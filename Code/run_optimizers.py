#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh independence study driver.

Uses a sequence of *already generated* meshes for varying global size h,
solves the reduced optimal control problem on each mesh, and records
stability metrics. Produces a summary JSON and plots vs h.

Requirements:
- FEniCS for PDE solves

Outputs per h:
- results/mesh_h_{h}/u_star.npy, y_star.npy
- plots/mesh_h_{h}/*.png (per-mesh plots)
- plots/mesh_independence_*.png (aggregated curves)
- results/mesh_independence_summary.json
"""
import os
import json
import math
import argparse
import sys
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# Support both invocation styles:
# - as a module:  python -m Code.mesh_independence
# - as a script:  python Code/mesh_independence.py
try:
    from .reduced_oc_model import ReducedOCModelExternal
    from .optimizers import bb, gd_fixed, nesterov_constant_ml
except ImportError:
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from Code.reduced_oc_model import ReducedOCModelExternal
    from Code.optimizers import bb, gd_fixed, nesterov_constant_ml


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_function(V, vec: np.ndarray):
    import fenics as fe
    f = fe.Function(V)
    f.vector().set_local(vec)
    f.vector().apply('insert')
    return f


def plot_mesh(mesh, outpath: str, title: str = 'Mesh of $[0,1]^2$'):
    import matplotlib.tri as mtri
    coords = mesh.coordinates()
    cells = mesh.cells()
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
    plt.figure(figsize=(5, 5))
    plt.triplot(tri, color='0.4', linewidth=0.5)
    plt.title(title)
    plt.xlabel('x'); plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_function(fun, title: str, outpath: str):
    import matplotlib.tri as mtri
    import fenics as fe
    V = fun.function_space()
    mesh = V.mesh()
    coords = mesh.coordinates(); cells = mesh.cells()
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
    v2d = fe.vertex_to_dof_map(V)
    vals = fun.vector().get_local()[v2d]
    plt.figure(figsize=(6, 5))
    tpc = plt.tripcolor(tri, vals, shading='gouraud')
    plt.colorbar(tpc)
    plt.title(title)
    plt.xlabel('x'); plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_function_3d(fun, title: str, outpath: str):
    import matplotlib.tri as mtri
    import fenics as fe
    V = fun.function_space()
    mesh = V.mesh()
    coords = mesh.coordinates(); cells = mesh.cells()
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
    v2d = fe.vertex_to_dof_map(V)
    vals = fun.vector().get_local()[v2d]
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(tri, vals, cmap='viridis', linewidth=0.1, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.7)
    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('value')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _h_tokens(h: float) -> List[str]:
    """Return a small set of stable string renderings for directory names."""
    candidates = [
        str(h),
        f"{h:g}",
        f"{h:.6f}".rstrip('0').rstrip('.'),
        f"{h:.3f}".rstrip('0').rstrip('.'),
    ]
    out: List[str] = []
    seen = set()
    for s in candidates:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def resolve_mesh_dir(mesh_root: str, h: float, mesh_prefix: str = 'mesh_h_') -> str:
    """Resolve an existing mesh directory for h.

    Expected convention:
    - mesh/{mesh_prefix}{h}/mesh_cells.xdmf  (default: mesh/mesh_h_{h}/...)

    Returns empty string if not found.
    """
    tokens = _h_tokens(h)
    for t in tokens:
        candidate_dir = os.path.join(mesh_root, f"{mesh_prefix}{t}")
        if os.path.exists(os.path.join(candidate_dir, 'mesh_cells.xdmf')):
            return candidate_dir
    return ""


def norm_L2_from_M(vec: np.ndarray, M) -> float:
    val = float(vec.T @ (M @ vec))
    return math.sqrt(max(val, 0.0))


def cg_solve_with_gradnorm_history(model: ReducedOCModelExternal,
                                  q_matvec,
                                  rhs: np.ndarray,
                                  rtol: float = 1e-3,
                                  atol: float = 0.0,
                                  maxiter: int | None = None,
                                  x0: np.ndarray | None = None) -> tuple[np.ndarray, int, Dict[str, List[float]]]:
    """Conjugate Gradient for SPD system Qx=rhs with gradient-norm history.

    We solve the stationarity system in coefficient space:
      Q u = rhs,  Q = B^T A^{-1} M A^{-1} B + beta * MU.

    The reduced gradient in the U-metric satisfies
      grad_U(u) = MU^{-1} (Q u - rhs).
    Hence, if we maintain the CG residual r = rhs - Q u, then
      ||grad_U(u)||_U^2 = (Q u - rhs)^T MU^{-1} (Q u - rhs) = r^T MU^{-1} r,
    which we can compute cheaply via one MU solve per iteration.
    """
    n = int(model.n)
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float, copy=True)

    # r = b - A x
    r = rhs - q_matvec(x)
    p = r.copy()
    rsold = float(r @ r)
    bnorm = float(np.linalg.norm(rhs))
    # Termination threshold as in SciPy-style: ||r|| <= max(atol, rtol*||b||)
    thresh = max(float(atol), float(rtol) * bnorm)

    grad_norm_hist: List[float] = []
    # Store grad norm at the initial iterate
    z = model.MU_fac.solve(r)  # MU^{-1} r
    grad_norm_hist.append(float(math.sqrt(max(float(r @ z), 0.0))))

    if float(np.linalg.norm(r)) <= thresh:
        return x, 0, {'grad_norm': grad_norm_hist}

    k = 0
    while True:
        if maxiter is not None and k >= int(maxiter):
            return x, k, {'grad_norm': grad_norm_hist}

        Ap = q_matvec(p)
        pAp = float(p @ Ap)
        if pAp <= 0.0:
            # Should not happen for SPD; treat as breakdown.
            return x, -1, {'grad_norm': grad_norm_hist}

        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        # Record gradient norm in U metric
        z = model.MU_fac.solve(r)
        grad_norm_hist.append(float(math.sqrt(max(float(r @ z), 0.0))))

        rnorm = float(np.linalg.norm(r))
        if rnorm <= thresh:
            return x, 0, {'grad_norm': grad_norm_hist}

        rsnew = float(r @ r)
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
        k += 1


def run_one_mesh(h: float,
                 beta: float,
                 omega_id: int,
                 mesh_root: str = 'mesh',
                 mesh_prefix: str = 'mesh_h_',
                 plots_dir: str = 'plots',
                 results_root: str = 'results',
                 per_mesh_plots: bool = True) -> Dict[str, Any]:
    """Load existing mesh for h, build model, compute u_* via CG, and collect metrics."""
    h_dir_token = _h_tokens(h)[0]

    mesh_dir = resolve_mesh_dir(mesh_root=mesh_root, h=h, mesh_prefix=mesh_prefix)
    ensure_dir(plots_dir)
    per_mesh_plots_dir = os.path.join(plots_dir, f'mesh_h_{h_dir_token}')
    if per_mesh_plots:
        ensure_dir(per_mesh_plots_dir)
    res_dir = os.path.join(results_root, f'mesh_h_{h_dir_token}')
    ensure_dir(res_dir)

    if not mesh_dir:
        raise FileNotFoundError(
            f"No existing mesh found for h={h}. "
            f"Expected '{mesh_root}/{mesh_prefix}{h_dir_token}/mesh_cells.xdmf' (or equivalent float formatting)."
        )

    cells_path = os.path.join(mesh_dir, 'mesh_cells.xdmf')
    facets_path = os.path.join(mesh_dir, 'mesh_facets.xdmf')

    # 2) Build model and compute u_* via CG
    if not os.path.exists(cells_path):
        raise FileNotFoundError(
            f"Mesh file not found: {cells_path}. "
            f"Expected an existing mesh in '{mesh_root}/{mesh_prefix}{{h}}'."
        )
    model = ReducedOCModelExternal(mesh_cells_xdmf=cells_path, omega_id=omega_id, beta=beta)

    # Ensure a mesh visualization exists alongside the mesh files.
    # This does NOT regenerate meshes; it only renders an image from the existing XDMF mesh.
    try:
        mesh_png_path = os.path.join(mesh_dir, 'mesh.png')
        plot_mesh(model.mesh, mesh_png_path, title=f"Mesh (h={h_dir_token})")
    except Exception as e:
        print(f"Warning: could not write mesh.png for h={h}: {e}")

    # Per-mesh plots
    # - Keep the full target plot only for the representative mesh h=0.02 (used in the report).
    if per_mesh_plots and h_dir_token == '0.02':
        try:
            import fenics as fe
            yd_fun = fe.Function(model.V)
            yd_fun.vector().set_local(model.y_d)
            yd_fun.vector().apply('insert')
            plot_function_3d(
                yd_fun,
                rf'Target $y_d$ (3D, h={h_dir_token})',
                os.path.join(per_mesh_plots_dir, 'y_d_3d.png'),
            )
        except Exception as e:
            print(f"Warning: per-mesh target plot failed for h={h}: {e}")

    # Compute CG-based u_* (normal equations Q u = B^T A^{-1} M y_d)
    n = model.n
    def q_matvec(v: np.ndarray) -> np.ndarray:
        return model.MU @ model.hess_U(v)
    z = model.M @ model.y_d
    p = model.adjoint(z)
    rhs = model.B.transpose() @ p

    u_star, info_cg, h_cg = cg_solve_with_gradnorm_history(
        model=model,
        q_matvec=q_matvec,
        rhs=rhs,
        rtol=1e-3,
        atol=0.0,
        maxiter=None,
        x0=None,
    )
    iters_cg = int(max(len(h_cg['grad_norm']) - 1, 0))
    F_star = float(model.cost(u_star))
    g_star = model.grad_U(u_star)
    grad_norm_star = float(model.norm_U(g_star))

    # Save arrays
    np.save(os.path.join(res_dir, 'u_star.npy'), u_star)
    y_star = model.state(u_star)
    np.save(os.path.join(res_dir, 'y_star.npy'), y_star)

    # Metrics
    entry: Dict[str, Any] = {
        'h': float(h),
        'n_dofs': int(model.n),
        'beta': float(beta),
        'omega_id': int(omega_id),
        'L_est': float(model.estimate_L(iters=30, tol=1e-6)),
        'F_star': F_star,
        'grad_norm_star': grad_norm_star,
        'u_star_norm_U': float(model.norm_U(u_star)),
        'y_star_norm_L2': float(norm_L2_from_M(y_star, model.M)),
        'cg_info_u_star': int(info_cg),
        'cg_iters_u_star': int(iters_cg)
    }
    # Run optimizers for iteration counts and per-mesh plots
    try:
        u0 = model.grad_U(np.zeros(n))
        L_fixed = model.estimate_L(iters=30, tol=1e-6)
        L_ml, m_ml = model.estimate_L_m(tol=1e-6)

        tol_abs = 1e-5
        tol_rel = 1e-3

        u_bb, h_bb = bb(model, u0=u0, tol_abs=tol_abs, tol_rel=tol_rel, max_iter=500)
        u_gd, h_gd = gd_fixed(model, u0=u0, tol_abs=tol_abs, tol_rel=tol_rel, max_iter=500, L=L_fixed)
        # Only use the strongly-convex constant-parameter variant (m,L) and label it "Nesterov".
        u_nesterov, h_nesterov = nesterov_constant_ml(model, u0=u0, tol_abs=tol_abs, tol_rel=tol_rel, max_iter=500, L=L_ml, m=m_ml)

        entry['iterations'] = {
            'CG': int(iters_cg),
            'BB': int(len(h_bb['cost'])),
            'GD': int(len(h_gd['cost'])),
            'Nesterov': int(len(h_nesterov['cost']))
        }

        entry['final_costs'] = {
            'CG': float(F_star),
            'BB': float(h_bb['cost'][-1]),
            'GD': float(h_gd['cost'][-1]),
            'Nesterov': float(h_nesterov['cost'][-1])
        }

        if per_mesh_plots:
            # 3D plots for each algorithm: final control u and resulting state y(u)
            try:
                # Conjugate Gradient (direct solve u_*)
                u_cg_fun = to_function(model.V, u_star)
                y_cg_fun = to_function(model.V, y_star)
                plot_function_3d(u_cg_fun, f"CG: optimal control u (h={h_dir_token})", os.path.join(per_mesh_plots_dir, 'u_cg_3d.png'))
                plot_function_3d(y_cg_fun, f"CG: optimal state y(u) (h={h_dir_token})", os.path.join(per_mesh_plots_dir, 'y_cg_3d.png'))

                u_bb_fun = to_function(model.V, u_bb)
                y_bb_fun = to_function(model.V, model.state(u_bb))
                plot_function_3d(u_bb_fun, f"BB: optimal control u (h={h_dir_token})", os.path.join(per_mesh_plots_dir, 'u_bb_3d.png'))
                plot_function_3d(y_bb_fun, f"BB: optimal state y(u) (h={h_dir_token})", os.path.join(per_mesh_plots_dir, 'y_bb_3d.png'))

                u_gd_fun = to_function(model.V, u_gd)
                y_gd_fun = to_function(model.V, model.state(u_gd))
                plot_function_3d(u_gd_fun, f"GD: optimal control u (h={h_dir_token})", os.path.join(per_mesh_plots_dir, 'u_gd_3d.png'))
                plot_function_3d(y_gd_fun, f"GD: optimal state y(u) (h={h_dir_token})", os.path.join(per_mesh_plots_dir, 'y_gd_3d.png'))

                u_nes_fun = to_function(model.V, u_nesterov)
                y_nes_fun = to_function(model.V, model.state(u_nesterov))
                plot_function_3d(u_nes_fun, f"Nesterov: optimal control u (h={h_dir_token})", os.path.join(per_mesh_plots_dir, 'u_nesterov_3d.png'))
                plot_function_3d(y_nes_fun, f"Nesterov: optimal state y(u) (h={h_dir_token})", os.path.join(per_mesh_plots_dir, 'y_nesterov_3d.png'))

                # Convergence plot: gradient norms
                plt.figure(figsize=(7, 5))
                # CG: full convergence curve from the stationarity-system solve
                plt.semilogy(h_cg['grad_norm'], label='CG')
                plt.semilogy(h_bb['grad_norm'], label='BB')
                plt.semilogy(h_gd['grad_norm'], label='GD 1/L')
                plt.semilogy(h_nesterov['grad_norm'], label='Nesterov')
                plt.xlabel('Iteration'); plt.ylabel(r'$\|\nabla F(u_k)\|_U$')
                plt.grid(True, which='both', ls='--', alpha=0.5)
                plt.legend(); plt.title(f'Gradient Norm Convergence (h={h_dir_token})')
                plt.tight_layout()
                plt.savefig(os.path.join(per_mesh_plots_dir, 'grad_norm.png'), dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Warning: per-mesh optimizer plots failed for h={h}: {e}")
    except Exception as e:
        print(f"Warning: optimizer runs/plots failed for h={h}: {e}")

    # Write a small LaTeX snippet for the representative-mesh table in the report.
    # This follows the same pattern as the mesh-independence tables: the report inputs
    # this file inside a tabular environment.
    try:
        it = entry.get('iterations', {})
        fc = entry.get('final_costs', {})
        methods = [
            ('CG ($Q\\mathbf{u}=\\mathbf{b}$)', 'CG'),
            ('BB', 'BB'),
            ('GD ($1/L$)', 'GD'),
            ('Nesterov', 'Nesterov'),
        ]

        lines = []
        for idx, (label, key) in enumerate(methods):
            cost = float(fc.get(key, np.nan))
            iters = it.get(key, np.nan)
            # 7 digits after the decimal point for costs, as requested.
            row = f"{label} & {cost:.7f} & {iters}"
            if idx < len(methods) - 1:
                row += r"\\"
            lines.append(row)

        with open(os.path.join(res_dir, 'numerics_table.tex'), 'w') as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"Warning: could not write numerics_table.tex for h={h}: {e}")
    return entry


def make_plots(summary: List[Dict[str, Any]], plots_dir: str = 'plots'):
    ensure_dir(plots_dir)
    # Sort by h ascending for plot readability
    summary_sorted = sorted(summary, key=lambda e: e['h'])
    H = [e['h'] for e in summary_sorted]
    N = [e['n_dofs'] for e in summary_sorted]
    methods = ['CG', 'BB', 'GD', 'Nesterov']
    final_costs_by_method: Dict[str, List[float]] = {m: [] for m in methods}
    iters_by_method: Dict[str, List[float]] = {m: [] for m in methods}
    for e in summary_sorted:
        fc = e.get('final_costs', {})
        it = e.get('iterations', {})
        for m in methods:
            final_costs_by_method[m].append(float(fc.get(m, np.nan)))
            iters_by_method[m].append(float(it.get(m, np.nan)))

    # Cost vs h
    plt.figure(figsize=(6.5, 4.8))
    for m in methods:
        plt.plot(H, final_costs_by_method[m], marker='o', label=m)
    plt.xlabel('Global mesh size h')
    plt.ylabel('Final objective value')
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    plt.title('Mesh Independence: Final Cost vs h')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mesh_independence_cost_vs_h.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # (Removed L vs h and gradnorm vs h per streamlined plots request)

    # Optional: cost vs DOFs
    plt.figure(figsize=(6.5, 4.8))
    for m in methods:
        plt.plot(N, final_costs_by_method[m], marker='o', label=m)
    plt.xlabel('Degrees of freedom (n)')
    plt.ylabel('Final objective value')
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    plt.title('Mesh Independence: Final Cost vs DOFs')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mesh_independence_cost_vs_ndofs.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Iterations vs h
    plt.figure(figsize=(6.5, 4.8))
    for m in methods:
        plt.plot(H, iters_by_method[m], marker='o', label=m)
    plt.xlabel('Global mesh size h')
    plt.ylabel('Iterations to tolerance')
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    plt.title('Mesh Independence: Iterations vs h')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mesh_independence_iterations_vs_h.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Mesh independence study for reduced OC problem.')
    ap.add_argument('--h-list', type=float, nargs='+', default=[0.06, 0.04, 0.03, 0.02, 0.015, 0.012, 0.01, 0.008],
                    help='List of global mesh size h values')
    ap.add_argument('--beta', type=float, default=1e-3, help='Tikhonov parameter')
    ap.add_argument('--omega-id', type=int, default=2, help='Control region physical id in cells')
    ap.add_argument('--mesh-root', type=str, default='mesh', help='Root directory for mesh outputs')
    ap.add_argument('--mesh-prefix', type=str, default='mesh_h_', help='Mesh directory prefix under mesh-root (default: mesh_h_)')
    ap.add_argument('--plots-dir', type=str, default='plots', help='Directory to save plots')
    ap.add_argument('--results-root', type=str, default='results', help='Root to save results per h and summary')
    ap.add_argument('--no-per-mesh-plots', action='store_true', help='Disable per-mesh plots under plots/mesh_h_{h}/')
    args = ap.parse_args()

    summary: List[Dict[str, Any]] = []
    for h in args.h_list:
        try:
            entry = run_one_mesh(h=float(h),
                                 beta=args.beta,
                                 omega_id=args.omega_id,
                                 mesh_root=args.mesh_root,
                                 mesh_prefix=args.mesh_prefix,
                                 plots_dir=args.plots_dir,
                                 results_root=args.results_root,
                                 per_mesh_plots=(not args.no_per_mesh_plots))
            summary.append(entry)
            print(f"Completed h={h}: F_star={entry['F_star']:.6g}, n={entry['n_dofs']}, grad*={entry['grad_norm_star']:.3e}")
        except Exception as e:
            print(f"Warning: h={h} run failed: {e}")

    # Write summary JSON
    ensure_dir(args.results_root)
    summary_path = os.path.join(args.results_root, 'mesh_independence_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'beta': args.beta, 'omega_id': args.omega_id, 'summary': summary}, f, indent=2)
    print(f"Wrote summary to {summary_path}")

    # Plots
    try:
        make_plots(summary, plots_dir=args.plots_dir)
        print(f"Saved plots to {args.plots_dir}")
    except Exception as e:
        print(f"Warning: could not generate plots: {e}")

    # Write LaTeX tables for the report (to top-level results folder)
    try:
        report_results_dir = args.results_root
        ensure_dir(report_results_dir)

        # Summary table (metrics)
        rows = sorted(summary, key=lambda e: e['h'], reverse=True)
        lines = []
        for idx, e in enumerate(rows):
            row = (
                f"{e['h']} & {e['n_dofs']} & {e['F_star']:.7f} & {e['u_star_norm_U']:.6f} & {e['y_star_norm_L2']:.6f}"
            )
            if idx < len(rows) - 1:
                row += r"\\"
            lines.append(row)
            if idx < len(rows) - 1:
                lines.append(r"\hline")
        with open(os.path.join(report_results_dir, 'mesh_independence_table.tex'), 'w') as f:
            f.write("\n".join(lines) + "\n")

        # Iteration count table
        lines2 = []
        for idx, e in enumerate(rows):
            it = e.get('iterations', {})
            bb_it = it.get('BB', '')
            gd_it = it.get('GD', '')
            ne_it = it.get('Nesterov', '')
            cg_it = it.get('CG', e.get('cg_iters_u_star', ''))
            row = f"{e['h']} & {cg_it} & {bb_it} & {gd_it} & {ne_it}"
            if idx < len(rows) - 1:
                row += r"\\"
            lines2.append(row)
            if idx < len(rows) - 1:
                lines2.append(r"\hline")
        with open(os.path.join(report_results_dir, 'mesh_iterations_table.tex'), 'w') as f:
            f.write("\n".join(lines2) + "\n")
        print(f"Wrote LaTeX tables to {report_results_dir}")
    except Exception as e:
        print(f"Warning: could not write LaTeX tables: {e}")


if __name__ == '__main__':
    main()
