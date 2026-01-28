#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh independence study driver.

Generates a sequence of meshes for varying global size h via Gmsh,
solves the reduced optimal control problem on each mesh, and records
stability metrics. Produces a summary JSON and plots vs h.

Requirements:
- Gmsh + meshio for mesh generation (see mesh/gmsh_mesh.py)
- FEniCS for PDE solves

Outputs per h:
- results/mesh_h_{h}/u_star.npy, y_star.npy
- plots/mesh_independence_*.png (aggregated curves)
- results/mesh_independence_summary.json
"""
import os
import json
import math
import argparse
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from .reduced_oc_model import ReducedOCModelExternal
from mesh.gmsh_mesh import build_mesh as gmsh_build_mesh
from .optimizers import bb, gd_fixed, nesterov_constant_ml


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def norm_L2_from_M(vec: np.ndarray, M) -> float:
    val = float(vec.T @ (M @ vec))
    return math.sqrt(max(val, 0.0))


def run_one_mesh(h: float,
                 beta: float,
                 omega_id: int,
                 cx: float = 0.5,
                 cy: float = 0.5,
                 r: float = 0.1,
                 refine_circle: bool = True,
                 mesh_root: str = 'mesh',
                 plots_dir: str = 'plots',
                 results_root: str = 'results',
                 make_per_mesh_plots: bool = False) -> Dict[str, Any]:
    """Generate mesh for given h, build model, compute u_* via CG, and collect metrics."""
    mesh_outdir = os.path.join(mesh_root, f'out_h_{h}')
    ensure_dir(mesh_outdir)
    ensure_dir(plots_dir)
    res_dir = os.path.join(results_root, f'mesh_h_{h}')
    ensure_dir(res_dir)

    # 1) Build mesh
    gmsh_build_mesh(cx=cx, cy=cy, r=r, h=h, refine_circle=refine_circle, outdir=mesh_outdir, plots_dir=plots_dir)

    # 2) Build model and compute u_* via CG
    cells_path = os.path.join(mesh_outdir, 'mesh_cells.xdmf')
    model = ReducedOCModelExternal(mesh_cells_xdmf=cells_path, omega_id=omega_id, beta=beta)

    # Compute CG-based u_* (normal equations Q u = B^T A^{-1} M y_d)
    from scipy.sparse.linalg import LinearOperator, cg
    n = model.n
    def q_matvec(v: np.ndarray) -> np.ndarray:
        return model.MU @ model.hess_U(v)
    Qop = LinearOperator((n, n), matvec=q_matvec, rmatvec=q_matvec, dtype=float)
    z = model.M @ model.y_d
    p = model.adjoint(z)
    rhs = model.B.transpose() @ p
    iters = [0]
    def _cb(_):
        iters[0] += 1
    u_star, info_cg = cg(Qop, rhs, rtol=1e-10, maxiter=None, callback=_cb)
    iters_cg = int(iters[0])
    F_star = float(model.cost(u_star))
    g_star = model.grad_U(u_star)
    grad_norm_star = float(model.norm_U(g_star))

    # Save arrays
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
    # Optional: run optimizers for iteration counts and per-mesh plots
    try:
        u0 = model.grad_U(np.zeros(n))
        L_fixed = model.estimate_L(iters=30, tol=1e-6)
        L_ml, m_ml = model.estimate_L_m(tol=1e-6)

        u_bb, h_bb = bb(model, u0=u0, tol=1e-8, max_iter=500)
        u_gd, h_gd = gd_fixed(model, u0=u0, tol=1e-8, max_iter=500, L=L_fixed)
        u_ne_ml, h_ne_ml = nesterov_constant_ml(model, u0=u0, tol=1e-8, max_iter=500, L=L_ml, m=m_ml)

        entry['iterations'] = {
            'BB': int(len(h_bb['cost'])),
            'GD': int(len(h_gd['cost'])),
            'Nesterov_mL': int(len(h_ne_ml['cost']))
        }

        if make_per_mesh_plots:
            # Per-mesh plots disabled in streamlined study
            pass
    except Exception as e:
        print(f"Warning: optimizer runs/plots failed for h={h}: {e}")
    return entry


def make_plots(summary: List[Dict[str, Any]], plots_dir: str = 'plots'):
    ensure_dir(plots_dir)
    # Sort by h ascending for plot readability
    summary_sorted = sorted(summary, key=lambda e: e['h'])
    H = [e['h'] for e in summary_sorted]
    N = [e['n_dofs'] for e in summary_sorted]
    F = [e['F_star'] for e in summary_sorted]
    U = [e['u_star_norm_U'] for e in summary_sorted]
    Y = [e['y_star_norm_L2'] for e in summary_sorted]

    # Cost vs h
    plt.figure(figsize=(6.5, 4.8))
    plt.plot(H, F, marker='o')
    plt.xlabel('Global mesh size h')
    plt.ylabel(r'$F(u_*^{(h)})$')
    plt.grid(True, ls='--', alpha=0.5)
    plt.title('Mesh Independence: Cost vs h')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mesh_independence_cost_vs_h.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Norms vs h
    plt.figure(figsize=(6.5, 4.8))
    plt.plot(H, U, marker='o', label=r'$\|u_*^{(h)}\|_U$')
    plt.plot(H, Y, marker='s', label=r'$\|y(u_*^{(h)})\|_{L^2}$')
    plt.xlabel('Global mesh size h')
    plt.ylabel('Norm value')
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    plt.title('Mesh Independence: Norms vs h')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mesh_independence_norms_vs_h.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # (Removed L vs h and gradnorm vs h per streamlined plots request)

    # Optional: cost vs DOFs
    plt.figure(figsize=(6.5, 4.8))
    plt.plot(N, F, marker='o')
    plt.xlabel('Degrees of freedom (n)')
    plt.ylabel(r'$F(u_*^{(h)})$')
    plt.grid(True, ls='--', alpha=0.5)
    plt.title('Mesh Independence: Cost vs DOFs')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mesh_independence_cost_vs_ndofs.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Mesh independence study for reduced OC problem.')
    ap.add_argument('--h-list', type=float, nargs='+', default=[0.06, 0.04, 0.03, 0.02, 0.015, 0.012, 0.01],
                    help='List of global mesh size h values')
    ap.add_argument('--beta', type=float, default=1e-3, help='Tikhonov parameter')
    ap.add_argument('--omega-id', type=int, default=2, help='Control region physical id in cells')
    ap.add_argument('--cx', type=float, default=0.5, help='Circle center x')
    ap.add_argument('--cy', type=float, default=0.5, help='Circle center y')
    ap.add_argument('--r', type=float, default=0.1, help='Circle radius')
    ap.add_argument('--no-refine-circle', action='store_true', help='Disable circle-near refinement')
    ap.add_argument('--mesh-root', type=str, default='mesh', help='Root directory for mesh outputs')
    ap.add_argument('--plots-dir', type=str, default='plots', help='Directory to save plots')
    ap.add_argument('--results-root', type=str, default='results', help='Root to save results per h and summary')
    args = ap.parse_args()

    summary: List[Dict[str, Any]] = []
    for h in args.h_list:
        try:
            entry = run_one_mesh(h=float(h),
                                 beta=args.beta,
                                 omega_id=args.omega_id,
                                 cx=args.cx,
                                 cy=args.cy,
                                 r=args.r,
                                 refine_circle=(not args.no_refine_circle),
                                 mesh_root=args.mesh_root,
                                 plots_dir=args.plots_dir,
                                 results_root=args.results_root,
                                 make_per_mesh_plots=False)
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
            lines.append(
                f"{e['h']} & {e['n_dofs']} & {e['F_star']:.7f} & {e['u_star_norm_U']:.6f} & {e['y_star_norm_L2']:.6f} "
                + r"\\"
            )
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
            ne_it = it.get('Nesterov_mL', '')
            cg_it = e.get('cg_iters_u_star', '')
            lines2.append(
                f"{e['h']} & {cg_it} & {bb_it} & {gd_it} & {ne_it} " + r"\\"
            )
            if idx < len(rows) - 1:
                lines2.append(r"\hline")
        with open(os.path.join(report_results_dir, 'mesh_iterations_table.tex'), 'w') as f:
            f.write("\n".join(lines2) + "\n")
        print(f"Wrote LaTeX tables to {report_results_dir}")
    except Exception as e:
        print(f"Warning: could not write LaTeX tables: {e}")


if __name__ == '__main__':
    main()
