#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner to compare BB, GD(1/L), Nesterov on the external-mesh reduced OC model.

Discretization & Optimizer Notes
--------------------------------------------
- Spaces: V=P1 with homogeneous Dirichlet for state/adjoint; U=H=L2 with P1
    coefficients (no BCs) and inner product via the control mass MU.
- Control region: B=χ_ω implemented by dx(omega_id, subdomain_data=subdomains)
    in the model; Dirichlet BCs applied to A only, with RHS zeroing at boundary DOFs.
- Gradient: ∇F(u) = MU^{-1} B^T p + β u with adjoint solve A p = M (y - y_d).
- Stepsizes: GD uses α=1/L with L estimated by power iteration on the Hessian;
    BB alternates BB1/BB2 using MU-induced norms; Nesterov uses 1/L with optional restart.

Saves convergence plots; optional interactive display.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from reduced_oc_model import ReducedOCModelExternal
from optimizers import bb, gd_fixed, nesterov

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def to_function(V, vec: np.ndarray):
    import fenics as fe
    f = fe.Function(V)
    f.vector().set_local(vec)
    f.vector().apply('insert')
    return f

def plot_mesh(mesh, outpath: str):
    import matplotlib.tri as mtri
    coords = mesh.coordinates()
    cells = mesh.cells()
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
    plt.figure(figsize=(5, 5))
    plt.triplot(tri, color='0.4', linewidth=0.5)
    plt.title('Mesh of $[0,1]^2$')
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

def build_mask_function(V, subdomains, omega_id):
    import fenics as fe
    DG0 = fe.FunctionSpace(V.mesh(), 'DG', 0)
    mask = fe.Function(DG0)
    arr = mask.vector().get_local()
    sub_arr = subdomains.array()
    arr[:] = (sub_arr == omega_id).astype(float)
    mask.vector().set_local(arr); mask.vector().apply('insert')
    return fe.interpolate(mask, V)


def build_model(mesh_cells_xdmf: str, omega_id: int, beta: float):
    """Construct the external-mesh reduced model with control region tag `omega_id` and Tikhonov β."""
    return ReducedOCModelExternal(mesh_cells_xdmf=mesh_cells_xdmf, omega_id=omega_id, beta=beta)


def main():
    parser = argparse.ArgumentParser(description='Compare BB, GD(1/L), Nesterov on external-mesh Ex3.')
    parser.add_argument('--mesh-cells-xdmf', type=str, default='mesh/out/mesh_cells.xdmf', help='Path to cell XDMF')
    parser.add_argument('--omega-id', type=int, default=2, help='Physical id for omega subdomain')
    parser.add_argument('--beta', type=float, default=1e-3, help='Tikhonov parameter')
    parser.add_argument('--plots-dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save optimizer outputs')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    args = parser.parse_args()

    ensure_outdir(args.plots_dir)
    ensure_outdir(args.results_dir)

    model = build_model(mesh_cells_xdmf=args.mesh_cells_xdmf, omega_id=args.omega_id, beta=args.beta)

    # Mesh and data figures
    plot_mesh(model.mesh, os.path.join(args.plots_dir, 'mesh.png'))
    omega_fun = build_mask_function(model.V, model.subdomains, args.omega_id)
    plot_function(omega_fun, r'Control region $\omega$', os.path.join(args.plots_dir, 'omega.png'))
    import fenics as fe
    yd_fun = fe.Function(model.V)
    yd_fun.vector().set_local(model.y_d); yd_fun.vector().apply('insert')
    plot_function(yd_fun, r'Target $y_d$', os.path.join(args.plots_dir, 'y_d.png'))

    n = model.n
    # Initialize at u0 = ∇F(0) and estimate Lipschitz L for GD/Nesterov.
    u0 = model.grad_U(np.zeros(n))
    L = model.estimate_L(iters=30, tol=1e-6)

    u_bb, h_bb = bb(model, u0=u0, tol=1e-8, max_iter=500)
    u_gd, h_gd = gd_fixed(model, u0=u0, tol=1e-8, max_iter=500, L=L)
    u_ne, h_ne = nesterov(model, u0=u0, tol=1e-8, max_iter=500, L=L, restart=True)

    # Choose best final cost and plot optimized control/state
    finals = [h_bb['cost'][-1], h_gd['cost'][-1], h_ne['cost'][-1]]
    u_opt = [u_bb, u_gd, u_ne][int(np.argmin(finals))]
    u_fun = to_function(model.V, u_opt)
    plot_function(u_fun, r'Optimized control $u_*$', os.path.join(args.plots_dir, 'u_opt.png'))
    y_opt = model.state(u_opt)
    y_fun = to_function(model.V, y_opt)
    plot_function(y_fun, r'State $y(u_*)$', os.path.join(args.plots_dir, 'y_opt.png'))

    fig1 = plt.figure(figsize=(7, 5))
    plt.semilogy(h_bb['grad_norm'], label='BB')
    plt.semilogy(h_gd['grad_norm'], label='GD 1/L')
    plt.semilogy(h_ne['grad_norm'], label='Nesterov 1/L + restart')
    plt.xlabel('Iteration'); plt.ylabel(r'$\|\nabla F(u_k)\|_U$')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend(); plt.title('Gradient Norm Convergence')
    fig1.savefig(os.path.join(args.plots_dir, 'grad_norm.png'), dpi=150, bbox_inches='tight')

    fig2 = plt.figure(figsize=(7, 5))
    plt.semilogy(h_bb['cost'], label='BB')
    plt.semilogy(h_gd['cost'], label='GD 1/L')
    plt.semilogy(h_ne['cost'], label='Nesterov 1/L + restart')
    plt.xlabel('Iteration'); plt.ylabel(r'$F(u_k)$')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend(); plt.title('Cost Convergence')
    fig2.savefig(os.path.join(args.plots_dir, 'cost.png'), dpi=150, bbox_inches='tight')

    # Save optimizer results
    try:
        import json
        method_idx = int(np.argmin(finals))
        method_names = ['BB', 'GD', 'Nesterov']
        summary = {
            'omega_id': int(args.omega_id),
            'beta': float(args.beta),
            'L_est': float(L),
            'final_costs': {
                'BB': float(finals[0]),
                'GD': float(finals[1]),
                'Nesterov': float(finals[2])
            },
            'best_method': method_names[method_idx],
            'iterations': {
                'BB': int(len(h_bb['cost'])),
                'GD': int(len(h_gd['cost'])),
                'Nesterov': int(len(h_ne['cost']))
            }
        }
        with open(os.path.join(args.results_dir, 'optimizer_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        np.save(os.path.join(args.results_dir, 'u_opt.npy'), u_opt)
        np.save(os.path.join(args.results_dir, 'y_opt.npy'), y_opt)
    except Exception as e:
        print(f"Warning: could not save optimizer results: {e}")

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
