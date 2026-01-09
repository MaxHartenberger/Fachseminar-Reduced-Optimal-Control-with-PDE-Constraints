#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate figures for the Ex3 reduced optimal control problem:
- Mesh of the domain
- Control region mask (omega)
- Target y_d
- Optimized control u_opt
- Corresponding state y(u_opt)
- Convergence plots (grad norm, cost)

Outputs PNGs into ../results by default.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from Code_v1.reduced_oc_model import ReducedOCModel
from Code_v1.optimizers import bb, gd_fixed, nesterov


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def to_function(V, vec: np.ndarray):
    import fenics as fe
    f = fe.Function(V)
    f.vector().set_local(vec)
    f.vector().apply('insert')
    return f


def plot_mesh(mesh, outpath: str):
    import fenics as fe
    plt.figure(figsize=(5, 5))
    fe.plot(mesh)
    plt.title('Mesh of $[0,1]^2$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_function(fun, title: str, outpath: str):
    import fenics as fe
    plt.figure(figsize=(6, 5))
    c = fe.plot(fun)
    plt.colorbar(c)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def build_mask_function(V, radius: float):
    import fenics as fe
    omega = fe.Expression(
        "((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) < R*R) ? 1.0 : 0.0",
        degree=1, R=float(radius)
    )
    return fe.interpolate(omega, V)


def build_yd_function(V):
    import fenics as fe
    expr = fe.Expression("0.5 - fmax(fabs(x[0]-0.5), fabs(x[1]-0.5))", degree=1)
    return fe.interpolate(expr, V)


def main(nx=64, beta=1e-3, radius=0.1, outdir='../results'):
    ensure_outdir(outdir)

    model = ReducedOCModel(h=1.0/float(nx), beta=beta, radius=radius)

    # Optimization runs (and L estimate)
    n = model.n
    u0 = model.grad_U(np.zeros(n))
    L = model.estimate_L(iters=30, tol=1e-6)

    u_bb, h_bb = bb(model, u0=u0, tol=1e-8, max_iter=500)
    u_gd, h_gd = gd_fixed(model, u0=u0, tol=1e-8, max_iter=500, L=L)
    u_ne, h_ne = nesterov(model, u0=u0, tol=1e-8, max_iter=500, L=L, restart=True)

    # Choose the best among methods by cost for illustration
    costs = [h_bb['cost'][-1], h_gd['cost'][-1], h_ne['cost'][-1]]
    methods = [u_bb, u_gd, u_ne]
    u_opt = methods[int(np.argmin(costs))]

    # Mesh
    plot_mesh(model.mesh, os.path.join(outdir, 'mesh.png'))

    # Control region mask omega
    omega_fun = build_mask_function(model.V, radius=radius)
    plot_function(omega_fun, r'Control region $\omega$', os.path.join(outdir, 'omega.png'))

    # Target y_d
    yd_fun = build_yd_function(model.V)
    plot_function(yd_fun, r'Target $y_d$', os.path.join(outdir, 'y_d.png'))

    # Optimal control and corresponding state
    u_fun = to_function(model.V, u_opt)
    plot_function(u_fun, r'Optimized control $u_*$', os.path.join(outdir, 'u_opt.png'))

    y_opt = model.state(u_opt)
    y_fun = to_function(model.V, y_opt)
    plot_function(y_fun, r'State $y(u_*)$', os.path.join(outdir, 'y_opt.png'))

    # Convergence plots
    plt.figure(figsize=(7, 5))
    plt.semilogy(h_bb['grad_norm'], label='BB')
    plt.semilogy(h_gd['grad_norm'], label='GD 1/L')
    plt.semilogy(h_ne['grad_norm'], label='Nesterov 1/L + restart')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\|\nabla F(u_k)\|_U$')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.title('Gradient Norm Convergence')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'grad_norm.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.semilogy(h_bb['cost'], label='BB')
    plt.semilogy(h_gd['cost'], label='GD 1/L')
    plt.semilogy(h_ne['cost'], label='Nesterov 1/L + restart')
    plt.xlabel('Iteration')
    plt.ylabel(r'$F(u_k)$')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.title('Cost Convergence')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'cost.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Also dump some scalars for LaTeX if needed
    with open(os.path.join(outdir, 'summary.txt'), 'w') as f:
        f.write(f"n = {n}\n")
        f.write(f"beta = {beta}\n")
        f.write(f"radius = {radius}\n")
        f.write(f"L_est = {L}\n")
        f.write(f"F(u_bb) = {h_bb['cost'][-1]}\n")
        f.write(f"F(u_gd) = {h_gd['cost'][-1]}\n")
        f.write(f"F(u_ne) = {h_ne['cost'][-1]}\n")


if __name__ == '__main__':
    # Default paths relative to this file's location
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(os.path.dirname(here), 'results')
    main(nx=64, beta=1e-3, radius=0.1, outdir=out)
