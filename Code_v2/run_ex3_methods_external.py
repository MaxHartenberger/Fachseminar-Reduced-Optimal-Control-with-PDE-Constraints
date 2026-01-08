#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner to compare BB, GD(1/L), Nesterov on the external-mesh reduced OC model.
Saves convergence plots; optional interactive display.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from Code_v2.reduced_oc_model_external import ReducedOCModelExternal
from Code_v2.optimizers import bb, gd_fixed, nesterov


def build_model(mesh_cells_xdmf: str, omega_id: int, beta: float):
    return ReducedOCModelExternal(mesh_cells_xdmf=mesh_cells_xdmf, omega_id=omega_id, beta=beta)


def main():
    parser = argparse.ArgumentParser(description='Compare BB, GD(1/L), Nesterov on external-mesh Ex3.')
    parser.add_argument('--mesh-cells-xdmf', type=str, default='mesh/out/mesh_cells.xdmf', help='Path to cell XDMF')
    parser.add_argument('--omega-id', type=int, default=2, help='Physical id for omega subdomain')
    parser.add_argument('--beta', type=float, default=1e-3, help='Tikhonov parameter')
    parser.add_argument('--outdir', type=str, default='results_v2', help='Directory to save plots')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    model = build_model(mesh_cells_xdmf=args.mesh_cells_xdmf, omega_id=args.omega_id, beta=args.beta)

    n = model.n
    u0 = model.grad_U(np.zeros(n))
    L = model.estimate_L(iters=30, tol=1e-6)

    u_bb, h_bb = bb(model, u0=u0, tol=1e-8, max_iter=500)
    u_gd, h_gd = gd_fixed(model, u0=u0, tol=1e-8, max_iter=500, L=L)
    u_ne, h_ne = nesterov(model, u0=u0, tol=1e-8, max_iter=500, L=L, restart=True)

    fig1 = plt.figure(figsize=(7, 5))
    plt.semilogy(h_bb['grad_norm'], label='BB')
    plt.semilogy(h_gd['grad_norm'], label='GD 1/L')
    plt.semilogy(h_ne['grad_norm'], label='Nesterov 1/L + restart')
    plt.xlabel('Iteration'); plt.ylabel('||∇F(u_k)||_U')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend(); plt.title('Gradient Norm Convergence')
    fig1.savefig(os.path.join(args.outdir, 'grad_norm.png'), dpi=150, bbox_inches='tight')

    fig2 = plt.figure(figsize=(7, 5))
    plt.semilogy(h_bb['cost'], label='BB')
    plt.semilogy(h_gd['cost'], label='GD 1/L')
    plt.semilogy(h_ne['cost'], label='Nesterov 1/L + restart')
    plt.xlabel('Iteration'); plt.ylabel('F(u_k)')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend(); plt.title('Cost Convergence')
    fig2.savefig(os.path.join(args.outdir, 'cost.png'), dpi=150, bbox_inches='tight')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
