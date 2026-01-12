#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner to compare gradient methods (BB, fixed GD 1/L, Nesterov) on Ex3.
Saves convergence plots to an output directory; optional interactive display.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from Code_v1.reduced_oc_model import ReducedOCModel
from Code_v1.optimizers import bb, gd_fixed, nesterov


def build_model(nx=64, beta=1e-3, radius=0.1):
    h = 1.0 / float(nx)
    return ReducedOCModel(h=h, beta=beta, radius=radius)


def main():
    parser = argparse.ArgumentParser(description='Compare BB, GD(1/L), Nesterov on Ex3.')
    parser.add_argument('--nx', type=int, default=64, help='Grid resolution per axis.')
    parser.add_argument('--beta', type=float, default=1e-3, help='Tikhonov parameter.')
    parser.add_argument('--radius', type=float, default=0.1, help='Control region radius.')
    parser.add_argument('--outdir', type=str, default='results', help='Directory to save plots.')
    parser.add_argument('--show', action='store_true', help='Show plots interactively.')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    model = build_model(nx=args.nx, beta=args.beta, radius=args.radius)

    n = model.n
    u0 = model.grad_U(np.zeros(n))  # u0 = ∇F(0)
    L = model.estimate_L(iters=30, tol=1e-6)

    u_bb, h_bb = bb(model, u0=u0, tol=1e-8, max_iter=500)
    u_gd, h_gd = gd_fixed(model, u0=u0, tol=1e-8, max_iter=500, L=L)
    u_ne, h_ne = nesterov(model, u0=u0, tol=1e-8, max_iter=500, L=L, restart=True)

    # Plot gradient norms
    fig1 = plt.figure(figsize=(7, 5))
    plt.semilogy(h_bb['grad_norm'], label='BB')
    plt.semilogy(h_gd['grad_norm'], label='GD 1/L')
    plt.semilogy(h_ne['grad_norm'], label='Nesterov 1/L + restart')
    plt.xlabel('Iteration')
    plt.ylabel('||∇F(u_k)||_U')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.title('Gradient Norm Convergence')
    fig1.savefig(os.path.join(args.outdir, 'grad_norm.png'), dpi=150, bbox_inches='tight')

    # Plot cost values
    fig2 = plt.figure(figsize=(7, 5))
    plt.semilogy(h_bb['cost'], label='BB')
    plt.semilogy(h_gd['cost'], label='GD 1/L')
    plt.semilogy(h_ne['cost'], label='Nesterov 1/L + restart')
    plt.xlabel('Iteration')
    plt.ylabel('F(u_k)')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.title('Cost Convergence')
    fig2.savefig(os.path.join(args.outdir, 'cost.png'), dpi=150, bbox_inches='tight')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
