#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use an external Gmsh-generated mesh to assemble the reduced OC model, solve,
run optimizers from Code/optimizers.py, and generate plots in test_meshing/out_figs/.

This does NOT modify files in Code/.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from test_meshing.external_model import ExternalReducedOCModel
from Code.optimizers import bb, gd_fixed, nesterov


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def to_function(V, vec):
    import fenics as fe
    f = fe.Function(V)
    f.vector().set_local(vec)
    f.vector().apply('insert')
    return f


def plot_mesh(mesh, outpath):
    import matplotlib.tri as mtri
    coords = mesh.coordinates()
    cells = mesh.cells()
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
    plt.figure(figsize=(5, 5))
    plt.triplot(tri, color='0.4', linewidth=0.5)
    plt.title('External mesh of $[0,1]^2$')
    plt.xlabel('x'); plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout(); plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_function(fun, title, outpath):
    import matplotlib.tri as mtri
    V = fun.function_space()
    mesh = V.mesh()
    coords = mesh.coordinates()
    cells = mesh.cells()
    tri = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
    # Map vertex order to dof order for CG1
    import fenics as fe
    v2d = fe.vertex_to_dof_map(V)
    vals = fun.vector().get_local()[v2d]
    plt.figure(figsize=(6, 5))
    tpc = plt.tripcolor(tri, vals, shading='gouraud')
    plt.colorbar(tpc)
    plt.title(title)
    plt.xlabel('x'); plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout(); plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def build_mask_function(V, subdomains, omega_id):
    import fenics as fe
    DG0 = fe.FunctionSpace(V.mesh(), 'DG', 0)
    mask = fe.Function(DG0)
    arr = mask.vector().get_local()
    # subdomains is a MeshFunction over cells; access via array()
    sub_arr = subdomains.array()
    arr[:] = (sub_arr == omega_id).astype(float)
    mask.vector().set_local(arr); mask.vector().apply('insert')
    return fe.interpolate(mask, V)


def main(mesh_cells_xdmf='test_meshing/out_mesh/mesh_cells.xdmf', omega_id=2, beta=1e-3, outdir='test_meshing/out_figs'):
    ensure_dir(outdir)

    model = ExternalReducedOCModel(mesh_cells_xdmf=mesh_cells_xdmf, omega_id=omega_id, beta=beta)

    # Mesh figure
    plot_mesh(model.mesh, os.path.join(outdir, 'mesh.png'))

    # Control region mask visualization
    omega_fun = build_mask_function(model.V, model.subdomains, omega_id)
    plot_function(omega_fun, r'Control region $\omega$', os.path.join(outdir, 'omega.png'))

    # Target y_d
    import fenics as fe
    yd_fun = fe.Function(model.V)
    yd_fun.vector().set_local(model.y_d); yd_fun.vector().apply('insert')
    plot_function(yd_fun, r'Target $y_d$', os.path.join(outdir, 'y_d.png'))

    # Run optimizers using the external model and reuse Code/optimizers
    n = model.n
    u0 = model.grad_U(np.zeros(n))
    L = model.estimate_L(iters=30, tol=1e-6)

    u_bb, h_bb = bb(model, u0=u0, tol=1e-8, max_iter=500)
    u_gd, h_gd = gd_fixed(model, u0=u0, tol=1e-8, max_iter=500, L=L)
    u_ne, h_ne = nesterov(model, u0=u0, tol=1e-8, max_iter=500, L=L, restart=True)

    # Best by final cost
    finals = [h_bb['cost'][-1], h_gd['cost'][-1], h_ne['cost'][-1]]
    u_opt = [u_bb, u_gd, u_ne][int(np.argmin(finals))]

    # Control and state plots
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
    plt.xlabel('Iteration'); plt.ylabel(r'$\|\nabla F(u_k)\|_U$')
    plt.grid(True, which='both', ls='--', alpha=0.5); plt.legend()
    plt.title('Gradient Norm Convergence')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'grad_norm.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.semilogy(h_bb['cost'], label='BB')
    plt.semilogy(h_gd['cost'], label='GD 1/L')
    plt.semilogy(h_ne['cost'], label='Nesterov 1/L + restart')
    plt.xlabel('Iteration'); plt.ylabel(r'$F(u_k)$')
    plt.grid(True, which='both', ls='--', alpha=0.5); plt.legend()
    plt.title('Cost Convergence')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'cost.png'), dpi=150, bbox_inches='tight')
    plt.close()

    with open(os.path.join(outdir, 'summary.txt'), 'w') as f:
        f.write(f"n={n}\n")
        f.write(f"beta={beta}\n")
        f.write(f"omega_id={omega_id}\n")
        f.write(f"L_est={L}\n")
        f.write(f"F(u_bb)={h_bb['cost'][-1]}\n")
        f.write(f"F(u_gd)={h_gd['cost'][-1]}\n")
        f.write(f"F(u_ne)={h_ne['cost'][-1]}\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh-cells-xdmf', default='test_meshing/out_mesh/mesh_cells.xdmf')
    ap.add_argument('--omega-id', type=int, default=2)
    ap.add_argument('--beta', type=float, default=1e-3)
    ap.add_argument('--outdir', default='test_meshing/out_figs')
    args = ap.parse_args()
    main(mesh_cells_xdmf=args.mesh_cells_xdmf, omega_id=args.omega_id, beta=args.beta, outdir=args.outdir)
