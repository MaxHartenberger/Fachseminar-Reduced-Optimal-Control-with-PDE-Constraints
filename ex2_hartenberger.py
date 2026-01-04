#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 13:44:32 2025

@author: max
"""

import fenics as fe
from fenics import dx, grad, dot
import numpy as np
import matplotlib.pyplot as plt

# c)
mesh = fe.UnitSquareMesh(32, 32)
V = fe.FunctionSpace(mesh, 'Lagrange', 1)
U = fe.FunctionSpace(mesh, 'DG', 0)

y = fe.TrialFunction(V)
v = fe.TestFunction(V)
u = fe.TrialFunction(U)
w = fe.TestFunction(U)

M = fe.assemble(y * v * dx)
K = fe.assemble(u * w * dx)
A = fe.assemble(dot(grad(y), grad(v)) * dx)
P = fe.assemble(u * v * dx)

bc = fe.DirichletBC(V, fe.Constant(0.0), "on_boundary")

# d)
ud_1 = fe.Constant(1.0)
yd_1 = fe.Function(V)
fe.solve(dot(grad(yd_1), grad(v)) * dx == ud_1 * v * dx, yd_1, bc)

ud_2 = fe.Expression("sin(2*pi*x[0]) * cos(2*pi*x[1])", degree=4, pi=np.pi)
yd_2 = fe.Function(V)
fe.solve(dot(grad(yd_2), grad(v)) * dx == ud_2 * v * dx, yd_2, bc)

# Convert matrices to numpy
import scipy.sparse as sp
M_np = sp.csr_matrix(fe.as_backend_type(M).mat().getValuesCSR()[::-1])
K_np = sp.csr_matrix(fe.as_backend_type(K).mat().getValuesCSR()[::-1])
A_np = sp.csr_matrix(fe.as_backend_type(A).mat().getValuesCSR()[::-1])
P_np = sp.csr_matrix(fe.as_backend_type(P).mat().getValuesCSR()[::-1])

# Apply BC to A and M
bc.apply(fe.as_backend_type(A).mat())
bc.apply(fe.as_backend_type(M).mat())
A_np = sp.csr_matrix(fe.as_backend_type(A).mat().getValuesCSR()[::-1])
M_np = sp.csr_matrix(fe.as_backend_type(M).mat().getValuesCSR()[::-1])

# e)
def solve_lq(yd, sigma):
    # Optimality system:
    # A y = P u
    # A^T p = M (y - yd)
    # sigma K u + P^T p = 0
    
    # From third: u = - (1/sigma) K^{-1} P^T p
    # Substitute: A y = - (1/sigma) P K^{-1} P^T p
    # A^T p = M y - M yd
    
    # System: [A, (1/sigma) P K^{-1} P^T; -M, A^T] [y; p] = [0; -M yd]
    
    from scipy.sparse.linalg import spsolve
    
    # Compute K_inv = inv(K)
    K_inv = sp.linalg.inv(K_np)
    
    # Compute B = P K^{-1} P^T
    B = P_np @ K_inv @ P_np.T
    
    # System matrix
    zero = sp.csr_matrix((A_np.shape[0], A_np.shape[0]))
    top_left = A_np
    top_right = (1/sigma) * B
    bottom_left = -M_np
    bottom_right = A_np.T
    
    K_sys = sp.bmat([[top_left, top_right], [bottom_left, bottom_right]], format='csr')
    
    # RHS
    rhs = np.concatenate([np.zeros(A_np.shape[0]), -M_np @ yd.vector().get_local()])
    
    # Solve
    sol = spsolve(K_sys, rhs)
    y_vec = sol[:A_np.shape[0]]
    p_vec = sol[A_np.shape[0]:]
    
    # Compute u
    u_vec = - (1/sigma) * spsolve(K_np, P_np.T @ p_vec)
    
    return y_vec, u_vec, p_vec

# Test for yd_1 and yd_2
sigmas = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

for yd, label in [(yd_1, "yd_1"), (yd_2, "yd_2")]:
    print(f"\nFor {label}:")
    for sigma in sigmas:
        y_opt, u_opt, p_opt = solve_lq(yd, sigma)
        # ||u||_L2 = sqrt(u^T K u)
        norm_u = np.sqrt(u_opt.T @ K_np @ u_opt)
        print(f"sigma={sigma}: ||u||_L2 = {norm_u:.6f}")
        
        if sigma == 0.001:
            # Plot
            y_func = fe.Function(V)
            y_func.vector().set_local(y_opt)
            u_func = fe.Function(U)
            u_func.vector().set_local(u_opt)
            p_func = fe.Function(V)
            p_func.vector().set_local(p_opt)
            
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            fe.plot(y_func, title=f"y_opt {label}")
            plt.subplot(1, 3, 2)
            fe.plot(u_func, title=f"u_opt {label}")
            plt.subplot(1, 3, 3)
            fe.plot(p_func, title=f"p_opt {label}")
            plt.show()
