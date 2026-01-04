#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 3: Control reduced approach with Barzilai-Borwein method
"""

import fenics
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


class elliptic_control_model:
    """
    Model for elliptic control problem: -Delta y = B u, y=0 on boundary
    With B = chi_omega, omega circle center 0.5 radius 0.1
    """
    def __init__(self, dx=0.01, beta=1e-3):
        self.dx = dx
        self.beta = beta
        self.a1, self.b1, self.a2, self.b2 = 0, 0, 1, 1
        self._setup()

    def _setup(self):
        # Mesh and space
        lower_left = fenics.Point(self.a1, self.b1)
        upper_right = fenics.Point(self.a2, self.b2)
        nx = int((self.a2 - self.a1) / self.dx)
        ny = int((self.b2 - self.b1) / self.dx)
        self.mesh = fenics.RectangleMesh(lower_left, upper_right, nx, ny)
        self.V = fenics.FunctionSpace(self.mesh, 'P', 1)
        self.dofs = self.V.dim()

        # Forms
        y = fenics.TrialFunction(self.V)
        v = fenics.TestFunction(self.V)

        # Stiffness A
        self.A = csr_matrix(fenics.as_backend_type(fenics.assemble(fenics.dot(fenics.grad(y), fenics.grad(v)) * fenics.dx)).mat().getValuesCSR()[::-1])

        # Mass M
        self.M = csr_matrix(fenics.as_backend_type(fenics.assemble(y * v * fenics.dx)).mat().getValuesCSR()[::-1])

        # Control operator B: chi_omega
        omega_expr = fenics.Expression("((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) < 0.01) ? 1 : 0", degree=0)
        B_form = omega_expr * y * v * fenics.dx
        self.B = csr_matrix(fenics.as_backend_type(fenics.assemble(B_form)).mat().getValuesCSR()[::-1])

        # Dirichlet BCs
        self.bcs = [fenics.DirichletBC(self.V, fenics.Constant(0.0), "on_boundary")]
        for bc in self.bcs:
            bc.apply(fenics.as_backend_type(self.A).mat())
            bc.apply(fenics.as_backend_type(self.M).mat())
            bc.apply(fenics.as_backend_type(self.B).mat())

        # yd
        yd_expr = fenics.Expression("0.5 - fmax(fabs(x[0]-0.5), fabs(x[1]-0.5))", degree=1)
        yd_func = fenics.interpolate(yd_expr, self.V)
        self.yd = yd_func.vector().get_local()

    def solve_state(self, u):
        """
        Solve A y = B u
        """
        rhs = self.B @ u
        for bc in self.bcs:
            bc.apply(rhs)
        y = spsolve(self.A, rhs)
        return y

    def solve_adjoint(self, z):
        """
        Solve A p = z
        """
        rhs = z.copy()
        for bc in self.bcs:
            bc.apply(rhs)
        p = spsolve(self.A, rhs)
        return p

    def cost(self, u):
        """
        F(u) = 1/2 ||S u - yd||_M^2 + beta/2 ||u||_M^2
        """
        y = self.solve_state(u)
        diff = y - self.yd
        term1 = 0.5 * diff.T @ self.M @ diff
        term2 = self.beta / 2 * u.T @ self.M @ u
        return term1 + term2

    def gradient(self, u):
        """
        grad F(u) = B^T p + beta u, where A p = M (S u - yd)
        """
        y = self.solve_state(u)
        z = self.M @ (y - self.yd)
        p = self.solve_adjoint(z)
        grad = self.B.T @ p + self.beta * (self.M @ u)
        return grad


def barzilai_borwein(model, tol=1e-8, max_iter=1000):
    """
    Barzilai-Borwein gradient method
    """
    u_prev = np.zeros(model.dofs)
    grad_prev = model.gradient(u_prev)
    u = grad_prev.copy()  # initial u0 = grad F(0)

    grad_norms = [np.sqrt(grad_prev.T @ model.M @ grad_prev)]

    k = 0
    while np.sqrt(u.T @ model.M @ u) > tol and k < max_iter:
        if k > 0:
            s = u - u_prev
            d = grad_prev - model.gradient(u_prev)
            if k % 2 == 0:
                alpha = (d.T @ model.M @ d) / (s.T @ model.M @ d)
            else:
                alpha = (s.T @ model.M @ d) / (s.T @ model.M @ s)
            u_new = u - alpha * model.gradient(u)
        else:
            # First step
            alpha = 1.0  # or something
            u_new = u - alpha * model.gradient(u)

        u_prev = u
        grad_prev = model.gradient(u)
        u = u_new

        grad_norm = np.sqrt(u.T @ model.M @ u)
        grad_norms.append(grad_norm)

        k += 1

    return u, grad_norms


if __name__ == "__main__":
    model = elliptic_control_model(dx=0.01, beta=1e-3)

    # Solve with BB
    u_opt, grad_norms = barzilai_borwein(model)

    # Plot convergence
    plt.semilogy(grad_norms)
    plt.xlabel('Iteration')
    plt.ylabel('||grad F||_U')
    plt.title('Convergence of Barzilai-Borwein')
    plt.show()

    # Plot optimal u and y
    y_opt = model.solve_state(u_opt)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fenics.plot(fenics.Function(model.V, u_opt), ax=axes[0], title="Optimal control u")
    fenics.plot(fenics.Function(model.V, y_opt), ax=axes[1], title="Optimal state y")
    plt.show()