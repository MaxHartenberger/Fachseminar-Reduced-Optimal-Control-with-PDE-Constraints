#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 2: Discretization and solution of LQ optimal control problem
"""

import fenics
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


class elliptic_model_ex2:
    """
    Model for elliptic PDE -Delta y = u, y=0 on boundary
    """
    def __init__(self, a1=0, b1=0, a2=1, b2=1, dx=0.01):
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2
        self.dx = dx
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
        self.A_form = fenics.dot(fenics.grad(y), fenics.grad(v)) * fenics.dx
        self.A = csr_matrix(fenics.as_backend_type(fenics.assemble(self.A_form)).mat().getValuesCSR()[::-1])

        # Mass M
        self.M_form = y * v * fenics.dx
        self.M = csr_matrix(fenics.as_backend_type(fenics.assemble(self.M_form)).mat().getValuesCSR()[::-1])

        # Dirichlet BCs
        self.bcs = [fenics.DirichletBC(self.V, fenics.Constant(0.0), "on_boundary")]
        for bc in self.bcs:
            bc.apply(fenics.as_backend_type(self.A).mat())
            bc.apply(fenics.as_backend_type(self.M).mat())

    def solve_state(self, u):
        """
        Solve A y = M u
        """
        rhs = self.M @ u
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


def solve_optimal_control(model, yd, sigma):
    """
    Solve min 1/2 ||y - yd||_M^2 + sigma/2 ||u||_M^2 s.t. A y = M u
    """
    # Optimality system:
    # A y = M u
    # A p = M (y - yd)
    # sigma M u + M p = 0

    # From third: u = - (1/sigma) p
    # From second: A p = M y - M yd
    # From first: A y = M u = - (1/sigma) M p

    # So A y + (1/sigma) M p = 0
    # A p - M y = - M yd

    # System:
    # [A, (1/sigma) M; -M, A] [y; p] = [0; -M yd]

    from scipy.sparse import bmat, csc_matrix

    zero = csc_matrix((model.dofs, model.dofs))
    Ms = (1/sigma) * model.M
    A = model.A
    M = model.M

    # Block matrix
    K = bmat([[A, Ms], [-M, A]], format='csc')
    rhs = np.concatenate([np.zeros(model.dofs), -M @ yd])

    sol = spsolve(K, rhs)
    y = sol[:model.dofs]
    p = sol[model.dofs:]
    u = - (1/sigma) * p  # from sigma u + p = 0

    return y, u, p


if __name__ == "__main__":
    model = elliptic_model_ex2(dx=0.01)

    # Generate yd for ud = 1
    ud1 = np.ones(model.dofs)
    yd1 = model.solve_state(ud1)

    # ud = sin(2 pi x1) cos(2 pi x2)
    ud_expr = fenics.Expression("sin(2*M_PI*x[0])*cos(2*M_PI*x[1])", degree=3)
    ud_func = fenics.interpolate(ud_expr, model.V)
    ud2 = ud_func.vector().get_local()
    yd2 = model.solve_state(ud2)

    # Solve for different sigma
    sigmas = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    for yd, label in [(yd1, "yd1"), (yd2, "yd2")]:
        print(f"\nFor {label}:")
        for sigma in sigmas:
            y_opt, u_opt, p_opt = solve_optimal_control(model, yd, sigma)
            norm_grad = np.sqrt(u_opt.T @ model.M @ u_opt)  # ||u||_L2
            print(f"sigma={sigma}: ||u||_L2 = {norm_grad:.6f}")

            # Plot for sigma=1e-3
            if sigma == 1e-3:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                fenics.plot(fenics.Function(model.V, y_opt), ax=axes[0], title=f"y_opt {label}")
                fenics.plot(fenics.Function(model.V, u_opt), ax=axes[1], title=f"u_opt {label}")
                fenics.plot(fenics.Function(model.V, p_opt), ax=axes[2], title=f"p_opt {label}")
                plt.show()