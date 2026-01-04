#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 1.2: Implementation of analytical_problem and elliptic_model classes
"""

import fenics
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class analytical_problem:
    """
    Class containing the domain, boundary conditions, and data for the elliptic problem.
    """
    def __init__(self, a1, b1, a2, b2, gamma_D, gamma_N, f_expr, gN_expr, c, kappa):
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2
        self.gamma_D = gamma_D  # list of boundary indices for Dirichlet
        self.gamma_N = gamma_N  # list of boundary indices for Neumann
        self.f = f_expr  # FEniCS Expression for f
        self.gN = gN_expr  # FEniCS Expression for gN
        self.c = c
        self.kappa = kappa


class elliptic_model:
    """
    Class for discretizing and solving the elliptic PDE.
    """
    def __init__(self, problem, dx=0.005):
        self.problem = problem
        self.dx = dx
        self.mesh = None
        self.V = None
        self.dirichlet_conditions = None
        self.A1 = None
        self.A2 = None
        self.B = None
        self.L = None
        self.dofs = None

        self._setup_mesh_and_space()
        self._setup_forms()
        self._assemble_matrices()

    def _setup_mesh_and_space(self):
        # Generate mesh
        lower_left = fenics.Point(self.problem.a1, self.problem.b1)
        upper_right = fenics.Point(self.problem.a2, self.problem.b2)
        self.mesh = fenics.RectangleMesh(lower_left, upper_right, int((self.problem.a2 - self.problem.a1)/self.dx), int((self.problem.b2 - self.problem.b1)/self.dx))
        self.V = fenics.FunctionSpace(self.mesh, 'P', 1)
        self.dofs = self.V.dim()

        # Dirichlet BCs: homogeneous on gamma_D
        self.dirichlet_conditions = []
        for i in self.problem.gamma_D:
            if i == 1:  # bottom
                bc = fenics.DirichletBC(self.V, fenics.Constant(0.0), "near(x[1], {})".format(self.problem.b1))
            elif i == 2:  # left
                bc = fenics.DirichletBC(self.V, fenics.Constant(0.0), "near(x[0], {})".format(self.problem.a1))
            elif i == 3:  # top
                bc = fenics.DirichletBC(self.V, fenics.Constant(0.0), "near(x[1], {})".format(self.problem.b2))
            elif i == 4:  # right
                bc = fenics.DirichletBC(self.V, fenics.Constant(0.0), "near(x[0], {})".format(self.problem.a2))
            self.dirichlet_conditions.append(bc)

    def _setup_forms(self):
        y = fenics.TrialFunction(self.V)
        v = fenics.TestFunction(self.V)

        # Bilinear forms
        self.a1_form = fenics.dot(fenics.grad(y), fenics.grad(v)) * fenics.dx
        self.a2_form = y * v * fenics.dx

        # Linear form l
        self.l_form = self.problem.f * v * fenics.dx

        # Neumann BC
        if self.problem.gamma_N:
            for i in self.problem.gamma_N:
                if i == 1:  # bottom
                    self.l_form += self.problem.gN * v * fenics.ds(1)
                elif i == 2:  # left
                    self.l_form += self.problem.gN * v * fenics.ds(2)
                elif i == 3:  # top
                    self.l_form += self.problem.gN * v * fenics.ds(3)
                elif i == 4:  # right
                    self.l_form += self.problem.gN * v * fenics.ds(4)

        # Control form b
        self.b_form = y * v * fenics.dx  # since B is identity for L2 control

    def _assemble_matrices(self):
        # Assemble A1, A2, L
        self.A1 = csr_matrix(fenics.as_backend_type(fenics.assemble(self.a1_form)).mat().getValuesCSR()[::-1])
        self.A2 = csr_matrix(fenics.as_backend_type(fenics.assemble(self.a2_form)).mat().getValuesCSR()[::-1])
        self.L = fenics.assemble(self.l_form).get_local()

        # B is mass matrix for L2 control
        self.B = self.A2.copy()

        # Apply Dirichlet BCs
        for bc in self.dirichlet_conditions:
            bc.apply(fenics.as_backend_type(self.A1).mat())
            bc.apply(fenics.as_backend_type(self.A2).mat())
            bc.apply(self.L)

        # Convert to sparse
        self.A1 = csr_matrix(fenics.as_backend_type(self.A1).mat().getValuesCSR()[::-1])
        self.A2 = csr_matrix(fenics.as_backend_type(self.A2).mat().getValuesCSR()[::-1])

    def solution_operator(self, u):
        """
        Solve (kappa A1 + c A2) y = L + B u
        """
        A = self.problem.kappa * self.A1 + self.problem.c * self.A2
        rhs = self.L + self.B @ u
        y = spsolve(A, rhs)
        return y


# Test problems
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Problem 1
    prob1 = analytical_problem(0, 0, 0.5, 0.8, [1,2,3,4], [], fenics.Expression("1 - (x[0]-0.5)*(x[1]-0.5)", degree=1), fenics.Constant(0.0), 0, 1)
    model1 = elliptic_model(prob1)

    # Control u=0
    u0 = np.zeros(model1.dofs)
    y0 = model1.solution_operator(u0)

    # Control u = -10 on [0,0.25)x[0,0.8] + 10 on [0.25,0.5)x[0,0.8]
    u_expr = fenics.Expression("-10*(x[0] >= 0 && x[0] < 0.25) + 10*(x[0] >= 0.25 && x[0] < 0.5)", degree=0)
    u_func = fenics.interpolate(u_expr, model1.V)
    u1 = u_func.vector().get_local()
    y1 = model1.solution_operator(u1)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fenics.plot(fenics.Function(model1.V, y0), ax=axes[0], title="u=0")
    fenics.plot(fenics.Function(model1.V, y1), ax=axes[1], title="u=piecewise")
    plt.show()

    # Problem 2
    prob2 = analytical_problem(0, 0, 1, 1, [1,2,4], [3], fenics.Constant(0.0), fenics.Expression("x[0]*x[0]", degree=2), 1, 1)
    model2 = elliptic_model(prob2)

    # u=0
    u0_2 = np.zeros(model2.dofs)
    y0_2 = model2.solution_operator(u0_2)

    # u = 50 sin(4 pi x) cos(4 pi y)
    u_expr2 = fenics.Expression("50*sin(4*M_PI*x[0])*cos(4*M_PI*x[1])", degree=3)
    u_func2 = fenics.interpolate(u_expr2, model2.V)
    u2_2 = u_func2.vector().get_local()
    y2_2 = model2.solution_operator(u2_2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fenics.plot(fenics.Function(model2.V, y0_2), ax=axes[0], title="u=0")
    fenics.plot(fenics.Function(model2.V, y2_2), ax=axes[1], title="u=sin cos")
    plt.show()