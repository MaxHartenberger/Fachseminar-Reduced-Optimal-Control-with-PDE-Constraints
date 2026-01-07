#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 1.2: Separate forms, assemble matrices, and provide a solution operator.
This file keeps ex1_Hartenberger.py unchanged and implements the task-sheet structure here.
"""

import fenics as fe
import numpy as np

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


class AnalyticalProblem:
    """
    Data holder for domain, boundary parts, and PDE data.
    - Domain: [a1, a2] x [b1, b2]
    - gamma_D: list of boundary ids with Dirichlet (ids 1: left, 2: right, 3: bottom, 4: top)
    - gamma_N: list of boundary ids with Neumann
    - f, gN: FEniCS Expressions
    - c, kappa: constants
    """
    def __init__(self, a1, a2, b1, b2, gamma_D, gamma_N, f_expr, gN_expr, c, kappa):
        self.a1, self.a2 = a1, a2
        self.b1, self.b2 = b1, b2
        self.gamma_D = gamma_D or []
        self.gamma_N = gamma_N or []
        self.f = f_expr
        self.gN = gN_expr
        self.c = float(c)
        self.kappa = float(kappa)


class EllipticModel:
    """
    Finite element model implementing the task-sheet structure:
    - boundary marking
    - a(y,v) = kappa*a1(y,v) + c*a2(y,v)
    - l(v) with f and Neumann terms
    - optional B as mass matrix and solution operator on vectors
    """
    def __init__(self, problem: AnalyticalProblem, h=0.005, degree=2):
        self.p = problem
        self.h = h
        self.degree = degree

        # Mesh and space
        self.mesh = fe.RectangleMesh(fe.Point(self.p.a1, self.p.b1),
                                     fe.Point(self.p.a2, self.p.b2),
                                     int((self.p.a2 - self.p.a1)/h),
                                     int((self.p.b2 - self.p.b1)/h))
        self.V = fe.FunctionSpace(self.mesh, "P", 1)

        # Boundary markers and measures
        self.boundaries = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)
        TOL = 1e-14
        class Left(fe.SubDomain):
            def inside(_, x, on_boundary):
                return on_boundary and fe.near(x[0], self.p.a1, TOL)
        class Right(fe.SubDomain):
            def inside(_, x, on_boundary):
                return on_boundary and fe.near(x[0], self.p.a2, TOL)
        class Bottom(fe.SubDomain):
            def inside(_, x, on_boundary):
                return on_boundary and fe.near(x[1], self.p.b1, TOL)
        class Top(fe.SubDomain):
            def inside(_, x, on_boundary):
                return on_boundary and fe.near(x[1], self.p.b2, TOL)
        Left().mark(self.boundaries, 1)
        Right().mark(self.boundaries, 2)
        Bottom().mark(self.boundaries, 3)
        Top().mark(self.boundaries, 4)
        self.ds = fe.Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)

        # Dirichlet BCs (homogeneous)
        self.bcs = [fe.DirichletBC(self.V, fe.Constant(0.0), self.boundaries, m) for m in self.p.gamma_D]

        # Forms
        y = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)
        self.a1_form = fe.dot(fe.grad(y), fe.grad(v)) * fe.dx
        self.a2_form = y * v * fe.dx
        # l(v): source + Neumann
        self.l_form = self.p.f * v * fe.dx
        for gid in self.p.gamma_N:
            self.l_form += self.p.gN * v * self.ds(gid)
        # b(u,v): conceptual; discretely B equals mass matrix
        self.b_form_symbolic = None  # not used directly, see compute_Bu()

        # Assemble core matrices and vector (dolfin types)
        self.A1 = fe.assemble(self.a1_form)
        self.A2 = fe.assemble(self.a2_form)
        self.L = fe.assemble(self.l_form)

    def plot_mesh(self, fname="mesh.pdf"):
        try:
            import matplotlib.pyplot as plt
            fe.plot(self.mesh)
            plt.title("Mesh")
            plt.savefig(fname, bbox_inches='tight')
            plt.close()
        except Exception:
            pass

    def plot_surface(self, y_fun, resolution=100, title="Solution y(x,y)"):
        """Plot a 3D surface of the FE function over the rectangular domain and save a PDF."""
        import numpy as np
        import matplotlib.pyplot as plt
        x = np.linspace(self.p.a1, self.p.a2, resolution)
        y = np.linspace(self.p.b1, self.p.b2, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(lambda xx, yy: float(y_fun(xx, yy)))(X, Y)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        fig.colorbar(surf, shrink=0.5, aspect=10)
        fig.savefig(f"{title}.pdf", bbox_inches='tight')
        plt.close(fig)

    def interpolate_control(self, u_expr_str):
        """Interpolate control expression to Uh and return Function and coefficient vector."""
        uh_fun = fe.interpolate(fe.Expression(u_expr_str, degree=self.degree), self.V)
        return uh_fun, uh_fun.vector().get_local()

    def compute_B(self):
        """Mass matrix B (same as a2_form matrix)."""
        return fe.assemble(self.a2_form)

    def compute_Bu(self, uh_fun):
        """Compute Bu via assemble of b(uh, v) = ∫ uh v dx (preferred in FEniCS)."""
        v = fe.TestFunction(self.V)
        bu_vec = fe.assemble(uh_fun * v * fe.dx)
        return bu_vec

    def solution_operator(self, uh_fun=None, u_vec=None, use_scipy=False):
        """
        Solve (kappa*A1 + c*A2) y = L + B u with homogeneous Dirichlet BCs applied.
        - Provide either `uh_fun` (FEniCS Function) or `u_vec` (numpy vector). If `u_vec` is provided,
          we multiply by B (mass matrix). If `uh_fun` is provided, we assemble b(uh,v) directly.
        - use_scipy: if True and SciPy is available, export to CSR and solve via spsolve; otherwise use FEniCS.
        Returns a FEniCS Function y_h.
        """
        # Build A and Bu
        A = self.p.kappa * self.A1 + self.p.c * self.A2
        if uh_fun is not None:
            Bu = self.compute_Bu(uh_fun)
        else:
            # Multiply mass matrix B with u_vec
            B = self.compute_B()
            # PETSc multiplication: create a dolfin Vector and set values
            u_dof = fe.Function(self.V)
            u_dof.vector().set_local(u_vec if u_vec is not None else np.zeros(self.V.dim()))
            u_dof.vector().apply("insert")
            # Assemble Bu via form for robustness (avoids manual Mat*vec conversions)
            v = fe.TestFunction(self.V)
            Bu = fe.assemble(u_dof * v * fe.dx)

        rhs = self.L.copy()
        rhs.axpy(1.0, Bu)  # rhs = L + Bu

        # Apply BCs to A and rhs
        for bc in self.bcs:
            bc.apply(A, rhs)

        # Solve
        y_fun = fe.Function(self.V)
        if use_scipy and SCIPY_AVAILABLE:
            # Export A to CSR and rhs to numpy, solve, then set back into y_fun
            from dolfin import as_backend_type
            petsc_A = as_backend_type(A).mat()
            IA, JA, AA = petsc_A.getValuesCSR()
            A_csr = csr_matrix((AA, JA, IA), shape=(self.V.dim(), self.V.dim()))
            rhs_np = rhs.get_local()
            sol_np = spsolve(A_csr, rhs_np)
            y_fun.vector().set_local(sol_np)
            y_fun.vector().apply("insert")
        else:
            fe.solve(A, y_fun.vector(), rhs)

        return y_fun


def run_exercises():
    """Run the four problems as in the task sheet and plot 3D surface results."""

    # 1.1 and 1.2 on [0,0.5]x[0,0.8], Dirichlet everywhere
    f_expr = fe.Expression("1 - (x[0] - 0.5)*(x[1] - 0.5)", degree=2)
    gN0 = fe.Expression("0.0", degree=1)
    prob1 = AnalyticalProblem(0.0, 0.5, 0.0, 0.8, gamma_D=[1,2,3,4], gamma_N=[], f_expr=f_expr, gN_expr=gN0, c=0.0, kappa=1.0)
    model1 = EllipticModel(prob1, h=0.005)
    # u = 0
    u0_fun = fe.interpolate(fe.Expression("0.0", degree=0), model1.V)
    y0 = model1.solution_operator(uh_fun=u0_fun)
    # piecewise control
    u_expr = "(x[0] <= 0.25) ? -10.0 : (x[0] > 0.25 && x[0] <= 0.5) ? 10.0 : 0.0"
    u1_fun = fe.interpolate(fe.Expression(u_expr, degree=0), model1.V)
    y1 = model1.solution_operator(uh_fun=u1_fun)

    # 3D surface plots like in ex1_Hartenberger.py
    model1.plot_surface(y0, title="Problem 1")
    model1.plot_surface(y1, title="Problem 2")

    # 2.1 and 2.2 on [0,1]^2, Neumann on right edge (x=1), Dirichlet elsewhere
    prob2 = AnalyticalProblem(0.0, 1.0, 0.0, 1.0, gamma_D=[1,3,4], gamma_N=[2], f_expr=fe.Expression("0.0", degree=0), gN_expr=fe.Expression("x[0]*x[0]", degree=2), c=1.0, kappa=1.0)
    model2 = EllipticModel(prob2, h=0.005)
    # u = 0
    u0_fun2 = fe.interpolate(fe.Expression("0.0", degree=0), model2.V)
    y0_2 = model2.solution_operator(uh_fun=u0_fun2)
    # oscillatory control
    u_expr2 = "50*sin(4*pi*x[0])*cos(4*pi*x[1])"
    u2_fun = fe.interpolate(fe.Expression(u_expr2, degree=4), model2.V)
    y2_2 = model2.solution_operator(uh_fun=u2_fun)

    model2.plot_surface(y0_2, title="Problem 3")
    model2.plot_surface(y2_2, title="Problem 4")


if __name__ == "__main__":
    # Optional: quick mesh plot for the first problem
    # You can comment this out if not needed.
    run_exercises()