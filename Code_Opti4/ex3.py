#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 3: Control-reduced approach with Barzilai–Borwein method
- Correct BC handling (apply to A and RHS only)
- Separate L2 inner product for U/H (mass matrix without BC)
- Gradient in U: ∇F(u) = M_U^{-1} B^T p + β u
- Proper BB step sizes and U-inner-product norms
"""

import fenics as fe
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, splu
import matplotlib.pyplot as plt


class elliptic_control_model:
    """
    - PDE: -Δ y = B u in Ω, y=0 on ∂Ω
    - Spaces: V=P1 with homogeneous Dirichlet for state/adjoint; U=H=L2 with P1 coefficients
    - B = χ_ω, ω: circle centered at (0.5,0.5) radius 0.1
    """
    def __init__(self, dx=0.01, beta=1e-3):
        self.dx = dx
        self.beta = beta
        self.a1, self.b1, self.a2, self.b2 = 0.0, 0.0, 1.0, 1.0
        self._setup()

    def _setup(self):
        # Mesh and space
        nx = max(2, int((self.a2 - self.a1) / self.dx))
        ny = max(2, int((self.b2 - self.b1) / self.dx))
        self.mesh = fe.RectangleMesh(fe.Point(self.a1, self.b1), fe.Point(self.a2, self.b2), nx, ny)
        self.V = fe.FunctionSpace(self.mesh, 'P', 1)
        self.n = self.V.dim()

        # Forms
        y = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)

        # Dolfin matrices
        A_d = fe.assemble(fe.dot(fe.grad(y), fe.grad(v)) * fe.dx)
        M_d = fe.assemble(y * v * fe.dx)
        omega = fe.Expression("((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) < 0.1*0.1) ? 1.0 : 0.0", degree=1)
        B_d = fe.assemble(omega * y * v * fe.dx)

        # Dirichlet BC and dirichlet dofs
        self.bc = fe.DirichletBC(self.V, fe.Constant(0.0), "on_boundary")
        # Apply BC to A only
        self.bc.apply(A_d)

        # Convert to SciPy CSR
        from dolfin import as_backend_type
        def to_csr(mat):
            petsc = as_backend_type(mat).mat()
            IA, JA, AA = petsc.getValuesCSR()
            return csr_matrix((AA, JA, IA), shape=(mat.size(0), mat.size(1)))

        self.A = to_csr(A_d).tocsc()
        # U/H mass matrix without BC application
        self.MU = to_csr(M_d).tocsc()
        # B maps U->V; do not BC-apply B
        self.B = to_csr(B_d).tocsc()

        # Dirichlet dof indices for RHS zeroing
        self.dir_dofs = np.array(list(self.bc.get_boundary_values().keys()), dtype=int)

        # Factorizations for repeated solves
        self.A_fac = splu(self.A)
        self.MU_fac = splu(self.MU)

        # Target y_d
        yd_expr = fe.Expression("0.5 - fmax(fabs(x[0]-0.5), fabs(x[1]-0.5))", degree=1)
        yd_fun = fe.interpolate(yd_expr, self.V)
        self.yd = yd_fun.vector().get_local()

    def _apply_dirichlet_to_rhs(self, rhs):
        if self.dir_dofs.size:
            rhs = rhs.copy()
            rhs[self.dir_dofs] = 0.0
        return rhs

    def solve_state(self, u):
        """Solve A y = B u with homogeneous Dirichlet via zeroed RHS at boundary dofs."""
        rhs = self.B @ u
        rhs = self._apply_dirichlet_to_rhs(rhs)
        y = self.A_fac.solve(rhs)
        return y

    def solve_adjoint(self, z):
        """Solve A p = z with homogeneous Dirichlet via zeroed RHS at boundary dofs."""
        rhs = self._apply_dirichlet_to_rhs(z)
        p = self.A_fac.solve(rhs)
        return p

    def cost(self, u):
        """F(u) = 1/2 ||S u - y_d||_H^2 + (β/2) ||u||_U^2 with H=U=L2, both via MU."""
        y = self.solve_state(u)
        diff = y - self.yd
        term1 = 0.5 * diff.T @ (self.MU @ diff)
        term2 = 0.5 * self.beta * (u.T @ (self.MU @ u))
        return float(term1 + term2)

    def gradient_U(self, u):
        """
        Return the gradient in U (with respect to (.,.)_U): ∇F(u) = MU^{-1} B^T p + β u,
        where A p = MU (S u - y_d).
        """
        y = self.solve_state(u)
        z = self.MU @ (y - self.yd)
        p = self.solve_adjoint(z)
        Bt_p = self.B.transpose() @ p
        g = self.MU_fac.solve(Bt_p) + self.beta * u
        return g


def barzilai_borwein(model: elliptic_control_model, tol=1e-8, max_iter=1000):
    """Barzilai–Borwein in U with (.,.)_U induced by MU."""
    MU = model.MU
    ip = lambda a, b: float(a.T @ (MU @ b))
    normU = lambda a: np.sqrt(max(ip(a, a), 0.0))

    u_m1 = np.zeros(model.n)                 # u_{-1}
    g_m1 = model.gradient_U(u_m1)            # ∇F(u_{-1})
    u = g_m1.copy()                           # u_0 = ∇F(0)
    g = model.gradient_U(u)                  # ∇F(u_0)

    grad_norms = [normU(g_m1)]
    k = 0
    while normU(g) > tol and k < max_iter:
        s = u - u_m1                          # s_k
        d = g - g_m1                          # d_k
        sd = ip(s, d)
        if abs(sd) < 1e-30:
            alpha = 1.0
        else:
            if k % 2 == 0:
                alpha = ip(d, d) / sd
            else:
                alpha = sd / ip(s, s)

        # Update: u_{k+1} = u_k - (1/α_k) ∇F(u_k)
        u_next = u - (1.0 / alpha) * g

        # Shift iterates
        u_m1, u = u, u_next
        g_m1, g = g, model.gradient_U(u)
        grad_norms.append(normU(g))
        k += 1

    return u, grad_norms


if __name__ == "__main__":
    model = elliptic_control_model(dx=0.02, beta=1e-3)

    # Solve with BB
    u_opt, grad_norms = barzilai_borwein(model, tol=1e-8, max_iter=500)

    # Plot convergence in semilog scale
    plt.figure()
    plt.semilogy(grad_norms)
    plt.xlabel('Iteration')
    plt.ylabel('||∇F(u_k)||_U')
    plt.title('Convergence of Barzilai–Borwein')
    plt.tight_layout(); plt.show()

    # Plot optimal u and y
    y_opt = model.solve_state(u_opt)
    uf = fe.Function(model.V); uf.vector().set_local(u_opt)
    yf = fe.Function(model.V); yf.vector().set_local(y_opt)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plt.sca(axes[0]); fe.plot(uf); axes[0].set_title('Optimal control u')
    plt.sca(axes[1]); fe.plot(yf); axes[1].set_title('Optimal state y')
    plt.tight_layout(); plt.show()