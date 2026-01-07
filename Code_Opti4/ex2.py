#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 2: Discretization and solution of LQ optimal control problem
Implements:
- Assembly of A, M on V and K on U, and coupling P (U->V)
- Clearing Dirichlet dofs in matrices/vectors per task (2c)
- Target generation yd = S(ud) for ud ≡ 1 and sin-cos (2d)
- KKT solver for (2e) using sparse factorization; visualization and norms
"""

import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix, bmat
from scipy.sparse.linalg import spsolve, splu


class EllipticOCModel:
    """
    - Domain Ω=[a1,a2]x[b1,b2]
    - V = P1 (state/adjoint), U = DG0 (control)
    - PDE: find y in V with y|∂Ω=0 s.t. ∫Ω ∇y·∇v = ∫Ω u v
    """
    def __init__(self, a1=0.0, a2=1.0, b1=0.0, b2=1.0, h=0.02):
        self.a1, self.a2, self.b1, self.b2 = a1, a2, b1, b2
        self.h = h
        self._build_spaces()
        self._assemble_operators()

    def _build_spaces(self):
        nx = max(2, int((self.a2 - self.a1) / self.h))
        ny = max(2, int((self.b2 - self.b1) / self.h))
        self.mesh = fe.RectangleMesh(fe.Point(self.a1, self.b1), fe.Point(self.a2, self.b2), nx, ny)
        self.V = fe.FunctionSpace(self.mesh, 'P', 1)
        self.U = fe.FunctionSpace(self.mesh, 'DG', 0)
        self.bc = fe.DirichletBC(self.V, fe.Constant(0.0), 'on_boundary')
        self.nV = self.V.dim()
        self.nU = self.U.dim()

    def _assemble_operators(self):
        y = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)
        u = fe.TrialFunction(self.U)
        w = fe.TestFunction(self.U)

        # Assemble dolfin matrices
        A = fe.assemble(fe.dot(fe.grad(y), fe.grad(v)) * fe.dx)      # VxV
        M = fe.assemble(y * v * fe.dx)                               # VxV
        K = fe.assemble(u * w * fe.dx)                               # UxU
        P = fe.assemble(u * v * fe.dx)                               # VxU

        # Apply BCs to A and M
        self.bc.apply(A)
        self.bc.apply(M)

        # Convert to SciPy
        from dolfin import as_backend_type
        def to_csr(mat, shape=None):
            petsc = as_backend_type(mat).mat()
            IA, JA, AA = petsc.getValuesCSR()
            return csr_matrix((AA, JA, IA), shape=shape if shape else (mat.size(0), mat.size(1)))

        A_csr = to_csr(A, (self.nV, self.nV))
        M_csr = to_csr(M, (self.nV, self.nV))
        K_csr = to_csr(K, (self.nU, self.nU))
        P_csr = to_csr(P, (self.nV, self.nU))

        # Clear Dirichlet rows in P (map into free V dofs only)
        dir_dofs = np.array(list(self.bc.get_boundary_values().keys()), dtype=int)
        if dir_dofs.size > 0:
            P_csr = P_csr.tolil()
            P_csr[dir_dofs, :] = 0.0
            P_csr = P_csr.tocsr()

        self.A = A_csr
        self.M = M_csr
        self.K = K_csr
        self.P = P_csr

    def interpolate_u(self, expr_str, degree=3):
        u_fun = fe.interpolate(fe.Expression(expr_str, degree=degree, pi=np.pi), self.U)
        return u_fun.vector().get_local(), u_fun

    def constant_u(self, val):
        u_fun = fe.interpolate(fe.Constant(val), self.U)
        return u_fun.vector().get_local(), u_fun

    def state_from_u(self, u_vec):
        rhs = self.P @ u_vec
        y = spsolve(self.A, rhs)
        return y

    def plot_function_V(self, vec, title):
        f = fe.Function(self.V)
        f.vector().set_local(vec)
        fe.plot(f)
        plt.title(title)
        plt.show()

    def plot_function_U(self, vec, title):
        f = fe.Function(self.U)
        f.vector().set_local(vec)
        fe.plot(f)
        plt.title(title)
        plt.show()


def solve_kkt(model: EllipticOCModel, yd_vec: np.ndarray, sigma: float):
    """
    Solve KKT in reduced form using elimination of u:
      A y + (1/s) P K^{-1} P^T p = 0
      A^T p - M y = - M yd
    Return y, u, p (vectors).
    """
    A = model.A.tocsc()
    M = model.M.tocsc()
    P = model.P.tocsc()
    K = model.K.tocsc()

    # Factorize K once and build B = (1/sigma) P K^{-1} P^T
    K_fac = splu(K)
    PT = P.transpose().tocsc()
    # Solve K X = P^T  => X = K^{-1} P^T
    X = K_fac.solve(PT.toarray())  # dense solve on multiple RHS; acceptable for moderate sizes
    B = (1.0 / sigma) * (P @ X)    # VxV

    n = model.nV
    Z = csc_matrix((n, n))
    KKT = bmat([[A, B], [-M, A.T]], format='csc')
    rhs = np.concatenate([np.zeros(n), -(M @ yd_vec)])

    sol = spsolve(KKT, rhs)
    y = sol[:n]
    p = sol[n:]
    # Recover u = - (1/sigma) K^{-1} P^T p
    u = - (1.0 / sigma) * K_fac.solve(PT @ p)
    return y, u, p


if __name__ == "__main__":
    model = EllipticOCModel(h=0.02)

    # d) Targets yd = S(ud)
    u1_vec, _ = model.constant_u(1.0)
    yd1 = model.state_from_u(u1_vec)

    u2_vec, _ = model.interpolate_u("sin(2*pi*x[0]) * cos(2*pi*x[1])", degree=4)
    yd2 = model.state_from_u(u2_vec)

    # e) Solve optimal control for a range of sigma
    sigmas = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    for yd_vec, label in [(yd1, "yd1"), (yd2, "yd2")]:
        print(f"\nFor {label}:")
        for s in sigmas:
            y_opt, u_opt, p_opt = solve_kkt(model, yd_vec, s)
            # ||u||_L2 = sqrt(u^T K u)
            norm_u = np.sqrt(u_opt.T @ (model.K @ u_opt))
            print(f"sigma={s}: ||u||_L2 = {norm_u:.6f}")

            if abs(s - 1e-3) < 1e-12:
                # Plot representative results
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                yf = fe.Function(model.V); yf.vector().set_local(y_opt)
                uf = fe.Function(model.U); uf.vector().set_local(u_opt)
                pf = fe.Function(model.V); pf.vector().set_local(p_opt)
                plt.sca(axes[0]); fe.plot(yf); axes[0].set_title(f"y_opt {label}")
                plt.sca(axes[1]); fe.plot(uf); axes[1].set_title(f"u_opt {label}")
                plt.sca(axes[2]); fe.plot(pf); axes[2].set_title(f"p_opt {label}")
                plt.tight_layout(); plt.show()