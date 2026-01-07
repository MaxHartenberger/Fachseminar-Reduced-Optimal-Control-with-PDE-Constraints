#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduced optimal control model for the Ex3 problem (control-reduced formulation).
Provides: state(), adjoint(), cost(), grad_U(), hess_U() and Lipschitz estimator.
"""

import fenics as fe
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu


class ReducedOCModel:
    """
    - Domain Ω=[0,1]^2
    - V = P1 with homogeneous Dirichlet (state/adjoint)
    - U = H = L2 with P1 coefficients and L2 inner product MU (no BCs applied)
    - PDE: -Δ y = B u with B = χ_ω (circle center (0.5,0.5), radius r)
    - Target: y_d(x) = 0.5 - max(|x-0.5|, |y-0.5|)
    """

    def __init__(self, h: float = 0.02, beta: float = 1e-3, radius: float = 0.1, degree: int = 1):
        self.h = h
        self.beta = float(beta)
        self.radius = float(radius)
        self.degree = degree
        self._build_spaces()
        self._assemble_operators()

    # ------------------------ assembly ------------------------
    def _build_spaces(self):
        nx = max(2, int(1.0 / self.h))
        ny = max(2, int(1.0 / self.h))
        self.mesh = fe.RectangleMesh(fe.Point(0.0, 0.0), fe.Point(1.0, 1.0), nx, ny)
        self.V = fe.FunctionSpace(self.mesh, 'P', 1)
        self.n = self.V.dim()
        self.bc = fe.DirichletBC(self.V, fe.Constant(0.0), 'on_boundary')

    def _assemble_operators(self):
        y = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)

        # Dolfin matrices
        A_d = fe.assemble(fe.dot(fe.grad(y), fe.grad(v)) * fe.dx)
        M_d = fe.assemble(y * v * fe.dx)
        omega = fe.Expression(
            "((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) < R*R) ? 1.0 : 0.0",
            degree=max(1, self.degree), R=self.radius
        )
        B_d = fe.assemble(omega * y * v * fe.dx)

        # Apply BC to A only; MU and B are left untouched
        self.bc.apply(A_d)

        # Convert to SciPy CSR
        from dolfin import as_backend_type
        def to_csr(mat):
            petsc = as_backend_type(mat).mat()
            IA, JA, AA = petsc.getValuesCSR()
            return csr_matrix((AA, JA, IA), shape=(mat.size(0), mat.size(1)))

        self.A = to_csr(A_d)
        self.MU = to_csr(M_d)
        self.B = to_csr(B_d)

        # Dirichlet dofs for RHS zeroing
        self.dir_dofs = np.array(list(self.bc.get_boundary_values().keys()), dtype=int)

        # Factorizations
        self.A_fac = splu(self.A.tocsc())
        self.MU_fac = splu(self.MU.tocsc())

        # Target y_d
        yd_expr = fe.Expression("0.5 - fmax(fabs(x[0]-0.5), fabs(x[1]-0.5))", degree=max(1, self.degree))
        yd_fun = fe.interpolate(yd_expr, self.V)
        self.y_d = yd_fun.vector().get_local()

    # ------------------------ PDE solves ------------------------
    def _apply_bc_rhs(self, rhs: np.ndarray) -> np.ndarray:
        if self.dir_dofs.size:
            rhs = rhs.copy()
            rhs[self.dir_dofs] = 0.0
        return rhs

    def state(self, u: np.ndarray) -> np.ndarray:
        rhs = self.B @ u
        rhs = self._apply_bc_rhs(rhs)
        return self.A_fac.solve(rhs)

    def adjoint(self, z: np.ndarray) -> np.ndarray:
        rhs = self._apply_bc_rhs(z)
        return self.A_fac.solve(rhs)

    # ------------------------ objective & derivatives ------------------------
    def cost(self, u: np.ndarray) -> float:
        y = self.state(u)
        d = y - self.y_d
        term_state = 0.5 * float(d.T @ (self.MU @ d))
        term_ctrl = 0.5 * self.beta * float(u.T @ (self.MU @ u))
        return term_state + term_ctrl

    def grad_U(self, u: np.ndarray) -> np.ndarray:
        y = self.state(u)
        z = self.MU @ (y - self.y_d)
        p = self.adjoint(z)
        Bt_p = self.B.transpose() @ p
        return self.MU_fac.solve(Bt_p) + self.beta * u

    def hess_U(self, v: np.ndarray) -> np.ndarray:
        # H v = MU^{-1} B^T A^{-1} MU A^{-1} B v + beta v
        w = self.A_fac.solve(self.B @ v)      # w = A^{-1} B v
        q = self.A_fac.solve(self.MU @ w)     # q = A^{-1} MU w
        Bt_q = self.B.transpose() @ q
        return self.MU_fac.solve(Bt_q) + self.beta * v

    # ------------------------ inner products & L estimator ------------------------
    def dot_U(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(a.T @ (self.MU @ b))

    def norm_U(self, a: np.ndarray) -> float:
        val = self.dot_U(a, a)
        return float(np.sqrt(max(val, 0.0)))

    def estimate_L(self, iters: int = 20, tol: float = 1e-8, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(self.n)
        x /= (self.norm_U(x) + 1e-16)
        lam_old = 0.0
        for _ in range(iters):
            y = self.hess_U(x)
            lam = self.dot_U(x, y) / max(self.dot_U(x, x), 1e-16)
            if abs(lam - lam_old) <= tol * max(1.0, abs(lam_old)):
                return float(max(lam, 0.0))
            lam_old = lam
            # U-orthonormalize next iterate
            x = y / (self.norm_U(y) + 1e-16)
        return float(max(lam_old, 0.0))
