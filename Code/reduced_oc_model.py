#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduced optimal control model using an external XDMF mesh with a circular
subdomain (omega) tagged in the cell data.

Overview
========
- Domain Ω is provided by the external mesh (typically Ω=[0,1]^2).
- V = P1 with homogeneous Dirichlet BCs for state/adjoint.
- U = H = L2 with P1 coefficients and L2 inner product MU (no BCs applied).
- PDE: -Δ y = B u with B = χ_ω (ω is the tagged circular region in cell data).
- Target: y_d(x) = 0.5 - max(|x-0.5|, |y-0.5|) (pyramid-like profile).

This implementation mirrors the API of the internal-mesh model and is
compatible with Code_v2/optimizers.py.

Control-Reduced Notes (from Ex3)
--------------------------------
- Correct BC handling: apply Dirichlet BC to the stiffness matrix `A` and zero
    RHS entries at boundary DOFs for state/adjoint solves; do not BC-apply `B` or
    the control-space mass `MU`.
- Separate L2 inner product for U/H: the control space uses the L2 mass matrix
    `MU` without BCs as its inner product and Riesz map.
- Gradient in U: ∇F(u) = MU^{-1} B^T p + β u, with adjoint defined by
    A p = M (y - y_d) in this external-mesh variant.
- Barzilai–Borwein (BB): use proper BB1/BB2 stepsizes and norms induced by `MU`.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu, eigsh, LinearOperator

import fenics as fe
from dolfin import Mesh, MeshValueCollection, MeshFunction, XDMFFile, as_backend_type


def _to_csr(mat):
    petsc = as_backend_type(mat).mat()
    IA, JA, AA = petsc.getValuesCSR()
    return csr_matrix((AA, JA, IA), shape=(mat.size(0), mat.size(1)))


def load_external_mesh(mesh_cells_xdmf: str, subdomain_name: str = "subdomains"):
    mesh = Mesh()
    with XDMFFile(mesh_cells_xdmf) as xdmf:
        xdmf.read(mesh)
    mvc_cells = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile(mesh_cells_xdmf) as xdmf:
        xdmf.read(mvc_cells, subdomain_name)
    subdomains = MeshFunction("size_t", mesh, mvc_cells)
    return mesh, subdomains


class ReducedOCModelExternal:
    """
    - Domain Ω from external XDMF mesh
    - V = P1 with homogeneous Dirichlet (state/adjoint)
    - U = H = L2 with P1 coefficients and L2 inner product MU (no BCs applied)
    - PDE: -Δ y = B u with B = χ_ω (subdomain tag `omega_id` in cells)
    - Target: y_d(x) = 0.5 - max(|x-0.5|, |y-0.5|)
    """
    def __init__(self, mesh_cells_xdmf: str, omega_id: int = 2, beta: float = 1e-3, degree: int = 1):
        self.beta = float(beta)
        self.degree = int(degree)
        self.mesh, self.subdomains = load_external_mesh(mesh_cells_xdmf)
        self.omega_id = int(omega_id)

        # Quick Reference (Ex3)
        # - PDE: -Δ y = B u in Ω, y=0 on ∂Ω
        # - Spaces: V=P1 with homogeneous Dirichlet (state/adjoint);
        #           U=H=L2 with P1 coefficients (no BCs), inner product via MU
        # - Control region: B = χ_ω implemented by dx(omega_id, subdomain_data=subdomains)
        # - Gradient in U: ∇F(u) = MU^{-1} B^T p + β u with A p = M (y - y_d)

        self.V = fe.FunctionSpace(self.mesh, 'P', 1)
        self.n = self.V.dim()
        self.bc = fe.DirichletBC(self.V, fe.Constant(0.0), 'on_boundary')

        y = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)
        uU = fe.TrialFunction(self.V)
        wU = fe.TestFunction(self.V)

        # Dolfin bilinear forms
        a = fe.dot(fe.grad(y), fe.grad(v)) * fe.dx
        M = y * v * fe.dx
        B_form = uU * v * fe.dx(self.omega_id, subdomain_data=self.subdomains)
        MU_form = uU * wU * fe.dx

        # Assemble and apply BCs to A only; MU and B are left untouched
        A_d = fe.assemble(a)
        self.bc.apply(A_d)
        M_d = fe.assemble(M)
        B_d = fe.assemble(B_form)
        MU_d = fe.assemble(MU_form)

        # Convert to SciPy CSR
        self.A = _to_csr(A_d)
        self.M = _to_csr(M_d)
        self.B = _to_csr(B_d)
        self.MU = _to_csr(MU_d)

        # Dirichlet dofs for RHS zeroing and factorizations
        self.dir_dofs = np.array(list(self.bc.get_boundary_values().keys()), dtype=int)
        self.A_fac = splu(self.A.tocsc())
        self.MU_fac = splu(self.MU.tocsc())

        # Target y_d
        yd_expr = fe.Expression("0.5 - fmax(fabs(x[0]-0.5), fabs(x[1]-0.5))", degree=max(1, self.degree))
        yd_fun = fe.interpolate(yd_expr, self.V)
        self.y_d = yd_fun.vector().get_local()

    def _apply_bc_rhs(self, rhs: np.ndarray) -> np.ndarray:
        if self.dir_dofs.size:
            rhs = rhs.copy()
            rhs[self.dir_dofs] = 0.0
        return rhs

    # ------------------------ PDE solves ------------------------
    def state(self, u: np.ndarray) -> np.ndarray:
        """Solve A y = B u with homogeneous Dirichlet via zeroed RHS at boundary DOFs."""
        rhs = self.B @ u
        rhs = self._apply_bc_rhs(rhs)
        return self.A_fac.solve(rhs)

    def adjoint(self, z: np.ndarray) -> np.ndarray:
        """Solve A p = z with homogeneous Dirichlet via zeroed RHS at boundary DOFs."""
        rhs = self._apply_bc_rhs(z)
        return self.A_fac.solve(rhs)

    # ------------------------ objective & derivatives ------------------------
    def cost(self, u: np.ndarray) -> float:
        y = self.state(u)
        d = y - self.y_d
        term_state = 0.5 * float(d.T @ (self.M @ d))
        term_ctrl = 0.5 * self.beta * float(u.T @ (self.MU @ u))
        return term_state + term_ctrl

    def grad_U(self, u: np.ndarray) -> np.ndarray:
        """
        Gradient in U (with respect to (.,.)_U): ∇F(u) = MU^{-1} B^T p + β u,
        where the adjoint solves A p = M (y - y_d) in this external-mesh variant.
        """
        y = self.state(u)
        z = self.M @ (y - self.y_d)
        p = self.adjoint(z)
        Bt_p = self.B.transpose() @ p
        return self.MU_fac.solve(Bt_p) + self.beta * u

    def hess_U(self, v: np.ndarray) -> np.ndarray:
        # H v = MU^{-1} B^T A^{-1} M A^{-1} B v + beta v
        w = self.A_fac.solve(self.B @ v)
        q = self.A_fac.solve(self.M @ w)
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
            x = y / (self.norm_U(y) + 1e-16)
        return float(max(lam_old, 0.0))

    def estimate_L_m(self, tol: float = 1e-8, maxiter: int = None) -> tuple[float, float]:
        """
        Compute extreme generalized eigenvalues of (Q, M_U):
        Q = B^T A^{-1} M A^{-1} B + beta M_U.
        Returns (L, m) = (lambda_max, lambda_min).
        Uses eigsh with A as a LinearOperator via Q v = M_U @ hess_U(v).
        """
        n = self.n

        def q_matvec(v: np.ndarray) -> np.ndarray:
            return self.MU @ self.hess_U(v)

        Qop = LinearOperator((n, n), matvec=q_matvec, rmatvec=q_matvec, dtype=float)

        # Largest eigenvalue (w.r.t. M_U)
        vals_max, _ = eigsh(A=Qop, M=self.MU, k=1, which='LM', tol=tol, maxiter=maxiter)
        L = float(vals_max[-1])

        # Smallest eigenvalue (w.r.t. M_U)
        # For SPD generalized problem, 'SM' targets the smallest eigenvalue.
        vals_min, _ = eigsh(A=Qop, M=self.MU, k=1, which='SM', tol=tol, maxiter=maxiter)
        m = float(vals_min[0])

        # Safety clamps
        L = max(L, 0.0)
        m = max(m, 0.0)
        return L, m
