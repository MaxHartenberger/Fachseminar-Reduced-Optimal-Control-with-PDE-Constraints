#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduced optimal control model using an external XDMF mesh with a circular
subdomain (omega) tagged in the cell data.
Compatible with Code_v2/optimizers.py API.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

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
    def __init__(self, mesh_cells_xdmf: str, omega_id: int = 2, beta: float = 1e-3, degree: int = 1):
        self.beta = float(beta)
        self.degree = int(degree)
        self.mesh, self.subdomains = load_external_mesh(mesh_cells_xdmf)
        self.omega_id = int(omega_id)

        self.V = fe.FunctionSpace(self.mesh, 'P', 1)
        self.n = self.V.dim()
        self.bc = fe.DirichletBC(self.V, fe.Constant(0.0), 'on_boundary')

        y = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)
        uU = fe.TrialFunction(self.V)
        wU = fe.TestFunction(self.V)

        a = fe.dot(fe.grad(y), fe.grad(v)) * fe.dx
        M = y * v * fe.dx
        B_form = uU * v * fe.dx(self.omega_id, subdomain_data=self.subdomains)
        MU_form = uU * wU * fe.dx

        A_d = fe.assemble(a)
        self.bc.apply(A_d)
        M_d = fe.assemble(M)
        B_d = fe.assemble(B_form)
        MU_d = fe.assemble(MU_form)

        self.A = _to_csr(A_d)
        self.M = _to_csr(M_d)
        self.B = _to_csr(B_d)
        self.MU = _to_csr(MU_d)

        self.dir_dofs = np.array(list(self.bc.get_boundary_values().keys()), dtype=int)
        self.A_fac = splu(self.A.tocsc())
        self.MU_fac = splu(self.MU.tocsc())

        yd_expr = fe.Expression("0.5 - fmax(fabs(x[0]-0.5), fabs(x[1]-0.5))", degree=max(1, self.degree))
        yd_fun = fe.interpolate(yd_expr, self.V)
        self.y_d = yd_fun.vector().get_local()

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

    def cost(self, u: np.ndarray) -> float:
        y = self.state(u)
        d = y - self.y_d
        term_state = 0.5 * float(d.T @ (self.M @ d))
        term_ctrl = 0.5 * self.beta * float(u.T @ (self.MU @ u))
        return term_state + term_ctrl

    def grad_U(self, u: np.ndarray) -> np.ndarray:
        y = self.state(u)
        z = self.M @ (y - self.y_d)
        p = self.adjoint(z)
        Bt_p = self.B.transpose() @ p
        return self.MU_fac.solve(Bt_p) + self.beta * u

    def hess_U(self, v: np.ndarray) -> np.ndarray:
        w = self.A_fac.solve(self.B @ v)
        q = self.A_fac.solve(self.M @ w)
        Bt_q = self.B.transpose() @ q
        return self.MU_fac.solve(Bt_q) + self.beta * v

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
