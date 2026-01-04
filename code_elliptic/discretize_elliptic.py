#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elliptic discretization helper (analog zu github_Kartmann/code/discretize.py)
Erstellt ein FEM Full Order Model (FOM) für das stationäre elliptische Problem
-Delta y = B u  in Omega, y = 0 auf dOmega

Die Funktion `discretize` liefert ein `model`-Objekt wie im Referenzprojekt.
"""

import fenics as fenics
import numpy as np
from scipy.sparse import csr_matrix, diags, identity


# Lightweight replacements for `methods.collection` and `model.model`
class Collection:
    """Minimal container (attribute access) used in place of the original `collection`.
    Behaves like an empty object where attributes can be set dynamically.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def as_dict(self):
        return self.__dict__.copy()

    def __repr__(self):
        return f"Collection({self.__dict__})"


class Model:
    """Small wrapper that holds pde/cost/time/space/options.
    The original `model` class in the reference repo offers many methods; here we
    only provide a lightweight container so the discretizer can return a single
    object. If you later need solver methods, add them to this class or replace
    the class with the project's real `model` implementation.
    """

    def __init__(self, pde, cost, time, space, options=None):
        self.pde = pde
        self.cost = cost
        self.time = time
        self.space = space
        self.options = options

    def __repr__(self):
        return f"Model(pde={len(self.pde.__dict__)} attrs, cost={len(self.cost.__dict__)}, time={self.time}, space={self.space})"

    # minimal convenience accessors
    @property
    def pde_obj(self):
        return self.pde

    @property
    def cost_obj(self):
        return self.cost

    @property
    def time_obj(self):
        return self.time

    @property
    def space_obj(self):
        return self.space


def discretize(dx=50):
    # options
    options = Collection()
    options.factorize = True

    # time placeholder for stationary problem: single time slice K=1
    time = Collection()
    time.t0 = 0.0
    time.T = 0.0
    time.K = 1
    time.dt = 1.0
    time.t_v = np.array([0.0])
    time.D = np.ones(time.K)
    time.D_diag = diags(time.D)
    time.cost_zero = 1.0

    # space discretization (unit square)
    space = Collection()
    L = 1.0
    lower_left = fenics.Point(0.0, 0.0)
    upper_right = fenics.Point(L, L)
    mesh = fenics.RectangleMesh(lower_left, upper_right, dx, dx)
    V = fenics.FunctionSpace(mesh, 'P', 1)
    space.V = V
    space.mesh = mesh
    space.dx = dx
    space.DirichletBC = None

    # pde assembly (elliptic: diffusion operator A = -Delta)
    pde = Collection()
    y = fenics.TrialFunction(V)
    v = fenics.TestFunction(V)

    # diffusion term (stiffness)
    A_diff_form = fenics.assemble(fenics.dot(fenics.nabla_grad(y), fenics.nabla_grad(v)) * fenics.dx)
    A_diff = csr_matrix(fenics.as_backend_type(A_diff_form).mat().getValuesCSR()[::-1])
    # optional reaction term 0 for pure Laplace
    pde.A = A_diff

    # control operator B: piecewise indicator functions on an n_rhs grid (like reference)
    n_rhs = np.array([4, 4])
    n_u = int(np.prod(n_rhs))
    x_grid = np.linspace(0, 1, n_rhs[0] + 1)
    y_grid = np.linspace(0, 1, n_rhs[1] + 1)
    tol = 1e-14
    B_list = []
    for i in range(n_rhs[1]):  # y direction
        for j in range(n_rhs[0]):  # x direction
            expr = fenics.Expression(
                "x1 <= x[0] && x[0] <= x2 + tol && y1 <= x[1] && x[1] <= y2 + tol ? k1 : k2",
                x1=float(x_grid[j]), x2=float(x_grid[j + 1]), y1=float(y_grid[i]), y2=float(y_grid[i + 1]),
                degree=0, tol=tol, k1=1.0, k2=0.0)
            vec = fenics.assemble(expr * v * fenics.dx).get_local()
            B_list.append(vec)
    B = np.array(B_list).T  # shape (state_dofs, n_u)
    pde.B = B

    # mass matrix M
    M_form = fenics.assemble(y * v * fenics.dx)
    M = csr_matrix(fenics.as_backend_type(M_form).mat().getValuesCSR()[::-1])
    pde.M = M

    # products: L2 and H10 (energy) and H1
    L2_form = fenics.assemble(y * v * fenics.dx)
    L2 = csr_matrix(fenics.as_backend_type(L2_form).mat().getValuesCSR()[::-1])
    H10_form = fenics.assemble(fenics.dot(fenics.nabla_grad(y), fenics.nabla_grad(v)) * fenics.dx)
    H10 = csr_matrix(fenics.as_backend_type(H10_form).mat().getValuesCSR()[::-1])
    H1 = H10 + L2
    pde.state_products = {'H1': H1, 'L2': L2, 'H10': H10}

    # input product (identity for control coefficients)
    pde.input_product = identity(n_u)

    # y0 (unused for stationary, keep zero vector)
    y0_fun = fenics.Constant(0.0)
    pde.y0 = fenics.interpolate(y0_fun, V).vector().get_local()
    space.dofs = len(pde.y0)

    # F (right-hand side): stationary problem -> single column zeros
    pde.F = np.zeros((space.dofs, time.K))

    # dims/meta
    pde.input_dim = n_u
    pde.state_dim = space.dofs
    pde.type = 'FOM_elliptic'

    # cost (stationary: bounds per control dof, no time dimension)
    cost = Collection()
    cost.ua = -2.0 * np.ones(pde.input_dim * time.K)
    cost.ub = 5.0 * np.ones(pde.input_dim * time.K)
    cost.regularization_parameter = 1e-3
    cost.terminal_parameter = 0
    cost.YT = 0
    cost.Ud = None
    cost.Yd = None

    # create model
    fom = Model(pde, cost, time, space, options)
    return fom


def expression_to_vector(V, fenics_expression):
    """Interpolate a FEniCS expression into the FE basis and return the nodal vector."""
    return fenics.interpolate(fenics_expression, V).vector().get_local()


def time_expression_to_vector(V, fenics_expression, time=None):
    """If fenics_expression depends on time via attribute .t, sample it at time.t_v (if provided) and
    return an array shape (dofs, K). For stationary problems K=1.
    """
    if time is None:
        K = 1
        t_v = [0.0]
    else:
        K = time.K
        t_v = time.t_v

    vecs = []
    for t in t_v:
        # set attribute if exists
        if hasattr(fenics_expression, 't'):
            try:
                fenics_expression.t = float(t)
            except Exception:
                # ignore if cannot set
                pass
        vecs.append(fenics.interpolate(fenics_expression, V).vector().get_local())
    return np.array(vecs).T  # shape (dofs, K)

