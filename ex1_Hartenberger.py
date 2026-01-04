#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:06:04 2025

@author: max
"""

import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

class EllipticModel:
    def __init__(self, a1, a2, b1, b2, f_expr, u_expr, h=0.005, 
                 kappa=1.0, c=0.0, gD_expr="0.0", gN_expr="0.0",
                 gD_ids=[1,2,3,4], gN_id=None, degree=2):
        
        self.a1, self.a2, self.b1, self.b2 = a1, a2, b1, b2
        self.h = h
        self.gD_ids = gD_ids
        self.gN_id = gN_id

        self.f = fe.Expression(f_expr, degree=degree)
        self.u = fe.Expression(u_expr, degree=degree)
        self.kappa = fe.Constant(kappa)
        self.c = fe.Constant(c)
        self.gD = fe.Expression(gD_expr, degree=1)
        self.gN = fe.Expression(gN_expr, degree=degree)

        self.mesh = fe.RectangleMesh(fe.Point(a1, b1), fe.Point(a2, b2),
                                     int((a2 - a1)/h), int((b2 - b1)/h))
        self.V = fe.FunctionSpace(self.mesh, "P", 1)
        self._mark_boundaries()
        self.ds = fe.Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)

        self._define_variational_problem()

    def _mark_boundaries(self):
        self.boundaries = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(0)
        TOL = 1e-14
        class Left(fe.SubDomain):  
            def inside(_, x, on_boundary): return on_boundary and fe.near(x[0], self.a1, TOL)
        class Right(fe.SubDomain): 
            def inside(_, x, on_boundary): return on_boundary and fe.near(x[0], self.a2, TOL)
        class Bottom(fe.SubDomain):
            def inside(_, x, on_boundary): return on_boundary and fe.near(x[1], self.b1, TOL)
        class Top(fe.SubDomain):   
            def inside(_, x, on_boundary): return on_boundary and fe.near(x[1], self.b2, TOL)
        Left().mark(self.boundaries, 1)
        Right().mark(self.boundaries, 2)
        Bottom().mark(self.boundaries, 3)
        Top().mark(self.boundaries, 4)

    def _define_variational_problem(self):
        y = fe.TrialFunction(self.V)
        v = fe.TestFunction(self.V)

        self.a = self.kappa * fe.dot(fe.grad(y), fe.grad(v)) * fe.dx + self.c * y * v * fe.dx
        self.l = self.f * v * fe.dx + self.u * v * fe.dx
        if self.gN_id is not None:
            if isinstance(self.gN_id, list):
                for gid in self.gN_id:
                    self.l += self.gN * v * self.ds(gid)
            else:
                self.l += self.gN * v * self.ds(self.gN_id)

        self.bcs = [fe.DirichletBC(self.V, self.gD, self.boundaries, i) for i in self.gD_ids]

    def solve(self):
        A = fe.assemble(self.a)
        rhs = fe.assemble(self.l)
        for bc in self.bcs:
            bc.apply(A, rhs)

        self.solution = fe.Function(self.V)
        fe.solve(A, self.solution.vector(), rhs)
        return self.solution

    def plot_solution(self, resolution=100, title="Lösung y(x, y)"):
        x = np.linspace(self.a1, self.a2, resolution)
        y = np.linspace(self.b1, self.b2, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(lambda x, y: self.solution(x, y))(X, Y)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm')
        plt.title(title)
        fig.colorbar(surf, shrink=0.5, aspect=10)
        fig.savefig(f"{title}.pdf", bbox_inches='tight')
        plt.show()
        


#(iii)
#1.1 
a1=0.0
a2=0.5
b1=0.0
b2=0.8
f_expr="1 - (x[0] - 0.5)*(x[1] - 0.5)"
u_expr="0.0"
h = 0.005
kappa = 1.0
c = 0.0
gD_expr="0.0"
gN_expr="0.0"
gD_ids=[1,2,3,4]
gN_id=None
model = EllipticModel(a1, a2, b1, b2, f_expr, u_expr, h, kappa, c, gD_expr, gN_expr, gD_ids, gN_id)
sol = model.solve()
model.plot_solution(title="Problem 1")

#1.2
u_expr='(x[0] <= 0.25 && x[1] >= 0.0 && x[1] <= 0.8) ? -10.0 : (x[0] > 0.25 && x[0] <= 0.5 && x[1] >= 0.0 && x[1] <= 0.8) ? 10.0 : 0.0'
model = EllipticModel(a1, a2, b1, b2, f_expr, u_expr, h, kappa, c, gD_expr, gN_expr, gD_ids, gN_id)
sol = model.solve()
model.plot_solution(title="Problem 2")

#2.1
a1 = 0.0
a2 = 1.0
b1 = 0.0
b2 = 1.0
f_expr="0.0"
u_expr="0.0"
h = 0.005
kappa = 1.0
c = 1.0
gD_expr="0.0"
gN_expr="x[0]*x[0]"
gD_ids=[1, 3, 4]
gN_id=[2]
model = EllipticModel(a1, a2, b1, b2, f_expr, u_expr, h, kappa, c, gD_expr, gN_expr, gD_ids, gN_id)
sol = model.solve()
model.plot_solution(title="Problem 3")

#2.2
u_expr="50*sin(4*pi*x[0])*cos(4*pi*x[1])"
model = EllipticModel(a1, a2, b1, b2, f_expr, u_expr, h, kappa, c, gD_expr, gN_expr, gD_ids, gN_id)
sol = model.solve()
model.plot_solution(title="Problem 4")