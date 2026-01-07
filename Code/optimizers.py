#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizers for reduced OC: Barzilai–Borwein, fixed-step GD (alpha=1/L), Nesterov.
Require a model exposing: cost(u), grad_U(u), norm_U(x), dot_U(a,b), estimate_L().
"""

import numpy as np


def bb(model, u0=None, tol=1e-8, max_iter=1000):
    """Barzilai–Borwein in U with inner product from model."""
    n = model.n
    if u0 is None:
        u_m1 = np.zeros(n)
        g_m1 = model.grad_U(u_m1)
        u = g_m1.copy()  # u0 = ∇F(0)
    else:
        u_m1 = np.zeros(n)
        g_m1 = model.grad_U(u_m1)
        u = u0.copy()
    g = model.grad_U(u)

    hist = {
        'grad_norm': [model.norm_U(g_m1)],
        'cost': [model.cost(u_m1)],
    }

    k = 0
    while model.norm_U(g) > tol and k < max_iter:
        s = u - u_m1
        d = g - g_m1
        sd = model.dot_U(s, d)
        if abs(sd) < 1e-30:
            alpha = 1.0
        else:
            if k % 2 == 0:
                alpha = model.dot_U(d, d) / sd
            else:
                alpha = sd / max(model.dot_U(s, s), 1e-30)
        u_next = u - (1.0 / alpha) * g

        u_m1, u = u, u_next
        g_m1, g = g, model.grad_U(u)
        hist['grad_norm'].append(model.norm_U(g))
        hist['cost'].append(model.cost(u))
        k += 1

    return u, hist


def gd_fixed(model, u0=None, tol=1e-8, max_iter=1000, L=None):
    """Gradient descent with fixed step alpha=1/L (L estimated if None)."""
    n = model.n
    if u0 is None:
        u = model.grad_U(np.zeros(n))  # u0 = ∇F(0)
    else:
        u = u0.copy()
    if L is None or L <= 0:
        L = model.estimate_L(iters=20, tol=1e-6)
    alpha = 1.0 / L

    hist = {
        'grad_norm': [],
        'cost': [],
        'L': L,
    }

    for _ in range(max_iter):
        g = model.grad_U(u)
        gn = model.norm_U(g)
        hist['grad_norm'].append(gn)
        hist['cost'].append(model.cost(u))
        if gn <= tol:
            break
        u = u - alpha * g

    return u, hist


def nesterov(model, u0=None, tol=1e-8, max_iter=1000, L=None, restart=False):
    """Nesterov acceleration (FISTA-style) with step 1/L. Optional restart if cost increases."""
    n = model.n
    if u0 is None:
        u = model.grad_U(np.zeros(n))
    else:
        u = u0.copy()
    y = u.copy()
    if L is None or L <= 0:
        L = model.estimate_L(iters=20, tol=1e-6)
    alpha = 1.0 / L
    t = 1.0

    hist = {
        'grad_norm': [],
        'cost': [],
        'L': L,
    }

    f_prev = model.cost(u)
    for _ in range(max_iter):
        g = model.grad_U(y)
        gn = model.norm_U(g)
        hist['grad_norm'].append(gn)
        hist['cost'].append(f_prev)
        if gn <= tol:
            break

        u_next = y - alpha * g
        f_next = model.cost(u_next)

        if restart and f_next > f_prev:
            # Restart: reset momentum
            y = u.copy()
            t = 1.0
            continue

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        beta = (t - 1.0) / t_next
        y = u_next + beta * (u_next - u)
        u = u_next
        t = t_next
        f_prev = f_next

    return u, hist
