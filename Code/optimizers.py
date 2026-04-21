#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizers for reduced OC: Barzilai–Borwein, fixed-step GD (alpha=1/L), Nesterov.
Require a model exposing: cost(u), grad_U(u), norm_U(x), dot_U(a,b), estimate_L(), estimate_m().
"""
import numpy as np


def _stop_threshold(model, tol: float | None, tol_abs: float | None, tol_rel: float | None) -> float:
    """Return a stopping threshold for ||grad_U(u)||_U.

    If tol_abs or tol_rel is provided, uses
        tol_abs + tol_rel * ||grad_U(0)||_U.
    Otherwise falls back to the legacy absolute tolerance `tol`.
    """
    if tol_abs is None and tol_rel is None:
        if tol is None:
            return 0.0
        return float(tol)

    tol_abs_val = 0.0 if tol_abs is None else float(tol_abs)
    tol_rel_val = 0.0 if tol_rel is None else float(tol_rel)
    g0 = model.grad_U(np.zeros(model.n))
    g0n = float(model.norm_U(g0))
    return tol_abs_val + tol_rel_val * g0n


def bb(model, u0=None, tol=1e-8, tol_abs=None, tol_rel=None, max_iter=1000):
    """Barzilai–Borwein in U with inner product from model."""
    n = model.n
    if u0 is None:
        u_m1 = np.zeros(n)
        g_m1 = model.grad_U(u_m1)
        u = g_m1.copy()
    else:
        u_m1 = np.zeros(n)
        g_m1 = model.grad_U(u_m1)
        u = u0.copy()
    g = model.grad_U(u)

    hist = {
        'grad_norm': [model.norm_U(g_m1)],
        'cost': [model.cost(u_m1)],
        'u_seq': [u_m1.copy()],
        'updates': 0,
    }

    k = 0
    stop_thr = _stop_threshold(model, tol=tol, tol_abs=tol_abs, tol_rel=tol_rel)
    while model.norm_U(g) > stop_thr and k < max_iter:
        s = u - u_m1
        d = g - g_m1
        sd = model.dot_U(s, d)
        if abs(sd) < 1e-30:
            alpha = 1.0
        else:
            if k % 2 == 0:
                # Step-length form (update uses u_{k+1} = u_k - alpha_k g_k):
                # alpha = <s,d>/<d,d>
                alpha = sd / max(model.dot_U(d, d), 1e-30)
            else:
                # alpha = <s,s>/<s,d>
                alpha = model.dot_U(s, s) / sd
        u_next = u - alpha * g

        u_m1, u = u, u_next
        g_m1, g = g, model.grad_U(u)
        hist['grad_norm'].append(model.norm_U(g))
        hist['cost'].append(model.cost(u))
        hist['u_seq'].append(u.copy())
        k += 1
        hist['updates'] = k

    return u, hist


def gd_fixed(model, u0=None, tol=1e-8, tol_abs=None, tol_rel=None, max_iter=1000, L=None):
    """Gradient descent with fixed step alpha=1/L (L estimated if None)."""
    n = model.n
    if u0 is None:
        u = model.grad_U(np.zeros(n))
    else:
        u = u0.copy()
    if L is None or L <= 0:
        L = model.estimate_L(iters=20, tol=1e-6)
    alpha = 1.0 / L

    hist = {
        'grad_norm': [],
        'cost': [],
        'L': L,
        'u_seq': [],
        'updates': 0,
    }

    stop_thr = _stop_threshold(model, tol=tol, tol_abs=tol_abs, tol_rel=tol_rel)

    for _ in range(max_iter):
        g = model.grad_U(u)
        gn = model.norm_U(g)
        hist['grad_norm'].append(gn)
        hist['cost'].append(model.cost(u))
        hist['u_seq'].append(u.copy())
        if gn <= stop_thr:
            break
        u = u - alpha * g
        hist['updates'] += 1

    return u, hist


def nesterov_constant_ml(model, u0=None, tol=1e-8, tol_abs=None, tol_rel=None, max_iter=1000, L=None, m=None):
    """Nesterov with constant parameters for strongly convex QP.

    Uses alpha = 1/L and beta = (sqrt(kappa)-1)/(sqrt(kappa)+1), kappa = L/m,
    with update
        y_k = u_k + beta (u_k - u_{k-1}),
        u_{k+1} = y_k - alpha * grad_U(u_k),
    where L and m are the largest/smallest generalized eigenvalues of (Q, M_U).
    If L or m is not provided, they are computed via model.estimate_L() and model.estimate_m().
    """
    n = model.n
    if u0 is None:
        u = model.grad_U(np.zeros(n))
    else:
        u = u0.copy()
    u_prev = u.copy()

    if L is None or L <= 0:
        L = model.estimate_L(iters=20, tol=1e-6)
    if m is None or m <= 0:
        m = model.estimate_m(iters=20, tol=1e-6)

    alpha = 1.0 / L
    kappa = max(L / max(m, 1e-30), 1.0)
    sqk = np.sqrt(kappa)
    beta = (sqk - 1.0) / (sqk + 1.0)

    hist = {
        'grad_norm': [],
        'cost': [],
        'L': L,
        'm': m,
        'beta': beta,
        'kappa': kappa,
        'u_seq': [],
        'updates': 0,
    }

    stop_thr = _stop_threshold(model, tol=tol, tol_abs=tol_abs, tol_rel=tol_rel)

    for _ in range(max_iter):
        g = model.grad_U(u)
        gn = model.norm_U(g)
        hist['grad_norm'].append(gn)
        hist['cost'].append(model.cost(u))
        hist['u_seq'].append(u.copy())
        if gn <= stop_thr:
            break

        y = u + beta * (u - u_prev)
        u_next = y - alpha * g
        u_prev, u = u, u_next
        hist['updates'] += 1

    return u, hist
