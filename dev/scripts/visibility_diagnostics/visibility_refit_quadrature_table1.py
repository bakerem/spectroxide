#!/usr/bin/env python3
"""Recompute the paper's Table 1 visibility fit with a grid-independent cost.

The fit behind Fig. 2 and Table 1 (dev/scripts/fit_visibility_from_table.py)
minimises an unweighted sum over frequency *nodes*,

    C = sum_z sum_i [ x_i^3 (model_i - data_i) ]^2 ,

so the effective weight is x^3 times the local density of our PDE grid. That is
not a functional of the spectra: interpolating the same spectra onto
CosmoTherm's x nodes moves the fitted z_y by 5.4%. Replacing the sum with the
quadrature Int [x^3 r]^2 dx removes the grid dependence (0.2% residual, versus
5.9% for the node sum) and is what this script fits.

Same configuration as the paper otherwise: NC-stripped spectra, seven free
parameters (z_y, alpha_y, z_mu, alpha_mu, A, B, beta) with z_th = 1.98e6 and
alpha_th = 5/2 held at their analytic values.
"""
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from spectroxide.greens import mu_shape, y_shape, g_bb, KAPPA_C

MU_TO_ENERGY = 3.0 / KAPPA_C
DATADIR = ROOT / "dev" / "data"

NAMES = ["z_y", "alpha_y", "z_mu", "alpha_mu", "A", "B", "beta"]
LIT = np.array([6.0e4, 2.58, 5.8e4, 1.88, 0.983, 0.0381, 2.29])
PAPER = np.array([63128.023, 2.6533894, 58428.697, 1.9465843, 0.99191, 0.048283, 2.0724])
BOUNDS = [(2e4, 2e5), (1.0, 5.0), (2e4, 2e5), (1.0, 4.0), (0.5, 1.0), (0.0, 0.5), (0.5, 5.0)]
Z_TH, A_TH = 1.98e6, 2.5

d = np.load(DATADIR / "visibility_table.npz")
z_h, x, dn_nc, drho = d["z_h"], d["x"], d["dn_nc"], d["drho"]

M_x, Y_x, G_x = mu_shape(x), y_shape(x), g_bb(x)
G_int = np.trapz(x**2 * G_x, x)
M_nc = M_x - np.trapz(x**2 * M_x, x) / G_int * G_x
Y_nc = Y_x - np.trapz(x**2 * Y_x, x) / G_int * G_x   # G_nc == 0 identically

mask = (x >= 0.5) & (x <= 20.0)
xs = x[mask]
W2 = xs**6                                   # (x^3)^2


def measure(kind):
    if kind == "node":
        return np.ones_like(xs)
    if kind == "dx":
        return np.gradient(xs)
    raise ValueError(kind)


def model_amps(p):
    z_y, a_y, z_mu, a_mu, A, B, beta = p
    jb = np.maximum(A * np.exp(-((z_h / Z_TH) ** A_TH))
                    * (1.0 - B * (z_h / Z_TH) ** beta), 0.0)
    am = MU_TO_ENERGY * (1.0 - np.exp(-(((1.0 + z_h) / z_mu) ** a_mu))) * jb
    ay = 0.25 / (1.0 + ((1.0 + z_h) / z_y) ** a_y)
    return am, ay


class Cost:
    def __init__(self, kind):
        self.q = W2 * measure(kind)
        self.M, self.Y = M_nc[mask], Y_nc[mask]
        self.D = dn_nc[:, mask]
        self.kind = kind

    def __call__(self, p):
        am, ay = model_amps(p)
        r = (drho * am)[:, None] * self.M[None, :] \
            + (drho * ay)[:, None] * self.Y[None, :] - self.D
        return float(np.sum(self.q[None, :] * r * r))

    def floor(self):
        s = np.sqrt(self.q)
        A = np.column_stack([s * self.M, s * self.Y])
        coef, *_ = np.linalg.lstsq(A, (s[None, :] * self.D).T, rcond=None)
        r = A @ coef - (s[None, :] * self.D).T
        return float(np.sum(r * r))


def fit(cost, seed=7):
    lb = np.array([np.log(b[0]) if b[0] > 0 else -30.0 for b in BOUNDS])
    ub = np.array([np.log(b[1]) for b in BOUNDS])
    bnd = list(zip(lb, ub))
    r = differential_evolution(lambda t: cost(np.exp(t)), bounds=bnd, seed=seed,
                              maxiter=900, popsize=32, tol=1e-14, polish=False, init="sobol")
    q = minimize(lambda t: cost(np.exp(t)), r.x, method="L-BFGS-B", bounds=bnd,
                 options=dict(maxiter=40000, ftol=1e-16))
    t = q.x if q.fun < r.fun else r.x
    return np.exp(t), cost(np.exp(t))


out = {}
for kind in ("node", "dx"):
    cost = Cost(kind)
    p, c = fit(cost)
    fl, clit, cpap = cost.floor(), cost(LIT), cost(PAPER)
    print("=" * 96)
    print(f"measure = {kind}    (NC-stripped, 7 free params, z_th and alpha_th fixed analytic)")
    print(f"  floor (per-z free M,Y)            = {fl:.6g}")
    print(f"  C(literature)                     = {clit:.6g}   ({100 * (clit / fl - 1):+.2f}% above floor)")
    print(f"  C(paper Table 1)                  = {cpap:.6g}   ({100 * (cpap / fl - 1):+.2f}% above floor)")
    print(f"  C(best fit, this measure)         = {c:.6g}   ({100 * (c / fl - 1):+.2f}% above floor)")
    print(f"  C_lit / C_fit                     = {clit / c:.4f}")
    print(f"\n  {'param':10s}{'this fit':>13s}{'literature':>13s}{'dev %':>10s}"
          f"{'paper Tab.1':>13s}{'paper dev %':>13s}")
    for i, nm in enumerate(NAMES):
        lo, hi = BOUNDS[i]
        at = "  AT BOUND" if (p[i] <= lo * (1 + 1e-3) or p[i] >= hi * (1 - 1e-3)) else ""
        print(f"  {nm:10s}{p[i]:13.5g}{LIT[i]:13.5g}{100 * (p[i] - LIT[i]) / LIT[i]:+9.2f}%"
              f"{PAPER[i]:13.5g}{100 * (PAPER[i] - LIT[i]) / LIT[i]:+12.2f}%{at}")
    print()
    out[kind] = dict(params={nm: float(p[i]) for i, nm in enumerate(NAMES)},
                     cost=c, floor=fl, cost_lit=clit, cost_paper=cpap)

(DATADIR / "visibility_refit_quadrature_table1.json").write_text(json.dumps(out, indent=1))
print(f"wrote {DATADIR / 'visibility_refit_quadrature_table1.json'}")
