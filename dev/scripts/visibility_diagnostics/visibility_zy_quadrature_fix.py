#!/usr/bin/env python3
"""Is the Fig. 2 z_y offset an artefact of summing the cost over grid nodes?

The fit in dev/scripts/fit_visibility_from_table.py minimises

    C = sum_z sum_i [ w_i (model_i - data_i) ]^2 ,      w_i = x_i^3

an unweighted sum over frequency *nodes*. That is not a functional of the
spectra alone: the effective weight is x^3 times the local node density, so two
codes sampled on different x grids are being fitted with different metrics even
when the cost formula is written identically. Our PDE grid is log-spaced at low
x and linear at high x; CosmoTherm's database grid is not, and interpolating our
own spectra onto CosmoTherm's nodes moves z_y from 6.42e4 to 6.08e4 with the
spectra held fixed.

The fix is to make the cost a quadrature,

    C = sum_z Int w(x)^2 (model - data)^2 dx        ('dx'  : uniform measure)
    C = sum_z Int w(x)^2 (model - data)^2 dlnx      ('dlnx': log measure)

which is grid-independent up to discretisation error. If the two grids then
agree on z_y, the node sum was the whole story.
"""
import pathlib
import sys

import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from spectroxide.greens import mu_shape, y_shape, g_bb, KAPPA_C
from spectroxide.cosmotherm import load_greens_database, cosmotherm_gf_to_delta_n

MU_TO_ENERGY = 3.0 / KAPPA_C
DATADIR = ROOT / "dev" / "data"
JBB_LIT = dict(A=0.983, z_th=1.98e6, a_th=2.5, B=0.0381, beta=2.29)
P_LIT = np.array([6.0e4, 2.58, 5.8e4, 1.88])
P_BOUNDS = [(2e4, 2e5), (1.0, 5.0), (2e4, 2e5), (1.0, 4.0)]


def j_bb_star(z, q=JBB_LIT):
    return np.maximum(q["A"] * np.exp(-((z / q["z_th"]) ** q["a_th"]))
                      * (1.0 - q["B"] * (z / q["z_th"]) ** q["beta"]), 0.0)


def j_mu(z, z_mu, a_mu):
    return 1.0 - np.exp(-(((1.0 + z) / z_mu) ** a_mu))


def j_y(z, z_y, a_y):
    return 1.0 / (1.0 + ((1.0 + z) / z_y) ** a_y)


z_ct, x_ct, g_stored, _ = load_greens_database(include_metadata=True)
ours = np.load(DATADIR / "visibility_table.npz")
z_o, x_o, dn_raw, drho = ours["z_h"], ours["x"], ours["dn_raw"], ours["drho"]

dn_ct = np.column_stack([
    cosmotherm_gf_to_delta_n(x_ct, (g_stored * np.exp(-((z_ct / 2e6) ** 2.5))[None, :])[:, k])
    for k in range(len(z_ct))]).T
dn_o = dn_raw / drho[:, None] - (0.25 * (1 - j_bb_star(z_o)))[:, None] * g_bb(x_o)[None, :]

zsel = (z_ct >= z_o.min()) & (z_ct <= z_o.max())
Z_CT = z_ct[zsel]


def regrid(data, x_src, z_src, x_new, z_new):
    tmp = np.array([np.interp(np.log(x_new), np.log(x_src), r) for r in data])
    return np.array([np.interp(np.log(z_new), np.log(z_src), tmp[:, j])
                     for j in range(len(x_new))]).T


class Metric:
    """measure = 'node' (the paper's sum), 'dx', or 'dlnx' (both quadratures)."""

    def __init__(self, data, x, z, measure="node", x_lo=0.5, x_hi=20.0, p=3.0):
        m = (x >= x_lo) & (x <= x_hi)
        xs = x[m]
        w2 = xs ** (2 * p)
        if measure == "node":
            q = np.ones_like(xs)
        elif measure == "dx":
            q = np.gradient(xs)
        elif measure == "dlnx":
            q = np.gradient(np.log(xs))
        else:
            raise ValueError(measure)
        self.q = w2 * q                       # full per-node quadrature weight
        self.x, self.z = xs, z
        self.M, self.Y = mu_shape(xs), y_shape(xs)
        self.D = data[:, m]
        self.jb = j_bb_star(z)

    def cost(self, p):
        z_y, a_y, z_mu, a_mu = p
        am = MU_TO_ENERGY * j_mu(self.z, z_mu, a_mu) * self.jb
        ay = 0.25 * j_y(self.z, z_y, a_y)
        r = am[:, None] * self.M[None, :] + ay[:, None] * self.Y[None, :] - self.D
        return float(np.sum(self.q[None, :] * r * r))

    def floor(self):
        s = np.sqrt(self.q)
        A = np.column_stack([s * self.M, s * self.Y])
        coef, *_ = np.linalg.lstsq(A, (s[None, :] * self.D).T, rcond=None)
        r = A @ coef - (s[None, :] * self.D).T
        return float(np.sum(r * r))


def fit(metric, seed=11):
    lb, ub = np.log([b[0] for b in P_BOUNDS]), np.log([b[1] for b in P_BOUNDS])
    bnd = list(zip(lb, ub))
    r = differential_evolution(lambda t: metric.cost(np.exp(t)), bounds=bnd, seed=seed,
                              maxiter=500, popsize=24, tol=1e-12, polish=False, init="sobol")
    q = minimize(lambda t: metric.cost(np.exp(t)), r.x, method="L-BFGS-B", bounds=bnd,
                 options=dict(maxiter=20000, ftol=1e-16))
    return np.exp(q.x if q.fun < r.fun else r.x)


CASES = [
    ("spectroxide (native x grid)", dn_o, x_o, z_o),
    ("spectroxide (on CT x grid) ", regrid(dn_o, x_o, z_o, x_ct, Z_CT), x_ct, Z_CT),
    ("CosmoTherm  (native x grid)", dn_ct[zsel], x_ct, Z_CT),
]

for measure in ("node", "dx", "dlnx"):
    print("=" * 88)
    print(f"measure: {measure}   (x^3 weighting, x in [0.5, 20])")
    print(f"  {'case':30s}{'z_y':>11s}{'dev vs 6e4':>12s}{'alpha_y':>10s}"
          f"{'z_mu':>10s}{'alpha_mu':>10s}{'C_lit/C_fit':>13s}")
    for label, data, xg, zg in CASES:
        met = Metric(data, xg, zg, measure)
        p = fit(met)
        print(f"  {label:30s}{p[0]:11.5g}{100 * (p[0] - 6e4) / 6e4:+11.2f}%"
              f"{p[1]:10.4f}{p[2]:10.5g}{p[3]:10.4f}"
              f"{met.cost(P_LIT) / met.cost(p):13.4f}")
    print()

# How much does the grid-density mismatch weight the two ends differently?
print("=" * 88)
print("effective x-weight from node density alone, normalised to unit total")
for lo, hi in [(0.5, 1), (1, 2), (2, 5), (5, 10), (10, 20)]:
    a = ((x_o >= lo) & (x_o < hi)).sum() / ((x_o >= 0.5) & (x_o <= 20)).sum()
    b = ((x_ct >= lo) & (x_ct < hi)).sum() / ((x_ct >= 0.5) & (x_ct <= 20)).sum()
    print(f"  x in [{lo:4g}, {hi:4g}) : ours {100 * a:5.1f}%   CT {100 * b:5.1f}%"
          f"   ratio {a / b:5.2f}")
