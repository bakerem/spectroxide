#!/usr/bin/env python3
"""Isolate what moves the fitted z_y: the spectra, the x grid, or the z sampling.

Fitting our PDE spectra on our own 118-node redshift grid gives z_y ~ 6.4e4.
Fitting the same spectra, interpolated onto CosmoTherm's redshift nodes, gives
~6.06e4. The spectra are identical, so the difference is in the fit, not the
physics. This script turns the three candidate causes on and off one at a time.

Also reports two sampling-insensitive variants of the cost:

  per-z normalised : each redshift contributes its *relative* misfit, so the
                     answer does not depend on how many nodes sit where the
                     signal is large.
  dlnz-weighted    : trapezoidal in ln z, i.e. the continuum limit of the sum.
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
z_ours, x_ours, dn_raw, drho = ours["z_h"], ours["x"], ours["dn_raw"], ours["drho"]

dn_ct = np.column_stack([
    cosmotherm_gf_to_delta_n(x_ct, (g_stored * np.exp(-((z_ct / 2e6) ** 2.5))[None, :])[:, k])
    for k in range(len(z_ct))]).T
dn_ours = dn_raw / drho[:, None] - (0.25 * (1.0 - j_bb_star(z_ours)))[:, None] * g_bb(x_ours)[None, :]


def regrid(data, x_src, z_src, x_new, z_new):
    tmp = np.array([np.interp(np.log(x_new), np.log(x_src), row) for row in data])
    return np.array([np.interp(np.log(z_new), np.log(z_src), tmp[:, j])
                     for j in range(len(x_new))]).T


class Metric:
    def __init__(self, data, x, z, mode="plain", x_lo=0.5, x_hi=20.0, p=3.0):
        m = (x >= x_lo) & (x <= x_hi)
        self.x, self.z, self.mode = x[m], z, mode
        w = self.x ** p
        self.Mw, self.Yw = w * mu_shape(self.x), w * y_shape(self.x)
        self.D = w[None, :] * data[:, m]
        self.jb = j_bb_star(z)
        if mode == "per-z":                       # each z weighted by 1/|signal|^2
            self.zw = 1.0 / np.maximum(np.sum(self.D ** 2, axis=1), 1e-300)
        elif mode == "dlnz":                      # trapezoid in ln z
            lz = np.log(z)
            self.zw = np.gradient(lz)
        else:
            self.zw = np.ones(len(z))

    def _resid(self, p):
        z_y, a_y, z_mu, a_mu = p
        am = MU_TO_ENERGY * j_mu(self.z, z_mu, a_mu) * self.jb
        ay = 0.25 * j_y(self.z, z_y, a_y)
        return am[:, None] * self.Mw[None, :] + ay[:, None] * self.Yw[None, :] - self.D

    def cost(self, p):
        r = self._resid(p)
        return float(np.sum(self.zw * np.sum(r * r, axis=1)))


def fit(metric, seed=11):
    lb, ub = np.log([b[0] for b in P_BOUNDS]), np.log([b[1] for b in P_BOUNDS])
    bnd = list(zip(lb, ub))
    r = differential_evolution(lambda t: metric.cost(np.exp(t)), bounds=bnd, seed=seed,
                              maxiter=500, popsize=24, tol=1e-12, polish=False, init="sobol")
    q = minimize(lambda t: metric.cost(np.exp(t)), r.x, method="L-BFGS-B", bounds=bnd,
                 options=dict(maxiter=20000, ftol=1e-16))
    return np.exp(q.x if q.fun < r.fun else r.x)


zsel = (z_ct >= z_ours.min()) & (z_ct <= z_ours.max())
Z_CT, Z_OURS = z_ct[zsel], z_ours

print(f"our z grid  : {len(Z_OURS)} nodes, {Z_OURS.min():.4g} to {Z_OURS.max():.4g}")
print(f"CT z grid   : {len(Z_CT)} nodes over the same range")
for lo, hi in [(3e3, 3e4), (3e4, 3e5), (3e5, 3e6), (3e6, 5e6)]:
    a = ((Z_OURS >= lo) & (Z_OURS < hi)).sum()
    b = ((Z_CT >= lo) & (Z_CT < hi)).sum()
    print(f"  nodes in [{lo:7.2g}, {hi:7.2g}) :  ours {a:3d}   CT {b:3d}")
print()

CASES = [
    ("ours: native x, native z", dn_ours, x_ours, Z_OURS),
    ("ours: CT x,     native z", regrid(dn_ours, x_ours, Z_OURS, x_ct, Z_OURS), x_ct, Z_OURS),
    ("ours: native x, CT z    ", regrid(dn_ours, x_ours, Z_OURS, x_ours, Z_CT), x_ours, Z_CT),
    ("ours: CT x,     CT z    ", regrid(dn_ours, x_ours, Z_OURS, x_ct, Z_CT), x_ct, Z_CT),
    ("CosmoTherm: CT x, CT z  ", dn_ct[zsel], x_ct, Z_CT),
]

for mode in ("plain", "per-z", "dlnz"):
    print("=" * 84)
    print(f"cost mode: {mode}    (x^3 weighting, x in [0.5, 20])")
    print(f"  {'case':28s}{'z_y':>11s}{'dev vs 6e4':>12s}{'alpha_y':>10s}"
          f"{'z_mu':>10s}{'alpha_mu':>10s}")
    for label, data, xg, zg in CASES:
        p = fit(Metric(data, xg, zg, mode))
        print(f"  {label:28s}{p[0]:11.5g}{100 * (p[0] - 6e4) / 6e4:+11.2f}%"
              f"{p[1]:10.4f}{p[2]:10.5g}{p[3]:10.4f}")
    print()
