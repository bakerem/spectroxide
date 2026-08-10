#!/usr/bin/env python3
"""Fit J_mu and J_y to CosmoTherm's own Green's function database.

The question this answers. Our PDE spectra agree with the CosmoTherm Green's
function to sub-percent (paper Fig. 3), yet the visibility parameters we fit
differ from Ref. [Chluba2013] by up to 5-8% in z_y (Fig. 2, Table 1). Those two
statements are only compatible if the map from spectra to visibility parameters
is badly conditioned. The clean test is to run *CosmoTherm's own* Green's
function through the *same* fitting pipeline. Two possible outcomes:

  (a) CosmoTherm gives z_y ~ 6.0e4, ours gives ~6.3e4  -> a real spectral
      difference, amplified by the poor conditioning.
  (b) CosmoTherm also gives z_y ~ 6.3e4                -> the offset is in the
      fitting procedure (weighting / x-range / observable), not in the code.
      Chluba's published parameters then differ from a fit to his own output,
      and the Fig. 2 discrepancy is not a code discrepancy at all.

Setup. CosmoTherm stores the mu+y residual WITHOUT the G_bb temperature shift
(tracked separately via Tgin/Tglast) and applies exp(-(z/2e6)^{5/2})
analytically at convolution time. So its stored entries, times that factor, are
exactly the "spectrum minus the known temperature-shift term" that our own raw
fit constructs by subtracting 0.25 (1 - J_bb*) G_bb. The two are directly
comparable.

Everything is evaluated on one common grid: CosmoTherm's x nodes in
[x_lo, x_hi] and CosmoTherm's z_h nodes over the range both tables cover. Our
spectra are interpolated onto it, so the only difference between the two fits
is the spectra themselves.
"""
import json
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

P_NAMES = ["z_y", "alpha_y", "z_mu", "alpha_mu"]
P_LIT = np.array([6.0e4, 2.58, 5.8e4, 1.88])
P_PAPER = np.array([63128.023, 2.6533894, 58428.697, 1.9465843])
P_BOUNDS = [(2e4, 2e5), (1.0, 5.0), (2e4, 2e5), (1.0, 4.0)]


def j_bb_star(z, q=JBB_LIT):
    return np.maximum(
        q["A"] * np.exp(-((z / q["z_th"]) ** q["a_th"]))
        * (1.0 - q["B"] * (z / q["z_th"]) ** q["beta"]), 0.0)


def j_mu(z, z_mu, a_mu):
    return 1.0 - np.exp(-(((1.0 + z) / z_mu) ** a_mu))


def j_y(z, z_y, a_y):
    return 1.0 / (1.0 + ((1.0 + z) / z_y) ** a_y)


# --------------------------------------------------------------------------
# Load both datasets and put them on a common (x, z) grid
# --------------------------------------------------------------------------
z_ct, x_ct, g_stored, md = load_greens_database(include_metadata=True)
ours = np.load(DATADIR / "visibility_table.npz")
z_ours, x_ours = ours["z_h"], ours["x"]
dn_raw, drho_tab = ours["dn_raw"], ours["drho"]

# CosmoTherm mu+y response per unit Drho/rho, as occupation number.
g_muy_jy = g_stored * np.exp(-((z_ct / 2.0e6) ** 2.5))[np.newaxis, :]
dn_ct_full = np.column_stack(
    [cosmotherm_gf_to_delta_n(x_ct, g_muy_jy[:, k]) for k in range(len(z_ct))]
).T                                        # (N_z, N_x)

# Our raw spectra, temperature-shift term removed the same way the fit does.
G_ours = g_bb(x_ours)
dn_ours_muy = dn_raw - (drho_tab * 0.25 * (1.0 - j_bb_star(z_ours)))[:, None] * G_ours[None, :]
dn_ours_muy /= drho_tab[:, None]           # per unit Drho/rho, matching CT

z_lo = max(z_ct.min(), z_ours.min())
z_hi = min(z_ct.max(), z_ours.max())
zsel = (z_ct >= z_lo) & (z_ct <= z_hi)
Z = z_ct[zsel]

# Interpolate our spectra onto (Z, x_ct): log-log in x, log in z.
def regrid(x_new, z_new):
    tmp = np.empty((len(z_ours), len(x_new)))
    for k in range(len(z_ours)):
        tmp[k] = np.interp(np.log(x_new), np.log(x_ours), dn_ours_muy[k])
    out = np.empty((len(z_new), len(x_new)))
    for j in range(len(x_new)):
        out[:, j] = np.interp(np.log(z_new), np.log(z_ours), tmp[:, j])
    return out


class Metric:
    """x^p-weighted least squares over x in [x_lo, x_hi] on the common grid."""

    def __init__(self, data, x, z, x_lo=0.5, x_hi=20.0, p=3.0):
        m = (x >= x_lo) & (x <= x_hi)
        self.x, self.z = x[m], z
        w = self.x ** p
        self.Mw, self.Yw = w * mu_shape(self.x), w * y_shape(self.x)
        self.D = w[None, :] * data[:, m]
        self.jb = j_bb_star(z)
        self.label = f"x^{p:g}, x in [{x_lo:g},{x_hi:g}], {m.sum()} pts"

    def cost(self, p):
        z_y, a_y, z_mu, a_mu = p
        am = MU_TO_ENERGY * j_mu(self.z, z_mu, a_mu) * self.jb
        ay = 0.25 * j_y(self.z, z_y, a_y)
        r = am[:, None] * self.Mw[None, :] + ay[:, None] * self.Yw[None, :] - self.D
        return float(np.sum(r * r))

    def floor(self):
        A = np.column_stack([self.Mw, self.Yw])
        coef, *_ = np.linalg.lstsq(A, self.D.T, rcond=None)
        r = A @ coef - self.D.T
        return float(np.sum(r * r))


def fit(metric, seed=11, maxiter=500):
    lb, ub = np.log([b[0] for b in P_BOUNDS]), np.log([b[1] for b in P_BOUNDS])
    bnd = list(zip(lb, ub))
    r = differential_evolution(lambda t: metric.cost(np.exp(t)), bounds=bnd, seed=seed,
                              maxiter=maxiter, popsize=24, tol=1e-12, polish=False,
                              init="sobol")
    q = minimize(lambda t: metric.cost(np.exp(t)), r.x, method="L-BFGS-B",
                 bounds=bnd, options=dict(maxiter=20000, ftol=1e-16))
    t = q.x if q.fun < r.fun else r.x
    return np.exp(t), metric.cost(np.exp(t))


DATASETS = {"CosmoTherm": dn_ct_full[zsel], "spectroxide": regrid(x_ct, Z)}

print(f"common grid: {len(Z)} redshifts in [{z_lo:.4g}, {z_hi:.4g}], "
      f"CosmoTherm x nodes\n")

# Sanity: how well do the two spectra agree on this grid?
print("=" * 92)
print("Spectral agreement of the two datasets on the common grid (x^3-weighted RMS)")
print("=" * 92)
mtest = (x_ct >= 0.5) & (x_ct <= 20.0)
wt = x_ct[mtest] ** 3
a, b = DATASETS["CosmoTherm"][:, mtest], DATASETS["spectroxide"][:, mtest]
num = np.sqrt(np.sum((wt * (a - b)) ** 2, axis=1))
den = np.sqrt(np.sum((wt * a) ** 2, axis=1))
rel = num / np.maximum(den, 1e-300)
print(f"  median {100 * np.median(rel):.3f}%   90th pct {100 * np.percentile(rel, 90):.3f}%"
      f"   max {100 * rel.max():.3f}% at z_h = {Z[np.argmax(rel)]:.3g}")
for zt in (1e4, 3e4, 6e4, 1e5, 3e5, 1e6):
    k = int(np.argmin(np.abs(Z - zt)))
    print(f"    z_h = {Z[k]:9.3g}   rel diff = {100 * rel[k]:6.3f}%")
print()

out = {}
for pw, xlo, xhi in [(3.0, 0.5, 20.0), (0.0, 0.5, 20.0), (2.0, 0.5, 20.0),
                     (4.0, 0.5, 20.0), (3.0, 1.0, 10.0), (3.0, 0.1, 30.0)]:
    print("=" * 92)
    print(f"metric: x^{pw:g} weighting, x in [{xlo:g}, {xhi:g}]")
    print(f"  {'dataset':14s}{'z_y':>11s}{'dev%':>9s}{'alpha_y':>10s}{'z_mu':>11s}"
          f"{'alpha_mu':>10s}{'C_fit/floor':>13s}{'C_lit/C_fit':>13s}")
    for name, data in DATASETS.items():
        met = Metric(data, x_ct, Z, xlo, xhi, pw)
        p, c = fit(met)
        fl, clit = met.floor(), met.cost(P_LIT)
        print(f"  {name:14s}{p[0]:11.5g}{100 * (p[0] - 6e4) / 6e4:+8.2f}%"
              f"{p[1]:10.4f}{p[2]:11.5g}{p[3]:10.4f}"
              f"{c / fl:13.4f}{clit / c:13.4f}")
        out[f"{name}|x^{pw:g}|{xlo}-{xhi}"] = dict(
            zip(P_NAMES, map(float, p)), cost=c, floor=fl, cost_lit=clit)
    print()

(DATADIR / "visibility_fit_cosmotherm_gf.json").write_text(json.dumps(out, indent=1))
print(f"wrote {DATADIR / 'visibility_fit_cosmotherm_gf.json'}")
