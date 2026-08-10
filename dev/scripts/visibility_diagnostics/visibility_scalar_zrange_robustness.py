#!/usr/bin/env python3
"""Is the reconstructed Chluba procedure robust to the redshift range fitted?

Result 7 recovers Chluba's four published parameters to <2% by fitting J_y and
J_mu separately, by ordinary least squares, to the per-redshift visibility
sequences extracted with x^3 weighting and J_bb* held at its analytic form. That
scan varied the dataset, the J_bb* form, the temperature-shift treatment and the
weighting exponent, but always fitted the full redshift range each table covers.

The redshift range is the remaining free choice, and it is not innocuous: J_y
and J_mu are sigmoids, so including or excluding the flat tails changes how much
leverage the transition region carries. If the recovered parameters move by more
than the ~1% agreement they currently show, the reconstruction is tuned rather
than robust.
"""
import itertools
import pathlib
import sys

import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from spectroxide.greens import mu_shape, y_shape, g_bb, KAPPA_C
from spectroxide.cosmotherm import (load_greens_database, reconstruct_full_gf,
                                    cosmotherm_gf_to_delta_n)

MU_TO_ENERGY = 3.0 / KAPPA_C
DATADIR = ROOT / "dev" / "data"
Z_TH, A_TH = 1.98e6, 2.5
TARGET = np.array([6.0e4, 2.58, 5.8e4, 1.88])
B_Y = [(2e4, 2e5), (1.0, 5.0)]
B_MU = [(2e4, 2e5), (1.0, 4.0)]
JBB_AN = dict(A=1.0, B=0.0, beta=2.29)

tab = np.load(DATADIR / "visibility_table.npz")
DATA = {"spectroxide": (tab["z_h"], tab["x"], tab["dn_raw"] / tab["drho"][:, None])}
z_ct, x_ct, g_stored, md = load_greens_database(include_metadata=True)
g_full = reconstruct_full_gf(x_ct, g_stored, z_ct, md, apply_exp=True)
dn_ct = np.column_stack([cosmotherm_gf_to_delta_n(x_ct, g_full[:, k])
                         for k in range(len(z_ct))]).T
sel = (z_ct >= tab["z_h"].min()) & (z_ct <= tab["z_h"].max())
DATA["CosmoTherm"] = (z_ct[sel], x_ct, dn_ct[sel])


def j_bb_star(z):
    return np.maximum(np.exp(-((z / Z_TH) ** A_TH)), 0.0)


def j_mu(z, p):
    return 1.0 - np.exp(-(((1.0 + z) / p[0]) ** p[1]))


def j_y(z, p):
    return 1.0 / (1.0 + ((1.0 + z) / p[0]) ** p[1])


def amplitudes(dataset, power=3.0, x_lo=0.5, x_hi=20.0):
    z, x, dn = DATA[dataset]
    jb = j_bb_star(z)
    G = g_bb(x)
    M, Y = mu_shape(x), y_shape(x)
    D = dn - (0.25 * (1.0 - jb))[:, None] * G[None, :]
    m = (x >= x_lo) & (x <= x_hi)
    xs = x[m]
    qw = xs ** (2 * power) * np.gradient(xs)
    Ms, Ys, Ds = M[m], Y[m], D[:, m]
    Gmm, Gmy, Gyy = (np.sum(qw * Ms * Ms), np.sum(qw * Ms * Ys), np.sum(qw * Ys * Ys))
    P, Q = Ds @ (qw * Ms), Ds @ (qw * Ys)
    det = Gmm * Gyy - Gmy**2
    return z, (Gyy * P - Gmy * Q) / det, (Gmm * Q - Gmy * P) / det, jb


def lsq(fn, seq, z, bounds, seed=5):
    lb, ub = np.log([c[0] for c in bounds]), np.log([c[1] for c in bounds])
    bnd = list(zip(lb, ub))

    def cost(t):
        r = fn(z, np.exp(t)) - seq
        return float(np.sum(r * r))

    r = differential_evolution(cost, bounds=bnd, seed=seed, maxiter=600, popsize=24,
                              tol=1e-14, polish=False, init="sobol")
    q = minimize(cost, r.x, method="L-BFGS-B", bounds=bnd,
                 options=dict(maxiter=20000, ftol=1e-18))
    return np.exp(q.x if q.fun < r.fun else r.x)


RANGES = [("full table", 0.0, np.inf),
          ("[1e4, 5e6]", 1e4, 5e6),
          ("[1e4, 1e6]", 1e4, 1e6),
          ("[3e4, 3e5]", 3e4, 3e5),
          ("[1e4, 3e6]", 1e4, 3e6),
          ("[3e3, 1e6]", 3e3, 1e6)]

hdr = (f"{'dataset':12s}{'z range':13s}{'N_y':>5s}{'N_mu':>6s}"
       f"{'z_y':>9s}{'a_y':>8s}{'z_mu':>9s}{'a_mu':>8s}{'RMS dev':>9s}")
print(hdr)
print("-" * len(hdr))
rows = []
for dset in DATA:
    z, a, b, jb = amplitudes(dset)
    for lab, lo, hi in RANGES:
        my = (z >= lo) & (z <= hi)
        mm = my & (jb > 1e-3)
        if my.sum() < 8 or mm.sum() < 8:
            continue
        py = lsq(j_y, 4.0 * b[my], z[my], B_Y)
        pm = lsq(j_mu, a[mm] / (MU_TO_ENERGY * jb[mm]), z[mm], B_MU)
        p = np.concatenate([py, pm])
        dev = 100 * (p - TARGET) / TARGET
        rms = float(np.sqrt(np.mean(dev**2)))
        rows.append((dset, lab, p, dev, rms))
        print(f"{dset:12s}{lab:13s}{my.sum():5d}{mm.sum():6d}"
              f"{p[0]:9.0f}{p[1]:8.3f}{p[2]:9.0f}{p[3]:8.3f}{rms:8.2f}%")

print(f"\nChluba (2013) published:{'':13s}"
      f"{TARGET[0]:9.0f}{TARGET[1]:8.2f}{TARGET[2]:9.0f}{TARGET[3]:8.2f}")

print("\nspread across redshift ranges (excluding the narrow [3e4, 3e5] window):")
for dset in DATA:
    sub = [r for r in rows if r[0] == dset and r[1] != "[3e4, 3e5]"]
    arr = np.array([r[2] for r in sub])
    for i, nm in enumerate(["z_y", "alpha_y", "z_mu", "alpha_mu"]):
        lo, hi = arr[:, i].min(), arr[:, i].max()
        print(f"  {dset:12s} {nm:9s} {lo:9.4g} to {hi:9.4g}   "
              f"({100 * (hi - lo) / np.mean(arr[:, i]):.2f}% spread)")
