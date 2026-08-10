#!/usr/bin/env python3
"""Least squares on the extracted visibility values, not on the spectra.

"A least-squares fit" of a visibility function has a second natural reading. It
need not mean minimising a spectral residual over (x, z). It can mean:

  1. decompose each Green's function into component amplitudes at its own
     injection redshift, giving sequences a_M(z_k) and a_Y(z_k);
  2. convert to visibilities via the Ansatz,
         J_y(z_k)  = 4 a_Y(z_k)
         J_mu(z_k) = a_M(z_k) kappa_c / (3 J_bb*(z_k)) ;
  3. fit (z_y, alpha_y) and (z_mu, alpha_mu) by ordinary least squares to those
     two O(1) sequences, independently of one another.

This is far less sensitive to the frequency weighting than a direct spectral
fit, because the weighting enters only through step 1. It also fits each
visibility function on its own scale, which a joint spectral cost does not:
there a_M ~ 1.4 and a_Y ~ 0.25, so the mu term carries ~5.6x the leverage.

Earlier attempts at scalar fits (dev/data/visibility_scalar_fit.json,
visibility_scalar_ec_fit.json) minimised *relative* errors and landed far from
the literature values (z_y = 1.38e5 and 6.6e4). Relative errors put enormous
weight on the tails where mu and y are ~0, so those runs do not test this
reading. Absolute least squares on the sequences is what is scanned here.
"""
import itertools
import json
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

TARGET_Y = np.array([6.0e4, 2.58])
TARGET_MU = np.array([5.8e4, 1.88])
B_Y = [(2e4, 2e5), (1.0, 5.0)]
B_MU = [(2e4, 2e5), (1.0, 4.0)]

JBB = {"analytic": dict(A=1.0, B=0.0, beta=2.29),
       "literature": dict(A=0.983, B=0.0381, beta=2.29)}


def j_bb_star(z, q):
    r = z / Z_TH
    return np.maximum(q["A"] * np.exp(-(r ** A_TH)) * (1.0 - q["B"] * r ** q["beta"]), 0.0)


def j_mu(z, p):
    return 1.0 - np.exp(-(((1.0 + z) / p[0]) ** p[1]))


def j_y(z, p):
    return 1.0 / (1.0 + ((1.0 + z) / p[0]) ** p[1])


tab = np.load(DATADIR / "visibility_table.npz")
DATA = {"spectroxide": (tab["z_h"], tab["x"], tab["dn_raw"] / tab["drho"][:, None])}

z_ct, x_ct, g_stored, md = load_greens_database(include_metadata=True)
g_full = reconstruct_full_gf(x_ct, g_stored, z_ct, md, apply_exp=True)
dn_ct = np.column_stack([cosmotherm_gf_to_delta_n(x_ct, g_full[:, k])
                         for k in range(len(z_ct))]).T
sel = (z_ct >= tab["z_h"].min()) & (z_ct <= tab["z_h"].max())
DATA["CosmoTherm"] = (z_ct[sel], x_ct, dn_ct[sel])


def _strip(x, arr, G, Gnorm):
    return arr - (np.trapz(x**2 * arr, x) / Gnorm) * G


def amplitudes(dataset, jbb_key, resid, power, x_lo=0.5, x_hi=20.0):
    """Per-redshift free (a_M, a_Y) from a two-amplitude weighted least squares."""
    z, x, dn = DATA[dataset]
    jb = j_bb_star(z, JBB[jbb_key])
    G = g_bb(x)
    Gnorm = np.trapz(x**2 * G, x)
    if resid == "NC":
        M = _strip(x, mu_shape(x), G, Gnorm)
        Y = _strip(x, y_shape(x), G, Gnorm)
        D = np.array([_strip(x, row, G, Gnorm) for row in dn])
    else:
        M, Y = mu_shape(x), y_shape(x)
        D = dn - (0.25 * (1.0 - jb))[:, None] * G[None, :]

    m = (x >= x_lo) & (x <= x_hi)
    xs = x[m]
    qw = xs ** (2 * power) * np.gradient(xs)
    Ms, Ys, Ds = M[m], Y[m], D[:, m]
    Gmm = np.sum(qw * Ms * Ms)
    Gmy = np.sum(qw * Ms * Ys)
    Gyy = np.sum(qw * Ys * Ys)
    P, Q = Ds @ (qw * Ms), Ds @ (qw * Ys)
    det = Gmm * Gyy - Gmy**2
    a = (Gyy * P - Gmy * Q) / det
    b = (Gmm * Q - Gmy * P) / det
    return z, a, b, jb


def lsq(fn, target_seq, z, bounds, seed=5):
    lb = np.log([c[0] for c in bounds])
    ub = np.log([c[1] for c in bounds])
    bnd = list(zip(lb, ub))

    def cost(t):
        r = fn(z, np.exp(t)) - target_seq
        return float(np.sum(r * r))

    r = differential_evolution(cost, bounds=bnd, seed=seed, maxiter=800, popsize=28,
                              tol=1e-14, polish=False, init="sobol")
    q = minimize(cost, r.x, method="L-BFGS-B", bounds=bnd,
                 options=dict(maxiter=40000, ftol=1e-18))
    t = q.x if q.fun < r.fun else r.x
    return np.exp(t), cost(t)


rows = []
for dset, jk, rs, pw in itertools.product(DATA, JBB, ("NC", "sub"), (0.0, 2.0, 3.0, 4.0)):
    z, a, b, jb = amplitudes(dset, jk, rs, pw)

    jy_obs = 4.0 * b
    ok_mu = jb > 1e-3                       # J_mu undefined where J_bb* -> 0
    jmu_obs = np.full_like(a, np.nan)
    jmu_obs[ok_mu] = a[ok_mu] / (MU_TO_ENERGY * jb[ok_mu])

    py, cy = lsq(j_y, jy_obs, z, B_Y)
    pm, cm = lsq(lambda zz, p: j_mu(zz, p), jmu_obs[ok_mu], z[ok_mu], B_MU)

    dev = np.array([100 * (py[0] - TARGET_Y[0]) / TARGET_Y[0],
                    100 * (py[1] - TARGET_Y[1]) / TARGET_Y[1],
                    100 * (pm[0] - TARGET_MU[0]) / TARGET_MU[0],
                    100 * (pm[1] - TARGET_MU[1]) / TARGET_MU[1]])
    rows.append(dict(dataset=dset, jbb=jk, resid=rs, power=pw,
                     params=np.concatenate([py, pm]), dev=dev,
                     rms=float(np.sqrt(np.mean(dev**2))),
                     maxdev=float(np.abs(dev).max()),
                     rms_jy=float(np.sqrt(cy / len(z))),
                     rms_jmu=float(np.sqrt(cm / ok_mu.sum()))))

rows.sort(key=lambda r: r["rms"])

hdr = (f"{'dataset':12s}{'J_bb*':11s}{'resid':6s}{'p':>3s}"
       f"{'z_y':>9s}{'a_y':>8s}{'z_mu':>9s}{'a_mu':>8s}"
       f"{'RMS dev':>9s}{'max dev':>9s}{'rms J_y':>10s}{'rms J_mu':>10s}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    p = r["params"]
    print(f"{r['dataset']:12s}{r['jbb']:11s}{r['resid']:6s}{r['power']:3.0f}"
          f"{p[0]:9.0f}{p[1]:8.3f}{p[2]:9.0f}{p[3]:8.3f}"
          f"{r['rms']:8.2f}%{r['maxdev']:8.2f}%{r['rms_jy']:10.5f}{r['rms_jmu']:10.5f}")

print(f"\nChluba (2013) published:{'':16s}"
      f"{TARGET_Y[0]:9.0f}{TARGET_Y[1]:8.2f}{TARGET_MU[0]:9.0f}{TARGET_MU[1]:8.2f}")

print("\ntop 5 by RMS deviation from the published four:")
for r in rows[:5]:
    print(f"  {r['dataset']:12s} J_bb* {r['jbb']:10s} resid {r['resid']:3s} "
          f"p={r['power']:.0f}   RMS {r['rms']:.2f}%")
    print("      " + ", ".join(f"{n} {d:+.2f}%" for n, d in
                               zip(["z_y", "alpha_y", "z_mu", "alpha_mu"], r["dev"])))

(DATADIR / "visibility_scalar_leastsq_chluba.json").write_text(json.dumps(
    [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
     for r in rows], indent=1))
print(f"\nwrote {DATADIR / 'visibility_scalar_leastsq_chluba.json'}")
