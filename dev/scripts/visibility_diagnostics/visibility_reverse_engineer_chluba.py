#!/usr/bin/env python3
"""Which fit procedure reproduces Chluba's published J_mu and J_y parameters?

Hypothesis under test. In Chluba's original treatment the thermalisation
visibility carries no free parameters: J_bb* is the analytic
exp(-(z/z_th)^{5/2}) with z_th = 1.98e6 and alpha_th = 5/2 fixed by the
double-Compton opacity scaling. If the spectra are number-conservation stripped,
the G_bb temperature-shift direction is removed exactly (G_nc is identically
zero), so the only free parameters left are those of J_mu and J_y, fitted by
least squares. That is a well-posed four-parameter problem and a plausible
reconstruction of what produced

    z_y = 6.0e4,  alpha_y = 2.58,  z_mu = 5.8e4,  alpha_mu = 1.88 .

Scan over the procedure space, for our PDE spectra and for CosmoTherm's own
Green's function database:

    J_bb*      analytic (A=1, B=0)  |  literature (A=0.983, B=0.0381, beta=2.29)
    residual   NC-stripped          |  G_bb term subtracted analytically
    weighting  x^p, p = 0 .. 4      (p=0 is least squares on Delta n,
                                     p=3 is least squares on Delta I ~ x^3 Delta n)
    measure    node sum (no dx)     |  trapezoidal dx quadrature

Implementation note. The cost is quadratic in the two amplitudes at each
redshift, so all frequency integrals are precomputed once as Gram matrices and
each cost evaluation is O(N_z) rather than O(N_z * N_x):

    C(p) = sum_k [ a_k^2 Gmm + 2 a_k b_k Gmy + b_k^2 Gyy
                   - 2 a_k P_k - 2 b_k Q_k + R_k ]

with a_k, b_k the model amplitudes and Gmm, Gmy, Gyy, P_k, Q_k, R_k fixed.
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

TARGET = np.array([6.0e4, 2.58, 5.8e4, 1.88])
NAMES = ["z_y", "alpha_y", "z_mu", "alpha_mu"]
BOUNDS = [(2e4, 2e5), (1.0, 5.0), (2e4, 2e5), (1.0, 4.0)]
POWERS = (0.0, 1.0, 2.0, 3.0, 4.0)

JBB = {
    "analytic":   dict(A=1.0,   B=0.0,    beta=2.29),
    "literature": dict(A=0.983, B=0.0381, beta=2.29),
}


def j_bb_star(z, q):
    r = z / Z_TH
    return np.maximum(q["A"] * np.exp(-(r ** A_TH)) * (1.0 - q["B"] * r ** q["beta"]), 0.0)


def j_mu(z, z_mu, a_mu):
    return 1.0 - np.exp(-(((1.0 + z) / z_mu) ** a_mu))


def j_y(z, z_y, a_y):
    return 1.0 / (1.0 + ((1.0 + z) / z_y) ** a_y)


# --------------------------------------------------------------------------
# Datasets: full Delta n per unit Delta rho/rho, on each code's native x grid
# --------------------------------------------------------------------------
tab = np.load(DATADIR / "visibility_table.npz")
DATA = {"spectroxide": (tab["z_h"], tab["x"], tab["dn_raw"] / tab["drho"][:, None])}

z_ct, x_ct, g_stored, md = load_greens_database(include_metadata=True)
g_full = reconstruct_full_gf(x_ct, g_stored, z_ct, md, apply_exp=True)
dn_ct = np.column_stack([cosmotherm_gf_to_delta_n(x_ct, g_full[:, k])
                         for k in range(len(z_ct))]).T
sel = (z_ct >= tab["z_h"].min()) & (z_ct <= tab["z_h"].max())
DATA["CosmoTherm"] = (z_ct[sel], x_ct, dn_ct[sel])


def _strip(x, arr, G, Gnorm):
    """Enforce Int x^2 arr dx = 0 by removing a multiple of G_bb."""
    return arr - (np.trapz(x**2 * arr, x) / Gnorm) * G


_shape_cache = {}


def shapes(dataset, jbb_key, resid):
    """(M, Y, D, jb) after the chosen temperature-shift treatment. Cached."""
    key = (dataset, resid, jbb_key if resid == "sub" else None)
    if key in _shape_cache:
        return _shape_cache[key]
    z, x, dn = DATA[dataset]
    jb = j_bb_star(z, JBB[jbb_key])
    G = g_bb(x)
    Gnorm = np.trapz(x**2 * G, x)
    if resid == "NC":
        M = _strip(x, mu_shape(x), G, Gnorm)
        Y = _strip(x, y_shape(x), G, Gnorm)
        D = np.array([_strip(x, row, G, Gnorm) for row in dn])
    elif resid == "sub":
        M, Y = mu_shape(x), y_shape(x)
        D = dn - (0.25 * (1.0 - jb))[:, None] * G[None, :]
    else:
        raise ValueError(resid)
    _shape_cache[key] = (M, Y, D)
    return _shape_cache[key]


class Problem:
    def __init__(self, dataset, jbb_key, resid, meas, power, x_lo=0.5, x_hi=20.0):
        z, x, _ = DATA[dataset]
        M, Y, D = shapes(dataset, jbb_key, resid)
        self.z = z
        self.jb = j_bb_star(z, JBB[jbb_key])

        m = (x >= x_lo) & (x <= x_hi)
        xs = x[m]
        qw = xs ** (2 * power) * (np.ones_like(xs) if meas == "node" else np.gradient(xs))
        Ms, Ys, Ds = M[m], Y[m], D[:, m]

        self.Gmm = float(np.sum(qw * Ms * Ms))
        self.Gmy = float(np.sum(qw * Ms * Ys))
        self.Gyy = float(np.sum(qw * Ys * Ys))
        self.P = Ds @ (qw * Ms)
        self.Q = Ds @ (qw * Ys)
        self.R = float(np.sum(qw[None, :] * Ds * Ds))

    def cost(self, p):
        z_y, a_y, z_mu, a_mu = p
        a = MU_TO_ENERGY * j_mu(self.z, z_mu, a_mu) * self.jb
        b = 0.25 * j_y(self.z, z_y, a_y)
        return float(np.sum(a * a) * self.Gmm + 2.0 * np.sum(a * b) * self.Gmy
                     + np.sum(b * b) * self.Gyy
                     - 2.0 * np.sum(a * self.P) - 2.0 * np.sum(b * self.Q) + self.R)

    def floor(self):
        det = self.Gmm * self.Gyy - self.Gmy**2
        a = (self.Gyy * self.P - self.Gmy * self.Q) / det
        b = (self.Gmm * self.Q - self.Gmy * self.P) / det
        return float(self.R - np.sum(a * self.P) - np.sum(b * self.Q))


def fit(prob, seed=3):
    lb = np.log([c[0] for c in BOUNDS])
    ub = np.log([c[1] for c in BOUNDS])
    bnd = list(zip(lb, ub))
    r = differential_evolution(lambda t: prob.cost(np.exp(t)), bounds=bnd, seed=seed,
                              maxiter=800, popsize=28, tol=1e-14, polish=False,
                              init="sobol")
    q = minimize(lambda t: prob.cost(np.exp(t)), r.x, method="L-BFGS-B", bounds=bnd,
                 options=dict(maxiter=40000, ftol=1e-18))
    t = q.x if q.fun < r.fun else r.x
    return np.exp(t), prob.cost(np.exp(t))


rows = []
combos = list(itertools.product(DATA, JBB, ("NC", "sub"), ("node", "dx"), POWERS))
for i, (dset, jk, rs, ms, pw) in enumerate(combos, 1):
    prob = Problem(dset, jk, rs, ms, pw)
    p, c = fit(prob)
    dev = 100.0 * (p - TARGET) / TARGET
    rms = float(np.sqrt(np.mean(dev**2)))
    rows.append(dict(dataset=dset, jbb=jk, resid=rs, meas=ms, power=pw,
                     params=p, dev=dev, maxdev=float(np.abs(dev).max()),
                     rms=rms, cost=c, floor=prob.floor(), clit=prob.cost(TARGET)))
    print(f"[{i:3d}/{len(combos)}] {dset:12s} {jk:10s} {rs:3s} {ms:4s} p={pw:.0f}  "
          f"z_y={p[0]:8.0f}  RMS={rms:6.2f}%", flush=True)

rows.sort(key=lambda r: r["rms"])

print()
hdr = (f"{'dataset':12s}{'J_bb*':11s}{'resid':6s}{'meas':6s}{'p':>3s}"
       f"{'z_y':>9s}{'a_y':>8s}{'z_mu':>9s}{'a_mu':>8s}"
       f"{'RMS dev':>9s}{'max dev':>9s}{'C_lit/C_fit':>12s}{'above floor':>12s}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    p, d = r["params"], r["dev"]
    print(f"{r['dataset']:12s}{r['jbb']:11s}{r['resid']:6s}{r['meas']:6s}{r['power']:3.0f}"
          f"{p[0]:9.0f}{p[1]:8.3f}{p[2]:9.0f}{p[3]:8.3f}"
          f"{r['rms']:8.2f}%{r['maxdev']:8.2f}%"
          f"{r['clit'] / r['cost']:12.4f}{100 * (r['cost'] / r['floor'] - 1):11.2f}%")

print(f"\nChluba (2013) published:{'':25s}"
      f"{TARGET[0]:9.0f}{TARGET[1]:8.2f}{TARGET[2]:9.0f}{TARGET[3]:8.2f}")

print("\ntop 5 procedures by RMS deviation from the published four:")
for r in rows[:5]:
    print(f"  {r['dataset']:12s} J_bb* {r['jbb']:10s} resid {r['resid']:3s} "
          f"meas {r['meas']:4s} p={r['power']:.0f}   RMS {r['rms']:.2f}%")
    print("      " + ", ".join(f"{n} {d:+.2f}%" for n, d in zip(NAMES, r["dev"])))

(DATADIR / "visibility_reverse_engineer_chluba.json").write_text(json.dumps(
    [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
     for r in rows], indent=1))
print(f"\nwrote {DATADIR / 'visibility_reverse_engineer_chluba.json'}")
