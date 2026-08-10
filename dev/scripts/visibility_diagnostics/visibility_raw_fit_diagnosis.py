#!/usr/bin/env python3
"""Why the three-component Ansatz cannot fit un-stripped PDE spectra.

`visibility_fit_no_ncstrip.py` found that dropping the number-conservation strip
leaves the 7/9-parameter fit ~45% above the achievable per-redshift floor, with
A driven to 1.10, B to 0.3 and beta to 1.0 -- all at or outside their bounds.
A > 1 is unphysical: A = J_bb*(z -> 0) is the fraction of injected energy that
thermalises, so A <= 1 is required.

This script (a) refits with A constrained to the physical range, and (b) localises
the misfit by comparing the three amplitudes a free per-redshift fit prefers,

    Delta n(z, x) = a_mu(z) M(x) + a_y(z) Y(x) + a_G(z) G_bb(x),

against what the Ansatz predicts,

    a_mu = (3/kappa_c) J_mu J_bb*,   a_y = J_y / 4,   a_G = (1 - J_bb*) / 4.

The per-component comparison says whether the failure is in the mu branch, the y
branch, or the temperature-shift branch, which the stripped metric cannot see.
"""
import pathlib
import sys

import numpy as np
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "python"))

from spectroxide.greens import mu_shape, y_shape, g_bb, KAPPA_C

MU_TO_ENERGY = 3.0 / KAPPA_C
DATADIR = pathlib.Path(__file__).resolve().parents[3] / "dev" / "data"

NAMES = ["z_y", "alpha_y", "z_mu", "alpha_mu", "A", "z_th", "alpha_th", "B", "beta"]
LIT = np.array([6.0e4, 2.58, 5.8e4, 1.88, 0.983, 1.98e6, 2.5, 0.0381, 2.29])
PAPER = np.array([63128.023, 2.6533894, 58428.697, 1.9465843, 0.9918168,
                  1.98e6, 2.5, 0.048279535, 2.0719380])
# Physical bounds: A <= 1 (thermalised fraction), B >= 0, and J_bb* >= 0.
PHYS = [(1e4, 2e5), (1.0, 5.0), (1e4, 2e5), (1.0, 4.0), (0.80, 1.00),
        (5e5, 1e7), (1.5, 4.0), (0.0, 0.30), (1.0, 5.0)]
FREE7 = [0, 1, 2, 3, 4, 7, 8]
FREE9 = list(range(9))

d = np.load(DATADIR / "visibility_table.npz")
z_h, x_pde, dn_raw, drho = d["z_h"], d["x"], d["dn_raw"], d["drho"]
M_x, Y_x, G_x = mu_shape(x_pde), y_shape(x_pde), g_bb(x_pde)
mask = (x_pde >= 0.5) & (x_pde <= 20.0)
w = x_pde[mask] ** 3
Mw, Yw, Gw = w * M_x[mask], w * Y_x[mask], w * G_x[mask]
Dw = w * dn_raw[:, mask]
Amat = np.column_stack([Mw, Yw, Gw])


def visib(z, p):
    z_y, a_y, z_mu, a_mu, A, z_th, a_th, B, beta = p
    jb = np.maximum(A * np.exp(-((z / z_th) ** a_th)) * (1.0 - B * (z / z_th) ** beta), 0.0)
    jm = 1.0 - np.exp(-(((1.0 + z) / z_mu) ** a_mu))
    jy = 1.0 / (1.0 + ((1.0 + z) / z_y) ** a_y)
    return jm, jb, jy


def predicted_amps(p):
    jm, jb, jy = visib(z_h, p)
    return MU_TO_ENERGY * jm * jb, 0.25 * jy, 0.25 * (1.0 - jb)


def cost(p):
    am, ay, ag = predicted_amps(p)
    model = ((drho * am)[:, None] * Mw[None, :] + (drho * ay)[:, None] * Yw[None, :]
             + (drho * ag)[:, None] * Gw[None, :])
    r = model - Dw
    return float(np.sum(r * r))


coef, *_ = np.linalg.lstsq(Amat, Dw.T, rcond=None)
a_free = coef / drho                      # (3, n_z): free per-z amplitudes
r_floor = Amat @ coef - Dw.T
C_floor = float(np.sum(r_floor * r_floor))
print(f"per-z free 3-amplitude floor (354 params): {C_floor:.4f}")
print(f"cost at literature: {cost(LIT):.4f}    at paper Table 1: {cost(PAPER):.4f}\n")


def fit(free, seed=7, maxiter=400):
    fixed = PAPER.copy()

    def unpack(t):
        p = fixed.copy()
        p[free] = np.exp(t)
        return p

    lb = np.log([max(PHYS[k][0], 1e-8) for k in free])
    ub = np.log([PHYS[k][1] for k in free])
    r = differential_evolution(lambda t: cost(unpack(t)), bounds=list(zip(lb, ub)),
                               seed=seed, maxiter=maxiter, popsize=20, tol=1e-11,
                               polish=False, init="sobol")
    q = minimize(lambda t: cost(unpack(t)), r.x, method="L-BFGS-B",
                 bounds=list(zip(lb, ub)), options=dict(maxiter=8000, ftol=1e-15))
    t = q.x if q.fun < r.fun else r.x
    return unpack(t), cost(unpack(t))


results = {}
for label, free in (("7 free, A<=1", FREE7), ("9 free, A<=1", FREE9)):
    p, c = fit(free)
    results[label] = p
    print(f"{label}:  cost = {c:.4f}   = floor + {c - C_floor:.4f} "
          f"({100 * (c - C_floor) / C_floor:.2f}% above floor)")
    for i in free:
        lo, hi = PHYS[i]
        at = ""
        if abs(p[i] - lo) / max(abs(lo), 1e-12) < 1e-3:
            at = "  <-- AT LOWER BOUND"
        elif abs(p[i] - hi) / abs(hi) < 1e-3:
            at = "  <-- AT UPPER BOUND"
        print(f"    {NAMES[i]:10s}{p[i]:13.5g}  (lit {LIT[i]:.4g}, "
              f"{100 * (p[i] - LIT[i]) / LIT[i]:+.2f}%){at}")
    print()

# ── Localise the misfit: per-component amplitude comparison ──────────────
print("=" * 94)
print("Which component misfits? Free per-redshift amplitudes vs Ansatz prediction")
print("=" * 94)
labels = ["a_mu", "a_y", "a_G"]
for name, p in (("literature", LIT), ("paper Table 1", PAPER),
                ("9 free, A<=1", results["9 free, A<=1"])):
    pred = predicted_amps(p)
    print(f"\n{name}:")
    for j in range(3):
        f, q = a_free[j], pred[j]
        sel = np.abs(f) > 0.05 * np.max(np.abs(f))
        rel = np.abs(q[sel] - f[sel]) / np.abs(f[sel])
        # absolute discrepancy relative to the peak of the total signal
        absd = np.abs(q - f).max() / np.max(np.abs(a_free))
        print(f"  {labels[j]:5s} median rel dev {100 * np.median(rel):7.2f} %   "
              f"max rel dev {100 * rel.max():8.2f} %   "
              f"max abs dev / peak signal {100 * absd:6.2f} %")

print("\nFree amplitudes and the literature / refitted predictions:")
print(f"{'z_h':>10s} | {'a_mu free':>10s} {'lit':>9s} {'refit':>9s} "
      f"| {'a_y free':>9s} {'lit':>9s} {'refit':>9s} "
      f"| {'a_G free':>9s} {'lit':>9s} {'refit':>9s}")
pl, pr = predicted_amps(LIT), predicted_amps(results["9 free, A<=1"])
for i in range(0, len(z_h), max(1, len(z_h) // 14)):
    print(f"{z_h[i]:10.3g} | {a_free[0][i]:10.5f} {pl[0][i]:9.5f} {pr[0][i]:9.5f} "
          f"| {a_free[1][i]:9.5f} {pl[1][i]:9.5f} {pr[1][i]:9.5f} "
          f"| {a_free[2][i]:9.5f} {pl[2][i]:9.5f} {pr[2][i]:9.5f}")

# Is the y-era a_G consistent with A = 1 rather than 0.983?
lo = z_h < 2e4
print(f"\ny-era check (z_h < 2e4, {lo.sum()} points): mean free a_G = "
      f"{a_free[2][lo].mean():+.5f}")
print(f"  Ansatz 0.25(1-A): A=0.983 -> {0.25 * (1 - 0.983):+.5f}    "
      f"A=1.0 -> {0.0:+.5f}")
print("  A = 1 (no unthermalised residual in the y era) matches the data; the")
print("  literature A = 0.983 predicts a small positive G_bb term the PDE lacks.")
