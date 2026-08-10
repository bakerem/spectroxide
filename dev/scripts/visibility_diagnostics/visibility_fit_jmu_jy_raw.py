#!/usr/bin/env python3
"""Fit only J_mu and J_y to un-NC-stripped PDE spectra, with J_bb* held fixed.

Rationale. Fitting all 7-9 parameters to un-stripped spectra fails: A runs to
1.10 (unphysical, since A = J_bb*(z->0) is a thermalised fraction and must be
<= 1), B and beta pin to their bounds, and the result sits ~45% above the
achievable per-redshift floor. But J_bb* is the one visibility function with an
analytic backbone -- z_th = 1.98e6 and alpha_th = 5/2 follow from the redshift
scaling of the double-Compton opacity (Hu & Silk 1993) -- so it should be held
fixed rather than fitted. That leaves the four parameters the paper calls
primary,

    J_mu: z_mu, alpha_mu        J_y: z_y, alpha_y

to be fit against the full spectra, with the temperature-shift term
0.25 (1 - J_bb*) G_bb now a known function of z rather than something the strip
discards.

Three configurations, all on the same x^3-weighted metric over x in [0.5, 20]:

  raw,  J_bb* = literature (A = 0.983)
  raw,  J_bb* = literature but A = 1      (the y-era data prefers this; see below)
  NC,   J_bb* = literature                (for direct comparison with Fig. 2)

The floor quoted for each is the per-redshift unconstrained 2-amplitude fit of
(M, Y) to the data minus the fixed G_bb term: a lower bound for any J_mu/J_y.
"""
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "python"))

from spectroxide.greens import mu_shape, y_shape, g_bb, KAPPA_C

MU_TO_ENERGY = 3.0 / KAPPA_C
DATADIR = pathlib.Path(__file__).resolve().parents[3] / "dev" / "data"

# Fixed J_bb* parameters (Chluba 2015 Eq. 13 / Chluba 2013)
JBB_LIT = dict(A=0.983, z_th=1.98e6, a_th=2.5, B=0.0381, beta=2.29)
JBB_A1 = dict(JBB_LIT, A=1.0)

# Free parameters: z_y, alpha_y, z_mu, alpha_mu
P_NAMES = ["z_y", "alpha_y", "z_mu", "alpha_mu"]
P_LIT = np.array([6.0e4, 2.58, 5.8e4, 1.88])
P_PAPER = np.array([63128.023, 2.6533894, 58428.697, 1.9465843])
P_BOUNDS = [(2e4, 2e5), (1.0, 5.0), (2e4, 2e5), (1.0, 4.0)]

d = np.load(DATADIR / "visibility_table.npz")
z_h, x_pde = d["z_h"], d["x"]
dn_raw, dn_nc, drho = d["dn_raw"], d["dn_nc"], d["drho"]
n_z = len(z_h)

M_x, Y_x, G_x = mu_shape(x_pde), y_shape(x_pde), g_bb(x_pde)
G_int = np.trapz(x_pde**2 * G_x, x_pde)
M_nc = M_x - np.trapz(x_pde**2 * M_x, x_pde) / G_int * G_x
Y_nc = Y_x - np.trapz(x_pde**2 * Y_x, x_pde) / G_int * G_x

mask = (x_pde >= 0.5) & (x_pde <= 20.0)
w = x_pde[mask] ** 3


def j_bb_star(z, q):
    return np.maximum(
        q["A"] * np.exp(-((z / q["z_th"]) ** q["a_th"]))
        * (1.0 - q["B"] * (z / q["z_th"]) ** q["beta"]), 0.0)


def j_mu(z, z_mu, a_mu):
    return 1.0 - np.exp(-(((1.0 + z) / z_mu) ** a_mu))


def j_y(z, z_y, a_y):
    return 1.0 / (1.0 + ((1.0 + z) / z_y) ** a_y)


class Config:
    def __init__(self, mode, jbb):
        self.mode, self.jbb = mode, jbb
        self.jb = j_bb_star(z_h, jbb)
        if mode == "NC":
            self.Mw, self.Yw = w * M_nc[mask], w * Y_nc[mask]
            # G_nc is identically zero, so the temperature-shift term drops out.
            self.Dw = w * dn_nc[:, mask]
            self.fixed = np.zeros_like(self.Dw)
        else:
            self.Mw, self.Yw = w * M_x[mask], w * Y_x[mask]
            self.Dw = w * dn_raw[:, mask]
            Gw = w * G_x[mask]
            self.fixed = (drho * 0.25 * (1.0 - self.jb))[:, None] * Gw[None, :]
        self.resid_target = self.Dw - self.fixed   # what M and Y must explain

    def cost(self, p):
        z_y, a_y, z_mu, a_mu = p
        am = MU_TO_ENERGY * j_mu(z_h, z_mu, a_mu) * self.jb
        ay = 0.25 * j_y(z_h, z_y, a_y)
        model = (drho * am)[:, None] * self.Mw[None, :] + (drho * ay)[:, None] * self.Yw[None, :]
        r = model - self.resid_target
        return float(np.sum(r * r))

    def floor(self):
        A = np.column_stack([self.Mw, self.Yw])
        coef, *_ = np.linalg.lstsq(A, self.resid_target.T, rcond=None)
        r = A @ coef - self.resid_target.T
        return float(np.sum(r * r)), coef / drho


def fit(cfg, seed=11, maxiter=600):
    lb = np.log([b[0] for b in P_BOUNDS])
    ub = np.log([b[1] for b in P_BOUNDS])
    r = differential_evolution(lambda t: cfg.cost(np.exp(t)), bounds=list(zip(lb, ub)),
                              seed=seed, maxiter=maxiter, popsize=24, tol=1e-12,
                              polish=False, init="sobol")
    q = minimize(lambda t: cfg.cost(np.exp(t)), r.x, method="L-BFGS-B",
                 bounds=list(zip(lb, ub)), options=dict(maxiter=20000, ftol=1e-16))
    t = q.x if q.fun < r.fun else r.x
    return np.exp(t), cfg.cost(np.exp(t))


CONFIGS = [
    ("raw, J_bb* = literature (A=0.983)", "raw", JBB_LIT),
    ("raw, J_bb* = literature but A=1.0", "raw", JBB_A1),
    ("NC-stripped, J_bb* = literature  ", "NC", JBB_LIT),
]

out = {}
print(f"{n_z} redshifts, {int(mask.sum())} freq points in x in [0.5, 20], x^3 weighting")
print("Free parameters: z_y, alpha_y, z_mu, alpha_mu.  J_bb* held fixed.\n")

for label, mode, jbb in CONFIGS:
    cfg = Config(mode, jbb)
    c_floor, _ = cfg.floor()
    c_lit, c_paper = cfg.cost(P_LIT), cfg.cost(P_PAPER)
    p, c = fit(cfg)

    print("=" * 94)
    print(label)
    print(f"  floor (per-z free M,Y: {2 * n_z} params) = {c_floor:11.4f}")
    print(f"  cost at literature (z_y,a_y,z_mu,a_mu)  = {c_lit:11.4f}   "
          f"({100 * (c_lit - c_floor) / c_floor:+.2f}% above floor)")
    print(f"  cost at paper Table 1 values            = {c_paper:11.4f}   "
          f"({100 * (c_paper - c_floor) / c_floor:+.2f}% above floor)")
    print(f"  cost at best fit                        = {c:11.4f}   "
          f"({100 * (c - c_floor) / c_floor:+.2f}% above floor)")
    print(f"  C_lit / C_fit = {c_lit / c:.4f}\n")
    print(f"  {'param':10s}{'fitted':>13s}{'literature':>13s}{'dev %':>9s}"
          f"{'paper Fig.2':>13s}{'dev %':>9s}")
    for i, nm in enumerate(P_NAMES):
        lo, hi = P_BOUNDS[i]
        at = "  AT BOUND" if (abs(p[i] - lo) / lo < 1e-3 or abs(p[i] - hi) / hi < 1e-3) else ""
        print(f"  {nm:10s}{p[i]:13.5g}{P_LIT[i]:13.5g}"
              f"{100 * (p[i] - P_LIT[i]) / P_LIT[i]:+8.2f}%"
              f"{P_PAPER[i]:13.5g}{100 * (p[i] - P_PAPER[i]) / P_PAPER[i]:+8.2f}%{at}")
    print()
    out[label.strip()] = {
        "params": {nm: float(p[i]) for i, nm in enumerate(P_NAMES)},
        "cost_fit": c, "cost_lit": c_lit, "cost_paper": c_paper, "cost_floor": c_floor,
    }

(DATADIR / "visibility_fit_jmu_jy_raw.json").write_text(json.dumps(out, indent=1))
print(f"wrote {DATADIR / 'visibility_fit_jmu_jy_raw.json'}")
