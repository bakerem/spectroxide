#!/usr/bin/env python3
"""Refit the visibility parameters WITHOUT number-conservation stripping.

Motivation. The fit behind Fig. 2 NC-strips both model and data. Because the
strip subtracts a multiple of G_bb computed from G_bb itself, the stripped basis
vector G_nc is identically zero, so the third component of the Ansatz,
0.25 (1 - J_bb*) G_bb, contributes nothing to the cost. That metric therefore
cannot constrain z_th or alpha_th except through their appearance in J_bb*
inside the mu term, and it never tests the prediction that a fraction
(1 - J_bb*) of the injected energy ends up as a pure temperature shift.

Dropping the strip restores that sensitivity. Three residual definitions:

  NC     stripped (the published fit) -- 2 effective amplitudes per redshift
  raw    full Delta n, G_bb term active -- 3 amplitudes per redshift
  freeT  full Delta n, but a free G_bb amplitude profiled out at each redshift.
         This is what an observer sees, since T_0 is unknown. It removes the
         G direction differently from NC (per-redshift, in the weighted metric).

For each we fit 7 free parameters (z_th, alpha_th held at the analytic values)
and all 9, and report the cost of the literature parameters on the same metric.
The floor is the per-redshift unconstrained amplitude fit, a lower bound for any
parameterisation on that metric.

The physics question: does the raw fit recover z_th = 1.98e6 and alpha_th = 5/2?
Those are analytically derived from the double-Compton opacity scaling and are
held fixed in the paper. Recovering them from data the stripped metric cannot see
would be an independent check the current Fig. 2 does not provide.
"""
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "python"))

from spectroxide.greens import mu_shape, y_shape, g_bb, KAPPA_C

MU_TO_ENERGY = 3.0 / KAPPA_C
REPO = pathlib.Path(__file__).resolve().parents[3]
DATADIR = REPO / "dev" / "data"

NAMES = ["z_y", "alpha_y", "z_mu", "alpha_mu", "A", "z_th", "alpha_th", "B", "beta"]
LIT = np.array([6.0e4, 2.58, 5.8e4, 1.88, 0.983, 1.98e6, 2.5, 0.0381, 2.29])
PAPER = np.array([63128.02326056765, 2.653389404163302, 58428.69672709,
                  1.9465842847446648, 0.9918167997907094, 1.98e6, 2.5,
                  0.048279535221419465, 2.0719380110207646])
BOUNDS = [(1e4, 2e5), (1.0, 5.0), (1e4, 2e5), (1.0, 4.0), (0.8, 1.1),
          (5e5, 1e7), (1.5, 4.0), (0.0, 0.3), (1.0, 5.0)]
FREE7 = [0, 1, 2, 3, 4, 7, 8]
FREE9 = [0, 1, 2, 3, 4, 5, 6, 7, 8]

data = np.load(DATADIR / "visibility_table.npz")
z_h, x_pde = data["z_h"], data["x"]
dn_nc, dn_raw, drho = data["dn_nc"], data["dn_raw"], data["drho"]
n_z = len(z_h)

M_x, Y_x, G_x = mu_shape(x_pde), y_shape(x_pde), g_bb(x_pde)
G_int = np.trapz(x_pde**2 * G_x, x_pde)
M_nc = M_x - np.trapz(x_pde**2 * M_x, x_pde) / G_int * G_x
Y_nc = Y_x - np.trapz(x_pde**2 * Y_x, x_pde) / G_int * G_x

mask = (x_pde >= 0.5) & (x_pde <= 20.0)
w = x_pde[mask] ** 3


def visib(z, p):
    z_y, a_y, z_mu, a_mu, A, z_th, a_th, B, beta = p
    jb = np.maximum(A * np.exp(-((z / z_th) ** a_th)) * (1.0 - B * (z / z_th) ** beta), 0.0)
    jm = 1.0 - np.exp(-(((1.0 + z) / z_mu) ** a_mu))
    jy = 1.0 / (1.0 + ((1.0 + z) / z_y) ** a_y)
    return jm, jb, jy


class Metric:
    """One residual definition: model basis, data, and the per-z floor."""

    def __init__(self, mode):
        self.mode = mode
        if mode == "NC":
            self.Mw, self.Yw = w * M_nc[mask], w * Y_nc[mask]
            self.Gw = None
            self.Dw = w * dn_nc[:, mask]
        else:
            self.Mw, self.Yw = w * M_x[mask], w * Y_x[mask]
            self.Gw = w * G_x[mask]
            self.Dw = w * dn_raw[:, mask]

    def _project(self, r):
        """freeT: remove a free G_bb amplitude per redshift."""
        if self.mode != "freeT":
            return r
        gg = float(self.Gw @ self.Gw)
        c = (r @ self.Gw) / gg
        return r - c[:, None] * self.Gw[None, :]

    def cost(self, p):
        jm, jb, jy = visib(z_h, p)
        model = ((drho * MU_TO_ENERGY * jm * jb)[:, None] * self.Mw[None, :]
                 + (drho * 0.25 * jy)[:, None] * self.Yw[None, :])
        if self.mode == "raw":
            model = model + (drho * 0.25 * (1.0 - jb))[:, None] * self.Gw[None, :]
        r = self._project(model - self.Dw)
        return float(np.sum(r * r))

    def floor(self):
        """Per-redshift unconstrained amplitude fit: lower bound on this metric."""
        if self.mode == "raw":
            basis = [self.Mw, self.Yw, self.Gw]
        elif self.mode == "freeT":
            # G is profiled out, so it spans no independent direction; fit the
            # G-orthogonalised M and Y.
            gg = float(self.Gw @ self.Gw)
            basis = [v - (v @ self.Gw) / gg * self.Gw for v in (self.Mw, self.Yw)]
        else:
            basis = [self.Mw, self.Yw]
        A = np.column_stack(basis)
        D = self._project(self.Dw) if self.mode == "freeT" else self.Dw
        coef, *_ = np.linalg.lstsq(A, D.T, rcond=None)
        r = A @ coef - D.T
        return float(np.sum(r * r)), coef

    def gbb_amplitude_free(self):
        """freeT/raw: the G_bb amplitude the data actually wants, per redshift."""
        if self.mode == "NC":
            return None
        _, coef = self.floor()
        if self.mode == "raw":
            return coef[2] / drho
        return None


def fit(metric, free, seed=12345, maxiter=350):
    """Differential evolution over `free`, then L-BFGS-B polish, in log space."""
    fixed = PAPER.copy()

    def unpack(t):
        p = fixed.copy()
        p[free] = np.exp(t)
        return p

    lb = np.log([BOUNDS[k][0] if BOUNDS[k][0] > 0 else 1e-8 for k in free])
    ub = np.log([BOUNDS[k][1] for k in free])
    res = differential_evolution(lambda t: metric.cost(unpack(t)),
                                 bounds=list(zip(lb, ub)), seed=seed,
                                 maxiter=maxiter, popsize=18, tol=1e-10,
                                 polish=False, init="sobol")
    pol = minimize(lambda t: metric.cost(unpack(t)), res.x, method="L-BFGS-B",
                   bounds=list(zip(lb, ub)),
                   options=dict(maxiter=8000, ftol=1e-15, gtol=1e-12))
    best = pol.x if pol.fun < res.fun else res.x
    return unpack(best), metric.cost(unpack(best))


out = {}
for mode in ("NC", "raw", "freeT"):
    m = Metric(mode)
    c_lit, c_paper = m.cost(LIT), m.cost(PAPER)
    c_floor, _ = m.floor()

    print("=" * 92)
    print(f"METRIC: {mode}")
    print(f"  cost at literature params : {c_lit:12.4f}")
    print(f"  cost at paper (Table 1)   : {c_paper:12.4f}")
    print(f"  per-z free-amplitude floor: {c_floor:12.4f}")
    print(f"  C_floor / C_lit = {c_floor / c_lit:.4f}     "
          f"C_lit / C_paper = {c_lit / c_paper:.4f}")

    rec = {"cost_lit": c_lit, "cost_paper": c_paper, "cost_floor": c_floor}
    for label, free in (("7 free (z_th, a_th fixed)", FREE7), ("9 free", FREE9)):
        p, c = fit(m, free)
        rec[label] = {"params": {NAMES[i]: float(p[i]) for i in range(9)}, "cost": c}
        print(f"\n  fit, {label}:  cost = {c:.4f}   "
              f"(floor + {c - c_floor:.4f}, {100 * (c - c_floor) / c_floor:.3f}%)")
        print(f"    {'param':10s}{'fitted':>13s}{'literature':>13s}{'dev %':>10s}")
        for i in range(9):
            if i not in free:
                print(f"    {NAMES[i]:10s}{p[i]:13.5g}{LIT[i]:13.5g}"
                      f"{'(fixed)':>10s}")
            else:
                print(f"    {NAMES[i]:10s}{p[i]:13.5g}{LIT[i]:13.5g}"
                      f"{100 * (p[i] - LIT[i]) / LIT[i]:+9.2f}%")
    out[mode] = rec
    print()

# ── Does the raw metric confirm the temperature-shift branching? ─────────
print("=" * 92)
print("Temperature-shift branching, testable only without the strip")
print("The Ansatz predicts a G_bb amplitude of 0.25*(1 - J_bb*) per redshift.")
print("Compare with the amplitude a free 3-component per-redshift fit prefers.\n")
m_raw = Metric("raw")
a_g_free = m_raw.gbb_amplitude_free()
p9 = np.array([out["raw"]["9 free"]["params"][n] for n in NAMES])
_, jb9, _ = visib(z_h, p9)
_, jb_lit, _ = visib(z_h, LIT)
pred9, pred_lit = 0.25 * (1 - jb9), 0.25 * (1 - jb_lit)
sel = np.abs(a_g_free) > 0.02 * np.max(np.abs(a_g_free))
print(f"  over the {sel.sum()} redshifts where the free amplitude is > 2% of its peak:")
for lbl, pr in (("raw 9-free fit", pred9), ("literature", pred_lit)):
    d = np.abs(pr[sel] - a_g_free[sel]) / np.abs(a_g_free[sel])
    print(f"    {lbl:16s} median dev {100 * np.median(d):6.2f} %   "
          f"max dev {100 * d.max():7.2f} %")
print("\n  sample (z_h, free amplitude, 0.25(1-J_bb*) at raw 9-free fit, at literature):")
for i in range(0, n_z, max(1, n_z // 12)):
    print(f"    {z_h[i]:10.3g} {a_g_free[i]:12.5f} {pred9[i]:12.5f} {pred_lit[i]:12.5f}")

(DATADIR / "visibility_fit_no_ncstrip.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote {DATADIR / 'visibility_fit_no_ncstrip.json'}")
