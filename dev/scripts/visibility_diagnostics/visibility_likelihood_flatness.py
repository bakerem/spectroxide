#!/usr/bin/env python3
"""Quantify how weakly the visibility-function parameters are constrained.

Referee response for Fig. 2 (paper.tex fig:visibility): our PDE-derived
visibility parameters differ from the Chluba (2013, 2015) fitting formulas by
up to 5% in the primary parameters and ~27% in B. This script tests whether
that difference is statistically meaningful by asking three questions:

  Q1  How much does the spectral cost actually improve between the literature
      parameters and our fit?  ->  cost_lit vs cost_fit.

  Q2  How much of the residual is *irreducible*?  Under number-conservation
      stripping the g_bb basis vector vanishes identically (G_nc == 0), so the
      three-component Ansatz has only TWO free amplitudes per redshift:
          a_mu(z) = (3/kappa_c) * J_mu(z) * J_bb*(z)
          a_y(z)  = (1/4) * J_y(z)
      A per-redshift unconstrained least-squares fit of those two amplitudes is
      therefore a hard lower bound on the cost of ANY visibility
      parameterisation. If cost_floor ~ cost_fit ~ cost_lit, the residual is set
      by the Ansatz shape, not by the parameters.

  Q3  How flat is the cost in each parameter?  1-D profiles + the Hessian give
      the curvature and the degeneracy directions.

Outputs a summary table and dev/data/visibility_flatness.json.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "python"))

from spectroxide.greens import mu_shape, y_shape, g_bb, KAPPA_C

MU_TO_ENERGY = 3.0 / KAPPA_C

REPO = pathlib.Path(__file__).resolve().parents[3]
DATADIR = REPO / "dev" / "data"

# ── Parameter sets ──────────────────────────────────────────────────────
PARAM_NAMES = ["z_y", "alpha_y", "z_mu", "alpha_mu", "A", "z_th", "alpha_th", "B", "beta"]
# Chluba 2013 Eq. 5 + Chluba 2015 Eq. 13
LIT = np.array([6.0e4, 2.58, 5.8e4, 1.88, 0.983, 1.98e6, 2.5, 0.0381, 2.29])
# spectral_05_20 solution, dev/data/visibility_spectral_fit_v2.json
FIT = np.array([63128.02326056765, 2.653389404163302, 58428.69672709,
                1.9465842847446648, 0.9918167997907094, 1.98e6, 2.5,
                0.048279535221419465, 2.0719380110207646])
# Free (non-fixed) parameter indices
FREE = [0, 1, 2, 3, 4, 7, 8]

# ── Load PDE reference table ────────────────────────────────────────────
data = np.load(DATADIR / "visibility_table.npz")
z_h = data["z_h"]
x_pde = data["x"]
dn_nc = data["dn_nc"]
drho = data["drho"]
n_z, n_x = len(z_h), len(x_pde)

# ── Basis shapes, NC-stripped exactly as in fit_visibility_from_table.py ──
M_x, Y_x, G_x = mu_shape(x_pde), y_shape(x_pde), g_bb(x_pde)
G_int = np.trapz(x_pde**2 * G_x, x_pde)
M_nc = M_x - np.trapz(x_pde**2 * M_x, x_pde) / G_int * G_x
Y_nc = Y_x - np.trapz(x_pde**2 * Y_x, x_pde) / G_int * G_x
G_nc = G_x - np.trapz(x_pde**2 * G_x, x_pde) / G_int * G_x  # == 0


def _j_bb(z, z_th, a_th):
    return np.exp(-((z / z_th) ** a_th))


def _j_bb_star(z, A, z_th, a_th, B, beta):
    return np.maximum(A * _j_bb(z, z_th, a_th) * (1.0 - B * (z / z_th) ** beta), 0.0)


def _j_mu(z, z_mu, a_mu):
    return 1.0 - np.exp(-(((1.0 + z) / z_mu) ** a_mu))


def _j_y(z, z_y, a_y):
    return 1.0 / (1.0 + ((1.0 + z) / z_y) ** a_y)


def amplitudes(params):
    """(a_mu, a_y) per redshift implied by a visibility parameter vector."""
    z_y, a_y, z_mu, a_mu, A, z_th, a_th, B, beta = params
    jm = _j_mu(z_h, z_mu, a_mu)
    jb = _j_bb_star(z_h, A, z_th, a_th, B, beta)
    jy = _j_y(z_h, z_y, a_y)
    return MU_TO_ENERGY * jm * jb, 0.25 * jy


def make_machinery(x_lo=0.5, x_hi=20.0, weight_power=3):
    mask = (x_pde >= x_lo) & (x_pde <= x_hi)
    w = x_pde[mask] ** weight_power
    Mw, Yw = w * M_nc[mask], w * Y_nc[mask]
    Dw = w * dn_nc[:, mask]            # weighted PDE data, (n_z, n_mask)

    def cost(params):
        a_mu, a_y = amplitudes(params)
        # model_w[i] = drho[i] * (a_mu[i] Mw + a_y[i] Yw)
        model_w = (drho * a_mu)[:, None] * Mw[None, :] + (drho * a_y)[:, None] * Yw[None, :]
        r = model_w - Dw
        return float(np.sum(r * r))

    def cost_floor():
        """Per-redshift free 2-amplitude least squares: hard lower bound."""
        Amat = np.column_stack([Mw, Yw])                    # (n_mask, 2)
        coef, *_ = np.linalg.lstsq(Amat, Dw.T, rcond=None)  # (2, n_z)
        r = Amat @ coef - Dw.T
        return float(np.sum(r * r)), coef[0] / drho, coef[1] / drho

    def data_norm():
        return float(np.sum(Dw * Dw))

    return cost, cost_floor, data_norm, int(mask.sum())


cost, cost_floor, data_norm, n_mask = make_machinery()
n_pts = n_z * n_mask

c_lit, c_fit = cost(LIT), cost(FIT)
c_floor, a_mu_free, a_y_free = cost_floor()
c_data = data_norm()
c_zero = c_data  # cost of the null model (model == 0)

print(f"table: {n_z} redshifts x {n_mask} freq points in [0.5, 20]  =  {n_pts} points")
print(f"z_h in [{z_h[0]:.3g}, {z_h[-1]:.3g}]\n")

print("Q1/Q2  weighted sum-of-squares cost (x^3 weighting, arbitrary units)")
print(f"  null model (Delta n = 0)        C_null  = {c_zero:12.4f}")
print(f"  Chluba 2013/2015 parameters     C_lit   = {c_lit:12.4f}")
print(f"  our PDE fit                     C_fit   = {c_fit:12.4f}")
print(f"  per-z free amplitudes (floor)   C_floor = {c_floor:12.4f}")
print()
print(f"  C_lit / C_fit                        = {c_lit / c_fit:.4f}   "
      f"({100 * (c_lit - c_fit) / c_fit:+.2f}% )")
print(f"  fraction of residual that is irreducible Ansatz error:")
print(f"    C_floor / C_lit = {c_floor / c_lit:.4f}    C_floor / C_fit = {c_floor / c_fit:.4f}")
print(f"  reducible-by-parameters part of C_lit: {100 * (c_lit - c_floor) / c_lit:.2f}%")
print(f"  signal explained: 1 - C_fit/C_null = {1 - c_fit / c_zero:.6f}")
print()

# RMS fractional residual, as a physically readable number
rms_lit = np.sqrt(c_lit / c_data)
rms_fit = np.sqrt(c_fit / c_data)
rms_floor = np.sqrt(c_floor / c_data)
print("  weighted RMS residual as a fraction of the weighted signal:")
print(f"    literature {100 * rms_lit:6.3f} %   fit {100 * rms_fit:6.3f} %   floor {100 * rms_floor:6.3f} %")
print()

# ── Q2b: how far are LIT / FIT amplitudes from the per-z optimum? ────────
a_mu_lit, a_y_lit = amplitudes(LIT)
a_mu_fit, a_y_fit = amplitudes(FIT)


def amp_dev(a, a_free, label, floor_frac=0.05):
    """Max fractional deviation from the free per-z amplitude, where the
    amplitude is above floor_frac of its peak (elsewhere it is unmeasurable)."""
    m = np.abs(a_free) > floor_frac * np.max(np.abs(a_free))
    d = np.abs(a[m] - a_free[m]) / np.abs(a_free[m])
    return f"{label}: max {100 * d.max():5.2f} %  median {100 * np.median(d):5.2f} %"


print("Q2b  deviation of the implied amplitudes from the per-redshift optimum")
print("     " + amp_dev(a_mu_lit, a_mu_free, "a_mu  literature"))
print("     " + amp_dev(a_mu_fit, a_mu_free, "a_mu  our fit   "))
print("     " + amp_dev(a_y_lit, a_y_free, "a_y   literature"))
print("     " + amp_dev(a_y_fit, a_y_free, "a_y   our fit   "))
print()

# ── Q3: 1-D cost profiles and the Delta-C = (C_lit - C_fit) contour ──────
print("Q3  1-D profiles: vary one parameter, hold the rest at the fitted values")
print(f"    reference scale: C_lit - C_fit = {c_lit - c_fit:.4f}")
print()
print(f"    {'param':10s} {'fit':>12s} {'lit':>12s} {'lit-fit %':>10s} "
      f"{'dC to lit':>11s} {'range for dC<=(C_lit-C_fit)':>34s}")

dC_ref = c_lit - c_fit
profiles = {}
intervals = {}
for k in FREE:
    p = FIT.copy()
    lo_frac, hi_frac = 0.3, 3.0
    if PARAM_NAMES[k] == "A":
        lo_frac, hi_frac = 0.9, 1.1
    grid = FIT[k] * np.linspace(lo_frac, hi_frac, 601)
    if PARAM_NAMES[k] == "B":
        grid = np.linspace(0.0, 3.0 * FIT[k], 601)
    cs = []
    for v in grid:
        p[k] = v
        cs.append(cost(p))
    cs = np.asarray(cs)
    profiles[PARAM_NAMES[k]] = (grid.tolist(), cs.tolist())

    # single-parameter deviation to the literature value
    p = FIT.copy()
    p[k] = LIT[k]
    dC_1p = cost(p) - c_fit

    ok = grid[cs <= c_fit + dC_ref]
    if len(ok):
        iv = f"[{ok.min():.4g}, {ok.max():.4g}]"
        intervals[PARAM_NAMES[k]] = [float(ok.min()), float(ok.max())]
    else:
        iv = "(empty)"
        intervals[PARAM_NAMES[k]] = None
    print(f"    {PARAM_NAMES[k]:10s} {FIT[k]:12.5g} {LIT[k]:12.5g} "
          f"{100 * (LIT[k] - FIT[k]) / FIT[k]:+9.2f}% {dC_1p:11.4f} {iv:>34s}")

# ── Hessian at the fit: curvature and degeneracies ──────────────────────
print("\nHessian at the fitted minimum (free parameters, log-scaled)")
th = np.log(FIT[FREE])


def cost_log(t):
    p = FIT.copy()
    p[FREE] = np.exp(t)
    return cost(p)


nf = len(FREE)
H = np.zeros((nf, nf))
h = 1e-3
for i in range(nf):
    for j in range(i, nf):
        tpp, tpm, tmp, tmm = th.copy(), th.copy(), th.copy(), th.copy()
        tpp[i] += h; tpp[j] += h
        tpm[i] += h; tpm[j] -= h
        tmp[i] -= h; tmp[j] += h
        tmm[i] -= h; tmm[j] -= h
        H[i, j] = H[j, i] = (cost_log(tpp) - cost_log(tpm) - cost_log(tmp) + cost_log(tmm)) / (4 * h * h)

ev = np.linalg.eigvalsh(H)
print("  eigenvalues (d^2C / dlnp^2): " + "  ".join(f"{e:.3g}" for e in ev))
print(f"  condition number = {ev.max() / max(ev.min(), 1e-30):.3g}")
# fractional parameter move along each eigen-direction that costs dC_ref
evals, evecs = np.linalg.eigh(H)
print("  fractional parameter excursion costing dC = C_lit - C_fit, per eigen-direction:")
for i in range(nf):
    if evals[i] <= 0:
        print(f"    dir {i}: non-positive curvature")
        continue
    step = np.sqrt(2 * dC_ref / evals[i])
    dom = PARAM_NAMES[FREE[int(np.argmax(np.abs(evecs[:, i])))]]
    print(f"    dir {i} (dominated by {dom:9s}): {100 * step:8.2f} % in ln p")

out = {
    "n_z": n_z, "n_x_masked": n_mask, "n_points": n_pts,
    "cost": {"null": c_zero, "lit": c_lit, "fit": c_fit, "floor": c_floor},
    "cost_ratios": {
        "lit_over_fit": c_lit / c_fit,
        "floor_over_lit": c_floor / c_lit,
        "floor_over_fit": c_floor / c_fit,
        "reducible_fraction_of_lit": (c_lit - c_floor) / c_lit,
    },
    "rms_frac_residual": {"lit": rms_lit, "fit": rms_fit, "floor": rms_floor},
    "hessian_log_eigenvalues": ev.tolist(),
    "dC_ref": dC_ref,
    "intervals_at_dC_ref": intervals,
    "profiles": profiles,
}
(DATADIR / "visibility_flatness.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote {DATADIR / 'visibility_flatness.json'}")
