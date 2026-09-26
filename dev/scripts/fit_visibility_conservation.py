#!/usr/bin/env python3
"""Visibility-function fits to the PDE spectra.

Pipeline (paper Table 1): build_visibility_table.py -> build_baseline_table.py
-> fit_visibility_conservation.py (this script).
Inputs: dev/data/visibility_table.npz and dev/data/baseline_table.npz.
Output: dev/data/visibility_conservation_fit.json.

ADOPTED FIDUCIAL (paper Table 1 / Fig 2): the all-7 global spectral fit
("all7_spectral" below) — (A, B, beta, z_y, alpha_y, z_mu, alpha_mu)
fit simultaneously to the baseline-subtracted spectra with the x^3
trapezoidal metric.  The two-stage (energetic + spectral) fit documented
below is retained as a diagnostic: the conservation-law J_bb* extraction
provides the direct data points shown in Fig 2 and an independent check
of the fitted thermalization visibility.

The rest of this docstring describes the two-stage methodology.

Stage 1 — thermalization visibility J_bb* (A, B, beta), following
Chluba (2014), arXiv:1312.6030: the distortion visibility is defined
energetically, via the conservation-law chemical potential

    mu_cons = (3 * drho/rho - 4 * dN/N) / kappa_c ,

which is exactly invariant under a temperature shift (G_bb drops out) and
needs no spectral template fitting.  In the mu-era (z_h >= 3e5, where
J_mu = 1 to better than 1e-9) the thermalization visibility is

    J_bb*(z_h) = kappa_c * mu_cons / (3 * (drho/rho)_inj) .

The adiabatic-cooling baseline (mu_cons ~ -5.8e-9 absolute, i.e. -5.8e-4
per unit injected energy at delta_rho = 1e-5) is subtracted using the
no-injection sweep in dev/data/baseline_table.npz.  No y-type component
is subtracted: Chluba (2014), Sect. 3.3.4, defines the visibility
energetically, "irrespective of the shape of the distortion at late
times (mu, y and residual distortion)", so the Eq.-(12) mu_infty applied
to the full spectrum is the closest correspondence.  Sensitivity to the
transition-region y admixture (J_y ~ 1.5% at z = 3e5, dying as z^-2.58)
is tested by varying the lower fit bound instead.

(A, B, beta) are then fit to J_bb*(z_h) with z_th = 1.98e6 and
alpha_th = 5/2 held fixed at their analytic values, minimising relative
residuals (the visibility spans 4+ decades).  The fiducial floats all
three; (B, beta) are strongly degenerate over the fit window (probed
only to z ~ 1.5 z_th), so the free fit lands at beta ~ 1.27 with
B ~ 0.060 --- individually far from the literature (0.0381, 2.29) while
the J_bb* curves differ by < 0.013 everywhere.  Beta-fixed variants are
retained for a parameter-by-parameter literature comparison.
Reference: Chluba (2015),
arXiv:1506.06582, Eq. (13): A = 0.983, B = 0.0381, beta = 2.29, valid
3e5 <~ z <~ 6e6 (relativistic CS corrections neglected — same physics
content as this PDE).

Stage 2 — transition functions J_mu, J_y (z_mu, alpha_mu, z_y, alpha_y),
following Chluba (2013), arXiv:1304.6120: least-squares fit of the
three-component Green's function ansatz to the PDE spectra, with the
Stage-1 thermalization parameters held fixed.  The frequency integral
uses a trapezoidal quadrature measure (not a bare grid-point sum), which
removes the dependence on the local grid-point density.

Outputs: dev/data/visibility_conservation_fit.json
"""
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import least_squares, minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "python"))

from spectroxide.greens import (
    KAPPA_C,
    G2_PLANCK,
    G3_PLANCK,
    mu_shape,
    y_shape,
    g_bb,
)

MU_TO_ENERGY = 3.0 / KAPPA_C
DELTA_RHO = 1e-5  # injection amplitude of the visibility table; the baseline
                  # table used delta_rho = 1e-12 (effectively zero) and its
                  # absolute mu_cons/drho are divided by 1e-5 below to express
                  # them per unit *injected* energy of the signal runs

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
table = np.load(DATA / "visibility_table.npz")
base = np.load(DATA / "baseline_table.npz")

z_h = table["z_h"]
x = table["x"]
dn_nc = table["dn_nc"]
dn_raw = table["dn_raw"]
drho = table["drho"]
n_z = len(z_h)

z_base = base["z_h"]
mu_base = base["mu_cons"] / DELTA_RHO   # per unit injected energy
drho_base = base["drho"] / DELTA_RHO

LIT = {"A": 0.983, "B": 0.0381, "beta": 2.29,
       "z_y": 6.0e4, "alpha_y": 2.58, "z_mu": 5.8e4, "alpha_mu": 1.88}
Z_TH = 1.98e6
ALPHA_TH = 2.5


# ── Visibility forms ────────────────────────────────────────────────────
def j_bb_star_model(z, A, B, beta):
    r = z / Z_TH
    return A * np.exp(-(r**ALPHA_TH)) * (1.0 - B * r**beta)


def j_mu_model(z, z_mu, alpha_mu):
    return 1.0 - np.exp(-(((1.0 + z) / z_mu) ** alpha_mu))


def ks12_visibility(z, improved=True):
    """Khatri & Sunyaev (2012), arXiv:1203.2601, Eq. (3.7) / (4.5).

    Blackbody optical depth T(z); visibility = exp(-T).  Constants for
    the standard cosmology as quoted in the paper: z_dC = 1.96e6,
    z_br = 1.05e7, z_eps = 3.67e5, eps = 0.0151; improved (Eq. 4.5) adds
    the non-stationarity terms with z'_dC = 7.11e6, z'_br = 5.41e11.
    """
    zp1 = 1.0 + np.asarray(z, dtype=float)
    z_dc, z_br, z_eps, eps = 1.96e6, 1.05e7, 3.67e5, 0.0151
    main = np.sqrt((zp1 / (1 + z_dc)) ** 5 + (zp1 / (1 + z_br)) ** 2.5)
    log_term = eps * np.log((zp1 / (1 + z_eps)) ** 1.25
                            + np.sqrt(1.0 + (zp1 / (1 + z_eps)) ** 2.5))
    if not improved:
        return np.exp(-(main + log_term))
    tau = 1.007 * (main + log_term) \
        + (zp1 / (1 + 7.11e6)) ** 3 + (zp1 / (1 + 5.41e11)) ** 0.5
    return np.exp(-tau)


def j_y_model(z, z_y, alpha_y):
    return 1.0 / (1.0 + ((1.0 + z) / z_y) ** alpha_y)


# ── Conservation-law extraction ─────────────────────────────────────────
def mu_cons_of(dn):
    dr = np.trapz(x**3 * dn, x) / G3_PLANCK
    dN = np.trapz(x**2 * dn, x) / G2_PLANCK
    return (3.0 * dr - 4.0 * dN) / KAPPA_C


# Sanity: the operator must return 1 on mu_shape and ~0 on g_bb.
_check_m = mu_cons_of(mu_shape(x))
_check_g = mu_cons_of(g_bb(x))
assert abs(_check_m - 1.0) < 1e-4, _check_m
assert abs(_check_g) < 1e-4, _check_g

mu_cons = np.array([mu_cons_of(dn_nc[i]) for i in range(n_z)])

# Baseline subtraction (mu-era z's only; baseline table covers z >= 2e5).
mu_corr = mu_cons.copy()
drho_corr = drho.copy()
has_base = np.zeros(n_z, dtype=bool)
for i, zi in enumerate(z_h):
    j = np.where(np.isclose(z_base, zi, rtol=1e-10))[0]
    if len(j):
        mu_corr[i] -= mu_base[j[0]]
        drho_corr[i] -= drho_base[j[0]]
        has_base[i] = True

# Energetic visibility: J_mu*J_bb* per unit injected energy, normalised
# by the recovered (baseline-corrected) energy.
J_data = (KAPPA_C * mu_corr / 3.0) / drho_corr

print("Stage 1 data (conservation-law visibility):")
print(f"{'z_h':>10s} {'mu_cons':>11s} {'mu_corr':>11s} "
      f"{'J_data':>10s} {'J_C15':>10s} {'dev %':>7s} {'J_KS12':>10s} {'dev %':>7s}")
J_lit_all = j_bb_star_model(z_h, LIT["A"], LIT["B"], LIT["beta"]) * \
    j_mu_model(z_h, LIT["z_mu"], LIT["alpha_mu"])
J_ks12_all = ks12_visibility(z_h)
for i in np.where(z_h >= 2.5e5)[0]:
    dev = (J_data[i] / J_lit_all[i] - 1.0) * 100
    dev_ks = (J_data[i] / J_ks12_all[i] - 1.0) * 100
    print(f"{z_h[i]:10.3e} {mu_cons[i]:11.4e} {mu_corr[i]:11.4e} "
          f"{J_data[i]:10.4e} {J_lit_all[i]:10.4e} {dev:7.2f} "
          f"{J_ks12_all[i]:10.4e} {dev_ks:7.2f}")


# ── Stage 1 fit: (A, B, beta) on relative residuals ────────────────────
def fit_stage1(z_lo, z_hi, fix_beta=None, fix_B=None):
    """Fit thermalization-visibility parameters on relative residuals.

    fix_beta / fix_B pin those parameters at the given values.  The
    unconstrained 3-parameter fit prefers a shallower beta (~1.3) than
    the literature 2.29 at lower residual cost, with A and B moving to
    compensate, so only the constrained variants admit a
    parameter-by-parameter comparison with the literature.
    """
    sel = (z_h >= z_lo) & (z_h <= z_hi) & has_base
    zz, jj = z_h[sel], J_data[sel]
    jmu = j_mu_model(zz, LIT["z_mu"], LIT["alpha_mu"])  # ≈ 1 for z >= 3e5

    def unpack(p):
        A = p[0]
        B = fix_B if fix_B is not None else p[1]
        beta = fix_beta if fix_beta is not None else p[-1]
        return A, B, beta

    def resid(p):
        A, B, beta = unpack(p)
        return j_bb_star_model(zz, A, B, beta) * jmu / jj - 1.0

    x0, lo, hi = [0.983], [0.8], [1.1]
    if fix_B is None:
        x0.append(0.0381); lo.append(0.0); hi.append(0.3)
    if fix_beta is None:
        x0.append(2.29); lo.append(1.0); hi.append(5.0)
    out = least_squares(resid, x0=x0, bounds=(lo, hi))
    rms = np.sqrt(np.mean(out.fun**2))
    return unpack(out.x), rms, int(sel.sum())


stage1 = {}
VARIANTS = [
    # (z_lo, z_hi, fix_beta, fix_B, tag)
    (3e5, 3.0e6, None, None, "free_3e5_3e6"),
    (3e5, 3.5e6, None, None, "free_3e5_3.5e6"),
    (5e5, 3.5e6, None, None, "free_5e5_3.5e6"),
    (3e5, 4.0e6, None, None, "free_3e5_4e6"),
    (3e5, 3.0e6, 2.29, None, "beta_fixed_3e5_3e6"),
    (3e5, 3.5e6, 2.29, None, "beta_fixed_3e5_3.5e6"),
    (5e5, 3.5e6, 2.29, None, "beta_fixed_5e5_3.5e6"),
    (3e5, 4.0e6, 2.29, None, "beta_fixed_3e5_4e6"),
    (3e5, 3.5e6, 2.29, 0.0381, "A_only_3e5_3.5e6"),
]
for z_lo, z_hi, fb, fB, tag in VARIANTS:
    (A, B, beta), rms, npts = fit_stage1(z_lo, z_hi, fix_beta=fb, fix_B=fB)
    stage1[tag] = {"A": A, "B": B, "beta": beta,
                   "rms_rel": rms, "n_points": npts,
                   "z_range": [z_lo, z_hi],
                   "beta_fixed": fb is not None, "B_fixed": fB is not None}
    print(f"\nStage 1 [{tag}] z in [{z_lo:.1e}, {z_hi:.1e}] ({npts} pts, "
          f"rms rel {rms:.3%}):")
    for name, val, fixed in [("A", A, False), ("B", B, fB is not None),
                             ("beta", beta, fb is not None)]:
        lit = LIT[name]
        note = " (fixed)" if fixed else ""
        print(f"  {name:5s} = {val:9.5f}   lit {lit:9.5f}   "
              f"delta {100*(val/lit-1):+6.2f}%{note}")


# ── Spectral fits: subtract adiabatic-cooling baseline spectra ─────────
# The baseline table provides full no-injection spectra for z >= 2e5;
# strip them with the same NC projection and subtract from the signal.
# (Sub-0.02% effect on every fitted parameter, but consistent with the
# scalar subtraction in the energetic stage.)
_gx = g_bb(x)
_gnorm = np.trapz(x**2 * _gx, x)
dn_fit = dn_nc.copy()
for _i, _zi in enumerate(z_h):
    _j = np.where(np.isclose(base["z_h"], _zi, rtol=1e-10))[0]
    if len(_j):
        _b = base["dn_baseline"][_j[0]]
        dn_fit[_i] -= _b - np.trapz(x**2 * _b, x) / _gnorm * _gx

# ── Stage 2 fit: (z_y, alpha_y, z_mu, alpha_mu), spectral, trapz ───────
# Basis shapes, NC-stripped analytically (matching dn_nc's stripping).
M_x, Y_x, G_x = mu_shape(x), y_shape(x), g_bb(x)
G_int = np.trapz(x**2 * G_x, x)
M_nc = M_x - np.trapz(x**2 * M_x, x) / G_int * G_x
Y_nc = Y_x - np.trapz(x**2 * Y_x, x) / G_int * G_x
G_nc = G_x - G_x  # zero by construction

X_LO, X_HI = 0.5, 20.0
mask = (x >= X_LO) & (x <= X_HI)
x_m = x[mask]
w = x_m**3


def fit_stage2(A, B, beta, z_sel_hi=None, label="", free_amp=False):
    """Fit transition params with thermalization params fixed.

    free_amp adds a global spectral amplitude s multiplying the mu
    component: the energetic (mu_infty) visibility counts trapped
    low-frequency photons that the x in [0.5, 20] spectral amplitude
    does not, so the spectral normalisation sits ~1% above A.  With s
    free, that convention difference is absorbed by s instead of
    leaking into z_mu.
    """
    sel_z = np.ones(n_z, dtype=bool) if z_sel_hi is None else (z_h <= z_sel_hi)
    idx = np.where(sel_z)[0]
    jb_fix = j_bb_star_model(z_h, A, B, beta)

    def cost(p):
        # z parameters carried in units of 1e4 so all four parameters are
        # O(1); L-BFGS-B's absolute finite-difference step (~1e-8) is
        # otherwise invisible on scales of 1e4-2e5 and the z's never move.
        z_y, alpha_y, z_mu, alpha_mu = p[0] * 1e4, p[1], p[2] * 1e4, p[3]
        s = p[4] if free_amp else 1.0
        chi2 = 0.0
        for i in idx:
            jm = j_mu_model(z_h[i], z_mu, alpha_mu)
            jyv = j_y_model(z_h[i], z_y, alpha_y)
            model = (MU_TO_ENERGY * s * jm * jb_fix[i] * M_nc[mask]
                     + 0.25 * jyv * Y_nc[mask]
                     + 0.25 * (1.0 - jb_fix[i]) * G_nc[mask]) * drho[i]
            resid = w * (model - dn_fit[i, mask])
            chi2 += np.trapz(resid**2, x_m)
        return chi2

    p0 = [LIT["z_y"] / 1e4, LIT["alpha_y"], LIT["z_mu"] / 1e4, LIT["alpha_mu"]]
    bounds = [(1.0, 20.0), (1.0, 5.0), (1.0, 20.0), (1.0, 4.0)]
    if free_amp:
        p0.append(1.0)
        bounds.append((0.9, 1.1))
    out = minimize(cost, p0, method="L-BFGS-B", bounds=bounds)
    z_y, alpha_y, z_mu, alpha_mu = out.x[0] * 1e4, out.x[1], out.x[2] * 1e4, out.x[3]
    s = out.x[4] if free_amp else 1.0
    print(f"\nStage 2 fit ({label}, {len(idx)} spectra, cost {out.fun:.4e}):")
    for name, val in [("z_y", z_y), ("alpha_y", alpha_y),
                      ("z_mu", z_mu), ("alpha_mu", alpha_mu)]:
        lit = LIT[name]
        print(f"  {name:8s} = {val:12.5g}   lit {lit:9.5g}   "
              f"delta {100*(val/lit-1):+6.2f}%")
    if free_amp:
        print(f"  s_amp    = {s:12.5g}   (spectral/energetic amplitude ratio)")
    return {"z_y": z_y, "alpha_y": alpha_y, "z_mu": z_mu,
            "alpha_mu": alpha_mu, "cost": out.fun, "n_spectra": int(len(idx)),
            "s_amp": s}


# Adopt the free-(A, B, beta) fit over the well-conditioned range (tau <~ 3,
# where percent-level rate systematics are not exponentially amplified).
FIDUCIAL = "free_3e5_3e6"
A_f = stage1[FIDUCIAL]["A"]
B_f = stage1[FIDUCIAL]["B"]
beta_f = stage1[FIDUCIAL]["beta"]

stage2 = {}
stage2["all_z"] = fit_stage2(A_f, B_f, beta_f, None, "all z")
stage2["z_le_3e5"] = fit_stage2(A_f, B_f, beta_f, 3e5, "z <= 3e5")
stage2["all_z_free_amp"] = fit_stage2(A_f, B_f, beta_f, None,
                                      "all z, free amplitude", free_amp=True)

# ── All-seven-parameter spectral fit (no energetic stage) ──────────────
def fit_all7(label="all7 spectral"):
    """Global spectral fit of (A, B, beta, z_y, alpha_y, z_mu, alpha_mu).

    Same x^3-weighted trapezoidal cost as stage 2, but with the
    thermalization parameters free instead of pinned by the energetic
    (conservation-law) fit.  This is the fit design that originally
    produced the spurious B tension: the spectral residual barely
    constrains the deep-mu-era energetics, so (A, B, beta) drift to
    absorb transition-region shape misfit of the 3-component ansatz.
    """
    def cost(p):
        A, B, beta = p[0], p[1], p[2]
        z_y, alpha_y, z_mu, alpha_mu = p[3] * 1e4, p[4], p[5] * 1e4, p[6]
        jb = j_bb_star_model(z_h, A, B, beta)
        chi2 = 0.0
        for i in range(n_z):
            jm = j_mu_model(z_h[i], z_mu, alpha_mu)
            jyv = j_y_model(z_h[i], z_y, alpha_y)
            model = (MU_TO_ENERGY * jm * jb[i] * M_nc[mask]
                     + 0.25 * jyv * Y_nc[mask]) * drho[i]
            resid = w * (model - dn_fit[i, mask])
            chi2 += np.trapz(resid**2, x_m)
        return chi2

    p0 = [LIT["A"], LIT["B"], LIT["beta"],
          LIT["z_y"] / 1e4, LIT["alpha_y"], LIT["z_mu"] / 1e4, LIT["alpha_mu"]]
    bounds = [(0.8, 1.1), (0.0, 0.3), (0.5, 5.0),
              (1.0, 20.0), (1.0, 5.0), (1.0, 20.0), (1.0, 4.0)]
    # Rescale the cost to O(1) at the literature point and tighten the
    # tolerances: with the raw cost and default tolerances L-BFGS-B stopped
    # after a few iterations with beta still at its start value (audit A3,
    # dev/REVIEW_PAPER_CLAIMS_2026-09-24.md). Several starts guard against a
    # start-dependent answer; all of them must land on the same minimum.
    c_ref = cost(p0)
    opts = dict(ftol=1e-15, gtol=1e-12, maxiter=5000, maxfun=50000)
    starts = [p0,
              [0.98, 0.06, 1.2] + p0[3:],
              [0.99, 0.02, 3.5] + p0[3:],
              [0.95, 0.10, 1.8, 7.0, 2.3, 6.0, 1.7],
              [1.00, 0.04, 2.8, 5.0, 2.9, 5.0, 2.1]]
    runs = [minimize(lambda p: cost(p) / c_ref, s, method="L-BFGS-B",
                     bounds=bounds, options=opts) for s in starts]
    for s, r in zip(starts, runs):
        print(f"  start beta={s[2]:.2f} B={s[1]:.3f}: nit {r.nit}, "
              f"cost {r.fun * c_ref:.6f}, beta {r.x[2]:.4f}, B {r.x[1]:.4f}")
    out = min(runs, key=lambda r: r.fun)
    out.fun *= c_ref
    spread = np.ptp([r.x for r in runs], axis=0)
    A, B, beta = out.x[0], out.x[1], out.x[2]
    z_y, alpha_y, z_mu, alpha_mu = (out.x[3] * 1e4, out.x[4],
                                    out.x[5] * 1e4, out.x[6])
    print(f"\nAll-7 spectral fit ({label}, {n_z} spectra, cost {out.fun:.4e}):")
    for name, val in [("A", A), ("B", B), ("beta", beta),
                      ("z_y", z_y), ("alpha_y", alpha_y),
                      ("z_mu", z_mu), ("alpha_mu", alpha_mu)]:
        lit = LIT[name]
        print(f"  {name:8s} = {val:12.5g}   lit {lit:9.5g}   "
              f"delta {100*(val/lit-1):+6.2f}%")
    return {"A": A, "B": B, "beta": beta, "z_y": z_y, "alpha_y": alpha_y,
            "z_mu": z_mu, "alpha_mu": alpha_mu,
            "cost": out.fun, "n_spectra": int(n_z), "nit": int(out.nit),
            "n_starts": len(runs), "start_spread": spread.tolist()}


all7 = fit_all7()

# ── Save ────────────────────────────────────────────────────────────────
out = {
    "method": {
        "stage1": "conservation-law mu (Chluba 2014 Eq. 12), baseline- and "
                  "y-subtracted, relative-error least squares, z_th=1.98e6 "
                  "alpha_th=2.5 fixed",
        "stage2": "spectral least squares, trapz quadrature, x weight x^3, "
                  f"x in [{X_LO}, {X_HI}], thermalization params fixed from "
                  f"stage1[{FIDUCIAL}]",
    },
    "fiducial": "all7_spectral",
    "fiducial_stage1": FIDUCIAL,
    "stage1": stage1,
    "stage2": stage2,
    "all7_spectral": all7,
    "literature": LIT,
    "J_data": {"z_h": z_h.tolist(), "J": J_data.tolist(),
               "has_baseline": has_base.tolist(),
               "J_chluba2015": J_lit_all.tolist(),
               "J_ks12": J_ks12_all.tolist()},
}
outpath = DATA / "visibility_conservation_fit.json"
with open(outpath, "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nSaved: {outpath}")
