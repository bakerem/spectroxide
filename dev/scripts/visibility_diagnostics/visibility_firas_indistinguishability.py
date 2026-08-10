#!/usr/bin/env python3
"""Is the Fig. 2 visibility-parameter difference observable?

The spectral cost minimised in dev/scripts/fit_visibility_from_table.py has no
noise model, so its absolute value carries no statistical meaning. This script
converts the question into one with real error bars: given the *same* energy
release, how different are the spectra predicted by the Chluba (2013, 2015)
visibility parameters and by our PDE fit, measured against the FIRAS covariance?

For each injection redshift we build the three-component Green's-function
spectrum with both parameter sets at a fixed Delta rho / rho, take the
difference, and compute

    S = sqrt( d^T Cinv d )                     (raw)
    S = sqrt( d^T Cinv d - (d^T Cinv T)(T^T Cinv T)^-1 (T^T Cinv d) )

where the nuisance templates T are the blackbody temperature shift (FIRAS
cannot measure the absolute CMB temperature, so any G_bb component is
unobservable by construction) and the galactic dust template. Because the
spectra are linear in Delta rho / rho, S scales linearly with it, so we can
report the energy release at which FIRAS would distinguish the two
parameterisations at 1 sigma, and compare that with the FIRAS limit itself.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "python"))

from spectroxide.firas import FIRASData, _dn_to_dI_kJy
from spectroxide.greens import mu_shape, y_shape, g_bb, KAPPA_C

MU_TO_ENERGY = 3.0 / KAPPA_C

LIT = dict(z_y=6.0e4, a_y=2.58, z_mu=5.8e4, a_mu=1.88,
           A=0.983, z_th=1.98e6, a_th=2.5, B=0.0381, beta=2.29)
# Grid-independent refit (trapezoidal dx quadrature; see
# dev/scripts/visibility_diagnostics/visibility_refit_quadrature_table1.py). The published Table 1
# values came from a cost summed over frequency grid *nodes*, which is not a
# functional of the spectra: its effective weight is x^3 times the local node
# density. Correcting it moves z_y from +5.2% to +0.25% of the literature value.
FIT = dict(z_y=60150.403457354725, a_y=2.626476759129253,
           z_mu=56425.03815010834, a_mu=1.9305751269413123,
           A=0.9869858437036658, z_th=1.98e6, a_th=2.5,
           B=0.04416077341968499, beta=2.4153027387017425)

# The published (node-sum) fit, kept so the shift attributable to the metric
# defect can be quoted alongside the corrected one.
FIT_NODESUM = dict(z_y=63128.02326056765, a_y=2.653389404163302,
                   z_mu=58428.69672709, a_mu=1.9465842847446648,
                   A=0.9918167997907094, z_th=1.98e6, a_th=2.5,
                   B=0.048279535221419465, beta=2.0719380110207646)


def visibilities(z, p):
    jb = max(p["A"] * np.exp(-((z / p["z_th"]) ** p["a_th"]))
             * (1.0 - p["B"] * (z / p["z_th"]) ** p["beta"]), 0.0)
    jm = 1.0 - np.exp(-(((1.0 + z) / p["z_mu"]) ** p["a_mu"]))
    jy = 1.0 / (1.0 + ((1.0 + z) / p["z_y"]) ** p["a_y"])
    return jm, jb, jy


def gf_dn(x, z, p, drho):
    """Three-component Green's-function Delta n for energy release drho."""
    jm, jb, jy = visibilities(z, p)
    return drho * (MU_TO_ENERGY * jm * jb * mu_shape(x)
                   + 0.25 * jy * y_shape(x)
                   + 0.25 * (1.0 - jb) * g_bb(x))


firas = FIRASData()
x_f = firas.x
Cinv = firas.cov_inv

# Nuisance templates: unobservable temperature shift, and galactic dust.
T_dT = firas.gbb_template_kJy()
T_gal = firas.galactic_template_kJy()


def significance(d_kJy, marginalise):
    s2 = float(d_kJy @ Cinv @ d_kJy)
    if marginalise:
        T = np.column_stack(marginalise)
        A = T.T @ Cinv @ T
        b = T.T @ Cinv @ d_kJy
        s2 -= float(b @ np.linalg.solve(A, b))
    return np.sqrt(max(s2, 0.0))


DRHO = 1e-5          # reference energy release; results scale linearly
z_grid = np.array([3e3, 1e4, 3e4, 5e4, 8e4, 1.5e5, 3e5, 1e6, 2e6, 3e6, 5e6])

print(f"FIRAS 95% limits for context: |mu| < {firas.upper_limit_mu():.3g}   "
      f"|y| < {firas.upper_limit_y():.3g}")
print("Per-redshift Delta rho/rho limits below use the Green's function itself as")
print("the template, marginalised over the temperature shift and galactic dust.")
print(f"Reference energy release for S_raw / S_marg columns: {DRHO:.0e}\n")

def firas_limit_drho(z, p):
    """Self-consistent FIRAS 95% two-sided limit on Delta rho/rho at this z_h,
    using the Green's function itself as the signal template and marginalising
    over the temperature shift and dust."""
    tmpl = _dn_to_dI_kJy(x_f, gf_dn(x_f, z, p, 1.0), firas.t_cmb)
    r = firas.fit_amplitude_marginalised(tmpl, [T_dT, T_gal])
    return abs(r["amplitude"]) + 1.959963985 * r["sigma"]


# Above this Delta rho/rho the "limit" is not a constraint at all; such z_h are
# excluded from the worst-case statement (at z_h = 5e6 the GF is ~zero, so the
# fitted limit runs to order unity and the reported sigma is a ratio of two
# vanishing numbers).
LIM_MEANINGFUL = 1e-3

hdr = (f"{'z_h':>9s} {'d(mu) %':>8s} {'S_marg':>9s} {'S_marg+amp':>11s} "
       f"{'FIRAS lim':>10s} {'S at lim':>9s} {'dchi2':>10s} {'lim shift %':>12s}")
print(hdr)
print("-" * len(hdr))

rows = []
for z in z_grid:
    dn_l = gf_dn(x_f, z, LIT, DRHO)
    dn_f = gf_dn(x_f, z, FIT, DRHO)
    d_kJy = _dn_to_dI_kJy(x_f, dn_f - dn_l, firas.t_cmb)
    tmpl_lit = _dn_to_dI_kJy(x_f, gf_dn(x_f, z, LIT, 1.0), firas.t_cmb)

    s_marg = significance(d_kJy, [T_dT, T_gal])
    # Also profile out the overall amplitude of the distortion itself: Delta
    # rho/rho is unknown a priori, so any part of the LIT-vs-FIT difference that
    # is a pure rescaling of the template is unobservable too.
    s_amp = significance(d_kJy, [T_dT, T_gal, tmpl_lit])

    jm_l, jb_l, _ = visibilities(z, LIT)
    jm_f, jb_f, _ = visibilities(z, FIT)
    mu_l, mu_f = 1.401 * DRHO * jm_l * jb_l, 1.401 * DRHO * jm_f * jb_f

    lim_l = firas_limit_drho(z, LIT)
    lim_f = firas_limit_drho(z, FIT)
    s_at_lim = s_marg * lim_l / DRHO
    shift = 100 * (lim_f - lim_l) / lim_l

    rows.append((z, s_marg, s_amp, lim_l, s_at_lim, shift))
    flag = "" if lim_l <= LIM_MEANINGFUL else "  (no constraint)"
    print(f"{z:9.3g} {100 * (mu_f - mu_l) / mu_l if mu_l else float('nan'):+8.2f} "
          f"{s_marg:9.3e} {s_amp:11.3e} {lim_l:10.3e} {s_at_lim:9.4f} "
          f"{s_at_lim ** 2:10.6f} {shift:+11.2f}%{flag}")

good = [r for r in rows if r[3] <= LIM_MEANINGFUL]
worst = max(good, key=lambda r: r[4])
print(f"\nOver the range where FIRAS actually constrains the energy release "
      f"(Delta rho/rho limit <= {LIM_MEANINGFUL:g}):")
print(f"  worst case z_h = {worst[0]:.3g}, FIRAS 95% limit {worst[3]:.3e}")
print(f"    difference = {worst[4]:.4f} sigma  ->  Delta chi2 = {worst[4] ** 2:.6f}")
print(f"  1-sigma difference would need Delta rho/rho = {DRHO / worst[1]:.3e} "
      f"({DRHO / worst[1] / worst[3]:.1f}x the FIRAS limit)")

# The number a referee actually wants: does the choice of visibility parameters
# move the *published constraint*?
wshift = max(good, key=lambda r: abs(r[5]))
print(f"\nShift in the derived FIRAS 95% limit on Delta rho/rho between the two")
print(f"parameterisations (this is what propagates into a published constraint):")
print(f"  largest over the constrained range: {wshift[5]:+.2f}% at z_h = {wshift[0]:.3g}")
for z, _, _, lim_l, _, shift in good:
    print(f"    z_h = {z:9.3g}   limit {lim_l:.3e}   shift {shift:+6.2f}%")

# Dense scan for the true worst-case limit shift, so no between-node max is missed.
z_dense = np.logspace(np.log10(3e3), np.log10(2e6), 120)
shifts = []
for z in z_dense:
    ll = firas_limit_drho(z, LIT)
    if ll > LIM_MEANINGFUL:
        continue
    shifts.append((z, ll, 100 * (firas_limit_drho(z, FIT) - ll) / ll))
zb, lb, sb = max(shifts, key=lambda t: abs(t[2]))
print(f"\n  dense scan (120 pts, z_h <= 2e6): worst limit shift {sb:+.2f}% "
      f"at z_h = {zb:.4g} (limit {lb:.3e})")
