#!/usr/bin/env python3
r"""Landau-Zener check of the dark-photon NWA conversion rate γ_con (Part II §II.4).

The code's NWA (src/dark_photon.rs::gamma_con, Chluba & Cyr 2024 Eq. 6):

    γ_con(ε, m) = π ε² m² / (d · T_cmb_ev · H_ev)     evaluated at z_res,
    d = |d ln ω_pl² / d ln a|,   ω_pl(z_res) = m.

This is the resonant photon→dark-photon conversion probability for a photon of
energy ω, P(ω) = π ε² m² / (ω · d · H_ev), evaluated at the thermal energy
ω = T_cmb_ev(z_res). We verify it by directly integrating the 2-level γ–A′
mixing system through the resonance with the ACTUAL ω_pl(z) profile and
comparing the extracted conversion probability to the Landau-Zener result

    P_LZ(ω) = 1 − exp(−π ε² m² / (ω · d · H_ev)) = 1 − exp(−ε² γ_con(1,m))   at ω=T_cmb.

Mixing equations (relativistic WKB, common A′ phase removed; natural units, eV):

    i dA_γ /dt = [ (ω_pl²(t) − m²)/(2ω) ] A_γ + [ ε m²/(2ω) ] A_{A'}
    i dA_{A'}/dt = [ ε m²/(2ω) ] A_γ

with dt = −dz / (H_ev(z) (1+z)). Reference for the resonant LZ probability:
Mirizzi, Redondo & Sigl (2009), arXiv:0901.0014.

Run: python dev/scripts/gamma_con_landau_zener.py
Writes findings to dev/audit/gamma_con_lz_check.md.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from spectroxide import DEFAULT_COSMO  # noqa: E402
from spectroxide.dark_photon import (  # noqa: E402
    _cosmo_hubble,
    dln_omega_pl_sq_dlna,
    gamma_con,
    plasma_frequency_ev,
    resonance_redshift,
)

HBAR_EV_S = 6.582_119_569e-16  # eV·s
K_B_EV_PER_K = 8.617_333_262e-5  # eV/K


def h_ev(z, cosmo):
    """Hubble rate at z in eV (natural units). _cosmo_hubble returns SI [1/s]."""
    return HBAR_EV_S * _cosmo_hubble(z, cosmo)


def t_cmb_ev(z, cosmo):
    return K_B_EV_PER_K * cosmo["t_cmb"] * (1.0 + z)


def integrate_conversion(eps, m_ev, omega_ev, z_res, cosmo, n_widths=80.0):
    """Integrate the 2-level mixing ODE through resonance; return |A_{A'}|²."""
    # Resonance half-width in z: |ω_pl²−m²| ~ κ·2ω at the LZ turning region.
    # ω_pl² ∝ (1+z)^p locally, so a fractional Δz/(1+z) ~ (n_widths·ε) brackets it.
    dln = abs(dln_omega_pl_sq_dlna(z_res, cosmo))
    # width in z spanning n_widths off-diagonal couplings each side:
    w = n_widths * max(eps, 1e-12) / max(dln, 1e-6)
    w = min(max(w, 1e-6), 0.5)
    z_hi = z_res * (1.0 + w)
    z_lo = z_res * (1.0 - w)

    kappa = eps * m_ev * m_ev / (2.0 * omega_ev)  # off-diagonal [eV]

    def rhs(z, y):
        ar_g, ai_g, ar_p, ai_p = y
        opl = plasma_frequency_ev(z, cosmo)
        delta = (opl * opl - m_ev * m_ev) / (2.0 * omega_ev)  # [eV]
        # dt/dz = -1/(H_ev (1+z))
        dtdz = -1.0 / (h_ev(z, cosmo) * (1.0 + z))
        # dA/dt = -i (H A); multiply by dt/dz for dA/dz.
        # A_γ: -i(delta A_γ + kappa A_p)
        dg = -1j * (delta * (ar_g + 1j * ai_g) + kappa * (ar_p + 1j * ai_p))
        dp = -1j * (kappa * (ar_g + 1j * ai_g))
        dg *= dtdz
        dp *= dtdz
        return [dg.real, dg.imag, dp.real, dp.imag]

    # Post-resonance |A_{A'}|² is constant in magnitude but the endpoint can
    # carry residual ripple; sample the last 30% of the (fully post-resonance)
    # trajectory and average to read the asymptotic plateau (plan §II.4 point 1).
    z_tail = np.linspace(z_res * (1.0 - 0.3 * w), z_lo, 200)
    sol = solve_ivp(
        rhs,
        (z_hi, z_lo),
        [1.0, 0.0, 0.0, 0.0],
        rtol=1e-11,
        atol=1e-14,
        method="DOP853",
        dense_output=True,
        max_step=(z_hi - z_lo) / 20000.0,
    )
    ys = sol.sol(z_tail)
    p_tail = ys[2] ** 2 + ys[3] ** 2
    return float(np.mean(p_tail))


def main():
    cosmo = DEFAULT_COSMO
    m_ev = 1e-7  # resonance in the fully-ionized era (matches src test)
    z_res = resonance_redshift(m_ev, cosmo)
    assert z_res is not None, "no resonance"
    omega = t_cmb_ev(z_res, cosmo)  # thermal photon: ω = T_cmb(z_res)
    gc1, _ = gamma_con(1.0, m_ev, cosmo)  # γ_con at ε=1  == P_NWA(ω=T_cmb)/ε²

    lines = []

    def emit(s):
        print(s)
        lines.append(s)

    emit(f"# Landau-Zener check of γ_con (m = {m_ev:.1e} eV)")
    emit(f"z_res = {z_res:.5e},  ω = T_cmb(z_res) = {omega:.4e} eV")
    emit(f"γ_con(ε=1, m) [code] = P_NWA(ω=T_cmb, ε=1) = {gc1:.5e}")
    emit(f"|d ln ω_pl²/d ln a| at z_res = {abs(dln_omega_pl_sq_dlna(z_res, cosmo)):.4f}")
    emit("")
    emit("| regime | ε | P_NWA=ε²γ_con(1) | P_LZ=1−e^(−P_NWA) | P_numeric | rel err |")
    emit("|--------|---|------------------|-------------------|-----------|---------|")

    # ε chosen to land in the non-adiabatic, boundary, and adiabatic regimes.
    eps_ref = 1.0 / np.sqrt(gc1)  # ε at which P_NWA = 1
    cases = [
        ("non-adiabatic", 0.05 * eps_ref),
        ("boundary", 1.0 * eps_ref),
        ("adiabatic", 3.0 * eps_ref),
    ]
    worst = 0.0
    for label, eps in cases:
        p_nwa = eps * eps * gc1
        p_lz = 1.0 - np.exp(-p_nwa)
        p_num = integrate_conversion(eps, m_ev, omega, z_res, cosmo)
        rel = abs(p_num - p_lz) / max(p_lz, 1e-300)
        worst = max(worst, rel)
        emit(f"| {label} | {eps:.4e} | {p_nwa:.4e} | {p_lz:.4e} | {p_num:.4e} | {rel:.3e} |")

    emit("")
    verdict = "CONFIRMED" if worst < 0.05 else "DISCREPANCY"
    emit(f"Worst rel err (P_numeric vs P_LZ) = {worst:.3e} → NWA {verdict} (threshold 5%).")
    emit("")
    if worst < 0.05:
        emit(
            "The numerically-integrated conversion probability through the actual "
            "ω_pl(z) profile matches the Landau-Zener / NWA formula across the "
            "non-adiabatic, boundary, and adiabatic regimes. The code's γ_con "
            "(= P_NWA at ω=T_cmb) is validated against the underlying mixing "
            "dynamics. The ~22% discrepancy vs Bryce's frozen-absorption curve "
            "(memory: axion-dp-distortion) is therefore NOT in γ_con — it lives "
            "elsewhere (frozen-vs-thermalized treatment)."
        )
    else:
        emit(
            "The NWA does NOT reproduce the direct integration — γ_con needs "
            "revisiting before any regression test is pinned."
        )

    note = Path(__file__).resolve().parents[1] / "audit" / "gamma_con_lz_check.md"
    note.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {note}")
    return 0 if worst < 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
