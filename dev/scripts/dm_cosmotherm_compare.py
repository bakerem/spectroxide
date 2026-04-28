#!/usr/bin/env python3
"""Compare PDE and Green's function predictions for DM scenarios against CosmoTherm GF.

Computes spectral distortions for decaying particle and DM annihilation using:
1. CosmoTherm GF database (convolution with heating rate)
2. Our analytic GF (greens.py)
3. Our PDE solver (reads from sweep_output.json)

IMPORTANT: Uses proper cosmology with radiation in Hubble, matching the Rust code exactly.

NOTE on CosmoTherm GF normalization:
    The Greens_data.dat stores the TOTAL PDE Green's function (including G_bb temperature
    shift). CosmoTherm's C++ code (Greens.cpp line 847) applies exp(-(z/2e6)^{5/2})
    when reading to extract the primordial-only (non-G_bb) distortion for convolution.
    We use the RAW database values here because we compare total spectra.

Usage:
    python scripts/dm_cosmotherm_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from spectroxide import cosmotherm, greens

# Physical constants (SI) - matching Rust CODATA 2018
K_BOLTZMANN = 1.380_649e-23
HPLANCK = 6.626_070_15e-34
HBAR = HPLANCK / (2 * np.pi)
C_LIGHT = 2.997_924_58e8
EV_IN_JOULES = 1.602_176_634e-19
T_CMB = 2.726
M_PROTON = 1.672_621_923_69e-27
MPC_IN_METERS = 3.085_677_581e22
G_NEWTON = 6.674_30e-11
Y_P = 0.24
N_EFF = 3.046

# Cosmology matching Rust Default (Chluba 2013 params)
OMEGA_B = 0.044
OMEGA_M = 0.26  # Omega_b + Omega_cdm = 0.044 + 0.216
h_hubble = 0.71
H0 = h_hubble * 100 * 1e3 / MPC_IN_METERS

# Radiation density (needed in Hubble at high z)
rho_gamma_0 = np.pi**2 / 15.0 * K_BOLTZMANN**4 * T_CMB**4 / (HBAR**3 * C_LIGHT**3)
rho_crit_0 = 3 * H0**2 * C_LIGHT**2 / (8 * np.pi * G_NEWTON)
OMEGA_GAMMA = rho_gamma_0 / rho_crit_0
OMEGA_REL = OMEGA_GAMMA * (1 + N_EFF * (7 / 8) * (4 / 11) ** (4 / 3))
OMEGA_LAMBDA = 1.0 - OMEGA_M - OMEGA_REL


def n_h_0():
    rho_b_0 = 3 * H0**2 / (8 * np.pi * G_NEWTON) * OMEGA_B
    return rho_b_0 * (1 - Y_P) / M_PROTON


def rho_gamma(z):
    kt = K_BOLTZMANN * T_CMB * (1 + z)
    return np.pi**2 / 15.0 * kt**4 / (HBAR**3 * C_LIGHT**3)


def n_h(z):
    return n_h_0() * (1 + z) ** 3


def hubble(z):
    """Hubble rate including radiation - matches Rust cosmology.rs exactly."""
    opz = 1 + z
    return H0 * np.sqrt(OMEGA_M * opz**3 + OMEGA_REL * opz**4 + OMEGA_LAMBDA)


def cosmic_time(z):
    integrand = lambda zp: 1.0 / (hubble(zp) * (1 + zp))
    result, _ = quad(integrand, z, 1e9, limit=500, epsabs=1e-30, epsrel=1e-10)
    return result


def cosmotherm_gf_convolve(z_h_db, x_db, g_th_dn, drho_per_dz_positive,
                            z_min=1e3, z_max=3e6):
    """Convolve CosmoTherm GF with a POSITIVE heating rate per dz.

    Parameters
    ----------
    drho_per_dz_positive : callable
        Positive d(Drho/rho)/dz: amount of energy deposited per unit dz.
    """
    mask = (z_h_db >= z_min) & (z_h_db <= z_max)
    z_pts = z_h_db[mask]
    if len(z_pts) < 2:
        return np.zeros(len(x_db))
    delta_n = np.zeros(len(x_db))
    dq_vals = np.array([drho_per_dz_positive(z) for z in z_pts])
    for ix in range(len(x_db)):
        g_at_x = g_th_dn[ix, mask]
        delta_n[ix] = np.trapz(g_at_x * dq_vals, z_pts)
    return delta_n


def decompose_dn(x, delta_n):
    """Decompose Delta_n into mu, y, DT/T (simple least-squares)."""
    ex = np.exp(np.clip(x, 0, 500))
    n_pl = 1.0 / (ex - 1.0)
    g_bb = x * ex / (ex - 1) ** 2
    y_sz = g_bb * (x * (ex + 1) / (ex - 1) - 4)
    m_x = n_pl * (n_pl + 1) * (x / 2.19229 - 1)
    mask = (x >= 0.5) & (x <= 15.0)
    if mask.sum() < 10:
        return 0.0, 0.0, 0.0
    A = np.column_stack([m_x[mask], y_sz[mask], g_bb[mask]])
    b = delta_n[mask]
    result = np.linalg.lstsq(A, b, rcond=None)
    coeffs = result[0]
    return coeffs[0] * 1.401, coeffs[1] * 0.25, coeffs[2]


def main():
    # Load CosmoTherm GF database
    z_h_db, x_db, g_th_raw = cosmotherm.load_greens_database()
    g_th_dn = np.zeros_like(g_th_raw)
    for iz in range(g_th_raw.shape[1]):
        g_th_dn[:, iz] = cosmotherm.cosmotherm_gf_to_delta_n(x_db, g_th_raw[:, iz])

    print(f"CosmoTherm GF: {len(z_h_db)} redshifts, {len(x_db)} frequencies")
    print(f"  z range: [{z_h_db.min():.0f}, {z_h_db.max():.0e}]")
    print()

    # ================================================================
    # Scenario 1: Decaying Particle
    # ================================================================
    f_x_ev = 1e6  # eV
    gamma_x = 1e-10  # 1/s

    print(f"SCENARIO 1: Decaying Particle (f_X={f_x_ev:.0e} eV, Gamma={gamma_x:.0e} 1/s)")
    print(f"  Lifetime = {1 / gamma_x:.0e} s")

    def decay_drho_per_dz(z):
        t = cosmic_time(z)
        rate_per_t = (
            f_x_ev * EV_IN_JOULES * gamma_x * n_h(z) * np.exp(-gamma_x * t)
            / rho_gamma(z)
        )
        return rate_per_t / (hubble(z) * (1 + z))

    total_drho, _ = quad(decay_drho_per_dz, 500, 3e6, limit=500)
    print(f"  Total Drho/rho injected = {total_drho:.6e}")

    # CosmoTherm GF convolution
    dn_ct = cosmotherm_gf_convolve(z_h_db, x_db, g_th_dn, decay_drho_per_dz,
                                    z_min=1e3, z_max=2.5e6)
    mu_ct, y_ct, dt_ct = decompose_dn(x_db, dn_ct)
    drho_ct = np.trapz(x_db**3 * dn_ct, x_db) / greens.G3_PLANCK

    # Analytic GF
    mu_gf = greens.mu_from_heating(decay_drho_per_dz, 1e3, 2.5e6, 5000)
    y_gf = greens.y_from_heating(decay_drho_per_dz, 1e3, 2.5e6, 5000)

    print(f"\n  {'Method':<12} {'mu':>12} {'y':>12} {'drho/rho':>12}")
    print(f"  {'-' * 50}")
    print(f"  {'CT GF':<12} {mu_ct:>12.4e} {y_ct:>12.4e} {drho_ct:>12.4e}")
    print(f"  {'Analytic GF':<12} {mu_gf:>12.4e} {y_gf:>12.4e}")

    # Try to load PDE spectrum
    sweep_path = Path(__file__).resolve().parent.parent / "sweep_output.json"
    if sweep_path.exists():
        with open(sweep_path) as f:
            pde = json.load(f)
        if pde.get("injection") == "decaying-particle":
            r = pde["results"][0]
            print(f"  {'PDE':<12} {r['pde_mu']:>12.4e} {r['pde_y']:>12.4e} {r['drho']:>12.4e}")

            # Spectral comparison
            pde_x = np.array(r["x"])
            pde_dn = np.array(r["delta_n"])
            pde_interp = interp1d(pde_x, pde_dn, kind='linear',
                                   fill_value=0, bounds_error=False)
            print(f"\n  Spectral comparison (PDE vs CT GF):")
            print(f"  {'nu GHz':>8} {'x':>8} {'PDE':>12} {'CT GF':>12} {'% diff':>8}")
            print(f"  {'-' * 52}")
            for nu_ghz in [30, 60, 100, 150, 200, 300, 400, 600, 857]:
                x = HPLANCK * nu_ghz * 1e9 / (K_BOLTZMANN * T_CMB)
                ix = np.argmin(np.abs(x_db - x))
                xc = x_db[ix]
                d_ct = dn_ct[ix]
                d_pde = pde_interp(xc)
                pct = ((d_pde - d_ct) / abs(d_ct) * 100) if abs(d_ct) > 1e-30 else 0
                print(f"  {nu_ghz:>8} {xc:>8.3f} {d_pde:>12.4e} {d_ct:>12.4e} {pct:>7.1f}%")

    # ================================================================
    # Scenario 2: DM Annihilation (s-wave)
    # ================================================================
    f_ann_ev = 1e-24  # eV*m^3/s

    print(f"\n{'=' * 70}")
    print(f"SCENARIO 2: DM Annihilation s-wave (f_ann={f_ann_ev:.0e} eV*m^3/s)")

    def ann_drho_per_dz(z):
        rate_per_t = f_ann_ev * EV_IN_JOULES * n_h(z) ** 2 / rho_gamma(z)
        return rate_per_t / (hubble(z) * (1 + z))

    total_drho2, _ = quad(ann_drho_per_dz, 500, 3e6, limit=500)
    print(f"  Total Drho/rho injected = {total_drho2:.6e}")

    dn_ct2 = cosmotherm_gf_convolve(z_h_db, x_db, g_th_dn, ann_drho_per_dz,
                                      z_min=1e3, z_max=2.5e6)
    mu_ct2, y_ct2, dt_ct2 = decompose_dn(x_db, dn_ct2)
    drho_ct2 = np.trapz(x_db**3 * dn_ct2, x_db) / greens.G3_PLANCK

    mu_gf2 = greens.mu_from_heating(ann_drho_per_dz, 1e3, 2.5e6, 5000)
    y_gf2 = greens.y_from_heating(ann_drho_per_dz, 1e3, 2.5e6, 5000)

    print(f"\n  {'Method':<12} {'mu':>12} {'y':>12} {'drho/rho':>12}")
    print(f"  {'-' * 50}")
    print(f"  {'CT GF':<12} {mu_ct2:>12.4e} {y_ct2:>12.4e} {drho_ct2:>12.4e}")
    print(f"  {'Analytic GF':<12} {mu_gf2:>12.4e} {y_gf2:>12.4e}")


if __name__ == "__main__":
    main()
