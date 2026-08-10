#!/usr/bin/env python3
"""Per-injection-redshift comparison: spectroxide GF table vs CosmoTherm database.

Isolates the Green's functions themselves from any scenario or convolution
bookkeeping.  Both sides are number-conserving-stripped and compared as spectra
over the plotted window; the CosmoTherm entries get the
``exp(-(z/2e6)^{5/2})`` factor its own convolution applies analytically.

The fitted mu/y of individual entries are printed but **must not be read as a
physics comparison** through the transition era: the three-shape least-squares is
degenerate there (spectra agreeing to 0.01% of peak can show a 50% "y"
difference).  Use the spectrum RMS column.

Usage::

    python dev/scripts/dm_residual_diagnostics/gf_entry_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from spectroxide.cosmotherm import load_greens_database, strip_gbb
from spectroxide.greens_table import GreensTable

K_B = 1.380_649e-23
H_PL = 6.626_070_15e-34
T_CMB = 2.726
C_LIGHT = 2.997_924_58e8

X_FIT_LO, X_FIT_HI = 0.5, 15.0
X_CMP = np.linspace(0.5, 12.0, 400)
STRIDE = 4


def shapes(x):
    ex = np.exp(np.clip(x, 0.0, 500.0))
    n_pl = 1.0 / (ex - 1.0)
    gbb = x * ex / (ex - 1.0) ** 2
    y_sz = gbb * (x * (ex + 1.0) / (ex - 1.0) - 4.0)
    m_x = n_pl * (n_pl + 1.0) * (x / 2.1922939 - 1.0)
    return m_x, y_sz, gbb


def decompose(x, dn):
    m_x, y_sz, gbb = shapes(x)
    mask = (x >= X_FIT_LO) & (x <= X_FIT_HI)
    a = np.column_stack([m_x[mask], y_sz[mask], gbb[mask]])
    c, *_ = np.linalg.lstsq(a, dn[mask], rcond=None)
    return c[0] * 1.401, c[1] * 0.25


def main() -> None:
    z_ct, x_ct, g_ct = load_greens_database()
    tbl = GreensTable.load(Path.home() / ".spectroxide" / "greens_table_hq.npz")
    print(f"table: {len(tbl.z_h)} z_h in [{tbl.z_h.min():.0f}, {tbl.z_h.max():.3g}], "
          f"{len(tbl.x)} x in [{tbl.x.min():.3g}, {tbl.x.max():.3g}]")
    print(f"CosmoTherm: {len(z_ct)} z_h, {len(x_ct)} x")
    print()
    print(f"{'z_h':>10} {'mu_CT':>11} {'mu_sx':>11} {'d%':>7} "
          f"{'y_CT':>11} {'y_sx':>11} {'d%':>7} {'spec rms/peak':>14}")

    nu_hz = x_ct * K_B * T_CMB / H_PL
    for iz in range(0, len(z_ct), STRIDE):
        zh = z_ct[iz]
        if zh < tbl.z_h.min() or zh > tbl.z_h.max():
            continue
        gf_jy = g_ct[:, iz] * np.exp(-((zh / 2.0e6) ** 2.5))
        dn_ct = gf_jy * 1e-26 * C_LIGHT**2 / (2.0 * H_PL * nu_hz**3)
        dn_ct_s = strip_gbb(x_ct, dn_ct)[0]
        dn_sx_s = strip_gbb(tbl.x, tbl.greens_function(tbl.x, zh))[0]

        mu_c, y_c = decompose(x_ct, dn_ct_s)
        mu_s, y_s = decompose(tbl.x, dn_sx_s)

        a = np.interp(X_CMP, tbl.x, dn_sx_s)
        b = np.interp(X_CMP, x_ct, dn_ct_s)
        rms = np.sqrt(np.mean((a - b) ** 2)) / np.max(np.abs(b)) * 100.0

        dmu = (mu_s / mu_c - 1.0) * 100.0 if abs(mu_c) > 1e-12 else np.nan
        dy = (y_s / y_c - 1.0) * 100.0 if abs(y_c) > 1e-12 else np.nan
        print(f"{zh:10.3e} {mu_c:11.4e} {mu_s:11.4e} {dmu:+7.2f} "
              f"{y_c:11.4e} {y_s:11.4e} {dy:+7.2f} {rms:13.2f}%")

    print()
    print("per-entry injected energy actually delivered by the solver "
          "(table field delta_rho_over_rho, nominal 1e-5):")
    print(f"{'z_h':>10} {'drho/rho':>12} {'deficit%':>9}")
    for i in range(0, len(tbl.z_h), 10):
        d = tbl.delta_rho_over_rho[i]
        print(f"{tbl.z_h[i]:10.3e} {d:12.5e} {(d / 1e-5 - 1) * 100:+9.3f}")


if __name__ == "__main__":
    main()
