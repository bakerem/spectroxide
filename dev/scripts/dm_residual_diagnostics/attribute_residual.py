#!/usr/bin/env python3
"""Attribute the Fig. 4 residual across PDE / spectroxide-GF-table / CosmoTherm-GF.

Reproduces the three-way RMS table in ``dev/audit/dm_comparison_residual.md``.
The PDE side is read from the notebook's cache
(``~/.spectroxide/dm_pde_results.json``); pass extra JSON files with the same
layout on the command line to compare alternative solver tolerances.

Usage::

    python dev/scripts/dm_residual_diagnostics/attribute_residual.py [extra.json ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from spectroxide import DEFAULT_COSMO, cosmic_time, delta_n_to_delta_I
from spectroxide.cosmotherm import (
    convolve_cosmotherm_gf,
    ct_heating_rate_decay,
    ct_heating_rate_pwave,
    ct_heating_rate_swave,
    load_greens_database,
    strip_gbb,
)
from spectroxide.greens_table import GreensTable

K_B = 1.380_649e-23
H_PL = 6.626_070_15e-34
T_CMB = 2.726
C_LIGHT = 2.997_924_58e8

# Figure parameters, kept in sync with notebooks/paper_figures/dm_scenario_comparison.ipynb
Z_START, Z_END, N_Z = 5.0e6, 1001.0, 5000
F_X, F_ANN_SW, F_ANN_PW = 7.757e5, 3.758e-20, 5.789e-26
RESID_MASK_FRAC = 0.02
NAMES = ["Decay", "s-wave", "p-wave"]

# The output grid only has to resolve the plotted shapes; 800 log points over the
# CosmoTherm database range keeps the convolution's query array small enough to
# avoid thrashing on a small-memory machine.
N_X_OUT = 800


def heating_rates():
    gamma_x = 1.0 / cosmic_time(5.0e4 + 10.0, DEFAULT_COSMO)
    return {
        "Decay": lambda z: ct_heating_rate_decay(z, F_X, gamma_x),
        "s-wave": lambda z: ct_heating_rate_swave(z, F_ANN_SW),
        "p-wave": lambda z: ct_heating_rate_pwave(z, F_ANN_PW),
    }


def di_from_jy(x, di_jy):
    """CosmoTherm GF output [Jy/sr] -> Delta_n."""
    nu_hz = x * K_B * T_CMB / H_PL
    return di_jy * 1e-26 * C_LIGHT**2 / (2.0 * H_PL * nu_hz**3)


def main() -> None:
    z_ct, x_ct, g_ct = load_greens_database()
    tbl = GreensTable.load(Path.home() / ".spectroxide" / "greens_table_hq.npz")
    heat = heating_rates()

    x_out = np.logspace(np.log10(x_ct.min()), np.log10(x_ct.max()), N_X_OUT)
    nu_ghz = x_out * K_B * T_CMB / H_PL / 1e9

    ct_dn, tab_dn = {}, {}
    for name in NAMES:
        _, di_raw = convolve_cosmotherm_gf(
            z_ct, x_ct, g_ct, heat[name],
            z_min=Z_END, z_max=Z_START, n_z=N_Z, x_out=x_out,
        )
        ct_dn[name] = strip_gbb(x_out, di_from_jy(x_out, di_raw))[0]
        dn = tbl.distortion_from_heating(tbl.x, heat[name], Z_END, Z_START, n_z=N_Z)
        tab_dn[name] = strip_gbb(tbl.x, dn)[0]

    def rms_pair(xa, dna, xb, dnb, x_ref, dn_ref):
        di_a = np.interp(x_out, xa, delta_n_to_delta_I(xa, dna)[1])
        di_b = np.interp(x_out, xb, delta_n_to_delta_I(xb, dnb)[1])
        di_ref = np.interp(x_out, x_ref, delta_n_to_delta_I(x_ref, dn_ref)[1])
        peak = np.max(np.abs(di_ref))
        mask = (
            (np.abs(di_ref) > RESID_MASK_FRAC * peak)
            & (nu_ghz > 30.0)
            & (nu_ghz < 857.0)
        )
        d = (di_a[mask] - di_b[mask]) / peak * 100.0
        return np.sqrt(np.mean(d**2)), np.max(np.abs(d))

    sources = {"notebook cache": Path.home() / ".spectroxide" / "dm_pde_results.json"}
    sources.update({Path(a).stem: Path(a) for a in sys.argv[1:]})

    for label, path in sources.items():
        if not path.exists():
            print(f"-- {label}: missing {path}")
            continue
        pde = json.load(open(path))["results"]
        print(f"\n### PDE source: {label}")
        print(f"{'scenario':10s} {'pair':24s} {'rms%':>8} {'max%':>8}")
        for name in NAMES:
            r = pde[name]
            x_p = np.array(r["x"])
            dn_p = strip_gbb(x_p, np.array(r["delta_n"]))[0]
            for plabel, a, b in [
                ("PDE - CT", (x_p, dn_p), (x_out, ct_dn[name])),
                ("GFtable - CT", (tbl.x, tab_dn[name]), (x_out, ct_dn[name])),
                ("PDE - GFtable", (x_p, dn_p), (tbl.x, tab_dn[name])),
            ]:
                rms, mx = rms_pair(*a, *b, x_p, dn_p)
                print(f"{name:10s} {plabel:24s} {rms:8.3f} {mx:8.3f}")
            print(f"{name:10s} {'drho/rho reported':24s} {r['drho']:.6e}")


if __name__ == "__main__":
    main()
