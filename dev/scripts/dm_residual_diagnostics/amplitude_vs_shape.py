#!/usr/bin/env python3
"""Split the Fig. 4 residual into an amplitude part and a shape part.

Fits a single multiplicative factor ``s`` minimising ``|s dI_PDE - dI_CT|`` over
the figure's mask, then reports the residual that survives the rescale.  Also
prints the distortion energies ``int x^3 dn dx / G3``, whose ratio should equal
``s`` if the residual is purely an energy-budget offset.

Caches the CosmoTherm convolution to ``ct_curves.npz`` next to this script.

Usage::

    python dev/scripts/dm_residual_diagnostics/amplitude_vs_shape.py [[label=]cache.json ...]

With no arguments it reads the legacy single-file cache; otherwise each argument is
a keyed cache entry from ``~/.spectroxide/dm_pde_cache/``, e.g.::

    ... amplitude_vs_shape.py dy0.005=~/.spectroxide/dm_pde_cache/<key>.json
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
from spectroxide.greens import G3_PLANCK

K_B = 1.380_649e-23
H_PL = 6.626_070_15e-34
T_CMB = 2.726
C_LIGHT = 2.997_924_58e8

Z_START, Z_END, N_Z = 5.0e6, 1001.0, 5000
F_X, F_ANN_SW, F_ANN_PW = 7.757e5, 3.758e-20, 5.789e-26
RESID_MASK_FRAC = 0.02
NAMES = ["Decay", "s-wave", "p-wave"]
N_X_OUT = 800
# Energy window: wide enough that x^3 dn has decayed at both ends.
X_LO, X_HI = 0.02, 30.0

CACHE = Path(__file__).resolve().parent / "ct_curves.npz"


def heating_rates():
    gamma_x = 1.0 / cosmic_time(5.0e4 + 10.0, DEFAULT_COSMO)
    return {
        "Decay": lambda z: ct_heating_rate_decay(z, F_X, gamma_x),
        "s-wave": lambda z: ct_heating_rate_swave(z, F_ANN_SW),
        "p-wave": lambda z: ct_heating_rate_pwave(z, F_ANN_PW),
    }


def load_ct_curves():
    if CACHE.exists():
        d = np.load(CACHE)
        return d["x"], {n: d[n] for n in NAMES}

    z_ct, x_ct, g_ct = load_greens_database()
    heat = heating_rates()
    x_out = np.logspace(np.log10(x_ct.min()), np.log10(x_ct.max()), N_X_OUT)
    curves = {}
    for name in NAMES:
        _, di_raw = convolve_cosmotherm_gf(
            z_ct, x_ct, g_ct, heat[name],
            z_min=Z_END, z_max=Z_START, n_z=N_Z, x_out=x_out,
        )
        nu_hz = x_out * K_B * T_CMB / H_PL
        dn = di_raw * 1e-26 * C_LIGHT**2 / (2.0 * H_PL * nu_hz**3)
        curves[name] = strip_gbb(x_out, dn)[0]
    np.savez(CACHE, x=x_out, **curves)
    print(f"cached CosmoTherm curves -> {CACHE}")
    return x_out, curves


def energy(x, dn):
    m = (x >= X_LO) & (x <= X_HI)
    return np.trapz(x[m] ** 3 * dn[m], x[m]) / G3_PLANCK


def main() -> None:
    x_out, ct_dn = load_ct_curves()
    nu_ghz = x_out * K_B * T_CMB / H_PL / 1e9

    # Arguments are cache files, optionally labelled as ``label=path`` so a scan
    # over solver tolerances prints readable rows instead of SHA-256 stems.
    if sys.argv[1:]:
        sources = {}
        for a in sys.argv[1:]:
            label, _, path = a.rpartition("=")
            sources[label or Path(path).stem] = Path(path)
    else:
        sources = {"notebook cache": Path.home() / ".spectroxide" / "dm_pde_results.json"}

    for label, path in sources.items():
        if not path.exists():
            print(f"-- {label}: missing {path}")
            continue
        pde = json.load(open(path))["results"]
        print(f"\n### {label}")
        print(
            f"{'scen':8s} {'rms%':>7} {'s-1 %':>8} {'rms after':>10} "
            f"{'E_CT':>12} {'E_PDE':>12} {'E_CT/E_PDE':>11} {'drho/rho':>12}"
        )
        for name in NAMES:
            r = pde[name]
            x_p = np.array(r["x"])
            dn_p = strip_gbb(x_p, np.array(r["delta_n"]))[0]
            di_p = np.interp(x_out, x_p, delta_n_to_delta_I(x_p, dn_p)[1])
            di_c = delta_n_to_delta_I(x_out, ct_dn[name])[1]
            peak = np.max(np.abs(di_p))
            mask = (
                (np.abs(di_c) > RESID_MASK_FRAC * peak)
                & (nu_ghz > 30.0)
                & (nu_ghz < 857.0)
            )
            d = (di_p[mask] - di_c[mask]) / peak * 100.0
            s = np.dot(di_p[mask], di_c[mask]) / np.dot(di_p[mask], di_p[mask])
            d2 = (s * di_p[mask] - di_c[mask]) / peak * 100.0
            e_ct, e_p = energy(x_out, ct_dn[name]), energy(x_p, dn_p)
            print(
                f"{name:8s} {np.sqrt(np.mean(d**2)):7.3f} {(s - 1) * 100:+8.3f} "
                f"{np.sqrt(np.mean(d2**2)):10.3f} {e_ct:12.5e} {e_p:12.5e} "
                f"{e_ct / e_p:11.5f} {r['drho']:12.5e}"
            )


if __name__ == "__main__":
    main()
