#!/usr/bin/env python3
"""Spread of the visibility parameters across equally-good fits.

Every stored fit variant in dev/data/ is re-evaluated on ONE common cost
function (x^3-weighted spectral residual, x in [0.5, 20], the one quoted in the
paper) so the costs are directly comparable. The point: the parameter spread
across fits that are indistinguishable in cost is comparable to, or larger
than, the difference between our quoted fit and the literature values.
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

NAMES = ["z_y", "a_y", "z_mu", "a_mu", "A", "z_th", "a_th", "B", "beta"]
KEYS = ["z_y", "α_y", "z_μ", "α_μ", "A", "z_th", "α_th", "B", "β"]

data = np.load(DATADIR / "visibility_table.npz")
z_h, x_pde, dn_nc, drho = data["z_h"], data["x"], data["dn_nc"], data["drho"]

M_x, Y_x, G_x = mu_shape(x_pde), y_shape(x_pde), g_bb(x_pde)
G_int = np.trapz(x_pde**2 * G_x, x_pde)
M_nc = M_x - np.trapz(x_pde**2 * M_x, x_pde) / G_int * G_x
Y_nc = Y_x - np.trapz(x_pde**2 * Y_x, x_pde) / G_int * G_x

mask = (x_pde >= 0.5) & (x_pde <= 20.0)
w = x_pde[mask] ** 3
Mw, Yw = w * M_nc[mask], w * Y_nc[mask]
Dw = w * dn_nc[:, mask]


def cost(p):
    z_y, a_y, z_mu, a_mu, A, z_th, a_th, B, beta = p
    jb = np.maximum(A * np.exp(-((z_h / z_th) ** a_th)) * (1 - B * (z_h / z_th) ** beta), 0.0)
    jm = 1.0 - np.exp(-(((1 + z_h) / z_mu) ** a_mu))
    jy = 1.0 / (1.0 + ((1 + z_h) / z_y) ** a_y)
    a_mu_amp, a_y_amp = MU_TO_ENERGY * jm * jb, 0.25 * jy
    model = (drho * a_mu_amp)[:, None] * Mw[None, :] + (drho * a_y_amp)[:, None] * Yw[None, :]
    r = model - Dw
    return float(np.sum(r * r))


variants = {}
LIT = [6.0e4, 2.58, 5.8e4, 1.88, 0.983, 1.98e6, 2.5, 0.0381, 2.29]
variants["literature (C13/C15)"] = LIT

r1 = json.load(open(DATADIR / "visibility_fit_results.json"))
for k, v in r1.items():
    if isinstance(v, dict) and "params" in v:
        variants[k] = [v["params"][kk] for kk in KEYS]

r2 = json.load(open(DATADIR / "visibility_spectral_fit_v2.json"))
for k in ("spectral_05_20", "spectral_1_15"):
    p = r2[k]["params"]
    label = "PAPER FIT (spectral_05_20)" if k == "spectral_05_20" else k
    variants[label] = [p["z_y"], p["α_y"], p["z_μ"], p["α_μ"], p["A"], 1.98e6, 2.5, p["B"], p["β"]]

costs = {k: cost(np.asarray(v, float)) for k, v in variants.items()}
cmin = min(costs.values())

hdr = f"{'variant':28s}" + "".join(f"{n:>10s}" for n in NAMES) + f"{'cost':>10s}{'dC':>8s}{'dC/C %':>8s}"
print(hdr)
print("-" * len(hdr))
for k, v in variants.items():
    c = costs[k]
    row = f"{k:28s}"
    for n, val in zip(NAMES, v):
        row += f"{val:10.4g}" if abs(val) < 1e4 else f"{val:10.5g}"
    row += f"{c:10.3f}{c - cmin:8.3f}{100 * (c - cmin) / cmin:8.3f}"
    print(row)

# Spread across the fitted variants only (exclude the literature row)
fit_keys = [k for k in variants if k != "literature (C13/C15)"]
arr = np.array([variants[k] for k in fit_keys], float)
print("\nSpread across the fitted variants (all within "
      f"{100 * (max(costs[k] for k in fit_keys) - cmin) / cmin:.3f}% in cost):")
print(f"{'param':10s}{'min':>12s}{'max':>12s}{'spread %':>10s}"
      f"{'literature':>12s}{'|lit-paper| %':>14s}{'inside?':>9s}")
paper = np.array(variants["PAPER FIT (spectral_05_20)"], float)
for j, n in enumerate(NAMES):
    lo, hi = arr[:, j].min(), arr[:, j].max()
    mid = 0.5 * (lo + hi)
    spread = 100 * (hi - lo) / mid if mid else 0.0
    dlit = 100 * abs(LIT[j] - paper[j]) / paper[j] if paper[j] else float("nan")
    inside = "yes" if lo <= LIT[j] <= hi else "NO"
    print(f"{n:10s}{lo:12.5g}{hi:12.5g}{spread:10.2f}{LIT[j]:12.5g}{dlit:14.2f}{inside:>9s}")
