#!/usr/bin/env python3
"""Build a high-resolution PDE reference table for visibility function fitting.

Runs single-burst PDE at ~120 injection redshifts with n_points=4000,
NC-strips each spectrum, and extracts per-spectrum (J_μ, J_bb*) by fitting
the three-component GF ansatz. Saves everything as an npz file so that
the visibility function fitting can be done separately without re-running
the PDE.

Outputs:
  dev/data/visibility_table.npz  — NumPy archive with:
    z_h          (N_z,)        injection redshifts
    x            (N_x,)        PDE frequency grid
    dn_raw       (N_z, N_x)    raw Δn per Δρ/ρ
    dn_nc        (N_z, N_x)    NC-stripped Δn per Δρ/ρ
    drho         (N_z,)        recovered Δρ/ρ (should be ≈ 1.0)
    pde_mu       (N_z,)        μ from PDE decomposition
    pde_y        (N_z,)        y from PDE decomposition
    j_mu_fit     (N_z,)        per-spectrum fitted J_μ
    j_bb_fit     (N_z,)        per-spectrum fitted J_bb*
    j_y_fixed    (N_z,)        J_y from Chluba formula (fixed per z_h)
    n_points     scalar         grid resolution used
"""
import sys
import pathlib
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "python"))

from spectroxide import run_sweep, strip_gbb
from spectroxide.greens import (
    j_bb_star,
    j_mu,
    j_y,
    mu_shape,
    y_shape,
    g_bb,
    decompose_distortion,
    Z_MU,
    KAPPA_C,
    G3_PLANCK,
)

MU_TO_ENERGY = 3.0 / KAPPA_C

# ── Configuration ───────────────────────────────────────────────────────
DELTA_RHO = 1e-5
N_POINTS = 4000
TIMEOUT = 7200  # 2 hours

# Dense z_h grid: ~120 points from y-era through deep thermalization.
# Extra density in transition region (3e4 - 2e5).
z_low = np.logspace(np.log10(3e3), np.log10(3e4), 20)        # y-era
z_trans = np.logspace(np.log10(3e4), np.log10(2e5), 40)       # transition
z_mu = np.logspace(np.log10(2e5), np.log10(2e6), 40)          # μ-era
z_therm = np.logspace(np.log10(2.5e6), np.log10(5e6), 20)     # thermalization
z_h_grid = np.unique(np.concatenate([z_low, z_trans, z_mu, z_therm]))

print(f"Visibility table generation")
print(f"  n_points = {N_POINTS}")
print(f"  N_z = {len(z_h_grid)} injection redshifts")
print(f"  z range: [{z_h_grid[0]:.2e}, {z_h_grid[-1]:.2e}]")
print(f"  Transition region (3e4–2e5): {np.sum((z_h_grid >= 3e4) & (z_h_grid <= 2e5))} points")
print()

# ── Run PDE sweep ───────────────────────────────────────────────────────
print(f"Running PDE sweep ({len(z_h_grid)} redshifts, n_points={N_POINTS})...")
t0 = time.time()
sweep = run_sweep(
    delta_rho=DELTA_RHO,
    z_injections=z_h_grid.tolist(),
    z_end=500.0,
    n_points=N_POINTS,
    timeout=TIMEOUT,
)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

results = sweep["results"]
x_pde = np.array(results[0]["x"])
n_x = len(x_pde)
n_z = len(z_h_grid)

print(f"  Grid: {n_x} frequency points, x ∈ [{x_pde[0]:.4f}, {x_pde[-1]:.2f}]")

# ── Extract and NC-strip spectra ────────────────────────────────────────
print("\nExtracting and NC-stripping spectra...")

dn_raw = np.zeros((n_z, n_x))
dn_nc = np.zeros((n_z, n_x))
drho_arr = np.zeros(n_z)
pde_mu_arr = np.zeros(n_z)
pde_y_arr = np.zeros(n_z)
j_mu_fit_arr = np.zeros(n_z)
j_bb_fit_arr = np.zeros(n_z)
j_y_fixed_arr = np.zeros(n_z)

for k, r in enumerate(results):
    z_h = z_h_grid[k]
    dn = np.array(r["delta_n"])
    gf = dn / DELTA_RHO  # per unit Δρ/ρ

    dn_raw[k, :] = gf

    # NC-strip
    gf_nc, _alpha = strip_gbb(x_pde, gf)
    dn_nc[k, :] = gf_nc

    # Energy integral
    dx = np.diff(x_pde)
    x_mid = 0.5 * (x_pde[:-1] + x_pde[1:])
    gf_mid = 0.5 * (gf[:-1] + gf[1:])
    drho_arr[k] = np.sum(x_mid**3 * gf_mid * dx) / G3_PLANCK

    # Per-spectrum decomposition (extracts J_mu, J_bb* fits)
    dec = decompose_distortion(x_pde, dn, z_h=z_h, method="gf_fit")
    pde_mu_arr[k] = dec["mu"] / DELTA_RHO
    pde_y_arr[k] = dec["y"] / DELTA_RHO
    j_mu_fit_arr[k] = dec["j_mu_fit"]
    j_bb_fit_arr[k] = dec["j_bb_star_fit"]
    j_y_fixed_arr[k] = dec["j_y"]

    if (k + 1) % 20 == 0 or k == n_z - 1:
        print(f"  {k + 1}/{n_z} spectra processed")

# ── Save ────────────────────────────────────────────────────────────────
outdir = pathlib.Path(__file__).resolve().parent.parent / "data"
outdir.mkdir(exist_ok=True)
outpath = outdir / "visibility_table.npz"

np.savez(
    outpath,
    z_h=z_h_grid,
    x=x_pde,
    dn_raw=dn_raw,
    dn_nc=dn_nc,
    drho=drho_arr,
    pde_mu=pde_mu_arr,
    pde_y=pde_y_arr,
    j_mu_fit=j_mu_fit_arr,
    j_bb_fit=j_bb_fit_arr,
    j_y_fixed=j_y_fixed_arr,
    n_points=N_POINTS,
)
print(f"\nSaved: {outpath}")
print(f"  z_h:      ({n_z},)")
print(f"  x:        ({n_x},)")
print(f"  dn_raw:   ({n_z}, {n_x})")
print(f"  dn_nc:    ({n_z}, {n_x})")

# ── Quick summary ───────────────────────────────────────────────────────
print(f"\n{'z_h':>10s}  {'Δρ/ρ':>8s}  {'μ':>10s}  {'y':>10s}  "
      f"{'J_μ fit':>8s}  {'J_bb* fit':>8s}  {'J_y':>8s}")
print("=" * 75)
for k in range(0, n_z, max(1, n_z // 25)):
    print(f"{z_h_grid[k]:10.3e}  {drho_arr[k]:8.4f}  {pde_mu_arr[k]:10.4f}  "
          f"{pde_y_arr[k]:10.4e}  {j_mu_fit_arr[k]:8.4f}  "
          f"{j_bb_fit_arr[k]:8.4f}  {j_y_fixed_arr[k]:8.4f}")

print(f"\nTotal runtime: {time.time() - t0:.0f}s")
