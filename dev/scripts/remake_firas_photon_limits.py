"""Regenerate FIRAS photon injection limits figure for the paper.

Thins PDE points near the peak and increases timeout for z_h=2e6.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from spectroxide import apply_style, strip_gbb, decompose_distortion
from spectroxide.solver import run_sweep
from spectroxide.greens import (
    ALPHA_RHO, mu_from_photon_injection,
    greens_function_photon, mu_shape, y_shape, g_bb,
)
from spectroxide.style import C, SINGLE_COL
import matplotlib.pyplot as plt

apply_style()

FIG_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "figures"

# FIRAS 68% CL limits
MU_FIRAS = 4.5e-5
Y_FIRAS = 7.5e-6


def y_from_photon_injection_gf(x_inj, z_h, delta_n_over_n):
    x_grid = np.linspace(0.5, 25, 2000)
    gf_spec = greens_function_photon(x_grid, x_inj, z_h) * delta_n_over_n
    M = mu_shape(x_grid)
    Y = y_shape(x_grid)
    G = g_bb(x_grid)
    A_mat = np.column_stack([M, Y, G])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, gf_spec, rcond=None)
    return coeffs[1]


def run_pde_photon(x_inj, z_h, delta_n_over_n=1e-5, sigma_x=None,
                   z_end=500, n_points=None, timeout=600):
    if sigma_x is None:
        sigma_x = 0.05 * x_inj
    sweep_kwargs = dict(
        injection={
            'type': 'monochromatic_photon',
            'x_inj': x_inj,
            'delta_n_over_n': delta_n_over_n,
            'z_h': z_h,
            'sigma_x': sigma_x,
        },
        z_start=z_h + 7 * max(z_h * 0.04, 100),
        z_end=z_end,
        number_conserving=False,
        nc_z_min=0,
        timeout=timeout,
    )
    if n_points is not None:
        sweep_kwargs['n_points'] = n_points
    result = run_sweep(**sweep_kwargs)
    r = result['results'][0]
    return np.array(r['x']), np.array(r['delta_n']), r['pde_mu'], r['pde_y']


# ================================================================
# GF results (fast)
# ================================================================
x_i_gf = np.logspace(np.log10(0.1), np.log10(15), 300)

z_values = [2e6, 1e6, 3e5]
z_labels = [r"$z_h = 2\times10^6$", r"$z_h = 10^6$", r"$z_h = 3\times10^5$"]
colors = [C["blue"], C["orange"], C["teal"]]

print("Computing GF curves...", flush=True)
gf_results = {}
for z_h in z_values:
    mu_arr = np.array([mu_from_photon_injection(x, z_h, 1.0) for x in x_i_gf])
    y_arr = np.array([y_from_photon_injection_gf(x, z_h, 1.0) for x in x_i_gf])
    dn_max_mu = np.where(np.abs(mu_arr) > 1e-20, MU_FIRAS / np.abs(mu_arr), np.inf)
    dn_max_y = np.where(np.abs(y_arr) > 1e-20, Y_FIRAS / np.abs(y_arr), np.inf)
    gf_results[z_h] = np.minimum(dn_max_mu, dn_max_y)
    print(f"  z={z_h:.0e} done")

# ================================================================
# PDE results — thinned near peak, longer timeout for z_h=2e6
# ================================================================
# Removed 3.3, 3.8, 4.5 near the peak to speed up
x_i_pde = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                     5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0])
dn_over_n = 1e-5

print("\nRunning PDE photon injections...", flush=True)
t0_total = time.time()
pde_results = {}

for z_h in z_values:
    mu_list = []
    y_list = []
    t0 = time.time()
    # Longer timeout for z_h=2e6 where low-x runs are slow
    timeout = 1800 if z_h >= 2e6 else 600

    for i, x_inj in enumerate(x_i_pde):
        print(f"  z={z_h:.0e}, x_i={x_inj:5.2f} ({i+1}/{len(x_i_pde)}) ...",
              end="", flush=True)
        try:
            _, _, mu, y = run_pde_photon(x_inj, z_h, delta_n_over_n=dn_over_n,
                                          timeout=timeout)
            mu_list.append(mu)
            y_list.append(y)
            print(f" mu={mu:.3e}, y={y:.3e}")
        except Exception as e:
            print(f" FAILED: {e}")
            mu_list.append(np.nan)
            y_list.append(np.nan)

    mu_per_unit = np.array(mu_list) / dn_over_n
    y_per_unit = np.array(y_list) / dn_over_n
    dn_max_mu = np.where(np.abs(mu_per_unit) > 1e-20, MU_FIRAS / np.abs(mu_per_unit), np.inf)
    dn_max_y = np.where(np.abs(y_per_unit) > 1e-20, Y_FIRAS / np.abs(y_per_unit), np.inf)
    pde_results[z_h] = np.minimum(dn_max_mu, dn_max_y)

    dt = time.time() - t0
    print(f"  z={z_h:.0e} done in {dt:.0f}s")

print(f"\nTotal PDE time: {time.time() - t0_total:.0f}s")

# ================================================================
# Figure
# ================================================================
LW = 1.2
LW_THIN = 0.6
LEGEND_SIZE = 7

fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))

for z_h, label, color in zip(z_values, z_labels, colors):
    ax.semilogy(x_i_gf, gf_results[z_h], color=color, lw=LW, label=label)
    pde_dn = pde_results[z_h]
    valid = np.isfinite(pde_dn) & (pde_dn > 0) & (pde_dn < 1e10)
    ax.plot(x_i_pde[valid], pde_dn[valid], 'o', color=color, ms=3,
            markeredgecolor='none', alpha=0.8)

x0 = 4.0 / (3.0 * ALPHA_RHO)
ax.axvline(x0, color="gray", ls="--", lw=LW_THIN, alpha=0.6)
ax.text(x0 + 0.2, 2e-6, r"$x_0$", fontsize=7, color="gray")

ax.set_xlabel(r"Injection frequency $x_i$")
ax.set_ylabel(r"$\Delta N_\gamma / N_\gamma$ (68\% CL)")
ax.set_xlim(0.1, 15)
ax.set_ylim(5e-6, 1e1)
ax.set_xscale("log")
ax.legend(fontsize=LEGEND_SIZE, loc="upper left")

plt.tight_layout()
outpath = FIG_DIR / "firas_photon_limits_paper.pdf"
fig.savefig(outpath, bbox_inches="tight")
print(f"\nSaved {outpath}")
plt.close()
