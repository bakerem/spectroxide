#!/usr/bin/env python3
"""Plot visibility functions: literature vs PDE-fitted parameters.

Fitted parameters are read from dev/data/visibility_conservation_fit.json,
produced by fit_visibility_from_table.py's successor
(fit_visibility_conservation.py): thermalization parameters (A, B, beta)
from the conservation-law mu-era fit, transition parameters
(z_y, alpha_y, z_mu, alpha_mu) from the quadrature spectral fit.
"""
import sys
import json
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "python"))

import matplotlib
import matplotlib.pyplot as plt
from spectroxide.style import apply_style, C, DOUBLE_COL

apply_style()

datadir = pathlib.Path(__file__).resolve().parent.parent / "data"
with open(datadir / "visibility_conservation_fit.json") as f:
    fit = json.load(f)

# Adopted fiducial: the all-7 global spectral fit.
a7 = fit[fit["fiducial"]]
A_F, B_F, BETA_F = a7["A"], a7["B"], a7["beta"]
Z_Y, ALPHA_Y = a7["z_y"], a7["alpha_y"]
Z_MU_T, ALPHA_MU = a7["z_mu"], a7["alpha_mu"]

z = np.logspace(3, 7, 500)

Z_TH = 1.98e6  # analytic, fixed in both literature and this-work fits
ALPHA_TH = 2.5


def j_bb(z):
    return np.exp(-((z / Z_TH) ** ALPHA_TH))


# --- Literature (Chluba 2013; Chluba 2015 Eq. 13) ---
def j_bb_star_lit(z):
    r = z / Z_TH
    return np.maximum(0.983 * j_bb(z) * (1.0 - 0.0381 * r**2.29), 0.0)

def j_mu_lit(z):
    return 1.0 - np.exp(-(((1.0 + z) / 5.8e4) ** 1.88))

def j_y_lit(z):
    return 1.0 / (1.0 + ((1.0 + z) / 6.0e4) ** 2.58)


# --- This work (conservation-law + quadrature spectral fit) ---
def j_bb_star_new(z):
    r = z / Z_TH
    return np.maximum(A_F * j_bb(z) * (1.0 - B_F * r**BETA_F), 0.0)

def j_mu_new(z):
    return 1.0 - np.exp(-(((1.0 + z) / Z_MU_T) ** ALPHA_MU))

def j_y_new(z):
    return 1.0 / (1.0 + ((1.0 + z) / Z_Y) ** ALPHA_Y)


# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COL, 3.5),
                         gridspec_kw={"height_ratios": [3, 1]},
                         sharex=True)

# --- Top panel: visibility functions ---
ax = axes[0]

# Era shading
ax.axvspan(1e3, 5e4, alpha=0.06, color=C["teal"])
ax.axvspan(5e4, 2e5, alpha=0.06, color=C["purple"])
ax.axvspan(2e5, 2e6, alpha=0.06, color=C["orange"])
ax.axvspan(2e6, 1e7, alpha=0.06, color=C["blue"])
ax.text(8e3, 1.02, r"$y$-era", fontsize=7, color=C["teal"], ha="center")
ax.text(1e5, 1.02, "transition", fontsize=6, color=C["purple"], ha="center")
ax.text(6e5, 1.07, r"$\mu$-era", fontsize=7, color=C["orange"], ha="center")
ax.text(4e6, 0.7, "therm.", fontsize=7, color=C["blue"], ha="center")

# Literature curves (solid)
ax.semilogx(z, j_bb_star_lit(z), color=C["blue"], lw=1.5,
            label=r"$\mathcal{J}_{\mathrm{bb}}^*$ Chluba (2013, 2015)")
ax.semilogx(z, j_mu_lit(z), color=C["orange"], lw=1.5,
            label=r"$\mathcal{J}_\mu$ Chluba (2013)")
ax.semilogx(z, j_y_lit(z), color=C["teal"], lw=1.5,
            label=r"$\mathcal{J}_y$ Chluba (2013)")

# New fitted curves (dashed)
ax.semilogx(z, j_bb_star_new(z), color=C["blue"], lw=1.2, ls="--",
            label=r"$\mathcal{J}_{\mathrm{bb}}^*$ this work")
ax.semilogx(z, j_mu_new(z), color=C["orange"], lw=1.2, ls="--",
            label=r"$\mathcal{J}_\mu$ this work")
ax.semilogx(z, j_y_new(z), color=C["teal"], lw=1.2, ls="--",
            label=r"$\mathcal{J}_y$ this work")

ax.axvline(5e4, color=C["gray"], ls=":", lw=0.5)
ax.axvline(2e5, color=C["gray"], ls=":", lw=0.5)
ax.axvline(2e6, color=C["gray"], ls=":", lw=0.5)

ax.set_ylabel("Visibility function")
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=6, loc="center left", ncol=1)

# --- Bottom panel: absolute difference ---
ax2 = axes[1]
ax2.semilogx(z, j_bb_star_new(z) - j_bb_star_lit(z), color=C["blue"], lw=1.0,
             label=r"$\Delta\mathcal{J}_{\mathrm{bb}}^*$")
ax2.semilogx(z, j_mu_new(z) - j_mu_lit(z), color=C["orange"], lw=1.0,
             label=r"$\Delta\mathcal{J}_\mu$")
ax2.semilogx(z, j_y_new(z) - j_y_lit(z), color=C["teal"], lw=1.0,
             label=r"$\Delta\mathcal{J}_y$")

ax2.axhline(0, color="k", lw=0.5)
ax2.set_xlabel(r"Injection redshift $z_h$")
ax2.set_ylabel("Residual")
ax2.set_xlim(1e3, 1e7)
ax2.set_ylim(-0.06, 0.06)
ax2.legend(fontsize=7, loc="upper left", ncol=3)

fig.tight_layout()
outpath = pathlib.Path(__file__).resolve().parent.parent.parent / "notebooks" / "figures" / "pde_visibility_fit.pdf"
fig.savefig(outpath)
print(f"Saved: {outpath}")

print("\nFitted parameters (this work):")
print(f"  A={A_F:.4f} B={B_F:.5f} beta={BETA_F:.3f} (z_th={Z_TH:.3g}, "
      f"alpha_th={ALPHA_TH} fixed)")
print(f"  z_y={Z_Y:.5g} alpha_y={ALPHA_Y:.4f} z_mu={Z_MU_T:.5g} "
      f"alpha_mu={ALPHA_MU:.4f}")
