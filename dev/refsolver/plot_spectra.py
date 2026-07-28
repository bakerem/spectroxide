#!/usr/bin/env python3
"""Plot the five reference-solver distortion spectra (dev/output/refsolver/)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
DEST = os.path.join(HERE, "..", "output", "refsolver")
CASES = ["heat_z2e6", "heat_z2e5", "heat_z5e3", "adiabatic", "photon_x0.1_z3e5"]

os.makedirs(DEST, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for c in CASES:
    d = np.genfromtxt(os.path.join(OUT, f"spectrum_{c}.csv"), delimiter=",",
                      names=True)
    x, dn = d["x"], d["delta_n"]
    npl = 1.0 / np.expm1(x)
    axes[0].plot(x, np.abs(dn) / np.abs(dn).max(), label=c, lw=1.2)
    axes[1].plot(x, dn / npl, label=c, lw=1.2)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("x = h nu / k T_gamma")
axes[0].set_ylabel("|Delta n| / max|Delta n|")
axes[0].set_ylim(1e-10, 3)
axes[1].set_xscale("log")
axes[1].set_yscale("symlog", linthresh=1e-10)
axes[1].set_xlabel("x = h nu / k T_gamma")
axes[1].set_ylabel("Delta n / n_pl")
axes[1].axvspan(0.5, 18.0, color="0.9", zorder=0)
axes[1].legend(fontsize=7, loc="best")
for a in axes:
    a.axvline(0.5, ls=":", c="0.6", lw=0.8)
    a.axvline(18.0, ls=":", c="0.6", lw=0.8)
fig.suptitle("R3 clean-room reference solver: distortion spectra at z_end = 200"
             "  (shaded = fit window)", fontsize=9)
fig.tight_layout()
for p in (os.path.join(DEST, "refsolver_spectra.pdf"),
          os.path.join(OUT, "refsolver_spectra.png")):
    fig.savefig(p, dpi=130)
    print("wrote", os.path.abspath(p))
