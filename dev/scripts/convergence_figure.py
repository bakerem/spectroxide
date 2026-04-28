"""Generate convergence figure for the paper.

Parses CONV lines from the Rust convergence_study_full_data test
and produces a two-panel figure: (a) mu/y vs N_grid (spatial),
(b) mu/y vs dy_max (temporal). Both panels show convergence order.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from spectroxide.style import apply_style, C, SINGLE_COL, DOUBLE_COL
import matplotlib.pyplot as plt

apply_style()

FIG_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "figures"
DATA_FILE = Path("/tmp/convergence_data_full.txt")

# --- Parse CONV lines ---
# Format: CONV|scenario|sweep_type|n_points|dy_max|mu|y|drho|l2_norm|steps
records = []
with open(DATA_FILE) as f:
    for line in f:
        line = line.strip()
        if not line.startswith("CONV|"):
            continue
        parts = line.split("|")
        records.append({
            "scenario": parts[1],
            "sweep": parts[2],
            "n_points": int(parts[3]),
            "dy_max": float(parts[4]),
            "mu": float(parts[5]),
            "y": float(parts[6]),
            "drho": float(parts[7]),
            "l2_norm": float(parts[8]),
            "steps": int(parts[9]),
        })

# --- Extract joint sweep (simultaneous N + dt refinement) ---
joint_fp = [r for r in records if r["scenario"] == "full_physics" and r["sweep"] == "joint"]
joint_fp.sort(key=lambda r: r["n_points"])

n_joint = np.array([r["n_points"] for r in joint_fp])
mu_joint = np.array([r["mu"] for r in joint_fp])
y_joint = np.array([r["y"] for r in joint_fp])

# --- Extract temporal sweep (N=4000 fixed) ---
temp_fp = [r for r in records if r["scenario"] == "full_physics" and r["sweep"] == "temporal"]
temp_fp.sort(key=lambda r: r["dy_max"], reverse=True)

dy_temp = np.array([r["dy_max"] for r in temp_fp])
mu_temp = np.array([r["mu"] for r in temp_fp])
y_temp = np.array([r["y"] for r in temp_fp])

# --- Extract spatial sweep ---
spat_fp = [r for r in records if r["scenario"] == "full_physics" and r["sweep"] == "spatial"]
spat_fp.sort(key=lambda r: r["n_points"])

n_spat = np.array([r["n_points"] for r in spat_fp])
mu_spat = np.array([r["mu"] for r in spat_fp])
y_spat = np.array([r["y"] for r in spat_fp])

# Also grab spectral L2 norms for Gaussian (cleanest convergence)
spat_gauss = [r for r in records if r["scenario"] == "gaussian" and r["sweep"] == "spectral"]
spat_gauss.sort(key=lambda r: r["n_points"])
n_gauss = np.array([r["n_points"] for r in spat_gauss])
l2_gauss = np.array([r["l2_norm"] for r in spat_gauss])

# --- Richardson extrapolation for finest reference ---
# Use last two joint levels to estimate converged value
# mu_extrap = (N2^p * mu2 - N1^p * mu1) / (N2^p - N1^p) for order p
# Simpler: just use finest as reference, show relative deviation

# --- Figure: two panels ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

# (a) Spatial: L2 spectral error (Gaussian IC)
if len(l2_gauss) > 0:
    ax1.loglog(n_gauss, l2_gauss, "o-", ms=4, lw=1, color=C["blue"], label="Spectral $L_2$ error")
    # Reference slope: O(dx^2) => O(1/N^2)
    n_ref = np.array([n_gauss[0], n_gauss[-1]])
    l2_ref = l2_gauss[0] * (n_ref[0] / n_ref) ** 2
    ax1.loglog(n_ref, l2_ref, "--", color="gray", lw=0.8, label=r"$\propto N^{-2}$")
elif len(spat_fp) > 0:
    # Fallback: use mu deviation from finest
    mu_ref = mu_spat[-1]
    mu_err = np.abs(mu_spat[:-1] - mu_ref) / np.abs(mu_ref) * 100
    ax1.loglog(n_spat[:-1], mu_err, "o-", ms=4, lw=1, color=C["blue"], label=r"$|\mu - \mu_{\rm ref}|/\mu_{\rm ref}$")
    n_ref = np.array([n_spat[0], n_spat[-2]])
    err_ref = mu_err[0] * (n_spat[0] / n_ref) ** 2
    ax1.loglog(n_ref, err_ref, "--", color="gray", lw=0.8, label=r"$\propto N^{-2}$")

ax1.set_xlabel(r"Grid points $N$")
ax1.set_ylabel(r"$x^3$-weighted $L_2$ error")
ax1.legend(fontsize=7)
ax1.set_title("(a) Spatial convergence", fontsize=8)

# (b) Joint: mu and y vs refinement level
if len(joint_fp) >= 2:
    # Show relative change from finest
    mu_ref_j = mu_joint[-1]
    y_ref_j = y_joint[-1]
    mu_dev_j = np.abs(mu_joint[:-1] - mu_ref_j) / np.abs(mu_ref_j) * 100
    y_dev_j = np.abs(y_joint[:-1] - y_ref_j) / np.abs(y_ref_j) * 100

    ax2.loglog(n_joint[:-1], mu_dev_j, "s-", ms=4, lw=1, color=C["blue"], label=r"$\mu$")
    ax2.loglog(n_joint[:-1], y_dev_j, "^-", ms=4, lw=1, color=C["red"], label=r"$y$")


ax2.set_xlabel(r"Grid points $N$")
ax2.set_ylabel(r"Relative deviation from finest [\%]")
from matplotlib.ticker import FixedLocator, FixedFormatter
ax2.xaxis.set_major_locator(FixedLocator(n_joint[:-1]))
ax2.xaxis.set_major_formatter(FixedFormatter([str(int(n)) for n in n_joint[:-1]]))
ax2.xaxis.set_minor_locator(FixedLocator([]))
ax2.legend(fontsize=7)
ax2.set_title("(b) Joint convergence", fontsize=8)

fig.tight_layout()
outpath = FIG_DIR / "convergence_study.pdf"
fig.savefig(outpath)
print(f"Saved to {outpath}")
plt.close()
