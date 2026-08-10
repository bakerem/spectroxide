"""Generate MMS convergence figure for the paper (Appendix B, verification).

Parses MMS| lines emitted by tests/mms_convergence.rs (spectroxide repo,
run with --nocapture) and produces a two-panel figure:
(a) spatial convergence of the manufactured-solution error on three grids,
(b) temporal convergence of the Crank-Nicolson and backward-Euler branches.

Regenerate the data with:
    cargo test --release --test mms_convergence -- --nocapture 2> mms_run.log
    grep -E '^MMS\\|' mms_run.log > dev/data/mms_convergence_data.txt
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from spectroxide.style import apply_style, C, DOUBLE_COL
import matplotlib.pyplot as plt

apply_style()

FIG_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "figures"
DATA_FILE = Path(__file__).resolve().parents[2] / "dev" / "data" / "mms_convergence_data.txt"

# --- Parse MMS lines ---
# Spatial format:  MMS|case|spatial|N=<n>|rel_l2=<e>
# Temporal format: MMS|case|temporal diffs: [d1, d2, ...], orders: [p1, ...]
spatial = {}   # case -> list of (N, err)
temporal = {}  # case -> list of successive solution differences
with open(DATA_FILE) as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 5 and parts[2] == "spatial":
            n = int(parts[3].split("=")[1])
            e = float(parts[4].split("=")[1])
            spatial.setdefault(parts[1], []).append((n, e))
        elif len(parts) >= 3 and parts[2].startswith("temporal diffs"):
            arr = parts[2].split("[")[1].split("]")[0]
            temporal[parts[1]] = [float(v) for v in arr.split(",")]

for case in spatial:
    spatial[case].sort()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

# (a) Spatial: manufactured-solution error vs N on three grids
cases = [
    ("pure_kompaneets", C["blue"], "o", "log grid"),
    ("coupled_dcbr", C["red"], "s", "log grid + DC/BR"),
    ("production_grid", C["teal"], "^", "production grid"),
]
for case, color, marker, label in cases:
    n = np.array([p[0] for p in spatial[case]])
    e = np.array([p[1] for p in spatial[case]])
    ax1.loglog(n, e, marker + "-", ms=4, lw=1, color=color, label=label)

n_ref = np.array([1300.0, 3600.0])
e_anchor = spatial["pure_kompaneets"][0][1]
ax1.loglog(n_ref, 0.45 * e_anchor * (400.0 / n_ref) ** 2, "--", color="0.4",
           lw=0.8)
ax1.text(2300, 0.45 * e_anchor * (400.0 / 1650.0) ** 2, r"$\propto N^{-2}$",
         fontsize=7, color="0.3", ha="left")
ax1.set_xlabel(r"Grid points $N$")
ax1.set_ylabel(r"$x^3$-weighted relative $L_2$ error")
ax1.legend(fontsize=7, loc="lower left")
ax1.set_title("(a) Spatial convergence (MMS)", fontsize=8)

# (b) Temporal: successive solution differences vs number of time steps.
# Ladders: M = 16/32/64/... halvings; diff k is between M_k and 2 M_k.
temp_cases = [
    ("pure_kompaneets", C["blue"], "o", "Crank–Nicolson"),
    ("coupled_dcbr", C["red"], "s", "backward Euler"),
]
for case, color, marker, label in temp_cases:
    d = np.array(temporal[case])
    m = 16 * 2 ** np.arange(len(d))
    ax2.loglog(m, d, marker + "-", ms=4, lw=1, color=color, label=label)

m_ref = np.array([40.0, 110.0])
d_cn = temporal["pure_kompaneets"][0]
d_be = temporal["coupled_dcbr"][0]
ax2.loglog(m_ref, 0.5 * d_cn * (16.0 / m_ref) ** 2, "--", color="0.4", lw=0.8)
ax2.text(115, 0.5 * d_cn * (16.0 / 110.0) ** 2, r"$\propto M^{-2}$",
         fontsize=7, color="0.3", ha="left", va="center")
ax2.loglog(m_ref, 0.5 * d_be * (16.0 / m_ref) ** 1, "--", color="0.4", lw=0.8)
ax2.text(115, 0.5 * d_be * (16.0 / 110.0) ** 1, r"$\propto M^{-1}$",
         fontsize=7, color="0.3", ha="left", va="center")
ax2.set_xlabel(r"Time steps $M$")
ax2.set_ylabel(r"Successive-refinement difference")
ax2.set_xlim(right=230)
ax2.legend(fontsize=7, loc="lower left")
ax2.set_title("(b) Temporal convergence (MMS)", fontsize=8)

fig.tight_layout()
outpath = FIG_DIR / "mms_convergence.pdf"
fig.savefig(outpath)
print(f"Saved to {outpath}")
plt.close()
