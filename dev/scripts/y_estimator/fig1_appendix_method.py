"""Fig. 1 with the Appendix (Bianchini & Fabbian, method="bf") mu/y extraction.

Same sweep as notebooks/paper_figures/mu_y_vs_injection_redshift.ipynb. Open markers:
the paper's visibility fit (method="gf_fit"), for comparison. Both are per unit of the
energy actually in the spectrum.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectroxide import apply_style, C, SINGLE_COL
from spectroxide.plot_params import LW, LW_THIN, MS_SMALL, ANNOT_SIZE
from spectroxide.solver import run_sweep
from spectroxide.greens import decompose_distortion, j_bb_star, j_mu, j_y

apply_style()
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "dev" / "figures"
DATA = ROOT / "dev" / "data"
MU_NORM = 1.401
z_inj = np.geomspace(1e3, 3e6, 40)
results = []
for zs, dr in ((z_inj[z_inj < 2e3], 1e-7), (z_inj[z_inj >= 2e3], 1e-5)):
    results += run_sweep(delta_rho=dr, z_injections=zs, n_points=4000,
                         production_grid=True, timeout=1200)["results"]

rows = []
for r in results:
    x, dn, z = np.asarray(r["x"]), np.asarray(r["delta_n"]), r["z_h"]
    bf = decompose_distortion(x, dn, method="bf")
    gf = decompose_distortion(x, dn, z_h=z, method="gf_fit")
    drho = bf["drho"] if "drho" in bf else r["drho"]
    rows.append(dict(z_h=z, mu_bf=bf["mu"], y_bf=bf["y"], drho=drho,
                     P_gf=gf["j_mu_fit"] * gf["j_bb_star_fit"], Jy_gf=gf["j_y"]))
json.dump(rows, open(DATA / "fig1_appendix_method.json", "w"), indent=1)

z_h = np.array([r["z_h"] for r in rows])
mu_bf = np.array([r["mu_bf"] / r["drho"] for r in rows])
y4_bf = np.array([4 * r["y_bf"] / r["drho"] for r in rows])
mu_gf = MU_NORM * np.array([r["P_gf"] for r in rows])
y4_gf = np.array([r["Jy_gf"] for r in rows])

for lo, hi, name, a, b in ((3e5, 2e6, "mu", mu_bf, mu_gf), (1e3, 1e4, "4y", y4_bf, y4_gf),
                           (1e4, 3e4, "4y", y4_bf, y4_gf)):
    m = (z_h >= lo) & (z_h <= hi)
    print(f"{name}/drho for {lo:.0e}-{hi:.0e}: appendix {a[m].min():.4f}-{a[m].max():.4f}, "
          f"visibility fit {b[m].min():.4f}-{b[m].max():.4f}, max |diff| {np.max(np.abs(a[m]-b[m])):.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(SINGLE_COL * 2, 2.8))
fig.subplots_adjust(wspace=0.35)
m1 = z_h >= 2e5
zs = np.geomspace(2e5, 3e6, 300)
ax1.semilogx(z_h[m1], mu_bf[m1], "o", color=C["blue"], ms=MS_SMALL, mew=0.5, zorder=5,
             label="PDE, Appendix fit")
ax1.semilogx(z_h[m1], mu_gf[m1], "o", mfc="none", color=C["gray"], ms=MS_SMALL + 1, mew=0.6,
             zorder=4, label="PDE, visibility fit")
ax1.semilogx(zs, MU_NORM * j_bb_star(zs) * j_mu(zs), "-", color=C["orange"], lw=LW,
             label="Analytic GF")
ax1.axhline(MU_NORM, color=C["gray"], ls=":", lw=LW_THIN)
ax1.axvspan(3e5, 2e6, alpha=0.05, color=C["orange"])
ax1.set(xlabel=r"Injection redshift $z_h$", ylabel=r"$\mu\,/\,\Delta\rho/\rho$",
        xlim=(1.5e5, 4e6), ylim=(0, 1.6))
ax1.legend(loc="center left", fontsize=ANNOT_SIZE)

m2 = z_h <= 3e4
zs = np.geomspace(1e3, 3e4, 300)
ax2.semilogx(z_h[m2], y4_bf[m2], "s", color=C["teal"], ms=MS_SMALL, mew=0.5, zorder=5,
             label="PDE, Appendix fit")
ax2.semilogx(z_h[m2], y4_gf[m2], "s", mfc="none", color=C["gray"], ms=MS_SMALL + 1, mew=0.6,
             zorder=4, label="PDE, visibility fit")
ax2.semilogx(zs, j_y(zs), "-", color=C["orange"], lw=LW, label="Analytic GF")
ax2.axhline(1.0, color=C["gray"], ls=":", lw=LW_THIN)
ax2.axvspan(1e3, 1e4, alpha=0.05, color=C["teal"])
ax2.set(xlabel=r"Injection redshift $z_h$", ylabel=r"$4y\,/\,\Delta\rho/\rho$", xlim=(7e2, 3e4),
        ylim=(0, max(1.15, 1.05 * np.nanmax(y4_bf[m2]))))
ax2.legend(loc="center left", fontsize=ANNOT_SIZE)
fig.savefig(OUT / "fig1_appendix_method.pdf")
fig.savefig(OUT / "fig1_appendix_method.png", dpi=130)
print("saved")
