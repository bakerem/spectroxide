"""Plot mu and 4y per unit energy for three estimators, PDE (markers) and CosmoTherm (lines),
with residuals relative to the visibility fit. Usage: plot.py est.json out.pdf"""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/bakerem/spectroxide/python")
from spectroxide import apply_style, C, DOUBLE_COL
from spectroxide.greens import j_y, j_mu, j_bb_star
apply_style()
d = json.load(open(sys.argv[1])); p = d["pde"]; c = [r for r in d["ct"] if r["z"] < 4e5]
col = lambda rows, k: np.array([r[k] for r in rows])
zp, zc = col(p, "z"), col(c, "z")
zs = np.geomspace(1e3, 5e6, 400)
EST = [  # key for mu, key for 4y, label, color, marker
    ("vis_mu", "vis_Jy", "visibility fit ($x^3$, $\\Delta T$ removed)", C["orange"], "o"),
    ("bf_mu", "bf_4y", "appendix fit (unweighted, as in paper)", C["blue"], "s"),
    ("bf3_mu", "bf3_4y", "appendix fit, $x^3$-weighted", C["teal"], "^"),
    ("bfnc_mu", "bfnc_4y", "appendix fit, unweighted, $\\Delta T$ removed", C["purple"], "D"),
]
for rows in (p, c):
    for r in rows: r["vis_mu"] = 1.401 * r["vis_P"]
fig, ax = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
form = {0: 1.401 * j_mu(zs) * j_bb_star(zs), 1: j_y(zs)}
for j, (lab, fk) in enumerate([(r"$\mu/(\Delta\rho/\rho)$", 0), (r"$4y/(\Delta\rho/\rho)$", 1)]):
    top, bot = ax[0, j], ax[1, j]
    top.semilogx(zs, form[fk], "-", color=C["gray"], lw=1, label="Chluba formulas")
    ref_p, ref_c = col(p, EST[0][j]), col(c, EST[0][j])
    for kmu_ky in EST:
        k, name, cl, mk = kmu_ky[j], kmu_ky[2], kmu_ky[3], kmu_ky[4]
        top.semilogx(zc, col(c, k), "-", color=cl, lw=1.1)
        top.semilogx(zp, col(p, k), mk, color=cl, ms=2.6, mfc="none", mew=0.6, label=name)
        bot.semilogx(zc, col(c, k) - ref_c, "-", color=cl, lw=1.1)
        bot.semilogx(zp, col(p, k) - ref_p, mk, color=cl, ms=2.6, mfc="none", mew=0.6)
    fz = j_mu(zp) * j_bb_star(zp) * 1.401 if fk == 0 else j_y(zp)
    bot.semilogx(zp, fz - ref_p, "-", color=C["gray"], lw=1)
    bot.axhline(0, color="k", lw=0.4)
    top.set_ylabel(lab); bot.set_ylabel("minus visibility fit")
    bot.set_xlabel(r"Injection redshift $z_h$"); bot.set_xlim(1e3, 5e6)
ax[0, 1].axhline(1, color=C["gray"], ls=":", lw=0.6)
ax[0, 0].legend(fontsize=5.5, loc="upper left", frameon=False)
ax[0, 1].text(0.97, 0.95, "markers: PDE\nlines: CosmoTherm", transform=ax[0, 1].transAxes, ha="right", va="top", fontsize=6)
fig.tight_layout(); fig.savefig(sys.argv[2]); print("saved", sys.argv[2])
