import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/bakerem/spectroxide/python")
from spectroxide import apply_style, C, DOUBLE_COL
from spectroxide.greens import j_y, j_mu, j_bb_star
apply_style()
d = json.load(open(sys.argv[1])); p = d["pde"]; c = [r for r in d["ct"] if r["z"] < 4e5]
zs = np.geomspace(1e3, 5e6, 400)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.9))
f = lambda rows, k: (np.array([r["z"] for r in rows]), np.array([r[k] for r in rows]))
a1.semilogx(zs, 1.401 * j_mu(zs) * j_bb_star(zs), "-", color=C["gray"], lw=1, label="Chluba (2013) formula")
z, v = f(c, "vis_P"); a1.semilogx(z, 1.401 * v, "-", color=C["orange"], lw=1.2, label="CosmoTherm, visibility fit")
z, v = f(p, "vis_P"); a1.semilogx(z, 1.401 * v, "o", color=C["orange"], ms=3, mfc="none", label="PDE, visibility fit")
z, v = f(c, "bf_mu"); a1.semilogx(z, v, "-", color=C["blue"], lw=1.2, label="CosmoTherm, paper appendix fit")
z, v = f(p, "bf_mu"); a1.semilogx(z, v, "s", color=C["blue"], ms=3, mfc="none", label="PDE, paper appendix fit")
a1.set_xlabel(r"Injection redshift $z_h$"); a1.set_ylabel(r"$\mu/(\Delta\rho/\rho)$"); a1.set_xlim(1e3, 5e6)
a2.semilogx(zs, j_y(zs), "-", color=C["gray"], lw=1)
z, v = f(c, "vis_Jy"); a2.semilogx(z, v, "-", color=C["orange"], lw=1.2)
z, v = f(p, "vis_Jy"); a2.semilogx(z, v, "o", color=C["orange"], ms=3, mfc="none")
z, v = f(c, "bf_4y"); a2.semilogx(z, v, "-", color=C["blue"], lw=1.2)
z, v = f(p, "bf_4y"); a2.semilogx(z, v, "s", color=C["blue"], ms=3, mfc="none")
a2.axhline(1, color=C["gray"], ls=":", lw=0.6)
a2.set_xlabel(r"Injection redshift $z_h$"); a2.set_ylabel(r"$4y/(\Delta\rho/\rho)$"); a2.set_xlim(1e3, 5e6)
a1.legend(fontsize=6, loc="upper left", frameon=False)
fig.tight_layout(); fig.savefig(sys.argv[2]); print("saved", sys.argv[2])
