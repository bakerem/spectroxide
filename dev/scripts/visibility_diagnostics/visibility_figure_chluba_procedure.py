#!/usr/bin/env python3
"""Fig. 2 regenerated under the reconstructed Chluba fit procedure.

Produces `notebooks/figures/pde_visibility_fit_chluba.pdf`.

The published Fig. 2 compares a seven-parameter joint spectral fit of ours
against parameters Chluba obtained from a four-parameter fit with J_therm held
at its analytic form. That is not like-for-like, and it inflates the apparent
disagreement (dev/audit/visibility_fit_degeneracy.md, Results 6 and 7). This
figure applies Chluba's procedure to our spectra:

  1. J_bb* fixed analytically, exp(-(z/1.98e6)^{5/2}); no free parameters.
  2. Subtract the resulting temperature-shift term 0.25 (1 - J_bb*) G_bb.
  3. At each injection redshift, fit free (M, Y) amplitudes by least squares on
     the intensity, i.e. x^3 weighting with a dx quadrature over x in [0.5, 20].
  4. Convert to visibilities: J_y = 4 a_Y, J_mu = a_M kappa_c / (3 J_bb*).
  5. Fit (z_y, alpha_y) and (z_mu, alpha_mu) separately by ordinary least
     squares to those two sequences.

Unlike the published figure this shows the extracted J_y points as well as
J_mu; the old version fixed J_y from Chluba's formula and never plotted it.
"""
import pathlib
import sys

import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectroxide.style import apply_style, C, DOUBLE_COL
from spectroxide.greens import mu_shape, y_shape, g_bb, KAPPA_C

apply_style()

MU_TO_ENERGY = 3.0 / KAPPA_C
FIG_DIR = ROOT / "notebooks" / "figures"
DATADIR = ROOT / "dev" / "data"
Z_TH, A_TH = 1.98e6, 2.5

LIT_Y, LIT_MU = np.array([6.0e4, 2.58]), np.array([5.8e4, 1.88])
B_Y = [(2e4, 2e5), (1.0, 5.0)]
B_MU = [(2e4, 2e5), (1.0, 4.0)]


def j_bb_star_analytic(z):
    return np.exp(-((z / Z_TH) ** A_TH))


def j_bb_star_lit(z):
    r = z / Z_TH
    return np.maximum(0.983 * np.exp(-(r ** A_TH)) * (1.0 - 0.0381 * r**2.29), 0.0)


def j_mu(z, p):
    return 1.0 - np.exp(-(((1.0 + z) / p[0]) ** p[1]))


def j_y(z, p):
    return 1.0 / (1.0 + ((1.0 + z) / p[0]) ** p[1])


# --- Step 1-4: extract per-redshift visibilities from our PDE spectra --------
tab = np.load(DATADIR / "visibility_table.npz")
z_h, x = tab["z_h"], tab["x"]
dn = tab["dn_raw"] / tab["drho"][:, None]

jb = j_bb_star_analytic(z_h)
D = dn - (0.25 * (1.0 - jb))[:, None] * g_bb(x)[None, :]

m = (x >= 0.5) & (x <= 20.0)
xs = x[m]
qw = xs**6 * np.gradient(xs)                  # (x^3)^2 dx
Ms, Ys, Ds = mu_shape(x)[m], y_shape(x)[m], D[:, m]

Gmm = np.sum(qw * Ms * Ms)
Gmy = np.sum(qw * Ms * Ys)
Gyy = np.sum(qw * Ys * Ys)
P, Q = Ds @ (qw * Ms), Ds @ (qw * Ys)
det = Gmm * Gyy - Gmy**2
a_M = (Gyy * P - Gmy * Q) / det
a_Y = (Gmm * Q - Gmy * P) / det

jy_obs = 4.0 * a_Y
ok = jb > 1e-3
jmu_obs = np.full_like(a_M, np.nan)
jmu_obs[ok] = a_M[ok] / (MU_TO_ENERGY * jb[ok])


# --- Step 5: separate least squares on each sequence ------------------------
def lsq(fn, seq, zz, bounds, seed=5):
    lb, ub = np.log([c[0] for c in bounds]), np.log([c[1] for c in bounds])
    bnd = list(zip(lb, ub))

    def cost(t):
        r = fn(zz, np.exp(t)) - seq
        return float(np.sum(r * r))

    r = differential_evolution(cost, bounds=bnd, seed=seed, maxiter=600, popsize=24,
                              tol=1e-14, polish=False, init="sobol")
    q = minimize(cost, r.x, method="L-BFGS-B", bounds=bnd,
                 options=dict(maxiter=20000, ftol=1e-18))
    return np.exp(q.x if q.fun < r.fun else r.x)


P_Y = lsq(j_y, jy_obs, z_h, B_Y)
P_MU = lsq(j_mu, jmu_obs[ok], z_h[ok], B_MU)

print("Chluba procedure applied to spectroxide spectra:")
for nm, v, lit in (("z_y", P_Y[0], 6.0e4), ("alpha_y", P_Y[1], 2.58),
                   ("z_mu", P_MU[0], 5.8e4), ("alpha_mu", P_MU[1], 1.88)):
    print(f"  {nm:9s} {v:10.4f}   literature {lit:8.4g}   {100 * (v - lit) / lit:+6.2f}%")

# --- Figure -----------------------------------------------------------------
z = np.logspace(3, 7, 500)

fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COL, 3.5),
                         gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

ax = axes[0]
ax.axvspan(1e3, 5e4, alpha=0.06, color=C["teal"])
ax.axvspan(5e4, 2e5, alpha=0.06, color=C["purple"])
ax.axvspan(2e5, 2e6, alpha=0.06, color=C["orange"])
ax.axvspan(2e6, 1e7, alpha=0.06, color=C["blue"])
ax.text(8e3, 1.08, r"$y$-era", fontsize=7, color=C["teal"], ha="center", va="center")
ax.text(1e5, 1.08, "transition", fontsize=7, color=C["purple"], ha="center", va="center")
ax.text(6e5, 1.08, r"$\mu$-era", fontsize=7, color=C["orange"], ha="center", va="center")
ax.text(4e6, 1.08, "therm.", fontsize=7, color=C["blue"], ha="center", va="center")

ax.semilogx(z, j_bb_star_lit(z), color=C["blue"], lw=1.5,
            label=r"$J_{\mathrm{bb}}^*$ Chluba (2013)")
ax.semilogx(z, j_mu(z, LIT_MU), color=C["orange"], lw=1.5,
            label=r"$J_\mu$ Chluba (2013)")
ax.semilogx(z, j_y(z, LIT_Y), color=C["teal"], lw=1.5,
            label=r"$J_y$ Chluba (2013)")

ax.semilogx(z, j_bb_star_analytic(z), color=C["blue"], lw=1.2, ls="--",
            label=r"$J_{\mathrm{bb}}^*$ analytic (fixed)")
ax.semilogx(z, j_mu(z, P_MU), color=C["orange"], lw=1.2, ls="--",
            label=r"$J_\mu$ this work")
ax.semilogx(z, j_y(z, P_Y), color=C["teal"], lw=1.2, ls="--",
            label=r"$J_y$ this work")

# Plot only where J_mu is well determined. The extraction divides by J_bb*, so
# once the thermalisation branch has shut off the quotient is amplified without
# limit; those points carry no information and are excluded from the display.
# The fit itself is insensitive to them (see visibility_scalar_zrange_robustness).
show = jb > 0.05
ax.semilogx(z_h[show], jmu_obs[show], ".", color=C["orange"], ms=2.5, alpha=0.6)
ax.semilogx(z_h, jy_obs, ".", color=C["teal"], ms=2.5, alpha=0.6)

for zv in (5e4, 2e5, 2e6):
    ax.axvline(zv, color=C["gray"], ls=":", lw=0.5)

ax.set_ylabel("Visibility function")
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=6, loc="center left", ncol=1)

ax2 = axes[1]
d_bb = j_bb_star_analytic(z) - j_bb_star_lit(z)
d_mu = j_mu(z, P_MU) - j_mu(z, LIT_MU)
d_y = j_y(z, P_Y) - j_y(z, LIT_Y)
ax2.semilogx(z, d_bb, color=C["blue"], lw=1.0, label=r"$\Delta J_{\mathrm{bb}}^*$")
ax2.semilogx(z, d_mu, color=C["orange"], lw=1.0, label=r"$\Delta J_\mu$")
ax2.semilogx(z, d_y, color=C["teal"], lw=1.0, label=r"$\Delta J_y$")
ax2.axhline(0, color="k", lw=0.5)
ax2.set_xlabel(r"Injection redshift $z_h$")
ax2.set_ylabel("Residual")
ax2.set_xlim(1e3, 1e7)
lim = max(0.02, 1.25 * max(np.abs(d_bb).max(), np.abs(d_mu).max(), np.abs(d_y).max()))
ax2.set_ylim(-lim, lim)
ax2.legend(fontsize=7, loc="upper left", ncol=3)

fig.tight_layout()
out = FIG_DIR / "pde_visibility_fit_chluba.pdf"
fig.savefig(out)
print(f"\nSaved: {out}")

print("peak |residual| vs Chluba (2013):")
for nm, arr in (("J_bb*", d_bb), ("J_mu", d_mu), ("J_y", d_y)):
    print(f"  {nm:6s} {np.abs(arr).max():.4f}  at z_h = {z[np.argmax(np.abs(arr))]:.3g}")
