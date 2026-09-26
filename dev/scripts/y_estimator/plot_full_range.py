"""J_mu and J_y from 1e3 to 5e6 for three decompositions of the same PDE spectra, in the style of
Chluba & Jeong (2014, arXiv:1306.5751) Fig. 1.

Estimators:
  appendix   : the paper's appendix fit (Bianchini & Fabbian 2022), mu inside the exponential, mu, y
               and the temperature shift free, unweighted L2 on Delta n over x in [0.5, 18]. This is
               what spectroxide.greens._decompose_nonlinear_be did before ADR 0006; it is fitted here
               directly so the curve does not change when the library does.
  CJ2014     : linear {M, Y_SZ, G} fit to intensity in PIXIE-like channels, 30-1000 GHz in 15 GHz steps,
               flat sum (their Appendix A).
  appendix3  : the appendix model fitted to x^3 Delta n (intensity) over the same band (ADR 0006).
  appendix3nc: x^3-weighted, temperature shift removed first: G_bb is stripped from the data and from
               the shapes by photon-number conservation (spectroxide.strip_gbb convention, over the whole
               grid), then mu (linearised BE shape -G/x) and y are fitted. Linearising mu changes the
               appendix fit by < 3e-5 in J_mu at drho = 1e-5.
J_mu = mu / (1.401 Drho/rho), J_y = 4 y / (Drho/rho), with Drho/rho integrated from each spectrum.

Inputs: dev/data/visibility_table.npz (z_h >= 3e3) and dev/data/visibility_table_lowz.npz
(lowz_sweep.py), both stored per unit Drho/rho and rescaled to Drho/rho = 1e-5 before fitting.
Usage: plot_full_range.py out.pdf
"""
import sys, warnings
import numpy as np, matplotlib
from scipy.optimize import least_squares
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R = "/home/bakerem/spectroxide"
sys.path.insert(0, R + "/python")
from spectroxide import apply_style, C, DOUBLE_COL
from spectroxide.greens import mu_shape, y_shape, g_bb
apply_style(); warnings.simplefilter("ignore", RuntimeWarning)

G3, ALPHA = np.pi**4 / 15, 1.401
nu = np.arange(30.0, 1000.1, 15.0) * 1e9
X_CH = 6.62607015e-34 * nu / (1.380649e-23 * 2.725)          # CJ2014 channels, T0 = 2.725 K

def planck(x): return 1.0 / np.expm1(x)

def band(x, lo=0.5, hi=18.0):
    """Mask and trapezoid widths dx inside the band (widths from the full grid)."""
    dx = np.gradient(x)
    m = (x >= lo) & (x <= hi)
    return m, dx[m]

def fit_appendix(x, dn, power):
    m, w = band(x); xb = x[m]
    s = np.sqrt(w) * xb**power
    A = np.array([-g_bb(xb) / xb, g_bb(xb), y_shape(xb)]).T      # linearised start
    p0 = np.linalg.lstsq(A * s[:, None], dn[m] * s, rcond=None)[0]
    f = lambda p: s * (planck(xb + p[0]) - planck(xb) + p[1] * g_bb(xb) + p[2] * y_shape(xb) - dn[m])
    mu, _, y = least_squares(f, p0, x_scale=np.abs(p0) + 1e-12, xtol=1e-14, ftol=1e-14).x
    return mu, y

def strip(x, f):
    """Remove the G_bb component that carries the photon-number change of f."""
    return f - np.trapz(x**2 * f, x) / np.trapz(x**2 * g_bb(x), x) * g_bb(x)

def fit_appendix3_nc(x, dn):
    m, w = band(x); s = np.sqrt(w) * x[m]**3
    A = np.array([strip(x, -g_bb(x) / x)[m], strip(x, y_shape(x))[m]]).T
    mu, y = np.linalg.lstsq(A * s[:, None], strip(x, dn)[m] * s, rcond=None)[0]
    return mu, y

def fit_cj(x, dn):
    d = np.interp(X_CH, x, dn) * X_CH**3
    A = np.array([mu_shape(X_CH), y_shape(X_CH), g_bb(X_CH)]).T * X_CH[:, None]**3
    mu, y, _ = np.linalg.lstsq(A, d, rcond=None)[0]
    return mu, y

AMP = 1e-5  # spectra are stored per unit Drho/rho; the nonlinear fits need the physical amplitude

def estimate(x, dn):
    dn = AMP * dn
    E = np.trapz(x**3 * dn, x) / G3
    out = {"appendix": fit_appendix(x, dn, 0), "appendix3": fit_appendix(x, dn, 3),
           "appendix3nc": fit_appendix3_nc(x, dn), "cj": fit_cj(x, dn)}
    return {k: (mu / (ALPHA * E), 4 * y / E) for k, (mu, y) in out.items()}

hi = np.load(R + "/dev/data/visibility_table.npz"); lo = np.load(R + "/dev/data/visibility_table_lowz.npz")
rows = [(z, estimate(lo["x"], d)) for z, d in zip(lo["z_h"], lo["dn_raw"]) if z < 2999]
rows += [(z, estimate(hi["x"], d)) for z, d in zip(hi["z_h"], hi["dn_raw"])]
# overlap check at z_h = 3000: low-z run (difference quotient) vs the table (drho = 1e-5)
e_lo = estimate(lo["x"], lo["dn_raw"][-1]); e_hi = rows[[r[0] for r in rows].index(hi["z_h"][0])][1]
print("overlap z_h = 3000:", {k: np.round(np.subtract(e_lo[k], e_hi[k]), 5) for k in e_lo})
z = np.array([r[0] for r in rows])

EST = [("appendix", "appendix fit (paper): unweighted $\\Delta n$", C["blue"], "s"),
       ("cj", "Chluba \\& Jeong 2014: $x^3$, 15 GHz channels", C["orange"], "o"),
       ("appendix3", "appendix fit, $x^3$-weighted", C["teal"], "^"),
       ("appendix3nc", "appendix fit, $x^3$-weighted, $\\Delta T$ removed", C["purple"], "D")]
fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.3))
ax.axhline(0, color="k", lw=0.4); ax.axhline(1, color=C["gray"], lw=0.6, ls=":")
for key, lab, col, mk in EST:
    jm = np.array([r[1][key][0] for r in rows]); jy = np.array([r[1][key][1] for r in rows])
    ax.semilogx(z, jm, "-", color=col, lw=1.2, marker=mk, ms=2.2, mfc="none", mew=0.5, markevery=3)
    ax.semilogx(z, jy, "--", color=col, lw=1.2, marker=mk, ms=2.2, mfc="none", mew=0.5, markevery=3)
    print(key, "max J_mu %.3f at %.3g, max J_y %.3f at %.3g" % (jm.max(), z[jm.argmax()], jy.max(), z[jy.argmax()]))
ax.text(1.4e3, 0.93, r"$\mathcal{J}_y$", fontsize=8, va="top")
ax.text(2.2e6, 0.93, r"$\mathcal{J}_\mu$", fontsize=8, va="top", ha="right")
h = [Line2D([], [], color=c, marker=m, mfc="none", ms=3, lw=1.2, label=l) for _, l, c, m in EST]
h += [Line2D([], [], color="k", lw=1, ls="-", label=r"$\mathcal{J}_\mu = \mu/(1.401\,\Delta\rho/\rho)$"),
      Line2D([], [], color="k", lw=1, ls="--", label=r"$\mathcal{J}_y = 4y/(\Delta\rho/\rho)$")]
ax.legend(handles=h, fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.84))
ax.set_xlim(1e3, 5e6); ax.set_xlabel(r"Injection redshift $z_h$"); ax.set_ylabel("Fraction of energy")
ax.set_title("PDE single bursts, same spectra through four decompositions", fontsize=8)
fig.tight_layout(); fig.savefig(sys.argv[1]); fig.savefig(sys.argv[1].replace(".pdf", ".png"), dpi=200)
print("saved", sys.argv[1])
