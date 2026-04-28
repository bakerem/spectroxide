#!/usr/bin/env python3
"""Publication-quality comparison figures for spectroxide."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load publication style
plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())

DOUBLE_COL = 6.75

COLORS = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'teal': '#009988',
    'red': '#CC3311',
    'magenta': '#EE3377',
    'gray': '#BBBBBB',
    'cyan': '#33BBEE',
    'purple': '#AA3377',
}

# Load data
with open("sweep_output.json") as f:
    data = json.load(f)

drho_inj = data["delta_rho_inj"]
results = data["results"]

z_arr = np.array([r["z_h"] for r in results])
pde_mu = np.array([r["pde_mu"] for r in results])
gf_mu = np.array([r["gf_mu"] for r in results])
pde_y = np.array([r["pde_y"] for r in results])
gf_y = np.array([r["gf_y"] for r in results])
pde_drho = np.array([r["drho"] for r in results])


# ── Analytical reference shapes ──────────────────────────────────────────
def y_sz_shape(x):
    ex = np.exp(x)
    g_bb = x * ex / (ex - 1)**2
    coth = np.cosh(x / 2) / np.sinh(x / 2)
    return g_bb * (x * coth - 4)

def mu_shape(x):
    beta_mu = 2.1923
    ex = np.exp(x)
    g_bb = x * ex / (ex - 1)**2
    return (x / beta_mu - 1) * g_bb / x


# ── Figure: 4-panel comparison ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.5))

# ── (a) Thermalization efficiency ────────────────────────────────────────
ax = axes[0, 0]

# PDE
ax.plot(z_arr, np.abs(pde_mu), 'o-', color=COLORS['blue'], label=r'PDE $|\mu|$', ms=3, zorder=3)
ax.plot(z_arr, np.abs(pde_y), 's-', color=COLORS['orange'], label=r'PDE $|y|$', ms=3, zorder=3)

# Green's function
ax.plot(z_arr, np.abs(gf_mu), '--', color=COLORS['blue'], alpha=0.6, label=r'GF $|\mu|$')
ax.plot(z_arr, np.abs(gf_y), '--', color=COLORS['orange'], alpha=0.6, label=r'GF $|y|$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1.5e3, 7e5)
ax.set_ylim(1e-9, 5e-4)
ax.set_xlabel(r'Injection redshift $z_{\rm inj}$')
ax.set_ylabel(r'Distortion parameter')
ax.legend(ncol=2, loc='upper left')
ax.set_title(r'(a) Thermalization efficiency')

# Shade the mu-era and y-era
ax.axvspan(5e4, 7e5, alpha=0.04, color=COLORS['blue'], zorder=0)
ax.axvspan(1.5e3, 5e4, alpha=0.04, color=COLORS['orange'], zorder=0)
ax.text(1.5e5, 2e-9, r'$\mu$-era', color=COLORS['blue'], alpha=0.5)
ax.text(5e3, 2e-9, r'$y$-era', color=COLORS['orange'], alpha=0.5)

# ── (b) Energy conservation ──────────────────────────────────────────────
ax = axes[0, 1]

ratio = pde_drho / drho_inj
ax.plot(z_arr, ratio, 'o-', color=COLORS['teal'], ms=3)
ax.axhline(1.0, ls='--', color='k', lw=0.5, alpha=0.3)
ax.axhspan(0.95, 1.05, alpha=0.08, color=COLORS['gray'])
ax.set_xscale('log')
ax.set_xlim(1.5e3, 7e5)
ax.set_ylim(0.88, 1.12)
ax.set_xlabel(r'Injection redshift $z_{\rm inj}$')
ax.set_ylabel(r'$(\Delta\rho/\rho)_{\rm PDE}\, /\, (\Delta\rho/\rho)_{\rm inj}$')
ax.set_title(r'(b) Energy conservation')
ax.text(3e3, 0.96, r'$\pm 5\%$ band', color='gray')

# ── (c) Spectral distortion shapes at different z_inj ───────────────────
ax = axes[1, 0]

target_z = [5000, 10000, 50000, 200000]
colors_spec = [COLORS['orange'], COLORS['teal'], COLORS['blue'], COLORS['purple']]

for zt, col in zip(target_z, colors_spec):
    # Find closest z_h
    idx = np.argmin(np.abs(z_arr - zt))
    r = results[idx]
    x = np.array(r["x"])
    dn = np.array(r["delta_n"])
    mask = (x > 0.5) & (x < 15)
    if zt >= 1e4:
        exp = int(np.log10(zt))
        mant = zt / 10**exp
        if mant == 1.0:
            label = r'$z_{{\rm inj}} = 10^{{{0}}}$'.format(exp)
        else:
            label = r'$z_{{\rm inj}} = {0:.0f}\times 10^{{{1}}}$'.format(mant, exp)
    else:
        label = r'$z_{{\rm inj}} = {0:.0f}$'.format(zt)
    ax.plot(x[mask], dn[mask], color=col, label=label, lw=0.9)

ax.axhline(0, ls='--', color='k', lw=0.3, alpha=0.3)
ax.set_xlabel(r'$x = h\nu / k_B T_z$')
ax.set_ylabel(r'$\Delta n(x)$')
ax.set_xlim(0.5, 15)
ax.legend(loc='lower right')
ax.set_title(r'(c) Distortion shapes, $\Delta\rho/\rho = 10^{-5}$')

# ── (d) PDE vs Green's function spectral shape ─────────────────────────
ax = axes[1, 1]

# μ-era injection (z = 2×10⁵) — the key comparison
idx = np.argmin(np.abs(z_arr - 2e5))
r = results[idx]
x = np.array(r["x"])
dn_pde = np.array(r["delta_n"])
mask = (x > 0.5) & (x < 15)

ax.plot(x[mask], dn_pde[mask], color=COLORS['blue'], label='PDE', lw=1.2, zorder=3)

# Green's function spectral shape
if "delta_n_gf" in r:
    dn_gf = np.array(r["delta_n_gf"])
    ax.plot(x[mask], dn_gf[mask], '--', color=COLORS['orange'],
            label=r"Green's fn", lw=1.0)

# Also show the analytical μ × M(x) for reference
x_ref = np.linspace(0.5, 15, 500)
mu_gf_val = r["gf_mu"]
ax.plot(x_ref, mu_gf_val * mu_shape(x_ref), ':', color=COLORS['red'],
        label=r'$\mu_{\rm GF} \times M(x)$', lw=0.8, alpha=0.7)

ax.axhline(0, ls='--', color='k', lw=0.3, alpha=0.3)
ax.set_xlabel(r'$x = h\nu / k_B T_z$')
ax.set_ylabel(r'$\Delta n(x)$')
ax.set_xlim(0.5, 15)
ax.legend(loc='lower right')
ax.set_title(r'(d) PDE vs GF, $z_{\rm inj} = 2\times 10^5$')

fig.savefig('comparison_with_literature.pdf')
fig.savefig('comparison_with_literature.png')
print("Saved comparison_with_literature.pdf and .png")
plt.close()
