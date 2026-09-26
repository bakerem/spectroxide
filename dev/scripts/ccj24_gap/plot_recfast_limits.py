"""Dark-photon FIRAS limit against CCJ24 with our X_e, RECFAST and HyRec (record: dev/audit/ccj24_postrec_gap.md).

No PDE runs: the template limit gamma_lim is the same at every mass below 5e-8 eV (to 0.2%),
so eps = sqrt(gamma_lim / (gamma_con/eps^2)) and only the X_e history in the mapping changes.
Statistic: CCJ24-literal, Delta chi^2 = 4. RECFAST and HyRec X_e come from CLASS v3.2.
Output: dev/figures/dp_ccj24_recfast_limits.{pdf,png}.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from postrec_gap import HERE, ROOT, X0, GMap, gamma_limit, xe_class, xe_ours
from spectroxide import apply_style
from spectroxide.cosmology import DEFAULT_COSMO, PLANCK2018_COSMO as P18
from spectroxide.plot_params import C, DOUBLE_COL, LEGEND_SIZE, LW, LW_THIN, MS

apply_style()
TM = np.load(HERE / 'grid_templates.npz', allow_pickle=True)
mt = TM['m'].astype(float)
gl = np.array([gamma_limit(np.interp(X0, TM['x'][i], TM['dn_per_gc'][i]))[0] for i in range(len(mt))])
glim = lambda m: np.exp(np.interp(np.log(m), np.log(mt), np.log(gl)))
VEC = np.loadtxt(HERE / 'ccj24_fig8_firas_vector.txt')
ev = lambda m: 10**np.interp(np.log10(m), np.log10(VEC[:, 0]), np.log10(VEC[:, 1]))
AL = np.loadtxt(ROOT / 'dev/AxionLimits/limit_data/DarkPhoton/COBEFIRAS_Chluba.txt')
AL = AL[AL[:, 0] <= 1.5e-4]

xo, xr, xh = xe_ours(P18), xe_class(P18, 'RECFAST'), xe_class(P18)
maps = [  # label, mapping, style
    ('ours (Saha He), default cosmology', GMap(*xe_ours(DEFAULT_COSMO), DEFAULT_COSMO), dict(color=C['gray'], ls='--')),
    ('ours (Saha He), Planck 2018', GMap(*xo, P18), dict(color=C['red'])),
    ('RECFAST, Planck 2018', GMap(*xr, P18), dict(color=C['blue'])),
    ('HyRec-2020, Planck 2018', GMap(*xh, P18), dict(color=C['teal'], ls=':')),
]
ms = np.geomspace(1.5e-12, 1e-7, 80)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.9), gridspec_kw={'wspace': 0.28})

# (a) X_e relative to HyRec
z = np.geomspace(150, 8000, 1500)
sh = CubicSpline(np.log1p(xh[0]), np.log(xh[1]))
for (zz, xx), lab, col in [(xo, 'ours (Saha He)', C['red']), (xr, 'RECFAST', C['blue'])]:
    s = CubicSpline(np.log1p(zz), np.log(xx))
    a1.semilogx(z, np.exp(s(np.log1p(z)) - sh(np.log1p(z))), color=col, lw=LW, label=lab)
a1.axhline(1, color='k', lw=LW_THIN)
a1.axvspan(1700, 2700, color=C['gray'], alpha=0.25, lw=0)
a1.text(2150, 1.012, 'He I', ha='center', fontsize=LEGEND_SIZE)
a1.set_xticks([200, 500, 1000, 2000, 5000])
a1.set_xticklabels(['200', '500', '1000', '2000', '5000'])
a1.set_xlabel('redshift $z$')
a1.set_ylabel(r'$X_e / X_e^{\rm HyRec}$ (Planck 2018)')
a1.set_ylim(0.93, 1.03)
a1.legend(fontsize=LEGEND_SIZE, loc='lower left')

# (b) eps limit relative to CCJ24 Fig. 8
for lab, g, st in maps:
    e = np.array([np.sqrt(glim(m) / g.gpe2(m)[0]) for m in ms])
    a2.semilogx(ms, e / ev(ms), lw=LW, label=lab, **st)
a2.semilogx(ms, 10**np.interp(np.log10(ms), np.log10(AL[:, 0]), np.log10(AL[:, 1])) / ev(ms),
            color=C['black'], lw=LW_THIN, ls='-.', label='CCJ24 AxionLimits file')
cache = np.load(ROOT / 'dev/data/dp_firas_pde_limits.npz')
sel = (cache['m'] >= ms[0]) & (cache['m'] <= ms[-1])
zc = np.array([GMap(*xe_ours(DEFAULT_COSMO), DEFAULT_COSMO).gpe2(m)[1] for m in cache['m'][sel]])
r = cache['e_pl'][sel] / ev(cache['m'][sel])
a2.plot(cache['m'][sel], r / np.median(r[(zc > 3e3) & (zc < 5e4)]), 'o', color=C['orange'], ms=MS,
        mfc='none', label='paper Fig. 8 PDE (rescaled)')
a2.axhline(1, color='k', lw=LW_THIN)
a2.axvspan(1.1e-9, 2.6e-9, color=C['gray'], alpha=0.25, lw=0)
a2.set_xlabel(r"dark-photon mass $m_{A'}$ [eV]")
a2.set_ylabel(r'$\epsilon_{95} / \epsilon_{\rm CCJ24\ Fig.\,8}$')
a2.set_ylim(0.92, 1.12)
a2.legend(fontsize=LEGEND_SIZE - 1, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
ax2 = a2.secondary_xaxis('top')
ax2.set_xticks([1.5e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7])
ax2.set_xticklabels([f'{GMap(*xh, P18).gpe2(m)[1]:.0f}' for m in [1.5e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]],
                    fontsize=LEGEND_SIZE)
ax2.set_xlabel(r'$z_{\rm res}$', fontsize=LEGEND_SIZE + 1)
for ext in ('pdf', 'png'):
    fig.savefig(ROOT / f'dev/figures/dp_ccj24_recfast_limits.{ext}', bbox_inches='tight', dpi=200)
