"""Follow-up to postrec_gap.py (record: dev/audit/ccj24_postrec_gap.md, section "2026-09-25 follow-up").

No PDE runs. Three checks:
1. CCJ24 Fig. 2 (z_con vs m, eps/con_plot.pdf in the arXiv:2409.12115 source) against our
   resonance redshift, and the eps ratio that Fig. 2 would imply if it were the Fig. 8 mapping.
   Regenerate the curve with `python postrec_followup.py --extract /path/to/con_plot.pdf`.
2. Per-mass eps_ours / eps_CCJ24 (Fig. 8 vector curve, Delta chi^2 = 4, normalized to
   z_res 3e3-5e4) for our X_e, HyRec-2020 and RecfastCLASS, all at Planck 2018.
3. Attribution by splicing: our X_e with HyRec above z = 1550 (helium), HyRec with ours
   above z = 800 (low-z hydrogen).
"""
import re
import subprocess
import sys

import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import brentq

from postrec_gap import EV, HBAR, HERE, KB, ROOT, X0, GMap, gamma_limit, xe_class, xe_ours
from spectroxide.cosmology import DEFAULT_COSMO, PLANCK2018_COSMO, _cosmo_hubble

P18 = PLANCK2018_COSMO


def extract_fig2(pdf):
    """Blue z_con(m) path of CCJ24 Fig. 2. Axes: m 1e-13..1e-1 eV, z 1e1..1e9."""
    svg = subprocess.run(['pdftocairo', '-svg', str(pdf), '-'], capture_output=True, text=True).stdout
    x0, x1, y0, y1 = 45.872794, 402.992558, 39.200291, 305.310246   # frame, from the tick paths
    for p in re.findall(r'<path ([^>]*?)/?>', svg):
        if 'stroke="rgb(12.156677%' in p:
            v = np.array([float(t) for t in re.findall(r'-?\d+\.?\d*', re.search(r' d="([^"]*)"', p).group(1))])
            v = v.reshape(-1, 2)
    m = 10**(-13 + 12 * (v[:, 0] - x0) / (x1 - x0))
    z = 10**(1 + 8 * (v[:, 1] - y0) / (y1 - y0))
    np.savetxt(HERE / 'ccj24_zcon_vector.txt', np.column_stack([m, z]),
               header='m [eV]  z_con   CCJ24 Fig. 2 (con_plot.pdf), vector path')


def fig2_mapping():
    """gamma_con/eps^2 implied by Fig. 2: n_e(z_con) = m^2 m_e/(4 pi alpha), slope from a smoothed curve.
    The path is a staircase in z (steps of 1.5%), so collapse each z level to its mean ln m first."""
    d = np.loadtxt(HERE / 'ccj24_zcon_vector.txt')
    lz, lm = np.round(np.log1p(d[:, 1]), 6), np.log(d[:, 0])
    U = np.unique(lz)
    LM = np.array([lm[lz == u].mean() for u in U])
    sp = UnivariateSpline(U, LM, s=len(U) * 0.01**2, k=4)
    def gpe2(m, c):
        x = brentq(lambda t: sp(t) - np.log(m), U[0], U[-1])
        zr = np.expm1(x)
        return np.pi * m**2 / (2 * sp.derivative()(x) * KB * c['t_cmb'] * (1 + zr) / EV
                               * HBAR / EV * _cosmo_hubble(zr, c)), zr
    return d, gpe2


def splice(lo, hi, zcut):
    (za, xa), (zb, xb) = lo, hi
    return np.concatenate([za[za < zcut], zb[zb >= zcut]]), np.concatenate([xa[za < zcut], xb[zb >= zcut]])


def main():
    if '--extract' in sys.argv:
        extract_fig2(sys.argv[sys.argv.index('--extract') + 1])
    TM = np.load(HERE / 'grid_templates.npz', allow_pickle=True)
    mt = TM['m'].astype(float)
    gl = np.array([gamma_limit(np.interp(X0, TM['x'][i], TM['dn_per_gc'][i]))[0] for i in range(len(mt))])
    glim = lambda m: np.exp(np.interp(np.log(m), np.log(mt), np.log(gl)))
    VEC = np.loadtxt(HERE / 'ccj24_fig8_firas_vector.txt')
    ev = lambda m: 10**np.interp(np.log10(m), np.log10(VEC[:, 0]), np.log10(VEC[:, 1]))

    xo, xh = xe_ours(P18), xe_class(P18)
    ours, hyrec = GMap(*xo, P18), GMap(*xh, P18)

    print('1. CCJ24 Fig. 2 z_con against our z_res (Planck 2018)')
    d, g2 = fig2_mapping()
    zc = lambda m: 10**np.interp(np.log10(m), np.log10(d[:, 0]), np.log10(d[:, 1]))
    lm = lambda u: 10**(-13 + 12 * (u - 45.872794) / (402.992558 - 45.872794))
    print(f'   z_eq marker: m = {lm(180.998134):.3e} eV, Fig. 2 z_con there = {zc(lm(180.998134)):.0f}')
    for name, c in [('default', DEFAULT_COSMO), ('Planck 2018', P18)]:
        orh2 = 2.4728e-5 * (c['t_cmb'] / 2.7255)**4 * (1 + 0.2271 * c['n_eff'])
        print(f'   z_eq {name}: {c["omega_m"] * c["h"]**2 / orh2 - 1:.0f}')
    ms = np.geomspace(1.5e-12, 4e-7, 60)
    yb = lambda zr: (zr > 3e3) & (zr < 5e4)
    rows = np.array([(m, g2(m, P18)[1], hyrec.gpe2(m)[1], np.sqrt(g2(m, P18)[0] / hyrec.gpe2(m)[0])) for m in ms])
    n = np.median(rows[yb(rows[:, 2]), 3])
    print('   m [eV]   z_con(Fig.2)  z_res(HyRec)  eps_ours/eps_CCJ if Fig. 2 were the Fig. 8 mapping')
    for r in rows[::4]:
        print(f'   {r[0]:.2e} {r[1]:9.0f} {r[2]:11.0f}   {r[3] / n:.4f}')

    print('\n2. eps_ours / eps_CCJ24 Fig. 8 vector curve, Delta chi2 = 4, normalized to z_res 3e3-5e4')
    M = {'ours': ours, 'HyRec': hyrec, 'Recfast': GMap(*xe_class(P18, 'RECFAST'), P18),
         'ours+HyRec(z>1550)': GMap(*splice(xo, xh, 1550), P18),
         'HyRec+ours(z>800)': GMap(*splice(xh, xo, 800), P18)}
    ms = np.geomspace(1.5e-12, 4e-7, 120)
    E = {k: np.array([np.sqrt(glim(m) / g.gpe2(m)[0]) for m in ms]) / ev(ms) for k, g in M.items()}
    zr = np.array([ours.gpe2(m)[1] for m in ms])
    E = {k: v / np.median(v[yb(zr)]) for k, v in E.items()}
    print('   m [eV]    z_res  ' + ''.join(f'{k:>20s}' for k in M))
    for i in range(0, len(ms), 3):
        print(f'   {ms[i]:.2e} {zr[i]:6.0f}  ' + ''.join(f'{E[k][i]:20.4f}' for k in M))
    print('\n   band medians')
    for lo, hi in [(250, 800), (800, 1550), (1550, 2700), (2700, 4800), (4800, 7000), (7000, 5e4)]:
        s = (zr > lo) & (zr <= hi)
        print(f'   z_res {lo:>5}-{hi:<6.0f}' + ''.join(f'{np.median(E[k][s]):20.4f}' for k in M)
              + f'   (min/max ours {E["ours"][s].min():.4f}/{E["ours"][s].max():.4f})')

    print('\n3. X_e and s = dln X_e/dln(1+z), ours against HyRec-2020 (Planck 2018)')
    so, sh = CubicSpline(np.log1p(xo[0]), np.log(xo[1])), CubicSpline(np.log1p(xh[0]), np.log(xh[1]))
    for z in [200, 277, 350, 450, 600, 800, 1000, 1500, 1700, 1900, 2100, 2300, 2600, 3000, 6000]:
        L = np.log1p(z)
        print(f'   z = {z:5d}  X_e ours/HyRec = {np.exp(so(L) - sh(L)):.4f}   s ours {so(L, 1):7.3f}  HyRec {sh(L, 1):7.3f}')


if __name__ == '__main__':
    main()
