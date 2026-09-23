"""Post-recombination part of the dark-photon FIRAS gap to CCJ24 (record: dev/audit/ccj24_postrec_gap.md).

No PDE runs. Uses the 30 PDE templates in grid_templates.npz, the CCJ24-literal statistic
(diagonal errors, residual column, G and dust projected from the model, limit at
Delta chi^2 = 4 above the no-distortion model), and swaps one ingredient of the
eps -> gamma_con mapping at a time.

Inputs
- grid_templates.npz (make_grid_templates.py).
- ccj24_fig8_firas_vector.txt: the solid blue FIRAS curve of CCJ24 Fig. 8, read from the
  vector paths of eps/eps_limits_new.pdf in the arXiv:2409.12115 source. Regenerate with
  `python postrec_gap.py --extract /path/to/eps_limits_new.pdf` (needs pdftocairo).
- classy (CLASS v3.2 Python wrapper) for HyRec-2020 and RecfastCLASS X_e(z).

Every ratio is eps_ours / eps_CCJ24 at the same statistic, divided by its own median over
z_res in (3e3, 5e4], so the statistic's overall level (Delta chi^2 = 3.84 or 4) drops out:
the template gives the same k = a_hat/sigma_a = -0.348 at every mass below 5e-8 eV.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from spectroxide.cosmology import (DEFAULT_COSMO, PLANCK2015_COSMO, PLANCK2018_COSMO,
                                   _cosmo_hubble, _cosmo_n_h, ionization_fraction)
from spectroxide.dark_photon import gc_per_epsilon_sq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ALPHA, ME, C, HBAR, KB, EV = (7.2973525693e-3, 9.1093837015e-31, 2.99792458e8,
                              1.054571817e-34, 1.380649e-23, 1.602176634e-19)


def extract_fig8(pdf):
    """Solid blue path of CCJ24 Fig. 8 -> (m, eps). Axes: m 1e-12..1e-3, eps 1e-10..1e-4."""
    svg = subprocess.run(['pdftocairo', '-svg', str(pdf), '-'], capture_output=True, text=True).stdout
    x0, x1, y0, y1 = 768.085938, 8777.890625, 676.71875, 5772.226562   # frame, from the tick paths
    best = np.zeros((0, 2))
    for p in re.findall(r'<path ([^>]*)/>', svg):
        if 'stroke="rgb(0%, 0%, 100%)"' in p and 'dasharray' not in p:
            v = np.array([float(t) for t in re.findall(r'-?\d+\.?\d*', re.search(r' d="([^"]*)"', p).group(1))])
            v = v.reshape(-1, 2)
            if len(v) > len(best):
                best = v
    m = 10**(-12 + 9 * (best[:, 0] - x0) / (x1 - x0))
    e = 10**(-10 + 6 * (best[:, 1] - y0) / (y1 - y0))
    np.savetxt(HERE / 'ccj24_fig8_firas_vector.txt', np.column_stack([m, e]),
               header='m [eV]  eps   CCJ24 Fig. 8 FIRAS CosmoTherm curve (solid blue), vector PDF')


# ---------------------------------------------------------------- FIRAS, CCJ24-literal statistic
f_cm, _, RESID, SIGMA, GAL = np.loadtxt(ROOT / 'data/firas_monopole_spec_v1.txt').T
H_PL, K_B = 6.62607015e-34, 1.380649e-23
NU = f_cm * C * 100
X0 = H_PL * NU / (K_B * 2.725)
PREF = 2 * H_PL * NU**3 / C**2 / 1e-23
NUIS = np.column_stack([PREF * X0 * np.exp(X0) / np.expm1(X0)**2, GAL])
CI = np.diag(1 / SIGMA**2)


def gamma_limit(dn_at_x0, t=4.0):
    s = PREF * dn_at_x0
    s = s - NUIS @ np.linalg.solve(NUIS.T @ CI @ NUIS, NUIS.T @ CI @ s)
    sig = 1 / np.sqrt(s @ CI @ s)
    k = (RESID @ CI @ s) * sig
    return (k + np.sqrt(k**2 + t) if k < 0 else k + np.sqrt(t)) * sig, k


# ---------------------------------------------------------------- eps -> gamma_con mapping
class GMap:
    """gamma_con / eps^2 (CCJ24 Eq. 6) for a given X_e(z) table and cosmology."""

    def __init__(self, z, xe, cosmo, ne_scale=1.0):
        self.c, self.k = cosmo, ne_scale
        self.s = CubicSpline(np.log1p(z), np.log(xe))
        self.zlo, self.zhi = z.min(), z.max()

    def gpe2(self, m):
        wpl = lambda z: HBAR / EV * np.sqrt(4 * np.pi * ALPHA * HBAR * C / ME * self.k
                                            * np.exp(self.s(np.log1p(z))) * _cosmo_n_h(z, self.c))
        zr = brentq(lambda z: wpl(z) - m, max(self.zlo, 11), 0.999 * self.zhi, xtol=1e-10, rtol=1e-13)
        d = abs(3.0 + self.s(np.log1p(zr), 1))
        return np.pi * m**2 / (d * KB * self.c['t_cmb'] * (1 + zr) / EV
                               * HBAR / EV * _cosmo_hubble(zr, self.c)), zr


def xe_ours(c):
    z = np.concatenate([np.arange(10, 9000, 1.0), np.geomspace(9000, 2e7, 3000)])
    return z, ionization_fraction(z, c)


def xe_class(c, rec='HyRec'):
    import classy
    M = classy.Class()
    M.set({'h': c['h'], 'omega_b': c['omega_b'] * c['h']**2, 'omega_cdm': (c['omega_m'] - c['omega_b']) * c['h']**2,
           'T_cmb': c['t_cmb'], 'YHe': c['y_p'], 'N_ur': c['n_eff'], 'N_ncdm': 0,
           'recombination': rec, 'reio_parametrization': 'reio_none', 'output': ''})
    M.compute()
    th = M.get_thermodynamics()
    z, xe = th['z'], th['x_e']
    M.struct_cleanup()
    o = np.argsort(z)
    z, xe = z[o], xe[o]
    keep = np.concatenate([[True], np.diff(z) > 0]) & (z > 5)
    return z[keep], xe[keep]


def main():
    if '--extract' in sys.argv:
        extract_fig8(sys.argv[sys.argv.index('--extract') + 1])
    TM = np.load(HERE / 'grid_templates.npz', allow_pickle=True)
    mt = TM['m'].astype(float)
    dn = [np.interp(X0, TM['x'][i], TM['dn_per_gc'][i]) for i in range(len(mt))]
    gl = np.array([gamma_limit(d)[0] for d in dn])
    glim = lambda m: np.interp(np.log(m), np.log(mt), gl)

    print('1. gamma_con limit per PDE template (Delta chi2 = 4), and PDE / analytic Eq. 25 template')
    npl = 1 / np.expm1(X0)
    g_an = gamma_limit(-npl / X0)[0]
    for i in range(0, 18):
        print(f'   m = {mt[i]:.3e}  z_res = {TM["zr"][i]:7.0f}  gamma_lim = {gl[i]:.4e}  PDE/analytic = {gl[i] / g_an:.4f}'
              f'  k = {gamma_limit(dn[i])[1]:.3f}')

    AL = np.loadtxt(ROOT / 'dev/AxionLimits/limit_data/DarkPhoton/COBEFIRAS_Chluba.txt')
    AL = AL[(AL[:, 0] > 1.4e-12) & (AL[:, 0] < 4.6e-5)]
    VEC = np.loadtxt(HERE / 'ccj24_fig8_firas_vector.txt')
    ev = lambda m: 10**np.interp(np.log10(m), np.log10(VEC[:, 0]), np.log10(VEC[:, 1]))
    D, P18 = DEFAULT_COSMO, PLANCK2018_COSMO
    base = GMap(*xe_ours(D), D)
    print('\n   check: this mapping / spectroxide.dark_photon.gc_per_epsilon_sq =',
          ' '.join(f'{base.gpe2(m)[0] / gc_per_epsilon_sq(m)[0]:.4f}' for m in (3e-12, 1e-10, 1e-9, 1e-7)))
    V = {'Peebles, default cosmo (ours)': base,
         'HyRec-2020, default': GMap(*xe_class(D), D),
         'RecfastCLASS, default': GMap(*xe_class(D, 'RECFAST'), D),
         'Peebles, T0 = 2.7255': GMap(*xe_ours({**D, 't_cmb': 2.7255}), {**D, 't_cmb': 2.7255}),
         'Peebles, Planck 2018': GMap(*xe_ours(P18), P18),
         'HyRec-2020, Planck 2015': GMap(*xe_class(PLANCK2015_COSMO), PLANCK2015_COSMO),
         'HyRec-2020, Planck 2018': GMap(*xe_class(P18), P18)}
    zp18 = xe_class(P18)
    for s in (1.02, 1.0789):
        V[f'HyRec, Planck 2018, n_e x {s}'] = GMap(*zp18, P18, ne_scale=s)
    bands = [(250, 800), (800, 1100), (1100, 1450), (1450, 3000), (3000, 5e4)]
    head = ''.join(f'{f"{a:.0f}-{b:.0f}":>11}' for a, b in bands[:-1])
    for name, ms, ef in (('AxionLimits file', AL[:, 0], lambda m: np.interp(np.log(m), np.log(AL[:, 0]), np.log(AL[:, 1]))),
                         ('Fig. 8 vector curve', VEC[(VEC[:, 0] > 1.4e-12) & (VEC[:, 0] < 4.6e-5), 0], lambda m: np.log(ev(m)))):
        zr = np.array([base.gpe2(m)[1] for m in ms])
        print(f'\n2. eps_ours/eps_CCJ vs {name} ({len(ms)} masses), normalized to z_res 3e3-5e4; last column = that median')
        print(f'   {"mapping":34s}{head}     norm')
        for k, g in V.items():
            r = np.array([np.sqrt(glim(m) / g.gpe2(m)[0]) / np.exp(ef(m)) for m in ms])
            med = [np.median(r[(zr > a) & (zr <= b)]) for a, b in bands]
            print(f'   {k:34s}' + ''.join(f'{x / med[-1]:11.4f}' for x in med[:-1]) + f'   {med[-1]:.4f}')
        print('   masses per band', [int(((zr > a) & (zr <= b)).sum()) for a, b in bands])

    zr = np.array([base.gpe2(m)[1] for m in AL[:, 0]])
    r = AL[:, 1] / ev(AL[:, 0])
    print('\n3. AxionLimits file / Fig. 8 vector curve, median [min, max] by band')
    for a, b in bands + [(5e4, 1e6)]:
        s = (zr > a) & (zr <= b)
        print(f'   z_res {a:.0f}-{b:.0f}: {np.median(r[s]):.4f} [{r[s].min():.4f}, {r[s].max():.4f}]  ({s.sum()} nodes)')

    print('\n4. Coarse grids in CCJ24 (20,000 models): bias of the gridded curve')
    g = V['HyRec-2020, Planck 2018']
    eps = lambda m: np.sqrt(9.3e-5 / g.gpe2(m)[0])
    mf = np.geomspace(1.5e-12, 4e-9, 400)
    ef_ = np.array([eps(m) for m in mf])
    zf = np.array([g.gpe2(m)[1] for m in mf])
    for N in (100, 141, 200):
        mg = np.geomspace(1e-12, 1e-3, N)
        mg = mg[mg < 5e-9]
        ei = 10**np.interp(np.log10(mf), np.log10(mg), np.log10([eps(m) for m in mg]))
        rr = ef_ / ei
        print(f'   mass grid N = {N}: exact/gridded median by band',
              ' '.join(f'{np.median(rr[(zf > a) & (zf <= b)]):.4f}' for a, b in bands[:-1]))
    k = -0.348
    ut = k + np.sqrt(k**2 + 4)
    for N in (100, 141, 200):
        grid = np.geomspace(1e-10, 3e-5, N)

        def bias(et):
            y = (ut * (grid / et)**2 - k)**2 - k**2 - 4
            j = np.nonzero(y >= 0)[0][0]
            return np.exp(np.interp(0, y[j - 1:j + 1], np.log(grid[j - 1:j + 1]))) / et
        b_post = np.mean([bias(e) for e in np.geomspace(4e-8, 4e-6, 2001)])
        b_y = np.mean([bias(e) for e in np.geomspace(3.85e-8, 4.1e-8, 50)])
        print(f'   eps grid N = {N}: log-interpolated crossing / exact, post-rec average {b_post:.4f}, y-era {b_y:.4f}')

    # 5. CCJ24's gamma*_con = 1e-4 contour (their Fig. 7, eps/gamma_con_contour.pdf) against their Fig. 8.
    # With a mass-independent gamma_con limit, eps_Fig8 / eps_contour must be flat in mass.
    pdf = Path('/tmp/claude-1000/ccj24_src/eps/gamma_con_contour.pdf')
    if pdf.exists():
        svg = subprocess.run(['pdftocairo', '-svg', str(pdf), '-'], capture_output=True, text=True).stdout
        lines = []
        for p in re.findall(r'<path ([^>]*)/>', svg):
            if 'stroke="rgb(0%, 0%, 100%)"' in p:
                a, _, _, d_, _, f = [float(t) for t in re.search(r'matrix\(([^)]*)\)', p).group(1).split(',')]
                v = np.array([float(t) for t in re.findall(r'-?\d+\.?\d*', re.search(r' d="([^"]*)"', p).group(1))])
                v = v.reshape(-1, 2)
                px, py = a * v[:, 0], f + d_ * v[:, 1]
                lines.append(np.column_stack([10**(-12 + 9 * (px - 55.765625) / (412.199219 - 55.765625)),
                                              10**(-10 + 6 * (276.78125 - py) / (276.78125 - 11.179688))]))
        lines.sort(key=lambda L: L[-1, 1])
        L = lines[1]   # gamma* = 1e-4
        print('\n5. CCJ24 Fig. 8 / their own gamma* = 1e-4 contour (flat if both use one mapping)')
        print('   ' + ' '.join(f'{m:.0e}:{ev(m) / 10**np.interp(np.log10(m), np.log10(L[:, 0]), np.log10(L[:, 1])):.3f}'
                             for m in (2e-12, 1e-11, 5e-11, 1e-10, 3e-10, 6e-10, 1e-9, 3e-9, 1e-8, 1e-7, 1e-6, 1e-5)))


if __name__ == '__main__':
    main()
