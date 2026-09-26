import os
import numpy as np, glob
from stats import *
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ccj = np.loadtxt(os.path.join(REPO_ROOT, 'dev/AxionLimits/limit_data/DarkPhoton/COBEFIRAS_Chluba.txt'))
ccj = ccj[ccj[:, 0] <= 1.5e-4]  # drop the contour-closure row at m ~ 1e-3 eV
epub = lambda m: 10**np.interp(np.log10(m), np.log10(ccj[:,0]), np.log10(ccj[:,1]))
steps = [
 ('S0 CCJ24: diag, resid, T0 fixed, floor a>=0', dict()),
 ('S1 drop a>=0 floor', dict(clip='ul')),
 ('S2 + full cov', dict(clip='ul', cov='full')),
 ('S3 + col2 spectrum data', dict(clip='ul', cov='full', dkind='spec')),
 ('S4 + floating T (=paper)', dict(clip='ul', cov='full', dkind='spec', T='float')),
]
alt = [
 ('A1 CCJ24 + floating T only (floor kept)', dict(T='float', dkind='spec')),
 ('A2 CCJ24 + full cov only (floor kept)', dict(cov='full')),
 ('A3 paper with floor a>=0', dict(cov='full', dkind='spec', T='float')),
]
for f in sorted(glob.glob('tmpl_*.npz'), key=lambda s: float(s[5:-4])):
    d = np.load(f); m = float(d['m']); gp = float(d['gpe2'])
    xx, dn = d['x'], d['dn_per_gc']
    for lab, tmpl in [('PDE', lambda x: np.interp(x, xx, dn)), ('Eq25', analytic(float(d['zr'])))]:
        print(f'\n=== m={m:.1e} eV z_con={float(d["zr"]):.3g} template={lab}  eps_pub={epub(m):.3e}')
        prev = None; e0 = None
        for name, kw in steps + alt:
            r = limit(tmpl, **kw); e = np.sqrt(r['ul']/gp)
            if name.startswith('S0'): e0 = e
            fac = '' if prev is None or name[0] == 'A' else f'{e/prev:6.3f}'
            print(f'  {name:44s} eps={e:.3e} a/sig={r["a"]/r["s"]:+.3f} sig_eps2={r["s"]/gp:.3e} T={r["T"]:.6f} step={fac:>6s} cum={e/e0:.3f} /pub={e/epub(m):.3f}')
            if name[0] == 'S': prev = e
