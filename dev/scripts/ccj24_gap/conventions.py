import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from stats import *
def convs(a, s):
    k = a/s
    out = {}
    out['floor: max(a,0)+1.645s'] = max(k,0)+1.645
    out['unfloored a+1.645s (paper)'] = k+1.645
    out['Bayes flat prior >=0, 95%'] = k+norm.ppf(1-0.05*norm.cdf(k))  # truncated Gaussian
    out['chi2(g)-chi2(0)=2.71'] = k+np.sqrt(k*k+2.71) if k<0 else k+1.645
    out['chi2(g)-chi2(0)=3.84'] = k+np.sqrt(k*k+3.84) if k<0 else k+1.96
    out['Feldman-Cousins-like: 1.645 at k=0 bound'] = None
    return {kk: v for kk, v in out.items() if v is not None}
ccj = np.loadtxt('/home/bakerem/spectroxide/dev/data/cosmotherm_dp_lims.csv', delimiter=',')
epub = lambda m: 10**np.interp(np.log10(m), np.log10(ccj[:,0]), np.log10(ccj[:,1]))
import glob
for f in sorted(glob.glob('tmpl_*.npz'), key=lambda s: float(s[5:-4])):
    d=np.load(f); m=float(d['m']); gp=float(d['gpe2'])
    r = limit(lambda x: np.interp(x, d['x'], d['dn_per_gc']))
    print(f'm={m:.1e}: eps/pub under CCJ24 statistic (diag, T0, resid) with each threshold:')
    for k,v in convs(r['a'], r['s']).items():
        print(f'   {k:38s} {np.sqrt(v*r["s"]/gp)/epub(m):.3f}')
from spectroxide import mu_shape
for nm,t,conv in [('mu',mu_shape,1/1.401),('D (Eq26)',D,0.5421)]:
    r = limit(t)
    print(nm, 'Delta rho/rho 95% limits:', {k: f'{v*r["s"]*conv:.2e}' for k,v in convs(r['a'],r['s']).items()})
