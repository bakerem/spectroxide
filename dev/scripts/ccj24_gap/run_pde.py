import sys, numpy as np
from spectroxide import g_bb
from spectroxide.solver import solve
from spectroxide.dark_photon import gc_per_epsilon_sq
ccj = np.loadtxt('/home/bakerem/spectroxide/dev/data/cosmotherm_dp_lims.csv', delimiter=',')
def eps_pub(m):
    return 10**np.interp(np.log10(m), np.log10(ccj[:,0]), np.log10(ccj[:,1]))
m = float(sys.argv[1])
gpe2, zr = gc_per_epsilon_sq(m)
eps = eps_pub(m); gc = gpe2*eps**2
r = solve(injection={'type':'dark_photon_resonance','epsilon':eps,'m_ev':m},
          z_end=100, n_points=4000 if zr < 1e6 else 8000, timeout=3000)
x, dn = r.x, r.delta_n
alpha = np.trapz(x**2*dn, x)/np.trapz(x**2*g_bb(x), x)
np.savez(f'tmpl_{m:.1e}.npz', x=x, dn_per_gc=(dn-alpha*g_bb(x))/gc, m=m, zr=zr, gpe2=gpe2, eps_ref=eps, gc_ref=gc)
print(m, zr, gc, 'done')
