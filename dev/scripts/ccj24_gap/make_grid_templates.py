"""PDE dark-photon templates on a mass grid, run at eps = AxionLimits CCJ24 curve.

Output: grid_templates.npz with, per mass, the G_bb-stripped (number-conserving)
Delta n per unit gamma_con on the solver grid. Used by
notebooks/observational/dp_firas_limit_conventions.ipynb.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from spectroxide import g_bb
from spectroxide.solver import solve
from spectroxide.dark_photon import gc_per_epsilon_sq

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
AL = np.loadtxt(ROOT / 'dev/AxionLimits/limit_data/DarkPhoton/COBEFIRAS_Chluba.txt')
AL = AL[AL[:, 0] <= 1.5e-4]  # drop the contour-closure row at m ~ 1e-3 eV


def eps_ccj(m):
    return 10**np.interp(np.log10(m), np.log10(AL[:, 0]), np.log10(AL[:, 1]))


def run(m):
    gpe2, zr = gc_per_epsilon_sq(m)
    eps = eps_ccj(m)
    gc = gpe2 * eps**2
    r = solve(injection={'type': 'dark_photon_resonance', 'epsilon': eps, 'm_ev': m},
              z_end=100, n_points=4000 if zr < 1e6 else 8000, timeout=3000)
    x, dn = r.x, r.delta_n
    alpha = np.trapz(x**2 * dn, x) / np.trapz(x**2 * g_bb(x), x)
    print(f'm = {m:.3e} eV  z_res = {zr:.3g}  gamma_con = {gc:.2e}', flush=True)
    return m, zr, gpe2, eps, x, (dn - alpha * g_bb(x)) / gc


masses = np.geomspace(1.5e-12, 8e-5, 30)
with ThreadPoolExecutor(max_workers=6) as pool:
    out = list(pool.map(run, masses))
np.savez(HERE / 'grid_templates.npz',
         m=np.array([o[0] for o in out]), zr=np.array([o[1] for o in out]),
         gpe2=np.array([o[2] for o in out]), eps_ref=np.array([o[3] for o in out]),
         x=np.array([o[4] for o in out], dtype=object),
         dn_per_gc=np.array([o[5] for o in out], dtype=object))
print('saved', HERE / 'grid_templates.npz')
