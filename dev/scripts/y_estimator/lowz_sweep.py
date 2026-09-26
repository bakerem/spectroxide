"""Extend the Table 1 PDE spectra (dev/data/visibility_table.npz, z_h >= 3e3) down to z_h = 1e3.

Same settings as dev/scripts/build_visibility_table.py (single burst, n_points = 4000, z_end = 500),
except the amplitude: at drho = 1e-5 the runs at z_h <= 1200 hit the T_e cap and lose up to 69% of
the heat. Here each z_h runs at two amplitudes below the cap, and the stored spectrum per unit drho is
their difference quotient, which also cancels the adiabatic-cooling baseline. Usage: lowz_sweep.py out.npz
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/bakerem/spectroxide/python")
from spectroxide import run_sweep

A1, A2, N_POINTS = 2e-8, 1e-8, 4000
z_h = np.array([1000.0, 1200.0, 1450.0, 1750.0, 2100.0, 2500.0, 3000.0])  # 3000 overlaps the table
t0 = time.time()
def spectra(a):
    res = run_sweep(delta_rho=a, z_injections=z_h.tolist(), z_end=500.0, n_points=N_POINTS,
                    timeout=3600, n_threads=2)["results"]
    return np.array(res[0]["x"]), np.array([r["delta_n"] for r in res])
x, d1 = spectra(A1)
_, d2 = spectra(A2)
dn = (d1 - d2) / (A1 - A2)
np.savez(sys.argv[1], z_h=z_h, x=x, dn_raw=dn, n_points=N_POINTS)
print(f"done in {time.time() - t0:.0f} s; saved {sys.argv[1]}")
