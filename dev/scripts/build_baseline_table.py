#!/usr/bin/env python3
"""Build the no-injection (adiabatic-cooling) baseline table.

Pipeline (paper Table 1): build_visibility_table.py -> build_baseline_table.py
(this script) -> fit_visibility_conservation.py.
Input: dev/data/visibility_table.npz (for the z_h grid; from
build_visibility_table.py).
Output: dev/data/baseline_table.npz (consumed by fit_visibility_conservation.py).

Runs the PDE with a negligible injection (delta_rho = 1e-12) at every z_h of
the visibility table with z_h >= 2e5, using the identical grid and z_start
logic as the injected sweep (z_start = z_h + 7 sigma).  The resulting spectra
are pure adiabatic-cooling distortions; the 1e-12 injected signal is ~1e-7 of
the cooling amplitude and irrelevant.

The baseline must be subtracted from the injected spectra before extracting
the thermalization visibility via conservation laws: the cooling distortion
contributes mu_cons ~ -5.8e-9 (absolute), i.e. -5.8e-4 per unit injected
energy at delta_rho = 1e-5, which floors the visibility extraction at
z_h >~ 4e6.

Output: dev/data/baseline_table.npz with z_h, x, dn_baseline (absolute Dn,
NOT divided by delta_rho), and the scalar mu_cons/drho/dN of each baseline.
"""
import sys
import pathlib
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "python"))

from spectroxide import run_sweep
from spectroxide.greens import KAPPA_C, G2_PLANCK, G3_PLANCK

N_POINTS = 4000
Z_MIN = 2e5

table = np.load(pathlib.Path(__file__).resolve().parent.parent / "data" / "visibility_table.npz")
z_all = table["z_h"]
z_run = z_all[z_all >= Z_MIN]

# Optional chunking: build_baseline_table.py <i0> <i1> processes z_run[i0:i1]
# and writes baseline_table_chunk_<i0>_<i1>.npz (merge with --merge).
if len(sys.argv) == 2 and sys.argv[1] == "--merge":
    chunks = sorted((pathlib.Path(__file__).resolve().parent.parent / "data").glob(
        "baseline_table_chunk_*.npz"))
    zs, dns, mus, drs, dNs = [], [], [], [], []
    for c in chunks:
        d = np.load(c)
        zs.append(d["z_h"]); dns.append(d["dn_baseline"])
        mus.append(d["mu_cons"]); drs.append(d["drho"]); dNs.append(d["dN"])
        x_merge = d["x"]
    z_cat = np.concatenate(zs)
    order = np.argsort(z_cat)
    outpath = pathlib.Path(__file__).resolve().parent.parent / "data" / "baseline_table.npz"
    np.savez(outpath, z_h=z_cat[order], x=x_merge,
             dn_baseline=np.concatenate(dns)[order],
             mu_cons=np.concatenate(mus)[order],
             drho=np.concatenate(drs)[order],
             dN=np.concatenate(dNs)[order], n_points=N_POINTS)
    print(f"Merged {len(chunks)} chunks ({len(z_cat)} redshifts) -> {outpath}")
    sys.exit(0)

chunk_tag = ""
if len(sys.argv) == 3:
    i0, i1 = int(sys.argv[1]), int(sys.argv[2])
    z_run = z_run[i0:i1]
    chunk_tag = f"_chunk_{i0:02d}_{i1:02d}"
print(f"Baseline sweep: {len(z_run)} redshifts, z in [{z_run[0]:.3e}, {z_run[-1]:.3e}]")

t0 = time.time()
sweep = run_sweep(
    delta_rho=1e-12,
    z_injections=z_run.tolist(),
    z_end=500.0,
    n_points=N_POINTS,
    timeout=14000,
)
print(f"Done in {time.time() - t0:.0f}s")

results = sweep["results"]
x = np.array(results[0]["x"])
dn_base = np.zeros((len(z_run), len(x)))
mu_base = np.zeros(len(z_run))
drho_base = np.zeros(len(z_run))
dN_base = np.zeros(len(z_run))

for k, r in enumerate(results):
    dn = np.array(r["delta_n"])
    dn_base[k, :] = dn
    dr = np.trapz(x**3 * dn, x) / G3_PLANCK
    dN = np.trapz(x**2 * dn, x) / G2_PLANCK
    drho_base[k] = dr
    dN_base[k] = dN
    mu_base[k] = (3 * dr - 4 * dN) / KAPPA_C

outpath = (pathlib.Path(__file__).resolve().parent.parent / "data"
           / f"baseline_table{chunk_tag}.npz")
np.savez(outpath, z_h=z_run, x=x, dn_baseline=dn_base,
         mu_cons=mu_base, drho=drho_base, dN=dN_base, n_points=N_POINTS)
print(f"Saved: {outpath}")
for k in range(0, len(z_run), 6):
    print(f"  z_h={z_run[k]:.3e}  mu_cons_abs={mu_base[k]:.4e}  drho_abs={drho_base[k]:.4e}")
