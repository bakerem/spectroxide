#!/usr/bin/env python3
"""Export the frozen cosmological ingredient table for the R3 reference solver.

Workstream R3 (dev/PLAN_VALIDATION_ROUND2_2026-07-06.md). The clean-room
Chang-Cooper reference solver consumes THIS table rather than re-deriving
recombination, so the cross-check isolates the PDE numerics (recombination has
its own external anchor, HyRec-2, Round 1).

Columns: z, x_e, H_z [1/s], n_e [1/m^3], n_H [1/m^3], T_gamma [K], t_C [s]
  x_e     = ionization fraction (Peebles + Saha, same as the PDE)
  H_z     = Hubble rate
  n_e     = free-electron number density = x_e * (n_H + He electrons)
  n_H     = hydrogen number density
  T_gamma = photon temperature = T_cmb (1+z)
  t_C     = Thomson time 1/(sigma_T n_e c)  [the PDE's Compton-time unit]

Cosmology: Cosmology.default() = Chluba 2013 defaults
  h=0.71, Omega_b=0.044, Omega_m=0.26, Y_p=0.24, T_cmb=2.726 K, N_eff=3.046.

Regenerate:  python dev/refsolver/inputs/export_history.py
"""
import numpy as np
import spectroxide.cosmology as cosmo

# CODATA (match src/constants.rs)
SIGMA_THOMSON = 6.652_458_7321e-29  # m^2
C_LIGHT = 2.997_924_58e8  # m/s

C = cosmo.Cosmology()  # default = Chluba 2013

# Log-spaced z grid, dense, spanning the full solver range down to z=1.
z = np.unique(np.concatenate([
    np.geomspace(1.0, 5e6, 4000),
    # extra density around recombination & the mu-y transition
    np.geomspace(800.0, 3000.0, 400),
    np.geomspace(3e4, 3e5, 400),
]))
z = np.sort(z)

d = C.to_dict()
x_e = np.asarray(cosmo.ionization_fraction(z, d))
H_z = np.asarray(cosmo.hubble(z, d))          # 1/s
n_H = np.asarray(cosmo.n_hydrogen(z, d))      # 1/m^3
n_e = np.asarray(cosmo.n_electron(z, d, x_e=x_e)) # 1/m^3
T_g = d["t_cmb"] * (1.0 + z)                  # K
t_C = 1.0 / (SIGMA_THOMSON * n_e * C_LIGHT)   # s

import os
out = os.path.join(os.path.dirname(__file__), "history.csv")
hdr = "z,x_e,H_z_per_s,n_e_per_m3,n_H_per_m3,T_gamma_K,t_C_s"
arr = np.column_stack([z, x_e, H_z, n_e, n_H, T_g, t_C])
np.savetxt(out, arr, delimiter=",", header=hdr, comments="", fmt="%.10e")
print(f"wrote {out}: {len(z)} rows, z in [{z.min():.3g}, {z.max():.3g}]")
print("cosmology:", d)
# spot values
for zc in (2e6, 2e5, 5e4, 1100.0, 100.0):
    i = int(np.argmin(np.abs(z - zc)))
    print(f"  z={z[i]:.4g}: x_e={x_e[i]:.4e} H={H_z[i]:.4e} n_e={n_e[i]:.4e} t_C={t_C[i]:.4e}")
