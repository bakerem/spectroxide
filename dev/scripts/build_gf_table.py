#!/usr/bin/env python3
"""Build a PDE Green's function table at CosmoTherm's exact z-grid.

Outputs:
  data/pde_gf_table_full.npz  — NumPy archive with:
    z_h       (118,)    injection redshifts (matching CosmoTherm)
    x_pde     (N,)      PDE frequency grid
    gf_dn     (N, 118)  GF in Δn per Δρ/ρ
    gf_jy     (N_ct, 118) GF in Jy/sr per Δρ/ρ, interpolated onto CT x-grid
    x_ct      (N_ct,)   CosmoTherm frequency grid
    pde_mu    (118,)    μ per Δρ/ρ
    pde_y     (118,)    y per Δρ/ρ
    drho      (118,)    energy Δρ/ρ per Δρ/ρ (should be ≈ 1.0)

Also prints a comparison table against CosmoTherm.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "python"))

import numpy as np
from spectroxide import (
    run_sweep, delta_n_to_delta_I, g_bb, mu_shape, y_shape,
    KAPPA_C, G3_PLANCK,
)
from spectroxide.cosmotherm import load_greens_database, cosmotherm_gf_to_delta_n

# ── Load CosmoTherm GF ──────────────────────────────────────────────────
z_ct, x_ct, gf_ct_jy = load_greens_database()
print(f"CosmoTherm GF: {len(z_ct)} redshifts, {len(x_ct)} freq points")
print(f"  z range: [{z_ct[0]:.2e}, {z_ct[-1]:.2e}]")

# Convert CT to Δn
gf_ct_dn = np.zeros_like(gf_ct_jy)
for j in range(len(z_ct)):
    gf_ct_dn[:, j] = cosmotherm_gf_to_delta_n(x_ct, gf_ct_jy[:, j])

# ── Build basis functions for spectral decomposition ─────────────────────
M_ct = np.array([mu_shape(xi) for xi in x_ct])
Y_ct = np.array([y_shape(xi) for xi in x_ct])
G_ct = np.array([g_bb(xi) for xi in x_ct])
mask_fit = (x_ct > 0.5) & (x_ct < 20)
w_fit = x_ct[mask_fit] ** 2

def fit_spectrum(dn, x=x_ct, M=M_ct, Y=Y_ct, G=G_ct, mf=mask_fit, w=w_fit):
    """Fit dn = mu*M + y*Y + dt*G. Returns (mu, y, dt)."""
    A = np.column_stack([M[mf]*w, Y[mf]*w, G[mf]*w])
    b = dn[mf] * w
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return coeffs

# ── Run PDE at all 118 CosmoTherm z-values ───────────────────────────────
DELTA_RHO = 1e-5
N_POINTS = 2000

print(f"\nRunning PDE solver at {len(z_ct)} redshifts (n_points={N_POINTS})...")
t0 = time.time()
sweep = run_sweep(
    delta_rho=DELTA_RHO,
    z_injections=[float(z) for z in z_ct],
    z_end=500.0,
    n_points=N_POINTS,
    timeout=7200,
)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

results = sweep["results"]
x_pde = np.array(results[0]["x"])
n_x = len(x_pde)

# ── Build tables ─────────────────────────────────────────────────────────
gf_pde_dn = np.zeros((n_x, len(z_ct)))     # Δn per Δρ/ρ, on PDE grid
gf_pde_jy_ct = np.zeros((len(x_ct), len(z_ct)))  # Jy/sr per Δρ/ρ, on CT grid
pde_mu = np.zeros(len(z_ct))
pde_y = np.zeros(len(z_ct))
pde_drho = np.zeros(len(z_ct))

nu_ct_ghz = x_ct * 1.380649e-23 * 2.726 / 6.62607015e-34 / 1e9

for k, r in enumerate(results):
    dn = np.array(r["delta_n"])
    gf = dn / DELTA_RHO
    gf_pde_dn[:, k] = gf

    # Convert to Jy/sr and interpolate onto CT freq grid
    nu_pde, di_pde = delta_n_to_delta_I(x_pde, gf)
    gf_pde_jy_ct[:, k] = np.interp(nu_ct_ghz, nu_pde, di_pde)

    pde_mu[k] = r.get("pde_mu", 0) / DELTA_RHO
    pde_y[k] = r.get("pde_y", 0) / DELTA_RHO
    pde_drho[k] = r.get("drho", 0) / DELTA_RHO

# ── Save ─────────────────────────────────────────────────────────────────
outpath = pathlib.Path(__file__).resolve().parent.parent / "data" / "pde_gf_table_full.npz"
np.savez(
    outpath,
    z_h=z_ct,
    x_pde=x_pde,
    gf_dn=gf_pde_dn,
    gf_jy=gf_pde_jy_ct,
    x_ct=x_ct,
    pde_mu=pde_mu,
    pde_y=pde_y,
    drho=pde_drho,
)
print(f"\nSaved: {outpath}")

# ── Comparison table ─────────────────────────────────────────────────────
print(f"\n{'z_h':>10s}  {'RMS_30_857':>10s}  {'mu_PDE':>10s}  {'mu_CT':>10s}  "
      f"{'mu_%dev':>10s}  {'y_PDE':>10s}  {'y_CT':>10s}  {'drho_PDE':>8s}")
print("=" * 95)

mask_freq = (nu_ct_ghz > 30) & (nu_ct_ghz < 857)

for k in range(len(z_ct)):
    z_h = z_ct[k]

    # RMS in 30-857 GHz
    peak = np.max(np.abs(gf_ct_jy[mask_freq, k]))
    if peak > 0:
        rms = np.sqrt(np.mean(((gf_pde_jy_ct[mask_freq, k] - gf_ct_jy[mask_freq, k]) / peak)**2)) * 100
    else:
        rms = 0.0

    # Fit CT spectrum
    mu_ct, y_ct_fit, dt_ct = fit_spectrum(gf_ct_dn[:, k])

    mu_dev = (pde_mu[k] - mu_ct) / max(abs(mu_ct), 1e-30) * 100 if abs(mu_ct) > 1e-10 else 0

    # Only print every ~5th entry + key transitions
    if (k % 5 == 0 or z_h > 4e5 or rms > 1.0 or k < 3 or k > len(z_ct) - 3):
        print(f"{z_h:10.3e}  {rms:9.2f}%  {pde_mu[k]:10.4f}  {mu_ct:10.4f}  "
              f"{mu_dev:9.1f}%  {pde_y[k]:10.4e}  {y_ct_fit:10.4e}  {pde_drho[k]:8.4f}")

# ── Detailed comparison at key z values ──────────────────────────────────
print("\n\n=== Detailed spectral comparison at key redshifts ===")
z_detail = [5e3, 5e4, 2e5, 5e5, 1e6, 2e6, 3e6]

for z_d in z_detail:
    ik = np.argmin(np.abs(z_ct - z_d))
    z_h = z_ct[ik]

    di_pde = gf_pde_jy_ct[:, ik]
    di_ct = gf_ct_jy[:, ik]

    peak = np.max(np.abs(di_ct[mask_freq]))
    if peak == 0:
        continue

    rms = np.sqrt(np.mean(((di_pde[mask_freq] - di_ct[mask_freq]) / peak)**2)) * 100

    print(f"\nz_h = {z_h:.3e}  (RMS = {rms:.2f}%)")
    print(f"  {'nu':>6s}  {'CT [Jy/sr]':>12s}  {'PDE [Jy/sr]':>12s}  {'ratio':>8s}  {'%dev':>8s}")
    for nu_c in [30, 60, 100, 150, 200, 300, 500, 857]:
        ic = np.argmin(np.abs(nu_ct_ghz - nu_c))
        ratio = di_pde[ic] / di_ct[ic] if abs(di_ct[ic]) > 1e-30 else 0
        pct = (di_pde[ic] - di_ct[ic]) / peak * 100
        print(f"  {nu_ct_ghz[ic]:6.0f}  {di_ct[ic]:12.4e}  {di_pde[ic]:12.4e}  "
              f"{ratio:8.4f}  {pct:7.2f}%")
