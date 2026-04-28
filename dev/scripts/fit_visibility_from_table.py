#!/usr/bin/env python3
"""Fit visibility function parameters from a pre-built PDE table.

Loads visibility_table.npz (from build_visibility_table.py) and performs
a global fit of the visibility function parameters to the NC-stripped
PDE spectra.

Tests:
  1. Baseline L-BFGS-B from literature initial guess
  2. L-BFGS-B from PDE-fitted initial guess (warm start)
  3. Differential evolution (global optimizer, no initial guess)
  4. Basin-hopping (global with local refinement)
  5. Extended x-range sensitivity ([0.3, 25] vs [0.5, 20])

Outputs a summary table of fitted parameters and cost reductions.
"""
import sys
import pathlib
import time

import numpy as np
from scipy.optimize import minimize as sp_minimize, differential_evolution, basinhopping

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "python"))

from spectroxide import strip_gbb
from spectroxide.greens import (
    mu_shape,
    y_shape,
    g_bb,
    Z_MU,
    KAPPA_C,
)

MU_TO_ENERGY = 3.0 / KAPPA_C

# ── Load table ──────────────────────────────────────────────────────────
datadir = pathlib.Path(__file__).resolve().parent.parent / "data"
tablepath = datadir / "visibility_table.npz"
if not tablepath.exists():
    print(f"ERROR: {tablepath} not found. Run build_visibility_table.py first.")
    sys.exit(1)

data = np.load(tablepath)
z_h = data["z_h"]
x_pde = data["x"]
dn_nc = data["dn_nc"]     # (N_z, N_x), per unit Δρ/ρ, NC-stripped
drho = data["drho"]        # (N_z,), should be ≈ 1.0
n_points = int(data["n_points"])

n_z = len(z_h)
n_x = len(x_pde)
print(f"Loaded visibility table: {n_z} redshifts, {n_x} freq points (n_points={n_points})")
print(f"  z range: [{z_h[0]:.2e}, {z_h[-1]:.2e}]")
print(f"  x range: [{x_pde[0]:.4f}, {x_pde[-1]:.2f}]")

# ── Parametric visibility functions ─────────────────────────────────────
def _j_bb(z, z_th, alpha_th):
    return np.exp(-((z / z_th) ** alpha_th))

def _j_bb_star(z, A, z_th, alpha_th, B, beta):
    ratio = z / z_th
    return np.maximum(A * _j_bb(z, z_th, alpha_th) * (1.0 - B * ratio ** beta), 0.0)

def _j_mu(z, z_mu, alpha_mu):
    return 1.0 - np.exp(-(((1.0 + z) / z_mu) ** alpha_mu))

def _j_y(z, z_y, alpha_y):
    return 1.0 / (1.0 + ((1.0 + z) / z_y) ** alpha_y)


def gf_model(x, z_h_val, params):
    """Three-component GF model spectrum per unit Δρ/ρ."""
    z_y, alpha_y, z_mu, alpha_mu, A, z_th, alpha_th, B, beta_val = params
    jm = _j_mu(z_h_val, z_mu, alpha_mu)
    jb = _j_bb_star(z_h_val, A, z_th, alpha_th, B, beta_val)
    jyv = _j_y(z_h_val, z_y, alpha_y)
    M = mu_shape(x)
    Y = y_shape(x)
    G = g_bb(x)
    return MU_TO_ENERGY * jm * jb * M + 0.25 * jyv * Y + 0.25 * (1.0 - jb) * G


def nc_strip(x, dn):
    """Enforce ∫x²·Δn·dx = 0 (number conservation)."""
    G = g_bb(x)
    alpha = np.trapz(x**2 * dn, x) / np.trapz(x**2 * G, x)
    return dn - alpha * G


# ── Pre-compute basis shapes (once) ────────────────────────────────────
M_x = mu_shape(x_pde)
Y_x = y_shape(x_pde)
G_x = g_bb(x_pde)
# NC-strip basis shapes
G_int = np.trapz(x_pde**2 * G_x, x_pde)
M_nc = M_x - np.trapz(x_pde**2 * M_x, x_pde) / G_int * G_x
Y_nc = Y_x - np.trapz(x_pde**2 * Y_x, x_pde) / G_int * G_x
G_nc = G_x - np.trapz(x_pde**2 * G_x, x_pde) / G_int * G_x  # = 0 by construction


# ── Cost function factory ───────────────────────────────────────────────
def make_cost(x_lo=0.5, x_hi=20.0, weight_power=3):
    """Return a cost function with the given x-range and weighting."""
    mask = (x_pde >= x_lo) & (x_pde <= x_hi)
    x_m = x_pde[mask]
    w = x_m ** weight_power

    # Pre-slice PDE data
    pde_sliced = dn_nc[:, mask]

    def cost(params):
        z_y, alpha_y, z_mu, alpha_mu, A, z_th, alpha_th, B, beta_val = params
        chi2 = 0.0
        for i in range(n_z):
            jm = _j_mu(z_h[i], z_mu, alpha_mu)
            jb = _j_bb_star(z_h[i], A, z_th, alpha_th, B, beta_val)
            jyv = _j_y(z_h[i], z_y, alpha_y)
            # Model per unit Δρ/ρ, NC-stripped analytically
            model_nc = (MU_TO_ENERGY * jm * jb * M_nc[mask]
                        + 0.25 * jyv * Y_nc[mask]
                        + 0.25 * (1.0 - jb) * G_nc[mask])
            # Scale by actual Δρ/ρ from PDE
            model_nc *= drho[i]
            resid = w * (model_nc - pde_sliced[i])
            chi2 += np.dot(resid, resid)
        return chi2

    return cost


# ── Parameter setup ─────────────────────────────────────────────────────
PARAM_NAMES = ["z_y", "α_y", "z_μ", "α_μ", "A", "z_th", "α_th", "B", "β"]
LIT_VALUES = np.array([6e4, 2.58, 5.8e4, 1.88, 0.983, 1.98e6, 2.5, 0.0381, 2.29])

BOUNDS = [
    (1e4, 2e5),    # z_y
    (1.0, 5.0),    # alpha_y
    (1e4, 2e5),    # z_mu
    (1.0, 4.0),    # alpha_mu
    (0.8, 1.1),    # A
    (5e5, 1e7),    # z_th
    (1.5, 4.0),    # alpha_th
    (0.0, 0.3),    # B
    (1.0, 5.0),    # beta
]


def print_params(label, params, ref=LIT_VALUES):
    """Print parameter comparison table."""
    print(f"\n  {label}")
    print(f"  {'Param':<8s} {'Value':>14s} {'Literature':>14s} {'Δ (%)':>8s}")
    print(f"  {'-' * 46}")
    for name, val, lit in zip(PARAM_NAMES, params, ref):
        pct = 100 * (val - lit) / lit if lit != 0 else 0
        if lit > 1000:
            print(f"  {name:<8s} {val:>14.1f} {lit:>14.1f} {pct:>8.1f}")
        else:
            print(f"  {name:<8s} {val:>14.4f} {lit:>14.4f} {pct:>8.1f}")


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Baseline L-BFGS-B, standard x-range [0.5, 20]
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: L-BFGS-B from literature values, x ∈ [0.5, 20]")
print("=" * 70)

cost_std = make_cost(0.5, 20.0)
cost0 = cost_std(LIT_VALUES)
print(f"  Initial cost (literature): {cost0:.6e}")

t0 = time.time()
res1 = sp_minimize(
    lambda p: cost_std(p) / cost0,
    LIT_VALUES,
    bounds=BOUNDS,
    method="L-BFGS-B",
    options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10},
)
t1 = time.time()
p1 = res1.x
c1 = cost_std(p1)
print(f"  Final cost: {c1:.6e} (reduction: {c1/cost0:.6f})")
print(f"  Converged: {res1.success}, nit={res1.nit}, time={t1-t0:.1f}s")
print(f"  Message: {res1.message}")
print_params("L-BFGS-B (standard)", p1)


# ═══════════════════════════════════════════════════════════════════════
# Test 2: L-BFGS-B with tighter tolerances and warm start from Test 1
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: L-BFGS-B warm start from Test 1")
print("=" * 70)

t0 = time.time()
res2 = sp_minimize(
    lambda p: cost_std(p) / cost0,
    p1,  # warm start
    bounds=BOUNDS,
    method="L-BFGS-B",
    options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-12},
)
t1 = time.time()
p2 = res2.x
c2 = cost_std(p2)
print(f"  Final cost: {c2:.6e} (reduction from lit: {c2/cost0:.6f})")
print(f"  Converged: {res2.success}, nit={res2.nit}, time={t1-t0:.1f}s")
shift = np.max(np.abs(p2 - p1) / np.maximum(np.abs(p1), 1e-10)) * 100
print(f"  Max param shift from Test 1: {shift:.4f}%")
print_params("L-BFGS-B (warm start)", p2, ref=p1)


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Differential evolution (global optimizer)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: Differential evolution (global), x ∈ [0.5, 20]")
print("=" * 70)

t0 = time.time()
res3 = differential_evolution(
    cost_std,
    bounds=BOUNDS,
    seed=42,
    maxiter=1000,
    tol=1e-12,
    polish=True,       # L-BFGS-B polish at the end
)
t1 = time.time()
p3 = res3.x
c3 = cost_std(p3)
print(f"  Final cost: {c3:.6e} (reduction from lit: {c3/cost0:.6f})")
print(f"  Converged: {res3.success}, nit={res3.nit}, time={t1-t0:.1f}s")
shift = np.max(np.abs(p3 - p1) / np.maximum(np.abs(p1), 1e-10)) * 100
print(f"  Max param shift from L-BFGS-B: {shift:.4f}%")
print_params("Differential evolution", p3, ref=p1)


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Basin-hopping (global + local)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: Basin-hopping from literature values, x ∈ [0.5, 20]")
print("=" * 70)

t0 = time.time()
res4 = basinhopping(
    cost_std,
    LIT_VALUES,
    niter=200,
    T=cost0 * 0.01,    # temperature ~ 1% of initial cost
    stepsize=0.05,       # relative step
    seed=42,
    minimizer_kwargs={
        "method": "L-BFGS-B",
        "bounds": BOUNDS,
        "options": {"maxiter": 2000, "ftol": 1e-14},
    },
)
t1 = time.time()
p4 = res4.x
c4 = cost_std(p4)
print(f"  Final cost: {c4:.6e} (reduction from lit: {c4/cost0:.6f})")
print(f"  nit={res4.nit}, time={t1-t0:.1f}s")
shift = np.max(np.abs(p4 - p1) / np.maximum(np.abs(p1), 1e-10)) * 100
print(f"  Max param shift from L-BFGS-B: {shift:.4f}%")
print_params("Basin-hopping", p4, ref=p1)


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Extended x-range [0.3, 25]
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 5: L-BFGS-B with extended x-range [0.3, 25]")
print("=" * 70)

cost_ext = make_cost(0.3, 25.0)
cost0_ext = cost_ext(LIT_VALUES)
print(f"  Initial cost (literature): {cost0_ext:.6e}")

t0 = time.time()
res5 = sp_minimize(
    lambda p: cost_ext(p) / cost0_ext,
    LIT_VALUES,
    bounds=BOUNDS,
    method="L-BFGS-B",
    options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10},
)
t1 = time.time()
p5 = res5.x
c5 = cost_ext(p5)
print(f"  Final cost: {c5:.6e} (reduction from lit: {c5/cost0_ext:.6f})")
print(f"  Converged: {res5.success}, nit={res5.nit}, time={t1-t0:.1f}s")
print_params("Extended x-range", p5, ref=p1)


# ═══════════════════════════════════════════════════════════════════════
# Test 6: Narrower x-range [1.0, 15] (Chluba's original)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 6: L-BFGS-B with narrow x-range [1.0, 15]")
print("=" * 70)

cost_nar = make_cost(1.0, 15.0)
cost0_nar = cost_nar(LIT_VALUES)
print(f"  Initial cost (literature): {cost0_nar:.6e}")

t0 = time.time()
res6 = sp_minimize(
    lambda p: cost_nar(p) / cost0_nar,
    LIT_VALUES,
    bounds=BOUNDS,
    method="L-BFGS-B",
    options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10},
)
t1 = time.time()
p6 = res6.x
c6 = cost_nar(p6)
print(f"  Final cost: {c6:.6e} (reduction from lit: {c6/cost0_nar:.6f})")
print(f"  Converged: {res6.success}, nit={res6.nit}, time={t1-t0:.1f}s")
print_params("Narrow x-range", p6, ref=p1)


# ═══════════════════════════════════════════════════════════════════════
# Summary comparison
# ═══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("SUMMARY: cost reduction vs literature across all tests")
print("=" * 70)

# Evaluate all fits on the standard cost
tests = [
    ("Literature", LIT_VALUES),
    ("L-BFGS-B (std)", p1),
    ("L-BFGS-B (warm)", p2),
    ("Diff. evolution", p3),
    ("Basin-hopping", p4),
    ("Extended x [0.3,25]", p5),
    ("Narrow x [1,15]", p6),
]

print(f"\n  {'Method':<22s} {'Cost (std)':>12s} {'Red. %':>8s} {'Cost (ext)':>12s} {'Red. %':>8s}")
print(f"  {'-' * 68}")
for label, p in tests:
    cs = cost_std(p)
    ce = cost_ext(p)
    print(f"  {label:<22s} {cs:12.6e} {(1-cs/cost0)*100:7.3f}%  "
          f"{ce:12.6e} {(1-ce/cost0_ext)*100:7.3f}%")


# ═══════════════════════════════════════════════════════════════════════
# Parameter stability: max spread across optimizers
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PARAMETER STABILITY across optimizers (Tests 1-4, same x-range)")
print("=" * 70)

all_p = np.array([p1, p2, p3, p4])
print(f"\n  {'Param':<8s} {'Min':>14s} {'Max':>14s} {'Spread %':>10s} {'Literature':>14s}")
print(f"  {'-' * 60}")
for i, name in enumerate(PARAM_NAMES):
    lo, hi = all_p[:, i].min(), all_p[:, i].max()
    spread = (hi - lo) / (0.5 * (hi + lo)) * 100 if (hi + lo) > 0 else 0
    lit = LIT_VALUES[i]
    if lit > 1000:
        print(f"  {name:<8s} {lo:14.1f} {hi:14.1f} {spread:9.4f}%  {lit:14.1f}")
    else:
        print(f"  {name:<8s} {lo:14.4f} {hi:14.4f} {spread:9.4f}%  {lit:14.4f}")

# Best parameters (lowest cost on standard metric)
best_idx = np.argmin([cost_std(p) for _, p in tests[1:5]]) + 1
best_label, best_p = tests[best_idx]
print(f"\n  Best fit: {best_label}")
print_params("BEST FIT", best_p)

# ── Save best-fit results ───────────────────────────────────────────────
import json
outpath = datadir / "visibility_fit_results.json"
results_dict = {
    "n_points": n_points,
    "n_z": n_z,
    "z_range": [float(z_h[0]), float(z_h[-1])],
    "x_range": [float(x_pde[0]), float(x_pde[-1])],
    "literature": {n: float(v) for n, v in zip(PARAM_NAMES, LIT_VALUES)},
}
for label, p in tests[1:]:
    key = label.lower().replace(" ", "_").replace("[", "").replace("]", "").replace(",", "_").replace(".", "")
    results_dict[key] = {
        "params": {n: float(v) for n, v in zip(PARAM_NAMES, p)},
        "cost_std": float(cost_std(p)),
        "cost_ext": float(cost_ext(p)),
    }
with open(outpath, "w") as f:
    json.dump(results_dict, f, indent=2)
print(f"\nResults saved to {outpath}")
