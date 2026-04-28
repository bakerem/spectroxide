#!/usr/bin/env python3
"""Reproduce the performance table timings for the paper.

Runs each operation from Table (performance) and prints wall-clock times.
Assumes the Rust binary is already compiled (cached).
"""
import sys
import pathlib
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "python"))
import spectroxide
from spectroxide.greens import mu_from_heating

# Warm up: ensure binary is compiled and cached
print("Warming up (compiling if needed)...")
t0 = time.perf_counter()
_ = spectroxide.solve(
    injection={"type": "single_burst", "z_h": 1e4},
    delta_rho=1e-5,
    method="pde",
)
t_warmup = time.perf_counter() - t0
print(f"  Warmup: {t_warmup:.1f}s\n")

results = []

# --- PDE single burst: y-era ---
print("PDE single burst (y-era, z_h=1e3)...")
t0 = time.perf_counter()
_ = spectroxide.solve(
    injection={"type": "single_burst", "z_h": 1e3},
    delta_rho=1e-5, method="pde",
)
dt = time.perf_counter() - t0
results.append(("PDE single burst (y-era, z_h=1e3)", dt))
print(f"  {dt:.1f}s")

# --- PDE single burst: mu-era ---
print("PDE single burst (mu-era, z_h=1e5)...")
t0 = time.perf_counter()
_ = spectroxide.solve(
    injection={"type": "single_burst", "z_h": 1e5},
    delta_rho=1e-5, method="pde",
)
dt = time.perf_counter() - t0
results.append(("PDE single burst (mu-era, z_h=1e5)", dt))
print(f"  {dt:.1f}s")

# --- PDE single burst: thermalization ---
print("PDE single burst (thermalization, z_h=3e6)...")
t0 = time.perf_counter()
_ = spectroxide.solve(
    injection={"type": "single_burst", "z_h": 3e6},
    delta_rho=1e-5, method="pde",
)
dt = time.perf_counter() - t0
results.append(("PDE single burst (thermalization, z_h=3e6)", dt))
print(f"  {dt:.1f}s")

# --- PDE decaying particle ---
print("PDE decaying particle...")
t0 = time.perf_counter()
_ = spectroxide.solve(
    injection={"type": "decaying_particle",
               "f_x": 2.5e5, "gamma_x": 1.07e-10},
    method="pde",
)
dt = time.perf_counter() - t0
results.append(("PDE decaying particle", dt))
print(f"  {dt:.1f}s")

# --- PDE s-wave annihilation ---
print("PDE s-wave annihilation...")
t0 = time.perf_counter()
_ = spectroxide.solve(
    injection={"type": "annihilating_dm", "f_ann": 1e-22},
    method="pde",
)
dt = time.perf_counter() - t0
results.append(("PDE s-wave annihilation", dt))
print(f"  {dt:.1f}s")

# --- PDE monochromatic photon injection ---
print("PDE monochromatic photon injection (x_inj=0.01, z_h=5e5)...")
t0 = time.perf_counter()
_ = spectroxide.solve(
    injection={
        "type": "monochromatic_photon",
        "x_inj": 0.01,
        "delta_n_over_n": 1e-5,
        "z_h": 5e5,
    },
    method="pde",
)
dt = time.perf_counter() - t0
results.append(("PDE monochromatic photon injection", dt))
print(f"  {dt:.1f}s")

# --- 15-point sweep (parallel) ---
print("15-point sweep (parallel)...")
z_list = np.geomspace(1e3, 3e6, 15).tolist()
t0 = time.perf_counter()
_ = spectroxide.run_sweep(delta_rho=1e-5, z_injections=z_list)
dt = time.perf_counter() - t0
results.append(("15-point sweep (parallel)", dt))
print(f"  {dt:.1f}s")

# --- GF single spectrum ---
print("GF single spectrum (1000 pts)...")
t0 = time.perf_counter()
for _ in range(1000):
    _ = spectroxide.solve(z_h=1e5, delta_rho=1e-5, method="greens_function")
dt_per = (time.perf_counter() - t0) / 1000
results.append(("GF single spectrum (1000 pts)", dt_per))
print(f"  {dt_per*1000:.2f}ms")

# --- GF convolution (1000x1000) ---
print("GF convolution (1000x1000)...")
from spectroxide.greens import greens_function
z_arr = np.geomspace(1e3, 3e6, 1000)
dq = np.ones_like(z_arr) * 1e-10
x_out = np.linspace(0.1, 30, 1000)
dz = np.gradient(z_arr)
t0 = time.perf_counter()
dn = np.zeros_like(x_out)
for i in range(len(z_arr)):
    dn += greens_function(x_out, z_arr[i]) * dq[i] * dz[i]
dt = time.perf_counter() - t0
results.append(("GF convolution (1000x1000)", dt))
print(f"  {dt*1000:.1f}ms")

# --- GF mu+y from heating ---
print("GF mu+y from heating...")
dq_dz_func = lambda z: 1e-10
t0 = time.perf_counter()
for _ in range(100):
    _ = mu_from_heating(dq_dz_func, 1e3, 3e6, n_z=1000)
dt_per = (time.perf_counter() - t0) / 100
results.append(("GF mu+y from heating", dt_per))
print(f"  {dt_per*1000:.2f}ms")

# --- Summary ---
print("\n" + "=" * 60)
print(f"{'Operation':<45s} {'Time':>10s}")
print("-" * 60)
for name, dt in results:
    if dt < 0.01:
        print(f"{name:<45s} {dt*1000:>8.2f}ms")
    elif dt < 1:
        print(f"{name:<45s} {dt*1000:>8.1f}ms")
    else:
        print(f"{name:<45s} {dt:>8.1f}s")
print("=" * 60)
