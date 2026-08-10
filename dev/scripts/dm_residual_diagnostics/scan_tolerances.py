#!/usr/bin/env python3
"""Scan solver tolerances against the PDE's distortion-energy deficit.

The Fig. 4 residual is an amplitude offset equal to the direct solve's energy
deficit (see ``dev/audit/dm_comparison_residual.md``).  This script measures that
deficit as a function of ``dtau_max``, ``dy_max`` and grid size for the s-wave
scenario, the worst of the three.

Reference: the injected total for these parameters, integrated with the same
cosmology the Rust solver uses, is 9.9972e-6.

Usage::

    python dev/scripts/dm_residual_diagnostics/scan_tolerances.py

Runtime is hours on a 4-core machine; run it detached.
"""
from __future__ import annotations

from spectroxide import solve

INJ = {"type": "annihilating-dm", "f_ann": 3.758e-20}
INJECTED_TOTAL = 9.9972e-06

# (n_points, dtau_max, dy_max, number_conserving)
CASES = [
    (2000, 20, 0.005, True),
    (2000, 10, 0.005, True),
    (2000, 5, 0.005, True),
    (2000, 3, 0.005, True),
    (2000, 10, 0.002, True),
    (2000, 10, 0.001, True),
    (2000, 3, 0.001, True),
    (4000, 10, 0.005, True),
    (2000, 10, 0.005, False),
]


def main() -> None:
    print(f"s-wave, injected total drho/rho = {INJECTED_TOTAL:.4e}")
    print(f"{'n_pts':>6} {'dtau':>5} {'dy':>7} {'NC':>6} "
          f"{'drho/rho':>13} {'deficit%':>9} {'mu':>12} {'y':>12}")
    for n_points, dtau, dy, nc in CASES:
        try:
            sw = solve(
                injection=INJ, delta_rho=1e-5, z_start=5.0e6, z_end=1001.0,
                n_points=n_points, number_conserving=nc, timeout=7200,
                dy_max=dy, dtau_max=dtau,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            print(f"{n_points:6d} {dtau:5g} {dy:7g} {str(nc):>6} "
                  f"FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        d = sw.delta_rho_over_rho
        print(f"{n_points:6d} {dtau:5g} {dy:7g} {str(nc):>6} {d:13.6e} "
              f"{(d / INJECTED_TOTAL - 1) * 100:+9.3f} {sw.mu:12.5e} {sw.y:12.5e}",
              flush=True)


if __name__ == "__main__":
    main()
