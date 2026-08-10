#!/usr/bin/env python3
"""Assemble raw Rust solver output into the Fig. 4 PDE cache.

``notebooks/paper_figures/dm_scenario_comparison.ipynb`` keys its direct-PDE
cache on (physics hash, scenario parameters, n_points, dy_max, dtau_max) and
stores one JSON per key under ``~/.spectroxide/dm_pde_cache/``.  Solving is done
by the sibling shell script, which runs the binary once per scenario with
``--output`` so no finished solve is ever lost::

    dev/scripts/dm_residual_diagnostics/run_dm_pde_cache.sh 0.002 0.001

This script then folds those raw files into a keyed cache entry the notebook
loads without re-solving::

    python dev/scripts/dm_residual_diagnostics/run_dm_pde_cache.py --config 0.002:1.0

The scenario parameters below MUST stay in sync with cells 3 and 11 of the
notebook and with the shell script; the printed key is the check (it matches the
notebook's only if every payload field agrees).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from spectroxide import DEFAULT_COSMO, cosmic_time, get_physics_hash

# --- kept in sync with the notebook -----------------------------------------
Z_START, Z_END = 5.0e6, 1001.0
F_X, F_ANN_SW, F_ANN_PW = 7.757e5, 3.758e-20, 5.789e-26
GAMMA_X = 1.0 / cosmic_time(5.0e4 + 10.0, DEFAULT_COSMO)

DM_INJECTIONS = {
    "Decay": {"type": "decaying-particle", "f_x": F_X, "gamma_x": GAMMA_X},
    "s-wave": {"type": "annihilating-dm", "f_ann": F_ANN_SW},
    "p-wave": {"type": "annihilating-dm-pwave", "f_ann": F_ANN_PW},
}

CACHE_DIR = Path.home() / ".spectroxide" / "dm_pde_cache"
RAW_DIR = Path.home() / ".spectroxide" / "dm_pde_raw"


def cache_key(n_points: int, dy_max: float, dtau_max: float) -> str:
    payload = {
        "phys_hash": get_physics_hash(),
        "Z_START": float(Z_START),
        "Z_END": float(Z_END),
        "n_points": n_points,
        "dy_max": dy_max,
        "dtau_max": dtau_max,
        "inj": {k: v for k, v in sorted(DM_INJECTIONS.items())},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True,
                    metavar="DY:DTAU", help="repeatable, e.g. --config 0.002:1.0")
    ap.add_argument("--n-points", type=int, default=8000)
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for spec in args.config:
        dy_max, dtau_max = (float(v) for v in spec.split(":"))
        key = cache_key(args.n_points, dy_max, dtau_max)
        dest = CACHE_DIR / f"{key}.json"
        print(f"dy={dy_max} dtau={dtau_max} n={args.n_points} -> {dest.name}")

        results, missing = {}, []
        for name in DM_INJECTIONS:
            # The shell runner tags files with the dtau string as typed ("1.0"),
            # which %g renders as "1"; accept either spelling.
            stem = f"{name}_n{args.n_points}_dy{dy_max:g}_dtau"
            raw = next(
                (p for p in (RAW_DIR / f"{stem}{s}.json"
                             for s in (f"{dtau_max:g}", f"{dtau_max:.1f}"))
                 if p.exists()),
                None,
            )
            if raw is None:
                missing.append(f"{stem}{dtau_max:g}.json")
                continue
            r = json.load(open(raw))["results"][0]
            results[name] = {
                "x": r["x"], "delta_n": r["delta_n"],
                "pde_mu": r["pde_mu"], "pde_y": r["pde_y"],
                "drho": r["drho"], "z_h": None,
            }
            print(f"  {name}: drho={r['drho']:.6e} mu={r['pde_mu']:.5e} "
                  f"y={r['pde_y']:.5e} steps={int(r['step_count'])}")
        if missing:
            print(f"  incomplete, not written (missing {', '.join(missing)})")
            continue
        with open(dest, "w") as f:
            json.dump({
                "key": key,
                "meta": {"n_points": args.n_points, "dy_max": dy_max,
                         "dtau_max": dtau_max, "phys_hash": get_physics_hash()},
                "results": results,
            }, f)
        print(f"  wrote {dest}")


if __name__ == "__main__":
    main()
