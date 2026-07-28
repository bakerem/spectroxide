#!/usr/bin/env python3
"""Numerical error budget for the spectroxide PDE solver (audit Phase 2, B3).

Parses the structured output of the convergence and MMS test suites and
produces a quantitative error budget: the estimated relative discretization
error of the production solver at its default and production settings, per
error source. This feeds the declared tolerances of the benchmark pack
(plan Part A).

Inputs (regenerate with the commands below if missing):
    dev/output/convergence_ci.log
        cargo test --release --test convergence_order -- --nocapture
    dev/output/mms_ci.log
        cargo test --release --test mms_convergence -- --nocapture

Outputs:
    dev/output/error_budget.md   — budget table with derivations
    dev/output/error_budget.pdf  — convergence curves with order guides

Method: Richardson extrapolation. For a sequence Q_N with measured order p,
Q_inf ≈ Q_N + (Q_N − Q_{N/2}) / (2^p − 1), and the error at level N is
|Q_N − Q_inf|. MMS true errors need no extrapolation (the exact solution is
known analytically) and are quoted directly.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "dev" / "output"

# Solver defaults these budget rows refer to.
DEFAULT_N = 2000
PRODUCTION_N = 4000
DEFAULT_DY = 0.02


def parse_conv(path: Path):
    """CONV|scenario|sweep_type|n_points|dy_max|mu|y|drho|l2_norm|steps"""
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"CONV\|([\w_]+)\|(\w+)\|([\d.e+-]+)\|([\d.e+-]+)\|(.*)", line)
        if not m:
            continue
        scen, sweep, n, dy, rest = m.groups()
        vals = rest.split("|")
        rows.append(
            dict(
                scenario=scen,
                sweep=sweep,
                n=int(float(n)),
                dy=float(dy),
                mu=float(vals[0]),
                y=float(vals[1]),
                drho=float(vals[2]),
                l2=float(vals[3]),
            )
        )
    return rows


def parse_mms(path: Path):
    """MMS|case|kind|key=val|rel_l2=..."""
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"MMS\|([\w_]+)\|(\w+)\|(?:N=(\d+)|dtau_max=([\d.]+))\|rel_l2=([\d.e+-]+)", line)
        if m:
            case, kind, n, dtau, err = m.groups()
            rows.append(
                dict(
                    case=case,
                    kind=kind,
                    n=int(n) if n else None,
                    dtau=float(dtau) if dtau else None,
                    err=float(err),
                )
            )
    return rows


def richardson(values, ratio=2.0):
    """Return (order, q_inf, err_at_finest) from a refinement sequence."""
    diffs = [abs(values[i] - values[i + 1]) for i in range(len(values) - 1)]
    orders = [
        math.log(diffs[i] / diffs[i + 1]) / math.log(ratio)
        for i in range(len(diffs) - 1)
        if diffs[i + 1] > 0
    ]
    p = sorted(orders)[len(orders) // 2] if orders else float("nan")
    if not math.isfinite(p) or p <= 0:
        p = 1.0
    q_inf = values[-1] + (values[-1] - values[-2]) / (ratio**p - 1.0)
    err_finest = abs(values[-1] - q_inf)
    return p, q_inf, err_finest


def main():
    conv_log = OUT / "convergence_ci.log"
    mms_log = OUT / "mms_ci.log"
    for f in (conv_log, mms_log):
        if not f.exists():
            sys.exit(f"missing {f} — regenerate per the module docstring")

    conv = parse_conv(conv_log)
    mms = parse_mms(mms_log)

    budget = []  # (source, setting, estimate, method)

    # --- 1. Spatial discretization, μ (full physics, self-convergence) ------
    sp = [r for r in conv if r["scenario"] == "full_physics" and r["sweep"] == "spatial"]
    sp.sort(key=lambda r: r["n"])
    mus = [r["mu"] for r in sp]
    ns = [r["n"] for r in sp]
    if len(mus) >= 3:
        p, mu_inf, _ = richardson(mus)
        for target_n in (DEFAULT_N, PRODUCTION_N):
            if target_n in ns:
                err = abs(mus[ns.index(target_n)] - mu_inf) / abs(mu_inf)
                budget.append(
                    (
                        "Spatial (μ, full physics)",
                        f"N={target_n}",
                        err,
                        f"Richardson p={p:.2f} over N={ns}",
                    )
                )

    # --- 2. Temporal stepping at PRODUCTION defaults -------------------------
    # At the default settings (dy_max=0.02, dtau_max=10), the binding step
    # cap for a z=2e5 burst run is dtau_max, not dy_max (10k steps vs 581 if
    # dy alone controlled). The production temporal error is therefore
    # measured by direct dtau_max refinement at fixed dy_max=0.02
    # (examples/temporal_error_check.rs → dev/output/temporal_check.log),
    # not by extrapolating the dy sweep (which the convergence tests run at
    # the non-default dtau_max=200).
    tc_log = OUT / "temporal_check.log"
    if tc_log.exists():
        seq = []  # (dtau, mu, y) descending dtau
        for line in tc_log.read_text().splitlines():
            m = re.match(r"dtau_max=([\d.]+): mu=([\d.e+-]+) y=([\d.e+-]+)", line)
            if m:
                seq.append(tuple(float(v) for v in m.groups()))
        seq.sort(key=lambda t: -t[0])
        if len(seq) >= 3:
            mus_t = [s[1] for s in seq]
            ys_t = [s[2] for s in seq]
            p_mu, mu_inf_t, _ = richardson(mus_t)
            p_y, y_inf_t, _ = richardson(ys_t)
            i10 = [s[0] for s in seq].index(10.0)
            budget.append(
                (
                    "Temporal (μ, full physics, production defaults)",
                    "dtau_max=10",
                    abs(mus_t[i10] - mu_inf_t) / abs(mu_inf_t),
                    f"direct dtau_max refinement, Richardson p={p_mu:.2f}",
                )
            )
            budget.append(
                (
                    "Temporal (y, full physics, production defaults)",
                    "dtau_max=10",
                    abs(ys_t[i10] - y_inf_t) / abs(y_inf_t),
                    f"direct dtau_max refinement, Richardson p={p_y:.2f}",
                )
            )

    # --- 3. MMS true errors (exact analytic anchors) -------------------------
    prod = [r for r in mms if r["case"] == "production_grid" and r["kind"] == "spatial"]
    prod.sort(key=lambda r: r["n"])
    for r in prod:
        if r["n"] in (DEFAULT_N, PRODUCTION_N):
            budget.append(
                (
                    "Spatial spectrum, MMS true error (Kompaneets, production grid)",
                    f"N={r['n']}",
                    r["err"],
                    "exact manufactured solution, rel. x³-weighted L2",
                )
            )
    solver_lvl = [r for r in mms if r["case"] == "solver_level"]
    solver_lvl_sorted = sorted(solver_lvl, key=lambda r: r["dtau"] or 0)
    if solver_lvl_sorted:
        finest = solver_lvl_sorted[0]
        budget.append(
            (
                "End-to-end solver MMS true error (incl. source splitting)",
                f"N=2000, dtau_max={finest['dtau']:g}",
                finest["err"],
                "exact manufactured solution through TabulatedPhotonSource",
            )
        )

    # --- write markdown ------------------------------------------------------
    md = OUT / "error_budget.md"
    lines = [
        "# Numerical error budget (audit Phase 2, B3)",
        "",
        "Relative discretization-error estimates for the production solver,",
        "from Richardson extrapolation of the self-convergence suites and the",
        "true-error MMS suite (`tests/mms_convergence.rs`). These set the",
        "floor for benchmark-pack tolerances (plan Part A).",
        "",
        "| Error source | Setting | Relative error | Method |",
        "|---|---|---|---|",
    ]
    for src, setting, err, method in budget:
        lines.append(f"| {src} | {setting} | {err:.2e} | {method} |")
    lines += [
        "",
        "Additional floors (measured elsewhere in the Phase 2 suites):",
        "",
        "- Newton-tolerance conservation floor: ~2×10⁻⁸ relative photon-number",
        "  drift over 10³ steps (`conservation_fuzz.rs`, pure Compton).",
        "- Energy-ledger closure under fuzzing: ≤ 0.7% across randomized heat",
        "  scenarios at N=800–1600 (dominated by spatial truncation).",
        "- Kernel MMS orders: Kompaneets CN 2.0 (space and time), coupled",
        "  DC/BR backward Euler 1.0 (time) / 2.0 (space) — design orders",
        "  confirmed against exact solutions.",
        "",
        "Regenerate: `python dev/scripts/error_budget.py` after refreshing the",
        "logs (see module docstring).",
    ]
    md.write_text("\n".join(lines) + "\n")
    print(f"wrote {md}")

    # --- figure ---------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["text.usetex"] = False  # dev figure; avoid TeX round-trips
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — skipped figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    ax = axes[0]
    for case, marker, label in [
        ("pure_kompaneets", "o", "kernel MMS, log grid"),
        ("production_grid", "s", "kernel MMS, production grid"),
    ]:
        rows = sorted(
            (r for r in mms if r["case"] == case and r["kind"] == "spatial"),
            key=lambda r: r["n"],
        )
        if rows:
            ax.loglog([r["n"] for r in rows], [r["err"] for r in rows], marker + "-", label=label)
    if rows:
        n0, e0 = rows[0]["n"], rows[0]["err"]
        nn = [r["n"] for r in rows]
        ax.loglog(nn, [e0 * (n0 / n) ** 2 for n in nn], "k--", lw=0.8, label=r"$\propto N^{-2}$")
    ax.set_xlabel("grid points $N$")
    ax.set_ylabel("relative $L^2$ error vs exact solution")
    ax.set_title("MMS spatial convergence (true error)")
    ax.legend(fontsize=8)

    ax = axes[1]
    sp_mu = sorted(
        (r for r in conv if r["scenario"] == "full_physics" and r["sweep"] == "spatial"),
        key=lambda r: r["n"],
    )
    if len(sp_mu) >= 3:
        mus = [r["mu"] for r in sp_mu]
        p, mu_inf, _ = richardson(mus)
        errs = [abs(m - mu_inf) / abs(mu_inf) for m in mus[:-1]]
        ax.loglog([r["n"] for r in sp_mu[:-1]], errs, "o-", label=r"$\mu$ (full physics)")
        ax.set_xlabel("grid points $N$")
        ax.set_ylabel(r"$|\mu_N - \mu_\infty|/\mu_\infty$")
        ax.set_title(rf"Richardson error, $\mu$ (p={p:.2f})")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "error_budget.pdf")
    print(f"wrote {OUT / 'error_budget.pdf'}")


if __name__ == "__main__":
    main()
