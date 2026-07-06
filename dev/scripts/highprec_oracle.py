#!/usr/bin/env python3
"""High-precision (mpmath) oracles for spectroxide's cancellation-critical paths.

Workstream R4.1 of dev/PLAN_VALIDATION_ROUND2_2026-07-06.md.

Replaces "we argue this cancellation is handled" with "a 50-digit computation
confirms it across the whole switch domain." Four parts:

  1. Recompute every spectral constant in src/constants.rs from its DEFINING
     integral/series (not the closed-form identity, which is tautological) at
     50 digits and diff against the hard-coded value.
  2. Pitfall #5 (DC/BR source near-cancellation): n_pl(x/rho_e) - n_pl(x) over
     a (x, rho_e) grid straddling the |rho_e-1| = 0.01 branch switch.
  3. Pitfall #4 (perturbative Delta rho_eq): confirm the perturbative form
     matches the 50-digit I4/4G3 to O(dn^2) while the float64 full-integral
     route shows the claimed ~1e-3 noise floor.
  4. Pitfall #1 (Kompaneets flux splitting): the analytic n_pl(1+n_pl) term vs
     the naive finite-difference form (~1000x signal ratio claim).

Run:  python dev/scripts/highprec_oracle.py [--dps 60] [--part N]
Output: prints a report; writes JSON summary to dev/output/highprec/oracle.json
Regenerates the tables in dev/audit/highprec_numerics.md.

All float64 branches are transcribed TERM-BY-TERM from the Rust source; the
transcription mapping is documented in dev/audit/highprec_numerics.md. Do not
paraphrase Rust expressions here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import mpmath as mp

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "highprec")


# ---------------------------------------------------------------------------
# Hard-coded values transcribed from src/constants.rs (the values under test).
# ---------------------------------------------------------------------------
RUST_CONST = {
    "ZETA_3": 1.202_056_903_159_594_3,
    "G1_PLANCK": 1.644_934_066_848_226_4,     # pi^2/6
    "G2_PLANCK": 2.404_113_806_319_188_6,     # 2 zeta(3)
    "G3_PLANCK": 6.493_939_402_266_829,       # pi^4/15
    "I4_PLANCK": 4.0 * 6.493_939_402_266_829, # 4 G3
    # BETA_MU, KAPPA_C, ALPHA_RHO, X_BALANCED are computed in Rust from the
    # above by closed-form expressions; we recompute them from their DEFINING
    # integrals below and compare against the Rust closed-form value.
    "BETA_MU": 3.0 * 1.202_056_903_159_594_3 / 1.644_934_066_848_226_4,
    "ALPHA_RHO": 2.404_113_806_319_188_6 / 6.493_939_402_266_829,
    "KAPPA_C": 12.0 / (3.0 * 1.202_056_903_159_594_3 / 1.644_934_066_848_226_4)
    - 9.0 * 2.404_113_806_319_188_6 / 6.493_939_402_266_829,
    "X_BALANCED": 4.0 / (3.0 * (2.404_113_806_319_188_6 / 6.493_939_402_266_829)),
}


def _quad_0_inf(f):
    """Quadrature over [0, inf) with the integrand written to avoid overflow.

    mpmath's tanh-sinh handles the endpoint singularity at 0 and the
    exponential decay at infinity; splitting at 1 improves conditioning.
    """
    return mp.quad(f, [0, 1, mp.inf])


def part1_constants(dps: int) -> dict:
    """Recompute spectral constants from defining integrals at `dps` digits."""
    mp.mp.dps = dps

    # Planck occupation n_pl(x) = 1/(e^x - 1). Its integer moments:
    #   G_n = int_0^inf x^n n_pl(x) dx = Gamma(n+1) zeta(n+1)
    # and the (1+n_pl)-weighted moment
    #   I_4 = int_0^inf x^4 e^x/(e^x-1)^2 dx = 4 G_3   (integration by parts).
    # We compute the INTEGRALS numerically, not the closed forms.
    def n_pl(x):
        return 1 / (mp.e ** x - 1)

    G1 = _quad_0_inf(lambda x: x * n_pl(x))
    G2 = _quad_0_inf(lambda x: x ** 2 * n_pl(x))
    G3 = _quad_0_inf(lambda x: x ** 3 * n_pl(x))
    # I4 integrand: x^4 e^x/(e^x-1)^2. Rewrite to avoid overflow at large x:
    #   e^x/(e^x-1)^2 = 1/( (e^x-1)(1-e^-x) ) = e^-x/(1-e^-x)^2
    I4 = _quad_0_inf(lambda x: x ** 4 * mp.e ** (-x) / (1 - mp.e ** (-x)) ** 2)

    zeta3 = mp.zeta(3)

    # Derived-constant DEFINING forms (from constants.rs docstrings):
    #   beta_mu  = 3 zeta(3) / zeta(2),   zeta(2) = G1
    #   alpha_rho = G2 / G3
    #   kappa_c  = 3 * int x^3 M(x) dx / G3,  M(x) = (x/beta_mu - 1) e^x/(e^x-1)^2
    #   x_balanced = 4/(3 alpha_rho)
    beta_mu = 3 * zeta3 / G1
    alpha_rho = G2 / G3
    x_balanced = 4 / (3 * alpha_rho)
    # kappa_c straight from its defining integral (NOT the 12/beta - 9G2/G3 form):
    kappa_integrand = lambda x: x ** 3 * (x / beta_mu - 1) * mp.e ** (-x) / (
        1 - mp.e ** (-x)
    ) ** 2
    kappa_c = 3 * _quad_0_inf(kappa_integrand) / G3

    oracle = {
        "ZETA_3": zeta3,
        "G1_PLANCK": G1,
        "G2_PLANCK": G2,
        "G3_PLANCK": G3,
        "I4_PLANCK": I4,
        "BETA_MU": beta_mu,
        "ALPHA_RHO": alpha_rho,
        "KAPPA_C": kappa_c,
        "X_BALANCED": x_balanced,
    }

    rows = []
    for name, oracle_val in oracle.items():
        rust_val = RUST_CONST[name]
        # digits of agreement = -log10(relative error)
        rel = abs(mp.mpf(rust_val) - oracle_val) / abs(oracle_val)
        digits = float(-mp.log10(rel)) if rel > 0 else float("inf")
        rows.append(
            {
                "name": name,
                "rust": repr(rust_val),
                "oracle": mp.nstr(oracle_val, 20),
                "rel_err": mp.nstr(rel, 4),
                "digits_agree": round(digits, 1),
            }
        )
    return {"dps": dps, "rows": rows}


# ---------------------------------------------------------------------------
# Float64 transcriptions of the production Rust branches (term-by-term).
# Mapping table is in dev/audit/highprec_numerics.md. `math` = float64.
# ---------------------------------------------------------------------------
def f64_planck(x: float) -> float:
    """n_pl(x) = 1/(e^x - 1). Rust `spectrum::planck` uses `x.exp_m1()`."""
    return 1.0 / math.expm1(x)


def neq_full_branch_f64(x: float, rho: float) -> float:
    """solver.rs:1295  neq = planck(xe) - npl,  xe = x * (1/rho).

    The xe>500 -> INFINITY guard (line 1281) affects `bose_factor`, not neq;
    neq itself is planck(xe)-planck(x). Reproduced verbatim.
    """
    inv_rho = 1.0 / rho
    xe = x * inv_rho
    return f64_planck(xe) - f64_planck(x)


def neq_taylor_branch_f64(x: float, rho: float) -> float:
    """solver.rs:1254  neq = x * delta_rho_inv * npl*(npl+1),
    delta_rho_inv = delta_rho * inv_rho_eq = (rho-1) * (1/rho).  (comment 1228)
    """
    inv_rho = 1.0 / rho
    delta_rho = rho - 1.0
    delta_rho_inv = delta_rho * inv_rho
    npl = f64_planck(x)
    return x * delta_rho_inv * npl * (npl + 1.0)


def part2_dcbr_cancellation(dps: int) -> dict:
    """Pitfall #5: neq = n_pl(x/rho) - n_pl(x) across the |rho-1|=0.01 switch."""
    mp.mp.dps = dps

    def neq_oracle(x, rho):
        xm, rm = mp.mpf(x), mp.mpf(rho)
        npl = lambda t: 1 / (mp.e ** t - 1)
        return npl(xm / rm) - npl(xm)

    xs = [float(10 ** e) for e in _linspace(-4, mp.log10(30), 25)]
    # rho-1 straddling 0.01, both signs, log-spaced in |rho-1|
    drhos = []
    for e in _linspace(-8, -1, 22):
        drhos.append(+float(10 ** e))
        drhos.append(-float(10 ** e))

    # For each branch: worst relative error over the (x, rho) grid, split by
    # which side of the 0.01 switch we are on.
    # The Taylor truncation error of neq grows like x*(rho-1) (the neglected
    # 2nd-order term), so its RELATIVE error is large at large x. But there neq
    # is exponentially tiny AND the DC emission coefficient carries e^{-2x}, so
    # the physically weighted error is negligible. We therefore report the
    # relative error both raw and bucketed by x, and an emission-weighted
    # ABSOLUTE error using the DC high-frequency suppression H_dc(x)=e^{-2x}*poly
    # (the x-dependence of K_DC) as the physical weight.
    def h_dc(xx):  # transcribed from double_compton.rs dc_high_freq_suppression
        return math.exp(-2.0 * xx) * (
            1.0
            + 1.5 * xx
            + 29.0 / 24.0 * xx ** 2
            + 11.0 / 16.0 * xx ** 3
            + 5.0 / 12.0 * xx ** 4
        )

    def scan(branch_fn, in_window):
        # returns max rel err (raw), max rel err for x<=5, max rel err for x<=10,
        # and max emission-weighted ABSOLUTE error, over the region where the
        # branch is USED (in_window True => |rho-1|<0.01).
        raw = x5 = x10 = wabs = 0.0
        for x in xs:
            w = h_dc(x)
            for drho in drhos:
                if (abs(drho) < 0.01) != in_window:
                    continue
                rho = 1.0 + drho
                oracle = neq_oracle(x, rho)
                if oracle == 0:
                    continue
                val = branch_fn(x, rho)
                aerr = float(abs(mp.mpf(val) - oracle))
                rel = aerr / float(abs(oracle))
                raw = max(raw, rel)
                if x <= 5.0:
                    x5 = max(x5, rel)
                if x <= 10.0:
                    x10 = max(x10, rel)
                wabs = max(wabs, aerr * w)
        return {"raw": raw, "x_le_5": x5, "x_le_10": x10, "emit_weighted_abs": wabs}

    taylor_in = scan(neq_taylor_branch_f64, in_window=True)
    full_out = scan(neq_full_branch_f64, in_window=False)
    full_in = scan(neq_full_branch_f64, in_window=True)

    # AS-USED error: the branch the code actually selects at each (x, rho).
    # Answers plan item R4.1.2(c): is there a region where the SELECTED branch
    # is inaccurate?  Bucketed by x; also the worst-case (x, rho) location.
    au_all = au_x5 = au_x10 = 0.0
    worst = None
    for x in xs:
        for drho in drhos:
            rho = 1.0 + drho
            oracle = neq_oracle(x, rho)
            if oracle == 0:
                continue
            fn = neq_taylor_branch_f64 if abs(drho) < 0.01 else neq_full_branch_f64
            rel = float(abs(mp.mpf(fn(x, rho)) - oracle) / abs(oracle))
            au_all = max(au_all, rel)
            if x <= 5.0:
                au_x5 = max(au_x5, rel)
            if x <= 10.0:
                au_x10 = max(au_x10, rel)
            if worst is None or rel > worst[0]:
                worst = (rel, x, drho)

    return {
        "dps": dps,
        "n_points": len(xs) * len(drhos),
        "taylor_where_used": taylor_in,   # |rho-1|<0.01 : branch IS used
        "full_where_used": full_out,      # |rho-1|>=0.01 : branch IS used
        "full_inside": full_in,           # cancellation region Taylor replaces
        "as_used_maxrel": au_all,
        "as_used_maxrel_x_le_5": au_x5,
        "as_used_maxrel_x_le_10": au_x10,
        "as_used_worst": {"rel": worst[0], "x": worst[1], "rho_minus_1": worst[2]},
    }


def _linspace(a, b, n):
    a, b = mp.mpf(a), mp.mpf(b)
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def part3_perturbative_rho_eq(dps: int) -> dict:
    """Pitfall #4: perturbative Delta rho_eq vs full I4/(4G3) noise floor.

    Uses a representative log grid [1e-4, 30], midpoint (half-cell) quadrature
    matching solver.rs::update_temperatures weights:
      delta_g3 += x_h^3 * dn_mid * dx
      delta_i4 += x_h^3 * x_h * (2 n_pl + 1) * dn_mid * dx
      exact_g3 += x_h^3 * n_full * dx
      exact_i4 += x_h^3 * x_h * n_full (1+n_full) * dx
    perturbative: delta_i4/(4 G3_const) - delta_g3/G3_const
    full:         exact_i4/(4 exact_g3) - 1
    """
    G3 = 6.493_939_402_266_829  # G3_PLANCK constant used by solver

    # log grid, N points
    N = 2000
    lo, hi = math.log(1e-4), math.log(30.0)
    x = [math.exp(lo + (hi - lo) * i / (N - 1)) for i in range(N)]

    def routes_f64(delta_n):
        dg3 = di4 = eg3 = ei4 = 0.0
        for i in range(1, N):
            dx = x[i] - x[i - 1]
            xh = 0.5 * (x[i] + x[i - 1])
            x3 = xh ** 3
            dn_mid = 0.5 * (delta_n[i] + delta_n[i - 1])
            npl = 0.5 * (f64_planck(x[i]) + f64_planck(x[i - 1]))
            dg3 += x3 * dn_mid * dx
            di4 += x3 * xh * (2.0 * npl + 1.0) * dn_mid * dx
            nf = max(npl + dn_mid, 0.0)
            eg3 += x3 * nf * dx
            ei4 += x3 * xh * nf * (1.0 + nf) * dx
        pert = di4 / (4.0 * G3) - dg3 / G3
        full = ei4 / (4.0 * eg3) - 1.0 if eg3 > 1e-30 else 0.0
        return pert, full

    def rho_eq_oracle(shape, eps):
        """Continuum rho_eq-1 = int x^4 n(1+n) / (4 int x^3 n) - 1 at high dps."""
        mp.mp.dps = dps
        npl = lambda t: 1 / (mp.e ** t - 1)
        n = lambda t: npl(t) + mp.mpf(eps) * shape(t)
        I4 = _quad_0_inf(lambda t: t ** 4 * n(t) * (1 + n(t)))
        G3c = _quad_0_inf(lambda t: t ** 3 * n(t))
        return I4 / (4 * G3c) - 1

    # Planck (Delta n = 0): full route noise floor; perturbative must be exact 0
    zero = [0.0] * N
    pert0, full0 = routes_f64(zero)

    # A small y-type distortion: Delta n = eps * x e^x/(e^x-1)^2 * (x/4 - 1)
    #   (the standard Compton-y occupation shape Y_SZ, unnormalised).
    eps = 1e-5

    def y_shape_mp(t):
        e = mp.e ** t
        gbb = t * e / (e - 1) ** 2
        return gbb * (t / 4 - 1)

    def y_shape_f64(t):
        e = math.exp(t)
        gbb = t * e / (e - 1) ** 2
        return gbb * (t / 4 - 1)

    dn_y = [eps * y_shape_f64(t) for t in x]
    pert_y, full_y = routes_f64(dn_y)
    true_y = float(rho_eq_oracle(y_shape_mp, eps))

    return {
        "dps": dps,
        "grid": f"log [1e-4,30], N={N}, midpoint",
        "planck_full_noise_floor": full0,
        "planck_pert": pert0,
        "y_eps": eps,
        "y_true_continuum": true_y,
        "y_pert_f64": pert_y,
        "y_full_f64": full_y,
        "y_pert_err_vs_true": abs(pert_y - true_y),
        "y_full_err_vs_true": abs(full_y - true_y),
    }


def part4_kompaneets_flux(dps: int) -> dict:
    """Pitfall #1: split flux vs naive FD flux for Planck + T_e=T_z.

    kompaneets.rs:60,68  split source (phi=1 => (phi-1) term = 0):
      F_split = x_h^4 [ ddn/dx + (phi-1) n_pl(1+n_pl) + phi(2n_pl+1)dn + phi dn^2 ]
    For Delta n = 0, phi = 1:  F_split = 0 analytically (every term vanishes).
    Naive form keeps dn_pl/dx by finite difference:
      F_naive = x_h^4 [ (n_pl[i+1]-n_pl[i])/dx + n_pl_half(1+n_pl_half) ]
    which does NOT cancel to 0 -> O(dx^2) residual.  The physical signal that
    the split form must resolve is O(rho_e-1) ~ 1e-5.
    """
    N = 2000
    lo, hi = math.log(1e-4), math.log(30.0)
    x = [math.exp(lo + (hi - lo) * i / (N - 1)) for i in range(N)]

    # The physical flux the split form must resolve is the (phi-1) source with
    # a realistic rho_e-1 ~ 1e-5:  F_signal(x) = x^4 (rho-1) n_pl(1+n_pl).
    # The naive form's spurious residual competes with THIS x-dependent flux,
    # not a flat scalar. Report both the max absolute spurious flux and the
    # max pointwise ratio spurious/signal.
    rho_minus_1 = 1e-5
    max_split = 0.0
    max_naive = 0.0
    max_ratio = 0.0
    for i in range(N - 1):
        xh = 0.5 * (x[i] + x[i + 1])
        dx = x[i + 1] - x[i]
        npl_h = f64_planck(xh)
        # split, phi=1, dn=0: ddn/dx=0, source=(1-1)*..=0, drift terms *dn=0
        f_split = xh ** 4 * (0.0 + 0.0 * npl_h * (npl_h + 1.0) + 0.0 + 0.0)
        # naive: FD of n_pl + analytic n_pl(1+n_pl)  (should cancel, doesn't)
        ddn_pl = (f64_planck(x[i + 1]) - f64_planck(x[i])) / dx
        f_naive = xh ** 4 * (ddn_pl + npl_h * (npl_h + 1.0))
        f_signal = xh ** 4 * rho_minus_1 * npl_h * (npl_h + 1.0)
        max_split = max(max_split, abs(f_split))
        max_naive = max(max_naive, abs(f_naive))
        if f_signal != 0:
            max_ratio = max(max_ratio, abs(f_naive) / abs(f_signal))

    return {
        "grid": f"log [1e-4,30], N={N}",
        "max_split_flux_planck": max_split,
        "max_naive_flux_planck": max_naive,
        "rho_minus_1_signal": rho_minus_1,
        "max_ratio_naive_over_signal": max_ratio,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dps", type=int, default=60)
    ap.add_argument("--part", type=int, default=0, help="0 = all parts")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    if args.part in (0, 1):
        r = part1_constants(args.dps)
        results["part1_constants"] = r
        print(f"\n=== Part 1: constants.rs vs {args.dps}-digit defining integrals ===")
        print(f"{'constant':<12} {'digits agree':>12}  {'rel err':>12}")
        for row in r["rows"]:
            print(f"{row['name']:<12} {row['digits_agree']:>12}  {row['rel_err']:>12}")
        # Stability re-check at higher dps (verify quadrature converged)
        r80 = part1_constants(args.dps + 20)
        drift = []
        for a, b in zip(r["rows"], r80["rows"]):
            da, db = mp.mpf(a["oracle"]), mp.mpf(b["oracle"])
            reld = abs(da - db) / abs(db) if db != 0 else mp.mpf(0)
            drift.append((a["name"], mp.nstr(reld, 3)))
        results["part1_stability_dps+20"] = drift
        print(f"\n  stability (dps {args.dps} vs {args.dps+20}, should be ~1e-{args.dps}):")
        for name, d in drift:
            print(f"    {name:<12} {d}")

    if args.part in (0, 2):
        r = part2_dcbr_cancellation(args.dps)
        results["part2_dcbr_cancellation"] = r
        print(f"\n=== Part 2: DC/BR near-cancellation neq = n_pl(x/rho)-n_pl(x) ===")
        print(f"  grid: {r['n_points']} (x, rho) points, {args.dps}-digit oracle")
        tu, fo, fi = r["taylor_where_used"], r["full_where_used"], r["full_inside"]
        w = r["as_used_worst"]
        print(f"  Taylor (USED |rho-1|<0.01): rel raw={tu['raw']:.2e}  x<=5={tu['x_le_5']:.2e}  x<=10={tu['x_le_10']:.2e}")
        print(f"  Full   (USED |rho-1|>=.01): rel raw={fo['raw']:.2e}  x<=5={fo['x_le_5']:.2e}")
        print(f"  Full   (INSIDE window, cancellation region Taylor replaces): rel raw={fi['raw']:.2e}")
        print(f"  AS-USED (selected branch): raw={r['as_used_maxrel']:.2e}  x<=5={r['as_used_maxrel_x_le_5']:.2e}  x<=10={r['as_used_maxrel_x_le_10']:.2e}")
        print(f"    worst: rel={w['rel']:.2e} at x={w['x']:.3g}, rho-1={w['rho_minus_1']:.2e}")

    if args.part in (0, 3):
        r = part3_perturbative_rho_eq(args.dps)
        results["part3_perturbative_rho_eq"] = r
        print(f"\n=== Part 3: perturbative Delta rho_eq vs full I4/(4G3) ===")
        print(f"  grid: {r['grid']}")
        print(f"  Planck (dn=0): full-route noise floor = {r['planck_full_noise_floor']:.3e}, pert = {r['planck_pert']:.3e}")
        print(f"  y-dist eps={r['y_eps']:.0e}: true(continuum)={r['y_true_continuum']:.4e}")
        print(f"    perturbative f64 = {r['y_pert_f64']:.4e}  (err {r['y_pert_err_vs_true']:.2e})")
        print(f"    full route   f64 = {r['y_full_f64']:.4e}  (err {r['y_full_err_vs_true']:.2e})")

    if args.part in (0, 4):
        r = part4_kompaneets_flux(args.dps)
        results["part4_kompaneets_flux"] = r
        print(f"\n=== Part 4: Kompaneets flux splitting (Planck, T_e=T_z) ===")
        print(f"  grid: {r['grid']}")
        print(f"  max |split flux|  = {r['max_split_flux_planck']:.3e}  (analytic 0)")
        print(f"  max |naive flux|  = {r['max_naive_flux_planck']:.3e}  (FD dn_pl/dx residual)")
        print(f"  max pointwise ratio naive/physical-signal (rho-1={r['rho_minus_1_signal']:.0e}) = {r['max_ratio_naive_over_signal']:.0f}x")

    with open(os.path.join(OUT_DIR, "oracle.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {os.path.join(OUT_DIR, 'oracle.json')}")


if __name__ == "__main__":
    main()
