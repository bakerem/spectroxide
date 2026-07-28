#!/usr/bin/env python3
"""Collect the R3 reference-solver run matrix into tables + a merged results.json.

Reads dev/refsolver/outputs/results*.json and prints:
  * the headline table (mu, y, dT/T per case)
  * the convergence table (grid halving/doubling, step halving/doubling)
  * the amplitude-linearity table (drho = 1e-3 vs 1e-5)
  * z_end = 200 vs 1, and BR on vs off
  * the photon-number ledger
It then writes the augmented outputs/results.json in place (adding the
convergence and sensitivity blocks to the baseline file).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
CASES = ["heat_z2e6", "heat_z2e5", "heat_z5e3", "adiabatic", "photon_x0.1_z3e5"]
VARIANTS = {
    "base": "results.json",
    "N1025": "results_N1025.json",
    "N4097": "results_N4097.json",
    "refine0.5": "results_refine0.5.json",
    "refine2": "results_refine2.json",
    "drho1e-5": "results_drho1e-5.json",
    "zend1": "results_zend1.json",
    "nobr": "results_nobr.json",
}


def load():
    d = {}
    for k, fn in VARIANTS.items():
        p = os.path.join(OUT, fn)
        if os.path.exists(p):
            with open(p) as fh:
                d[k] = json.load(fh)
    return d


def fit(d, v, c):
    try:
        return d[v]["cases"][c]["fit"]
    except KeyError:
        return None


def main():
    d = load()
    print("variants present:", ", ".join(sorted(d)))
    base = d["base"]

    print("\n" + "=" * 100)
    print("HEADLINE  (baseline: N=%d, refine=%g, z_end=%g, drho=%g, BR on)"
          % (base["config"]["N"], base["config"]["refine"],
             base["config"]["z_end"], base["config"]["drho"]))
    print("=" * 100)
    hdr = f"{'case':20s} {'mu':>14s} {'y':>14s} {'dT/T':>14s} {'resid/peak':>11s} {'nz':>7s} {'drho_meas':>11s}"
    print(hdr)
    for c in CASES:
        r = base["cases"][c]
        f = r["fit"]
        print(f"{c:20s} {f['mu']:+14.6e} {f['y']:+14.6e} {f['dT']:+14.6e} "
              f"{f['resid_rel']:11.2e} {r['nz']:7d} {r['drho_final_measured']:+11.3e}")

    print("\n" + "=" * 100)
    print("CONVERGENCE  (relative change vs baseline; error estimate = max|change| over the two refinements)")
    print("=" * 100)
    conv = {}
    for c in CASES:
        fb = base["cases"][c]["fit"]
        row = {}
        for v, lab in (("N1025", "grid/2"), ("N4097", "grid x2"),
                       ("refine0.5", "step x2 coarser"), ("refine2", "step /2 finer")):
            fv = fit(d, v, c)
            if fv is None:
                continue
            row[lab] = {k: (fv[k] - fb[k]) / abs(fb[k]) if fb[k] != 0 else float("nan")
                        for k in ("mu", "y", "dT")}
        conv[c] = row
        print(f"\n  {c}")
        print(f"    {'variant':18s} {'d mu/mu':>12s} {'d y/y':>12s} {'d(dT)/dT':>12s}")
        for lab, r in row.items():
            print(f"    {lab:18s} {r['mu']:+12.2e} {r['y']:+12.2e} {r['dT']:+12.2e}")
        err = {k: max(abs(r[k]) for r in row.values()) for k in ("mu", "y", "dT")} if row else {}
        if err:
            print(f"    {'-> error est.':18s} {err['mu']:12.2e} {err['y']:12.2e} {err['dT']:12.2e}")
            conv[c]["error_estimate_rel"] = err

    print("\n" + "=" * 100)
    print("AMPLITUDE LINEARITY  (mu, y, dT/T divided by drho/rho; drho=1e-3 vs 1e-5)")
    print("=" * 100)
    lin = {}
    if "drho1e-5" in d:
        print(f"  {'case':20s} {'quantity':10s} {'/1e-3':>14s} {'/1e-5':>14s} {'rel diff':>11s}")
        for c in CASES[:3]:
            f3 = base["cases"][c]["fit"]
            f5 = fit(d, "drho1e-5", c)
            if f5 is None:
                continue
            lin[c] = {}
            for k in ("mu", "y", "dT"):
                a, b = f3[k] / 1e-3, f5[k] / 1e-5
                rel = (b - a) / abs(a) if a != 0 else float("nan")
                lin[c][k] = dict(per_drho_1em3=a, per_drho_1em5=b, rel_diff=rel)
                print(f"  {c:20s} {k:10s} {a:+14.6e} {b:+14.6e} {rel:+11.2e}")

    print("\n" + "=" * 100)
    print("SENSITIVITY: z_end = 200 vs 1,   BR on vs off   (relative change)")
    print("=" * 100)
    sens = {}
    print(f"  {'case':20s} {'variant':10s} {'d mu/mu':>12s} {'d y/y':>12s} {'d(dT)/dT':>12s}")
    for c in CASES:
        fb = base["cases"][c]["fit"]
        sens[c] = {}
        for v, lab in (("zend1", "z_end=1"), ("nobr", "BR off")):
            fv = fit(d, v, c)
            if fv is None:
                continue
            r = {k: (fv[k] - fb[k]) / abs(fb[k]) if fb[k] != 0 else float("nan")
                 for k in ("mu", "y", "dT")}
            sens[c][lab] = r
            print(f"  {c:20s} {lab:10s} {r['mu']:+12.2e} {r['y']:+12.2e} {r['dT']:+12.2e}")

    print("\n" + "=" * 100)
    print("PHOTON LEDGER  (dN/N = trapz(x^2 dn dx)/(2 zeta(3)), uniform trapezoid, full x range)")
    print("=" * 100)
    pc = "photon_x0.1_z3e5"
    for v in ("base", "N4097", "refine2", "nobr", "N1025"):
        if v not in d or pc not in d[v]["cases"]:
            continue
        r = d[v]["cases"][pc]
        L = r["ledger"]
        print(f"  [{v}] N={r['N']} nz={r['nz']} x in [{r['xmin']:g}, {r['xmax']:g}]  "
              f"pts/sigma_x={L['pts_per_sigma_x']:.2f}")
        print(f"      (a) nominal dN/N                       = {L['dN_nominal']:.6e}")
        print(f"      (b) measured after window (z={L['z_after_window']:.4g})  = "
              f"{L['dN_after_window_trapz']:.6e}   (= {L['dN_after_window_trapz']/L['dN_nominal']:.5f} x nominal)")
        print(f"      (c) measured at z_end                  = "
              f"{L['dN_final_net_trapz']:.6e}   (= {L['surviving_fraction_vs_after_window']:.5f} x (b))")
        n = r["normalised_by_measured_dN"]
        print(f"      (d) mu={r['fit']['mu']:+.6e}  y={r['fit']['y']:+.6e}  dT/T={r['fit']['dT']:+.6e}")
        print(f"          mu/(b) = {n['mu_over_dN_after_window']:+.6f}   "
              f"mu/nominal = {n['mu_over_dN_nominal']:+.6f}")

    print("\n" + "=" * 100)
    print("FIT-WEIGHTING SENSITIVITY  (the contract's 'uniform weights on the grid' is")
    print("grid-dependent; on a log grid it is effectively w ~ 1/x over [0.5,18])")
    print("=" * 100)
    import numpy as np
    import refsolver as RS
    xr = np.linspace(0.5, 18.0, 1001)      # grid-free reference resampling
    print(f"  {'case':20s} {'q':4s} {'uniform-node':>14s} {'dx-weighted':>14s} "
          f"{'resampled-lin':>14s} {'max spread':>11s}")
    wsens = {}
    for c in CASES:
        r = base["cases"][c]
        p = os.path.join(OUT, f"spectrum_{c}.csv")
        rs = None
        if os.path.exists(p):
            d0 = np.genfromtxt(p, delimiter=",", names=True)
            dn = np.interp(xr, d0["x"], d0["delta_n"])
            rs = RS.decompose(dn, xr)
        wsens[c] = {}
        for k in ("mu", "y", "dT"):
            a, b = r["fit"][k], r["fit_dxweight"][k]
            cc = rs[k] if rs else float("nan")
            sp = max(abs(b - a), abs(cc - a)) / abs(a) if a != 0 else float("nan")
            wsens[c][k] = dict(uniform_node=a, dx_weighted=b, resampled_linear=cc,
                               max_rel_spread=sp)
            print(f"  {c:20s} {k:4s} {a:+14.6e} {b:+14.6e} {cc:+14.6e} {sp:11.2e}")
    base["fit_weighting_sensitivity"] = wsens

    print("\n" + "=" * 100)
    print("DIAGNOSTICS (baseline)")
    print("=" * 100)
    print(f"  {'case':20s} {'max dy/step':>12s} {'max|H_dcbr|':>12s} {'dri_max':>11s} "
          f"{'rho_e_max':>10s} {'capped':>7s} {'worstconv':>10s} {'ctl mu':>11s}")
    for c in CASES:
        r = base["cases"][c]
        print(f"  {c:20s} {r['max_dy_per_step']:12.3e} {r['max_abs_H_dcbr']:12.3e} "
              f"{r['dri_max']:11.3e} {r['rho_e_max']:10.5f} {r['n_newton_capped']:7d} "
              f"{r['worst_newton_conv']:10.1e} {r['fit_control_only']['mu']:+11.2e}")

    base["convergence"] = conv
    base["amplitude_linearity"] = lin
    base["sensitivity"] = sens
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(base, fh, indent=2, default=float)
    print("\nrewrote outputs/results.json with convergence/linearity/sensitivity blocks")


if __name__ == "__main__":
    main()
