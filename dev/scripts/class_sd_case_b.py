#!/usr/bin/env python3
"""Case B: clean deep-mu-era decaying particle, spectroxide vs CLASS sd.

The unambiguous cross-code check flagged in dev/audit/class_sd_comparison.md:
a particle decaying at z_X ~ 1e6 injects all its energy deep in the mu era,
where mu = 1.401 * J_bb(z) * (Delta rho/rho) and y ~ 0 -- no mu/y-split
ambiguity, no transition-era branching spread, no adiabatic-cooling hand-off
(each code supplies its own cooling; the injected channel is compared).

Gate (directive R1.4): before comparing mu/y, the two codes' heating
histories d(Delta rho/rho)/dz for the SAME (f_x, Gamma_X) must match to
<0.1% where the rate is significant. The spectroxide side is computed here
with the package's own cosmology functions (cosmic_time/n_hydrogen/rho_gamma
mirror src/cosmology.rs); the CLASS side is read from _sd_heating.dat with
sd_only_exotic=yes so the file contains ONLY the exotic-injection term.

Unit mapping (derived in class_sd_compare.py, verified by the gate here):
  CLASS:       dE/dt/dV = rho_cdm(z) * f_dec * Gamma * exp(-Gamma t)
  spectroxide: dE/dt/dV = f_x[J] * Gamma * n_H(z) * exp(-Gamma t)
  => f_dec = f_x[eV] * eV_J * (1-Yp) * Om_b / (m_H * Om_cdm)
  (CLASS converts baryon density to n_H with _m_H_, not m_p.)

Run: python dev/scripts/class_sd_case_b.py [--gamma-x 4.2e-8] [--f-x 100]
     [--skip-pde]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from class_sd_compare import (COSMO_CLASS, OUT, CLASS_BIN, CLASS_DIR, SPX_ROOT,
                              _grep_float, _dig, _M_H, _EV_J)

from spectroxide import cosmology as spxc
from spectroxide import mu_from_heating, y_from_heating


_C_LIGHT = 2.997_924_58e8  # m/s


def decay_fraction(f_x_ev):
    """CLASS DM_decay_fraction for a given spectroxide f_x [eV].

    CLASS's rho_cdm in injection.c is an ENERGY density [J/m^3], so the
    baryon mass density converts with a factor c^2:
      f_dec = f_x[J] * n_H0 / (rho_cdm,0 c^2-included)
            = f_x[J] * (1-Yp) * Om_b / (m_H * c^2 * Om_cdm)."""
    y_p = COSMO_CLASS["YHe"]
    om_b = COSMO_CLASS["omega_b"] / COSMO_CLASS["h"] ** 2
    om_cdm = COSMO_CLASS["omega_cdm"] / COSMO_CLASS["h"] ** 2
    return f_x_ev * _EV_J * (1.0 - y_p) * om_b / (_M_H * _C_LIGHT**2 * om_cdm)


def spx_dqdz(z, f_x_ev, gamma_x):
    """spectroxide's decay heating history d(Delta rho/rho)/dz, exactly the
    formula in src/energy_injection.rs (DecayingParticle) divided by H(1+z)."""
    z = np.asarray(z, dtype=float)
    t = np.array([spxc.cosmic_time(zi) for zi in z])
    rate = (f_x_ev * _EV_J * gamma_x * spxc.n_hydrogen(z)
            * np.exp(-gamma_x * t) / spxc.rho_gamma(z))       # d(drho/rho)/dt
    return rate / (spxc.hubble(z) * (1.0 + z))                 # -> per dz


def run_class_decay(gamma_x, f_dec, branching="exact", tag=None):
    tag = tag or f"B_decay_{branching}"
    ini = os.path.join(OUT, f"{tag}.ini")
    lines = [
        "output = Sd, mPk",
        f"sd_branching_approx = {branching}",
        "sd_only_exotic = yes",
        f"root = {os.path.join(OUT, tag)}_",
        "overwrite_root = yes",
        "write distortions = yes",
        "distortions_verbose = 2",
        f"DM_decay_Gamma = {gamma_x}",
        f"DM_decay_fraction = {f_dec}",
        "f_eff_type = on_the_spot",
        "f_eff = 1.",
    ]
    lines += [f"{k} = {v}" for k, v in COSMO_CLASS.items()]
    with open(ini, "w") as f:
        f.write("\n".join(lines) + "\n")
    p = subprocess.run([CLASS_BIN, ini], cwd=CLASS_DIR,
                       capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    mu = _grep_float(out, r"mu-parameter\s*=\s*([-\d.eE+]+)")
    y = _grep_float(out, r"y-parameter\s*=\s*([-\d.eE+]+)")
    heat = os.path.join(OUT, f"{tag}__sd_heating.dat")
    if mu is None:
        print(out[-2000:])
        raise RuntimeError(f"CLASS decay run ({branching}) failed")
    return mu, y, (heat if os.path.exists(heat) else None)


def history_gate(heat_path, f_x_ev, gamma_x, threshold=1e-3):
    """Compare CLASS's exotic-only heating column to spectroxide's analytic
    history. Returns (max relative deviation over significant region, arrays)."""
    data = np.loadtxt(heat_path)
    z, dq_class = data[:, 0], data[:, 1]
    order = np.argsort(z)
    z, dq_class = z[order], dq_class[order]
    dq_spx = spx_dqdz(z, f_x_ev, gamma_x)
    peak = dq_class.max()
    sig = dq_class > threshold * peak
    rel = np.abs(dq_spx[sig] / dq_class[sig] - 1.0)
    return rel.max(), z, dq_class, dq_spx, sig


def run_spx_pde(f_x_ev, gamma_x, z_start=4.9e6, z_end=500.0, n_points=2000):
    cmd = ["cargo", "run", "--release", "--bin", "spectroxide", "--",
           "solve", "decaying-particle",
           "--f-x", str(f_x_ev), "--gamma-x", str(gamma_x),
           "--z-start", str(z_start), "--z-end", str(z_end),
           "--n-points", str(n_points), "--format", "json"]
    p = subprocess.run(cmd, cwd=SPX_ROOT, capture_output=True, text=True,
                       timeout=3600)
    txt = p.stdout
    start = txt.find("{")
    if start < 0:
        print(p.stderr[-2000:])
        raise RuntimeError("spectroxide PDE produced no JSON")
    return json.loads(txt[start:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma-x", type=float, default=4.2e-8,
                    help="decay rate [1/s]; 4.2e-8 puts Gamma*t=1 at z~1e6")
    ap.add_argument("--f-x", type=float, default=100.0,
                    help="energy per baryon [eV]; 100 eV -> drho/rho ~ few e-8")
    ap.add_argument("--skip-pde", action="store_true")
    args = ap.parse_args()

    gx, fx = args.gamma_x, args.f_x
    f_dec = decay_fraction(fx)
    t1 = spxc.cosmic_time(1e6)
    print(f"Case B: Gamma_X = {gx:.3e}/s (Gamma*t(z=1e6) = {gx*t1:.3f}), "
          f"f_x = {fx:.3g} eV -> CLASS DM_decay_fraction = {f_dec:.6e}")

    # total injected energy, spectroxide convention
    zg = np.geomspace(1e3, 4.9e6, 4000)
    dq = spx_dqdz(zg, fx, gx)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    drho_tot = float(trapz(dq, zg))
    print(f"total injected Delta rho/rho (spx analytic) = {drho_tot:.4e}")

    # CLASS runs: exact + approximation sweep
    results = {}
    for br in ["exact", "soft_soft", "sharp_sharp"]:
        mu_c, y_c, heat = run_class_decay(gx, f_dec, branching=br)
        results[br] = {"mu": mu_c, "y": y_c}
        print(f"CLASS {br:12s}: mu = {mu_c:.4e}   y = {y_c:.4e}"
              f"   mu/1.401drho = {mu_c/(1.401*drho_tot):.4f}")

    # heating-history gate on the exact run's file
    mu_c, y_c, heat = run_class_decay(gx, f_dec, branching="exact")
    if heat is None:
        raise RuntimeError("CLASS did not write _sd_heating.dat")
    maxdev, z, dq_c, dq_s, sig = history_gate(heat, fx, gx)
    drho_class = float(trapz(dq_c, z))
    print(f"GATE: max |spx/CLASS - 1| over rate > 1e-3*peak: {maxdev:.2%} "
          f"({'PASS' if maxdev < 1e-3 else 'FAIL'} at 0.1%)")
    print(f"      integral CLASS = {drho_class:.4e}  vs spx = {drho_tot:.4e} "
          f"({drho_class/drho_tot - 1:+.2%})")

    # spectroxide GF on its own history
    def dq_dz(zq):
        return np.interp(zq, zg, dq, left=0.0, right=0.0)
    mu_gf = float(mu_from_heating(dq_dz, zg.min(), zg.max()))
    y_gf = float(y_from_heating(dq_dz, zg.min(), zg.max()))
    print(f"spectroxide GF:      mu = {mu_gf:.4e}   y = {y_gf:.4e}"
          f"   mu/1.401drho = {mu_gf/(1.401*drho_tot):.4f}")

    summary = {
        "case": "B_decay_deep_mu",
        "gamma_x": gx, "f_x_ev": fx, "class_DM_decay_fraction": f_dec,
        "drho_total_spx": drho_tot, "drho_total_class": drho_class,
        "history_gate_max_rel_dev": maxdev,
        "class": results, "spx_gf": {"mu": mu_gf, "y": y_gf},
    }

    if not args.skip_pde:
        res = run_spx_pde(fx, gx)
        mu_p, y_p = _dig(res, "mu"), _dig(res, "y")
        print(f"spectroxide PDE:     mu = {mu_p:.4e}   y = {y_p:.4e}"
              f"   mu/1.401drho = {mu_p/(1.401*drho_tot):.4f}")
        print(f"ratio PDE/CLASS(exact): mu = {mu_p/results['exact']['mu']:.4f}")
        summary["spx_pde"] = {"mu": mu_p, "y": y_p}

    with open(os.path.join(OUT, "comparison_case_b.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", os.path.join(OUT, "comparison_case_b.json"))


if __name__ == "__main__":
    main()
