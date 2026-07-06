#!/usr/bin/env python3
"""Cross-code comparison: spectroxide vs CLASS spectral-distortion (sd) module.

Workstream R1 of dev/PLAN_VALIDATION_ROUND2_2026-07-06.md — the like-for-like
independent-code check for the HEAT-INJECTION half of the paper. CLASS `sd`
(Lucca, Schoneberg, Hooper, Lesgourgues & Chluba 2020, JCAP 02 (2020) 026,
arXiv:1910.04619) computes mu, y from heating histories via
Green's-function/branching-ratio methods — a different group, language, and
numerical approach.

CLASS build: v3.3.0, commit 0ceb7a9, at /home/bakerem/CLASS (binary prebuilt).
Cosmology matched to Cosmology::default() (Chluba 2013 / CosmoTherm):
  h=0.71, Omega_b=0.044, Omega_m=0.26 -> omega_b=0.0221836, omega_cdm=0.1088856,
  T_cmb=2.726, Y_p=0.24, N_eff=3.046 (N_ur=3.046, N_ncdm=0).

Case A (this script): ADIABATIC LCDM. CLASS computes mu, y from its internal
heating history (acoustic dissipation + adiabatic cooling + recombination) with
`sd_branching_approx = exact` (Chluba's Green's data in external/distortions/
Greens_data.dat -> an *indirect CosmoTherm* comparison). We then feed CLASS's
OWN heating history `_sd_heating.dat` (column d(Q/rho)/dz, same convention as
spectroxide's TabulatedHeating rate_table) into the spectroxide PDE via the
`solve tabulated-heating` subcommand. Because both codes thermalise the SAME
heating history, the heating-history match (directive R1.4) is exact BY
CONSTRUCTION, so the mu/y comparison isolates the *thermalisation* numerics.

Cases B (decay), C (s-wave annihilation), D (mu/y transfer) additionally require
the CLASS injection unit mapping (DM_decay_Gamma / DM_annihilation_efficiency
-> spectroxide gamma_x,f_x / f_ann), derived and heating-history-verified before
comparison — scaffolded here, see `run_case_decay` TODO.

Run:  python dev/scripts/class_sd_compare.py --case A
Outputs: dev/output/class_sd/  (CLASS ini/outputs + spectroxide heating CSV)
         printed ratio table; JSON summary dev/output/class_sd/comparison.json
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
import numpy as np

CLASS_DIR = "/home/bakerem/CLASS"
CLASS_BIN = os.path.join(CLASS_DIR, "class")
SPX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(SPX_ROOT, "dev", "output", "class_sd")

# Chluba-2013 cosmology in CLASS parameterisation
COSMO_CLASS = {
    "h": 0.71,
    "omega_b": 0.044 * 0.71**2,          # 0.0221836
    "omega_cdm": (0.26 - 0.044) * 0.71**2,  # 0.1088856
    "T_cmb": 2.726,
    "YHe": 0.24,
    "N_ur": 3.046,
}


def run_class_lcdm(root_tag="A_lcdm"):
    """Run CLASS sd for adiabatic LCDM (no injection). Returns (mu, y, heating_path)."""
    os.makedirs(OUT, exist_ok=True)
    ini = os.path.join(OUT, f"{root_tag}.ini")
    lines = [
        "output = Sd, mPk",
        "sd_branching_approx = exact",
        "sd_only_exotic = no",
        f"root = {os.path.join(OUT, root_tag)}_",
        "overwrite_root = yes",
        "write distortions = yes",
        "distortions_verbose = 2",
    ]
    for k, v in COSMO_CLASS.items():
        lines.append(f"{k} = {v}")
    with open(ini, "w") as f:
        f.write("\n".join(lines) + "\n")

    proc = subprocess.run([CLASS_BIN, ini], cwd=CLASS_DIR,
                          capture_output=True, text=True, timeout=600)
    out = proc.stdout + proc.stderr
    mu = _grep_float(out, r"mu-parameter\s*=\s*([-\d.eE+]+)")
    y = _grep_float(out, r"y-parameter\s*=\s*([-\d.eE+]+)")
    heat = os.path.join(OUT, f"{root_tag}__sd_heating.dat")
    if mu is None or not os.path.exists(heat):
        print("CLASS output:\n", out[-2000:])
        raise RuntimeError("CLASS sd run failed to produce mu/y or heating file")
    return mu, y, heat


def _grep_float(text, pat):
    m = re.search(pat, text)
    return float(m.group(1)) if m else None


def class_heating_to_spx_csv(heat_path, csv_path):
    """Convert CLASS _sd_heating.dat (z, Heat=d(Q/rho)/dz) to spectroxide
    tabulated-heating CSV (z ascending, columns z,dq_dz). Same convention:
    both are d(Delta rho/rho)/dz, positive = heating."""
    data = np.loadtxt(heat_path)
    z, dqdz = data[:, 0], data[:, 1]
    order = np.argsort(z)
    z, dqdz = z[order], dqdz[order]
    with open(csv_path, "w") as f:
        f.write("z,dq_dz\n")
        for zi, r in zip(z, dqdz):
            f.write(f"{zi:.10e},{r:.10e}\n")
    return z.min(), z.max()


def run_spx_tabulated(csv_path, z_start, z_end, n_points=4000):
    """Run spectroxide PDE on the tabulated heating history. Returns parsed JSON."""
    cmd = [
        "cargo", "run", "--release", "--bin", "spectroxide", "--",
        "solve", "tabulated-heating",
        "--heating-table", csv_path,
        "--delta-rho", "1e-5",  # ignored for tabulated (table carries amplitude)
        "--z-start", str(z_start),
        "--z-end", str(z_end),
        "--n-points", str(n_points),
        "--format", "json",
    ]
    proc = subprocess.run(cmd, cwd=SPX_ROOT, capture_output=True, text=True, timeout=1200)
    # JSON is on stdout; find the JSON object
    txt = proc.stdout
    start = txt.find("{")
    if start < 0:
        print("spectroxide stderr:\n", proc.stderr[-2000:])
        print("spectroxide stdout:\n", txt[-2000:])
        raise RuntimeError("spectroxide produced no JSON")
    return json.loads(txt[start:])


def run_case_A():
    print("=== Case A: adiabatic LCDM (CLASS heating -> spectroxide PDE) ===")
    mu_c, y_c, heat = run_class_lcdm()
    print(f"CLASS (sd, exact branching):  mu = {mu_c:.4e}   y = {y_c:.4e}")

    csv = os.path.join(OUT, "A_lcdm_heating_spx.csv")
    zmin, zmax = class_heating_to_spx_csv(heat, csv)
    print(f"heating history: z in [{zmin:.4g}, {zmax:.4g}] (fed to spectroxide verbatim)")

    res = run_spx_tabulated(csv, z_start=min(zmax, 5e6), z_end=max(zmin, 1.0))
    mu_s = _dig(res, "mu")
    y_s = _dig(res, "y")
    print(f"spectroxide PDE:              mu = {mu_s:.4e}   y = {y_s:.4e}")
    print(f"ratio spx/CLASS:              mu = {mu_s/mu_c:.4f}   y = {y_s/y_c:.4f}")

    summary = {
        "case": "A_adiabatic_lcdm",
        "class_version": "v3.3.0 (0ceb7a9)",
        "class_branching": "exact (Greens_data.dat)",
        "cosmology": COSMO_CLASS,
        "mu_class": mu_c, "y_class": y_c,
        "mu_spx": mu_s, "y_spx": y_s,
        "ratio_mu": mu_s / mu_c, "ratio_y": y_s / y_c,
        "note": "heating history identical by construction; comparison isolates thermalisation",
    }
    with open(os.path.join(OUT, "comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {os.path.join(OUT, 'comparison.json')}")
    return summary


# --- Case B: decaying particle (unit mapping DERIVED, R1.4) ---------------
# CLASS  (external/heating/injection.c:748):
#   energy_rate = rho_cdm(z) * DM_decay_fraction * DM_decay_Gamma * exp(-Gamma t)
# spectroxide (energy_injection.rs DecayingParticle):
#   dE/dt = f_x[J] * gamma_x * n_H(z) * exp(-gamma_x t)
# Both ~ (1+z)^3 exp(-Gamma t) with the SAME Gamma, so gamma_x = DM_decay_Gamma
# directly, and the amplitude match (rho_cdm(z)=Om_cdm rho_crit (1+z)^3,
# n_H(z)=(1-Yp) Om_b rho_crit (1+z)^3 / m_p) gives (rho_crit cancels):
#   DM_decay_fraction = f_x[eV] * eV_J * (1-Yp) * Om_b / (m_p * Om_cdm)
# VERIFY by matching the two heating histories d(Delta rho/rho)/dz to <0.1%
# BEFORE comparing mu/y (directive R1.4). Only then is the mu/y comparison
# meaningful.
_EV_J = 1.602_176_634e-19
_M_P = 1.672_621_923_69e-27  # kg


def decay_fraction_for_spx(f_x_ev, y_p=0.24, omega_b=0.044, omega_cdm=None):
    if omega_cdm is None:
        # Om_cdm = omega_cdm / h^2 with the matched cosmology
        omega_cdm = COSMO_CLASS["omega_cdm"] / COSMO_CLASS["h"] ** 2
    return f_x_ev * _EV_J * (1.0 - y_p) * omega_b / (_M_P * omega_cdm)


def run_case_decay(gamma_x=1.1e-10, f_x_ev=7.8e5):
    """Case B — decaying particle. Paper Fig-4 values: gamma_x=1.1e-10/s,
    f_x=7.8e5 eV. TODO(run): needs CLASS + PDE builds (serialise vs other heavy
    builds on this 7GB box, see ROUND2_STATUS.md). Steps:
      1. f_dec = decay_fraction_for_spx(f_x_ev); write CLASS ini with
         DM_decay_Gamma=gamma_x, DM_decay_fraction=f_dec, sd_only_exotic=yes.
      2. Run CLASS -> mu,y + _sd_heating.dat.
      3. Run spectroxide `solve decaying-particle --gamma-x --f-x`, and ALSO
         export its heating history; assert the two d(Delta rho/rho)/dz overlap
         to <0.1% (the mapping gate). Only then compare mu,y.
    Deep-mu-era decay (z_X ~ 1e6) is the UNAMBIGUOUS check: mu=1.401 drho, y~0.
    """
    f_dec = decay_fraction_for_spx(f_x_ev)
    print(f"Case B mapping: gamma_x={gamma_x:.3e}/s, f_x={f_x_ev:.3e} eV "
          f"-> CLASS DM_decay_fraction={f_dec:.4e}, DM_decay_Gamma={gamma_x:.3e}")
    print("  (heating-history match gate not yet run — see run_case_decay TODO)")
    return {"gamma_x": gamma_x, "f_x_ev": f_x_ev, "class_DM_decay_fraction": f_dec,
            "class_DM_decay_Gamma": gamma_x, "status": "mapping_derived_run_pending"}


def _dig(res, key):
    """Pull mu/y out of the spectroxide JSON (schema-tolerant)."""
    if key in res:
        return float(res[key])
    for sub in ("distortion", "result", "snapshot", "final"):
        if sub in res and isinstance(res[sub], dict) and key in res[sub]:
            return float(res[sub][key])
    # deep search
    found = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == key and isinstance(v, (int, float)):
                    found.append(float(v))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(res)
    if found:
        return found[0]
    raise KeyError(f"{key} not found in spectroxide JSON; keys={list(res.keys())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="A", choices=["A", "B"])
    args = ap.parse_args()
    if args.case == "A":
        run_case_A()
    elif args.case == "B":
        run_case_decay()
