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
`solve tabulated-heating` subcommand.

*** READ THIS BEFORE QUOTING ANY CASE A NUMBER. ***
"Both codes thermalise the SAME heating history" is only true with
`--subtract-cooling`. CLASS's table already contains first-order adiabatic
cooling of photons on baryons, and the spectroxide PDE models that same cooling
internally and unconditionally (the Lambda*rho_e term). Feeding the table in
verbatim counts it twice — roughly half of the apparent PDE-vs-GF mu gap that
finding R1-A was built on. The original Case A run and R1-A are RETRACTED; see
dev/audit/class_sd_comparison.md. `--subtract-cooling` reconstructs CLASS's own
cooling term and removes it, and self-checks the reconstruction against the
literature mu ~ -3e-9 for pure cooling.

Cases B (decay), C (s-wave annihilation), D (mu/y transfer) additionally require
the CLASS injection unit mapping (DM_decay_Gamma / DM_annihilation_efficiency
-> spectroxide gamma_x,f_x / f_ann), derived and heating-history-verified before
comparison — scaffolded here, see `run_case_decay` TODO.

Run:  python dev/scripts/class_sd_compare.py --case A --subtract-cooling
Outputs: dev/output/class_sd/  (CLASS ini/outputs + spectroxide heating CSV)
         printed ratio table; JSON summary comparison_nocool.json
         (or comparison.json for the retracted verbatim configuration)
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
    """Run CLASS sd for adiabatic LCDM (no injection).

    Returns (mu, y, heating_path, thermo_path). `write thermodynamics = yes` is
    requested because x_e(z) is needed to reconstruct CLASS's own adiabatic-
    cooling term (see class_adiabatic_cooling_dqdz).
    """
    os.makedirs(OUT, exist_ok=True)
    ini = os.path.join(OUT, f"{root_tag}.ini")
    lines = [
        "output = Sd, mPk",
        "sd_branching_approx = exact",
        "sd_only_exotic = no",
        f"root = {os.path.join(OUT, root_tag)}_",
        "overwrite_root = yes",
        "write distortions = yes",
        "write thermodynamics = yes",
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
    thermo = os.path.join(OUT, f"{root_tag}__thermodynamics.dat")
    if not os.path.exists(thermo):
        thermo = None
    return mu, y, heat, thermo


def _grep_float(text, pat):
    """`pat` must capture the number with a proper float group — the sloppy
    [-\\d.eE+]+ charset swallows trailing sentence periods."""
    m = re.search(pat, text)
    return float(m.group(1).rstrip(".")) if m else None


# --- adiabatic-cooling double count (RETRACTION of Case A, 2026-07-30) -------
#
# CLASS's non-injection heating table INCLUDES first-order adiabatic cooling of
# photons on baryons:
#
#   external/heating/noninjection.c:197  noninjection_table[i] += dEdt(cooling)
#   external/heating/noninjection.c:313  *energy_rate = -heat_capacity*H*T_g
#   source/distortions.c:862             dQrho_dz_tot[i] = heat*a/(H*rho_g)
#
# and _sd_heating.dat is that total. The spectroxide PDE models adiabatic
# cooling ITSELF, unconditionally, via the Lambda*rho_e electron-temperature
# term — it cannot be switched off. Feeding the CLASS table in verbatim
# therefore counts the cooling twice. It is invisible to eyeball inspection of
# the file because acoustic dissipation dominates the total at every z, so no
# entry ever turns negative.
#
# Reconstruct the cooling column in the file's own units, exactly as CLASS
# builds it (H cancels between the rate and the dz Jacobian):
#
#   dQ/dz|_cool = -(3/2) k_B n_H(z) (1 + f_He + x_e(z)) T_g(z) / ((1+z) rho_g(z))
#
# with n_H(z) = n_H0 (1+z)^3, f_He = YHe/(_not4_ (1-YHe)), T_g = T_cmb (1+z),
# rho_g = a_rad T_g^4, and x_e(z) taken from CLASS's own thermodynamics output
# so the subtraction is self-consistent with the run being corrected.
_K_B = 1.380_649e-23              # J/K
_A_RAD = 7.565_733e-16            # J m^-3 K^-4  (= 4 sigma_SB / c)
_NOT4 = 3.9715                    # CLASS _not4_: He/H mass ratio
_M_H = 1.673_575_756_0e-27        # kg, CLASS _m_H_


def _n_h0(omega_b, h, y_he):
    """Hydrogen nucleus number density today [1/m^3], CLASS's pth->n_e."""
    # rho_crit,0 = 3 H0^2 c^2 / (8 pi G) expressed via the standard
    # 1.878e-26 h^2 kg/m^3, then the hydrogen mass fraction.
    rho_crit0 = 1.878_371_1e-26 * h**2          # kg/m^3
    return omega_b / h**2 * rho_crit0 * (1.0 - y_he) / _M_H


def _read_class_xe(thermo_path):
    """Return (z, x_e) from a CLASS _thermodynamics.dat, by header column name."""
    header = None
    with open(thermo_path) as f:
        for line in f:
            if line.startswith("#") and ":" in line:
                header = line
            elif not line.startswith("#"):
                break
    if header is None:
        raise RuntimeError(f"no column header found in {thermo_path}")
    # header looks like:
    #   "#  1:scale factor a   2:z   3:conf. time [Mpc]   4:x_e   5:kappa' [Mpc^-1] ..."
    # Split on the "<n>:" markers; names may themselves contain digits
    # ("[Mpc^-1]"), so anchor on the index tokens, not on the name charset.
    parts = re.split(r"(?:^|\s{2,})(\d+):", header.lstrip("# ").rstrip())
    names = {}
    for idx, name in zip(parts[1::2], parts[2::2]):
        names[name.strip().lower()] = int(idx) - 1
    try:
        iz = names["z"]
        ixe = next(i for n, i in names.items() if n.startswith("x_e"))
    except (KeyError, StopIteration) as exc:
        raise RuntimeError(
            f"could not find z / x_e columns in {thermo_path}; parsed {sorted(names)}"
        ) from exc
    data = np.loadtxt(thermo_path)
    z, xe = data[:, iz], data[:, ixe]
    order = np.argsort(z)
    return z[order], xe[order]


def class_adiabatic_cooling_dqdz(z, thermo_path, cosmo=None):
    """CLASS's own adiabatic-cooling contribution to the _sd_heating.dat column,
    evaluated on the grid `z`. Negative (cooling). Same units/convention as the
    file: dimensionless d(Q/rho_gamma)/dz."""
    c = dict(COSMO_CLASS if cosmo is None else cosmo)
    y_he, h, t0 = c["YHe"], c["h"], c["T_cmb"]
    f_he = y_he / (_NOT4 * (1.0 - y_he))
    n_h = _n_h0(c["omega_b"], h, y_he) * (1.0 + z) ** 3
    z_xe, xe_tab = _read_class_xe(thermo_path)
    # thermodynamics.dat stops at recombination-ish z; above the last tabulated
    # point the gas is fully ionised, so hold the highest-z value (x_e ~ 1+f_He).
    xe = np.interp(z, z_xe, xe_tab, left=xe_tab[0], right=xe_tab[-1])
    t_g = t0 * (1.0 + z)
    rho_g = _A_RAD * t_g**4
    heat_capacity = 1.5 * _K_B * n_h * (1.0 + f_he + xe)      # J/(K m^3)
    return -heat_capacity * t_g / ((1.0 + z) * rho_g)


def class_heating_to_spx_csv(heat_path, csv_path, subtract_cooling=False,
                             thermo_path=None):
    """Convert CLASS _sd_heating.dat (z, Heat=d(Q/rho)/dz) to spectroxide
    tabulated-heating CSV (z ascending, columns z,dq_dz). Same convention:
    both are d(Delta rho/rho)/dz, positive = heating.

    subtract_cooling=True removes CLASS's adiabatic-cooling term so the table
    carries only the physics the spectroxide PDE does NOT already model. Without
    it the comparison double-counts adiabatic cooling (see the block comment
    above and the retraction in dev/audit/class_sd_comparison.md)."""
    data = np.loadtxt(heat_path)
    z, dqdz = data[:, 0], data[:, 1]
    order = np.argsort(z)
    z, dqdz = z[order], dqdz[order]
    cool = None
    if subtract_cooling:
        if thermo_path is None:
            raise ValueError("subtract_cooling needs the CLASS _thermodynamics.dat "
                             "(add 'write thermodynamics = yes' to the .ini)")
        cool = class_adiabatic_cooling_dqdz(z, thermo_path)
        dqdz = dqdz - cool          # cool < 0, so this REMOVES a negative term
    with open(csv_path, "w") as f:
        f.write("z,dq_dz\n")
        for zi, r in zip(z, dqdz):
            f.write(f"{zi:.10e},{r:.10e}\n")
    return z.min(), z.max(), cool, z


def cooling_only_selfcheck(z, cool):
    """Sanity-check the reconstructed cooling term against the literature.

    Pure adiabatic cooling in LCDM gives mu ~ -3e-9 (Chluba 2011; Khatri,
    Sunyaev & Chluba 2012), reproduced in-repo by
    tests/cosmotherm_comparison.rs::test_adiabatic_cooling_mu_vs_cosmotherm
    against CosmoTherm's DI_cooling.dat. Push the reconstructed column alone
    through the Green's function and check we land there. This guards against a
    units or normalisation slip in the reconstruction, which would otherwise
    silently mis-correct the comparison."""
    from spectroxide import mu_from_heating, y_from_heating
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    drho = float(trapz(cool, z))

    def dq_dz(zq):
        return np.interp(zq, z, cool, left=0.0, right=0.0)

    mu = float(mu_from_heating(dq_dz, z.min(), z.max()))
    y = float(y_from_heating(dq_dz, z.min(), z.max()))
    print(f"  cooling-only: integral d(rho)/rho = {drho:.4e}"
          f"   mu = {mu:.4e}   y = {y:.4e}   (literature mu ~ -3e-9)")
    return {"cooling_drho_over_rho": drho, "cooling_mu": mu, "cooling_y": y}


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


def run_case_A(subtract_cooling=False):
    print("=== Case A: adiabatic LCDM (CLASS heating -> spectroxide PDE) ===")
    if not subtract_cooling:
        print("  !! WARNING: running WITHOUT --subtract-cooling reproduces the")
        print("  !! RETRACTED configuration: CLASS's table already contains")
        print("  !! adiabatic cooling, which the spectroxide PDE also models")
        print("  !! internally, so the cooling is counted twice (~half of the")
        print("  !! apparent PDE-vs-GF mu gap). See dev/audit/class_sd_comparison.md.")
    mu_c, y_c, heat, thermo = run_class_lcdm()
    print(f"CLASS (sd, exact branching):  mu = {mu_c:.4e}   y = {y_c:.4e}")

    tag = "A_lcdm_heating_spx_nocool.csv" if subtract_cooling else "A_lcdm_heating_spx.csv"
    csv = os.path.join(OUT, tag)
    zmin, zmax, cool, zgrid = class_heating_to_spx_csv(
        heat, csv, subtract_cooling=subtract_cooling, thermo_path=thermo)
    how = "with adiabatic cooling REMOVED" if subtract_cooling else "verbatim"
    print(f"heating history: z in [{zmin:.4g}, {zmax:.4g}] (fed to spectroxide {how})")
    cool_check = cooling_only_selfcheck(zgrid, cool) if cool is not None else None

    res = run_spx_tabulated(csv, z_start=min(zmax, 5e6), z_end=max(zmin, 1.0))
    # persist the raw solver JSON immediately so a schema hiccup in _dig
    # cannot lose a multi-minute PDE run
    raw_name = "A_spx_full_nocool.json" if subtract_cooling else "A_spx_full.json"
    with open(os.path.join(OUT, raw_name), "w") as f:
        json.dump(res, f)
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
        "subtract_cooling": subtract_cooling,
        "cooling_selfcheck": cool_check,
        "note": ("adiabatic cooling removed from the CLASS table; the PDE supplies "
                 "it internally, so this is the apples-to-apples configuration"
                 if subtract_cooling else
                 "RETRACTED CONFIGURATION: adiabatic cooling double-counted "
                 "(present in the CLASS table AND in the PDE). Do not quote."),
    }
    out_name = "comparison_nocool.json" if subtract_cooling else "comparison.json"
    with open(os.path.join(OUT, out_name), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {os.path.join(OUT, out_name)}")
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
    # CLASS's rho_cdm in injection.c is an ENERGY density [J/m^3] (the
    # original derivation above forgot the c^2 — caught by the Case B
    # heating-history gate, see class_sd_case_b.py).
    c2 = 2.997_924_58e8 ** 2
    if omega_cdm is None:
        # Om_cdm = omega_cdm / h^2 with the matched cosmology
        omega_cdm = COSMO_CLASS["omega_cdm"] / COSMO_CLASS["h"] ** 2
    return f_x_ev * _EV_J * (1.0 - y_p) * omega_b / (_M_P * c2 * omega_cdm)


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
    """Pull mu/y out of the spectroxide JSON (schema-tolerant).

    Current schema (output.rs): {"results":[{"pde_mu":..., "pde_y":...}], ...};
    older layouts had bare "mu"/"y" at various depths. Accept both the bare
    and the "pde_"-prefixed key anywhere in the tree."""
    keys = (key, f"pde_{key}")
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
                if k in keys and isinstance(v, (int, float)):
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
    ap.add_argument("--subtract-cooling", action="store_true",
                    help="remove CLASS's adiabatic-cooling term before handing the "
                         "heating table to the PDE (the PDE models cooling itself; "
                         "without this the comparison double-counts it)")
    args = ap.parse_args()
    if args.case == "A":
        run_case_A(subtract_cooling=args.subtract_cooling)
    elif args.case == "B":
        run_case_decay()
