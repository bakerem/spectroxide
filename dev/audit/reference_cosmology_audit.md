# B4-3 Audit — Reference-comparison cosmology parameters

**Date:** 2026-07-05
**Scope:** every site where the codebase compares against an external reference
(CosmoTherm DI files / GF database, Chluba 2013 & 2015 GF fits, CCJ24 dark-photon
limits, Fixsen 1996 FIRAS, Bolliet+2020 Gaunt, Procopio & Burigana 2009,
Arsenadze+2025), checked against the cosmology the reference actually used.
**Method:** reference cosmologies taken from primary sources — raw ar5iv text of
arXiv:1304.6120 and 1506.06582, the vendored `Greens.v1.0.3/Greens.cpp`, the
`data/cosmotherm/DI_*.dat` headers, the FIRAS monopole file header, and arXiv
metadata for the dark-photon papers — not from repo comments.

**Identification note:** CCJ24 = Chluba, Cyr & Johnson (2024), *Revisiting Dark
Photon Constraints from CMB Spectral Distortions*, is **arXiv:2409.12115** (the
audit plan's working notes cited 2402.03431, which is an unrelated
reactor-neutrino paper; repo code cites 12115 correctly). Arsenadze et al.
(2025) = arXiv:2409.12940.

---

## Reference cosmologies (verified from primary sources)

| Reference | Cosmology it used | Source of that statement |
|---|---|---|
| Chluba 2013 (1304.6120) GF fits | Y_p=0.24, Ω_m=0.26, Ω_b=0.044, Ω_Λ=0.74, h=0.71, N_eff=3.046, T₀=2.726 | paper §3 raw text ("We adopted the following cosmological parameters…") |
| Chluba 2015 (1506.06582) photon-injection GF | "standard cosmology" (same CosmoTherm standard as C13); x_c, y_γ≈4.9e-11(1+z)² consistent with ω_b≈0.022 | paper §2 raw text |
| CosmoTherm Greens.v1.0.3 GF database | Yp=0.24, Omega0=0.26 (**total** matter), OmegaB=0.044, h=0.71, **Nnu=3.04**, T0_CMB=2.726 | `Greens.cpp:104-112`, `Greens.h:41` |
| CosmoTherm DI files (cooling/damping/y_late/CRR) | Yp=0.2467, T0=2.726, Om=0.264737, Ob=0.049169, h=0.6727, Neff=3.046 = Planck 2015 (1502.01589, TT,TE,EE+lowP) with **Om = Ω_cdm** | file headers; 0.264737 = 0.1198/0.6727² and 0.049169 = 0.02225/0.6727² exactly |
| CCJ24 (2409.12115) | **not stated numerically** in the paper; uses "full numerical results from CosmoTherm" | paper raw text (searched: no parameter list) |
| Fixsen et al. 1996 FIRAS | cosmology-independent data; monopole file residuals defined w.r.t. a **2.725 K** blackbody | `data/firas_monopole_spec_v1.txt` header |
| Bolliet+2020 / Chluba, Ravenni & Bolliet 2020 Gaunt | cosmology-independent (atomic physics, g_ff(x, θ_e)) | — |
| Procopio & Burigana 2009 | benchmarks used here are analytic thermodynamic relations (φ_BE, μ–Δε) — cosmology-independent | — |
| Arsenadze+2025 Eq. D13–D16 | analytic functions of (x′, y_γ); cosmology enters only through the caller's y_γ | — |

On the DI-header "Om" convention: `Greens.cpp` treats its `Omega0` as **total**
matter (`OmegaL = 1 − Omega0 − …`), but the DI headers' Om=0.264737 reproduces
Planck 2015's Ω_cdm to 6 digits (total Ω_m would be 0.3156). The headers also
claim "Planck 2015" explicitly, so Om there must mean Ω_cdm. The Rust
`planck2015` preset adopts this interpretation and is exactly consistent.
Residual risk: if the CosmoTherm run in fact fed 0.264737 into a total-matter
slot, its low-z H(z) would be ~9 % higher than the preset's — the ~% level
DI_cooling/DI_damping residuals do not show a systematic of that size, so the
Ω_cdm reading is also supported empirically.

---

## Comparison-site table

| # | Site | Reference | Cosmology used by code | Verdict |
|---|---|---|---|---|
| 1 | `tests/cosmotherm_comparison.rs:126,268,414,479,643` | CosmoTherm DI_cooling/DI_damping (Planck 2015 run) | `Cosmology::planck2015()` = T=2.726, ω_b=0.02225, ω_cdm=0.1198, h=0.6727, N_eff=3.046, Y_p=0.2467 | ✅ exact match (all 7 params, incl. CT T₀ convention) |
| 2 | `tests/greens_function_checks.rs:34,264` | Chluba 2013 GF-fit limits; Bolliet Gaunt spot checks | `Cosmology::default()` = C13 params exactly | ✅ exact match (Gaunt part cosmology-independent) |
| 3 | `tests/heat_injection.rs` §5, §21, §30, §33 (~136 × `Cosmology::default()`) | Chluba 2013 Eq. 5/6, Chluba 2015 Eq. 30, CS2012 regime boundaries | default = C13 paper cosmology | ✅ match (C15 "standard cosmology" = same CosmoTherm standard) |
| 4 | `tests/heat_injection.rs:837` (`test_firas_limits_consistency`), FIRAS-limit assertions | Fixsen 1996 95 % CL | n/a | ✅ cosmology-independent |
| 5 | `tests/heat_injection.rs` §36 (P&B 2009), §BR/DC (CosmoTherm ln(2.25/xZ), Bolliet softplus) | P&B 2009, Bolliet+2020 | any | ✅ cosmology-independent oracles |
| 6 | `tests/heat_injection.rs:10135` (`test_solver_respects_cosmology_parameters`) | internal default-vs-planck2018 direction check | both presets | ✅ internal, no external ref |
| 7 | `tests/science_suite.rs:26,148,225` | Chluba 2013 Eq. 6 limits; Chluba & Thomas 2011 Peebles TLA (F=1.125) | `Cosmology::default()` | ✅ match |
| 8 | `notebooks/paper_figures/cosmotherm_comparison.ipynb`, `dm_scenario_comparison.ipynb`, `pathological_heating.ipynb`; `dev/notebooks/pde_greens_function.ipynb`, `pde_validation.ipynb` | CosmoTherm GF database (Greens.v1.0.3) | GF convolution: `COSMOTHERM_GF_COSMO` (default + n_eff=3.04) — matches `Greens.cpp` exactly; PDE side: `DEFAULT_COSMO` (n_eff=3.046) | ✅ match / documented ΔN_eff=0.006 (ΔΩ_rel ~0.1 %, negligible) |
| 9 | `notebooks/physics/adiabatic_cooling.ipynb` cell[3] | CosmoTherm DI_cooling.dat (T0=2.726 convention) | `Cosmology.planck2015().to_dict()` → **t_cmb=2.7255**, should be `PLANCK2015_COSMO` (t_cmb=2.726) | ⚠ minor mismatch (wrong preset pick; ~0.07 % in Jy/sr amplitude, 0.04 % in x-scale) |
| 10 | `notebooks/paper_figures/dark_photon_constraints.ipynb` (Fig. 8) cells[1,3,7,16] | CCJ24 digitized CosmoTherm limits (`dev/data/cosmotherm_dp_lims.csv`) | `DEFAULT_COSMO` (h=0.71, C13) for z_res/γ_con/gc_per_eps2; `PLANCK2018_COSMO` imported but unused | ⚠ unverifiable — CCJ24 states no numbers; quantified below |
| 11 | `notebooks/observational/dp_firas_method_comparison.ipynb` | CCJ24 (2409.12115) | `FIRASData()` (t_cmb=2.726) + library-default templates | ⚠ same CCJ24 caveat as #10 |
| 12 | `notebooks/paper_figures/firas_photon_limits.ipynb`, `notebooks/observational/firas_photon_limits.ipynb` | FIRAS 68 % CL μ=4.5e-5, y=7.5e-6 (Chluba 2015 convention) | library default | ✅ limits cosmology-independent |
| 13 | `notebooks/physics/photon_injection*.ipynb`, `paper_figures/photon_injection_spectra.ipynb` | Chluba 2015 Figs. 2/5/7; Arsenadze Eq. C8 | library default; hard-coded T=2.726 for x↔GHz only | ✅ match |
| 14 | `dev/scripts/dm_cosmotherm_compare.py:37-47` | CosmoTherm GF database | explicit constants = C13 default, **n_eff=3.046** (GF DB used 3.04) | ✅ match / same negligible ΔN_eff as #8 |
| 15 | `dev/scripts/build_gf_table.py:57`, `build_visibility_table.py:71`, `remake_firas_photon_limits.py:62`, `plot_visibility_comparison.py`, `fit_visibility_from_table.py` | CosmoTherm GF DB; Chluba 2013 fit params; FIRAS limits | `run_sweep` with no `cosmo_params` → Rust default (C13) | ✅ match |
| 16 | `python/spectroxide/cosmotherm.py:43` (`T_CMB_DEFAULT=2.726`), `:559` (`COSMOTHERM_GF_COSMO` fallback) | CosmoTherm data conversions | 2.726 = `Greens.h` T0_CMB | ✅ exact |
| 17 | `python/spectroxide/firas.py:64` (`_T_CMB = 2.726 # (Fixsen & Mather 2002)`) | Fixsen 1996 monopole file (residuals w.r.t. **2.725 K** BB) | 2.726 for x-grid and kJy/sr conversions | ⚠ 0.037 % convention offset + wrong attribution (F&M 2002 give 2.725±0.001; 2.726 is the CosmoTherm/Chluba convention). Absorbed by floating-T fits; negligible vs FIRAS σ |
| 18 | `notebooks/paper_figures/visibility_functions.ipynb`, `mu_y_vs_injection_redshift.ipynb`, `dev/notebooks/*` | Chluba 2013 fit formulas | library default / `DEFAULT_COSMO` | ✅ match |
| 19 | rotti2022 / ec26 Planck-μ comparison | Rotti+2022 | lives on a private working branch only (main has stale `.pyc`/target artifacts); preset bug already covered by P0-5 | — out of scope here |

---

## Findings (triaged)

### Confirmed mismatches

**RC-1 — `adiabatic_cooling.ipynb` uses `Cosmology.planck2015()` (t_cmb=2.7255)
against DI_cooling.dat (T0=2.726).** The repo defines `PLANCK2015_COSMO`
specifically for CT comparisons and this notebook doesn't use it. Impact:
0.04 % in the frequency mapping, ~0.07 % in the Jy/sr amplitude — invisible at
the notebook's percent-level agreement, but it contradicts the preset's stated
purpose (P0-6). Fix: switch cell[3] to `PLANCK2015_COSMO` (or resolve P0-6
first and route through the CT preset).

### Documented / negligible convention differences

**RC-2 — CCJ24 comparison cosmology is unverifiable (dark_photon_constraints,
dp_firas_method_comparison).** CCJ24 (2409.12115) publishes no numeric
cosmology; it uses CosmoTherm's full numerics (plausibly Planck-era
parameters). The notebooks run z_res/γ_con on `DEFAULT_COSMO` (h=0.71).
Quantified with the repo's own functions: γ_con(default)/γ_con(planck2018) =
1.030 at m=1e-9 eV (z_res≈1569), 0.995 at 1e-6 eV, 0.994 at 1e-4 eV; z_res
shifts ≤0.2 %. Since ε_lim ∝ γ_con^{−1/2}, the induced offset in the ε limits
is ≤1.5 % — far below the plot's decade-scale axis and below the unresolved
~22 % γ_con discrepancy against the decoded Bryce figure (separate memo).
Action: state the cosmology caveat in the Fig. 8 notebook (the
`PLANCK2018_COSMO` import is currently dead code — either use it for a
sensitivity band or drop the import).

**RC-3 — N_eff 3.046 vs 3.04.** The Chluba 2013 *paper* says N_eff=3.046; the
shipped `Greens.cpp` uses Nnu=3.04. Repo default (3.046, both languages)
matches the paper; `COSMOTHERM_GF_COSMO` (3.04) matches the code and is used
for all GF-database convolutions. ΔΩ_rel ≈ 0.1 % → sub-0.05 % in μ/y. The
Rust default's docstring (`src/cosmology.rs:416`) claims "matches CosmoTherm
v1.0.3 … N_eff=3.046", which is wrong about the code (3.04); it matches the
C13 paper. Doc fix only. `dm_cosmotherm_compare.py:42` similarly uses 3.046
against the 3.04 database — same negligible class.

**RC-4 — FIRAS T convention (firas.py).** The monopole file's residual column
is defined against a 2.725 K blackbody; `firas.py` converts frequencies and
distortion templates at 2.726 K, and the comment misattributes 2.726 to
Fixsen & Mather 2002 (who give 2.725±0.001). 2.726 is the CosmoTherm/Chluba
convention. 0.037 % scale offset, degenerate with the floated ΔT in all
fitting paths (`profile_limit_floating_T`, trial-T range 2.720–2.732).
Doc fix; optionally expose t_cmb=2.725 for strict Fixsen-convention fits.

**RC-5 — DI-header "Om" labelling.** `data/cosmotherm/README.md` copies the
header as "Omega_m = 0.264737"; the value is Ω_cdm (see convention analysis
above). `tests/cosmotherm_comparison.rs:9` already says "Om_cdm=0.264737"
(correct). Doc fix in the data README to prevent a future reader from
"fixing" `planck2015` into a genuine mismatch.

### Verified exact (no action)

- `Cosmology::planck2015()` ↔ CosmoTherm DI headers: all seven parameters
  exact, including the T₀=2.726 CT convention (Rust side). P0-6's open
  decision only affects the *Python* `Cosmology.planck2015()`; the dict
  `PLANCK2015_COSMO` used for CT work is already correct.
- `Cosmology::default()` / `DEFAULT_COSMO` ↔ Chluba 2013 paper cosmology:
  verified against the paper's raw text (Y_p=0.24, Ω_m=0.26, Ω_b=0.044,
  h=0.71, N_eff=3.046, and T₀=2.726 via CosmoTherm). Every heat-injection
  literature benchmark and GF-fit check runs on this preset — consistent.
- Fixsen 1996 limit values, Bolliet+2020 Gaunt, P&B 2009 thermodynamic
  relations, Arsenadze D13–D16 broadening: cosmology-independent oracles.

## Bottom line

No reference comparison uses a *wrong* cosmology at a level that affects any
published number. One notebook picks the wrong preset variant (RC-1,
sub-0.1 %), the CCJ24 comparison cosmology cannot be verified from the paper
but is bounded to ≤1.5 % in ε (RC-2), and there are three documentation-level
convention errors (RC-3/4/5). Recommended actions are one-line notebook and
docstring fixes plus a data-README correction.
