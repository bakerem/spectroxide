# Python API & Documentation Audit — 2026-07-08

**Scope**: `python/spectroxide/` (all 10 modules), README.md Python sections, `docs/*.rst`
(sources), CLAUDE.md §Python package, `pyproject.toml`, notebook usage as evidence.
**Method**: (A) scripted doc-accuracy checks — every code example executed, `inspect.signature`
vs NumPy-docstring `Parameters` diff over the full public surface; (B) manual API-consistency
review; (C) three fresh-eyes agents restricted to public docs (README + `help()` + `docs/*.rst`,
no source access) attempting canonical tasks. **Report-only: nothing was changed.**

Severity legend: **DOC-WRONG** (doc states something false), **DOC-MISSING** (needed doc absent),
**API-INCONSISTENT** (behavior/naming inconsistency), **API-PROPOSAL** (improvement, not a defect).
Findings ranked most-severe first within each section.

---

## 1. Critical doc defects (DOC-WRONG)

### 1.1 README quick-start decay example is physically broken
`README.md:100-103`:
```python
result = solve(
    injection={"type": "decaying_particle", "f_x": 5e5, "gamma_x": 5e4},
    z_start=5e6, z_end=1e3,
)
```
`gamma_x` is Γ_X in **1/s** (`src/energy_injection.rs:45-46`, `docs/api/solver.rst:71`).
`gamma_x = 5e4 s⁻¹` → lifetime 2×10⁻⁵ s, while `cosmic_time(5e6) ≈ 9.5×10⁵ s`: the particle has
fully decayed ~10 orders of magnitude in time before `z_start`. Executed verbatim, the example
returns μ = −2.3×10⁻⁹, y = −8.8×10⁻¹⁰, Δρ/ρ = −4.7×10⁻⁹ — numerical noise (adiabatic-cooling
residual), sign-flipped, no warning raised. Control: identical call with `gamma_x=1e-10` gives
Δρ/ρ = 6.6×10⁻⁶, μ = 1.5×10⁻⁶, y = 2.7×10⁻⁶ — the example is dead by ~3 orders of magnitude
in signal, not marginally off.
A plausible intent was a decay near z ~ 5×10⁴, i.e. Γ ~ 10⁻¹⁰ s⁻¹; as written the value looks
like a redshift pasted into a rate slot. This is the flagship copy-pasteable README example.

### 1.2 `docs/api/greens.rst` unit comment wrong by 10⁶ (Jy/sr vs MJy/sr)
`docs/api/greens.rst:250`:
```python
dI = delta_n_to_delta_I(x, dn)            # intensity in MJy/sr
```
and prose at `greens.rst:237` ("to intensity units (MJy/sr)"). The Python function returns
**Jy/sr** (`greens.py:1885-1886, 1896`: `di_jy = di_si / 1e-26`), as its own docstring correctly
states. Verified numerically (fresh-eyes run: raw peak 310.9 = 3.1×10⁻⁴ MJy/sr for ΔN/N = 10⁻⁶,
physically sane only in Jy/sr). A user trusting the .rst comment is off by six orders of
magnitude with no error. Root cause is finding 3.2 (Rust and Python converters use different
units); the .rst comment matches the Rust convention, not the Python function it annotates.

### 1.3 `docs/api/greens.rst` worked heating example is outside the linear regime
`docs/api/greens.rst:172-178`:
```python
dq_dz = lambda z: 1e-7 * np.exp(-1e-15 * cosmic_time(z))
mu = mu_from_heating(dq_dz, z_min=1e3, z_max=5e6)
```
∫dq/dz dz ≈ 10⁻⁷ × 5×10⁶ ≈ 0.5, i.e. Δρ/ρ ~ 0.5. Executed: **μ = 0.231**, y = 1.9×10⁻³ —
order-unity μ from a formalism that assumes μ ≪ 1, with no warning. Amplitude should be
~10⁻¹² for a Δρ/ρ ~ 10⁻⁵ demonstration. (Same snippet also uses the legacy import path
`from spectroxide.greens import cosmic_time`; see 3.4.)

### 1.4 `docs/api/solver.rst` required-keys table wrong in two rows
`docs/api/solver.rst:69-77`:
- `"monochromatic_photon"` row lists required keys `x_inj, delta_n_over_n, z_h, sigma_z, sigma_x`.
  Verified: `solve(injection={"type": "monochromatic_photon", "x_inj": 1.0,
  "delta_n_over_n": 1e-6, "z_h": 5e5})` succeeds with neither `sigma_z` nor `sigma_x`;
  `docs/cli.rst:42-43` correctly shows `[--sigma-x]` optional and no `sigma_z` at all.
  `sigma_z` appears to be copy-pasted from the `single_burst` row.
- `"single_burst"` row lists `z_h, sigma_z` as required. Verified: the `__init__.py` quick-start
  `solve(injection={'type': 'single_burst', 'z_h': 2e5}, delta_rho=1e-5)` runs without `sigma_z`.
  The table conflates "required" with "accepted".

### 1.5 `solve()` docstring names a nonexistent `intensity` property
`solver.py:1347`: "Returns a structured :class:`SolverResult` with frequency grid, distortion,
``μ``, ``y``, and an ``intensity`` property." `SolverResult` has no `intensity` attribute
(verified `AttributeError`); the property is `delta_I` (`solver.py:1300`), which the same
docstring's Returns section names correctly at `solver.py:1436`. Two contradictory names in
one docstring.

### 1.6 `docs/api/solver.rst` oversells `SolverResult` ΔT/T content
The page's summary sentence says `SolverResult` "bundles ... scalar μ/y/ΔT/T components", but the
class has no ΔT/T fit component — only `accumulated_delta_t`, a narrow PDE-only quantity
(temperature shift absorbed from photon-number non-conservation), which is 0.0 in ordinary runs.
Fresh-eyes agent used it as "the ΔT/T", got 0.0, and only caught the error by cross-checking
`decompose_distortion` (dT = 9.3×10⁻⁸ for the same run). Two different quantities both read
naturally as "ΔT/T" with no cross-reference; the actual decomposition route
(`decompose_distortion`) is not mentioned in README or `solver.rst` at all (see 4.3).

---

## 2. Other doc defects

### 2.1 DOC-WRONG — stale executed warning recommends a nonexistent function
`notebooks/paper_figures/cosmotherm_comparison.ipynb` (output cell, 1 occurrence): captured
`UserWarning` text "Use greens_function_with_cosmo or the cosmo-aware run_single (from
spectroxide import greens_function_with_cosmo, run_single)". `greens_function_with_cosmo` does
not exist anywhere in the package (grep of `python/`, `src/` — zero hits). The output predates
the current `_validation.py`; today's low-z warning (`_validation.py:357-366`) recommends the
PDE solver and fires at z < 1100, not the old z < 5000. Fix is a notebook re-execution, not a
code change. (This was the only dead API name found by the full notebook/docs sweep — a
mechanical check of every `spectroxide` identifier referenced in notebooks/README/.rst against
the package found no others.)

### 2.2 DOC-MISSING — CLAUDE.md §Python package omits `cosmology.py`
`CLAUDE.md:65-76` lists 10 files but not `cosmology.py` (691 lines; source of `Cosmology`,
the three presets, and 11 top-level exports). All other CLAUDE.md §Python claims verified
accurate: helper names exist (`solver.py:276,320`), `greens_function(x, z_h)` has no cosmology
kwarg, `greens_function_photon` takes `cosmo=`. Minor: `__init__.py` re-exports one `strip_gbb`
function; CLAUDE.md's "`strip_gbb*`" glob implies several.

### 2.3 DOC-MISSING — no Sphinx API pages for `cosmotherm`, `dark_photon`, `plot_params`
No `docs/api/{cosmotherm,dark_photon,plot_params}.rst`. `cosmotherm` (11 public functions) and
`dark_photon` (5) are actively imported by notebooks; `dark_photon` is the documented route to
reproduce dark-photon numbers. All 66 `__all__` names do appear somewhere in the .rst sources,
and all 50 autodoc directives resolve to real objects (verified by import).

### 2.4 DOC-MISSING — packaged distribution ships no long description
`pyproject.toml:9`: `readme` commented out ("outside package dir"). An sdist/wheel has an empty
PyPI description. Standard fix (when desired): `readme = {text = ...}` or copy README at build
time. Related: the `dev` extra (`pyproject.toml:36`) is identical to `notebook` and omits
`pytest`/`mutmut`, which the test/mutation workflow needs.

### 2.5 DOC-MISSING — two docstring gaps (the only ones on the whole surface)
Scripted diff over every `__all__` callable, all submodule-only publics, and all public methods
of `FIRASData`/`GreensTable`/`PhotonGreensTable`/`SolverResult` found exactly two gaps:
- `FIRASData.chi2_diagonal(model_kJy)` — no Parameters section (`firas.py`).
- `FIRASData.profile_limit_floating_T` — `use_diagonal` kwarg undocumented.
Everything else matches: zero undocumented params, zero ghost params, dataclass fields fully
documented via Attributes. Note: the pre-audit exploration claim that `FIRASData.__init__`
params are undocumented was **wrong** — `t_cmb`/`t_dust` are documented in the class docstring
(NumPy convention).

### 2.6 DOC-STALE — `dev/audit/census/census_python.json`
Lists three deleted test files (`test_anisotropy.py`, `test_dm_baryon.py`, `test_fh_basis.py`)
and omits two existing ones (`test_adversarial_inputs.py`, `test_literature_curves.py`).

### 2.7 DOC-NOTE — `SolverResult.z_h` is `None` for monochromatic photon injection
Docstring (`solver.py:1278-1280`): "*None* for continuous injection or custom heating
histories." Photon injection at fixed `z_h` is a burst, was passed `z_h=5e5`, yet
`result.z_h is None` (verified at runtime). Either populate it or document the actual rule
("single_burst only").

---

## 3. API inconsistencies (API-INCONSISTENT)

### 3.1 `cosmo=` accepts a `Cosmology`; `cosmo_params=` crashes on one
`solve(cosmo=Cosmology.planck2018())` works (`solver.py:1322` normalizes), but
`run_sweep(cosmo_params=Cosmology.planck2018())` raises
`TypeError: argument of type 'Cosmology' is not iterable`
(`solver.py:196` in `_build_cosmo_args`, via `run_sweep` → `_build_common_solver_args`).
Same concept, two names (`cosmo` in `solve`/all GF functions; `cosmo_params` in
`run_sweep`/`run_photon_sweep`/`run_photon_sweep_batch`/table builders), and only one path
normalizes the dataclass. The error message never says "pass a dict / use .to_dict()".
**Proposal**: accept both spellings everywhere (deprecation alias) or at minimum route
`cosmo_params` through the same normalizer `solve` uses. This is the highest-impact usability
fix available.

### 3.2 Intensity units differ between Rust and Python converters
Rust `distortion.rs:436` outputs **MJy/sr**; Python `delta_n_to_delta_I` outputs **Jy/sr**
(`greens.py:1885`). Both docstrings are individually correct, but the split caused defect 1.2
and makes any cross-language comparison a silent ×10⁶ trap. **Proposal**: standardize one unit
(or add `delta_n_to_delta_I_mjy` / a `unit=` kwarg) in a future breaking pass; until then, add
an explicit unit note wherever `delta_I` appears.

### 3.3 Three naming conventions for the same injection scenario
README table (`README.md:194-204`) uses Rust CamelCase (`MonochromaticPhotonInjection`), the
Python dict key is `"monochromatic_photon"` (documented only in `docs/api/solver.rst:76`), and
the Rust CLI error reports kebab-case (`monochromatic-photon`). A fresh-eyes agent's first
attempt used the README's CamelCase and got
`RuntimeError: ... Unknown injection type: 'MonochromaticPhotonInjection'`.
**Proposal**: add a "Python `type` key" column to the README table; optionally have the Python
wrapper translate CamelCase/kebab-case to the canonical key.
(Verified-consistent, for the record: README's "9 built-in scenarios" is correct — 10 Rust
variants minus `Custom`; `solve()` reaches 9 of them: 7 dict types + `dq_dz`/`photon_source`.)

### 3.4 Legacy re-export shim in `greens.py` duplicates the cosmology surface
`greens.py:659-696` re-imports ~30 names (public: `hubble`, `cosmic_time`,
`ionization_fraction`, presets; private: `_C_LIGHT`, `_cosmo_hubble`, …) from `cosmology` for
back-compat. Consequences: every cosmology function is importable from two module paths;
`COSMOTHERM_GF_COSMO` is importable but in no `__all__`; and the *official docs still teach the
legacy path* (`docs/api/greens.rst:174`: `from spectroxide.greens import cosmic_time`).
Consumers of the legacy path: `docs/api/greens.rst`, `notebooks/paper_figures/
dark_photon_constraints.ipynb` (also imports six private names through it).
**Proposal**: keep the shim, but (a) switch docs/notebooks to canonical paths, (b) add a
comment date/removal condition to the shim block.

### 3.5 Return-type mix across entry points
`solve()` → `SolverResult`; `run_sweep`/`run_photon_sweep`/`run_single` → raw `dict`;
`run_photon_sweep_batch` → `list[dict]`; related analysis functions return float
(`mu_from_heating`), tuple (`GreensTable.mu_y_from_heating`), array (`distortion_from_heating`),
dict (`decompose_distortion`). All individually documented, but a user must re-learn the access
pattern per function (`result.mu` vs `r["pde_mu"]` vs tuple unpacking).
**Proposal** (breaking-change candidate, not for now): sweep functions return
`list[SolverResult]` or a thin `SweepResult`.

### 3.6 `_validation` warnings vs binary stdout share one channel
Rust binary progress lines are surfaced as Python `RuntimeWarning`s
(e.g. `RuntimeWarning: spectroxide: Progress: z=2.5e5 ...`), the same channel as genuine
physics-validity warnings (ρ_e clamping, regime warnings). Users must visually filter noise
from signal. **Proposal**: route `Progress:` lines to logging/DEBUG, keep warnings for
actionable conditions.

### 3.7 Minor consistency notes (catalogue, no action urged)
- Only `__init__.py` defines `__all__`; the 9 submodules' public surface is naming-convention
  only. Per-module `__all__` would pin intent (relevant to 3.4).
- `timeout` default drifts across `run_*`: 600 s vs 3600 s for batch — each documented
  correctly; fine, but worth knowing.
- The ~10 solver-quality kwargs (`dy_max`, `n_points`, `dtau_max`, `dtau_max_photon_source`,
  `number_conserving`, `nc_z_min`, `no_dcbr`, `production_grid`, `n_threads`) are duplicated
  verbatim across all five `run_*`/`solve` signatures (17–20 params each). Scripted check:
  **no default drift** among them — the copies are currently in sync. A `QualitySettings`
  dataclass would remove the risk (API-PROPOSAL).
- Naming grab-bag: `z_h`/`z_injections`/`z_start`/`z_min`; `x`/`x_inj`/`x_obs`/`x_grid`
  (the `x` vs `x_min`/`x_max`/`n_x` dual path is fine — explicitly documented "Ignored when
  x is provided"); unit suffixes in `firas.py` method names (`predict_kJy`, `chi2_diagonal`);
  casing mixes `f_ann_CT`, `f_x_eV` (`cosmotherm.py`), `m_ev` (`dark_photon.py`).
- **Refuted lead**: there are *not* two `Cosmology` classes — `solver.Cosmology is
  cosmology.Cosmology` (single class, re-imported).

---

## 4. Fresh-eyes usability findings (Pass C)

Three agents (no repo context, public docs only) attempted: (a) decaying-particle μ/y vs FIRAS,
(b) monochromatic photon injection → MJy/sr spectrum, (c) PDE sweep + μ/y/ΔT decomposition.
All three completed their tasks; findings below are the friction that remained after
1.1–1.6/3.1–3.3 (their biggest blockers) are accounted for.

### 4.1 No fast-path story for scenario injections
README advertises "9 built-in injection scenarios" adjacent to the GF fast path, but the GF
route (`run_single`, `solve(method="greens_function")`) accepts only `z_h` (single burst) or a
raw `dq_dz` callable. Worse, `solve(method="greens_function", injection={...})` **silently
ignores** the `injection` dict and fails with
`ValueError: Either z_h (single burst) or dq_dz (custom heating) is required` — even when the
dict contains `z_h`. **Proposal**: error explicitly ("injection= is PDE-only; for GF pass
z_h=/dq_dz=") or translate the dict where feasible; state PDE-only-ness in `solve()`'s
`injection` docstring.

### 4.2 `f_x` normalization undefined in user docs
Both scenario tables give "`f_x` [eV]" with no statement of *per what* (it is energy released
per baryon, per `src/energy_injection.rs:43` — a doc comment users can't see). Blocks informed
parameter choice; users must cargo-cult examples, two of which are broken (1.1, 1.3).

### 4.3 μ/y/ΔT decomposition is undiscoverable from README/solver docs
`decompose_distortion` is mentioned in neither README nor `docs/api/solver.rst`; the sweep
docs print `pde_mu`/`pde_y` only. Agent found it via `dir(spectroxide)`. One README line and a
`See Also` on `run_sweep`/`SolverResult` would fix it. (Its own docstring is good — it
pre-empted a `z_h`-kwarg misuse with a clear warning.)

### 4.4 `FIRASData.upper_limit_mu()` default ≠ literature limit
Default `marginalise_y=True` gives 1.61×10⁻⁴ vs the cited |μ| < 9×10⁻⁵ (`MU_FIRAS_95`); the
docstring warns about this explicitly (good), but nothing short of reading the full docstring
surfaces it, and `MU_FIRAS_95`/`upper_limit_mu()` disagreeing by 1.8× invites inconsistent
pass/fail conclusions. **Proposal**: cross-reference the constants in the method docstring's
first line, not only in the buried warning.

### 4.5 No documented bridge from solver output to FIRAS χ²
`FIRASData.chi2` wants a (43,) array in kJy/sr on the FIRAS grid; solver output is Δn on its
own grid, `delta_I` in Jy/sr. No documented interpolation/unit path connects them (agent
hand-rolled `np.interp` + /1000). A `FIRASData.chi2_from_solver(result)` helper or a doc
example would close the gap.

### 4.6 Positive observations (for calibration)
All three tasks were completable within a 2-minute compute budget using documented knobs
(`debug=True`, `run_sweep` ~12 s for 3 redshifts). Runtime warnings (photon-injection grid
resolution, ρ_e clamping) were rated genuinely helpful. `decompose_distortion`, once found,
matched PDE-native μ/y to all printed digits; GF vs PDE photon-injection μ agreed to ~3%,
consistent with the documented bound.

---

## 5. Executed-example scorecard (Pass A1)

| Example | Result |
|---|---|
| `__init__.py` docstring quick-start (3 calls + submodule imports) | ✅ all run |
| README "Python: Green's function" block | ✅ runs, values sane |
| README "Python: PDE solver" block | ⚠️ runs, but decay example returns noise (1.1) |
| `docs/api/greens.rst` heating example | ❌ unphysical output, no warning (1.3) |
| `docs/api/greens.rst` decomposition example comment | ❌ unit comment wrong ×10⁶ (1.2) |
| `docs/api/solver.rst` decay example (`gamma_x=1e-15`) | ⚠️ runs; emits ρ_e-clamped-31× linearity warning |
| Doctest-style `>>>` examples (6 total, `cosmology.py`, `firas.py`) | ✅ pass |

## 6. Suggested fix order (all deferred; report-only)

1. README decay example parameters (1.1) — one-line change, flagship example.
2. `greens.rst` MJy/sr comment + heating amplitude (1.2, 1.3).
3. `solver.rst` required-keys rows + `intensity`→`delta_I` in `solve()` docstring (1.4, 1.5).
4. README scenario table: add Python `type`-key column (3.3); one line on `decompose_distortion` (4.3).
5. `cosmo_params` dataclass normalization or clearer TypeError (3.1).
6. The rest as convenient; breaking-change proposals (3.2, 3.5) for a future major version.
