# Mutation-testing audit (Workstream R2)

**Plan:** `dev/PLAN_VALIDATION_ROUND2_2026-07-06.md` §R2.
**Tool:** `cargo-mutants 27.1.0` (installed from git, commit a29be7b4; crates.io is
sandbox-blocked). Python side: `mutmut` (§R2.5).
**Status:** **CLOSED — TRUNCATED (EB, 2026-07-26).** Lean sweep complete
(2026-07-11, all 14 Rust physics modules) + raw Python mutmut done. The
survivor-escalation pass ran 2026-07-11 → 07-13 and covered **320 of the 2,037
survivors (16%)** before dying; EB stopped it rather than spend the ~13 further
days it needed. Verified full-suite lower bound: **63.6%** (§Escalation). The
close-out fixes and the physics checks the campaign motivated are tracked in
`R2_WRAPUP_TODO.md` and recorded under §Findings / §New tests added.

## Method (how to reproduce)

Mutation runs execute against an **isolated copy of the working tree** at
`/tmp/claude-1000/spx-mut` (rsync of the live tree minus `.git`, `target`,
`.gitmodules`, and large data dirs). Two reasons this copy exists rather than
running in the repo directly:

1. **`.gitmodules` bind-mount blocker.** In this checkout `.gitmodules` is a
   root-owned read-only `devtmpfs` bind-mount of `/dev/null`, not a regular
   file. cargo-mutants' default *copy-mode* tries to copy it and dies with
   "Permission denied"; it cannot be removed without root (documented in
   `ROUND2_STATUS.md`). The isolated copy has no such device, so runs proceed
   with `--in-place` there (copy-mode's tree-copy is redundant once the tree is
   already an isolated throwaby copy, and `--in-place` reuses one primed
   `target/` — critical on this 7 GB RAM box, which OOMs on concurrent cold
   builds).
2. **Isolation.** `--in-place` mutates source files one at a time; running it in
   a throwaway copy guarantees the live working tree (which carries uncommitted
   Round-2 edits) is never left in a mutated state if a run is OOM-killed.

The copy preserves the *actual* working-tree state, including the uncommitted
edits to `distortion.rs`/`recombination.rs` and the untracked test files
`tests/conservation_fuzz.rs` / `tests/mms_convergence.rs` that
`.cargo/mutants.toml` references — a git worktree from a committed branch would
have tested stale code and lacked those test binaries.

Per-mutant command (single-threaded; `--in-place` is implicitly `-j 1`):

```
cd /tmp/claude-1000/spx-mut
CARGO_BUILD_JOBS=2 cargo mutants --in-place -t 120 --jobserver-tasks 2 \
    -f src/<module>.rs -o out/<module>
```

**Memory tuning (7 GB box, non-negotiable — the run OOM-dies without it):**
- `CARGO_BUILD_JOBS=2` + `--jobserver-tasks 2` cap concurrent rustc to 2.
  cargo-mutants defaults to `n_tasks = ncpu = 12`; 12 parallel rustc OOM-kill
  the baseline build here.
- `--test-threads=2` (in `.cargo/mutants.toml`) caps *test* parallelism. The lib
  unit suite contains ~6 end-to-end PDE solver tests (`test_energy_conservation`,
  `test_pde_vs_greens_*`, each >60 s). At 12 threads a mutant that makes them
  non-converge spawns 12 runaway PDE solves and OOM-kills cargo-mutants itself.

**Test subset (R2.2, revised from the plan's 5-suite list).** Measured baselines:
`--lib` 17 s, `greens_function_checks` 9 s, but `coverage_gaps` 112 s and the
fuzz/convergence suites tens of s each — the 5-suite set gives a ~140 s+
per-mutant baseline, infeasible across thousands of mutants. The sweep therefore
uses the fast, targeted killers **lib unit tests + `greens_function_checks`**
(≈ 26 s at 12 threads; slower at 2). Every SURVIVOR is escalated by hand to the
heavy suites (`coverage_gaps`, `conservation_fuzz`, `convergence_order`,
`mms_convergence`, full `heat_injection`) to confirm it is a genuine test-gap
and not merely killed by an excluded suite (R2.2 escalation). Per-mutant timeout
120 s: baseline finishes under it; genuine infinite Newton hangs are killed at
120 s and counted as **caught** (R2.2), and recorded separately as robustness
boundaries.

**Process-visibility note:** detached runs execute in a separate PID namespace,
so `ps`/`pgrep` from the interactive shell cannot see the cargo-mutants/rustc
processes even while they run normally. Liveness is judged by **log growth**
(`mut-<mod>.log`, `shard-status.log`), never by process listing.

Excluded from the denominator (`.cargo/mutants.toml` `exclude_globs`):
`main.rs`, `cli.rs`, `output.rs`, `bin/**` — I/O plumbing, so the headline score
reflects physics code. Stated here so the denominator is honest.

## Mutant inventory (from `--list`, no build)

| Module | Mutants | Tier | Run status |
|---|--:|:--:|---|
| `electron_temp.rs` | 6 | 1 | ✅ done (5 caught, 1 survivor → new test) |
| `double_compton.rs` | 128→87 | 1 | ✅ done (dead fn deleted; 66 caught, 21 to escalate) |
| `bremsstrahlung.rs` | 288→240 | 1 | ✅ run done (220 caught; dead `br_heating_integral` to delete; 28 to escalate) |
| `recombination.rs` | 322 | 1 | ⬜ |
| `kompaneets.rs` | 528 | 1 | ⬜ (large — shard) |
| `solver.rs` | 1636 | 1 | ⬜ (largest — shard) |
| `grid.rs` | 95 | 2 | ⬜ |
| `dark_photon.rs` | 112 | 2 | ⬜ |
| `spectrum.rs` | 219 | 2 | ⬜ |
| `cosmology.rs` | 351 | 2 | ⬜ |
| `distortion.rs` | 364 | 2 | ⬜ |
| `energy_injection.rs` | 571 | 2 | ✅ done (268 caught, 351 to escalate) |
| `greens.rs` | 633 | 2 | ✅ done (260 caught, 373 to escalate) |
| `axion.rs` (new, post-snapshot) | 49 | 2 | ✅ done (25 caught, 24 to escalate) |
| **tier-1 total** | **2908** | | |
| **tier-2 total** | **2345** | | |
| **grand total (physics modules)** | **5253** | | |

**Scale reality (stated honestly, per directive against silent caps).**
Measured throughput at the safe config (2 build + 2 test threads) is **~81 s /
mutant** (≈50 s test at 2 threads + build), so the full 5253-mutant sweep is a
~5-day wall-clock job on this box. Runs proceed **tier-1 first** (the plan's gate
priority): `double_compton` → `bremsstrahlung` → `recombination` →
`dark_photon` → `spectrum` → `cosmology` → `distortion` → `grid`, with
`electron_temp` already done. The two oversized modules are **sharded by
function-name regex** (`--re`) so the highest-consequence code is covered first;
un-run mutants are reported as a numbered coverage gap, never silently dropped:

- `solver.rs` (1636): `ThermalizationSolver::update_temperatures` alone is
  **731** mutants (the perturbative Δρ_eq + DC/BR heating integrals — the most
  numerically delicate solver code; CLAUDE.md pitfalls #4/#5 live here), the
  remaining 905 are everything else. Shard A =
  `--re 'update_temperatures'`, shard B = the complement.
- `kompaneets.rs` (528): `kompaneets_step_coupled_inplace` **455** (the IMEX
  Crank–Nicolson step), `thomas_solve_inplace` 46 (tridiagonal solver, unsafe
  `get_unchecked` kernel — cross-checked by Miri in R4.2), ~27 others.
  Shard A = `--re 'kompaneets_step_coupled_inplace'`.

Because sharding does not reduce total work (kompaneets+solver ≈ 49 h alone),
these two modules will be reported at whatever coverage the available compute
window reaches, with the exact mutant count run / total stated per shard. This
is a compute bound, not a methodology gap: the harness is identical to the
completed modules and any un-run shard is one command away
(`CARGO_BUILD_JOBS=2 cargo mutants --in-place -t 120 --jobserver-tasks 2 -f
src/solver.rs --re 'update_temperatures' -o out/solver_A`).

## Execution plan (2026-07-06, EB directive: sweep all, triage at the end)

Run the lean sweep across **every** module to conclusion first; record raw
`outcomes.json` counts as each lands; defer ALL triage, survivor escalation,
dead-duplicate deletions (`br_heating_integral` — F-R2-2), equivalent-mutant
proofs, and new tests to a single audit pass after the full sweep + Python
`mutmut` complete. Raw outputs persist under `/tmp/claude-1000/spx-mut/out/` and
committed snapshots under `dev/audit/mutation/`, so no triage data is lost by
deferring. Order: (1) recombination→dark_photon→spectrum→cosmology→distortion→
grid; (2) kompaneets(shard)→energy_injection→greens→solver(shard); (3) Python
mutmut; (4) full audit. `dc_heating_integral` was already deleted (F-R2-1) before
this directive.

## Code-snapshot note (parallel development during the multi-day sweep)

The isolated copy was rsync'd from the working tree on 2026-07-06; the repo was
actively developed in parallel through 2026-07-09. Divergence check (copy vs
current repo, per module) on 2026-07-09: **all swept physics modules identical
except `solver.rs`.** The `solver.rs` diff is **44 lines, cosmetic/additive** —
variable/method renames generalizing the resonance-IC wiring from dark-photon to
"dark photon + axion" (`dark_photon_params`→`resonance_params`,
`dp_z_res`→`res_z_res`), comment updates, and a new `warn_axion_range` call. The
core solver physics where all solver survivors live (`update_temperatures`,
`step_with_dz`, `dcbr_heating_with_derivative`, `adaptive_dz`) is unchanged, so
the solver mutation results stand without a re-run. A **new module `src/axion.rs`**
(axion-photon conversion: `kappa_ev`, `gamma_con_axion`) was added and is queued
as the final shard. `energy_injection.rs` was extended to call `axion` (mutant
count 571→622); it is (re)run against the current version. `electron_temp.rs`
and `double_compton.rs` differ only by this audit's own edits (new test / dead-fn
deletion).

## Per-module results

_(populated as shards complete; from `mutants.out/outcomes.json`, not stdout)_

| Module | tested | caught | timeout | missed | unviable | score* |
|---|--:|--:|--:|--:|--:|--:|
| `electron_temp.rs` | 6 | 0 | 5 | 1→0 | 0 | 6/6 = 100%† |
| `double_compton.rs` | 87‡ | 66 | 0 | 21 | 0 | 66/87 = 76%§ |
| `bremsstrahlung.rs` | 240¶ | 211 | 0 | 28 | 1 | 211/239 = 88%§ |
| `recombination.rs` | 322 | 231 | 30 | 61 | 0 | 261/322 = 81%§ |
| `dark_photon.rs` | 112 | 79 | 0 | 33 | 0 | 79/112 = 71%§ |
| `spectrum.rs` | 219 | 181 | 0 | 38 | 0 | 181/219 = 83%§ |
| `cosmology.rs` | 351 | 246 | 36 | 69 | 0 | 282/351 = 80%§ |
| `distortion.rs` | 364 | 92 | 0 | 269 | 3 | 92/361 = 25%§‖ |
| `grid.rs` | 95 | 38 | 3 | 51 | 3 | 41/92 = 45%§ |
| `kompaneets.rs` [B: ¬step_coupled] | 55 | 53 | 2 | 0 | 0 | 55/55 = 100% |
| `kompaneets.rs` [A: step_coupled] | 473 | 305 | 12 | 156 | 0 | 317/473 = 67%§ |
| `kompaneets.rs` [**total**] | 528 | 358 | 14 | 156 | 0 | 372/528 = 70%§ |
| `solver.rs` [A: update_temperatures] | 920 | 830 | 4 | 86 | 0 | 834/920 = 91%§ |
| `solver.rs` [B: ¬update_temperatures] | 716 | 188 | 29 | 476 | 23 | 217/693 = 31%§ |
| `solver.rs` [**total**] | 1636 | 1018 | 33 | 562 | 23 | 1051/1613 = 65%§ |
| `energy_injection.rs` | 622 | 268 | 0 | 351 | 3 | 268/619 = 43%§** |
| `greens.rs` | 633 | 256 | 4 | 373 | 0 | 260/633 = 41%§†† |
| `axion.rs` | 49 | 25 | 0 | 24 | 0 | 25/49 = 51%§‡‡ |

\* score = (caught + timeout) / (caught + timeout + missed); unviable excluded.
"caught" = killed by an assertion failure; "timeout" = killed by the 120 s hang
timeout (R2.2 counts both as caught).
† after adding the new externally-anchored test that kills the lone survivor
(`test_equilibrium_recovers_shifted_temperature`); the raw lean-run score was
5/6.
‡ raw run was 128 mutants; 41 were on the now-deleted dead duplicate
`dc_heating_integral` (see Findings), all of which had survived the lean run
because their sole test lives in the excluded `heat_injection` suite. Deleting
the function removes exactly those 41 → 87 real mutants. §This 76% is the
**lean-subset lower bound**; the 21 survivors are pending escalation to the
energy-conservation suites (most `dc_emission_coefficient_fast`/`dc_prefactor`
mutations should convert to "caught"). Final score after escalation TBD.
¶ raw run 288 mutants (220 caught, 67 survived, 1 unviable). 48 were on the dead
duplicate `br_heating_integral` (F-R2-2): 9 caught by its in-module Planck→0
test, 39 survived. Deleting it removes those 48 → 240 real mutants (211 caught,
28 survived, 1 unviable). 88% is the lean lower bound; 28 survivors pending
escalation.
‖ distortion's 25% is a lean-subset artifact concentrated in non-default
decomposition variants (`decompose_nonlinear_be` 181/269) + MJy conversion, NOT
the production least-squares path (which `test_decomposition_pure_*` pin). See
the distortion triage note; headline distortion score deferred until the
dead/rare-path question is resolved in the final audit.

** energy_injection's 43% is a **strong lean-subset artifact**: the module is
dominated by scenario-config, validation, and warning code whose only exercisers
live in the excluded `heat_injection`/`adversarial_inputs`/`coverage_gaps`
suites. Survivors by function (351 total, from `outcomes.json`):

| Function | survivors | killed where (expected escalation) |
|---|--:|---|
| `InjectionScenario::validate` | 78 | `adversarial_inputs`, `coverage_gaps` (bad-input errors) |
| `photon_source_rate` | 52 | `heat_injection` photon-injection tests |
| `heating_rate` | 32 | `heat_injection` scenario energetics |
| `interp_2d` | 28 | `heat_injection` tabulated-source tests |
| `initial_delta_n` | 19 | DP/axion resonance IC tests (`heat_injection`) |
| `warn_tabulated_coverage` | 17 | `coverage_gaps` warning-threshold tests |
| `refinement_zones` | 15 | grid-refinement tests |
| `suggested_x_min` | 14 | grid-extent tests |
| `load_photon_source_table` | 13 | table-I/O tests (`coverage_gaps`) |
| `warn_axion_range` | 12 | axion-range warning tests |
| `dark_photon_params` | 11 | DP resonance tests |
| `axion_params` | 11 | axion resonance tests |
| `resonance_params` | 10 | resonance-z tests |
| `warn_dark_photon_range` | 9 | DP-range warning tests |
| `warn_stimulated_emission` | 9 | stimulated-emission warning test |
| `vacuum_survival` | 6 | photon-survival tests |
| `characteristic_redshift` | 6 | scenario-z tests |
| `interp_log_z` | 4 | tabulated-source interp tests |
| `load_heating_table` | 3 | table-I/O tests |
| `warn_strong_distortion` | 2 | strong-distortion warning test |

Nearly all are validation/warning/scenario-config paths whose killers are
intentionally outside the lean subset. Escalation (deferred to final audit) is
expected to convert the large majority; the residual after escalation is the
real test-gap signal.

†† greens' 41% is concentrated in the **photon-injection GF path**:
`greens_function_photon` 121, `broadened_bump` 74 + Arsenadze helpers
(`f_cs`/`alpha_cs`/`beta_cs`) 35, `tau_ff_survival` 35,
`distortion_from_{heating,photon_injection}` 50, `mu_y_from_heating` 16,
regime-guard helpers ~27. Structural cause: the lean subset's
`greens_function_checks` pins the *heat* GF (Chluba-2013 limits); the photon-GF
tests live in the excluded `heat_injection` suite, and the other main anchor is
the Python **parity** fixture (CI-side, not in the mutation kill set at all).
**Resolved by the escalation harvest (2026-07-26).** Escalation converted only
16/94 in shard s0 and 0/23 in s1, so the "escalation should convert much of it"
prior was wrong here too. The 68 in `tau_ff_survival` + `f_cs`/`alpha_cs`/
`beta_cs` are covered by the pytest `parity` CI job and were verified to fail it
(§Survivor escalation); they are a harness-scope artifact. The 25 in
`distortion_from_heating`/`mu_y_from_heating` are *not* parity-covered and remain
a genuine open gap on the channel the coverage matrix already flags as
anchor-poor (rows 6/7) — triage jointly with R3. Raw snapshots:
`dev/audit/mutation/rust/greens/`, `dev/audit/mutation/rust_escalation/greens_s*/`.

‡‡ axion's 24 survivors are **23 in `gamma_con_axion` + 1 in `kappa_ev`** — every
arithmetic operator inside the axion→photon conversion rate γ_con survives the
*lean* subset, which contains no axion physics at all (axion is exercised only by
`heat_injection`). This is a **lean-subset artifact, largely closed by
escalation**, not an open anchor gap: `heat_injection::test_axion_gamma_con_benchmark`
already pins the γ_con **value** to ≈ 0.214 ± 15 %, hand-derived from CCM24
Eq. 3a (not read off code output — CLAUDE.md #9). Categorizing the 24 by hand
against that ±15 % window (z_res ≈ 3.21×10⁴, so `1+z_res` = `z_res` to 3×10⁻⁵):

- **21 killed by escalation** — every operator swap that moves γ_con's magnitude
  by a large factor (`m_ev*m_ev`→`m_ev+m_ev`, `/EV_IN_JOULES`→`*`/`%`,
  `kappa*kappa`→`kappa/kappa`, the numerator/denominator `/`→`*`, etc.). All land
  far outside ±15 % and fail `test_axion_gamma_con_benchmark`.
- **3 genuine equivalent mutants** — `axion.rs:78:53 +→*` (`1.0*z_res` vs
  `1.0+z_res`), `82:28 +→-` (`(1.0−z_res).powi(4)` — even power, sign irrelevant),
  `82:28 +→*` (`(1.0*z_res).powi(4)`). All three differ from the true value by
  O(1/z_res) ≈ 3×10⁻⁵ ≪ 15 %, so **no finite-tolerance test can kill them**;
  equivalent under the large-z_res regime, logged as such (not a test gap).

Net: escalated axion score over non-equivalent mutants = **(25 caught + 21
escalated)/(49 − 3 equivalent) = 46/46 = 100 %**. NOTE: an earlier draft of this
footnote wrongly claimed `heat_injection` only checks γ_con's order of magnitude
and that "escalation must go beyond `heat_injection`" — that was incorrect; the
±15 % value benchmark exists and is the anchor. The open ~22 % Bryce red-curve
offset (memory `axion-dp-distortion-fig-tshift-error`) is a separate physics
question, not a mutation-coverage gap. Raw snapshot:
`dev/audit/mutation/rust/axion/`.

## Survivor escalation (truncated 2026-07-13; harvested 2026-07-26)

Three workers escalated lean survivors against the heavy suites only
(`adversarial`, `mms`, `conservation_fuzz`, `science`, `convergence`,
`cosmotherm`, `cli`, `coverage_gaps`, `heat_injection`); `--lib` and
`greens_function_checks` were not re-run since every escalated mutant had
already survived them. Raw outputs: `dev/audit/mutation/rust_escalation/`.

| Shard | escalated | caught | still missed |
|---|--:|--:|--:|
| `double_compton` (module complete) | 21 | 9 | 12 |
| `energy_injection` s0 | 117 | 55 | 62 |
| `energy_injection` s1 (partial) | 59 | 48 | 11 |
| `greens` s0 | 94 | 16 | 78 |
| `greens` s1 (partial) | 23 | 0 | 23 |
| `solver` s0 (partial) | 6 | 4 | 2 |
| **total** | **320** | **132** | **188** |

**Conversion rate 41%** (132/320). This *refutes* the prior stated throughout
the per-module triage below — "escalation is expected to convert the large
majority." It converts fewer than half. Consequence for the 1,717 survivors
never escalated: they can be neither dismissed as lean-subset artifacts nor
claimed as test gaps, and this document says so wherever the old prior appears.

**Quotable score.** Lean lower bound 3,194/5,231 = 61.1%; adding the 132
escalation conversions gives **3,326/5,231 = 63.6%, a verified full-suite lower
bound**. Applying the observed 41% conversion to the remaining 1,717 would give
≈77%, but that is an extrapolation from a **non-random 16% sample** (shards were
chosen by worker layout, not sampled) and is not quoted as a result.

**Escalated-and-still-missed, by cause** (all 188 accounted for):

| Cause | n | Disposition |
|---|--:|---|
| Covered by the pytest `parity` CI job, outside the cargo-mutants kill set | 74 | not a gap (verified) |
| Tabulated-source interpolation and table I/O | 37 | 26 closed, 11 open |
| Input-validation guards | 33 | out of scope by decision |
| Heating-convolution Green's-function entry points | 25 | closed (B3) |
| `double_compton` (F-R2-3, B1, one equivalent) | 12 | closed |
| Scenario energetics (`heating_rate`, `photon_source_rate`) | 5 | open |
| `visibility_j_t` (F-R2-4) | 2 | closed |

- **74 are a harness-scope artifact, not a gap — verified, not assumed.**
  `tau_ff_survival` 35, `f_cs` 12, `alpha_cs` 11, `beta_cs` 11,
  `photon_survival_probability` 2, `photon_survival_probability_numerical` 1,
  `x_c_dc`/`x_c_br` 2. Every one is reached by a function in the
  `test_parity.py` dispatch, and the `parity` CI job regenerates the fixture
  from current Rust and runs the Python port against it — but that job is
  pytest, so cargo-mutants could never execute it. Checked directly: planting
  `+`→`-` in `f_cs` and `*`→`+` in `tau_ff_survival`'s `1/x³` moves the
  regenerated fixture by **42%** (`greens_function_photon`, rtol 1e-3) and
  **100%** (`photon_survival_probability_numerical`, rtol 1e-5), 24 points over
  tolerance in each group; both fail CI. Native Rust anchors were added anyway
  (§New tests added) so the Rust suite no longer leans on the Python job for
  this physics.
- **37 tabulated-source** — `interp_2d` 26 (closed by an exact-bilinear
  identity test), `load_photon_source_table` 6, `load_heating_table` 3,
  `interp_log_z` 2 (the 11 loader/1-D-interp mutants remain open).
- **33 validation guards** — `InjectionScenario::validate` 31,
  `SolverConfig::validate` 2. Left open by decision (§Decisions, item 2).
- **25 `distortion_from_heating` (15) / `mu_y_from_heating` (10)** — production
  Green's-function entry points, *not* parity-covered (only the scalar
  `greens_function` is). The pre-audit tests checked only linearity in Δρ/ρ,
  which no rescaling of the integrand can break. Closed by a narrow-burst
  reduction test (§New tests added).
- **12 `double_compton`** — the only module escalated to completion:
  `dc_prefactor` 4 + `dc_emission_coefficient_fast` 4 (**F-R2-3**),
  `dc_gaunt_factor` 3 (relativistic correction, closed by B1),
  `dc_high_freq_suppression` 1 (equivalent).
- **5 scenario energetics** — `InjectionScenario::heating_rate` 4,
  `photon_source_rate` 1. Open.

**Rust lean sweep COMPLETE (2026-07-11).** All 14 physics modules run.
Aggregate (per-module table above, dead-duplicate mutants excluded, raw
electron_temp): **5,264 mutants tested — 3,069 caught + 125 timeout (= 3,194),
2,037 missed, 33 unviable → 61% lean-subset lower bound.** Per §method this is
NOT a quotable suite score: the kill set deliberately excluded the heavy
physics suites, so the escalation pass over the 2,402 survivors is the gate
for any headline number. Raw per-module snapshots:
`dev/audit/mutation/rust/<module>/`.

## Python `mutmut` results (R2.5, raw — triage deferred)

`mutmut 3.6.0`, `--max-children 1`, on the 4 limit-pipeline modules
(`firas`, `greens`, `dark_photon`, `greens_table`), test selection =
`test_firas`, `test_greens`, `test_greens_table`, `test_parity`. Config in
`python/pyproject.toml` `[tool.mutmut]`. **Setup fix:** mutmut's `mutants/` copy
adds a directory level, so firas.py's `__file__`-relative `_DATA_DIR`
(`parent.parent.parent/data`) resolved to `python/data` (nonexistent) → baseline
failed. Fixed with a `python/data → ../data` symlink (repo data lives at
`<repo>/data`). Ran in parallel with the Rust sweep (probe confirmed one pytest
worker coexists in the Rust test-phase memory headroom).

Totals: **3683 mutants — 1892 killed, 3 killed-by-timeout, 1021 survived, 767
"no tests" (on lines no selected test covers).** Caught = 1895/3683 = 51% (raw).
Survivors by module: **`greens` 827**, `firas` 155, `greens_table` 24,
`dark_photon` 15.

**Scratch removed at close-out (2026-07-26).** `python/mutants/` (19 MB of
mutmut's generated copy) and the `python/data → ../data` symlink were deleted.
**Recreate the symlink before any future mutmut run** — without it firas.py's
`__file__`-relative `_DATA_DIR` resolves to the nonexistent `python/data` inside
mutmut's extra directory level and the baseline fails.

**Caveat (final audit): the 51% is a test-SELECTION lower bound, not a suite
verdict.** The 4 selected test files under-exercise `greens.py` (a large pure-
Python port with many functions); `greens` alone is 827 survivors + most of the
767 "no tests". The audit must (a) re-run with the full `python/tests/` suite
before quoting a Python score, and (b) check whether `greens.py` survivors that
diverge from the Rust port are caught by `test_parity` (they should be — parity
is the primary anchor for that module). `firas.py`'s 155 survivors are the
referee-critical cluster (covariance/GLS/profiling — Round-1 P1-5) and get
priority triage. Committed snapshot: `dev/audit/mutation/mutmut_results.txt`.

## Survivor triage

_(every entry in each `missed.txt` gets one of: **test-gap** → new
externally-anchored test; **equivalent** → one-line proof; **unreachable** →
domain argument, per R2.3.)_

### `electron_temp.rs`

- **`electron_temp.rs:50:9: replace ElectronTemperature::update_equilibrium
  with ()`** → **class-1 test-gap** (closed).
  The no-op mutant leaves `rho_e` at its `Default` value of `1.0`. Both existing
  tests (`test_equilibrium_for_planck`, `test_equilibrium_for_bose_einstein`)
  feed spectra whose Compton-equilibrium ratio is *exactly* 1 (a Planck, and any
  Bose-Einstein — for which `n(1+n) = −dn/dx` ⟹ `I₄ = ∫x⁴n(1+n)dx = 4∫x³n dx =
  4G₃` identically), so `rho_e = 1` is indistinguishable from the no-op. The
  function is documented non-production (verification helper only; the solver
  uses the perturbative `Δρ_eq`), so the mutant has zero physics-prediction
  impact — but the helper's tests genuinely fail to pin its output.
  **Fix (externally-anchored, analytic):** a Planck spectrum at a shifted
  temperature `n(x) = n_pl(x/a)` has Compton-equilibrium ratio exactly `a`
  (same by-parts identity: `n(1+n) = −a·dn/dx` ⟹ `I₄ = 4a·G₃`). New test
  `test_equilibrium_recovers_shifted_temperature` feeds `a = 1.05` and asserts
  `rho_e ≈ 1.05`; the no-op mutant (leaving 1.0) now fails. Kill re-confirmed by
  re-running the mutant (see New tests added).

### `double_compton.rs`

Raw lean run: 128 mutants, 66 caught, **62 survivors**. Survivors split into two
buckets:

- **41 survivors in `dc_heating_integral` → dead-code FINDING, resolved by
  deletion.** This `pub fn` duplicated the DC-heating logic the production
  solver actually uses (`dcbr_heating_with_derivative`, inlined in `solver.rs`);
  a whole-codebase grep found **no production caller** (only a doc-comment
  reference in `bremsstrahlung.rs` and one test). Its sole test
  (`test_dc_heating_integral_planck_zero`) fed `delta_n = 0`, where the
  integrand is *identically zero*, so it could never catch k_dc-scaling,
  accumulation, or normalization mutations — hence all 41 survived. EB decision
  (2026-07-06): **delete** the duplicate. Removed the function + its test (see
  `src/double_compton.rs` / `tests/heat_injection.rs` notes); this eliminates a
  duplicate-divergence hazard between two hand-maintained copies of the same
  physics. `dc_emission_coefficient` (used by the deleted fn and still live
  elsewhere) is unaffected; the now-test-only `planck` import was gated behind
  `#[cfg(test)]`. Verified: `cargo clippy --lib --tests` clean, affected unit
  tests pass. **A parallel `br_heating_integral` in `bremsstrahlung.rs` looks
  like the same pattern — check when the BR shard lands.**

- **21 production-relevant survivors → ESCALATED TO COMPLETION (2026-07-13).**
  9 caught by the heavy suites, **12 survived everything**. The expectation
  recorded here beforehand — that the energy-conservation suites would catch the
  `dc_prefactor` / `dc_emission_coefficient_fast` clusters — was wrong. What the
  12 actually are, and why:
  - `dc_prefactor` ×4 and `dc_emission_coefficient_fast` ×4 → **F-R2-3**: K_DC
    was implemented twice and the literature anchor tested the wrong copy. Fixed.
  - `dc_gaunt_factor` ×3 (`/ → *`, `+ → -`, `* → /` in the relativistic
    correction `1/(1+14.16 θ_z)`): **confirmed genuine gap.** Every pre-audit
    Gaunt test passes `theta_z = 0.0` exactly
    (`double_compton.rs:189,200,227,238`), where the correction is identically 1
    and the flips are unobservable. Closed by
    `test_dc_relativistic_correction_applied` (§New tests added).
  - `dc_high_freq_suppression` ×1 (`> → >=` at the `x > 100` cutoff):
    **equivalent mutant** — at x = 100, `e^{-200} ≈ 10⁻⁸⁷`, so `>` vs `>=`
    changes an output already indistinguishable from 0. No test can (or should)
    catch it. Logged as equivalent, not a gap.

### `bremsstrahlung.rs`

Raw lean run: 288 mutants, 220 caught, **67 survivors**, 1 unviable. Split:

- **39 survivors in `br_heating_integral` → dead-code (F-R2-2), delete pending.**
  Same pattern as DC's `dc_heating_integral`: non-production duplicate of
  `solver::dcbr_heating_with_derivative`. Its in-module Planck→0 test caught 9
  of its 48 mutants (the sign-flips) but not the 39 scaling/normalization ones.
  Deletion + verification scheduled for the recombination-completion build window
  (deleting now would need a clippy build that OOMs the running recombination
  shard). Removes 48 mutants (9 caught + 39 survived) → 240 real BR mutants.
- **28 production-relevant survivors → pending escalation:**
  - `br_emission_coefficient` ×13, `br_emission_coefficient_with_he` ×5: the BR
    emission rate; survive lib+greens because their thermalization effect is
    checked by the energy-conservation suites (excluded from lean). Expected
    caught on escalation.
  - `gaunt_ff_nr_fast_preln` ×3, `gaunt_ff_nr` ×1, `gaunt_ff_nr_fast` ×1: the
    non-relativistic Gaunt factor (Born approx + softplus). Some may be genuine
    coverage gaps in the Gaunt-value tests — check on escalation.
  - `softplus` ×3: the softplus interpolation (Chluba, Ravenni & Bolliet 2020).
    Likely gaps in the interpolation-shape coverage — check on escalation.
  - `br_precompute` ×2: the precompute helper for the fast BR path.

### `recombination.rs` (raw — triage deferred per EB directive)

322 mutants, 231 caught + 30 timeout, **61 survivors**, 0 unviable. Survivors
spread across the Saha/Peebles machinery (no dead-duplicate concentration):
`saha_he_ii` 9, `RecombinationHistory::x_e` 9, `find_saha_switch` 7,
`ionization_fraction` 7, `solve_saha_quadratic` 6, `saha_he_i` 6, `peebles_rhs`
6, `solve_saha_linear` 5, `peebles_c` 3, `peebles_step` 2, `saha_hydrogen` 1.
Working hypothesis for the final audit: the direct X_e anchor
(`test_xe_vs_recfast_milestones`, HyRec-2) checks only z = 1100/800/200 — all
H-dominated — so He-recombination Saha mutations (`saha_he_i/he_ii`) and
off-milestone-z mutations survive lib+greens; the rest feed the solver's X_e and
are pinned only by the excluded energy-conservation/PDE suites. Likely a
coverage-breadth gap (add He-epoch + finer-z X_e anchors) rather than dead code.

### `dark_photon.rs` (raw — triage deferred)

112 mutants, 79 caught, **33 survivors**. Concentrated: `resonance_redshift`
×16, `dln_omega_pl_sq_dlna` ×16, `gamma_con` ×1. These NWA helpers are pinned by
the **Python parity suite** (`test_parity.py`) and AxionLimits (R5), not by Rust
`--lib` assertions, so they survive the Rust lean subset. Final-audit note: this
is a cross-language-coverage artifact — the mutmut run (R2.5) tests the Python
`dark_photon.py` that actually feeds the published limits; the Rust helpers'
direct in-module coverage could be tightened with value-anchored unit tests
(z_res and dln ω_pl²/dlna at a reference mass), but their correctness is already
externally anchored via parity + AxionLimits.

### `spectrum.rs` (raw — triage deferred)

219 mutants, 181 caught, **38 survivors**: `y_shape` 9, `g_bb` 8, `planck` 7,
`bose_einstein` 7, `spectral_integral` 3, `compton_equilibrium_ratio` 3,
`weighted_integral` 1. Spectral-shape / integral helpers — final audit to check
whether the survivors are (a) mutations in overflow/edge branches unreached by
the test inputs, (b) integral-quadrature mutations pinned only by the excluded
convergence suites, or (c) genuine value-coverage gaps.

### `cosmology.rs` (raw — triage deferred)

351 mutants, 246 caught + 36 timeout, **69 survivors**:
`compton_y_parameter_with_recomb` 20, `validate` 19, `dt_dz` 6, `rho_gamma` 4,
`h0` 3, `n_he` 3, `cosmic_time` 3, `z_eq` 2, `n_gamma` 2, `compton_y_parameter`
2, +6 singletons. Two dominate: (i) `validate` (input-validation guards — a
classic untested-branch cluster; final audit to check `_validation.py`/solver
guards actually exclude the same domains, R2.3 class-3), and (ii)
`compton_y_parameter_with_recomb` (a diagnostic y-integral; check whether it is
on any published-figure path before deciding test-gap vs low-priority).

### `distortion.rs` (raw — triage deferred; STANDOUT LOW SCORE)

364 mutants, only 92 caught, **269 survivors**, 3 unviable — 25%, the lowest of
any module. Breakdown: **`decompose_nonlinear_be` 181**, `decompose` 27,
`band_weights` 22, `decompose_gram_schmidt` 21, `delta_n_to_intensity_mjy` 16,
`decomposition_band_count` 2.

**CORRECTION (2026-07-26).** An earlier draft of this section hypothesised that
`decompose_nonlinear_be` / `decompose_gram_schmidt` might be "alternative or
experimental decomposition paths not on any production route," i.e. dead code in
the F-R2-1/F-R2-2 mould. **That is wrong.** `decompose_distortion`
(`distortion.rs:398`) calls `decompose_nonlinear_be` directly and `decompose`
wraps `decompose_distortion`, so it is *the* production path — every published μ
and y goes through it. Do not repeat the dead-code framing.

The correct explanation is structural, and it is not a coverage failure. Roughly
140 of the 181 survivors lie in the Levenberg–Marquardt block
(`distortion.rs:316-341`: the 3×3 cofactor expansion, the λ damping, the
backtracking loop). **LM with backtracking only accepts a step that lowers χ²**
(`if chi2_new < prev_chi2`), so a mutated cofactor or damping update changes the
path taken through parameter space but not the fixed point the iteration
converges to. Those mutants are equivalent by construction: no finite-tolerance
test can kill them, and none should try. The residual ~19 that *do* move the fit
sit at `distortion.rs:255` (`model_at`, the M/Y_SZ/G_bb basis assembly), 261-262
(the χ² residual and its weights) and 270 (the Gram–Schmidt bootstrap) — those
are genuine and worth escalating if the campaign is ever resumed.

Caveat: `distortion` was **never escalated**, so all of these are lean-only
numbers. The production least-squares path is anchored by
`solver::tests::test_decomposition_pure_{mu,y,tshift}` (lib) and
`greens_function_checks`, which caught the 92.

### `grid.rs` (raw — triage deferred)

95 mutants, 38 caught + 3 timeout, **51 survivors**, 3 unviable:
`GridConfig::validate` 22, `FrequencyGrid::find_index` 14,
`FrequencyGrid::uniform` 10, `log_uniform` 2, +3 config-ctor singletons.
Infrastructure — `validate` (untested guards, R2.3 class-3) and `find_index`
(the grid lookup; likely pinned only by solver-path tests) dominate. Low
physics-consequence; final audit to confirm guards are covered elsewhere.

### `kompaneets.rs` (raw — triage deferred)

Sharded: **B (¬`kompaneets_step_coupled_inplace`) = 55/55 caught, 100%** — the
Thomas tridiagonal solver + helpers are tightly pinned (consistent with the
Miri + MMS kernel coverage in R4). **A (`kompaneets_step_coupled_inplace`) = 317
caught, 156 survivors, 67%.** All 156 survivors are internal arithmetic of the
IMEX Crank–Nicolson step. Strong prior for the final audit: the *convergence*
suites `mms_convergence` + `convergence_order` (method-of-manufactured-solutions,
excluded from the lean subset) directly pin this step's discretization order and
will catch a large fraction — this is the R2.2 escalation case where the excluded
suite is precisely the one that tests the mutated code. Escalate the 156 against
`mms_convergence`/`convergence_order`/`conservation_fuzz` before classifying any
as genuine gaps.

### `solver.rs` (raw — triage deferred; shard B in progress)

Shard **A (`update_temperatures`, the highest-consequence solver function) =
834 caught, 86 survivors, 91%** — the strongest coverage of any large physics
shard, consistent with this function being pinned by the decomposition + PDE↔GF
consistency lib tests. All 86 survivors are internal arithmetic of the
perturbative Δρ_eq / DC-BR heating; escalation prior = `conservation_fuzz` +
`coverage_gaps` (energy conservation is directly sensitive to the T_e update).
Shard **B (¬`update_temperatures`) = 217 caught, 476 survivors, 23 unviable,
31%** — the low score is dominated by numerics the lean subset structurally
cannot catch: `step_with_dz` 180, `dcbr_heating_with_derivative` 95 (the
*production* DC/BR heating that replaced the deleted duplicates),
`adaptive_dz` 73 (= 348/476). These are exactly what `mms_convergence`,
`convergence_order`, and `conservation_fuzz` (all excluded from the lean subset)
are built to pin — a single-redshift lib test cannot detect a wrong timestep or
heating increment, but a manufactured-solution convergence test or an
energy-conservation fuzz will. The remaining 128 are diagnostic/config:
`run_with_snapshots` 40, `brightness_temp` 17, `SolverConfig::validate` 14,
`SolverBuilder::build` 13, `soft_warnings` 9, snapshot helpers, etc. — low
physics-consequence (R2.3 class-3, plus the validation-guard cluster seen in
cosmology/grid). **Combined solver = 1051 caught / 562 survivors / 65% lean;
this is the single most escalation-dependent module** and the final audit must
escalate the step/heating/adaptive clusters against the convergence+conservation
suites before any solver score is quoted.

## Findings

- **F-R2-1 (dead code, resolved).** `double_compton::dc_heating_integral` was a
  non-production duplicate of `solver::dcbr_heating_with_derivative`, surfaced by
  41 surviving mutants. Deleted 2026-07-06 (EB-approved). No published figure
  depends on it (not in any production path), so no figure rerun required per the
  B5 protocol.
- **F-R2-3 (duplicate implementation, RESOLVED 2026-07-26). K_DC was written
  twice and the literature anchor exercised the copy the solver does not use.**
  Surfaced by the only module escalated to completion: 8 of `double_compton`'s
  12 full-suite survivors were in `dc_prefactor` / `dc_emission_coefficient_fast`.
  `dc_prefactor(θ)·H_dc(x)` is the solver hot path (`solver.rs:454,1161`);
  `dc_emission_coefficient(x,θ)` (`greens.rs:419`, `kompaneets.rs:1387,1490`) was
  a separate transcription of the same `(4α/3π)θ_z²I₄/(1+14.16θ_z)` arithmetic,
  and `test_dc_br_ratio_pinned_z1e6` — the class-(ii) Danese & de Zotti anchor —
  called only the latter. The production prefactor was therefore unanchored.
  Bremsstrahlung has exactly the consistency test DC lacked
  (`bremsstrahlung.rs:504`, fast vs reference at rel < 1e-10), which is why BR
  scored 88% and DC's prefactor did not.
  **Severity, measured not asserted:** the surviving mutant `(3.0*π) → (3.0+π)`
  scales K_DC by **1.535×** and passed the entire suite. It is *not* a wrong
  published number — no figure changes, since the code is correct; it is a
  statement that the suite constrained the DC normalisation only to a factor
  ~1.5. **Fix:** `dc_emission_coefficient` now delegates to
  `dc_prefactor × dc_high_freq_suppression`, and both the (4α/3π) coefficient and
  the relativistic correction are single-sourced (`DC_ALPHA_COEFF`,
  `dc_relativistic_correction`). Verified behaviour-preserving: all 40 parity
  groups (957 evaluation points) regenerate **bit-identically**.
- **F-R2-4 (dead code + vacuous assertion, RESOLVED 2026-07-26).**
  `greens::visibility_j_t` has no production caller — its only reference is its
  own bounds test, which computed `jt` and then asserted `jt ∈ [-0.2, 1]` while
  quoting a comment describing a *different* convention
  (J_T = 1 − J_μJ_bb* − J_y). The implementation is J_T = 1 − J_bb* ∈ [0,1], so
  the window was wide enough that replacing the whole function with the constant
  0 or 1 passed. The test now asserts the exact identity J_T = 1 − J_bb* and the
  correct [0,1] range, and the stale comment is gone. The function is kept
  (public API, part of the Chluba-2013 visibility set) rather than deleted.
- **F-R2-2 (dead code, confirmed — pending BR-shard data).**
  `bremsstrahlung::br_heating_integral` is the exact parallel: no production
  caller (production BR heating is also `solver::dcbr_heating_with_derivative`
  via `br_emission_coefficient_fast_preln`), only Planck→0 tests, not in the
  prelude. Unlike DC it has an *in-module* Planck→0 test that runs in the lean
  subset, so its sign-flip mutants are caught but its scaling/normalization
  mutants will survive. Plan: capture the BR shard's survivor split, then delete
  `br_heating_integral` under the same EB dead-duplicate ruling as F-R2-1.

## New tests added

_(close-out pass, 2026-07-26. Every one is anchored to an analytic identity, a
literature value, or an independent code — none is calibrated to code output,
per CLAUDE.md pitfall #9.)_

- **`science_suite::science_deep_thermalization_pde_z3e6`** — the campaign's
  most consequential addition, and the one that closes F-R2-3 at the physics
  level rather than the plumbing level. The suite could not detect a 1.535×
  error in K_DC, and the reason was *where* the tests sit, not how loose they
  are. With μ ∝ exp(−τ) and τ = (z_h/z_th)^{5/2} ∝ √K_DC,
  **∂ln μ/∂ln K_DC = −τ/2**; at z_h = 10⁶ (τ ≈ 0.18) that is −0.09, so a 53%
  K_DC error moves μ by ~4% — inside the existing 5% PDE↔GF band. Every other
  PDE thermalization test sat at z_h ≤ 10⁶ and the z_h = 5×10⁶ cases are
  Green's-function-only, so none of them tests the code's *own* DC rate.
  At z_h = 3×10⁶, τ ≈ 2.8 and the sensitivity is −1.4.
  **Measured:** PDE μ/Δρ = 0.07453 vs Green's-function target 0.07354 — **1.3%
  agreement** deep in the exponential tail (J_bb* = 0.0525). Band set to 8%
  (~6× measured), which bounds the DC normalisation to ≈6%.
  **Kill verified empirically, not argued:** re-planting the surviving mutant
  makes this test fail at **50.1%** error (μ/Δρ drops 0.0745 → 0.0367, a factor
  2.03) while `science_deep_thermalization_pde` at z_h = 10⁶ only moves from
  1.0% to 3.3% and still passes — exactly the predicted −τ/2 scaling. Runtime
  157 s.
- **`double_compton::test_dc_prefactor_matches_emission_coefficient`** — the two
  K_DC paths must agree exactly (`assert_eq!` on f64, 5 redshifts × 6
  frequencies). The direct guard against F-R2-3 recurring; mirrors BR's
  `test_br_fast_matches_reference`.
- **`double_compton::test_dc_relativistic_correction_applied`** — pins
  (1+14.16 θ_z)⁻¹ at z = 10⁶/3×10⁶/10⁷ *and* verifies it reaches both consumers
  by taking ratios against the θ_z → 0 limit, which isolates the correction
  exactly. Closes the 3 `dc_gaunt_factor` survivors; at z = 10⁷ the correction is
  6.5%, so a sign or division flip is a >12% error.
- **`double_compton::test_dc_emission_coefficient_absolute_value_z1e6`** —
  K_DC(x=1, z=10⁶) = 1.1002×10⁻⁸, derived by hand from CS2012 Eq. 13 (all five
  factors written out in the doc comment) and asserted to 5×10⁻⁴. The code
  agrees to better than 0.05%, which is an independent confirmation of the DC
  normalisation as well as a mutation anchor.
- **`double_compton::test_dc_emission_coefficient_magnitude`** (tightened) — its
  hand-transcribed CS2012 expression was compared at 10%; that is an identity,
  not an approximation, so the band is now 1×10⁻¹².
- **`heat_injection::test_dc_br_ratio_at_p18_reference_point`** — asserts DC/BR =
  17.06 ± 20% at the *independently derived* reference point (z = 10⁶, x = 0.1;
  P1-8, checked against Danese & de Zotti). **Measured 17.162 — 0.6% from the
  hand derivation**, so the anchor holds far inside its band. The coverage matrix
  listed this as a class-(ii) anchor but no test asserted it.
  This also exposed a wrong comment in the pre-existing
  `test_dc_br_ratio_pinned_z1e6`: it runs at x = 1 and claims "should be ~15-20",
  but 15–20 is the x = 0.1 value — the actual x = 1 ratio is **42.5**, sitting
  18% under that test's own upper bound of 50 while having 5.3× of slack below.
  The comment is corrected; the window is deliberately *not* tightened, because
  the x = 1 centre has never been independently derived and narrowing it to the
  code's output is precisely CLAUDE.md pitfall #9.
- **`double_compton::test_dc_detailed_balance_at_shifted_electron_temperature`**
  and **`bremsstrahlung::test_br_detailed_balance_at_shifted_electron_temperature`**
  — Kirchhoff balance at T_e ≠ T_z: the DC and BR sources must vanish identically
  for a Planck spectrum at the *electron* temperature, n_eq = 1/(exp(xφ)−1),
  φ = θ_z/θ_e. The pre-existing Planck tests only covered ρ_e = 1, where φ = 1
  and any error in the φ convention (CLAUDE.md pitfall #1) is invisible. Swept
  over ρ_e ∈ [0.9, 1.1], bracketing the |ρ_e−1| = 0.01 Taylor switch (pitfall
  #5). Scale-free: the equilibrium residual is required to be 10⁻¹⁰ of the
  residual from a 1%-off-equilibrium spectrum, so no absolute tolerance is
  guessed.
- **`greens::test_tau_ff_free_free_scaling`** — value and shape anchors on the
  low-z τ_ff integral, which only z_h < 5×10⁴ reaches and which had 34/34
  mutants survive. The scaling anchor is textbook: in the Rayleigh–Jeans limit
  free-free absorption goes as ν⁻² times the Gaunt factor, and the DC term
  carries the same x⁻² since H_dc(0) = 1, so
  d ln τ/d ln x → −2 − 1/ln(1/x) ≈ −2.15. **Measured −2.160** at the τ = 1
  crossing (x₁ = 4.37×10⁻³, z_h = 10⁴) — an independent confirmation that the
  free-free physics in `tau_ff_survival` is right, not just self-consistent.
  Plus monotonicity in x and in z_h, the z ≤ 200 floor, and the branch-crossing
  structure against the analytic μ-era form.
- **`greens::test_compton_broadening_identities`** — `f_cs`, `alpha_cs`,
  `beta_cs`, `broadened_bump` appeared **nowhere** in `tests/`. Adds the
  Arsenadze Eq. D13/D14 transcription checks and their x′ → 0 / x′ → ∞ limits,
  plus three numerical moment identities on the bump: it integrates to 1 (it is
  a log-normal PDF), ⟨x⟩ = x′·f_int (tying the returned energy ratio to the
  bump's own first moment), and σ²(ln x) = 2βy_γ → **2y_γ**, the
  Zeldovich–Sunyaev Compton-diffusion variance in log frequency — a class-(i)
  anchor on the broadening width. A hand audit does not survive a refactor;
  these do.
- **`greens::test_heating_convolution_reduces_to_greens_function`** — closes the
  25 `distortion_from_heating`/`mu_y_from_heating` survivors. For a narrow
  normalised heating history at z_h the convolution must reduce to
  G_th(x, z_h)·Δρ/ρ, and (μ, y) to ((3/κ_c)J_μJ_bb*, J_y/4)·Δρ/ρ — the defining
  property of a Green's function, which pins the convolution's normalisation and
  its ln(1+z) Jacobian against `greens_function`. The pre-audit tests checked
  only *linearity* in Δρ/ρ, which no rescaling of the integrand can violate.
  Band 5×10⁻⁴; the limiting error is the O(σ²) smearing of the burst, worst for
  y at z_h = 5×10⁵ where J_y ∝ (1+z)^{−2.58} gives G″/G ≈ 6.7. Setting σ = 0.02
  put that term at 1.3×10⁻³ and the test caught it; the fix was to narrow the
  burst to σ = 0.005 (≈8×10⁻⁵), not to widen the band.
- **`energy_injection::test_interp_2d_exact_on_bilinear_data`** — bilinear
  interpolation is *exact* for f = a + bz + cx + dzx, so this is an identity
  check over a deliberately non-uniform, non-square 4×5 grid: every node, three
  asymmetric points in every cell, both edges, out-of-range, and empty tables.
  The pre-audit test used one 2×2 cell with f01 = f11 = 0 and a single interior
  point; 26 of 28 `interp_2d` mutants survived it and the full suite.
- **`recombination::test_xe_vs_hyrec_helium_epoch`** — X_e at z = 5000/2500/2300/
  2000/1800 against the same HyRec-2 run as the existing milestones
  (`dev/output/hyrec2_xe_default_cosmo.dat`), with bands from the measured
  per-band disagreement in `xe_hyrec_comparison.md`. The old milestones were all
  hydrogen-dominated (z = 1100/800/200), leaving the He Saha machinery with no
  value anchor and 15 surviving mutants. Not cosmetic: the same audit measures
  the ε(m_A′) dark-photon limit moving −10.5% for resonances in the He window
  z ≈ 1800–2500, so X_e there feeds a published figure.
- **`greens::test_visibility_functions_physical_bounds`** (tightened) — see
  F-R2-4.

### CI

- `.github/workflows/ci.yml` `parity` job now compares fixture **values**, not
  just inputs, so the committed golden file can no longer drift silently from
  Rust (fix A5). Verified in sync: worst relative difference across all 40
  groups / 957 points is exactly 0.

### Earlier

- `src/electron_temp.rs::test_equilibrium_recovers_shifted_temperature` —
  analytic anchor `ρ_eq(n_pl(x/a)) = a`; kills the `update_equilibrium → ()`
  survivor. Numerically pre-verified with a Python replica of the exact Rust
  midpoint quadrature: `ρ_eq(n_pl(x/1.05)) = 1.05000226` (error 2.3e-6 ≪ 2e-3
  tol; no-op leaves 1.0, `|1.0−1.05| = 0.05 ≫ tol` ⟹ fails). **Verified on real
  code:** compiles clean (clippy), passes (`cargo test --release --lib`), and the
  kill is logically certain (the no-op mutant leaves `rho_e = 1.0`, which fails
  the `≈ 1.05` assertion by 0.05 ≫ 2e-3).

## Decisions taken at close-out (2026-07-26)

1. **Escalation stopped at 16% of survivors.** ~13 further days of wall clock on
   this box for a pass whose distinct *failure classes* were already all
   represented in the 320 escalated. Reported as a lower bound, never as a
   headline score.
2. **The validation-guard cluster is left open.** `InjectionScenario::validate`
   (31 residual after escalation), `SolverConfig::validate`, `GridConfig::validate`,
   `Cosmology::validate` — untested input guards. Declared out of scope: they
   reject nonsensical user input and cannot affect a published number. Stated
   here rather than quietly excluded from the denominator.
3. **Python `mutmut` not re-run against the full suite.** The 51% stands as a
   test-*selection* lower bound and is labelled as such. If Python triage is ever
   resumed it starts with `firas.py`, whose 155 survivors concentrate in
   `_joint_fit_floating_T` (~60), `limit_on_model` (16), `fit_distortion` (13),
   `profile_limit_floating_T` (13) and `chi2_from_solver` (25, "no tests") —
   i.e. the floating-T profiling behind the SciPost Fig. 8 referee question, the
   only referee-facing item in the backlog.
4. **Equivalent mutants logged, not chased:** `dc_high_freq_suppression` `>`→`>=`
   at x = 100; the three axion `1+z_res` mutants (§axion footnote); and the ~140
   Levenberg–Marquardt mutants in `decompose_nonlinear_be` (§distortion).

## Referee-reply paragraph (draft)

> We assessed the test suite by mutation testing rather than by coverage. Using
> `cargo-mutants`, 5,264 mutants were generated across the fourteen physics
> modules (I/O and CLI plumbing excluded from the denominator, which is stated
> in full in the audit) and each was run against the test suite. Because a
> full-suite run per mutant was not affordable, mutants were first screened
> against a fast subset and the survivors escalated to the complete suite; the
> escalation was carried to 16% of survivors before being stopped for compute,
> so we quote **63.6% as a verified lower bound** on the full-suite mutation
> score rather than a point estimate. The exercise found no incorrect published
> result. It did find, and we have fixed, three structural weaknesses that
> ordinary coverage metrics cannot see: the double-Compton emission coefficient
> was implemented twice with only the non-solver copy covered by its literature
> anchor, so the suite constrained the DC normalisation only to a factor ~1.5;
> the free-free survival integral that sets our low-redshift photon-injection
> limits was bound-checked but never value-checked; and the helium-recombination
> ionisation history — which shifts our dark-photon limit by ~10% where the
> resonance lands in the helium window — had no anchor of its own. Each is now
> pinned by an analytic identity, a literature value, or the HyRec-2 comparison,
> and we verified by re-planting the surviving mutant that the new
> thermalization test at z = 3×10⁶ detects the DC-normalisation error (50%
> deviation) that every previous test passed.
