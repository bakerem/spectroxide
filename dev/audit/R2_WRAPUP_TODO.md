# R2 mutation testing — close-out TODO (truncated by EB, 2026-07-26)

**Decision:** stop the survivor-escalation pass. Report R2 at its current
evidence level. This file is the complete list of what must happen to close it
honestly, plus the physics-validation checks the campaign surfaced.

---

## STATUS: close-out IMPLEMENTED 2026-07-26

| Item | | Item | |
|---|---|---|---|
| A1 DC deduplication | ✅ | B1 DC relativistic correction | ✅ |
| A2 `dc_emission_coefficient_fast` | ✅ (via A1) | B2 `interp_2d` exactness | ✅ |
| A3 `visibility_j_t` | ✅ | B3 `distortion_from_*` | ✅ |
| A4 DC/BR anchor | ✅ (new x=0.1 test) | B4 He-epoch X_e | ✅ |
| A5 CI parity values | ✅ | B5 validation guards | ⬜ out of scope (§8) |
| A6 cleanup | ✅ | C1–C5 reporting | ✅ |
| P1 PDE thermalization z=3e6 | ✅ **mutant-kill verified** | P4 detailed balance | ✅ |
| P2 τ_ff anchors | ✅ | P5 R1 CLASS B–D | ⬜ separate workstream |
| P3 broadening identities | ✅ | P6 R3 refsolver | ⬜ separate workstream |

Two findings were promoted to numbered audit findings: **F-R2-3** (K_DC
implemented twice, literature anchor on the wrong copy) and **F-R2-4**
(`visibility_j_t` dead + vacuous assertion). Details, measured numbers and the
referee-reply draft live in `mutation_audit.md`.

**Corrections to the record made during implementation:**
- §5 C2's guess that the 68 greens survivors are parity-covered was **verified**,
  not assumed: planted mutants move the regenerated fixture by 42% / 100%.
- §7 P2's proposed invariant ("raw τ_ff must always absorb more than the
  analytic form") is **false as stated** and was corrected. The two branches
  carry different x-scalings (x⁻² free-free vs x⁻¹ quasi-stationary), so they
  cross where P_s ≈ O(1). The invariant holds only in the absorption region and
  the test now scopes it there. The code's doc comment is correct.
- §7 P1's sensitivity estimate was right: predicted "factor ~2 in μ", measured
  factor 2.03.
- §2's "188 still missed" breakdown was recounted exactly from the harvested
  `missed.txt` files: the parity-covered group is **74**, not 68 (it also
  includes `photon_survival_probability`, `x_c_dc`, `x_c_br`). The corrected
  full classification is the table in `mutation_audit.md` §Survivor escalation.

Of the 188 escalated-and-still-missed mutants, **139 are now closed or
classified as non-gaps** (74 parity-covered, 26 `interp_2d`, 25 heating
convolution, 12 `double_compton`, 2 `visibility_j_t`). The remaining 49 are open
by decision: 33 validation guards (§8), 11 table-loader / 1-D-interp, 5 scenario
energetics. Also open by decision: the 1,717 never-escalated survivors and the
Python `mutmut` re-run. §6 and §8 explain each.

---

**Why stopping is right (measured, not asserted):** the 3-worker escalation ran
2026-07-11 → 2026-07-13 and covered **320 of 2,037 lean survivors (16%)** before
dying. At that throughput the remaining 1,717 need ~13 more days of wall clock
on this 7 GB box. The marginal information per day is low — see §2, where 16% of
the survivors already pin down every distinct failure *class*.

---

## 1. Data harvested (done this session)

`/tmp/claude-1000/spx-mut{,-b,-c}/out_esc/` → `dev/audit/mutation/rust_escalation/`
(856 KB: `outcomes.json`, `missed/caught/timeout/unviable.txt` per shard, plus
`ESCALATION_README.md`). **/tmp is volatile — this was the one urgent item.**
The three worker trees can now be deleted.

| Shard | escalated | caught | still missed |
|---|--:|--:|--:|
| `double_compton` (module complete) | 21 | 9 | 12 |
| `energy_injection` s0 | 117 | 55 | 62 |
| `energy_injection` s1 (partial) | 59 | 48 | 11 |
| `greens` s0 | 94 | 16 | 78 |
| `greens` s1 (partial) | 23 | 0 | 23 |
| `solver` s0 (partial) | 6 | 4 | 2 |
| **total** | **320** | **132** | **188** |

**Escalation conversion rate = 41%** (132/320). The audit's stated prior was
"escalation is expected to convert the large majority" — that prior is **wrong**
and must be corrected in `mutation_audit.md`. Consequence: the remaining 1,717
lean survivors can be neither dismissed as lean-subset artifacts nor claimed as
test gaps.

## 2. The score that can be quoted

- Lean lower bound (all 14 modules, 5,231 viable mutants): **3,194 / 5,231 = 61.1%**
- With the 132 escalation conversions: **3,326 / 5,231 = 63.6% — verified full-suite lower bound.**
- Extrapolating the observed 41% conversion to the un-escalated 1,717:
  **~77%.** This is a projection from a **non-random 16% sample** (shards were
  picked by worker layout, not sampled) — label it as such or omit it.

Quote 63.6% as the number. Anything higher is an estimate.

---

## 3. Code fixes (do these)

**A1 — DC coefficient is implemented twice; only the unused-by-solver copy is
anchored.** `dc_prefactor(θ)·H_dc(x)` (solver hot path, `solver.rs:454,1161`) and
`dc_emission_coefficient(x,θ)` (`greens.rs:419`, `kompaneets.rs:1387,1490`) are
two hand-maintained copies of K_DC. `test_dc_br_ratio_pinned_z1e6` exercises only
the second. BR has exactly the consistency test DC lacks
(`bremsstrahlung.rs:504`, fast vs reference, rel < 1e-10) — which is why BR
scored 88% and DC's production prefactor did not.
→ Make `dc_emission_coefficient` call `dc_prefactor(θ) * dc_high_freq_suppression(x)`
(single implementation), or add the 1e-15 identity test. Kills 4 survivors.

**A2 — `dc_emission_coefficient_fast` is test-only.** Production inlines
`dc_pre * dc_high_freq_suppression(x)` rather than calling it. Its 4 survivors
(`→0.0`, `→1.0`, `*→+`, `*→/`) are dead-code artifacts. Either have `solver.rs`
call it (preferred — removes the duplication) or `#[cfg(test)]`-gate it. This is
**F-R2-3**, the third instance of the `dc_heating_integral` / `br_heating_integral`
pattern.

**A3 — `visibility_j_t` has no production caller** (`greens.rs:105`; the only
reference is `greens.rs:961`, its own bounds test, where `jt` is computed and
never asserted). `→0.0`/`→1.0` survive trivially. Delete, or assert it in the
bounds test. **F-R2-4.**

**A4 — `test_dc_br_ratio_pinned_z1e6` tolerance is ±2.5×** (asserts
`8 < ratio < 50` for a quantity independently derived as ≈17.06 at z=1e6, P1-8).
CLAUDE.md #9 applies. Tighten to the derived value ±20% — but **re-derive at
x=1 first**; the 17.06 anchor is at x=0.1 and the test runs x=1. A1+A4 together
are what actually kill the `dc_prefactor` survivors; neither alone suffices.

**A5 — CI parity check compares inputs only.** `.github/workflows/ci.yml:130-144`
asserts the committed fixture's *inputs* match the freshly generated ones, never
the outputs. The committed golden file's outputs can drift from Rust silently.
One-line fix: compare outputs too, within tolerance.

**A6 — cleanup.** Delete `python/mutants/` (19 MB, untracked mutmut scratch) and
the `python/data → ../data` symlink (mutmut setup workaround), or gitignore both.
Delete `/tmp/claude-1000/spx-mut{,-b,-c}`.

## 4. Tests to add (evidence-backed gaps)

**B1 — DC relativistic correction `1/(1+14.16 θ_z)` is unpinned at any θ_z where
it matters.** Every existing DC gaunt test passes `θ_z = 0.0` exactly
(`double_compton.rs:174,185,227,238`), so the correction is identically 1 and
`/→*`, `+→-` are unobservable. 3 `dc_gaunt_factor` + 2 `dc_prefactor` full-suite
survivors. → Add a value test at z ≈ 3×10⁶ (θ_z ≈ 1.4×10⁻³, correction ≈2%) and
z ≈ 10⁷ (≈6%). Anchor: Chluba+ 2007 / CS2012.

**B2 — `interp_2d` — 26 of 28 mutants survive the full suite.** The bilinear
(z,x) interpolation behind `TabulatedPhotonSource`. Pure numerics, trivially
anchored: interpolate an exactly-bilinear function and require machine-precision
recovery, plus a corner/edge/out-of-range set.

**B3 — `distortion_from_heating` (15) and `mu_y_from_heating` (10) survive the
full suite** despite being production Green's-function entry points. Not covered
by the parity fixture (only scalar `greens_function` is). Genuine gap.

**B4 — He-recombination Saha.** `saha_he_i`/`saha_he_ii` = 15 lean survivors; the
X_e anchor `test_xe_vs_recfast_milestones` checks only z = 1100/800/200, all
H-dominated. The HyRec-2 He-epoch reference numbers already exist in
`dev/audit/xe_hyrec_comparison.md` (5.7% at z ≈ 1600–2000) — they are just never
asserted. → Add milestones at z ≈ 1600/2000/2500.

**B5 — validation-guard clusters** (`InjectionScenario::validate` 31 residual,
`SolverConfig::validate`, `GridConfig::validate`, `Cosmology::validate`). Low
physics consequence, cheap to close, or explicitly declare them out of scope in
the referee reply. Pick one and say so.

## 5. Reporting corrections (`mutation_audit.md`)

**C1 — the distortion hypothesis is wrong.** The audit (lines 451-465) flags
`decompose_nonlinear_be` as a "candidate alternative/experimental decomposition
path not on any production route." It is **the production path**:
`decompose_distortion` (`distortion.rs:398`) calls it directly, and `decompose`
wraps that. Every published μ and y goes through it.
The correct explanation of its 181 survivors is structural, not dead code:
~140 sit in the Levenberg–Marquardt block (`distortion.rs:316-341` — cofactor
expansion, damping, backtracking). **LM with backtracking only accepts a step
that lowers χ², so a mutated search direction changes the path, not the fixed
point.** Those are equivalent-by-construction, and no finite-tolerance test can
or should kill them. The ~19 at lines 255 (`model_at` basis assembly), 261-262
(residual and weights) and 270 (GS bootstrap) *do* move the fit and are real.
Caveat: distortion was never escalated — these are lean-only numbers.

**C2 — the greens photon-path survivors are mostly a harness-scope artifact, not
a gap.** `tau_ff_survival` (34/34 missed) and `alpha_cs`/`beta_cs`/`f_cs` (34/34
missed) are reached by `photon_survival_probability_numerical` and
`greens_function_photon`, both in the `test_parity.py` dispatch. The `parity` CI
job regenerates the fixture from current Rust and runs Python against it, so a
Rust mutation shows up as a Python↔Rust disagreement — but that job is pytest,
never `cargo test`, so cargo-mutants could not run it. → Spot-check 2-3
representative mutants against `pytest tests/test_parity.py` before writing this
down as fact; then reclassify all 68.

**C3 — state the truncation honestly.** Escalation covered 16% of survivors at a
41% conversion rate; the campaign is reported as a verified ≥63.6% lower bound
with per-module lean bounds, not a headline score. No silent caps (CLAUDE.md).

**C4 —** update `ROUND2_STATUS.md`: R2 → CLOSED (truncated). **C5 —** write the
referee-reply paragraph (`mutation_audit.md` placeholder, line 542).

## 6. Optional — the one run worth doing before closing (~4 h, not 13 days)

Instead of escalating all 1,717 remaining survivors, escalate **one
representative mutant per un-sampled cluster** (~20 mutants × ~13 min ≈ 4 h on a
single worker). Clusters that matter: `kompaneets::kompaneets_step_coupled_inplace`
(156), `solver::step_with_dz` (180), `solver::dcbr_heating_with_derivative` (95),
`solver::adaptive_dz` (73), `distortion::decompose_nonlinear_be` lines 255/261-262,
`recombination::saha_he_*`. The audit's prior is that `mms_convergence` /
`convergence_order` / `conservation_fuzz` catch these; the measured 41% conversion
rate says test that prior rather than assert it. Config is already in place
(`.cargo/mutants.toml` in the worker copies).

---

## 7. Physics validation checks worth adding

**P1 — PDE thermalization test at z_h ≈ 3×10⁶ (highest value).**
The campaign's most informative single result: scaling K_DC by **1.535×**
(`(3.0*π) → (3.0+π)` in `dc_prefactor`) passes the entire suite, including
`science_deep_thermalization_pde` (PDE μ vs Chluba-GF μ at z_h = 1e6, 5%
tolerance). This is **not** sloppiness — it is intrinsic insensitivity:

  μ ∝ exp(−τ), τ = (z/z_th)^{5/2} ∝ √K_DC  ⟹  ∂lnμ/∂lnK_DC = −τ/2.

At z_h = 1e6, τ ≈ 0.18, so −∂lnμ/∂lnK_DC ≈ 0.09 and a 53% K_DC error moves μ by
only ~4% — under the 5% tolerance. Every PDE thermalization test sits at
z_h ≤ 1e6; the z_h = 5e6 tests are Green's-function-only and by construction
cannot test the code's own DC rate.
→ At z_h ≈ 3×10⁶, τ ≈ 2.8, the same 1.535× error changes μ by a **factor 2**,
with μ/Δρ ≈ 0.08 (comfortably resolvable). One PDE run converts the DC emission
normalization from "constrained to a factor ~1.5" to "constrained to ~10%", and
gives the paper's z_th ≈ 1.98×10⁶ claim a direct PDE-side anchor it currently
lacks (`Z_MU` appears only as a constant and in GF tests).
*Confidence: the sensitivity scaling is analytic; the claim that the new test
kills the mutant is a prediction, verify by running it.*

**P2 — τ_ff value anchors at z_h < 5×10⁴.** `tau_ff_survival` is the
low-z branch of `photon_survival_probability_numerical` (the z_h > 5e4 branch
short-circuits to the analytic μ-era form). Its in-module tests
(`greens.rs:876-903`) only bounds-check 0 ≤ P_s ≤ 1. This is the free-free
absorption that sets the FIRAS photon-injection limits in **this fork's core
regime** (post-recombination, locked-in distortions). → Anchor the critical
frequency: solve τ_ff(x_c) = 1 and compare against the Chluba 2015 x_c formula;
add the x ≫ x_c → 1 and x ≪ x_c → 0 limits as class-(i) checks.

**P3 — Compton-broadening moment identity.** `broadened_bump` and its
`f_cs`/`alpha_cs`/`beta_cs` helpers appear **nowhere in `tests/`**. They were
verified line-by-line against Arsenadze App. D in Round 1
(`arsenadze_broadening_audit.md`, no bugs) but have no regression test.
→ Kompaneets diffusion gives ⟨Δx²⟩ = 2x²y_γ to leading order; assert the
broadened bump's second moment against that in the small-y_γ limit. Class (i),
and it survives future refactors in a way a one-time hand audit does not.

**P4 — DC/BR detailed balance.** With n = n_pl(x/ρ_e) and T_e = T_z·ρ_e the net
DC+BR source must vanish identically at every x — a single test that pins the
sign *and* relative normalization of both coefficients at once, and exercises the
CLAUDE.md #5 Taylor branch on both sides of the |ρ_e−1| = 0.01 switch.
*Grep for `detailed.balance|kirchhoff` in `tests/` found nothing; confirm it
isn't present under another name before writing it.*

**P5 — R1 CLASS `sd` Cases B–D** (decay / s-wave / transition). Still the
largest available independent-code (iii) upgrade for the heat channel
(coverage-matrix rows 1, 2, 4, T1). Blocked on the
`DM_decay_Gamma` → spectroxide unit mapping, which was derived in the Round-2
session and is recorded in `ROUND2_STATUS.md`.

**P6 — R3 refsolver photon case — now the top priority after R2.** Three
independent lines of evidence converge on the same place: the coverage matrix
flags rows 6/7 as the only channel without an independent-code anchor; R5 was
cancelled, removing the digitized-curve fallback; and the mutation campaign
independently found the photon Green's-function path (τ_ff, broadening,
`distortion_from_*`) to be the least-pinned code in the repository. R3 is the
only remaining anchor for it.

## 8. Explicitly not doing

- Full escalation of the remaining 1,717 survivors (~13 days).
- Python `mutmut` re-run against the full `python/tests/` suite. The raw 51% is a
  test-*selection* lower bound and stays labelled as such. **If any Python triage
  happens, it is `firas.py`:** its 155 survivors concentrate in
  `_joint_fit_floating_T` (~60), `limit_on_model` (16), `fit_distortion` (13) and
  `profile_limit_floating_T` (13), plus `chi2_from_solver` (25, "no tests"). That
  is exactly the floating-T profiling code behind the SciPost Fig. 8 referee
  question — referee-facing, unlike the rest of the backlog.
