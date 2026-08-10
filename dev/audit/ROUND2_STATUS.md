# Validation Round 2 — status & running log

**Plan:** dev/PLAN_VALIDATION_ROUND2_2026-07-06.md. Complements the Round-1
`AUDIT_SUMMARY.md` (kept separate to avoid entangling this session's commits
with pre-existing uncommitted Round-1 edits in that file). B5 protocol: any
confirmed bug → fix → rerun affected figures → before/after numbers here.

Session 2026-07-06 progress:

| WS | Item | Status | Deliverable |
|---|---|---|---|
| R0 | Coverage matrix | ✅ done | `coverage_matrix.md` |
| R4.1 | mpmath oracles | ✅ done | `highprec_numerics.md`, `dev/scripts/highprec_oracle.py`, `dev/output/highprec/oracle.json` |
| R4.2 | Miri on unsafe kernel | ✅ **GREEN** | 7 kernel tests, `miri-kernel` CI job; 0 UB detected |
| R1 | CLASS `sd` comparison | 🔴 **Case A RETRACTED 2026-07-30** (adiabatic-cooling double count; R1-A withdrawn, new open item R1-A′ on the y excess). Fix shipped (`--subtract-cooling`, reconstruction validated to 5% of the literature μ_cool); **rerun not done**. B–D still scaffolded. | `class_sd_comparison.md`, `dev/scripts/class_sd_compare.py` |
| R2 | Mutation testing | 🛑 **CLOSED — TRUNCATED (EB, 2026-07-26)**. Lean sweep complete (5,264 mutants); escalation stopped after 320/2,037 survivors (16%, 41% conversion). Verified full-suite lower bound **63.6%**. Close-out fixes: `R2_WRAPUP_TODO.md` | `mutation_audit.md`, `R2_WRAPUP_TODO.md`, snapshots `dev/audit/mutation/{,rust_escalation/}` |
| R3 | Clean-room refsolver | ✅ **all five cases GREEN 2026-07-27** — every case inside the contract's acceptance band (0.32–0.87% vs bands of 2–5%); photon case needs one reconciliation before quoting (see below) | `dev/refsolver/` + `STATUS.md`, `outputs/results.json`, reference side `dev/audit/r3_reference_side.json`, `contract.md` §5 correction |
| R5 | Literature curves | ❌ **CANCELLED (EB, 2026-07-11)** | `digitization_request.md` kept for reference; `test_literature_curves.py` stays skipping. Consequence: R3 is now the *sole* planned independent anchor for the photon-injection channel (coverage matrix rows 6/7) — R3 priority raised. |
| R6 | Repro capsule | ⬜ not started | — |

## Findings (Round 2)

- **R4-1 (LOW, candidate refinement, no figure impact).** The DC/BR source
  near-cancellation Taylor branch (`solver.rs`) has relative error up to ~7% at
  large x near the |ρ−1|=0.01 window edge, but only where DC emission is
  exponentially suppressed (H_dc∝e⁻²ˣ). The 0.01 switch threshold could drop to
  ~10⁻³ to keep both branches <0.5% everywhere. Not a defect. **Flag to EB.**
- ~~**R1-A (paper-text scope, no bug).** Case A shows the paper's "PDE↔Green's-
  function agree to 2–5% in μ" holds for *clean* single-era injections but not
  for a broad transition-spanning heating history (there the PDE puts ~34% less
  in μ / more in y than the branching-ratio method, while total distortion
  energy still agrees to 5.6% across CLASS/PDE/GF). Recommend scoping the claim
  in the paper. **Flag to EB.**~~
- **R1-A — RETRACTED 2026-07-30. Do not act on it; do not scope the paper claim
  on its basis.** Case A double-counted adiabatic cooling: CLASS's
  `_sd_heating.dat` includes the first-order photon-baryon cooling term
  (`external/heating/noninjection.c:197,313` → `source/distortions.c:862`), and
  the spectroxide PDE models that same cooling internally and unconditionally via
  Λρ_e. Measured size of the spurious second copy: **μ = −2.83×10⁻⁹** (validated
  against the literature −2.7…−3×10⁻⁹ for pure ΛCDM cooling), i.e. **51% of the
  5.6×10⁻⁹ μ gap that R1-A was built on.** Corrected first-order estimate: μ gap
  34% → 17%. The eyeball convention check missed it because acoustic dissipation
  dominates the column at every z, so no entry ever turns negative.
  **New open item (R1-A′): the y excess is real and now larger.** Ours is
  4.64×10⁻⁹ vs CLASS 3.45×10⁻⁹; the correction *raises* it to ≈5.20×10⁻⁹
  (+34% → +51%). The double count was masking part of it. Unexplained.
  Fix shipped as `class_sd_compare.py --subtract-cooling`; the corrected
  comparison has **not** been run. Full write-up: `class_sd_comparison.md`.

- **F-R2-3 (test-anchoring defect, FIXED 2026-07-26).** K_DC was implemented
  twice — `dc_prefactor·H_dc` in the solver hot loop vs `dc_emission_coefficient`
  in greens/kompaneets — and the class-(ii) Danese & de Zotti anchor tested only
  the copy the solver does not use, leaving the production DC normalisation
  constrained to a factor ~1.5. Deduplicated; verified behaviour-preserving
  (all 40 parity groups regenerate bit-identically). **No published number
  changes** — the code was correct, the *test* was anchored to the wrong copy.
- **F-R2-4 (dead code + vacuous assertion, FIXED 2026-07-26).**
  `greens::visibility_j_t` has no production caller and its only test asserted a
  window ([-0.2,1]) wide enough to admit the constants 0 and 1, under a comment
  describing a different convention. Now pinned to the exact identity
  J_T = 1 − J_bb*.

**Zero confirmed production physics bugs this session** (consistent with
Round 1). F-R2-3/F-R2-4 are defects in the *tests*, not in the physics.

### R3 result — cross-code comparison (2026-07-27)

Both sides run at Δρ/ρ = 10⁻⁵ (the linear regime — see the amplitude note in
F-R3-3; at the contract's 10⁻³ the reference side is 16% nonlinear and the two
codes are not at the same *physical* amplitude even when asked for the same
nominal one). Reference side: production grid (4000), dtau_max = 3, z_end = 200.
Refsolver: Chang–Cooper, N = 2049, x ∈ [10⁻⁴, 40], z_end = 200. Both use the
corrected contract §5 recipe. Observable is normalised by each code's own
*measured* Δρ/ρ.

| case | obs | refsolver | spectroxide | ratio | band | verdict |
|---|---|---|---|---|---|---|
| `heat_z2e6` | μ/Δρ | 0.477808 | 0.479354 | 1.0032 | 2% | **PASS (0.32%)** |
| `heat_z2e5` | μ/Δρ | 1.377098 | 1.382180 | 1.0037 | 2% | **PASS (0.37%)** |
| `heat_z5e3` | y/Δρ | 0.253192 | 0.252176 | 0.9960 | 3% | **PASS (0.40%)** |
| `adiabatic` | μ | −2.248×10⁻⁹ | −2.25969×10⁻⁹ | 1.0052 | 5% | **PASS (0.52%)** |
| `photon_x0.1_z3e5` | μ | −1.717254×10⁻³ | −1.70240×10⁻³ | 0.9914 | 5% | PASS (0.87%), *but see caveat* |

**This is an independent-discretisation anchor, not an independent-code one.**
Chang–Cooper finite-volume vs IMEX Crank–Nicolson, independently written from
`contract.md`, agreeing to ≤0.9% on the dominant component of every case. What
it rules out is scheme-specific discretisation error. It does **not** rule out
shared-specification error: the spec is a common-mode channel, and it produced
two defects this session (F-R3-1, F-R3-3). Project CLAUDE.md is also
auto-injected into subagent context and contains the reference solver's flux
splitting and φ convention, so context isolation was imperfect by construction.
The claim to make in the paper is "independently written from a specification,
different discretisation scheme", not "independent code".

**Photon-case caveat (do not quote the 0.87% yet).** μ agrees to 0.87%, but the
two codes deliver *different energy* for the same nominal ΔN/N = 10⁻³: measured
Δρ/ρ = 4.7231×10⁻⁵ (reference) vs 3.7202×10⁻⁵ (refsolver), a 27% difference,
because the reference deposits over a window z_h ± 7σ_z (σ_z = 0.04 z_h) while
the refsolver deposits instantaneously at z_h. With a 27% energy mismatch, μ
agreeing to 0.87% is not yet demonstrably meaningful. Reconcile the deposition
convention before this row is quotable.

**Subdominant components disagree by 17–27% and that is expected**, not a
defect: the refsolver independently found that the contract's "uniform weights
on the x ∈ [0.5,18] grid" is grid-dependent (on a log grid it is effectively
w ∝ 1/x), and that switching to cell-width weights moves the dominant component
by ≤1.3% but the subdominant ones by 30–60%. Cross-code comparison of
subdominant components requires both sides to fix the weighting first.

**Independent corroborations the refsolver produced en route** (its own
`STATUS.md` has the full list): μ = 1.4006 Δρ/ρ derived from the template
moments alone (vs 3/κ_c = 1.401); implied z_μ = 1.970×10⁶ from
J_bb = μ/(1.4006 Δρ/ρ) at z_h = 2×10⁶ vs the literature 1.98×10⁶ (0.5%);
adiabatic Δρ/ρ = −4.913×10⁻⁹ from the quasi-stationary identity vs −4.854×10⁻⁹
measured (1.2%); DC/BR crossover z ≈ 2.9×10⁵ at x = 0.1, consistent with
F-PC-3's x-dependence finding from the other side.

### R3 findings (2026-07-27)

- **F-R3-1 (spec defect in our own contract, FIXED).** The R3 blocker recorded
  in `dev/refsolver/STATUS.md` — "a pure Compton up-scatter decomposes to a
  spurious μ" — was **not** an error in the clean-room solver. `contract.md` §5
  specified the shared decomposition templates as `G = G_bb/x` and
  `Y_SZ = (G_bb/x)(x coth(x/2) − 4)` but `M = G_bb(1/β_μ − 1/x)`: the three
  templates carry different powers of x, so no combination of them spans a
  pure-y spectrum and the least-squares fit dumps the mismatch into μ. The
  section also defined `M` twice, inconsistently, in one sentence.
  Measured on synthetic inputs through the old text: pure `y = 10⁻⁵` →
  `μ = −6.20×10⁻⁵`, `y = 4.40×10⁻⁵`, `ΔT/T = +2.87×10⁻⁵`, fit residual
  5.8×10⁻³; pure `ΔT/T = 10⁻⁵` → spurious `μ = +2.19×10⁻⁵ = β_μ·ΔT/T`; pure μ
  unaffected. With `G = G_bb` (the correct T-shift shape, since
  `Δn = δ·(−x ∂n_pl/∂x) = δ·G_bb`) all three round-trip to ≤10⁻²⁰ with residual
  ~10⁻¹⁶. Contract corrected in place; the isolated agent was sent the
  correction as a spec fix (contract-level, so isolation is intact).
  Script: `scratchpad/r3/contract_decomp.py`.
  **Lesson for the campaign: a clean-room spec needs its own round-trip
  self-test.** Three synthetic decompositions would have caught this before a
  subagent spent a session chasing it.
- **F-R3-2 (production robustness bug, NEW, open).** The coupled DC/BR Newton
  path produces NaN/Inf in `delta_n` and **panics** (`solver.rs:912`) at
  z ≈ 32.3 for a single burst with Δρ/ρ ≳ 10⁻³ integrated below z ≈ 50.
  Reproducer:
  `spectroxide solve single-burst --z-h 5e3 --delta-rho 1e-3 --sigma-z 200 --z-start 6400 --z-end 1`
  Characterisation: threshold between Δρ/ρ = 5×10⁻⁴ (ok) and 10⁻³ (panic);
  z_end = 50 ok, z_end = 10 panics; survives `--no-dcbr` and `--split-dcbr`
  (⟹ it is the *joint* Newton solve, not the DC/BR physics); unaffected by
  `--no-number-conserving`; grid-insensitive (z = 32.31 default grid → 30.70
  production grid). The `assert!` at `solver.rs:912` is a correct fail-loud
  guard; the NaN originates upstream in the joint Newton solve.
  **Why 400+ tests miss it: the two axes are never crossed.**
  `test_kompaneets_large_perturbation_stability` (`heat_injection.rs:2787`) is
  the one Δρ/ρ = 10⁻³ stability test and it stops at z_end = 10⁴ — three
  decades above the failure; `science_suite.rs:317` reaches z_end = 50 but at
  the usual 10⁻⁵. A single test at (10⁻³, z_end ≤ 10) would have caught it.
  **No published-figure impact** — every paper figure uses
  Δρ/ρ = 10⁻⁵ and z_end = 500, both far from the failure. Consequence for R3:
  the contract's "Δρ/ρ = 10⁻³, integrate to z = 1" cannot be run on the
  reference side; both sides moved to z_end = 200, which is measurably
  equivalent for heat injection (μ/y frozen well above z ~ 50).
- **F-R3-3 (comparison-protocol defect, FIXED in protocol; underlying question
  open).** The contract's photon case is specified by a *nominal* ΔN/N = 10⁻³,
  which is not a well-defined quantity to compare across two codes. Measured on
  the reference side with uniform-trapz weights on x²Δn/(2ζ(3)):
  ΔN/N = 9.8743×10⁻⁴ (−1.26% of nominal) once the injection window closes —
  **grid-independent to five digits across n = 2000/4000/8000**, so it is a
  property of the discretized Gaussian's normalisation, not resolution. It then
  drifts a further ~1.7% by z_end = 200 even with `--no-dcbr`, where Compton
  conserves photon number exactly; that part is consistent with the documented
  P2-2 pitfall (number conservation must be read in the kernel's own quadrature
  weights, not uniform trapz) rather than with lost photons.
  Full ledger at z_end = 200, production grid: 9.2354×10⁻⁴ default,
  9.7031×10⁻⁴ with `--no-dcbr`, 9.2349×10⁻⁴ with `--no-number-conserving` —
  so DC/BR absorption accounts for 4.7 points of the 7.6% net deficit and the
  number-conserving T-shift subtraction for ~0.005% (negligible).
  **Protocol fix:** both sides now report nominal ΔN/N, measured ΔN/N after the
  window closes, measured ΔN/N at z_end, and μ/y/ΔT/T, and the comparison is on
  μ/(measured ΔN/N). Open question deferred out of R3: whether the −1.26% is
  the injection normalisation or the quadrature, which needs the ledger
  recomputed in the kernel's own weights.
  **Also note for the comparison:** the contract's Δρ/ρ = 10⁻³ is outside the
  linear regime for the deep case. Measured μ/Δρ at z_h = 2×10⁶ is 0.5553 at
  10⁻³ vs 0.4793 at 10⁻⁵ (**16%**), and the returned Δρ overshoots the 10⁻³
  input by 19.6%; z_h = 2×10⁵ is linear to 0.3% and z_h = 5×10³ to 5%. Both
  amplitudes are therefore run on the reference side
  (`dev/audit/r3_reference_side.json`) and the agent was asked to do the same.

## R2 mutation testing — installed & verified, runs blocked

`cargo-mutants v27.1.0` (commit a29be7b4) installed and working. Verified it
parses the project and generates mutants (list-only, no test runs):
`src/electron_temp.rs` → 6, `src/dark_photon.rs` → 112, `src/bremsstrahlung.rs`
→ 288.

**BLOCKER — copy-mode (environment-level, needs root/EB):** `.gitmodules` in
this checkout is **not a regular file but a root-owned, read-only `devtmpfs`
bind-mount of `/dev/null`** (`crw-rw-rw- 1 nobody nogroup 1,3`;
`/proc/mounts`: `none on …/.gitmodules type devtmpfs (ro,…)`). EB OK'd
replacing it, but it **cannot be removed/replaced without superuser**
(`rm` → "Device or resource busy"; `umount` → "must be superuser"). Adding it
to `.git/info/exclude` does **not** make cargo-mutants skip it — its safe
copy-mode still tries to copy the device and dies with "Permission denied".
→ **The real fix is to remove the bind-mount at the environment/container
level (EB/root), or run mutation testing in an environment without it.**

**`--in-place` works (POC done on the `mutation-testing` branch).** In-place
skips the tree copy. Verified end-to-end on `electron_temp.rs`:

```
ok       Unmutated baseline in 20s build + 16s test
TIMEOUT  theta_e_with -> f64 with 0.0    (caught: mutation breaks solver
TIMEOUT  theta_e_with -> f64 with 1.0     convergence -> Newton hang -> killed
TIMEOUT  theta_e_with -> f64 with -1.0    by the per-mutant timeout, which
TIMEOUT  replace * with + in theta_e_with  counts as CAUGHT, per R2.2)
```

Baseline passes; the four `theta_e_with` mutants tested so far are all caught
(via timeout — they make the solver non-converge). The run was stopped by the
harness wall-clock after 4/6 mutants; `electron_temp.rs` was `git checkout`-restored.

**To complete R2 (next session):** on the `mutation-testing` branch (isolates
in-place tree mutation from `main`), run each module with `--in-place -t <~3x
baseline>` in an uninterrupted window (or detached, alone — no concurrent builds
or it OOMs), `git checkout` the mutated file after. `run_mutation_shards.sh`
uses copy-mode `-f`; add `--in-place` there once confirmed, OR remove the
`.gitmodules` mount so copy-mode (safer, parallel) works. Then triage survivors
per R2.3 (test-gap / equivalent / unreachable) and run `mutmut` on the 4 Python
limit-pipeline modules. **No mutation *score* until the runs complete.**

## Decisions / inputs needed from EB

1. ~~**P0-6**: planck2015 T_CMB convention~~ **RESOLVED 2026-07-06 (EB: use
   2.7255).** Rust `planck2015()` now → 2.7255 (paper/Fixsen-2009, matches
   Python); added `planck2015_cosmotherm()` → 2.726 for CT comparisons (mirrors
   Python `PLANCK2015_COSMO`). All 7 CosmoTherm tests pass unchanged.
2. ~~**R5 photon-injection digitization**~~ **CANCELLED 2026-07-11 (EB).**
   Dark-photon remains anchored via AxionLimits; photon-injection independent
   anchoring now rests entirely on R3.
3. **R2 `.gitmodules` bind-mount** (see R2 section): EB approved replacing it,
   but it is a **root-owned read-only bind-mount** — needs a *root/container*
   action to remove (I cannot). Until then R2 runs use `--in-place` on the
   `mutation-testing` branch. Removing the mount would let the safer parallel
   copy-mode work.
4. **R4-1 threshold** and **R1-A paper-text scope**: EB said *don't touch the
   paper now* — both are recorded as recommendations for later, no code/paper
   change made. R4-1 would need a coupled-path test rerun if the threshold is
   ever changed.

## Environment note (for the next session)

This is a **7 GB RAM** box: Miri sysroot compilation, the cargo-mutants
169-crate build, and a spectroxide PDE run **cannot run concurrently — they
OOM-kill each other** (observed repeatedly this session). Run heavy builds
one-at-a-time. Detached `setsid` jobs survive only if nothing else is building.

## Kompaneets moment-hierarchy workstream — merged 2026-07-30

Not a Round-2 item; a separate pass (plan
`dev/PLAN_KOMPANEETS_MOMENT_VERIFICATION_2026-07-07.md`) developed on branch
`kompaneets-validation` in worktree `~/spectroxide-kompaneets` and **merged into
`main` as files on 2026-07-30**, since every deliverable was a new untracked file.

**Why it exists.** MMS verifies the *discretization*, not the *equation*: the
manufactured residual is built from an operator transcribed from the code's own
flux form, so a wrong coefficient cancels and MMS still reports p = 2.00. This
suite pins the formulation against anchors derived outside the code. Now recorded
as **CLAUDE.md pitfall #11**.

**What landed:** `tests/kompaneets_moments.rs` (11),
`tests/compton_equilibrium_analytic.rs` (4), `tests/mu_photosphere_profile.rs` (2),
`tests/rate_coefficients_first_principles.rs` (3),
`python/tests/test_firas_coverage.py` (4) + `conftest.py`,
`dev/scripts/{compton_equilibrium_coefficients,gamma_con_landau_zener}.py`,
`dev/audit/{KOMPANEETS_VERIFICATION_RESULTS,gamma_con_lz_check,term_coverage_matrix}.md`.
Deliberately **not** merged: the worktree's `Cargo.toml` (benches stripped for
memory). `COVERAGE_MATRIX.md` renamed to `term_coverage_matrix.md` (case collision
with R0's `coverage_matrix.md`).

**Verification on merge (release mode, current `main`):** 20/20 new Rust tests
pass, 4/4 Python pass, `cargo clippy --release --all-targets -- -D warnings` clean,
and the **full suite passes with them in place — 478 tests, exit 0, zero
failures** (174 unit + 301 integration + 3 doc; 3 `#[ignore]`d; 481 declared).
`--features axion` also run green: 486 pass, 489 declared (the feature adds 8, not
4 — four unit tests in `src/axion.rs` plus the four documented in
`heat_injection.rs`). Python: 333 collected. The re-run mattered: the three commits
`main` had gained rewrote `double_compton.rs` (+231) and `greens.rs` (+457),
including the F-R2-3 K_DC deduplication that the first-principles DC test pins.

**Two findings that propagate into other records:**
1. **γ_con is exonerated** — Landau–Zener ODE integration reproduces the NWA to
   1.2% *at the adiabaticity boundary*, so the ~22% dark-photon offset lives in the
   frozen-vs-thermalized treatment, not the conversion rate. Four audit files that
   listed dead candidates were corrected.
2. **The FIRAS coverage MC is narrower than it looks** — it drives the
   single-amplitude fit, not the floating-`T` profile likelihood where the
   surviving `firas.py` mutants and the paper's published limits both live.

**Top open gap it identifies:** row 13 of `term_coverage_matrix.md`, the y_γ
broadening kernel — no identity, no amplitude anchor, no design-order check, and
∂lnμ/∂ln y_γ = −2.03. Largest unanchored O(1) lever in the code.

**One weakness unchanged:** the Chluba 2015 Eq. 25 coefficients behind the
μ-photosphere test were transcribed to match `src/greens.rs` rather than verified
against the paper. Human spot-check outstanding.

## Resume checklist

- R1 **Case A: rerun `python dev/scripts/class_sd_compare.py --case A
  --subtract-cooling`.** The subtraction is implemented and its reconstruction
  validated (μ_cool = −2.83×10⁻⁹ vs literature −2.7…−3×10⁻⁹); only the corrected
  comparison is missing. Expect the y excess to get *worse* (R1-A′).
- R1 B–D: derive CLASS `DM_decay_Gamma`/`DM_annihilation_efficiency` →
  spectroxide mapping, verify heating histories match <0.1%, then compare μ/y
  (clean deep-μ-era decay case is the unambiguous check — and it is also what
  discriminates the two candidate causes of R1-A′, since it removes the
  transition era).
- R2: **closed, truncated.** Escalation results harvested from volatile /tmp to
  `dev/audit/mutation/rust_escalation/`; the three `/tmp/claude-1000/spx-mut*`
  worker trees can be deleted. Remaining work is the fix/report list in
  `dev/audit/R2_WRAPUP_TODO.md` — no further mutation runs except the optional
  ~4 h representative spot-escalation (§6 there).
- R3: relaunch a fresh **isolated** subagent (contract.md + history.csv only);
  start from the decomposition-template check in `dev/refsolver/STATUS.md`.
- R6: `make figures` target + `figures.manifest.json` (hash plotted data, not
  PDF bytes) + Dockerfile.
