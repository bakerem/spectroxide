# Plan: Open Benchmark Pack + Codebase Correctness Audit

**Date:** 2026-07-02
**Context:** SciPost Report 1 (2026-06-30) — referee accepts the code as the core
contribution but wants the validation treated more critically. This plan covers
(A) a published, machine-readable benchmark pack as the substitute for a full
like-for-like CosmoTherm comparison, and (B) a broader correctness audit of the
codebase designed to be *convincing to a skeptical third party* — i.e. every
step produces an artifact whose correctness can be checked without trusting the
auditor.

**Design principle (both parts):** attribution over assertion. A claim counts as
validated only when it is anchored to a source outside the code (analytic
formula, primary-literature derivation, public dataset, independent
implementation) and the anchor is documented next to the claim.

---

## Part A — Open benchmark pack

**Goal:** a versioned `benchmarks/` directory of configs + reference outputs
that any distortion code (CosmoTherm included) can diff against. Flips the
referee's chicken-and-egg objection: the pack itself becomes the "source
outside the code" for future contributions.

### A1. Directory layout & schema

- [ ] `benchmarks/README.md` — the spec. Must pin down every convention that
      has historically caused cross-code confusion: definition of x (normalised
      by T_z), Δn vs ΔI units, G_bb-stripping convention (number-conserving vs
      least-squares), μ/y/ΔT decomposition method, cosmology parameter
      definitions, FIRAS statistic conventions.
- [ ] `benchmarks/configs/<case>.json` — one file per case: full cosmology,
      injection scenario + parameters, solver settings (grid, tolerances,
      z_start/z_end), spectroxide version + git commit + physics hash.
- [ ] `benchmarks/outputs/<case>/` — spectrum Δn(x) and ΔI(ν) (CSV), scalar
      summary (μ, y, ΔT/T, Δρ/ρ, ΔN/N energy/number ledger) (JSON).
- [ ] Every case carries a `target` field: `analytic` / `literature` /
      `public-dataset` / `reference-only`. Reference-only cases (no external
      anchor; for cross-code diff) are labelled honestly as such.

### A2. Case selection (span regimes × processes)

- [ ] Heat injection, single burst: z_h log grid 1e3–5e6 (μ-era, transition,
      y-era). Targets: μ = 1.401 Δρ/ρ, y = Δρ/4ρ, visibility function J_bb.
- [ ] ΛCDM adiabatic cooling (negative μ). Target: literature value.
- [ ] Decaying particle: 3–4 lifetimes spanning the eras.
- [ ] DM annihilation (s-wave, p-wave).
- [ ] Monochromatic photon injection: x_inj × z_h grid matching Chluba 2015 /
      Bolliet+2020 figure coverage.
- [ ] Dark photon resonances: mass grid including the FIRAS-limit templates
      (hooks directly to the CCJ24 comparison notebook).
- [ ] Numerics stress cases (narrow bursts, low/high x_inj) — reference-only,
      for cross-code numerical comparison.

### A3. Tooling

- [ ] Generator: `dev/scripts/generate_benchmark_pack.py` (or a `benchmarks`
      CLI subcommand) — deterministic regeneration from configs.
- [ ] `dev/scripts/compare_benchmark.py` — takes any code's outputs on the same
      configs, produces standard ratio plots + tolerance report. This is what
      we hand to Chluba/Cyr or any future code author.
- [ ] CI job: regenerate pack, assert current code matches committed outputs
      within stated tolerances (regression pinning with declared tolerances,
      not exact-match pinning).
- [ ] Versioning: semver + physics hash; CHANGELOG entry required whenever an
      output changes, with cause attribution.

### A4. Publication

- [ ] Zenodo archive with DOI; cite in the revised paper.
- [ ] Paper appendix: benchmark table (case, target, tolerance, status).
- [ ] CONTRIBUTING section + issue template inviting other codes to submit
      outputs; mini code-comparison request to Chluba/Cyr (3–5 cases, frozen
      X_e table and cosmology) rides on this infrastructure.

---

## Part B — Codebase correctness audit

**Threat model:** the referee's implicit worry is systematic error that
internal tests can't see, because the tests were written by the same agent that
wrote the code (CLAUDE.md pitfall #9 at project scale). The audit must
therefore be (i) anchored outside the code, (ii) adversarial, (iii) documented
per-finding, and (iv) reproducible.

### B0. Test-provenance census (do first — cheap, high signal)

- [ ] Script to extract every numeric assertion from the ~400 tests
      (tests/ + src unit tests) into a table: test name, asserted value,
      tolerance, provenance class = {analytic, literature, dimensional,
      cross-method, code-derived}.
- [ ] Classify each. Every `code-derived` target gets either (a) re-derived
      from an independent source, (b) reclassified as an explicit regression
      pin (allowed, but labelled), or (c) flagged as a gap.
- [ ] Artifact: `dev/audit/TEST_PROVENANCE.md`. This single table is the most
      direct answer to the referee's concern and should be cited in the reply.

### B1. Paper-to-code physics audit (per module, adversarial)

Order by risk (numerical delicacy × constraint impact):

- [ ] `kompaneets.rs` — flux splitting, φ convention, Newton linearisation.
- [ ] `double_compton.rs`, `bremsstrahlung.rs` — emission coefficients,
      Gaunt factor, dimensional check per pitfall #8.
- [ ] `electron_temp.rs` — perturbative Δρ_eq derivation.
- [ ] `greens.rs` + `python/spectroxide/greens.py` — visibility fits vs
      Chluba 2013; photon survival probability vs Chluba 2015 (includes the
      referee-flagged Saha-vs-Peebles divergence at the P_s call site).
- [ ] `dark_photon.rs`/`.py` — γ_con vs CCJ24 Eq. 6, z_res, NWA validity.
- [ ] `recombination.rs` — Peebles rates vs primary sources.
- [ ] `firas.py` — statistics implementations vs their definitions.
- [ ] `distortion.rs` — decomposition conventions.

Protocol per module (one physics-inquisitor pass per module, independent
context each):

1. Re-derive every coefficient from the *primary* reference (not the paper),
   including dimensions and limiting cases, **before** reading the code's
   numeric output.
2. Produce an equation↔code mapping table with per-line verdicts.
3. Findings triaged: confirmed bug / convention mismatch (documented) /
   false alarm (with refutation).
4. Artifact: `dev/audit/<module>_audit.md`.

Human spot-check: EB reviews a sample of memos per module (the audit's own
audit — this is what makes the AI-audit claim credible in the reply).

### B2. Cross-implementation parity (Rust ↔ Python)

The referee's adversarial review found a real Rust/Python divergence; make the
whole mirrored surface systematically tested instead of fixed point-wise.

- [ ] Enumerate all mirrored function pairs (greens, dark_photon, cosmology
      helpers, spectral shapes).
- [ ] Parity harness: sample inputs across validated domains, assert
      Rust-vs-Python agreement to stated tolerance; run in CI.
- [ ] Fix the known P_s ionization divergence first; quantify its effect on
      every published figure that used the Python path.

### B3. Adversarial numerics

- [ ] Method of manufactured solutions on the coupled operator: use
      `Custom`/`TabulatedPhotonSource` to inject a source that makes a known
      analytic Δn(x, z) exact; verify convergence order against it. Strongest
      available verification technique; check feasibility on the Kompaneets +
      DC/BR split first.
- [ ] Property-based conservation fuzzing: random scenario/grid draws, assert
      energy and photon-number ledger closure within budget.
- [ ] Extend convergence-order tests (Richardson bounds) to the coupled
      system; publish as an error-budget figure feeding Part A tolerances.

### B4. Independent replication anchors (paper-facing)

- [ ] X_e-swap experiment: rerun CosmoTherm-comparison figures with a
      CLASS/CosmoRec-quality tabulated recombination history; report how much
      of each residual is attributable to Peebles+Saha. Feeds rewritten
      Sect. 7 (per-simplification new-physics parameter-space bounds).
- [ ] Photon-injection FIRAS limits redone with the reference statistic
      (mirror of the completed Fig. 8 / CCJ24 notebook).
- [ ] Cosmology audit: verify each reference comparison uses the reference's
      exact parameters.

### B5. Findings protocol (non-negotiable)

Any confirmed bug: fix → rerun affected published figures → document impact in
`dev/audit/AUDIT_SUMMARY.md` (before/after numbers) → CHANGELOG. No
theoretical dismissals of impact (CLAUDE.md debugging philosophy).

---

## Phasing

| Phase | Content | Gate |
|---|---|---|
| 0 | B0 provenance census + B2 parity harness scaffold + fix known P_s bug | Census table complete; parity CI green |
| 1 | B1 module audits (risk order) | All memos written; findings triaged; EB spot-check done |
| 2 | B3 MMS + fuzzing + coupled convergence | Error budget quantified |
| 3 | A1–A3 benchmark pack + regeneration CI | Pack regenerates deterministically; tolerances asserted |
| 4 | B4 replication anchors + A4 publication (Zenodo, Chluba/Cyr request) | Figures regenerated; DOI minted |
| 5 | Paper revisions (Sect. 6 wording, Sect. 7 expansion, validation matrix appendix) + referee reply | — |

**Sequencing rationale:** audit before pack — benchmark outputs published
before the audit would risk publishing bugs as reference values; Phase 3
depends on Phase 1–2 findings being fixed.

## Deliverables checklist (referee-facing)

- [ ] `dev/audit/TEST_PROVENANCE.md` — every test target with its independent source
- [ ] `dev/audit/<module>_audit.md` × ~8 + `AUDIT_SUMMARY.md` with triage log
- [ ] Rust↔Python parity CI
- [ ] `benchmarks/` pack + Zenodo DOI + comparison tooling
- [ ] X_e-swap and statistic-swap notebooks
- [ ] Revised Sect. 7 with per-simplification parameter-space validity bounds
- [ ] Validation matrix table in paper appendix
