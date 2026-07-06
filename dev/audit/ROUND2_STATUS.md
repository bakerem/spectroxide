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
| R1 | CLASS `sd` comparison | 🟡 Case A done, B–D scaffolded | `class_sd_comparison.md`, `dev/scripts/class_sd_compare.py` |
| R2 | Mutation testing | 🚧 installed+verified; runs BLOCKED (see below) | `.cargo/mutants.toml`, `dev/scripts/run_mutation_shards.sh` |
| R3 | Clean-room refsolver | 🚧 WIP (subagent hit session limit) | `dev/refsolver/` + `STATUS.md` |
| R5 | Literature curves | 🟡 request + skipping test done | `digitization_request.md`, `test_literature_curves.py` |
| R6 | Repro capsule | ⬜ not started | — |

## Findings (Round 2)

- **R4-1 (LOW, candidate refinement, no figure impact).** The DC/BR source
  near-cancellation Taylor branch (`solver.rs`) has relative error up to ~7% at
  large x near the |ρ−1|=0.01 window edge, but only where DC emission is
  exponentially suppressed (H_dc∝e⁻²ˣ). The 0.01 switch threshold could drop to
  ~10⁻³ to keep both branches <0.5% everywhere. Not a defect. **Flag to EB.**
- **R1-A (paper-text scope, no bug).** Case A shows the paper's "PDE↔Green's-
  function agree to 2–5% in μ" holds for *clean* single-era injections but not
  for a broad transition-spanning heating history (there the PDE puts ~34% less
  in μ / more in y than the branching-ratio method, while total distortion
  energy still agrees to 5.6% across CLASS/PDE/GF). Recommend scoping the claim
  in the paper. **Flag to EB.**

**Zero confirmed production physics bugs this session** (consistent with
Round 1).

## R2 mutation testing — installed & verified, runs blocked

`cargo-mutants v27.1.0` (commit a29be7b4) installed and working. Verified it
parses the project and generates mutants (list-only, no test runs):
`src/electron_temp.rs` → 6, `src/dark_photon.rs` → 112, `src/bremsstrahlung.rs`
→ 288.

**BLOCKER (needs EB / a fix before real runs):** the repo's `.gitmodules` is a
**`/dev/null` character device** (`crw-rw-rw- 1 nobody nogroup 1,3`, a
submodule-disabling trick), not a regular file. cargo-mutants' default (safe)
mode copies the whole source tree to a scratch dir and **fails to copy the
`.gitmodules` device** ("Permission denied"). Workarounds and why each is
unsatisfactory *right now*:
- `--in-place` skips the copy and works, BUT it mutates the live working tree;
  combined with the 2-min Bash-tool timeout it left orphaned cargo-mutants
  processes mutating `electron_temp.rs` mid-run (cleaned up: file restored, no
  orphans left). Unsafe for unattended runs in this harness.
- Removing/replacing the `.gitmodules` device is possible but it is a
  deliberately-placed device node I did not create — **not touched** (surface to
  EB rather than modify).

**To run R2 (next session / EB):** either (a) have EB confirm it is safe to
replace `.gitmodules` with an empty regular file for the duration of the run
(then copy-mode works and `run_mutation_shards.sh tier1` can go detached), or
(b) run `--in-place` in a dedicated uninterrupted window with the tree committed
first and `git checkout` after. Then triage survivors per R2.3 into
test-gap / equivalent / unreachable, and run `mutmut` on the 4 Python
limit-pipeline modules. Nothing here should be reported as a mutation *score*
until the runs actually complete.

## Decisions / inputs needed from EB

1. ~~**P0-6**: planck2015 T_CMB convention~~ **RESOLVED 2026-07-06 (EB: use
   2.7255).** Rust `planck2015()` now → 2.7255 (paper/Fixsen-2009, matches
   Python); added `planck2015_cosmotherm()` → 2.726 for CT comparisons (mirrors
   Python `PLANCK2015_COSMO`). All 7 CosmoTherm tests pass unchanged.
2. **R5 photon-injection digitization** (`digitization_request.md` D1/D2):
   try the Chluba-2015 arXiv tarball first; digitize the remainder. Dark-photon
   is already anchored via AxionLimits (no action).
4. **R2 `.gitmodules` device** (see R2 section): OK to replace with an empty
   regular file during mutation runs? It currently blocks cargo-mutants
   copy-mode.
3. **R4-1 threshold** and **R1-A paper-text scope**: accept the recommendations
   above? R4-1 needs a coupled-path test rerun if the threshold is changed.

## Environment note (for the next session)

This is a **7 GB RAM** box: Miri sysroot compilation, the cargo-mutants
169-crate build, and a spectroxide PDE run **cannot run concurrently — they
OOM-kill each other** (observed repeatedly this session). Run heavy builds
one-at-a-time. Detached `setsid` jobs survive only if nothing else is building.

## Resume checklist

- R1 B–D: derive CLASS `DM_decay_Gamma`/`DM_annihilation_efficiency` →
  spectroxide mapping, verify heating histories match <0.1%, then compare μ/y
  (clean deep-μ-era decay case is the unambiguous check).
- R2: confirm `cargo-mutants --version`; verify flags in
  `run_mutation_shards.sh` against `--help`; launch `tier1` shard detached
  (alone). Then `mutmut` on the 4 Python limit-pipeline modules.
- R3: relaunch a fresh **isolated** subagent (contract.md + history.csv only);
  start from the decomposition-template check in `dev/refsolver/STATUS.md`.
- R6: `make figures` target + `figures.manifest.json` (hash plotted data, not
  PDF bytes) + Dockerfile.
