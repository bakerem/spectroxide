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
2. **R5 photon-injection digitization** (`digitization_request.md` D1/D2):
   try the Chluba-2015 arXiv tarball first; digitize the remainder. Dark-photon
   is already anchored via AxionLimits (no action).
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
