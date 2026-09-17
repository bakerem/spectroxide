# Performance Audit — 2026-08-18

Constraint: no functionality or accuracy changes. Findings classed (a) bit-identical,
(b) last-ulp equivalent, (c) changes results measurably (flagged, not applied).

**Status 2026-08-18 (later same day):** findings 1 and 6 are APPLIED to the tree
(analytic dh/dρ_e replacing the FD triple — user-approved with measured shifts
≤ 9e-9 relative, worst case photon-injection y; and the sweep work queue, verified
byte-identical 12-thread vs 1-thread). All 479 release tests green afterward,
including cosmotherm_comparison; a new unit test pins the analytic derivative
against a fresh FD (< 1e-6 at three states). Per-solve gain 25–35%.

**Status 2026-08-19:** findings 2–5 are also APPLIED (rebased over the analytic
derivative). Everything below is now in the tree; the stale patch file was removed.
Cumulative: z_h=1e5 single-burst 79.7 s → 30.2 s (≈2.6×); default sweep → 36 s.
Verification: 479 release tests green, clippy clean both feature configs,
debug-mode kompaneets/solver lib tests exercise the pitfall-#10 asserts;
findings 3/4/5 proven bit-identical by rebuilding intermediate configurations;
finding 2 measured at ≤3.2 ulp on K_BR (80k samples, no signed bias), worst
observable shift y at z_h=3e6 by 1.2e-9 relative (abs 1.9e-18 — a near-null
least-squares residual there, amplified by conditioning; see note below).

**Conditioning note (pre-existing, not introduced):** |dln y/dln K_BR| ≈ 1.2e5
at z_h = 3e6 (vs 0.08 at z_h = 1e5) because deep-μ-era y is a near-null fit
residual (y/μ ≈ 0.002 there). Since the Gaunt factor is itself a ~1%-accurate
fit, y at z_h ≳ 10⁶ is not numerically meaningful at its current amplitude —
fine (it is physically ≈0 there), but do not quote it.

## Measured baseline

`solve single-burst --z-h 1e5` (n=2000, 103,987 steps): **79.7 s**, single-threaded.
Attribution (rdtsc-instrumented build + differential timing):

| component | share |
|---|---|
| `dcbr_heating_with_derivative` (solver.rs) | 51% |
| DC/BR rate fill loop | 15% |
| Newton loop (2.73 iters/step avg: residual/Jacobian + 2× Thomas) | 32% |
| everything else (moments, NC shift, X_e/H lookups) | ~2% |

Cross-checks: `--no-dcbr` drops 77 s → 9.8 s at identical step count; `--split-dcbr`
is no faster (83 s), so the cost is DC/BR coefficient evaluation, not the Newton
coupling. ~66% of runtime is libm (`exp`/`ln`): ~24 transcendental calls per grid
cell per step, dominated by 3× Gaunt-factor evaluations in the heating
finite-difference triple.

## Findings (ranked)

1. **(c — NOT in patch)** `dh_drho` central-difference triple, `solver.rs:445–525`:
   the full DC/BR heating integral is evaluated 3× per step (θ_e, θ_e(1±1e-4)) on
   99.2% of steps. Measured end-to-end sensitivity of zeroing it entirely:
   Δμ/μ ≤ 1.2e-9 across burst/photon scenarios. Refreshing it every 8 steps gives a
   further −17% runtime at Δμ/μ ≈ 8e-13, but is class (c): needs CosmoTherm
   re-validation per project rules. An analytic derivative would be cleaner.
2. **(b)** Gaunt factor exp factorization, `bremsstrahlung.rs:61–99`: the softplus
   argument is affine in ln x, so `exp(arg) = x^(−√3/π) · e_Z(θ_e)`; `x^(−√3/π)` is
   grid-constant, `e_Z` is one scalar per step. Removes 2 of 5 transcendentals per
   BR eval (6 exp/cell/step in the FD loop). The ±20 softplus shortcut branches were
   measured never-taken over a full run. Last-ulp only; Gaunt fit is ~1% accurate.
3. **(a)** Thomas solver factors the same tridiagonal twice per Newton iteration for
   the bordered solve (`kompaneets.rs:855–871`); fuse into one two-RHS solve.
4. **(a)** Interface fluxes computed twice (f_r(i) = f_l(i+1)) in the Newton body and
   K_old precompute (`kompaneets.rs:595–617, 735–827`); restructure as one interface
   pass + one cell pass. Also hoist `1/ρ_e²` and per-step `dtau·em[i]`.
5. **(a)** Grid-constant per-step recomputation in `solver.rs`: `planck(x_mid)` per
   cell per step (grid.x_half/dx already exist), `1/x³` per point per step, a `match`
   inside the grid loop, `exact_i4/g3` accumulated for a rarely-taken branch,
   `photon_source_buf` zeroed when no source exists, `n_h(z_mid)` computed twice.
6. **(a)** Sweep scheduling, `cli.rs:1565/1699/1846`: `chunks(n_threads)` is a hard
   barrier per chunk; per-z_h cost spans 623× (128 → 79,813 steps) and the default
   grid is sorted ascending in cost, so the 17-point sweep runs at ~2× parallelism
   on 12 cores (measured: 62 s wall, 128 s user). Replace with an AtomicUsize work
   queue (std-only). Ceiling: wall → cost of the single heaviest point.

## Verification

- Patch build passes the full `cargo test --release` suite (all binaries, 0 failures).
- Independent spot-check (this session, separate binaries): 79.7 s → 46.9 s (**1.70×**);
  μ and y agree to 15 digits, step count identical, max relative Δn difference 1e-9
  (near-zero elements), consistent with class (b) last-ulp from finding 2 only.
- With finding 1 at stride 8 (not in patch): 37.4 s (**2.13×**), Δμ/μ ≈ 5e-13.

## Not worth doing

- Recombination lookup is already O(1) direct index (doc says binary search — doc is
  wrong in the harmless direction). X_e/H/θ_z lookups are ~0% of runtime.
- Release profile already maximal (fat LTO, codegen-units=1, opt-level=3).
- Step count is dtau_max-bound (linear: dtau 10→3 gives 3.33× steps, 3.9× time);
  reducing steps is an accuracy decision, out of scope.
