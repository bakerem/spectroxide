# Fix A A/B study: the Crank–Nicolson old half at the backward-Euler ρ_e (2026-09-23)

**Result.** Fix A removes the heat-delivery error that
`heat_delivery_near_recombination.md` traced to the ρ_e time centering. With
the fix, every heat-delivery and cooling-baseline check we ran for injection
at z ≤ 2e5 agrees with an independent integration to 0.25% or better at
default step sizes, and the result no longer depends on step size after
recombination. Without it, the same checks are off by up to 14%, and a decay
with lifetime at z = 1000 loses 15% of its heat (finding N-4, which has the
same cause). In the deep
μ era (z_h = 1e6 to 2e6) the fix halves the delivery error at default steps,
from 1.0–1.2% to 0.4–0.6%, but a separate first-order-in-Δτ residual
remains. Fix A moves every CosmoTherm comparison we
ran closer to CosmoTherm. The one regression is a test that asserts a finite
result on a 10-point grid, where both versions return garbage.

## What was compared

- **A:** `main` at `9e88ceb`.
- **B:** the same tree with Fix A. In `kompaneets_step_coupled_inplace`
  (`src/kompaneets.rs`), the old Crank–Nicolson half uses the step's
  backward-Euler predictor `theta_e` for `theta_e_old` and `phi_old`, in
  place of `rc.rho_e_old`. This makes the photon-side heating term consistent
  with the backward-Euler ρ_e row, so the heat the gas gives up equals the
  heat the photons receive.

Both builds came from one scratch worktree, switched at run time by an
environment variable (`SPX_FIX_A`). A second scratch variable
(`SPX_DZFRAC`) lowers the 5%-of-z step cap for the converged references. The
patch is `dev/scripts/heatloss/ledger_and_fix_a.patch` plus that switch.
Unless stated otherwise, runs use N = 4000, Δτ_max = 10, and z_end = 200.

"Independent" below means `dev/scripts/heatloss/heat_delivery_expectation.py`
or one of its two variants (cooling baseline; decaying particle). They
integrate the gas-temperature excess with Compton exchange and adiabatic
cooling, from CODATA constants typed into the script, and take only X_e(z)
from the solver.

## Heat delivery and the cooling baseline

The cooling baseline is the photon Δρ/ρ of a run with no injection.
Delivered heat is the Δρ/ρ of a run with injection minus the matching
baseline, divided by the injected energy.

| Case | A | B | Independent |
|---|---|---|---|
| Baseline, z_start = 1700, 5%-of-z steps | 0.864 | 1.0025 | 1 (−6.105e-10) |
| Baseline, z_start = 1700, Δτ_max = 1e-3, 0.1% steps | 0.9986 | 1.0002 | 1 |
| Baseline, z_start = 5e6 | 0.9774 | 0.9977 | 1 (−5.150e-9) |
| Baseline, z_start = 5e6, Δτ_max = 1 | 0.9874 | 0.9996 | 1 |
| Baseline, z_start = 5e6, Δτ_max = 1, 0.2% steps | 0.9982 | | 1 |
| Burst, z_h = 1000, Δτ_max = 10, 3, 1 | 0.938 at all three | 0.99985 at all three | 0.99986 |
| Same, Δτ_max = 0.3 | 0.939 | 0.99988 | 0.99986 |
| Decay, lifetime at z = 5e3 (z_start = 5e6) | 0.9912 | 0.9999 | 1.0000 |
| Decay, lifetime at z = 2e5 (z_start = 5e6) | 0.9970 | 0.9986 | 1.0000 |

For the baselines, the table gives the ratio to the independent value. A
converges toward B as the steps shrink. B barely moves. The injected energy
of a decay is computed per n_H with T_0 = 2.726 K, the solver's convention
(`energy_injection.rs:819`, `constants.rs:93`). The independent scripts use
T_0 = 2.7255 K; that choice does not change a delivered fraction.

## Bursts across eras (Δρ/ρ = 1e-5)

| z_h | Δμ/μ, B vs A | Δy/y, B vs A | Delivered, A | Delivered, B |
|---|---|---|---|---|
| 3e3 | | +0.26% | 0.9974 | 0.99998 |
| 1e4 | | +0.25% | 0.9976 | 0.9999 |
| 3e4 | +0.35% | +0.16% | 0.9985 | 0.9997 |
| 1e5 | +0.16% | +0.07% | 0.9982 | 0.9992 |
| 2e5 | +0.15% | +0.06% | 0.9973 | 0.9987 |
| 5e5 | +0.30% | +0.25% | 0.9944 | 0.9974 |
| 1e6 | +0.51% | +0.47% | 0.9904 | 0.9956 |
| 2e6 | +0.28% | +0.23% | 0.9880 | 0.9938 |

We leave Δμ/μ blank where μ is below 2% of y. In the μ era the delivered
fraction still converges at first order in Δτ with or without the fix:

| z_h = 2e5, Δτ_max | 10 | 3 | 1 | 0.3 |
|---|---|---|---|---|
| A | 0.99729 | 0.99924 | 0.99980 | 0.999996 |
| B | 0.99874 | 0.99968 | 0.99995 | 1.00004 |

At z_h = 2e6 and Δτ_max = 1, A delivers 1.00052 and B delivers 1.00111. Fix A
roughly halves the Δτ-dependent μ-era residual at Δτ_max = 10. It does not
address the deep μ-era excess, which is still open
(`energy-conservation` notes in project memory, I-1 in the review plan).

## Comparisons with CosmoTherm

| Check | A | B |
|---|---|---|
| `cosmotherm_comparison.rs` (8 tests, database included) | 8 pass | 8 pass |
| Adiabatic cooling μ vs CosmoTherm | −0.87% | −0.59% |
| Decay vs GF database, lifetime at z = 2e5: Δμ/μ, Δy/y | −0.16%, −0.93% | +0.01%, −0.87% |
| Decay vs GF database, lifetime at z = 5e5: Δμ/μ, Δy/y | −0.64%, −3.13% | −0.34%, −3.00% |
| Pathological-heating figure, RMS vs CosmoTherm GF: sine | 0.443% | 0.198% |
| Same: Gaussian | 0.139% | 0.074% |
| Same: double power law | 0.682% | 0.490% |

The pathological-heating runs use the paper-figure settings: N = 8000,
`dy_max` = 0.005, number-conserving mode, and z = 1e3 to 3e6 tabulated.

## Other outputs

- **Dark-photon FIRAS limits** at five masses (z_res = 222 to 1.4e5): ε
  changes by at most 5e-4 relative (m = 9e-7 eV). The four masses with
  z_res ≤ 6940 change by less than 3e-5.
- **Photon injection:** x_inj = 1, z_h = 1e4, ΔN/N = 5e-8: μ unchanged, y
  changes by 2e-5. For x_inj = 0.01 at z_h = 3e5, the energy-closure excess
  from I-1 falls from 22% to 13% (measured in the earlier session, recorded in
  the I-3 row of `dev/REVIEW_2026-09-22.md`; not rerun here).
- **Post-recombination cooling y** (baseline, z_start = 1700): +16%. This is
  the size of the step-dependent error in A, not a new physical effect.
  Figures that show post-recombination cooling should be regenerated if the
  fix lands.

## Test suites

- **Default release suite, B:** 505 pass, 1 fails.
- **Full suite with `--features axion`, B:** everything passes except the
  same test.
- **Convergence-order suite:** 8 pass in both A and B.

The failure is `coverage_gaps::smallest_accepted_grid_runs_and_warns`. It
runs a burst on a 10-point grid and asserts only that Δρ/ρ is finite. A
returns Δρ/ρ = 15.9 with ρ_e at its cap, which is finite and meaningless. B
returns NaN, which the solver reports as an error. At 20 points the two
builds swap: A fails and B returns garbage. At 50 points and above both are
finite and within 0.3% of each other. If the fix lands, move that test to
50 points, or have it accept the NaN error. Neither version is right on a
10-point grid.

## Finding N-4 has the same cause

N-4 in the review plan reported that a decay with lifetime at z = 1000 loses
17% of its heat at default settings. We reran it as a decay with Γ = 1/t(z =
1000) = 7.18e-14 s⁻¹, f_X = 10 eV (injected Δρ/ρ = 6.537e-9), z_start = 5e4,
and z_end = 200. The independent integration expects 0.9942 of the heat to
reach the photons: 0.53% goes to adiabatic cooling of the gas excess, and
0.05% is still in the gas at z = 200.

| Δτ_max | Steps | A | B |
|---|---|---|---|
| 10 | 960 | 0.849 | 0.9939 |
| 0.1 | 92,229 | 0.908 | 0.9942 |

A loses 15% at defaults, and a hundredfold smaller Δτ_max recovers less than
half of that. B agrees with the independent value to 0.02% at both step
sizes. So the N-4 loss is the ρ_e time-centering error, not missing
step-size refinement for continuous heating.

A first attempt used f_X = 1e4 eV. That drove T_e to its cap
(T_e/T_z = 1.5 at z = 988), and the solver warned as designed. Delivery was
0.42 (A) and 0.45 (B), and that run says nothing about N-4.

## Independent check

A fresh-context verifier (2026-09-23) rederived the energy mismatch of a
Crank–Nicolson photon step against a backward-Euler gas row. It reran the
z_h = 1000 burst with the ledger and recomputed every table from the JSON
files. The summed per-step mismatch is −5.7463e-10 in A, against
−5.7447e-10 predicted, and −8.5e-14 in B. The verifier corrected the decay
normalization and the Δτ_max = 0.3 row above. It also noted two points:

- With Fix A the mismatch is 2θ_z Δτ (ρ_p − ρ_new), where ρ_p is the
  predictor. It vanishes only when the predictor equals the Newton result.
  So some residual remains where they differ: the predictor's 1.5 clamp
  against 3 in Newton, and, in the μ era, the linearized DC/BR heating term
  evaluated at the old Δn. That fits the μ-era residual above.
- On `main` the Newton loop never refreshes the diffusion prefactor, so the
  "genuine CN in θ_e" comment in `kompaneets.rs` is already wrong. The
  `rho_coupling` doc comment becomes wrong with the fix. Both need rewriting
  if it lands.

## Not settled here

- **Deep μ-era energy excess:** unchanged in kind (see above).
- **Narrower variant:** moving only the (φ − 1) source term to the new ρ_e,
  and keeping Crank–Nicolson for the Δn terms, was not tested.

## Raw data

Session scratchpad, `fixA_runs/`: `out/` (JSON and stderr per run),
`an.py`, `baseline_expectation.py`, `decay_expect.py`, `patho.py`,
`suiteB_default.log`, `suiteB_axion.log`. The scratch worktree is `fixA/`.
