# Evaluate the old Crank–Nicolson half at the backward-Euler electron temperature

## Status

Accepted, 2026-09-23.

## Context

The coupled step (`kompaneets_step_coupled_inplace`, `src/kompaneets.rs`) solves the photon
occupation Δn and the electron temperature ρ_e = T_e/T_z together in a bordered Newton system.
The two halves of that system use different time centering:

- The photon rows are Crank–Nicolson. The old half of the flux, including the (φ − 1) heating
  term, uses the previous step's ρ_e (`rc.rho_e_old`) for both `theta_e_old` and `phi_old`.
- The ρ_e row is backward Euler at the new ρ_e.

So the photons receive 2θ_z Δτ (ρ_old + ρ_new − 2ρ_eq) per step while the gas gives up
4θ_z Δτ (ρ_new − ρ_eq). The difference, 2θ_z Δτ (ρ_old − ρ_new), does not cancel over a run
when X_e changes quickly. The quasi-stationary excess scales as q/X_e, so after recombination
each step loses about ½ Δln X_e of its heat. `dev/audit/heat_delivery_near_recombination.md`
found this with a per-step energy ledger. The predicted gap accounts for 92–99.7% of the
measured gap at z_h = 700 to 2000.

What the error does today (`dev/audit/fix_a_cn_old_half_ab.md`):

- A burst at z_h = 1000 delivers 0.938 of its heat. An independent integration, which takes only
  X_e(z) from the solver, expects 0.99986. Δτ_max barely changes the result because it does not
  bind after recombination.
- A decay with lifetime at z = 1000 delivers 0.849, or 0.908 at a hundredfold smaller Δτ_max.
  The expected value is 0.9942. This is finding N-4 in `dev/REVIEW_2026-09-22.md`.
- The no-injection cooling baseline from z = 1700 is 14% short, and it depends on the step
  layout.
- In the μ era the same lag adds to the first-order-in-Δτ energy residual. At z_h = 1e6 and
  default settings, delivery is 0.990.

The old choice was deliberate. The code keeps θ_e at the step-start value in the diffusion
prefactor, because iterating that prefactor would need an extra Jacobian column and caused
non-convergence after recombination. That comment addresses Newton
stability. It does not address energy bookkeeping between the photon and gas rows.

## Decision

In the coupled step, evaluate the old Crank–Nicolson half at the step's backward-Euler predictor
ρ_e (the `theta_e` argument) in place of `rc.rho_e_old`, for both `theta_e_old` and `phi_old`.
Δn keeps Crank–Nicolson time centering. Only the ρ_e that enters the old half changes, so the
heat the photons receive equals the heat the gas gives up.

The Jacobian and the Newton iteration do not change. The prefactor is still fixed for the whole
step, now at the predictor value. The heating term becomes effectively backward Euler, like the
gas row. A small mismatch remains wherever the predictor differs from the Newton result.

We did not test a narrower variant, which would move only the (φ − 1) source term and keep
`rho_e_old` in the Δn terms. We prefer the tested form.

## Consequences

Measured in a scratch A/B study (`dev/audit/fix_a_cn_old_half_ab.md`):

- **Heat delivery:** every check for injection at z ≤ 2e5 agrees with the independent
  integration to 0.25% at default settings, and the post-recombination results no longer depend
  on step size. The burst at z_h = 1000 delivers 0.99985, and the decay with lifetime at
  z = 1000 delivers 0.9939 (expected 0.9942).
- **μ era:** the delivery error at default settings halves, from 1.0–1.2% to 0.4–0.6% at
  z_h = 1e6 to 2e6. The remaining first-order-in-Δτ residual is a separate open item.
- **Changes to results:**
  - Burst μ and y move by +0.06% to +0.51% (Δρ/ρ = 1e-5, z_h = 3e3 to 2e6).
  - The post-recombination cooling y moves by +16%, which removes a step-dependent error.
  - Dark-photon FIRAS limits move by at most 5e-4.
- **CosmoTherm:** every comparison we ran moves closer to CosmoTherm. The cooling μ error goes
  from 0.87% to 0.59%. The pathological-heating RMS goes from 0.44/0.14/0.68% to
  0.20/0.07/0.49%.
- **Tests:** all tests pass in the default and `axion` configurations except
  `coverage_gaps::smallest_accepted_grid_runs_and_warns`. That test asserts a finite result on a
  10-point grid, where both versions return garbage. Move it to 50 points.
- **Follow-up:**
  - Figures that show post-recombination cooling need regenerating.
  - Close N-4 as a duplicate of this record.
  - The energy-closure skip in `src/solver.rs` (`ENERGY_CHECK_Z_LATE`,
    `ENERGY_CHECK_MAX_LATE_FRACTION`) assumes that about 5% of late heat "stays lost". That
    premise no longer holds. Narrow the skip to the real adiabatic loss in a later change.
- **Comments:** rewrite the `rho_coupling` doc comment and the "genuine CN in θ_e" comment in
  `src/kompaneets.rs`. The Newton loop never refreshes the prefactor, so that comment is already
  wrong on `main`.
- **Reversal:** easy, since the change is two expressions.

## Addendum (2026-09-23)

A physics audit found that the Decision overstates the energy balance. With a
discrete zero-flux Kompaneets step, photon energy gain minus gas energy loss
per step is

    M = 2θ_z Δτ G₃ [(ρ_a − ρ_new)(2 − ρ_eqⁿ⁺¹/ρ_new) + (ρ_eqⁿ − ρ_eqⁿ⁺¹)]
        + θ_z Δτ (Q − 4G₃)(ρ_e − 1),

where ρ_a is the ρ_e of the old half and of the prefactor (ρ_old before this
record, the predictor ρ_p after it), ρ_eqⁿ is the ρ_eq in `rho_source`, built
from the step-start Δn, and ρ_eqⁿ⁺¹ is the equilibrium of the new Δn. Q = Σ x⁴
n_pl(1 + n_pl) Δx is the discrete zero point, and the last term is first order
in ρ_e − 1. The derivation also treats the flux moment of Δn as equal to the
solver's ρ_eq. That holds to discretization accuracy, and up to the terms the
perturbative ρ_eq drops (Pitfall #4), which are second order in Δn times
ρ_e − 1. We checked this form against `kompaneets_step_coupled_inplace` and
`update_temperatures` in `src/solver.rs`, and a fresh-context verifier
rederived it with a symbolic check.

So the fix removes only the first term's (ρ_old − ρ_new) driver. Three
corrections to the Decision's "equals ... up to the difference between the
predictor and the Newton result":

- The predictor-minus-Newton term carries a factor 2 − ρ_eq/ρ_new, not 1. At
  z = 200, where T_e/T_z = 0.854 (`dev/scripts/heatloss/baseline_expectation.py`),
  that factor is 0.83.
- The ρ_eq-lag term ρ_eqⁿ − ρ_eqⁿ⁺¹ remains. It is the start-of-step ρ_eq lag
  that investigation I-1 (`dev/REVIEW_2026-09-22.md`) diagnosed and that the
  rejected ADR 0003 would have removed, and the (Q − 4G₃) term is the second
  I-1 cause. They are why the soft-photon energy residual survives this
  record: removing both took I-1's x = 0.01, z_h = 3e5 closure from +22% to
  −0.4%. In the μ era they add to the predictor term, and I-1 left −8% at
  z_h = 2e6 unexplained after removing both.
- The Context formula "photons receive 2θ_zΔτ(ρ_old + ρ_new − 2ρ_eq)" is first
  order in ρ − 1. The exact new-half contribution is (ρ_a/ρ_new)(ρ_new −
  ρ_eqⁿ⁺¹), not ρ_new − ρ_eq.

Known limitations, both on non-default paths that this record does not change:

- `cn_dcbr` pairs Crank–Nicolson DC/BR photon rows with a backward-Euler
  H_dcbr term in the gas row, so the DC/BR exchange does not balance per step.
- On the split DC/BR path (`!coupled_dcbr`), the predictor in
  `update_temperatures` includes H_dcbr while the Newton ρ_e row does not
  (no `DcbrCoupling` is passed). The predictor differs from the Newton result
  by construction there, so the first term of M does not vanish.

## Addendum 2026-09-24: energy-closure late rule

The follow-up on `ENERGY_CHECK_Z_LATE` is done (`d40e349`, merged into local `main`). The cutoff
is now z = 600, where burst heat loses 0.64% physically, and the late fraction that triggers the
allowance is 3%. Above that, the check no longer skips: it widens the shortfall bound by the late
heat and the excess bound by any late cooling. This is a worst-case bound, not the "real adiabatic
loss" named above, because heat injected just above z_end can be lost almost entirely by an amount
z_end sets. Loss table and reasoning: `dev/audit/heat_delivery_near_recombination.md`, section
"Energy-closure late rule".

## Addendum 2026-09-24: `cn_dcbr` removed

The first known limitation above no longer applies. `SolverConfig::cn_dcbr`, the builder setter,
and the `--cn-dcbr` flag are gone (bloat review P-5, `dev/REVIEW_BLOAT_2026-09-24.md`). Nothing
turned the option on, and pitfall #3 in CLAUDE.md already rules out Crank–Nicolson for the stiff
DC/BR rates at low x. DC/BR now always uses backward Euler inside the Newton solve.
