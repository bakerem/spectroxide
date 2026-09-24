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
