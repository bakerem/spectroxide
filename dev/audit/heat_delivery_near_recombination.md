# Heat delivery near recombination (2026-09-23)

**Result.** The 5% of burst heat missing at z_h = 700–1000 is a solver
discretization error, not physics of the model. The photon rows and the
electron-temperature row of the coupled Newton step use different time
centering for the Compton heating term. The model itself delivers 99.63%
(z_h = 700) and 99.986% (z_h = 1000) of the heat to the photons.

## Measurements

We ran bursts with Δρ/ρ = 1e-8, the default σ_z = max(0.04 z_h, 100),
z_start = z_h + 7σ_z, and z_end = 200. Delivered is the final photon Δρ/ρ
minus a no-injection run with the same settings, divided by 1e-8. N = 2000
unless stated.

| z_h | Δτ_max = 10 | Δτ_max = 1e-2 | Δτ_max = 1e-3 | Δτ_max = 1e-4 | Independent expectation | Test fix, Δτ_max = 10 |
|---|---|---|---|---|---|---|
| 700 | 0.9508 | | 0.9794 | 0.9954 | 0.99627 | 0.99618 |
| 1000 | 0.9381 (N = 4000: 0.9382) | 0.9881 | 0.99938 | | 0.99986 | 0.99979 (Δτ_max = 1e-3: 0.99976) |
| 1300 | 0.9717 | | | | | 0.99993 |
| 2000 | 0.9966 | | | | 0.999999 | 0.99991 |

The deficit is independent of N and shrinks roughly linearly with step size.
Δτ_max = 0.1 appeared not to converge it because it barely binds below
z ≈ 1100. There dτ per step is 1e-3 to 5e-2, and the step is set by
σ_z/10 = 10 inside the burst window and by 5% of z outside it.

## Independent expectation

`dev/scripts/heatloss/heat_delivery_expectation.py` integrates the gas
temperature excess with Compton exchange, adiabatic cooling, and the injected
heat. It uses CODATA constants typed into the script and takes only X_e(z)
from the solver. It gives delivered fractions of 0.996269 at z_h = 700
(adiabatic loss 3.7e-3) and 0.999862 at z_h = 1000 (adiabatic loss 1.2e-4).
The gas-side column of the solver ledger agrees to 1e-5, so the electron
temperature row, the heating normalization `q_rel * t_c / (4θ_z)`, and the
quasi-stationary balance are all correct. At z = 1000, R/(H t_C) = 2.9e4.
It falls to about 100 at z = 500 and 4 at z = 200. That is why adiabatic loss
matters at z_h = 700 and not at z_h = 1000.

## Cause

We added a per-step ledger (`dev/scripts/heatloss/ledger_and_fix_a.patch`,
`ledger_summary.py`). At z_h = 1000 the gas row hands 9.9986e-9 to the
photons, but the photon energy rises by only 9.381e-9. The mismatch is
−6.17e-10. The ledger also covers DC/BR, the number-conserving correction
(inactive below z = 5e4), and the ρ_e caps (no clamp warnings in any run). None of these
accounts for the gap.

The mismatch comes from `kompaneets_step_coupled_inplace` in
`src/kompaneets.rs`, where `theta_e_old` and `phi_old` are taken from
`rc.rho_e_old`. The Crank–Nicolson old half of the (φ − 1) heating term
therefore uses the previous step's ρ_e. The ρ_e row is backward Euler at the
new ρ_e. Photons receive 2θ_z Δτ (ρ_old + ρ_new − 2ρ_eq), and the gas loses
4θ_z Δτ (ρ_new − ρ_eq). The predicted gap, Σ 2θ_z Δτ (ρ_old − ρ_new), is
−6.14e-10, which is 99.5% of the measured gap. At z_h = 1300 it accounts for
99.7%, at z_h = 2000 for 97%, and at z_h = 700 for 92%.

The gap does not cancel over the burst. The quasi-stationary excess is
δρ_e ∝ q t_C ∝ q/X_e, and X_e falls by about 12% per Δz = 10 step at
z ≈ 1000. The stale half therefore loses about ½ Δln X_e of each step's heat.
The q-driven part telescopes to zero over the burst, but the X_e-driven part
does not.

## Proposed fix (tested in a scratch worktree only)

Evaluate the old half at the step's backward-Euler predictor ρ_e (the
`theta_e` argument) instead of `rc.rho_e_old`. The ledger mismatch then
falls below 1e-4 of the injection for z_h = 700–2000 (1.1e-3 at z_h = 2e5), and the result no
longer depends on step size (table, last column). The fix changes a
numerical method that was chosen on purpose ("genuine CN in θ_e"). It
therefore needs an ADR, and the full test suite must be rerun. A narrower
variant moves only the (φ − 1) source term to the new ρ_e and keeps CN for
the Δn terms. We did not test it.

## What does not matter

- Grid size: N = 4000 changes the result by 5e-5.
- The heating normalization after recombination: it agrees with the
  independent integration to 1e-5.
- The quasi-stationary approximation: stored gas heat is below 3e-13, and
  adiabatic loss of the gas excess is 1.2e-4 at z_h = 1000.
- DC/BR exchange, the ρ_e caps, and the θ_z cutoff.

## Open

- The same lag biases the no-injection adiabatic-cooling distortion. At
  z_start = 1700, the baseline is −5.70e-10 at Δτ_max = 10 and −6.12e-10 with
  the fix, where it no longer depends on step size. From z_start = 2.56e5 it
  is −3.35e-9 and −3.45e-9. Figures that rely on post-recombination cooling
  should be rechecked after a fix.
- `ENERGY_CHECK_Z_LATE` and `ENERGY_CHECK_MAX_LATE_FRACTION` in
  `src/solver.rs` rest on the premise that about 5% "stays lost". That premise
  is wrong. After a fix, the energy-closure skip could be narrowed to the real
  adiabatic loss.
- At z_h = 700, 0.3% of the gap is not explained by the ρ_e lag. It also
  vanishes with the fix or with smaller steps. Its term was not identified.
- The μ-era control (z_h = 2e5) moves from 0.9975 to 0.9989 with the fix,
  mostly through the baseline. We did not investigate its remaining 0.1%.
