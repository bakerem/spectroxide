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
- Done after ADR 0004: `ENERGY_CHECK_Z_LATE` and
  `ENERGY_CHECK_MAX_LATE_FRACTION` in `src/solver.rs` rested on the premise
  that about 5% "stays lost". See "Energy-closure late rule" below.
- At z_h = 700, 0.3% of the gap is not explained by the ρ_e lag. It also
  vanishes with the fix or with smaller steps. Its term was not identified.
- The μ-era control (z_h = 2e5) moves from 0.9975 to 0.9989 with the fix,
  mostly through the baseline. We did not investigate its remaining 0.1%.

## Energy-closure late rule (2026-09-23, after ADR 0004)

The post-run energy-closure check (R-1) used to skip any run with more than
10% of its heat below z = 2000, on the premise that about 5% of that heat
stays lost. ADR 0004 removed the numerical loss. The loss left is physical:
adiabatic cooling of the gas excess, plus heat still in the gas at z_end.
The table gives the fraction of injected heat that does not reach the
photons, from `heat_delivery_expectation.py` (bursts) and
`decay_delivery_expectation.py` (decays, f_X = 10 eV, z_start = 5e4), with
X_e from `spectroxide.ionization_fraction`.

| Injection | Lost, z_end = 200 | Lost, z_end = 100 |
|---|---|---|
| Burst z_h = 1500, σ_z = 60 | 2e-6 | 2e-6 |
| Burst z_h = 1000, σ_z = 40 (σ_z = 100) | 6.5e-5 (1.4e-4) | same |
| Burst z_h = 800, σ_z = 32 (σ_z = 100) | 9.3e-4 (1.4e-3) | |
| Burst z_h = 700, σ_z = 28 (σ_z = 100) | 2.7e-3 (3.7e-3) | same |
| Burst z_h = 650, σ_z = 26 | 4.2e-3 | |
| Burst z_h = 600, σ_z = 24 | 6.4e-3 | |
| Burst z_h = 550, σ_z = 22 | 9.6e-3 | |
| Burst z_h = 500, σ_z = 20 (σ_z = 100) | 1.4e-2 (2.6e-2) | 1.4e-2 (2.3e-2) |
| Burst z_h = 400, σ_z = 16 | 3.4e-2 | 3.4e-2 |
| Burst z_h = 300, σ_z = 12 | 0.12 | 0.10 |
| Burst z_h = 200, σ_z = 8 | 0.88 of the heat injected before z_end (0.86 still in the gas) | 0.41 |
| Decay, lifetime at z = 1000 (Γ = 7.184e-14 s⁻¹) | 5.8e-3 | 5.5e-3 |
| Decay, lifetime at z = 500 (Γ = 2.294e-14 s⁻¹) | 0.097 | 0.083 |
| Decay, lifetime at z = 300 (Γ = 1.015e-14 s⁻¹) | 0.22 | 0.29 |

The narrow width σ_z = 0.04 z_h measures the loss of heat injected at z_h.
Parentheses give the CLI default σ_z = max(0.04 z_h, 100). We leave out the
σ_z = 100 rows at z_h ≤ 300, because there the burst extends below z_end.
The scripts divide by the full burst amplitude; for z_h = z_end = 200 we
divide by the half injected before z_end instead.
Commands: `python heat_delivery_expectation.py Z_H --sigma S --z-end Z` and
`python decay_delivery_expectation.py 10 GAMMA 5e4 Z`.

The loss of heat injected at z passes 1% at z ≈ 550. Heat injected just
above z_end can be lost almost entirely, because most of it is still in
the gas.

**Rule.** `ENERGY_CHECK_Z_LATE` = 600, where heat loses 0.64%. If more than
`ENERGY_CHECK_MAX_LATE_FRACTION` = 3% of the gross heat falls below z = 600,
the tolerance allows all of that late heat to be lost
(`energy_closure_allowance` in `src/solver.rs`): the late heating widens the
shortfall bound and the late cooling widens the excess bound, since the gas
can equally keep a temperature deficit from the photons. A run that misses by
more than the tolerance plus all its late heat still warns. At or below 3%
the plain 5% tolerance applies: 5%, minus 0.7% of physical loss above
z = 600, minus up to 0.6% of numerical delivery error after ADR 0004
(z_h = 1e6 to 2e6, N = 4000, Δτ_max = 10), leaves 3.7% for lost late heat.
The ρ_e-cap rule is unchanged: a capped run may fall short by any amount.

A first version made the check one-sided (shortfall silent, excess warns)
above 3% of late heat. A code review found that this warns falsely on
cooling: a table that removes heat below z = 600 leaves the photons with
less deficit than injected, an excess. Splitting the late heat by sign fixes
that and keeps shortfalls larger than the late heat visible.

We did not build an expected-delivery model that subtracts the adiabatic
loss. It would need X_e(z), the cosmology, and z_end inside the check, and
the table shows that heat near z_end is lost by an amount that z_end sets,
not z_h. The worst-case bound needs none of these.

The check covers only scenarios with a known injected energy, single bursts
and heating tables; the decay rows above are for reference.

Effect: a burst at z_h = 1000 (σ_z = 100) has 3e-5 of its heat below
z = 600, so the check now applies the plain tolerance to it
(`solver::tests::test_energy_closure_late_rule`). Every run the old rule
skipped is now checked, at worst with the late heat added to the tolerance.
A run with 3–10% of its heat below z = 600 and at most 10% below z = 2000
had the plain tolerance before and now has the widened one.

**Where the warning fires now.** We logged every firing in a scratch build
over the full release suite, `cosmotherm_comparison` with the Green's function
database and ignored tests, and a default CLI `sweep`. It fires in seven
runs across six tests, all built to fail closure: six coarse-grid runs
(N = 50 to 500) and `test_extreme_large_injection` (Δρ/ρ = 1e-2,
+6.1%). The only new firing is the z_h = 1600 run on 100
points in `energy_closure_warns_on_coarse_grid`, which the old rule skipped.
The sweep and the CosmoTherm tests do not fire. CLI bursts at z_h = 1300 to
3000 with Δρ/ρ = 1e-7 to 1e-5 and z_end = 500 or 200 are now checked, and
all are silent. At z_h = 800 (Δρ/ρ ≥ 1e-7) and z_h = 1000 (Δρ/ρ ≥ 1e-6),
ρ_e reaches its cap and the photons end 18–93% short; the cap rule silences
that shortfall, as before, and the run warns "Substantial heating".
Late heat reaches the cap at small amplitude (a burst at z_h = 500 caps
at Δρ/ρ = 1e-8), so the cap rule, not the late rule, silences most
late-heat runs.
