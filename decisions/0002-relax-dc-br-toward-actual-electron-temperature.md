# Relax DC and BR toward the actual electron temperature

## Status

Accepted, 2026-09-22.

## Context

Double Compton (DC) and bremsstrahlung (BR) drive the photon spectrum toward a Planck spectrum at
a target temperature ρ_dcbr (`src/solver.rs:1176-1198`). Today the target is not the electron
temperature ρ_e. It subtracts this step's heating increment:
ρ_dcbr = ρ_e − R δρ_inj Δτ/[1 + Δτ(R(1 + H′) + H t_C)], clamped silently to [0.5, 2].
The code comment says that using ρ_e would count the injected energy twice
(`dev/REVIEW_2026-09-22.md`, finding P-8).

That justification is wrong. The photon rows gain Δτ·em·(neq − Δn) from DC and BR
(`src/kompaneets.rs:901-919`). The electron row loses H_dcbr, built from the same `em` and `neq`
arrays (`:978-991`). The energy the photons gain always equals what the electrons lose, whatever
the target. Chluba & Sunyaev (2012), Eq. 8, which the comment cites, relaxes toward n_pl at the
actual T_e. The subtraction also changes with regime: in tight coupling (R Δτ ≫ 1) it removes the
whole heating excess, and after recombination it removes only one step's increment.

An A/B run on 2026-09-22 measured the effect. It used a copy of HEAD 504e96e with an environment
switch, a production grid, z_start = 5e6, and z_end = 500. Raw output is in the session scratchpad
(`p8/`).

| Run | Δμ/μ | Δy/y | Δ(energy closure) |
|---|---|---|---|
| Burst, z_h = 2e6, Δτ_max = 10 | −5.6e-5 | −6.0e-5 | +2.8e-7 |
| Burst, z_h = 2e6, Δτ_max = 1 | −5.6e-5 | −5.9e-5 | −1.9e-7 |
| Burst, z_h = 5e5 | −1.9e-5 | −1.9e-5 | +8.0e-8 |
| Burst, z_h = 2e5 | −2.0e-5 | +5.4e-5 | +3.0e-8 |
| Decaying particle, Γ t(z = 1e6) = 1 | −2.9e-5 | −2.9e-5 | +2.1e-7 |

The sign matches the physical expectation: hotter electrons emit more soft photons, so μ is
slightly smaller at the same energy. The shift does not depend on Δτ_max, so it is not part of the
first-order-in-Δτ energy residual. That residual is 1.2% of the energy closure between
Δτ_max = 10 and 1, about 4e4 times larger. The clamp never engaged: the unclamped value stayed in
[0.988, 1.000013]. No run went below z = 500, where ρ_e is still close to 1.

## Decision

Set ρ_dcbr = ρ_e and delete the injection subtraction. Replace the silent clamp with a check that
records a solver warning when ρ_e leaves [0.5, 2], and keep the clamp as a guard. Rewrite the
comment to state the energy bookkeeping above.

The plan's decision rule placed this in the "below 1e-3" branch: the measured shift is 5.6e-5 at
z_h = 2e6. We switch because the current target has no physical basis, not because the numbers
require it.

## Consequences

- The DC/BR target follows the cited equation. It no longer depends on the coupling regime or on
  the step size.
- Heat-injection results shift by at most 6e-5 relative in μ and y. That is far below the 2–5%
  agreement with the Green's function and CosmoTherm, and invisible in every figure. Photon-injection,
  dark-photon, and axion runs do not change, because their δρ_inj is zero.
- Tests pinned to the current output within 1e-4 relative may need updating. Each such update gets
  a note that it follows from this record.
- A run down to z ≈ 100, where ρ_e ≈ 0.6, can now raise the new warning if ρ_e leaves the
  guard range. We do not know whether any current run does this.

## Addendum

2026-09-23.

The μ-era estimate holds. The A/B table in Context used the CLI `--production-grid` without
`--n-points`, which gives N = 2000, not the 4000-point production grid (finding N-3,
`src/cli.rs:1490`). A rerun at N = 4000 on the implementing branch agrees to two digits:

| Burst | Δμ/μ | Δy/y | Δ(energy closure) |
|---|---|---|---|
| z_h = 2e6 | −5.6e-5 | −5.9e-5 | +3.4e-7 |
| z_h = 5e5 | −1.9e-5 | −1.9e-5 | +7.8e-8 |
| z_h = 2e5 | −1.9e-5 | +5.4e-5 | +2.7e-8 |

The bound of 6e-5 does not hold for heating in the y-era. There, electrons stay hotter than T_z by
δρ_inj while heat goes in, and DC and BR now emit toward that temperature. The
result is a free-free excess at x ≲ 0.01 whose tail reaches the μ fit band. For a decaying
particle with Γ_X = 1e-13 s⁻¹ stopped at z = 3000, μ moves from +1.2e-9 to −5.1e-9 while
y = 6.9e-7: μ is a residual at the 1e-3 level of y, and this emission dominates it. A burst at
z_h = 1e4 (Δρ/ρ = 1e-5, to z_end = 100) shows the same: μ falls from 1.93e-8 to 1.21e-8 while y
changes by 9e-4 relative. The Chluba
(2013) Green's function omits it, so tests check energy closure in that regime, not μ.

For a μ-era decay (lifetime at z = 2e5 or 5e5), PDE μ agrees with the CosmoTherm Green's-function
database convolved with the same heating history to 0.6%, while the Chluba (2013) fit is off by 6%
in μ there (lifetime at z = 2e5).

Injection below z ≈ 1000 is outside the Green's-function formalism. That matches the range of the
CosmoTherm Green's-function database (z = 1000 to 5e6).

Heating strong enough to reach the solver's ρ_e caps (1.5 in the backward-Euler predictor, 3.0
after the coupled Newton solve) now pushes one warning per run. Such heating would ionize the gas,
while the solver holds X_e on its recombination history. In that regime the old target went
negative (ρ_dcbr = −3.2 at z ≈ 910 with the cap raised to 20) and was silently clamped to 0.5.

The implementation lowers the guard on the DC/BR target from [0.5, 2] to [0.05, 2], so that
adiabatic cooling of ρ_e below 0.5 near z ≈ 72 does not warn. Runs that go below z ≈ 72 therefore
change even when δρ_inj = 0: before, DC and BR relaxed toward ρ = 0.5 there, now toward the actual
ρ_e. For photon injection at x = 1, z_h = 1e4, to z_end = 50, μ and y change by less than 1e-9
relative and Δn changes by 1.2% of its peak, at x = 1e-4.
