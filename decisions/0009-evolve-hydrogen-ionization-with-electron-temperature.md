# Evolve hydrogen ionization with the electron temperature

## Status

Accepted, 2026-09-25 (EB).

Amends the ionization-history paragraphs of
[ADR 0007](0007-replace-electron-temperature-caps-with-a-sanity-guard.md): the "Hot gas" warning
stays, but its stated reason changes.

## Context

The solver evolves the electron temperature with the matter-temperature equation of RECFAST
(Seager et al. 1999), extended by the distortion terms (`update_temperatures` in
`src/solver.rs`):

    dρ_e/dτ = R [(ρ_eq + δρ_inj − H_dcbr) − ρ_e] − H t_C ρ_e.

The free-electron fraction X_e does not follow. `RecombinationHistory::new` integrates the
Peebles (1968) three-level atom once, before the run, and evaluates the case-B recombination
coefficient α_B at the radiation temperature T_z. The solver then reads X_e from that table, and
X_e sets t_C, the Compton optical depth per step, the coupling rate R, and the
bremsstrahlung rate. X_e never sees T_e, but T_e depends on X_e.

With no injection this costs little. During hydrogen recombination Compton scattering holds
T_e to T_z within about 1% (`peebles_rhs` docstring), and α_B changes by 0.7% for a 1%
temperature change. It is wrong when heating below z ≈ 1600 drives ρ_e = T_e/T_z above 1.
Heating q holds ρ_e − 1 ≈ q t_C/(4θ_z), and ADR 0007 measured peak ρ_e from 17 to 1600 for
bursts at z_h = 1000 and 64 to 200 at z_h = 600 to 400. The Péquignot et al. (1991) fit in
`alpha_recomb` gives, for z from 1500 to 600:

| ρ_e | α_B(ρ_e T_z) / α_B(T_z) |
|---|---|
| 2 | 0.58–0.60 |
| 10 | 0.14–0.16 |
| 100 | 0.014–0.019 |

Hot gas therefore recombines more slowly than the table assumes. It stays more ionized, couples
to the photons faster, and delivers its heat sooner. The fixed table gets the direction of this
feedback wrong for every heated run below z ≈ 1600. ADR 0007 answered the problem only with a
warning at ρ_e > 10 for z < 1500.

We have not measured the size of the effect on μ, y, or heat delivery. Measuring it is part of
the verification below.

## Decision

Evolve the hydrogen ionization fraction X_H as solver state, coupled to T_e, with the same
Peebles three-level atom and the same α_B fit, fudge factor, and Lyman-α escape as today.

- Evaluate α_B at T_m = ρ_e T_z. Keep the photoionization rate β_B and the Saha fraction X_S at
  T_z, since the CMB does the photoionizing.
- Keep the Saha-subtracted form and add the temperature dependence as a separate term:

      dX_H/dz_up = C n_H / [H(1+z)] · { α_B(T_z) [X_H² − X_S² (1 − X_H)/(1 − X_S)]
                                        + [α_B(T_m) − α_B(T_z)] X_H² }.

  The second term is exactly zero at ρ_e = 1, so a run whose gas stays at T_z reproduces the
  current table bit for bit, and the near-equilibrium cancellation stays under control.
- Couple X_H and ρ_e by a lagged split. Each step advances X_H across the step with ρ_e from the
  start of the step, sub-cycling the existing Heun step at dz ≤ 0.5. The midpoint X_e then feeds
  that step's T_e and Δn solve. The coupling is first order in the step. If the
  timestep-convergence check shows the split error is not negligible, we add one corrector pass
  with the step-averaged ρ_e.
- Above z_switch ≈ 1575, X_H is Saha at T_z, as today. Runs that end above z_switch are
  unchanged.
- Initialize X_H at z_start from the standard table, which assumes ρ_e = 1 before the run.
- Make this the default. The solver mode flag `fixed_ionization` (builder
  `.fixed_ionization()`, CLI `--fixed-ionization`) restores the fixed table for comparison. It
  sits beside `disable_dcbr` and the other mode flags, not in `SolverConfig`.
- Add `x_e` to `SolverSnapshot` and its JSON output.

Out of scope: helium stays Saha at T_z. There is no collisional ionization, no recombination
cooling or photoionization heating in the T_e equation, and no ionization by injected energy.
The Green's function, `ionization_fraction`, the dark-photon resonance redshift
(ADR 0008), and the Python port keep the standard history.

### Alternatives considered

- **Keep the fixed table and the warning (ADR 0007).** No code change, but the heated runs the
  warning flags stay wrong, in a known direction.
- **Solve (X_H, ρ_e, Δn) jointly in the Newton iteration.** Removes the splitting error, but
  adds a row and column to the bordered system in `kompaneets.rs`, the most delicate code in the
  repo, for a quantity that changes on a Hubble time.
- **A multilevel recombination code (HyRec, CosmoRec).** More accurate for the standard history,
  but EB ruled it out: the three-level atom is enough for this purpose.

## Consequences

- Heated runs below z ≈ 1600 keep X_e higher and deliver heat to the photons faster. The size
  of the change is unmeasured.
- Runs with no injection change little above z ≈ 500 and more below it, where T_m falls
  below T_z. The residual
  X_e should move toward RECFAST, which also evolves T_m. The existing
  `test_xe_vs_recfast_milestones` checks this.
- The targets in `tests/heat_delivery.rs` (0.999862 and 0.99417) were computed with the fixed
  X_e and must be recomputed. The independent oracle in `dev/scripts/heatloss/` must integrate
  the coupled X_H and T_m equations instead of reading X_e from the solver.
- Tests with z_end below about 1575 may shift. Each changed target needs an independent reason.
- The Green's function and the PDE now use different X_e histories for heated runs below
  z ≈ 1600. For small injections the difference is second order in the injection.
- Collisional ionization is still missing. Near z ≈ 850 it sets in at T_e ≈ 1.4e4 to 2.3e4 K
  (ρ_e ≈ 6 to 10, ADR 0007), so the three-level atom under-ionizes hotter gas. The "Hot gas"
  warning stays, with text that names collisional ionization, not the fixed history.
- Narrow bursts reach T_e ~ 1e6 K. We have not confirmed the temperature range over which the
  Péquignot et al. (1991) fit holds; this must be checked before quoting results there.
- Step cost rises by a sub-cycled scalar ODE below z_switch, at most about 100 Heun steps per
  PDE step. This is small next to the Newton solve.

## Addendum, 2026-09-25: measured after implementation

- **Step cap.** The review found that the lagged split was not the main error. The backward-Euler
  T_e step is first order, and at the old 0.05 z step cap (61 steps from z = 3000 to 200) it put
  ρ_e 2–8% off at z = 1000 to 800 and 25% off at z = 200, in fixed mode as in coupled mode. Once
  X_H follows T_e, X_e inherits that error: a decay heating through recombination came out 3–7%
  low. A trial T_e pass inside the step did not help, which located the error in ρ_e. Below
  z_switch, coupled mode now caps the step at `XE_COUPLED_DZ_FRAC` = 0.005 z. The decay's X_e is
  then within 0.8% of Δz = 0.25, and a production-grid run to z = 200 takes about 0.5 s longer
  (1.5 to 2.1 s). Fixed mode keeps the old steps, so it still reproduces runs made before this ADR.
- **Corrector.** After each step the X_H advance is redone with the step-averaged ρ_e. At the old
  steps this halved the X_e error of a burst at z_h = 600 (2.4% to 1.4%).
- **No injection.** Against HyRec-2, X_e is off by −0.23%, −0.21% and +0.08% at z = 200, 100 and
  50, where the fixed history is off by 1.9%, 5.2% and 10.2%. T_e matches HyRec's T_m to 0.05%,
  0.10% and 0.20%.
- **Heated runs** (default grid, to z = 200), delivered fraction, coupled against fixed:
  0.99980 against 0.99973 for (z_h, Δρ/ρ) = (1000, 1e-5), 0.9965 against 0.9903 for (600, 1e-6),
  and 0.9495 against 0.9031 for (400, 1e-6). X_e(200) reaches 3 times the standard value.
  An independent integration of the coupled X_H and T_m equations
  (`dev/scripts/heatloss/coupled_xe_expectation.py`) matches X_e(200) at z_h = 600 to 0.07%, and
  the coupling's change in delivered heat to 0.4%. The oracle's `--table-xe` control matches the
  solver's fixed mode; its `--fixed-xe` control does not, and the first version of the test
  compared against it (a 0.65% definition mismatch).
- **Linear regime.** The coupling is first order even as Δρ/ρ → 0: heating changes X_e, and X_e
  changes the heat flow of the no-injection run. The Consequences bullet calling the
  Green's-function/PDE difference second order is therefore wrong; it is first order, and a
  no-injection run's X_e already differs from the standard table by 2–10% below z = 200. The
  `heat_delivery.rs` burst target moves from 0.999862 to 0.999855 (the solver gives 0.999840).
  That shift is a quarter of the test's tolerance, so the test pins ADR 0004, not this ADR.
  Consequence 3 above overstated the fallout: no target left its tolerance.
- **Hot-gas warning.** With X_e coupled, a 1e-5 burst at z_h = 1000 peaks at ρ_e ≈ 6 (oracle)
  instead of 17 and no longer warns. `coverage_gaps::narrow_bursts_deliver_their_heat` now
  exercises the warning with 1e-3.
- **Collisional ionization.** The oracle estimates that collisional ionization runs 3e2 to 2e7
  times faster than the Hubble rate in the hot cases above. Both codes omit it, so they agree with
  each other, not with real gas. The "Hot gas" warning is the only guard.
- **α_B fit range.** Unresolved: we found no primary statement of the hydrogen range of the
  Péquignot et al. (1991) fit. The gas leaves the ~1e4 K regime only above ρ_e ≈ 10 near
  z = 850, where the "Hot gas" warning already fires.
- **Bit-for-bit claim.** It holds above z_switch and in fixed mode. Below z_switch at ρ_e = 1 the
  evolved X_H steps on different nodes from the table and agrees to 1e-4.
- **Below z ≈ 21** the solver stops updating ρ_e (θ_z < 1e-8), so X_H advances with a frozen gas
  temperature. X_e has frozen out by then; this existed before and is noted only because X_e now
  reads ρ_e.
- **Flag location.** The fixed-history switch is a solver mode flag, not a `SolverConfig` field
  (edited in the Decision above on the day of acceptance).
