# Replace the electron-temperature caps with a sanity guard

## Status

Accepted, 2026-09-24 (EB).

Amends the cap paragraph of the Addendum to
[ADR 0002](0002-relax-dc-br-toward-actual-electron-temperature.md).

## Context

The solver clamps ρ_e = T_e/T_z in three places in `src/solver.rs`:

- the backward-Euler predictor, at 1.5;
- the coupled Newton solve, at 3.0;
- the DC/BR target temperature, at [0.05, 2].

The comment on the predictor cap says it "prevents unphysical overshoot of the perturbative step
in weak-Compton regimes" and that a single range "degrades post-recombination accuracy". Both
claims date from the initial commit, and no record supports them. ADR 0002 later tied a warning
to the caps: reaching them means heating strong enough to change the ionization history, which
the solver holds fixed.

The caps do more than guard. Heat reaches the photons through the electrons, so a heating rate q
holds ρ_e − 1 ≈ q t_C/(4θ_z). A narrow burst drives ρ_e past the cap at any redshift, and every
capped step deletes the heat the electrons could not hold. We measured this on 2026-09-24 in a
scratch copy with the three caps read from the environment. The runs used `single_burst` with the
default width, the default grid, and `z_end = 200`. The linear reference is a run below the cap,
minus a run with no injection, scaled up by the amplitude ratio.

| z_h | Δρ/ρ | Delivered, current caps | Delivered, caps at 1e4 | Linear reference | Peak ρ_e |
|---|---|---|---|---|---|
| 20000 | 1e-2 | 0.916 | 1.006 | 1.000 | 1.7 |
| 5000 | 3e-3 | 0.328 | 1.001 | 1.000 | 5.3 |
| 3000 | 2e-4 | 0.831 | 1.000 | 1.000 | 1.9 |
| 1000 | 1e-5 | 0.313 | 0.9998 | 0.9998 | 17 |
| 1000 | 1e-3 | 0.014 | 1.0003 | 0.9998 | 1600 |
| 600 | 1e-6 | 0.061 | 0.9908 | 0.9908 | 64 |
| 400 | 1e-6 | 0.019 | 0.903 | 0.903 | 200 |

With the caps at 1e4, y/Δρ and Δn in 0.5 < x < 20 match the linear reference to 0.2–0.4%. Below
x = 0.1 the free-free tail differs from it by up to a factor of 2.7. The cause is physical: per
unit (ρ_e − 1), BR emission falls roughly as ρ_e^(−3/2), from θ_e^(−1/2) and the exp(x/ρ_e) − 1
factor. At x = 1e-3 the tail per unit amplitude is 0.43 of the linear value at peak ρ_e = 17 and
0.03 at peak ρ_e = 1600. A cap of 20 still loses heat, because post-recombination runs reach ρ_e = 50 to 1600.

The full release suite with the caps at 1e4 passes 445 tests and fails 6. Five of the six assert
the cap warning or its wording. The sixth,
`pde_heat::test_post_recombination_locked_in_distortion`, asserts that a burst at z_h = 800 puts
less than 10% of its heat into the photons. Without the caps the photons receive 99.9%, and
y = 0.2497 Δρ/ρ. That agrees with `heat_delivery.rs`, which pins 0.999862 at z_h = 1000 against an
independent integration. Nor can the gas keep the heat, since its thermal energy is about 1e-9
of the photons'. The test passes today only because the cap deletes the heat (CLAUDE.md
Pitfall #9).

After recombination the real limit is physical, because the solver holds X_e on the standard
history. At ρ_e = 2 the case-B recombination coefficient in `src/recombination.rs` falls by
about 40%. At z ≈ 850, ρ_e = 6 means T_e ≈ 1.4e4 K, close to the temperature at which
collisional ionization equilibrium leaves hydrogen half ionized. Before z ≈ 1500 hydrogen stays
ionized, and no such limit applies until θ_e nears 0.01.

## Decision

Raise all three upper caps to 1e4, and keep them as a sanity guard, not as physics. Nothing in
the solver overflows at large ρ_e: the ρ_e equation is linear in ρ_e, and the BR and Bose factors
are evaluated exactly. A value past the guard points to a failed solve.
Keep the lower DC/BR bound at 0.05 and the rejection of non-finite ρ_e.

Move the ionization-history warning off the clamp. Warn once per run when ρ_e exceeds 10 at
z < 1500 (`HOT_GAS_RHO_E`, `HOT_GAS_Z_MAX`), whether or not the guard engages. At ρ_e = 10 the
case-B recombination coefficient has fallen by about 85%, and at z ≈ 850 the gas is at
T_e ≈ 2.3e4 K, past the onset of collisional ionization. The warning therefore fires only once
X_e is already badly wrong. A threshold of 2 (a 40% drop) was the alternative. The choice is a
judgment call, not a derived bound.

Delete `test_post_recombination_locked_in_distortion`, since `heat_delivery.rs` already covers
delivery after recombination. Replace the cap-warning tests with a test that narrow bursts
deliver their heat, and move the late-decay test to the new warning. Drop the energy-closure
check's silence rule for capped runs.

## Consequences

- Narrow bursts and late heating deliver their heat to the photons instead of losing 8–99% of
  it. Energy closure holds in every case in the table.
- After recombination the solver now returns clean-looking results for gas hot enough to change
  X_e, where before it returned visibly wrong ones. The new warning is the only signal. In that
  regime the real gas would put part of the heat into recombination lines, not y.
- `test_post_recombination_locked_in_distortion` is deleted, along with three energy-closure
  tests in `coverage_gaps.rs` that asserted cap behavior.
  `coverage_gaps::narrow_bursts_deliver_their_heat` replaces them. It requires delivery
  within 1% for the z_h = 3000 and z_h = 1000 bursts in the table, which delivered 83% and 31%
  under the old caps. `pde_heat::test_decaying_particle_late_heating_warns` now asserts one
  "Hot gas" warning and no guard hit.
- The energy-closure check treats every run alike. A 25% excess from a coarse grid still warns
  (`energy_closure_tabulated_heating` covers that case).
- The x < 0.1 tail is untested in this nonlinear regime. In real gas this hot, collisional
  ionization would raise X_e and the BR rate with it, so the tail is qualitatively wrong once the
  "Hot gas" warning fires.
- The solver does not check θ_e itself. At the guard θ_e reaches 4.6e-3 at z = 1000 and passes
  0.01 above z ≈ 2000, beyond the nonrelativistic Kompaneets equation. No run we know of gets
  there.
- Physics outputs change only for runs that reached a cap. Every run below the caps is
  unchanged, since the clamp never engaged.
