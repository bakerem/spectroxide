# Set the photon-injection Green's function y-era limit to z = 10⁴

## Status

Accepted, 2026-09-24.

## Context

The photon-injection Green's function (GF) has two branches: a μ-era form, valid for
z_h ≥ 2×10⁵, and a y-era form. The y-era branch ends at z_h = 5×10⁴, and the GF raises an error
inside the gap between the two branches. The same 5×10⁴ also sets where the survival probability
P_s switches from the numerical free-free depth exp(−τ_ff) to the analytic exp(−x_c/x). The value
appears in four places:

- `src/greens.rs`: `PHOTON_GF_Y_ERA_Z_MAX = 5.0e4`, used by `assert_photon_gf_regime` and
  `in_photon_gf_transition_band`.
- `src/greens.rs`: `photon_survival_probability_numerical` branches on a literal `z_h > 5.0e4`.
- `python/spectroxide/_validation.py`: `PHOTON_GF_Y_ERA_Z_MAX = 5.0e4`, used by
  `greens_function_photon` and the history integrator in `greens.py`.
- `python/spectroxide/greens.py`: `photon_survival_probability_numerical` and
  `mu_from_photon_injection` branch on a literal `5.0e4`.

5×10⁴ is the middle of the μ–y transition, not the end of the y-era. With the Table 1 fit
(z_y = 6.0×10⁴, α_y = 2.58), J_y(5×10⁴) = 0.62: 38% of the injected energy no longer ends up as y.
J_y(10⁴) = 0.990. The equation label of the y-era photon GF in the arXiv source of Chluba (2015)
names the range 10³–10⁴ (not checked against the published text). The paper's own Fig. 2 notebook measured 4y/Δρ = 0.64 at z_h = 5×10⁴. The
paper now defines the y-era as z ≲ 10⁴, as the literature usually does (EB, 2026-09-24).

## Decision

Set the y-era limit to 10⁴ in Rust and Python, and use the one named constant for both the GF
validity window and the P_s branch switch. The GF then raises for 10⁴ < z_h < 2×10⁵, and P_s uses
exp(−τ_ff) only for z_h ≤ 10⁴.

The default number-conserving threshold `nc_z_min = 5e4` stays unchanged. It marks where DC/BR
photon-number violation matters for the PDE, not an era boundary. Changing it would move every
PDE result.

## Alternatives considered

- **Keep 5×10⁴ in the code and change only the paper text.** The code and paper would then
  disagree on where the photon GF is valid, and the code would keep accepting z_h where
  J_y = 0.62.
- **Warn instead of raise for 10⁴ < z_h ≤ 5×10⁴.** Existing callers would keep working, but the
  GF would still return values the paper says it cannot compute accurately.

## Consequences

- Calls with 10⁴ < z_h ≤ 5×10⁴ that used to succeed now raise. Known callers:
  - `python/tests/test_greens.py:538`, which asserts that 5×10⁴ is a valid boundary.
  - `src/greens.rs` test at z = 4.9×10⁴, which compares the two P_s branches at the switch;
    it moves to just below 10⁴.
  - `notebooks/physics/photon_injection.ipynb` cell 16 (z_h = 3×10⁴).
  - `notebooks/paper_figures/dark_photon_constraints.ipynb` cell 11. Its y-era GF branch covers
    3×10³ < z_res ≤ 3.2×10⁴ (m ≤ 10⁻⁷ eV), so the GF curve in Fig. 8 loses the masses with
    z_res in (10⁴, 3.2×10⁴]. Fig. 8 is not regenerated (EB, 2026-09-24), but the notebook's
    branch bound must drop to 10⁴ so that it still runs.
  - `notebooks/physics/photon_injection_validation.ipynb` cell 21, whose redshift grid runs
    from 3×10³ to 2×10⁶.
- P_s for direct callers with 10⁴ < z_h ≤ 5×10⁴ changes from exp(−τ_ff) to exp(−x_c/x).
  Neither is accurate in that range (paper Sec. 5), and the GF no longer uses either there.
- The history integrator in `greens.py` drops more samples in the transition band and warns.
- Paper Figs. 6 and 7 are unaffected: they use the GF at z_h = 10⁴ and z_h ≥ 3×10⁵.
- Rust and Python stay in parity. `test_parity.py` must still pass.

## Addendum 2026-09-24: μ-era limit at 3×10⁵

EB extended the decision the same day: the μ-era starts at z = 3×10⁵, in the paper, the figures
and the code. `PHOTON_GF_MU_ERA_Z_MIN` moves from 2×10⁵ to 3×10⁵ in `src/greens.rs` and
`python/spectroxide/_validation.py`, so the photon GF raises for 10⁴ < z_h < 3×10⁵. This is a
convention choice (EB, 2026-09-24), not a change forced by the energy branching: J_μ is already
0.99996 at 2×10⁵ in the Chluba (2013) fit.

- Unit tests that called the photon GF at z_h = 2×10⁵ move to 3×10⁵ (the band edge, which
  stays valid), and the history-integration test moves its burst to 3.5×10⁵ so that it lies
  inside the integration window.
- Paper Figs. 6 and 7 are unaffected: their smallest μ-era GF redshift is 3×10⁵.
- `notebooks/physics/photon_injection.ipynb` calls the photon GF at 2×10⁵ in several cells and at
  3×10⁴ in one. It raises if rerun, and needs rework before its next execution.
- The heat-GF transition warnings in `_validation.py` (3×10⁴ to 2×10⁵) are separate heuristics
  and are unchanged.

## Addendum 2026-09-24: the P_s branch switch stays at 5×10⁴

The Decision above tied the P_s branch switch to the GF's y-era limit. Implementing it showed
that at 10⁴ the analytic exp(−x_c/x) over-absorbs badly: it extrapolates the μ-era x_c fit
well outside its range. `_photon_survival_probability_numerical` with `DEFAULT_COSMO`:

| z | x | P_s from τ_ff | P_s analytic |
|---|---|---|---|
| 10⁴ | 0.03 | 0.985 | 0.236 |
| 10⁴ | 0.1 | 0.999 | 0.649 |
| 5×10⁴ | 0.03 | 0.938 | 0.612 |
| 5×10⁴ | 0.1 | 0.995 | 0.863 |

At 10⁴ the τ_ff form is the valid one (it is the paper's y-era form). EB decided (2026-09-24)
to decouple the two: the switch stays at 5×10⁴ under its own constant `P_S_TAU_FF_Z_MAX`
(Rust `src/greens.rs`, Python `greens.py`). The photon GF still rejects 10⁴ < z_h < 3×10⁵, so
the switch affects only direct callers of the survival probability inside that band. The
branch-comparison test and the parity grid keep their original redshifts.
