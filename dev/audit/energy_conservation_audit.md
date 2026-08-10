# Energy-conservation audit — where the Δρ/ρ deviation actually comes from

**Date:** 2026-07-27; §6 addendum 2026-08-05. **Harness:** `examples/energy_budget.rs`
(`cargo run --release --example energy_budget [all|quad|heat|photon|pb2009|figure|joint|deepmu|steps]`).

## Question

The suite's energy-conservation tests report deviations of 0.08–0.43% (heat) and
0.65–1.09% (photon injection) against tolerances of 2%, 1.5% and 3%. Is that a
residual bug in the energy bookkeeping, or a controlled discretization error?

## Answer

Two distinct causes, neither of them a bookkeeping bug.

1. **Photon injection: the test target is wrong**, by an amount that is exactly
   calculable. It accounts for ~0.75 of the ~0.9 percentage points reported.
   True conservation is ≤0.35%.
2. **Heat injection: the first-order-in-Δτ temporal error** of the coupled
   T_e / DC-BR step, i.e. the error already carried in
   `dev/output/error_budget.md` as *"Temporal (μ, production defaults,
   dtau_max=10) = 2.86e-3, Richardson p = 1.00"*. It is generated inside the
   injection window, is linear in the injected energy, and is controlled by
   `dtau_max` alone — the frequency grid does not move it.

## 1. The bookkeeping itself is clean

Δρ/ρ is `∫x³Δn dx / G₃` with the analytic `G3_PLANCK = π⁴/15` in the
denominator and a discrete quadrature in the numerator. That mismatch is
bounded by integrating the exact shapes, ∫x³G_bb dx = 4G₃ and
∫x³M dx = (κ_c/3)G₃, on the solver grid (`energy_budget quad`):

| grid | x³G_bb, trapz(x³Δn) | x³G_bb, midpoint-x | x³M, trapz | x³M, midpoint-x |
|---|---|---|---|---|
| default, N=2000 | −6.1e−6 | +1.1e−4 | −4.5e−6 | −7.6e−5 |
| production, N=4000 | −1.8e−6 | +3.8e−5 | −2.8e−6 | −5.2e−6 |

Two conventions are in use: `src/spectrum.rs::weighted_integral` evaluates
`x_mid³ · Δn_mid · dx`, while the test helpers trapezoid the product `x³Δn`.
Both are second order; the product form is ~18× more accurate on the default
grid. Either way the contribution is ≲10⁻⁴ — two to three orders below the
deviations under investigation. **Truncation at x_min/x_max and the G₃
normalisation are not the story.**

## 2. Photon injection — the target omits the IC's second moment

`setup_photon_injection` installs a Gaussian Δn with amplitude
A = (ΔN/N)·G₂/(x₀²·σ·√(2π)) and
`test_photon_injection_energy_conservation_tight` compares the final Δρ/ρ to
α_ρ x₀ (ΔN/N). The exact moments of that Gaussian are

    ∫x²Δn dx = A σ√(2π)(x₀² + σ²)     ⇒  ΔN/N = (ΔN/N)_nom (1 + σ²/x₀²)
    ∫x³Δn dx = A σ√(2π)(x₀³ + 3x₀σ²)  ⇒  Δρ/ρ = α_ρ x₀ (ΔN/N)_nom (1 + 3σ²/x₀²)

because the amplitude is normalised with `x₀²` instead of the exact second
moment. The test uses σ = max(0.05 x₀, 0.05), so σ/x₀ = 0.05 at every point it
samples and the target is low by 3(0.05)² = **0.750%** — independent of x₀,
which is why the reported errors cluster near 0.75% and why refining the grid
does not help. Measured on the default grid (`energy_budget photon`), the IC's
energy is +0.749…+0.752% above the target against the analytic +0.750%.

Separating the IC's content from the evolution (final Δρ/ρ ÷ the IC's Δρ/ρ
measured on the same grid):

| x_inj | reported err (old target) | IC vs target | **true conservation** |
|---|---|---|---|
| 1.5 | +0.845% | +0.749% | **+0.095%** |
| 3.6 | +0.649% | +0.752% | **−0.102%** |
| 5.0 | +0.675% | +0.750% | **−0.074%** |
| 8.0 | +0.828% | +0.750% | **+0.077%** |
| 12.0 | +1.086% | +0.750% | **+0.334%** |

x_inj = 12 is the outlier and it is temporal, not spatial: +0.334% at
dtau_max = 10 → +0.093% at 1 → +0.071% at 0.2, while N = 2000 → 4000 only moves
it to +0.298%.

**This is a test-only defect — the production scenario is exact.**
`InjectionScenario::MonochromaticPhotonInjection` builds its source as
G₂·gauss(x)/x², not (G₂/x₀²)·gauss(x), so

    ∫x²Δn dx = G₂ ∫gauss dx = G₂ (ΔN/N)        (exact)
    ∫x³Δn dx = G₂ (ΔN/N) ∫x gauss dx = G₂ (ΔN/N) x₀   (exact)

and α_ρ x₀ ΔN/N *is* the correct target for it, with no σ-dependent
correction. Only the test helper `setup_photon_injection`, which installs the
Gaussian directly into Δn with an x₀²-based amplitude, carries the bias. The
paper figure (`notebooks/paper_figures/energy_conservation.ipynb`), which uses
the scenario, is unaffected.

## 3. Heat injection — first-order temporal error in the T_e / DC-BR coupling

`test_heat_energy_conservation_sweep_tight` at the shipped defaults
(N = 2000, dtau_max = 10, dy_max = 0.02):

| z_h | 3e3 | 5e3 | 1e4 | 5e4 | 1e5 | 2e5 | 5e5 |
|---|---|---|---|---|---|---|---|
| err | −0.09% | −0.08% | −0.08% | −0.17% | −0.20% | −0.27% | −0.43% |

One-sided and growing with z_h. Decomposition (`energy_budget heat`; `err_net`
subtracts an identical zero-injection run, whose deficit is the physical
adiabatic-cooling distortion, −1.5×10⁻⁹ to −2.9×10⁻⁹ ≈ 0.015–0.03% of the
injected 10⁻⁵):

| config | z_h = 1e4 | z_h = 1e5 | z_h = 5e5 |
|---|---|---|---|
| dtau_max = 10 (default) | −0.061% | −0.176% | −0.394% |
| dtau_max = 10, N = 4000 | −0.061% | −0.182% | −0.524% |
| dtau_max = 10, dy_max = 0.005 | −0.061% | −0.176% | −0.394% |
| dtau_max = 10, DC/BR off | −0.001% | −0.070% | −0.098% |
| dtau_max = 2 | −0.061% | −0.026% | +0.083% |
| dtau_max = 1 | −0.029% | −0.007% | +0.142% |
| dtau_max = 0.2 | −0.005% | — | — |

Established properties:

- **Temporal, not spatial.** dy_max = 0.02 → 0.005 changes nothing to all
  printed digits at every z_h, and N = 2000 → 4000 changes nothing at z_h = 1e4
  and only −0.006 pp at 1e5; `dtau_max` is the knob that moves it. On the
  P&B-2009 scenario the finer grid is marginally *worse*
  (−0.284% → −0.306% at dtau_max = 10), which is why that test's two-grid
  structure could never have detected anything.
- **Deep μ-era caveat.** At z_h = 5e5 a second residual of comparable size
  appears and the two do not have the same sign: refining dtau_max crosses zero
  (−0.394% → +0.083% → +0.142%) while refining the grid at fixed dtau_max moves
  the other way (−0.394% → −0.524%). So the shipped defaults sit near a partial
  cancellation, and |error| still *improves* with dtau_max = 2 (0.394% →
  0.083%), but the deep-μ end is not converged in a single knob. μ itself is
  much better behaved there (dtau_max 2 → 1 moves it 0.06%).
- **DC/BR is the larger of two channels.** Disabling DC/BR removes ~60% of the
  deficit at z_h = 1e5 (−0.176% → −0.070%), 75% at 5e5 (→ −0.098%) and
  essentially all of it at z_h = 1e4. The remainder is the Kompaneets/T_e
  coupling. (DC/BR off is a diagnostic only — it changes μ by +2.7% at
  z_h = 5e5.)
- **`cn_dcbr = true` changes nothing** (identical to 5 significant figures at
  every z_h tested). The first-order term therefore does *not* live in the
  DC/BR photon-side operator, contrary to what the "coupled DC/BR backward
  Euler = 1.0 (time)" row of `error_budget.md` suggests. The candidate that
  survives is the electron-side/photon-side time-level and temperature
  mismatch: `update_temperatures` evaluates the DC/BR heating integral at
  ρ_e(old) including the injection boost, while the photon-side rates are built
  from `rho_eq_dcbr` with the injection contribution subtracted
  (`src/solver.rs:1182–1193`). Confirming that specific line is open work — the
  numerics below stand regardless of which of the two coupled terms dominates.
- **Generated inside the injection window** (`energy_budget steps`). Running
  Δρ/ρ tracks the analytically integrated Gaussian source to <10⁻⁴ before the
  burst; the residual is −0.249% immediately after the burst at dtau_max = 10
  (−0.027% at dtau_max = 2), then relaxes to −0.176% (−0.026%) by z = 500. The
  larger transient dip mid-burst (−1.0% at dtau_max = 10) is the half-step
  offset between the source evaluated at z_mid and the running total compared
  at step boundaries; it scales like dtau and is not a loss.
- **Linear in the injected energy.** At z_h = 1e5, dtau_max = 10, the relative
  deficit is −0.177%, −0.176%, −0.156%, +0.035% for Δρ/ρ = 1e−6, 1e−5, 1e−4,
  1e−3 — flat through 10⁻⁴, so it is a linear-response error, not amplitude
  nonlinearity (which only appears at 10⁻³). Roughly independent of burst width
  for σ_z/z_h ≥ 0.01.

## 4. Consequence for published numbers

The same first-order term biases μ and y, and at high z_h it is larger than the
grid systematic quantified in T-PC-8 (N = 4000 → 4400 moves μ by +0.54%):

| z_h | μ(dtau_max=10) vs dtau_max=1 | y |
|---|---|---|
| 1e4 | −0.08% | −0.03% |
| 1e5 | −0.22% | −0.13% |
| 5e5 | −0.54% | −0.46% |

(dtau_max = 2 is already within 0.03% / 0.06% of the dtau_max = 1 value at
z_h = 1e5 / 5e5, so the table's "converged" column is not extrapolation-limited.)

`error_budget.md` already carries 2.9×10⁻³ for μ at dtau_max = 10, measured at
z_h = 2×10⁵. That is the right order but it is quoted as a single number; the
term grows with z_h and reaches 5.4×10⁻³ by z_h = 5×10⁵. Cost of removing it:
dtau_max = 2 is ~5× the steps (15k → 77k at z_h = 5e5) and brings μ inside
0.06% and |energy| inside 0.09%.

`cn_dcbr = true` is *not* a shortcut here — it costs nothing and buys nothing
(see above), so the only lever is `dtau_max`.

**Recommendation:** keep dtau_max = 10 as the library default (the 0.2–0.5%
level sits below the μ/y error budget the paper quotes), but state the z_h
dependence in the budget rather than a single value.

The paper figure already runs at `dtau_max=3`
(`notebooks/paper_figures/energy_conservation.ipynb`, both panels, with
σ_z = 0.04 z_h, z_start = z_h + 7σ_z and N = 8000 from the `sweep` path), so what
it plots is mostly-but-not-fully converged. Measured at those settings but
N = 4000 (`energy_budget figure`):

| z_h | dev at dtau_max = 3 | dev at dtau_max = 1.5 | μ shift |
|---|---|---|---|
| 5e5 | −0.148% | −0.058% | +0.09% |
| 1e6 | −0.122% | +0.039% | +0.16% |
| 3e6 | +0.136% | **+0.323%** | −0.36% |

So the figure's plotted deviations are ≤0.15% everywhere, better than the tests'
configuration because the burst is wide (σ_z = 0.04 z_h) and dtau_max = 3.

**Open item — RESOLVED 2026-08-05, see §6.** At z_h = 3e6 refining dtau_max makes the deviation *worse*
(+0.136% → +0.323%, 266k → 532k steps), i.e. the deep-μ residual that survives
Δτ → 0 is positive and of order a few tenths of a percent, and the shipped
dtau_max merely cancels part of it. The same crossover is visible at z_h = 5e5
in the σ_z = 0.01 z_h series above (−0.394% → +0.083% → +0.142% for
dtau_max = 10 → 2 → 1). Whatever this term is, it is not the x-grid resolution
(N = 2000 → 4000 moves it by ~0.1 pp at most), not `dy_max`, and not the
analytic-vs-discrete energy of the number-conserving T-shift (that mismatch is
1.8×10⁻⁶ relative, four orders too small). Identifying it is the natural next
step if sub-0.1% energy closure is wanted in the deep μ-era; nothing in the
current suite is sensitive to it, since every energy test sits at z_h ≤ 5e5.

`dev/scripts/photon_energy_conservation.py` is the **stale** copy — it predates
the notebook, omits `dtau_max`, and uses different z grids; the notebook is the
figure's source of truth.

## 5. Test changes made

| test | before | after |
|---|---|---|
| `test_photon_injection_energy_conservation_tight` | final Δρ/ρ vs α_ρ x₀ ΔN/N, tol 3% | two assertions: IC energy vs the analytic (1+3σ²/x₀²) target to 5×10⁻⁴, then final vs IC to 0.5% |
| `test_pb2009_energy_conservation` | default grid 1.5% / production grid 0.5% | default grid at dtau_max = 10 (0.5%) and dtau_max = 2 (0.15%) — pins the convergence axis that actually controls the error |
| `test_heat_energy_conservation_sweep_tight` | 2% at 7 redshifts | 0.6%, with the measured z_h trend and its mechanism in the docstring |

## 6. Addendum (2026-08-05): the joint refinement limit closes the open items

The "positive residual that survives Δτ→0" reported above (heat z_h = 5e5:
+0.142%; photon x_inj = 12: +0.071%; deep-μ z_h = 3e6: +0.323%) was an
artifact of refining one knob at a time. The dtau series were run at N = 2000
(N = 4000 for the figure settings) and the N series at dtau_max = 10 — the
joint limit was never taken, and the temporal and spatial terms have opposite
signs, so each one-knob limit stalls at the other term's value. Taking the
joint limit (`energy_budget joint` and `deepmu`):

Heat, z_h = 5e5, σ_z = 0.01 z_h, err_net (baseline-subtracted):

| | dtau_max = 10 | 2 | 1 |
|---|---|---|---|
| N = 2000 | −0.394% | +0.083% | +0.142% |
| N = 4000 | −0.524% | −0.048% | **+0.012%** |
| N = 8000 | | | **−0.021%** |

Photon Gaussian IC, x_inj = 12 (out/IC−1): N = 2000/4000/8000 at
dtau_max = 0.2 give +0.071% / +0.035% / **+0.026%**, both knobs still
shrinking it monotonically.

Deep-μ figure settings (z_h = 3e6, σ_z = 0.04 z_h): N = 4000, dtau_max = 1.5
gives +0.323%; **N = 8000, dtau_max = 1.5 gives −0.012%** (N = 8000,
dtau_max = 3: −0.199%).

**Conclusion: there is no formulation-level energy leak.** Every deviation in
the suite decomposes into (a) the first-order-in-Δτ temporal term (negative,
controlled by `dtau_max`), and (b) a spatial term of opposite sign controlled
by `n_points`, larger at higher z_h (thermalization pushes the action to the
low-x DC/BR region). Joint closure is ≤0.03% at every point tested, including
the deep μ-era. Consistent with this, the bordered-Newton coupling is
discretely energy-consistent by construction: the electron-side
H_dcbr = h_norm·Σ wem_j (neq_j − Δn_j) in the ρ_e residual
(`src/kompaneets.rs`, ~line 836) is built from the same `wem`/`neq` as the
photon-side DC/BR update, so the exchange cancels identically at the converged
iterate; the half-grid `dcbr_heating_with_derivative` only feeds the
predictor.

Side observation: μ at z_h = 3e6 moves −1.1% from N = 4000 → 8000
(7.635e−7 → 7.551e−7 at dtau_max = 1.5) — the deep-μ grid systematic on μ is
an order of magnitude larger than at z_h ≤ 5e5 (T-PC-8). Relevant if deep-μ
μ values are ever quoted at sub-percent precision.
