# R3 reference solver — STATUS

**Updated:** 2026-07-27. Workstream R3 of PLAN_VALIDATION_ROUND2.

## Validated (all from runs executed this session)

- Decomposition templates: the contract's section 5 was corrected in place by the
  orchestrator (all three templates now carry the same power of x). The three
  round-trip self-tests (pure ΔT/T, pure y, pure μ synthetic inputs) recover
  their amplitudes to 1e-15 relative with fit residual ~1e-16. **The open issue
  from the previous session is closed: the defect was in the contract text, not
  in the transcription.** The round-trips are permanent self-tests in
  `refsolver.py` (`self_test_roundtrip`) and run on every invocation.
- Discrete moments: G3 of the discrete Planck reproduces π⁴/15 to 1e-13,
  I4/(4·G3) to 3.8e-13 (Simpson in ln x, x∈[1e-4, 40]).
- Chang–Cooper equilibrium: Planck at T_e is an exact fixed point. Relative
  interface flux 8e-14; spurious |dn/dτ|/n = 5e-13; drift over a single
  dτ = 1e4 step = 2.8e-13. Required rewriting the flux in the
  cancellation-free form `F = P·g·(φ/expm1(w))·expm1(Δψ+w)`, `ψ = ln[n/(1+n)]`;
  the naive `g = n/(1+n)` difference loses 11 digits at small x.
- Photon-number conservation under pure Compton: exactly 0 (cell-rule
  quadrature is annihilated by the discrete operator).
- Energy delivery: the heat burst delivers the nominal Δρ/ρ to 0.09%.
  This required defining ρ_eq from the *flux-consistent* discrete moments
  (`flux_moments`) rather than Simpson, plus a step-size cap θ_γ·dτ ≤ 0.05.
- Physics cross-checks against independent analytic expectations:
  heat_z5e3 → y = 2.529e-4 vs Δρ/(4ρ) = 2.5e-4 (+1.2%);
  heat_z2e6 → μ = 4.96e-4 vs 1.401·Δρ/ρ·exp[−(z/1.98e6)^2.5] = 5.02e-4 (−1.2%);
  adiabatic → μ = −2.25e-9 (literature ≈ −3e-9), correct sign.

- Analytic adiabatic-cooling cross-check: integrating the quasi-stationary
  identity d(Δρ/ρ)/dlnz = −3ζ(3)/G₃ · N_tot/N_γ over the frozen history gives
  −4.913e-9 for z = 3e6 → 200; the solver measures −4.854e-9 (1.2%).
- Template moment identities verified by quadrature: G_bb → (4, 3),
  Y_SZ → (4, 0), M → (0.713951, 0) for (Δρ/ρ, ΔN/N) per unit amplitude, so
  1/0.713951 = 1.4006 reproduces the μ = 1.401 Δρ/ρ constant from the templates.
- DC/BR crossover K_DC = K_BR at z ≈ 2.9e5 (x = 0.1) to 4.2e5 (x = 1e-3),
  consistent with the literature z_dc,br ~ few × 10⁵ — an independent check that
  the transcribed BR prefactor has the right magnitude.
- Implied blackbody-visibility redshift: heat_z2e6 gives
  J_bb = μ/(1.4006 Δρ/ρ) = 0.35401, i.e. z_μ = 2e6/(−ln J_bb)^{2/5} = 1.970e6
  vs the literature 1.98e6 (0.5%).
- Photon case vs the analytic BE limit built from the *measured* (ΔN/N, Δρ/ρ):
  μ_analytic = 1.4006 Δρ/ρ − 1.8674 ΔN/N = −1.7517e-3 vs solver −1.7172e-3
  (2.0%); ΔT/T = ΔN/(3N) = 3.220e-4 vs solver 3.582e-4 (11%, accounted for by
  the residual y = 5.2e-5 from incomplete Comptonisation at z = 3e5, y_tot = 4.3).

## Finding: the fit weighting is a convention, and it dominates the
## subdominant components

The contract's "uniform weights on the x∈[0.5,18] grid" is grid-dependent. On a
grid uniform in ln x it is effectively w ∝ 1/x. Switching to cell-width weights
moves the DOMINANT component by ≤1.3% in every case (μ for the two μ-era cases,
y for the y-era case, μ for the photon case), but moves the SUBDOMINANT ones by
30–60% (e.g. y for heat_z2e5: 1.004e-4 → 6.50e-5). A grid-free resampled fit
(1001 points linear on [0.5,18], interpolated from the shipped CSVs) is also
reported. Cross-code comparison of subdominant components is therefore only
meaningful once both sides agree on the weighting.

## Known limitation (finding, report to orchestrator)

Below z ≈ 20 the electrons have thermally decoupled (ρ_e = T_e/T_γ < 0.17,
φ > 6) and the CS2012 DC absorption rate Γ_DC = (K_DC/x³)(e^{φx}−1) becomes
spuriously enormous — Γ_DC·dτ > 1 for x ≳ 1 — because the DC Gaunt factor is
derived assuming a blackbody ambient field at T_e ≈ T_γ. The physical DC rate
there is negligible (K/x³·dτ ≲ 1e-15). DC is therefore gated off for φ > 2
(equivalently ρ_e < 0.5, z ≲ 70). BR needs no gate: its factor
e^{−φx}(e^{φx}−1) = 1−e^{−φx} ≤ 1 is bounded. **The contract's z_end = 200 is
unaffected** (ρ_e(200) ≈ 0.87, φ ≈ 1.15); only the z_end = 1 diagnostic needs
the gate.

## Second finding: Σ Z_i²N_i reconstruction from x_e

The table's `x_e` saturates at 1+2f_He = 1.1579 (He⁺⁺) at high z but falls to
1+f_He = 1.0789 (He⁺) by z ≈ 5000. A naive "He always He⁺⁺" mapping
Σ Z²N = N_H(x_e + 2f_He) therefore overestimates the BR ion weighting by 15%
for 1500 < z < 5500 and 28% at z ≈ 1300. Replaced by the recombination ladder
Σ Z²N/N_H = 3x_e − 2 − 2f_He for x_e > 1+f_He, else x_e. Effect on the reported
scalars is measured by comparing against the pre-fix baseline run (see the
matrix log); it is confined to the low-x spectrum shape.

## Not yet done

- Production run matrix (baseline / grid×2 / step×2 / Δρ/ρ=1e-5 / z_end=1 /
  BR off) — launched, see `outputs/results*.json`.
- `README.md` with the convergence numbers.
- Spectrum CSVs at production resolution.

## Next action

Collect the run matrix, tabulate convergence, write README.md.

## Isolation

No spectroxide solver source, test, notebook or audit file was read. Physics
taken from `contract.md` and the raw arXiv LaTeX of arXiv:1109.6552 (Chluba &
Sunyaev 2012) and arXiv:1911.08861 (Chluba, Ravenni & Bolliet 2020).
