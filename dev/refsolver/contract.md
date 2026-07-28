# Clean-room reference solver — I/O contract & physics specification (R3)

**You are implementing an independent reference solver for CMB spectral
distortions.** This is an N-version cross-check: a second, deliberately
different code, written *only* from the physics equations below, that will be
compared against an existing solver you must NOT look at.

## ISOLATION RULES (absolute — the entire value of this workstream depends on them)

- **Do NOT read** any of: `src/*.rs`, `python/spectroxide/greens.py`,
  `python/spectroxide/solver.py`, or any other spectroxide solver source. Do
  not read the existing `tests/`, `dev/audit/*`, or notebooks either.
- Allowed inputs: **this contract**, the frozen ingredient table
  `dev/refsolver/inputs/history.csv`, standard scientific Python
  (numpy/scipy/matplotlib), and — if you want to double-check an equation — the
  raw arXiv LaTeX sources of the two primary papers cited below (fetch
  `arxiv.org/e-print/<id>`, NOT ar5iv/HTML).
- If something disagrees with the (hidden) reference later, you debug *your*
  solver against the papers and its own convergence — never by peeking at the
  other code.
- Write everything under `dev/refsolver/` (solver + a `README.md`). Put outputs
  (CSVs, plots) under `dev/refsolver/outputs/` and `dev/output/refsolver/`.

## The physics you implement

Photon occupation number `n(x, z)`, dimensionless frequency `x = hν/(k T_γ)`
where `T_γ = T_cmb (1+z)` is the photon temperature (so x is normalised by the
photon, not electron, temperature). Solve the coupled photon Boltzmann equation
from high z down to z = 0.

### 1. Kompaneets (Compton scattering) operator

Standard Kompaneets equation with electron temperature `T_e` and photon
temperature `T_γ` (Kompaneets 1956; form as in Chluba & Sunyaev 2012,
arXiv:1109.6552):

    ∂n/∂τ |_Compton = (θ_e / x²) · ∂/∂x [ x⁴ ( ∂n/∂x + φ · n(1+n) ) ]

where
- `τ` = Thomson optical depth, `dτ = σ_T n_e c dt` (dimensionless time),
- `θ_e = k T_e / (m_e c²)` (dimensionless electron temperature),
- `φ = T_γ / T_e`.

Equilibrium check (must hold in your discretisation to machine precision as a
unit test): the flux `F = x⁴(∂n/∂x + φ n(1+n))` vanishes for a Planck spectrum
at the electron temperature, `n_eq(x) = 1/(exp(φx) − 1) = 1/(exp(hν/kT_e) − 1)`,
because `∂n_eq/∂x = −φ n_eq(1+n_eq)`.

**Recommended scheme (different by construction from the reference):**
**Chang–Cooper (1970)** finite-volume discretisation of this Fokker–Planck
operator, solving for the FULL occupation `n` (not a distortion `Δn`). Chang–
Cooper is positivity-preserving and reproduces the exact discrete Bose–Einstein
equilibrium — a genuinely different scheme. Because it evolves full n in double
precision it is *less* accurate for tiny (10⁻⁵) distortions; that is expected —
use test amplitudes Δρ/ρ ~ 10⁻³–10⁻² so the signal is comfortably resolvable,
and verify linearity by amplitude-scaling.

Time integration: implicit Euler or TR-BDF2 on the tridiagonal system via
`scipy.linalg.solve_banded`; fixed log-z steps, no adaptivity. Accept slowness
(it runs once).

### 2. Electron temperature (quasi-stationary Compton equilibrium)

Do NOT read history's T_e; compute it self-consistently. The electron
temperature relaxes to the Compton equilibrium set by the photon spectrum
(Chluba & Sunyaev 2012). Implement the quasi-stationary ratio

    ρ_eq = T_e/T_γ = I₄ / (4 G₃),
    I₄ = ∫ x⁴ n(1+n) dx,   G₃ = ∫ x³ n dx,

evaluated on the current full-n spectrum (well-conditioned at the Δρ/ρ~10⁻³
amplitudes used here — verify the conditioning numerically and report it). For
adiabatic cooling include the Hubble cooling term: the electrons also cool as
(1+z)² between scatterings; the net T_e is set by balancing Compton heating
against adiabatic + expansion cooling. State your T_e treatment in the README.
(For the heat-injection burst cases the injected energy raises the effective
equilibrium; see case definitions.)

### 3. Double-Compton (DC) and Bremsstrahlung (BR) emission (photon-number-changing)

These drive n toward the Planck spectrum at T_e and are essential at high z
(thermalisation) and low x. Emission/absorption term:

    ∂n/∂τ |_em = (K_DC + K_BR) / x³ · ( n_eq − n ) · (exp(φx) − 1)
               = (K_DC + K_BR) / x³ · [ 1 − n (exp(φx) − 1) ]

(the two forms are algebraically identical; `n_eq = 1/(exp(φx)−1)`). Handle this
term implicitly (backward Euler) — it is stiff at low x where K/x³ diverges.

**DC coefficient** (Chluba & Sunyaev 2012, their Eq. 13):

    K_DC = (4α / 3π) · θ_γ² · I₄^pl / (1 + 14.16 θ_γ) · H_dc(x)
    θ_γ = k T_γ / (m_e c²),   I₄^pl = 4π⁴/15,   α = fine-structure constant,
    H_dc(x) = exp(−2x) · (1 + (3/2)x + (29/24)x² + (11/16)x³ + (5/12)x⁴).

**BR coefficient** (non-relativistic Gaunt factor; Gaunt from Chluba, Ravenni &
Bolliet 2020, arXiv:1911.08861 — fetch it if you want the exact fit). A standard
form:

    K_BR = A_BR · θ_γ^(−7/2) · Σ_i Z_i² n_i / n_H · (1 − exp(−x)) / x_something · g_ff(x)

BR is subtle. If you cannot pin the exact prefactor from the paper, implement DC
only and **state in the README that BR was omitted and which cases that
affects** (BR matters at z ≲ 10⁵ low-x; for the z_h = 2×10⁶ and 2×10⁵ μ-era
cases DC dominates and BR omission is acceptable — say so). Do NOT invent a
prefactor. A documented omission is far better than a fabricated coefficient
(this is the single most important rule — see the reference-value directive).

### 4. Injection (heating and photon)

- **Heat injection (single burst):** energy Δρ/ρ deposited in a narrow Gaussian
  in z, `dQ/dz ∝ exp[−(z−z_h)²/(2σ_z²)]`, σ_z = 0.04 z_h, normalised so the
  total ∫ = Δρ/ρ. Heating raises the electron temperature (adds to the Compton
  equilibrium): convert the heating rate to a T_e source. **Define Δρ/ρ as the
  instantaneously-integrated fractional energy added to the photon field** —
  state your exact normalisation in the README (this is the #1 source of a fake
  discrepancy; the reference uses total integrated Δρ/ρ over the burst).
- **Adiabatic cooling:** no injection; the electrons cool as (1+z)² and Compton-
  drag continuously extracts a small amount of energy from the photons,
  producing a small negative μ and y. This case tests the T_e coupling sign.
- **Photon injection (one case):** inject photons as a narrow Gaussian in x,
  `ΔN ∝ exp[−(x−x_inj)²/(2σ_x²)]` with **σ_x = 0.05 x_inj**, total number
  ΔN/N = 10⁻³ (pin the shape — a numerical delta is grid-dependent), added to n
  at the injection redshift. Track the photon-number ledger
  `N = ∫ x² n dx` of the injection and the final spectrum.

### 5. Distortion decomposition (SHARED recipe — implement from this text)

At z = 0, decompose the distortion `Δn(x) = n(x) − n_pl(x)` (n_pl = Planck at
the final T_γ) into (μ, y, ΔT/T) by a **joint linear least-squares** fit over
`x ∈ [0.5, 18]` against the three template shapes:

    Δn(x) ≈ (ΔT/T) · G(x) + y · Y_SZ(x) + μ · M(x)

with (G_bb(x) ≡ x eˣ/(eˣ−1)²):
- Temperature shift:  `G(x)    = G_bb(x)`,
- Compton-y:          `Y_SZ(x) = G_bb(x) · (x·(eˣ+1)/(eˣ−1) − 4)`,
- Chemical potential:  `M(x)    = G_bb(x) · (1/β_μ − 1/x)`,
  with `β_μ = 3ζ(3)/ζ(2) ≈ 2.1923`.

All three templates carry the **same** power of x. This matters: a temperature
shift `T → T(1+δ)` gives `Δn = δ·(−x ∂n_pl/∂x) = δ·G_bb(x)`, so `G = G_bb`
with no `1/x`.

Solve the 3×3 normal equations with uniform weights on the `x∈[0.5,18]` grid.
Report μ, y, ΔT/T for each case. (Use exactly these formulas so both codes
implement the same decomposition from text; if unsure of a template's
normalisation, state it — the *ratios* between codes are what matter.)

**CORRECTION 2026-07-27 (orchestrator).** Revisions of this contract before
today wrote `G` and `Y_SZ` with an explicit `/x` while `M` had none, so the
three templates carried different powers of x and no combination of them could
represent a pure-y spectrum. Measured consequence of the old text: a synthetic
pure `y = 10⁻⁵` input decomposed to `μ = −6.20×10⁻⁵`, `y = 4.40×10⁻⁵`,
`ΔT/T = +2.87×10⁻⁵` with fit residual 5.8×10⁻³; a pure `ΔT/T = 10⁻⁵` input
gave a spurious `μ = +2.19×10⁻⁵ = β_μ × ΔT/T`. Pure μ was unaffected. With the
corrected templates above, all three round-trip exactly (recovered amplitude to
≤10⁻²⁰, residual ~10⁻¹⁶). **The spurious-μ-on-pure-y symptom recorded in
`STATUS.md` was this spec defect, not an implementation error.** Verification
script: `scratchpad/r3/contract_decomp.py`.

## The frozen ingredient table `dev/refsolver/inputs/history.csv`

Columns (SI units): `z, x_e, H_z_per_s, n_e_per_m3, n_H_per_m3, T_gamma_K, t_C_s`.
- `x_e` ionization fraction, `H_z` Hubble [1/s], `n_e` free-electron density
  [1/m³], `n_H` hydrogen density [1/m³], `T_gamma = T_cmb(1+z)` [K],
  `t_C = 1/(σ_T n_e c)` [s] the Thomson time.
- Consume this table (interpolate in log z) for ALL cosmology/recombination
  inputs — do not re-derive X_e. `dτ = dt / t_C` and `dt = −dz / [(1+z) H(z)]`,
  so `dτ = dz / [(1+z) H(z) t_C(z)]` (note dτ > 0 as z decreases).
- Cosmology (for constants only): T_cmb = 2.726 K, Y_p = 0.24, Ω_b = 0.044,
  Ω_m = 0.26, h = 0.71.

**Required domain (pitfall):** the grid MUST extend to `x_max ≥ 30` for the
energy/number integrals to converge (n_pl ~ e^{−x_max}); use x_min ≤ 10⁻³ with
many log-spaced points at low x where DC/BR act. Verify your G₃ integral
reproduces π⁴/15 = 6.4939 to <0.1% before trusting any μ/y.

## Comparison cases

Run these five and write results to `dev/refsolver/outputs/results.json`
(scalars) + per-case spectrum CSVs `dev/refsolver/outputs/spectrum_<case>.csv`
(columns `x, delta_n`):

| Case id | Type | Parameters |
|---|---|---|
| `heat_z2e6` | heat burst | z_h = 2×10⁶, Δρ/ρ = 10⁻³ (deep μ-era) |
| `heat_z2e5` | heat burst | z_h = 2×10⁵, Δρ/ρ = 10⁻³ (μ-era) |
| `heat_z5e3` | heat burst | z_h = 5×10³, Δρ/ρ = 10⁻³ (y-era) |
| `adiabatic` | cooling | no injection; small negative μ, y |
| `photon_x0.1_z3e5` | photon inj. | x_inj = 0.1, σ_x = 0.05 x_inj, ΔN/N = 10⁻³, z_h = 3×10⁵ |

For each: report μ, y, ΔT/T; for the photon case also the photon-number ledger
(ΔN/N in vs. final). Start each run at z_start = z_h + 7σ_z (bursts) or
z ≈ 3×10⁶ (adiabatic), integrate to z = 1 (treat as z≈0).

## Acceptance bands (for the orchestrator's later comparison — you just report)

μ within 2% (deep μ-era), y within 3% (y-era), transition within 5%, spectra
within ~5% pointwise where |Δn| > 1% of its peak. Also: **estimate your own
solver's numerical error by grid-doubling and step-halving**, and report it — a
disagreement is only meaningful relative to the combined error bars.

## Deliverables

1. `dev/refsolver/refsolver.py` (+ any helpers) — the Chang–Cooper solver.
2. `dev/refsolver/README.md` — scheme description, T_e/BR treatment decisions,
   normalisation conventions, self-convergence error estimate, and a statement
   that the isolation rules were followed (you did not read the reference code).
3. `dev/refsolver/outputs/results.json` + spectrum CSVs.
4. A convergence check (grid-doubling) documented in the README.

Return, as your final message: the results table (μ, y, ΔT/T per case), your
self-estimated numerical error, any physics you had to omit (e.g. BR) and why,
and any place you were forced to guess a convention.
