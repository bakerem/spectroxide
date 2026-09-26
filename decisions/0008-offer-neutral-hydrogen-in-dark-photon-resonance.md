# Offer neutral hydrogen in the dark-photon resonance as an option

## Status

Accepted, 2026-09-26 (EB).

## Context

The dark-photon scenario (`InjectionScenario::DarkPhotonResonance`) sets the photon mass to the
plasma frequency, m_γ = ω_pl, finds the one redshift z_res where ω_pl = m_A', and installs the
depletion Δn(x) = −[1 − exp(−γ_con/x)] n_pl(x) there. Audit finding C1
(`dev/REVIEW_PAPER_CLAIMS_2026-09-24.md`) flags that this drops the neutral-hydrogen term, which
matters after recombination.

The full photon mass is (Caputo, Liu, Mishra-Sharma & Ruderman 2020, PRD 102, 103533, Eq. 1;
also PRL 125, 221303, Eq. 2)

m_γ² = ω_pl² − 2ω²(n − 1)_H = ω_pl² − 4π α_H n_HI ω²,

with (n − 1)_H = 2π α_H n_HI and static polarizability α_H = 4.5 a₀³, so
4π α_H = 8.380×10⁻²⁴ cm³ (our calculation from CODATA a₀; Caputo et al. quote 8.4×10⁻²⁴).
Mirizzi, Redondo & Sigl (2009, JCAP 0903, 026), Eq. 20, has the same form with a coefficient
about 20% larger, a value for molecular hydrogen; Caputo et al. correct it in PRD footnote 1.
Chluba, Cyr & Johnson (2024, MNRAS 535, 1874; CCJ24) neglect the term in their Sec. 2 and use
m_γ = ω_pl, as we do. We took the CCJ24 statement from the task brief and did not reread the
paper.

The neutral term grows as x²(1+z)² relative to the plasma term at fixed x = ω/(k T_γ), and it
needs 1 − X_H to be large. After recombination it can dominate. For m_A' = 10⁻¹¹ eV the
plasma-only resonance is at z_res = 668.2, and there the neutral term is 1.88 times the plasma
term at x = 4 (our calculation, Python prototype, and the Rust test below). m_γ² is negative
at z_res for those photons, so they do not resonate there.

For each x, every crossing of m_γ²(z, x) = m² converts with the Landau–Zener probability
(Mirizzi et al. 2009, Eq. 18; Caputo et al. 2020, PRL Eq. 3)

P_i(x) = π ε² m² / (ω_i H_i |d ln m_γ²/d ln a|_i),   ω_i = x k T_γ(z_i),

with the slope at fixed x. The crossings add incoherently, so the photon survives with
probability exp(−τ), τ(x) = Σ_i P_i. Without the neutral term, τ = γ_con/x, the current code.

## Decision

Offer the full photon mass as an option, off by default. When it is on, compute the depletion
from the full photon mass, per frequency, average it over each grid cell, and install it as an
impulsive initial condition at the plasma-only z_res (alternative b below).

- The default stays m_γ = ω_pl, as in CCJ24: Δn = −[1 − exp(−γ_con/x)] n_pl at point values,
  the same arithmetic as before this ADR. The paper compares its limits with CCJ24, so its
  Fig. 8 uses the default, and the paper states that we neglect the term (EB, 2026-09-25).
- The option is the `neutral_hydrogen` field of `InjectionScenario::DarkPhotonResonance`, the
  CLI flag `--neutral-hydrogen`, and the key `"neutral_hydrogen": True` in the Python injection
  dict.

- `dark_photon::photon_mass_sq_ev2(z, x)` evaluates m_γ² in eV², with n_HI = (1 − X_H) n_H and
  X_H = X_e − x_He, clamped to [0, 1]. Using X_H instead of 1 − X_e keeps n_HI right while
  helium is still ionized, since X_e counts helium electrons. At z = 2000 the two differ
  (1 − X_H = 5.7×10⁻⁶, 1 − X_e = 2.1×10⁻⁶), but the neutral term is then below 10⁻⁷ of the
  plasma term for x ≤ 30.
- `dark_photon::conversion_probability(ε, m, x)` gives point values. It scans 3000 log-spaced
  redshifts over [10, 3×10⁷], bisects every sign change of m_γ² − m², and refines every scan
  extremum by golden-section search, so a pair of crossings closer than one scan step is still
  found. The slope is taken term by term in ln(1+z), with the same finite-difference step on
  X_e as `dln_omega_pl_sq_dlna`, so the plasma-only limit reproduces `gamma_con`.
- `dark_photon::cell_averaged_probability` averages 1 − exp(−τ) over each grid cell
  [x_{i−1/2}, x_{i+1/2}] (half cells at the ends), and `initial_delta_n` installs
  Δn_i = −⟨1 − exp(−τ)⟩_i n_pl(x_i). The reason is tangency, below.
- The builder and CLI start the solve at the plasma-only z_res, as before.
  `resonance_params` is unchanged.
- `gamma_con` and `resonance_redshift` stay plasma-only. They give the nominal (γ_con, z_res)
  for reporting and range warnings, and the axion module reuses them unchanged. A mass with no
  plasma-only resonance in [10, 3×10⁷] still has no resonance.
- The Python mirror (`spectroxide.dark_photon`) implements the same functions
  (`conversion_probability`, `cell_average` for any ascending, non-uniform grid,
  `tau_per_epsilon_sq`, cell-averaged by default). `tau_per_epsilon_sq` is the Green's-function
  counterpart of the option. The Green's-function frozen branch in
  `notebooks/paper_figures/dark_photon_constraints.ipynb` (z_res ≤ 3×10³) uses the plasma-only
  template −n_pl/x, like the PDE default, so the paper's PDE and GF curves use the same
  physics.

### Why install at z_res

The neutral term only lowers m_γ², so a crossing needs ω_pl(z) ≥ m. Since
ω_pl² ∝ X_e (1+z)³ grows with z, every crossing lies at z ≥ z_res: z_res is the latest any
frequency converts. Low-x photons (neutral term ∝ x²) cross at z_res itself. Higher-x photons
cross earlier, and installing them at z_res misses only the Compton redistribution in between.
The Compton parameter y = ∫ θ_γ σ_T n_e c dz / [H (1+z)] is 1.5×10⁻⁶ from z = 1189 to 225 and
1.4×10⁻⁵ from 1500 to 200 (our quadrature, 20000 points, T_e = T_γ). Bremsstrahlung does not
act on this gap at x ≳ 0.1.

We measured the alternative. For m = 10⁻¹¹ eV (z_res = 668.2) and ε = 10⁻⁵ we ran the PDE
(default grid, z_end = 100) twice: from z_res, and from z = 1188.7, the highest crossing for
x ≤ 30. Started early, bremsstrahlung refills the low-x hole before it physically exists. At
z = 100 the remaining depletion, as a fraction of the initial condition, is:

| x | start at z_res | start at 1188.7 |
|---|---|---|
| 10⁻⁴ | 0.9999 | 0.247 |
| 3×10⁻⁴ | 1.0000 | 0.835 |
| 10⁻³ | 1.0000 | 0.986 |
| ≥ 0.1 | 1.0000 | 1.0000 |

Over 0.5 < x < 11 the two runs differ by 3.4×10⁻⁶ of the peak Δn. The early start is harmless
in the FIRAS band and wrong at low x, so z_res is the install point.

### Tangency

Where m_γ²(z, x) touches m² tangentially, two crossings merge at some x_t. There
|d ln m_γ²/d ln a| → 0, and the narrow-width P_i diverges as (x_t − x)^(−1/2). At
m = 10⁻¹² eV this happens at both ends of the band x ∈ [2.73, 2.98], which has three
crossings. The divergence is integrable, but point values on a grid then depend on where the
points fall. On linear grids with Δx = 0.023–0.036 and ten offsets each, the band integral
I = ∫_{1.2}^{11} x³ n_pl τ dx from point values ranges from −3.9% to +20% of a fine-grid
reference (the ignored Rust test `dark_photon::tests::cell_average_convergence_study`; the
reference uses 20000 cells of Δx = 6×10⁻⁴).

`cell_averaged_probability` finds tangencies in two ways. It bisects the crossing count between
sub-samples, and it solves for them directly: a tangency has f = ∂f/∂z = 0 with
f = P(z) − N(z) x² − m², so x_t² = P'/N' where P − N P'/N' = m². The second way does not
depend on the grid, so it catches a band of extra crossings narrower than a sub-sample; each
estimate is confirmed by a change in the crossing count and refined by bisection. Every cell
within two cell widths of a tangency is split at it and integrated in u = |x − x_t|^(1/2), which
turns the divergence into a smooth integrand. A piece that lies between two tangencies diverges
at both ends; it is split at its midpoint, and each half is integrated toward its own tangency.
Elsewhere the cell average is the mean of 4 midpoint sub-samples. Convergence of I at
m = 10⁻¹², relative to the reference, over ten grid offsets (Rust; the Python mirror agrees to
10⁻⁴):

| method | Δx = 0.023 | Δx = 0.030 | Δx = 0.036 |
|---|---|---|---|
| point values | −3.1% to +12.2% | −3.7% to +16.4% | −3.9% to +20.3% |
| mean of 4 midpoints | −1.7% to +0.7% | −1.9% to +1.1% | −2.1% to +1.3% |
| mean of 64 midpoints | −0.35% to +0.51% | −0.46% to +0.35% | −0.52% to +0.25% |
| u-integration, 8 points | ±0.01% | ±0.01% | ±0.01% |
| u-integration, 32 points (chosen) | ±0.01% | ±0.01% | ±0.01% |

Plain midpoint averaging converges only as N^(−1/2) because it samples the singularity
directly. The u-integration is offset-independent to 10⁻⁴ from 8 points on; we use 32. On a
4000-point grid, `cell_averaged_probability` takes 165–177 ms in release Rust against 43–45 ms
for point values (m = 10⁻¹², 10⁻¹¹, 10⁻⁹ eV). The NumPy mirror takes 0.73 s against 0.16 s.

The narrowest case in the Fig. 8 mass grid is m = 1.49×10⁻¹² eV. Its two tangencies are
1.7×10⁻³ apart (x = 2.6632–2.6649), narrower than a sub-sample, and one cell usually holds both
divergent ends. We compared I with a reference that locates the tangencies independently (a
fine x scan of the crossing count, then bisection) and integrates by Gauss–Legendre in u toward
each tangency (Python script, ten offsets per grid; the Rust test
`cell_average_resolves_a_narrow_tangency_band` repeats it with five offsets). Before this fix
the band was often missed, and every piece was integrated toward one tangency only:

| grid | before | after |
|---|---|---|
| linear, Δx = 0.023 | −2.0% to −0.10% | −6.6×10⁻⁵ to +5.4×10⁻⁴ |
| linear, Δx = 0.036 | −2.2% to −0.09% | −5.8×10⁻⁵ to +3.7×10⁻⁴ |
| log, 200 points over [0.5, 20] | −2.3% to −0.45% | −4.6×10⁻⁵ to +1.1×10⁻⁵ |

At m = 10⁻¹² eV, the same comparison gives −6×10⁻⁵ to +1.1×10⁻⁴ on all three grids. For the
single cell [2.675, 3.025], which holds both m = 10⁻¹² tangencies, the old one-sided
integration was 5.6% low; the fix agrees with the reference to 1.6×10⁻⁴.

The remaining error of a few 10⁻⁴ at m = 1.49×10⁻¹² comes from the tabulated X_e, not the
quadrature. Within about 0.03 in u of a tangency, the two merging crossings are only a few
table spacings (Δz = 0.5) apart. There the finite-difference X_e slope in
|d ln m_γ²/d ln a| jumps from one linear table segment to the next, and τ·u, which the
(x_t − x)^(−1/2) law predicts to be constant, scatters by tens of percent. The cell averages of
the two cells at the band do not settle below about 1% as the sub-sample count rises to 2048.
Those cells carry a small share of I, so I stays within 6×10⁻⁴ of the reference.

Cell averaging makes the result grid-independent, not exact. It is still the narrow-width
value, which overstates the conversion near x_t. The true conversion there is finite; a uniform
stationary-phase (Airy-type) treatment would give it, and we do not attempt one. The direct
tangency solve can still miss a band whose two ends lie closer than its finite-difference
accuracy, which happens only for masses just below m ≈ 1.51×10⁻¹² eV, where the bands close.

## Alternatives

- **(a) Keep m_γ = ω_pl and cut the limit curve at z_res ≈ 1100.** No code change, and the
  paper states the omission. It discards the post-recombination masses (m ≲ 10⁻¹⁰ eV) instead
  of computing them, and CCJ24's curve there carries the same error, so a cut would hide a real
  difference.
- **(b) Per-x depletion, cell-averaged, installed at the plasma-only z_res (chosen for the
  option).** Reuses
  the existing initial-condition path. It is never early, and being late costs only the
  Compton y above.
- **(b′) The same depletion installed at the highest crossing (x ≤ 30).** This was the first
  draft of this ADR. It is never late, but bremsstrahlung refills the low-x hole early: 75% at
  x = 10⁻⁴ and 17% at 3×10⁻⁴ for m = 10⁻¹¹ eV (table above). Rejected.
- **(c) A time-dependent per-x depletion source, each frequency removed at its own crossings.**
  Exact in time, but it needs a new photon-sink term in the solver and step control around
  each crossing. The two-start comparison above bounds the timing error of (b) at a few parts
  per million in the FIRAS band, so the extra machinery buys nothing measurable now.

## Consequences

- The depletion changes for masses whose crossings fall after recombination. With ε fixed, the
  ratio of full to plasma-only point values of P (our Rust calculation, ε = 10⁻⁷) is:

  | m_A' (eV) | z_res | x = 0.5 | x = 1 | x = 2.8 | x = 4 | x = 10 | Δρ ratio |
  |---|---|---|---|---|---|---|---|
  | 10⁻¹² | 221.6 | 0.996 | 0.985 | 1.099 (3 crossings) | 4.3×10⁻⁵ | 9.8×10⁻⁷ | 0.65 |
  | 10⁻¹¹ | 668.2 | 0.972 | 0.885 | 0.264 | 0.074 | 0.0030 | 0.45 |
  | 10⁻¹⁰ | 985.9 | 0.998 | 0.995 | | 0.901 | 0.553 | 0.94 |
  | 10⁻⁹ | 1569.3 | 1.000 | 1.000 | | 0.999 | 0.993 | 0.9993 |
  | 10⁻⁸ | 6916.1 | | | | | | 1.0000 |

  The Δρ ratio is ∫x³ τ n_pl dx for the cell-averaged full τ over the plasma-only τ (Python
  mirror, 40000 points, 0 < x < 40). High-x photons cross earlier, where ω is larger and m_γ²
  changes faster, so their P drops. The limits for m ≲ 10⁻¹⁰ eV weaken; above about 10⁻⁹ eV
  nothing changes at the 0.1% level.
- The static polarizability and the neglect of helium are the main modeling errors, and τ is
  more sensitive to them than the coefficient is. Where the neutral term moves the crossing,
  d ln τ / d ln(coefficient) is about −1.5 to −2.9. Helium (α_He = 1.38 a₀³,
  n_He/n_H = 0.079) would add 2.4% to the coefficient and change τ by −3.7% to −7.0% at
  x = 4–10 for m = 10⁻¹² and 10⁻¹¹ eV. It would change τ by −1.3% at x = 10 for 10⁻¹⁰ eV and by
  under 0.3% at x ≤ 1. A single-pole dynamic polarizability, α(ω) = α_H/[1 − (ω/11.2 eV)²], at
  each crossing's ω changes τ by −6.4% at x = 10 and −0.7% to −1.6% at x = 4 for those masses
  (our Python calculation, scaling the coefficient at the crossing's ω). Both are within the
  model and not corrected here.
- The option changes the limits by the amounts above. With the option on, the Fig. 8 PDE
  limit weakens by 18% at 10⁻¹² eV, 7% at 1.1×10⁻¹¹ eV and under 0.1% above 10⁻¹⁰ eV
  (2026-09-24 cache). The FIRAS fit is dominated by x ≈ 1–3 after the temperature and dust
  projection, so the shift is much smaller than the x = 4 depletion ratio.
- The paper's Fig. 8 and its caches (`dev/data/dp_firas_pde_limits.npz`,
  `dp_firas_gf_limits.npz`) use the default, so they omit the term for m ≲ 10⁻¹⁰ eV, as CCJ24
  do. The paper says so in one clause and cites Caputo et al. (2020).
- With the option on, building the dark-photon initial condition costs about 0.2 s per solve
  on a 4000-point grid. The default adds nothing.
