# Fig. 4 (`fig:dm_comparison`) — where the PDE↔CosmoTherm residual comes from

**Date:** 2026-07-30
**Figure:** `notebooks/figures/pde_gf_dm_comparison.pdf`, notebook
`notebooks/paper_figures/dm_scenario_comparison.ipynb`
**Trigger:** referee objection to the residual disagreement between the
spectroxide PDE and the CosmoTherm Green's-function convolution.
**Reproduce:** scripts under `dev/scripts/dm_residual_diagnostics/` (see
"Scripts" below).

## Resolution (2026-07-30)

`dm_scenario_comparison.ipynb` now pins `PDE_DTAU_MAX = 1.0` (it previously took
the CLI default of 10, while `energy_conservation.ipynb` pinned 3 — see F-DM-4).
The figure's own residual metric improves:

| scenario | RMS resid, dtau_max = 10 | dtau_max = 1 |
|---|---|---|
| Decay  | 0.12 % | **0.07 %** |
| s-wave | 0.89 % | **0.37 %** |
| p-wave | 0.72 % | **0.46 %** |

Measured on the common 800-point grid used by the diagnostics below, with the
amplitude/shape split:

| scenario | RMS | rescale needed | RMS after rescale | Δρ_dist ratio CT/PDE | total Δρ/ρ deficit |
|---|---|---|---|---|---|
| Decay  | 0.019 % | −0.013 % | 0.018 % | 1.00027 | −0.06 % |
| s-wave | 0.290 % | +0.504 % | 0.135 % | 1.00420 | −0.43 % |
| p-wave | 0.366 % | +0.685 % | 0.021 % | 1.00715 | −0.13 % |

The s-wave deficit followed the measured dtau^0.47 scaling exactly (predicted
0.42 %, measured 0.43 %). Runtime for the three scenarios in parallel at
n_points = 8000 was under an hour.

**New finding from the tighter run (F-DM-6).** p-wave's *distortion*-energy
deficit (0.715 %) is now 5× its *total*-energy deficit (0.13 %). Energy is no
longer being lost — it is being misplaced into the thermalized channel, i.e. the
solver over-thermalizes by ~0.7 % relative to CosmoTherm for injection at
z ≳ 10⁶. **The stepping explanation given here was wrong — see the tolerance
scan below, which refutes it.**

## Tolerance scan (2026-08-05): the residual is a floor, not a step error

Six solves over `dy_max` and six over `dtau_max`, all at `n_points = 8000`,
`z ∈ [1001, 5×10⁶]`. Metric is the amplitude factor `s−1` from
`amplitude_vs_shape.py`; every configuration is cached under
`~/.spectroxide/dm_pde_cache/` keyed on (physics hash, scenario, tolerances),
so none of this has to be re-solved.

**`dy_max` does nothing** (at `dtau_max = 1`):

| scenario | dy 0.005 | dy 0.002 | dy 0.001 |
|---|---|---|---|
| Decay  | −0.013 % | −0.013 % | −0.013 % |
| s-wave | +0.504 % | +0.494 % | +0.490 % |
| p-wave | +0.685 % | +0.624 % | +0.623 % |

F-DM-6 predicted the opposite because it placed the step-selection crossover at
z ≈ 3.6×10⁶. That is the crossover for `dtau_max = 3`. The condition is
`dy_max/θ_e = dtau_max` with θ_z = 4.60×10⁻¹⁰(1+z), so at `dtau_max = 1` and
`dy_max = 0.005` it sits at **z ≈ 1.1×10⁷ — above `z_start`**, and `dtau_max`
sets *every* step in the domain. Lowering dy to 0.002 moves the crossover to
z ≈ 4.4×10⁶, leaving only a thin sliver of the domain where dy binds; that
sliver is the entire 0.06 pp of improvement seen in p-wave. Step counts confirm
it: 1.05M at dy = 0.002 vs 1.42M at dy = 0.001, for no change in the answer.

**`dtau_max` improves sublinearly; there is no floor at the level first claimed.**

| scenario | dtau 3.0 | dtau 1.0 | dtau 0.3 | effective power |
|---|---|---|---|---|
| Decay  | +0.020 % | −0.013 % | −0.024 % | — (already ≈0) |
| s-wave | +0.792 % | +0.504 % | +0.266 % | ≈ dtau^0.47 |
| p-wave | +0.758 % | +0.685 % | +0.577 % | ≈ dtau^0.12 |

A first-order-in-Δτ error would have taken p-wave from +0.758 % to +0.253 %
between dtau = 3 and dtau = 1; it went to +0.685 %, so the error is **not**
first order in Δτ. An intermediate write-up of this scan fitted `a + b·dtau` to
those two points and reported floors of +0.65 % (p-wave) and +0.36 % (s-wave).
The dtau = 0.3 run refuted both: measured +0.577 % and +0.266 %, i.e. below the
claimed floors. Two points cannot separate a floor from a weak power law —
that fit should not have been made.

s-wave's ≈dtau^0.47 reproduces the exponent already measured at dtau_max = 10
(see "Resolution" above) and is consistent with converging to zero. p-wave's
dependence is much weaker; whether it has a genuine floor is **unresolved** on
this evidence.

`E_PDE < E_CT` in every row and at every tolerance — the PDE thermalizes
slightly more than the CosmoTherm GF convolution, by an amount that tracks the
fraction of energy injected deep in the μ-era (p-wave annihilation ∝ v² weights
high z; Decay peaks at z ~ 5×10⁴ in the y-era, and shows no offset).

**The spatial (N) confounder is ruled out.** The energy-conservation work finds
the Δρ/ρ deficit carries an *opposite-sign spatial (N) term*, with only the
joint Δτ×N limit closing to ≤0.03 %, which raised the worry that a residual
measured at fixed `n_points = 8000` sits between two cancelling errors. It does
not, for this observable — doubling the grid at dtau = 1.0 changes nothing:

| scenario | n = 8000 | n = 16000 |
|---|---|---|
| Decay  | −0.013 % | −0.013 % |
| s-wave | +0.504 % | +0.502 % |
| p-wave | +0.685 % | +0.673 % |

So `s−1` is converged in N at 8000 and is driven by Δτ alone.

**p-wave has no floor either.** On log-log the local slope *steepens* as dtau
falls — 0.092 between dtau = 3→1, 0.143 between 1→0.3. A genuine floor would
flatten it toward zero, so the amplitude offset is still descending, just
slowly. Extrapolating the local power gives ≈0.50 % at dtau = 0.1, which would
cost ~48 h of solve time to confirm and is not worth it. **No evidence remains
for a CosmoTherm-side error; the residual is our Δτ discretization, converging
sublinearly.**

**What to say about the figure.** At every tolerance tested, p-wave's residual
*after* a single amplitude rescale is 0.020–0.022 % of peak, and the rescale
factor tracks `E_CT/E_PDE` to within 0.01–0.08 pp. The shape agreement is
tolerance-independent and excellent; the visible residual is one number per
panel — the distortion-energy deficit — which shrinks with `dtau_max` and does
not indicate a physics disagreement.

**Figure change.** The residual panel ran `set_ylim(-2, 2)` while the widest
curve reaches −0.83 % (p-wave PDE); it is now ±0.9, and cell 15 prints the
per-scenario spans so the limit can be re-checked rather than guessed.

## Summary (original diagnosis, at dtau_max = 10)

The residual is **not** a physics disagreement. Decomposed into amplitude and
shape it is, at the tolerances the notebook used:

| scenario | RMS resid, % of peak | after one free amplitude rescale | required rescale |
|---|---|---|---|
| Decay  | 0.074 | 0.023 | +0.135 % |
| s-wave | 0.748 | 0.250 | +1.382 % |
| p-wave | 0.617 | 0.016 | +1.158 % |

A *single multiplicative factor* removes 67–97 % of the residual, and that
factor equals the ratio of distortion energies to 4 significant figures:

| scenario | Δρ_dist/ρ (CosmoTherm) | Δρ_dist/ρ (PDE) | ratio | best-fit rescale |
|---|---|---|---|---|
| Decay  | 9.98334e-6 | 9.96612e-6 | 1.00173 | 1.00135 |
| s-wave | 8.42495e-6 | 8.31468e-6 | 1.01326 | 1.01382 |
| p-wave | 3.39785e-6 | 3.35794e-6 | 1.01189 | 1.01158 |

(∫x³Δn dx / G₃ over 0.02 ≤ x ≤ 30, after the number-conserving strip.)

So: **the shapes agree to 0.02–0.25 % of peak; the direct PDE run carries
0.17–1.33 % too little distortion energy.** The visible residual is our own
energy-budget error, not a spectral-shape difference with CosmoTherm.

## The energy deficit is the direct solve's, not the physics'

The same PDE code, entered through the Green's-function table, agrees with
CosmoTherm far better:

| pair | Decay | s-wave | p-wave |
|---|---|---|---|
| PDE direct − CosmoTherm GF  | 0.074 % | 0.747 % | 0.617 % |
| spectroxide GF table − CosmoTherm GF | 0.019 % | 0.167 % | 0.107 % |
| PDE direct − spectroxide GF table | 0.082 % | 0.897 % | 0.512 % |

(RMS in ΔI, normalised to the PDE peak, over the figure's mask.)
Distortion energies, GF table vs CosmoTherm: ratios 0.99989, 0.99817, 1.00224 —
i.e. **the thermalization efficiency and μ/y branching agree with CosmoTherm to
0.2 %.**

The reason the table is better is not a better solve: `_build_greens_table`
normalises every entry by the solver's *achieved* Δρ/ρ
(`scale = 1.0 / drho_actual`, `python/spectroxide/greens_table.py`), which
divides the energy deficit out. The direct solve gets no such renormalisation,
so its deficit lands in the figure.

Per-entry deficits are stored in the table (`delta_rho_over_rho`) and show where
the solver loses energy at `dy_max = 0.005`, `dtau_max = 10` (default),
`n_points = 8000`:

| z_h | 1.0e3 | 1.5e3 | 3.6e3 | 3.1e4 | 1.7e5 | 6.1e5 | 9.4e5 | 1.4e6 | 2.2e6 | 3.4e6 |
|---|---|---|---|---|---|---|---|---|---|---|
| deficit | −64.8 % | −0.96 % | −0.26 % | −0.17 % | −0.28 % | −0.71 % | −0.98 % | −0.84 % | −0.42 % | −0.21 % |

This ordering explains the scenario ordering in the figure: decay peaks at
z_X = 5e4 where the deficit is smallest (0.17 %); p-wave has
d(Δρ/ρ)/dz ≈ const so its weight sits at z ≳ 1e6 where the deficit is ~1 %;
s-wave is log-uniform in z and lands in between but comes out worst (1.33 %),
i.e. the continuous-source run is somewhat worse than the heating-weighted mean
of single bursts.

## Ruled out

Checked and too small to matter:

- **Convolution quadrature.** μ and y are converged to <1e-4 % between
  n_z = 1000 and 20000.
- **Upper integration limit.** z_max = 4e6 → 5e6 moves μ by 5e-3 % (decay),
  7e-3 % (s-wave), 0.05 % (p-wave). Truncating at 2.5e6 would matter (0.9 % for
  s-wave, 4.3 % for p-wave); 5e6 is safe.
- **Cosmology mismatch.** The Python heating rates default to
  `COSMOTHERM_GF_COSMO` (N_eff = 3.04) while the Rust PDE runs
  `Cosmology::default()` (N_eff = 3.046). Effect on μ, y: 0.02–0.04 %. Worth
  making consistent for tidiness, not a fix.
- **G_bb strip asymmetry.** The PDE grid runs to x_min = 1e-5, the CosmoTherm
  database to 1e-3, the GF table to 5e-3. α = ΔT/T is stable to <2 % against
  x_min over that range, and CosmoTherm's stored entries are already
  number-conserving (α_CT/μ ~ 5e-4), so the two conventions coincide. The
  residual G_bb amplitude after stripping is ≪ μ.
- **Nonlinearity of the CosmoTherm GF.** Every database entry was computed at
  Δρ/ρ = 1.0e-6 (header metadata), deep in the linear regime.

## Findings

**F-DM-1 (numerics, affects the figure).** The direct PDE runs behind Fig. 4 lose
0.17–1.33 % of the distortion energy at `dtau_max = 10` (the CLI default) and
`dy_max = 0.005`. This is the entire visible residual. Tolerance dependence for
the s-wave scenario at n_points = 2000: deficit 1.627 % → 1.223 % → 0.890 % for
`dtau_max` = 20 → 10 → 5, i.e. sublinear (≈ dtau^0.45), so `dtau_max` alone does
not reach 0.2 % at acceptable cost. Grid size is not the knob: n_points 2000 vs
8000 at fixed tolerances differ by 0.06 %. *Status: `dy_max` and
`number_conserving` legs of the scan in progress
(`dev/scripts/dm_residual_diagnostics/scan_tolerances.py`).*

**F-DM-5 (docs claim not supported).** `src/cli.rs` advertises
`--dtau-max <val>  Max Compton optical depth per step (default 10; use 3 for
<0.1% precision)`. For continuous injection that is wrong by ~7×: s-wave at
`dtau_max = 3` still loses 0.697 % of the injected energy. The claim presumably
came from a single-burst test. Reword or qualify it.

**F-DM-4 (figure suite is inconsistent, root cause of the referee's objection).**
`notebooks/paper_figures/energy_conservation.ipynb` pins `dtau_max = 3` — with an
in-cell comment that this is the only knob that moves the deficit — and the
published energy-conservation figure therefore reports min −0.352 %,
max −0.066 %, RMS 0.199 % over z_h ∈ [2e3, 3e6]. But `dm_scenario_comparison`,
`pathological_heating` and `cosmotherm_comparison` never pass `dtau_max`, so they
run at the CLI default of 10, where the single-burst deficit reaches −0.98 % at
z_h ≈ 9e5 (table above). Fig. 4 is thus computed at a looser temporal tolerance
than the figure that certifies its energy budget, and the residual the referee
objects to is the direct consequence. Only `energy_conservation.ipynb` and
`photon_injection_spectra.ipynb` pin `dtau_max` anywhere in the figure suite.

**F-DM-2 (GF table defect, small weight here).** The `z_h = 1000` table entry
loses 65 % of its injected energy: the burst's Gaussian has
σ_z = max(0.04 z_h, 100) = 100, so more than half of it lies below the run's
z_end = 1001 and is never injected. Self-normalisation then inflates a spectrum
built from the burst's tail only. This is why the entry-by-entry comparison
against CosmoTherm degrades to 3.45 % RMS at z_h = 1000 while staying ≤0.1 % for
1.4e3 ≤ z_h ≤ 2.9e6. Fix: build the table with z_end below the lowest z_h, or
drop the first entry. Weight in Fig. 4 is small (one Δln z bin at the bottom of
the integration range) but the defect is real.

**F-DM-3 (bookkeeping).** The cached HQ Green's-function table
(`~/.spectroxide/greens_table_hq.npz`, built 2026-04-27) carries a
`GreensTableHashMismatch` against the current binary: table hash
`93071c5ccb5b4026`, current `fa99e1d06338e2f0`. The GF-table numbers above are
therefore from a stale binary. Rebuild before the figure ships.

## Entry-by-entry GF comparison

spectroxide GF table vs CosmoTherm database, per injection redshift, spectra
compared on 0.5 ≤ x ≤ 12 after the number-conserving strip, RMS as % of the
CosmoTherm peak:

| z_h | 1.0e3 | 1.4e3 | 2.5e3 | 6.3e3 | 2.2e4 | 8.1e4 | 3.0e5 | 1.1e6 | 2.3e6 | 2.9e6 | 3.7e6 | 4.7e6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RMS | 3.45 % | 0.11 % | 0.07 % | 0.02 % | 0.01 % | 0.00 % | 0.00 % | 0.02 % | 0.03 % | 0.11 % | 0.71 % | 16.8 % |

The z_h > 3.7e6 entries are irrelevant: after the exp(−(z/2e6)^{5/2})
suppression their μ per unit Δρ/ρ is ≤2e-4 of the peak GF.

**Do not read the μ/y split of these entries.** A three-shape (M, Y_SZ, G_bb)
least-squares is degenerate through the transition era: at z_h = 1.6e5 the fitted
μ differs by −2.7 % and y by +51 % between the two codes while the *spectra*
agree to 0.01 % of peak. The same caveat applies to the scenario-level μ, y in
`dm_cosmotherm_compare.py` output: the ±7–11 % "y disagreements" there are
decomposition degeneracy, not physics. Compare spectra, or compare energies.

## Scripts

`dev/scripts/dm_residual_diagnostics/`

- `attribute_residual.py` — three-way RMS table (PDE, GF table, CosmoTherm).
- `amplitude_vs_shape.py` — one-parameter rescale test + distortion energies.
- `gf_entry_comparison.py` — per-z_h GF-vs-GF comparison.
- `scan_tolerances.py` — solver-tolerance scan for the energy deficit.
