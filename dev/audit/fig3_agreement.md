# Fig. 3 (`fig:spectral_shapes`) — quantifying the PDE ↔ CosmoTherm single-burst agreement

**Date:** 2026-07-30
**Figure:** `notebooks/figures/pde_cosmotherm_comparison.pdf`, notebook
`notebooks/paper_figures/cosmotherm_comparison.ipynb`
**Trigger:** referee 1 lists Fig. 3 among the figures showing "small differences
in the results", attributed to the Sect. 7 simplifications. Companion record for
Fig. 4 is `dev/audit/dm_comparison_residual.md`.
**Reproduce:** ⚠ the `dev/scripts/fig3_agreement/` scripts were not preserved
(lost with the session scratch directory; noticed 2026-08-10). The "Scripts"
section below documents what each did; reconstruction would follow the method
described there, against the same CosmoTherm database entries.

## Summary

There is no physics discrepancy in Fig. 3. Per injection redshift, comparing the
*observable* (number-conserving, G_bb-stripped) spectra over the FIRAS band
68–640 GHz:

| z_h | RMS, dτ_max = 10 → 3 | shape only (after one amplitude rescale) | energy deficit, 10 → 3 | FIRAS 95 % limit shift (dτ=3) | max diff / σ_FIRAS |
|---|---|---|---|---|---|
| 2 931 | 0.218 → 0.218 % | 0.166 % | 0.32 → 0.32 % | 0.13 % | 2.3e-3 |
| 10 018 | 0.161 → 0.072 % | 0.027 % | 0.29 → 0.14 % | 0.10 % | 5.7e-4 |
| 49 582 | 0.090 → 0.028 % | 0.018 % | 0.20 → 0.09 % | 0.04 % | 2.7e-4 |
| 201 688 | 0.164 → 0.057 % | 0.031 % | 0.31 → 0.12 % | 0.08 % | 5.4e-4 |
| 500 000 | 0.357 → 0.119 % | 0.020 % | 0.59 → 0.17 % | 0.20 % | 1.5e-3 |
| 1 944 078 | 0.823 → 0.434 % | 0.032 % | 1.26 → 0.27 % | 0.75 % | 5.8e-3 |

Three statements follow, each independent of the eye:

1. **The residual is one scalar per panel.** A single multiplicative rescale
   removes the residual down to 0.02–0.03 % of peak (0.17 % in the y-era panel).
2. **That scalar is the solver's own energy-budget error, which the solver
   reports.** The distortion-energy ratio ∫x³Δn dx (CT/PDE) equals
   1 − Δρ_achieved/Δρ_requested to three digits in every row
   (e.g. z_h = 1.94e6, dτ=3: ratio 1.00272, achieved Δρ/ρ = 9.97198e-6 against a
   requested 1e-5). It is the first-order-in-Δτ T_e/DC-BR residual documented in
   `dev/audit/energy_conservation_audit.md`, and it scales with `dtau_max`.
3. **It is far below the measurement.** Normalising the distortion to the FIRAS
   95 % upper limit on Δρ/ρ, the PDE−CT difference spectrum peaks at
   4.4e-4 – 1.7e-2 of the FIRAS 1σ error, and the limit itself moves by
   0.04–0.75 % depending on which code supplies the template.

   The limits are marginalised over the two nuisances always present in an
   absolute-spectrum fit — the unobservable G_bb temperature shift and the
   Fixsen (1996) §6.1 galactic dust template. That convention gives
   UL(Δρ/ρ) = 6.0e-5 at z_h = 5e5, i.e. μ_lim ≈ 1.401 × 6.0e-5 = 8.5e-5 against
   Fixsen's 9e-5 — the right ballpark. The unmarginalised `FIRASData.upper_limit`
   is 1.5–3.0× tighter and corresponds to no published constraint; the first
   version of this record used it and understated the difference-over-σ by that
   factor. The `ul_ratio` column below is still the unmarginalised one (the
   marginalised ratio needs the PDE templates, stored only after the script was
   patched); the ratio is far less sensitive to the convention than the absolute
   normalisation. Regenerated for the six Fig. 3 redshifts
   (`fig3_agreement_dtau3_marg.json`): the limit shift becomes 0.265, 0.129,
   0.023, 0.142, 0.240, 0.801 % against 0.131, 0.099, 0.043, 0.079, 0.203,
   0.749 % unmarginalised — same order, ≤2× either way. The sweep's `ul_ratio`
   column is unmarginalised for the same reason; rerunning 114 entries to change
   a number by ≤2× is not worth ~2 h of compute, but the paper should quote the
   marginalised values.

## Two features that are real and should be stated, not buried

- **y-era shape residual.** z_h ≈ 2931 is the only panel that does not improve
  with `dtau_max`, and it keeps a 0.166 % shape residual after rescaling. This is
  a genuine (small) shape difference, not a tolerance effect. Candidate cause is
  the ionisation history (Peebles vs HyRec, `dev/audit/xe_hyrec_comparison.md`);
  not yet diagnosed.
- **High-z amplitude amplification.** At z_h ≈ 1.9e6 the fitted amplitude offset
  (0.43 %) exceeds the total-energy offset (0.27 %). There most of the injected
  energy has already thermalised into G_bb (α_CT = 0.160 per unit Δρ/ρ), so the
  surviving μ is a small difference of large numbers and a 0.3 % energy error
  maps to ~0.4–0.7 % in the observable part.

**The 0.02–0.03 % shape floor is not an artefact of the comparison.** Linear vs
cubic interpolation of the CT database onto our x grid changes the stripped
spectrum by 1e-4 – 7e-4 % of peak, i.e. ~100× below the measured shape residual.

## Findings

**F-F3-1 (open; outside Fig. 3's range).** Below z_h ≈ 2000 the single-burst
energy deficit rises steeply and stops being tolerance-limited: at z_h = 1360 it
is 2.49 % and is unchanged by `z_end` (500 → 200: 2.487 → 2.489 %) and by
`dy_max` (0.005 → 0.001: 2.487 → 2.487 %); tightening both at once
(`dtau_max` 3 → 1 with `dy_max` = 0.001) moves it only to 2.472 %, whereas
z_h = 1996 gives 0.40 % and
z_h = 2931 gives 0.32 %. So it is not the burst-truncation defect of F-DM-2 in
`dm_comparison_residual.md`, and not the Δτ residual that explains every other
redshift. Mechanism unknown. Fig. 3's lowest panel is z_h = 2931, so this does
not affect the figure; it does set the worst case in the full-database sweep.

## Why the figure reads worse than the numbers

- No number is quoted anywhere in the caption or body text (`paper.tex:714–719`),
  so the referee has only the residual panels to judge by.
- The residual panels share a ±5 % axis between `PDE−CT` (ours, ≲0.2 % over the
  band) and `GF−CT`, where "GF" is the Chluba (2013) *fitting formulas*, which
  genuinely deviate by several % through the y–μ transition. The eye attributes
  the red curve's error to spectroxide.
- The only visible blue excursion reaches ≈5 % at x < 0.5 — below the FIRAS band,
  at the extreme left edge of the panel.
- The notebook runs at the CLI default `dtau_max = 10`, while
  `energy_conservation.ipynb` — the figure that certifies the energy budget —
  pins `dtau_max = 3` (finding F-DM-4 in `dm_comparison_residual.md`). Fig. 3 is
  therefore computed at a looser temporal tolerance than the figure that
  validates it.

## Recommendations

1. Regenerate Fig. 3 with `dtau_max = 3`, matching `energy_conservation.ipynb`.
2. Give `PDE−CT` its own residual scale; relabel the red curve as the analytical
   *fitting-formula* benchmark so it is not read as spectroxide's error.
3. Quote the RMS per panel in the caption, and the FIRAS-limit shift in the text.
4. Add the full-parameter-space sweep (below) — referee item 3 asks for exactly
   this ("full agreement across the complete parameter space").
5. Do **not** quote per-entry μ/y agreement: the three-shape least squares is
   degenerate through the transition era (+51 % apparent y difference at
   z_h = 1.6e5 while the spectra agree to 0.01 % of peak — see
   `dm_comparison_residual.md`).

## Full-parameter-space sweep

All 118 CosmoTherm database entries, PDE at `dtau_max = 3`, completed
2026-07-30. 114 ran; the four at z_h ≥ 4.19e6 are refused by the solver's own
guard, because z_start = 2.5 z_h > 1e7 implies θ_e > 5e-3 where the
Fokker–Planck approximation is invalid. Their surviving μ is ~1e-5 of
unsuppressed, so nothing observable is lost.

Statistics over the range where the metric is meaningful, 2e3 ≤ z_h ≤ 3e6
(99 entries):

| quantity | median | 90th pct | max |
|---|---|---|---|
| RMS \|ΔI_PDE − ΔI_CT\| / peak | 0.094 % | 0.368 % | 1.285 % |
| the same after one amplitude rescale (shape only) | 0.023 % | — | 0.256 % |
| shift in the FIRAS 95 % limit on Δρ/ρ | 0.124 % | — | 2.195 % |

Over the full 1.4e3–4e6 span the RMS median is 0.120 % and the shape median
0.027 %; both tails degrade, for different and identifiable reasons:

- **z_h ≲ 1500.** The energy deficit runs away (1.5 % at z_h = 1468, 2.5 % at
  1360, 37 % at 1080, 112 % at 1000) while the *shape* residual stays at
  1.3–1.4 %. Amplitude only — see F-F3-1, and F-DM-2 in
  `dm_comparison_residual.md` for the burst-truncation part of it.
- **z_h ≳ 3e6.** Here the energy deficit is flat at 0.26–0.29 % but the shape
  residual grows (0.14 % at 2.8e6 → 4.4 % at 3.9e6), so this tail is a genuine
  shape divergence. Expected: CosmoTherm multiplies its stored μ+y part by the
  analytic exp[−(z/2e6)^{5/2}] while the PDE computes the thermalization
  directly, and by z_h = 3.9e6 that factor is 4e-3, so the comparison is between
  two small residuals of nearly-complete thermalization. `dm_comparison_residual.md`
  independently found the CT entries above 3.7e6 carry ≤2e-4 of the peak GF.

Figure: `cross_code_agreement.pdf` (`plot_agreement.py`) — RMS, max, and
shape-only residual vs z_h, with the observational impact below and both edge
regions shaded.

## Scripts

`dev/scripts/fig3_agreement/` — **not preserved** (see "Reproduce" above); the
descriptions below are the record of what was run.

- `quantify_agreement.py` — the six Fig. 3 redshifts: RMS/max residual,
  amplitude-vs-shape split, distortion energies, FIRAS limit ratio and
  difference-over-σ. `--dtau-max` selects the tolerance; writes
  `fig3_agreement_<tag>.json`.
- `sweep_all_entries.py` — the same metrics for every database entry, 3 workers,
  resumable JSONL.
- `plot_agreement.py` — residual and observational-impact panels vs z_h.
