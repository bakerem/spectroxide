# External X_e anchor: HyRec-2 vs recombination.rs (B4 X_e-swap, ingredient level)

**Date:** 2026-07-05. Closes the P1-10 open item ("no CLASS/HyRec grid
comparison possible") and delivers the ingredient-level part of B4's X_e-swap
experiment.

## Setup

- Reference: HyRec-2 (github.com/nanoomlee/HyRec-2, master @ 2026-07-05
  shallow clone), built standalone with `gcc -O2` (no `-DCAMB`), run with the
  **exact spectroxide default cosmology**: h=0.71, T₀=2.726 K, Ω_b=0.044,
  Ω_cb=0.26, Ω_k=0, no massive ν, Y_He=0.24, N_eff=3.046, all exotic-physics
  switches off. Output z ∈ [1, 8000], Δz = 1.
- Artifacts: `dev/output/hyrec2_xe_default_cosmo.dat` (z, X_e, T_m) and
  `dev/output/hyrec2_input_default_cosmo.dat` (the input file, for exact
  reproduction).
- Our side: `python/spectroxide/cosmology.ionization_fraction` (parity-pinned
  to Rust `recombination::ionization_fraction` at ≤1e-5 by the Phase-0
  harness, so the comparison binds both implementations). Same X_e convention
  (n_e/n_H including He; both give 1.159 at z=8000).

## X_e comparison (max |rel| per band)

| z band | max rel diff | comment |
|---|---|---|
| 5000–8000 | 0.14% | He²⁺/He⁺ Saha region |
| 3000–5000 | 0.05% | |
| 1600–3000 | **5.7% (z≈2300)** | He⁺→He⁰: our Saha vs HyRec non-equilibrium He (HyRec higher, i.e. delayed recombination — expected direction) |
| 1100–1600 | 1.1% | |
| 800–1100 | 1.9% | H recombination; Peebles TLA F=1.125 |
| 200–800 | 1.9% (at z=200) | |
| 50–200 | 10% (at z=50) | post-freeze-out tail |
| 1–50 | **33% (at z=1)** | ours 2.30e-4 vs HyRec 1.73e-4 |

The tail excess is the documented α_B(T_radiation) convention (flagged in
P1-10 and in code comments): T_m < T_γ after decoupling, α_B(T_γ) < α_B(T_m),
so residual recombination is under-driven and X_e stays high. Sign and
magnitude are consistent with that explanation; no anomaly.

## Observable impact (X_e-swap, no theoretical dismissals)

HyRec X_e (cubic-log interpolation, z ≤ 8000; own X_e above) monkey-patched
into `spectroxide.cosmology/greens/dark_photon` simultaneously; every quantity
recomputed before/after on the default cosmology:

| Quantity | Where it matters | Max change | Where | Typical |
|---|---|---|---|---|
| P_s (numerical τ_ff) | firas_photon_limits | −0.9% for P_s ≳ 1e-3; −8.8% only where P_s ≲ 1e-11 | x_inj = 1e-3 deep-absorption tail | ≲0.06% for P_s > 0.5 |
| y_γ(z) bump broadening | photon-injection GF | +1.6% at z=1e3 (y_γ = 1.3e-7 there) | low-z, where broadening is negligible anyway | ≤0.15% for z ≥ 1e4 |
| γ_con/ε² | dark_photon_constraints | **+25% at m = 2.0e-9 eV (z_res ≈ 2450); −7.3% at m = 1.26e-9 eV** | z_res in the He-recombination window 1800–2500 | <1% outside it |
| ε limit (∝ γ_con^{-1/2}) | same figure | **−10.5% / +3.9%** at those masses | same window | <0.5% |
| z_res | same | 1.5% | m = 1.6e-9 eV | <0.1% |

Notes:

- The γ_con sensitivity is exactly the P1-4 prediction: the d-factor
  |d ln ω_pl²/d ln a| takes a finite difference of X_e, and the Saha He kink
  vs HyRec's smooth non-equilibrium He recombination differ most in the
  derivative. Our base γ_con(m) has a non-monotonic dip at m = 2e-9 eV
  (4.63e10 between 5.36e10 and 6.21e10) that the HyRec swap smooths away —
  the dip is a Saha-kink artifact. This is the one place where the Peebles+
  Saha simplification leaks tens of % into a published-figure ingredient, and
  it is confined to masses with z_res ≈ 1800–2500 (roughly
  1.2e-9 ≲ m ≲ 2.5e-9 eV). Feeds the rewritten Sect. 7 per-simplification
  validity bounds. ~~remains a candidate contributor to the unresolved ~22%
  cross-code γ_con offset~~ — **ruled out 2026-07-07/30**: the offset is
  broad-band while this effect is mass-localized (already noted), and the
  Landau–Zener integration now confirms γ_con itself to 1.2% at the adiabaticity
  boundary (`gamma_con_lz_check.md`), so no X_e-driven d-factor sensitivity can
  produce it. This finding stands on its own for Sect. 7; it is not evidence
  about the offset.
- PDE (Rust) results: μ-era thermalization runs at z ≳ 1e5 where hydrogen and
  helium are fully ionized and both codes agree to <0.2% — Peebles/Saha
  detail is irrelevant there. The y-era PDE sees X_e only through weak
  Compton/DC/BR rates at z < 8000; the ingredient-level bounds above cap the
  effect. No PDE rerun performed (no tabulated-X_e input mechanism in the
  Rust solver; adding one is optional Phase-4 work if full figure regeneration
  wants it).

## Actions taken

- `src/recombination.rs::test_xe_vs_recfast_milestones` tightened from
  order-of-magnitude bands to ±6% around the exact HyRec-2 values at
  z = 1100 / 800 / 200 (closes the P1-10 "tolerances 2–5× wide" item).
- HyRec table + input archived under `dev/output/` for reuse (benchmark pack,
  Sect. 7 figure).

## Verdict

No bug. Peebles(F=1.125)+Saha is within 1.9% of HyRec-2 everywhere it feeds
observables except (i) the He-recombination derivative entering γ_con
(up to 25%, mass window m ≈ 1.2–2.5e-9 eV → ≤10.5% on ε limits) and (ii) the
z ≲ 50 tail (33%, no observable consequence). Item (i) should be stated as a
validity bound on the dark-photon figure in the paper revision.
