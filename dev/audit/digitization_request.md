# Literature-figure digitization request (Workstream R5)

**Date:** 2026-07-06
**Plan:** dev/PLAN_VALIDATION_ROUND2_2026-07-06.md, Workstream R5
**For:** EB (digitization is EB's task — the agent must NOT invent digitized
data, directive 2). This file lists exactly what to digitize, the CSV schema,
and the spectroxide command that generates each comparison curve.

Round 1 anchored *coefficients* to papers; R5 anchors *curves* by turning
digitized published figures into a CI regression suite
(`python/tests/test_literature_curves.py`, which skips-with-notice until the
CSVs land).

## Already machine-readable — NO digitization needed

Before requesting any manual digitization, the agent checked for existing
machine-readable data:

1. **Dark-photon ε(m) limits (Fig 8 / `fig:dp_firas`).** The `dev/AxionLimits/`
   repo already ships the relevant published limit curves under
   `limit_data/DarkPhoton/`:
   - `COBEFIRAS_Chluba.txt` — **CCJ24** (Chluba, Cyr & Jitendran, arXiv:2409.12115),
     m ∈ [~10⁻¹⁵, 10⁻³] eV. This is the paper we reproduce (Round-1 CCJ24
     statistic, ~3%). **Primary dark-photon anchor.**
   - `COBEFIRAS_Cyr.txt` — Cyr et al. 2024 (arXiv:2411.13701, CosmoTherm), to
     m ≈ 1.8×10⁻⁴ eV.
   - `COBEFIRAS_Arsenadze.txt` — Arsenadze et al. 2024 (arXiv:2409.12940).
   These become direct (iii)-class regression anchors — see the comparison
   protocol below. **No digitization required for Fig 8.**

2. **Visibility functions (Fig 2 / Table 1).** These compare the PDE fit against
   the Chluba 2013 **analytic fitting formulas** (J_bb*, J_μ, J_y), which are
   already implemented in `greens.rs`/`greens.py` and were verified coefficient-
   by-coefficient in Round 1 (P1-9). The paper's Fig 2 residual (<0.05) *is* the
   comparison; no external curve to digitize (the "curve" is a closed-form
   function). **No digitization required.**

## Digitization requested (EB → `dev/audit/digitized/<paper>_<fig>.csv`)

The one channel lacking any curve anchor is **photon injection** (Fig 6,
`fig:photon_injection`). First try the arXiv source tarball (below); only
hand-digitize what that does not provide.

### Request D1 — Chluba 2015 photon-injection Green's function
- **Paper:** Chluba 2015, arXiv:1506.06582 (MNRAS 454, 4182), "Green's function
  of the cosmological thermalization problem — II. Effect of photon injection
  and constraints."
- **First check the arXiv tarball:** `arxiv.org/e-print/1506.06582` — Chluba
  often ships the plotted `.dat`/`.txt` next to figures, and hosts
  Green's-function data on his webpage. If found, that skips manual work and is
  a better anchor; record provenance in the memo.
- **Figure to digitize (if no data file):** the figure showing the
  distortion/μ from monochromatic photon injection as a function of injection
  frequency x_inj at fixed z (the balanced-frequency x₀≈3.6 zero-crossing panel
  is the most diagnostic — it pins the sign flip our Fig 6/7 depend on).
- **Curves:** each injection redshift shown.
- **CSV schema:** `x, y, curve_id` where x = x_inj (or ν), y = the plotted
  quantity (μ per ΔN/N, or Δn amplitude), curve_id = a short label per redshift.
- **Axis ranges/scales:** state log/linear per axis when digitizing (typically
  log-x, linear-or-log-y).
- **spectroxide comparison command:** `notebooks/paper_figures/photon_injection_spectra.ipynb`
  generation path, or `spectroxide solve photon --x-inj <X> --z-h <Z>
  --delta-n-over-n 1e-5` per point.

### Request D2 (optional) — Bolliet, Chluba & Battye 2020 photon-injection
- **Paper:** arXiv:2012.07292. Same check-tarball-first rule.
- **Figure:** μ(x_inj, z_h) surface / curves if it adds coverage beyond D1.
- **CSV schema:** same `x, y, curve_id`.

## Tolerance policy (per-figure, justified — no blanket 30% bars)

`test_literature_curves.py` compares within a tolerance built from:
- **digitization error:** 2–5% for log-log reads (estimate per-figure from axis
  span and marker size — state it in the CSV header comment),
- **Round-1 error budget:** ~0.3% spectroxide discretization,
- **known methodology deltas:** e.g. Chluba's exact vs our analytic P_s.
Any curve requiring >10% must be explained in the R5 memo, not merely tolerated.

## Dark-photon comparison protocol (agent-side, uses AxionLimits — like-for-like only)

For Fig 8, compare our ε(m) limit against `COBEFIRAS_Chluba.txt` (CCJ24) — but
**match the statistic and CL convention** (Round-1 P1-5/P1-6: convention
mismatch alone gives ~2× spread). Where our limit *should* differ, state the
expected offset and check the *observed* offset against it, rather than claiming
raw agreement:
- HyRec X_e sensitivity: γ_con up to +25% (ε −10.5%) at m ≈ 1.2–2.5×10⁻⁹ eV
  (Round-1 xe_hyrec_comparison.md).
- Unresolved ~22% γ_con offset vs the reference (Bryce) figure at the frozen-
  absorption masses (memory: axion-dp-distortion work).
Record which AxionLimits file + upstream paper each comparison uses.

## Status

- Dark-photon anchor: machine-readable, ready — no EB action.
- Photon-injection D1/D2: **awaiting EB** (try arXiv tarball first; digitize
  remainder). `test_literature_curves.py` will be committed skipping so the
  suite activates automatically as CSVs land in `dev/audit/digitized/`.
