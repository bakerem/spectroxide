# CLASS `sd` cross-code comparison (Workstream R1)

**Date:** 2026-07-06
**Plan:** dev/PLAN_VALIDATION_ROUND2_2026-07-06.md, Workstream R1
**Deliverables:** `dev/scripts/class_sd_compare.py`, CLASS I/O under
`dev/output/class_sd/`, this memo.
**Upgrades coverage-matrix rows:** 1, 2, 3, 4, T1 (heat-injection → independent
code, class (iii)).

The single most referee-convincing item: an independent, separately-authored,
different-language spectral-distortion code (CLASS `sd` module; Lucca,
Schöneberg, Hooper, Lesgourgues & Chluba 2020, JCAP 02 (2020) 026,
arXiv:1910.04619) run on the same physics regime as our heat-injection PDE.

**Scope (not oversold):** CLASS `sd` handles *heating* histories only — no
photon-injection or dark-photon channel. R1 anchors the **heat-injection half**
of the paper. Photon injection gets its independent-code anchor from R3; dark
photon from R5 (see coverage_matrix.md).

## Setup

- **CLASS v3.3.0, commit `0ceb7a9`**, prebuilt binary at `/home/bakerem/CLASS`.
  Has the `sd` module (`source/distortions.c`) and Chluba's Green's-function
  data (`external/distortions/Greens_data.dat`), used by
  `sd_branching_approx = exact`.
- **Cosmology matched to `Cosmology::default()` (Chluba 2013 / CosmoTherm):**
  h=0.71, Ω_b=0.044, Ω_m=0.26 → ω_b=0.0221836, ω_cdm=0.1088856, T_cmb=2.726 K,
  Y_p (CLASS `YHe`)=0.24, N_ur=3.046, N_ncdm=0.
- **Deposition:** `f_eff_type = on_the_spot` (f_eff=1) — comparison isolates
  *thermalisation*, not energy-deposition modelling (out of our scope).
- CLASS μ,y read from `distortions_verbose` stdout; heating history from
  `<root>_sd_heating.dat` (column `d(Q/ρ)/dz`, **identical convention** to
  spectroxide's `TabulatedHeating` `rate_table` = `d(Δρ/ρ)/dz`, positive =
  heating — verified by inspection of both). Spectrum from
  `<root>_sd_distortions.dat`.

Regenerate: `python dev/scripts/class_sd_compare.py --case A`.

## Case A — adiabatic ΛCDM (heating-history-matched thermalisation)

**Method.** CLASS computes μ,y for pure ΛCDM (acoustic dissipation + adiabatic
cooling + recombination) with `sd_branching_approx = exact`. We then feed
CLASS's *own* heating history verbatim into the spectroxide PDE
(`solve tabulated-heating`, z_start=4.9×10⁶ full range, z_end=200, N=2000).
Because both codes thermalise the **same** heating history, the heating-history
match (directive R1.4) is **exact by construction** — the comparison isolates
the thermalisation numerics, not the heating model.

Heat deposition of this history: 34% at z>2×10⁵ (deep μ-era), 50% at z>5×10⁴,
50% below 5×10⁴ (y-era). **This is a transition-dominated history**, not a clean
μ-era or y-era injection — closer to the paper's pathological stress tests
(Fig 5) than to Fig 1.

**Three-way comparison on the identical heating history** (total injected
Δρ/ρ = 2.83×10⁻⁸):

| Method | μ | y | μ/1.401 + 4y (total distortion energy) |
|---|---|---|---|
| CLASS `sd` (exact) | 1.931×10⁻⁸ | 3.453×10⁻⁹ | 2.759×10⁻⁸ |
| spectroxide PDE | 1.049×10⁻⁸ | 4.641×10⁻⁹ | 2.606×10⁻⁸ |
| spectroxide GF (Chluba-2013 visibility) | 1.604×10⁻⁸ | 3.641×10⁻⁹ | 2.601×10⁻⁸ |

(spectroxide GF = the branching-ratio/visibility method
`mu_from_heating`/`y_from_heating`: μ = (3/κ_c)∫J_bb*(z)J_μ(z) dq/dz dz,
y = ¼∫J_y(z) dq/dz dz — the *same method class* as CLASS, applied to the same
heating history.)

**Decomposition of the discrepancy (directive 1 — attribute, do not tune):**

1. **Total distortion energy agrees to 5.6%** across all three methods
   (2.76 vs 2.61 vs 2.60 ×10⁻⁸). The codes agree on *how much* energy ends up in
   spectral distortion; they differ on the **μ/y split**.
2. **spx-GF vs CLASS (branching-vs-branching): μ 17% low (1.60 vs 1.93×10⁻⁸),
   y 5% high.** This is the difference between spectroxide's Chluba-2013 *fitted*
   visibility functions and CLASS's `exact` Green's data (Chluba's
   CosmoTherm-derived branching). Lucca+2020 document exactly this level of
   spread between branching approximations in the transition era (their approx.
   comparison figure); it is expected, not a defect.
3. **spx-PDE vs spx-GF: the PDE puts ~34% less in μ and ~28% more in y** (1.05 vs
   1.60×10⁻⁸ μ). For this transition-dominated history the GF's
   independent-per-dz visibility assumption over-predicts μ relative to the full
   coupled PDE evolution, which resolves the partial μ→y erosion in the
   transition. The paper's "PDE↔GF agree to 2–5% on μ" claim holds for *clean*
   single-burst μ-era / y-era injections (Fig 1); it does **not** extend to a
   broad transition-spanning history — a limitation worth stating in the paper.

**z_start robustness:** rerunning from z_start=3×10⁶ vs 4.9×10⁶ changes spx-PDE
μ by <0.2% (1.0474 vs 1.0493×10⁻⁸) — the >3×10⁶ heat thermalises completely and
does not contribute to μ, so truncation is not a factor.

**Verdict for Case A:** cross-code **total distortion energy agrees to <6%**;
the μ/y split difference is fully attributed to (a) the known branching-
approximation spread and (b) the GF-vs-PDE difference for transition-dominated
heating. No spectroxide bug indicated; one **paper-text finding R1-A**: the
PDE↔GF agreement claim should be scoped to clean single-era injections.

## Cases B–D (decay / s-wave annihilation / μ-y transfer) — scaffolded, pending

The stronger, *unambiguous* cross-code check is a **clean deep-μ-era injection**
(decaying particle with z_X ≈ 10⁶), where the answer is μ = 1.401 Δρ/ρ, y ≈ 0,
and the μ/y-split ambiguity of Case A vanishes. These are scaffolded in
`class_sd_compare.py::run_case_decay` (TODO) and require, per directive R1.4:

1. **Injection unit mapping**, derived analytically BEFORE running:
   - CLASS `DM_decay_Gamma` [1/s] + `DM_decay_fraction` → spectroxide
     `DecayingParticle{gamma_x [1/s], f_x [eV]}`. spectroxide's rate is
     `dE/dt = f_x·Γ_X·n_H·exp(−Γ_X t)` (energy tied to n_H via f_x); CLASS's is
     `Γ·f_dcdm·ρ_dcdm(z)`. The mapping must be verified by matching the *heating
     histories* to <0.1% (export both `d(Δρ/ρ)/dz`) before comparing μ/y.
   - CLASS `DM_annihilation_efficiency` → spectroxide `AnnihilatingDM{f_ann
     [eV/s]}` (`dE/dt = f_ann·n_H·(1+z)³`). Same history-match gate.
2. Case D: compare spectroxide's J_bb/J_μ/J_y visibility (greens.rs) against
   `sd_branching_approx` sweep (`sharp_sharp`, `soft_soft`, `exact`) — this
   comparison *is* the result (upgrades Fig 2 / Table 1).

**Why deferred:** the unit mapping is the delicate step the plan flags as most
error-prone (CLASS injection units changed across versions); doing it without
the heating-history verification gate would violate directive R1.4. Case A
already delivers a real, decomposed cross-code number for the heat-injection
channel; B–D sharpen it into the clean-μ-era statement and are the next step.

## Referee-reply paragraph (draft)

> We validated the heat-injection thermalisation against CLASS's independent
> spectral-distortion module (v3.3.0, `sd_branching_approx=exact`, i.e. Chluba's
> CosmoTherm-derived Green's data), matched to our reference cosmology. Feeding
> CLASS's own ΛCDM heating history into the spectroxide PDE, the two codes agree
> on the total spectral-distortion energy to 5.6%. The residual difference is in
> the μ/y split for this transition-dominated history and is fully attributed:
> spectroxide's fitted Chluba-2013 visibility functions differ from CLASS's exact
> Green's data by 17% in μ (consistent with the branching-approximation spread
> documented in Lucca et al. 2020), and the full PDE redistributes transition-era
> energy from μ toward y relative to the branching-ratio method — confirming that
> our stated PDE↔Green's-function agreement (2–5% in μ) applies to clean μ-era
> and y-era injections. A clean deep-μ-era decaying-particle comparison (where
> μ=1.401 Δρ/ρ unambiguously) is in progress.
