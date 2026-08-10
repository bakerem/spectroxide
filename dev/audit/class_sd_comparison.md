# CLASS `sd` cross-code comparison (Workstream R1)

**Date:** 2026-07-06
**Plan:** dev/PLAN_VALIDATION_ROUND2_2026-07-06.md, Workstream R1
**Deliverables:** `dev/scripts/class_sd_compare.py`, CLASS I/O under
`dev/output/class_sd/`, this memo.
**Upgrades coverage-matrix rows:** ~~1, 2, 3, 4, T1 (heat-injection →
independent code, class (iii))~~ — **withdrawn 2026-07-30.** Case A
double-counts adiabatic cooling (retraction inline below); it currently upgrades
nothing and finding **R1-A is retracted**. Rerun with
`--subtract-cooling` before restoring the class-(iii) claim.

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

> ### ⚠ RETRACTED 2026-07-30 — Case A double-counts adiabatic cooling
>
> **The Case A comparison above is invalid as run, and finding R1-A is
> withdrawn.** Do not quote any number in this section.
>
> **The defect.** The setup feeds CLASS's `_sd_heating.dat` verbatim into the
> spectroxide PDE and asserts the two codes then thermalise "the same heating
> history." They do not. CLASS's non-injection heating table *includes* the
> first-order adiabatic cooling of photons on baryons:
>
> ```c
> /* First order cooling of photons due to adiabatic interaction with baryons */
> class_call(noninjection_rate_adiabatic_cooling(pni, z_coarse, &dEdt), ...);
> pni->noninjection_table[index_z] += dEdt;      // external/heating/noninjection.c:197
> ...
> *energy_rate = -pni->heat_capacity*pni->H*pni->T_g;   // :313, [J/(m^3 s)]
> ```
>
> and that total is what reaches the file
> (`psd->dQrho_dz_tot[index_z] = heat*a/(H*rho_g);`, `source/distortions.c:862`).
> The spectroxide PDE, meanwhile, **always** models adiabatic cooling itself,
> through the Λρ_e electron-temperature term — it is not optional and cannot be
> switched off. So the cooling appears once in the tabulated source and once again
> in the solver's own physics. The spectroxide column above is a ΛCDM history plus
> a spurious second copy of the cooling.
>
> **Why the setup note "identical convention … verified by inspection" did not
> catch it.** Inspection checked the *sign and units* of the column, which are
> right. It could not catch a double count, because acoustic dissipation dominates
> the total at every z in the file — the cooling contribution never flips an entry
> negative, so there is no visible signature to inspect. The check needed was a
> term-by-term audit of what CLASS puts in the table, not a convention check on
> what comes out.
>
> **Magnitude — measured, 2026-07-30.** The cooling column reconstructed from
> CLASS's own formula (`class_adiabatic_cooling_dqdz` in
> `dev/scripts/class_sd_compare.py`) over the file's z range [1.02×10³, 5×10⁶]:
>
> | quantity | reconstructed cooling term |
> |---|---|
> | ∫ d(Δρ/ρ)/dz dz | −4.746×10⁻⁹ |
> | μ (via GF) | **−2.829×10⁻⁹** |
> | y (via GF) | −5.566×10⁻¹⁰ |
> | \|cool\|/total in the file at z = 10³/10⁴/10⁵/10⁶ | 0.42 / 0.15 / 0.17 / 0.19 |
>
> **The reconstruction is validated, not assumed:** μ_cool = −2.83×10⁻⁹ lands on
> the literature value for pure adiabatic cooling in ΛCDM, ≈ −2.7 to −3×10⁻⁹
> (Chluba 2011; Khatri, Sunyaev & Chluba 2012), which is also reproduced in-repo
> by `test_adiabatic_cooling_mu_vs_cosmotherm` against CosmoTherm's
> `DI_cooling.dat`. A units or normalisation slip would not land within 5% of an
> independently published number. `cooling_only_selfcheck()` re-runs this check on
> every invocation. Sanity scaling also holds: |cool| ∝ (1+z)⁻¹ to 3 digits
> between z = 10⁴ and 10⁶ where x_e is flat, breaking only below recombination
> where x_e falls.
>
> The PDE−GF gap that R1-A was built on is 5.6×10⁻⁹ in μ (1.049 vs 1.604×10⁻⁸).
> At −2.83×10⁻⁹, **the double count is 51% of it. Half of what was written up as
> physics is bookkeeping.**
>
> **What survives, and what is now unexplained.**
> - The **total-distortion-energy agreement to 5.6%** is also contaminated and
>   must be re-derived, not quoted.
> - The **branching-approximation spread** result (item 2 above: spx-GF vs CLASS,
>   μ 17% low) is *unaffected* — both sides of that comparison use the same
>   history through the same class of method, so the double count enters both
>   identically and cancels in the ratio.
> - **First-order estimate of the corrected comparison** (linear, pending the
>   actual rerun): removing one copy of the cooling *raises* μ by 2.83×10⁻⁹ and y
>   by 5.57×10⁻¹⁰, giving μ_spx ≈ 1.33×10⁻⁸ (gap to spx-GF shrinks 34% → 17%;
>   ratio to CLASS 0.54 → 0.69) and y_spx ≈ 5.20×10⁻⁹.
> - **The y discrepancy therefore gets worse, and is unexplained.** Ours is
>   already high (4.64×10⁻⁹ vs spx-GF 3.64, CLASS 3.45). The correction pushes it
>   to ≈5.20×10⁻⁹ — from +34% to +51% against CLASS. The double count was
>   *masking* part of the y excess, not causing it. Whatever drives it is a
>   separate effect and is still open. **This is the one genuinely unresolved
>   item to come out of R1.**
>
> **The fix, and why the rerun is not in this memo.**
>
> ```bash
> python dev/scripts/class_sd_compare.py --case A --subtract-cooling
> ```
>
> `--subtract-cooling` reconstructs CLASS's own
> `noninjection_rate_adiabatic_cooling` on the file's z grid and removes it before
> handing the table to the PDE, leaving only the terms the PDE does not already
> model; it also runs `cooling_only_selfcheck()` against the literature μ and
> writes `comparison_nocool.json`. Running it without the flag now prints a
> warning naming this retraction. The reconstruction has been verified (table
> above); the corrected *comparison* has not been run, because that is a fresh
> measurement rather than a correction to a record — the numbers above are left
> standing under the retraction banner rather than silently overwritten.
>
> **Until that rerun exists, Case A is not an anchor for anything** and the
> coverage matrix must not count it as class (iii) for rows 1–4/T1.
>
> **The transferable lesson.** The setup note said "identical convention …
> verified by inspection of both," and that was true and useless. Convention
> checks (sign, units, variable) do not detect *term-content* mismatches. When one
> code hands a source term to another, the question is not "are the units the
> same" but "which physical processes are inside this number, and does the
> receiving code already model any of them?" Any process the receiver models
> unconditionally — here adiabatic cooling via Λρ_e — is a double-count hazard by
> construction, and the hazard is invisible whenever a larger term of the same
> sign dominates the column. Every future cross-code hand-off in this project must
> enumerate the terms on both sides before comparing numbers.

## Corrected Case A rerun (2026-08-04) — `--subtract-cooling`

The rerun the retraction called for. Same setup, CLASS's adiabatic-cooling
term reconstructed and removed before the hand-off (self-check: cooling-only
μ = −2.829×10⁻⁹ vs literature ≈ −3×10⁻⁹). PDE: z ∈ [4.9×10⁶, 1021],
n=4000. Raw solver output preserved in `A_spx_full_nocool.json`; summary in
`comparison_nocool.json`.

| Method | μ | y |
|---|---|---|
| CLASS `sd` exact (reported amplitudes) | 1.931×10⁻⁸ | 3.453×10⁻⁹ |
| spectroxide PDE (cooling-corrected) | 1.280×10⁻⁸ | 5.581×10⁻⁹ |
| CLASS exact spectrum, decomposed with OUR least-squares (converged) | 1.374×10⁻⁸ | 5.146×10⁻⁹ |

**The y "excess" (and most of the μ deficit) is decomposition convention,
not physics.** CLASS reports branching-ratio amplitudes and relegates
non-μ/non-y transition-era shapes to PCA residuals (`SD[e_0]`, `SD[e_1]` in
`_sd_distortions.dat`; e_0 is 23% of the spectrum's L2). An independent
verification pass confirmed the shape conventions are identical (our
decomposition applied to CLASS's *pure* μ and y columns recovers CLASS's
reported amplitudes exactly) and that the residual column alone projects
onto (μ, y) = (−6.47×10⁻⁹, +2.24×10⁻⁹) in our basis — the bulk of the
apparent μ deficit and y excess against CLASS's reported numbers.

**Quantitative caution (claim-verifier, 2026-08-04):** decomposing CLASS's
66-point tabulated spectrum on its own grid gives μ = 1.290×10⁻⁸,
y = 5.653×10⁻⁹ — within ~1% of the PDE — but that agreement is
quadrature-limited, not real: spline-refining the same spectrum converges to
μ = 1.374×10⁻⁸, y = 5.146×10⁻⁹. The honest same-convention comparison is
**PDE vs CLASS: μ −5.8%, y +6.2%** (both decomposed identically at
convergence). That residual ~6% is the genuine PDE-vs-exact-branching
difference for this transition-dominated history (real μ↔y redistribution
the per-dz branching cannot capture, plus fit-band choices) — versus the
naive −34%/+62% against CLASS's reported amplitudes.

**Pointwise spectrum agreement:** interpolating the PDE's final Δn(x) onto
CLASS's frequency grid, the two codes' total distortion spectra agree to
2.7% in L2 (2.4% with x³ intensity weighting, which is the more meaningful
metric — unweighted Δn L2 is dominated by the lowest-x points). Median
pointwise 3.2%, max 5.5% where |Δn| > 5% of peak (max at zero crossings,
where relative error inflates).

**Known minor issue found during verification:** `_band_trap_weights`
(`python/spectroxide/greens.py:1573`) uses full-width instead of half-width
trapezoid weights at the first/last *global* grid points. Harmless when the
grid extends beyond the [0.5, 18] fit band (the PDE grid does); a ~0.4%
μ bias when decomposing externally tabulated spectra whose grid starts
inside the band (as CLASS's does).

**Branching-approximation sweep (same ΛCDM run):** CLASS's own internal
spread is larger than any spectroxide-vs-CLASS difference in method class:

| CLASS branching | μ | y |
|---|---|---|
| sharp_sharp | 1.650×10⁻⁸ | 3.496×10⁻⁹ |
| sharp_soft | 1.552×10⁻⁸ | 3.496×10⁻⁹ |
| soft_soft / soft_soft_cons | 1.639×10⁻⁸ | 3.641×10⁻⁹ |
| exact | 1.931×10⁻⁸ | 3.453×10⁻⁹ |
| spectroxide GF (fitted Chluba-2013) | 1.604×10⁻⁸ | 3.641×10⁻⁹ |

Our GF sits 2.1% below CLASS `soft_soft` in μ and matches its y to four
digits (both use the same fitted visibility formulas — agreement essentially
by construction). The 17% GF-vs-`exact` μ gap equals CLASS's own
exact-vs-analytic-approximation spread and is not a spectroxide defect.

**What this upgrades:** the heat-injection channel now has a genuine
class-(iii) independent-code anchor: same heating history, cooling hand-off
audited term-by-term, spectra compared pointwise (2.7% L2, 2.4%
intensity-weighted), and amplitudes compared under a single converged
decomposition convention (~6% in μ and y for this transition-dominated
history; 1.5% in μ for the clean deep-μ Case B below). Rows 1–4/T1 of the
coverage matrix can be restored on this basis.

## Case B — clean deep-μ-era decay (2026-08-04) — run

`dev/scripts/class_sd_case_b.py`. Γ_X = 4.2×10⁻⁸ s⁻¹ (Γ·t = 1 at z = 10⁶),
f_x = 100 eV → Δρ/ρ = 6.393×10⁻¹¹. **Unit-mapping gate passed:** CLASS
`DM_decay_fraction` = f_x[J]·(1−Y_p)·Ω_b / (m_H **c²** Ω_cdm) = 1.649×10⁻⁸
(the derivation in `run_case_decay`'s comment had dropped the c² — CLASS's
`rho_cdm` in `injection.c` is an *energy* density; the history gate caught
it). With `sd_only_exotic = yes`, CLASS's exotic-only heating column matches
spectroxide's analytic decay history to **0.06% max deviation** (0.1% gate),
integrals to 0.05%.

| Method | μ | μ/(1.401 Δρ/ρ) |
|---|---|---|
| CLASS exact | 6.988×10⁻¹¹ | 0.780 |
| CLASS soft_soft | 6.964×10⁻¹¹ | 0.778 |
| CLASS sharp_sharp | 8.263×10⁻¹¹ | 0.923 |
| spectroxide GF | 6.792×10⁻¹¹ | 0.758 |
| **spectroxide PDE** | **6.884×10⁻¹¹** | **0.769** |

y ≈ 0 in all methods, as required. In the clean μ-era case the GF-vs-CLASS
`exact` gap collapses from 17% (transition-dominated Case A) to **2.8%**,
and the **PDE lands within 1.5% of CLASS `exact`** (ratio 0.985; 1.4% above
our own GF) — confirming the Case A μ split is transition-era branching
ambiguity. The μ/(1.401Δρ) ≈ 0.78 common factor is the blackbody visibility
J_bb at z_X = 10⁶ (partial thermalization), consistent across codes.

PDE leg method: the solver always includes ΛCDM adiabatic cooling, which at
f_x = 100 eV (Δρ/ρ = 6.4×10⁻¹¹) swamps the injection (total μ = −2.23×10⁻⁹).
Two runs at f_x = 100 and 10⁶ eV, differenced linearly, isolate the
injection-only distortion (both amplitudes are deep in the linear regime).
Raw outputs: `B_spx_full.json`, `B_spx_full_f1e6.json`; summary
`comparison_case_b.json`.

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

## Referee-reply paragraph (draft) — ⚠ DO NOT USE, built on the retracted run

**Superseded 2026-07-30.** Every number in the draft below traces to the
double-counted Case A configuration. It must not be sent to the referee in this
form. Two sentences are salvageable and two are not:

- *Salvageable:* the branching-approximation spread (spectroxide's fitted
  Chluba-2013 visibilities vs CLASS's exact Green's data, μ 17% low, consistent
  with Lucca et al. 2020) — the double count cancels in that ratio.
- **Not salvageable:** "agree on the total spectral-distortion energy to 5.6%"
  and "the full PDE redistributes transition-era energy from μ toward y." The
  first is contaminated; the second is roughly half an artefact, and the y half
  of it points the wrong way once corrected.

Rewrite only after `--subtract-cooling` has been run. Given that the corrected
y excess is *larger*, the honest reply may need to state an open discrepancy
rather than a clean agreement — which is the right thing to tell a referee who
asked for critical evaluation of the validation.

<details><summary>Retracted draft (kept for the record)</summary>

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

</details>
