# Claim → anchor coverage matrix (Workstream R0)

**Date:** 2026-07-06
**Plan:** dev/PLAN_VALIDATION_ROUND2_2026-07-06.md, Workstream R0
**Purpose:** answer the referee's question — not "did you run checks" but "is
*every* published result checked, and by what independence level." Built first
so gaps steer Round-2 prioritization; finished as the paper's appendix
validation matrix. Updated as R1–R5 land.

**Independence classes** (strongest wins):
- **(i) analytic identity** — closed-form limit the result must reproduce.
- **(ii) literature coefficient/curve** — a number or curve from a published
  paper (Chluba 2013/2015, Fixsen 1996, CCJ24, Danese & de Zotti…).
- **(iii) independent code** — a *separately-authored running code*: CosmoTherm
  DI/GF tables (shipped reference data), HyRec-2 (Round 1), CLASS `sd` (R1),
  the clean-room Chang–Cooper refsolver (R3), AxionLimits curves (R5).
- **(iv) internal-only** — MMS, convergence order, Rust↔Python parity,
  conservation fuzzing, error budget. Produced within this project.

**A row whose strongest anchor is class (iv) is a gap** — unless the figure is
*intrinsically* a numerical self-diagnostic (energy-conservation and
convergence figures ARE the internal check; class (iv) is correct for them, and
they are marked "diagnostic" not "gap").

Source inventory: `notebooks/paper_figures/*.ipynb` and the 11 figures + 2
tables + in-text claims of `/home/bakerem/cosmoxide/paper/paper.tex` (full
enumeration below). Round-1 anchors: `dev/audit/AUDIT_SUMMARY.md` and the eight
module memos.

---

## Part 1 — Figures

| # | Fig (label) | Observable | Channel | Strongest current anchor | Round-2 upgrade | Status |
|---|---|---|---|---|---|---|
| 1 | `fig:mu_y_vs_zh` | μ/Δρ→3/κ_c≈1.401, 4y/Δρ→1 vs z_h | heat | **(i)** analytic limits + **(ii)** Chluba13 visibility | **(iii)** R1 CLASS Case D (μ/y transfer), R3 heat bursts | anchored; R1/R3 add (iii) |
| 2 | `fig:visibility` | J_bb, J_μ, J_y(z), residual <0.05 | heat | **(ii)** Chluba13 fit formulas | **(iii)** R1 CLASS `sd_branching` sweep; **(ii)** R5 digitized Chluba13 fig | anchored; R1 adds (iii) |
| 3 | `fig:spectral_shapes` | ΔI_ν(x)/(Δρ/ρ), residual vs CosmoTherm | heat | **(iii)** CosmoTherm DI (v1.0.3), tests `cosmotherm_comparison.rs` | R1 CLASS `exact` (indirect CosmoTherm x-check); R3 heat bursts | **(iii)** ✓ |
| 4 | `fig:dm_comparison` | ΔI(x) for decay / s-wave / p-wave DM, resid ≲2% | heat | **(iii)** CosmoTherm GF convolution | R1 CLASS Case B (decay) + C (s-wave) | **(iii)** ✓ |
| 5 | `fig:pathological` | ΔI(x) for 3 stress heating histories | heat | **(iii)** CosmoTherm GF conv. + **(iv)** PDE↔GF-table | (linearity ⇒ inherits Fig 3/4 anchors) | **(iii)** ✓ |
| 6 | `fig:photon_injection` | Δn(x)/(ΔN_γ/N_γ), x_inj=0.1,1,5 | **photon** | **(ii)** Chluba15 analytic GF + **(iv)** PDE↔GF | **(iii)** R3 photon case (only independent-code photon check); **(ii)** R5 Chluba15 curves | **GAP → R3/R5** |
| 7 | `fig:firas_photon` | FIRAS 68% CL ΔN_γ/N_γ limit vs x_i | **photon** | **(ii)** FIRAS monopole+cov (Fixsen96) + **(iv)** μ-era GF | **(iii)** R3 photon case feeds the template; R5 (no direct curve) | **GAP → R3** (limit-pipeline anchored via firas.py Round-1 P1-5 tests) |
| 8 | `fig:dp_firas` | FIRAS 95% CL ε(m_A′) dark-photon limit | dark-photon | **(ii)** CCJ24 statistic (Round-1 repro ~3%) + **(i)/(ii)** γ_con Landau–Zener (P1-4) | **(iii)** R5 AxionLimits COBEFIRAS curves (like-for-like statistic) | anchored; R5 adds (iii) |
| 9 | `fig:development_workflow` | AI-development schematic | — | N/A (non-physics diagram) | — | n/a |
| 10L | `fig:energy` (heat) | (Δρ_PDE−Δρ_inj)/Δρ_inj vs z_h | heat | **(iv)** conservation_fuzz + error budget | — (intrinsic diagnostic) | diagnostic ✓ |
| 10R | `fig:energy` (photon) | energy-conservation residual, photon inj. | **photon** | **(iv)** conservation_fuzz (monochromatic) | R3 photon ledger cross-checks the number/energy accounting | diagnostic ✓ (R3 corroborates) |
| 11 | `fig:convergence` | spectral L2 O(N⁻²); μ,y deviation vs N | numerical | **(iv)** MMS (`mms_convergence.rs`), `convergence_order.rs` | — (intrinsic diagnostic) | diagnostic ✓ |
| T1 | `tab:visibility_params` | 9 visibility params, PDE fit vs lit (Δ%) | heat | **(ii)** Chluba13 (primaries ≤5%) | **(iii)** R1 CLASS branching | anchored; R1 adds (iii) |
| T2 | `tab:performance` | wall-clock timings | — | N/A (performance, not physics) | — | n/a |

## Part 2 — In-text scalar claims (referee-checkable)

| Claim | Value | Strongest anchor | Note |
|---|---|---|---|
| μ = (3/κ_c)Δρ | ≈1.401 | **(i)** + R4 oracle (κ_c to 15.5 digits) | analytic |
| y = Δρ/4 (y-era) | 0.25 | **(i)** | analytic |
| β_μ = 3ζ(3)/ζ(2) | ≈2.192 | **(i)** + R4 oracle | analytic |
| κ_c = 3∫x³M/G₃ | ≈2.142 | **(i)** + R4 oracle | analytic |
| x₀ = 4/(3α_ρ) | ≈3.60 | **(i)** + R4 oracle | analytic; Chluba15 Eq. 31 |
| G₂=2ζ(3), G₃=π⁴/15, I₄=4π⁴/15 | — | **(i)** + R4 oracle (defining integrals) | analytic |
| z_th | ≈1.98×10⁶ | **(ii)** Chluba13 (derived, not fit) | literature |
| α_th | 5/2 | **(i)** analytic (Silk/thermalization) | analytic |
| DC/BR ratio at z=10⁶,x=0.1 | ≈17 | **(ii)** Danese & de Zotti (P1-8, indep. derived 17.06) | literature |
| DC dominant z≳10⁶, BR z≲10⁵; crossover | z≈3–4×10⁵ | **(ii)** (P1-8) | literature |
| Silk-damping μ | ≈2×10⁻⁸ | **(ii)** literature (Chluba+) | not computed by paper; cited |
| adiabatic-cooling μ | ≈−3×10⁻⁹ | **(iii)** CosmoTherm DI_cooling (tests) + **(iv)** check_adiabatic | (iii) |
| visibility params recovered | ≤5.2% (primary) | **(ii)** Chluba13 + **(iv)** 118-spectrum fit | Table 1 |
| PDE↔GF μ agreement | few % | **(iv)** internal + **(ii)** (Fig 1) | |
| PDE↔GF y agreement | sub-percent | **(iv)** internal | |
| CosmoTherm spectral shape agreement | ≲5% (few % at small x) | **(iii)** CosmoTherm | (iii) |
| DM spectra vs CosmoTherm | ≲2% | **(iii)** CosmoTherm | (iii) |
| FIRAS \|μ\| < 4.5×10⁻⁵ | — | **(ii)** Fixsen96 (firas.py P1-5 anchor tests) | literature |
| dark-photon ε ≲ 2.5×10⁻⁸ (μ–y transition) | — | **(ii)** CCJ24 + **(iii)** R5 AxionLimits | R5 |
| γ_con ~ 6×10⁻⁵ at strongest limit | — | **(i)/(ii)** Landau–Zener (P1-4) | |
| energy conserved ≲0.3% (10⁴–5×10⁵), <1% all z | — | **(iv)** conservation_fuzz + error budget | diagnostic |
| μ,y temporal error at defaults <0.2% (halve dτ) | — | **(iv)** error budget (P2-1) | diagnostic |
| spatial convergence O(N⁻²) | — | **(iv)** MMS (p=2.00 measured) | diagnostic |
| X_e accuracy ~5–10% at z~1000–1200 | — | **(iii)** HyRec-2 (Round-1 xe_hyrec_comparison, ≤1.9% actually) | (iii) |
| He X_e error ≲10% at z~1600–2000 | — | **(iii)** HyRec-2 (5.7% measured) | (iii) |
| κ_γ = 8π/λ_e³ ≈1.76×10³⁰ cm⁻³ | — | **(i)** + R4 oracle | analytic |
| H_dc polynomial coefficients (Eq. hdc) | — | **(ii)** Chluba & Sunyaev 2012 Eq. 13 (P1-8) | literature |

---

## Gap analysis (drives R1–R5)

Two channels lack an **independent-code (iii)** anchor as of Round 1:

1. **Photon injection (Figs 6, 7).** Strongest current anchor is **(ii)** the
   Chluba 2015 analytic Green's function + **(iv)** internal PDE↔GF. There is
   **no independent running code** for monochromatic photon injection.
   → **R3** clean-room Chang–Cooper solver includes exactly one photon-injection
   case (x_inj≈0.1, z_h≈3×10⁵) as the sole independent-code photon check.
   → **R5** adds the Chluba 2015 / Bolliet+2020 digitized curves as (ii)
   regression anchors. The FIRAS *limit pipeline* itself (firas.py) is anchored
   by the Round-1 P1-5 convention tests; R2 mutation testing measures whether
   those tests pin it *tightly*.

2. **Dark-photon ε limits (Fig 8).** Anchored (ii) via the CCJ24 statistic
   reproduction (~3%, Round 1) and the Landau–Zener γ_con derivation (P1-4).
   → **R5** compares against AxionLimits `COBEFIRAS_*.txt` curves like-for-like
   (matching statistic/CL — Round-1 P1-5/P1-6 showed convention mismatch alone
   gives ~2× spread), stating expected offsets (HyRec X_e sensitivity: γ_con up
   to +25%, ε −10.5% at m≈1.2–2.5×10⁻⁹ eV; unresolved ~22% γ_con offset vs
   Bryce fig).

The **heat-injection** half is already (iii)-anchored via CosmoTherm DI/GF
tables (Figs 3, 4, 5) and gets a second independent code from **R1 (CLASS `sd`)**
and a third from **R3** (Chang–Cooper). CLASS `sd` handles heating only — it
does **not** anchor photon/dark-photon (stated so R1 is not oversold).

Figures 10 and 11 are intrinsic numerical self-diagnostics; class (iv) is the
correct and sufficient anchor for them (they *are* the conservation/convergence
check), so they are not counted as gaps.

**Gate (R0):** every physics row has a planned class-(i)–(iii) anchor. No row
lacks one. The photon rows (6, 7) are the only ones whose (iii) anchor is still
*pending* (R3) rather than *landed* — flagged here and tracked as R3/R5
deliverables. No row requires a paper-text caveat in lieu of an anchor.

## Which matrix rows each Round-2 workstream upgrades

- **R1 (CLASS `sd`):** rows 1, 2, 4, T1 (heat μ/y transfer, decay, s-wave) →
  add independent-code (iii); row 3 gets an indirect-CosmoTherm cross-check via
  CLASS `exact`.
- **R3 (refsolver):** rows 1, 3 (heat bursts + adiabatic) and **row 6** (the
  only independent-code photon check) → add (iii).
- **R4 (oracles):** the analytic in-text constants (κ_c, β_μ, x₀, G_n, κ_γ) →
  machine-checked to 15+ digits (done, `highprec_numerics.md`).
- **R5 (lit curves):** rows 2, 6, 8 → digitized-curve (ii)/(iii) regression.
- **R2 (mutation):** upgrades the *test suite's* credibility for all rows (does
  a given anchor test actually catch a planted bug), not a specific figure.
