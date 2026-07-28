# Physics-check gap audit — status & running log

**Started:** 2026-07-26. **Last update:** 2026-07-27. **Skill:** `/audit`.
Resume with `/audit resume dev/audit/PHYSICS_CHECKS_STATUS_2026-07-26.md`.

## Goal

Find *new* physics anchors (class (i) analytic / (ii) literature) that the suite
does not currently assert, after the Round-1 module audits and the Round-2
mutation close-out have been mined. Done = each candidate is either implemented
as a test with a measured number, or rejected with a reason.

Not a repeat of `coverage_matrix.md` (which grades *published figures* by anchor
class) nor of `R2_WRAPUP_TODO.md` §7 (whose P1–P4 are all ✅ in the working
tree). This pass asks the complementary question: **which exact identities does
the physics obey that nothing in `tests/` currently checks?**

## Scope

- **In:** `kompaneets.rs`, `greens.rs` (photon path), `spectrum.rs`,
  `cosmology.rs` + `recombination.rs` (the τ/y_γ integrand), `double_compton.rs`,
  `bremsstrahlung.rs`, `electron_temp.rs`.
- **Out:** validation-guard mutants (declared out of scope in `R2_WRAPUP_TODO.md`
  §8), Python `firas.py` mutation triage (same), R1 CLASS and R3 refsolver
  (separate workstreams, tracked in `ROUND2_STATUS.md`).

All new tests live in **`tests/physics_identities.rs`** except the bump-drift
characterisation, which needs crate-private helpers and sits in
`src/greens.rs::tests`.

## Findings

| ID | Sev | Description | Evidence |
|---|---|---|---|
| **F-PC-1** | LOW | The Arsenadze log-normal bump has the wrong y→0 mean-frequency drift. Exact linearised Kompaneets: `d⟨x⟩/dy = x[4 − x·coth(x/2)]`, zero at x = 3.8300. Code (transcribed exactly from arXiv:2409.12940): `x[4 − 2f_cs(x) − x]`, zero at **3.5889**. Max deviation of the fractional drift 0.2358 at x = 2.79 (peak absolute 0.744 at x = 3.56); **sign wrong for 3.589 < x < 3.830**. Exactness would require `f(x) = x/(e^x−1)`; the paper uses `e^{−x}(1+x²/2)`, correct only at x→0 and x→∞. **Not a code bug and not to be "fixed"** — see decision below. | `src/greens.rs::test_broadened_bump_first_moment_vs_kompaneets` |
| **F-PC-2** | LOW | **FIXED.** `test_photon_survival_regime_structure` closed with a disjunction (`xc_mid < xc_low \|\| xc_mid < xc_high`), which either half satisfied alone. Now a conjunction; both branches hold (x_c = 4.33×10⁻², 6.39×10⁻³, 8.69×10⁻³ at z = 10⁴, 2×10⁵, 2×10⁶). | `tests/heat_injection.rs` |
| **F-PC-3** | LOW | The in-text claim "DC/BR crossover z ≈ 3–4×10⁵" is **x-dependent** and holds at the P1-8 reference point but not generally: measured z_cross = 3.07×10⁵ at x = 0.1, **2.14×10⁵ at x = 1**. Paper-text precision issue, no code defect. **Flag to EB**, no paper edit (same handling as R4-1 / R1-A). | `test_dc_br_crossover_redshift` |
| **F-PC-4** | — | *Physics insight, not a defect.* The quasi-stationary T_e **returns the Kompaneets drift energy immediately as a y-distortion**: for a bump at x′ = 1 the photon energy moves by +0.014% where the drift alone predicts +6.47%, because the electron heat capacity is ~10⁻⁹ of the photons'. Consequences: (a) the first-moment identity is *unobservable* through the full solver and must be pinned at kernel level with θ_e = θ_z frozen; (b) the original T-PC-5 design (read y_c off the bump broadening at x′ = 3.83) is invalid — it measured y_c **28% low**, because the drift *shear* A′(x) < 0 at the fixed point compresses the bump and cancels part of the diffusive spreading. Both are now recorded in the test docs so the mistake is not repeated. | `test_quasistationary_te_returns_bump_energy` |

### F-PC-1 decision: keep the published form, pin the deviation

The question is not whether the paper's `f` is exact but whether it matters.
Measured by swapping `f_cs → x/(e^x−1)` and re-running the photon Green's
function (`scratchpad/fcs_impact.py`, L2 on x³G over x ∈ [0.01, 30]):

| z_h | y_γ | ΔL2(x³G) | Δμ |
|---|---|---|---|
| 2×10³ | 5.8×10⁻⁵ | ≤0.05% | <0.001% |
| 3×10⁴ | 3.9×10⁻² | **0.44–0.83%** | <0.001% |
| 2×10⁵ | 1.89 | ≤0.001% | <0.001% |
| 10⁶ | 47.7 | ~0 | <0.001% |

The effect peaks where y_γ ≈ 0.04 and dies at both ends — at large y_γ the
√(1+x′y_γ) and (1+x′y_γ) denominators suppress the f-dependence. So: no
published number moves, the code stays faithful to its citation, and the new
characterisation test locks the *deviation* (fixed point 3.5889, max fractional
drift error 0.2358) so a future edit to `f_cs` cannot be silent.

## Plan — candidate checks

Cost: **cheap** = pure function eval. **med** = one/few y-era runs. **exp** =
deep-μ-era runs.

| ID | Cost | Class | Check | Status |
|---|---|---|---|---|
| T-PC-1 | cheap | (ii) | **Thomson depth / last scattering.** τ(z) = 1 at z = **1090.69** (planck2018) / 1089.54 (default) vs Planck 2018 z_* = 1089.92 ± 0.25 → **0.07%**. Visibility peak z = 1079.60 at τ = 0.877, FWHM Δz = 203. Anchors X_e·n_e·σ_T·c/[H(1+z)] — the integrand behind y_γ and P_s (coverage rows 6/7). Band ±12 = reionization-convention ±7 (0.054/(dτ/dz)) ⊕ 15% integrand sensitivity (a +10% integrand moves z_* by −8.05). | ✅ DONE |
| T-PC-2 | cheap | (i) | **Six exact shape moments.** ∫x²G_bb = 3G₂ and ∫x³G_bb = 4G₃ (⟹ ΔN/N = 3ΔT/T, Δρ/ρ = 4ΔT/T) to 1.4×10⁻¹³; ∫x²M = **0** (β_μ's defining condition) to 5×10⁻¹³ relative; ∫x³M = 8ζ(2)ζ(4)/ζ(3) − 6ζ(3) = (κ_c/3)G₃ = 4.636351292964 to 12 digits (this is the identity behind μ = 1.401Δρ/ρ); ∫x²Y = **0** to 1.4×10⁻¹⁵ relative; ∫x³Y = 4G₃ (⟹ Δρ/ρ = 4y). Previously only the *constants* β_μ, κ_c were anchored, never the coded shapes. | ✅ DONE |
| T-PC-3 | cheap | (i) | **Kompaneets moment identities**, on the production flux split with θ_e = θ_z: dN/dy = 0 (measured ≤1.6×10⁻¹⁵), d⟨x⟩/dy = 4⟨x⟩ − ⟨x²coth(x/2)⟩ and d⟨x²⟩/dy = 10⟨x²⟩ − 2⟨x³coth(x/2)⟩, integrated along the trajectory. Agreement **0.002% / 0.026% / 0.16% / 0.025%** at x′ = 1, 3, 3.83, 5. Only anchor on the drift flux φ(2n_pl+1)Δn — conservation laws hold for *any* antisymmetric split. Includes the sign flip about x′ = 3.8300 (the Y_SZ zero crossing reached from the first-moment side). | ✅ DONE |
| T-PC-6 | cheap | (i) | **H-theorem, value and sign.** H = ∫x²[n ln n − (1+n)ln(1+n) + xn]dx obeys dH/dy = −∫x⁴n(1+n)(ψ′)², ψ = ln[n/(1+n)] + x. Monotone over 20 checkpoints and ΔH/predicted = **0.99982**. Fixes the *relative* weight of the diffusion and drift pieces, which conservation cannot. | ✅ DONE |
| T-PC-5 | med | (i) | **Redesigned** (see F-PC-4): quasi-stationary T_e returns the bump energy. Photon Δρ/ρ conserved to **0.014%** against a 6.47% drift-only prediction; ρ_e − 1 = −2.10×10⁻⁵ vs the number-weighted moment prediction (ΔN/N)(G₂/4G₃)⟨x[x coth(x/2) − 4]⟩ = −1.74×10⁻⁵ (ratio 1.21, band 30% for the bump's evolution during the run). | ✅ DONE |
| T-PC-7 | cheap | (ii) | **DC/BR crossover redshift** solved from K_DC = K_BR: **3.07×10⁵** at x = 0.1 vs the in-text 3–4×10⁵ (see F-PC-3 for the x-dependence). | ✅ DONE |
| T-PC-8 | med | (iv→i) | **Artificial-boundary independence.** At matched N = 4400, refining x_min ×3.3 moves μ by +0.137% and y by −0.224%; extending x_max to 100 moves them by −0.158% / +0.265%; Δρ/ρ moves <10⁻⁶ in both. For comparison the *resolution* step N = 4000→4400 alone moves μ/y by +0.542%/−0.879%, so the boundary systematic is the smaller of the two and the production choice is converged. First quantification of this systematic. | ✅ DONE |
| T-PC-4 | exp | (i)+(ii) | **α_th = 5/2 from the PDE.** Fitted **α_th = 2.4199** over z_h ∈ {2, 2.5, 3}×10⁶ against the analytic 5/2 and the Chluba J_bb* fit's own local slope 2.4700 over the same window. τ_eff = 1.109 / 1.899 / 2.960 (fit: 1.082 / 1.876 / 2.947). The z_h = 3×10⁵ calibration run divides out the constant PDE↔GF μ offset (measured **+2.65%**), which would otherwise bias α by ~δ/τ ≈ 0.1. Band 2.5 ± 0.15. `#[ignore]`d; runtime **339 s**. | ✅ DONE |
| T-PC-9 | med | (i)/(ii) | **T_e Compton/adiabatic balance.** ρ_e − 1 = −H/(Γ_C + H), Γ_C = (8σ_T u_γ)/(3m_e c)·n_e/(n_e+n_H+n_He) (Seager+1999, CS2012 Eq. 15–18). Solver/analytic ratio **5.45 / 1.775 / 1.121 / 1.0042 / 0.9763** at z = 10⁵ / 3×10⁴ / 10⁴ / 3×10³ / 1.5×10³. The balance term falls as z⁻² while the accumulated adiabatic-cooling *distortion* feeds back through δρ_eq, so it only dominates at low z: assertion is 3% at z ≤ 3×10³ plus a monotone-approach check. **DC/BR is not the residual** — disabling it moves every ratio by <0.1%. Fixes Γ_C, and hence the adiabatic-cooling μ ≈ −3×10⁻⁹, with O(1) sensitivity; previously anchored only internally and via CosmoTherm. | ✅ DONE |

## Sensitivity-directed pass — photon path (2026-07-27)

Method (the R2/P1 lesson generalised): **map ∂ln(observable)/∂ln(parameter)
first, then place the test where the derivative is O(1)** — do not tighten a
band at a point where the derivative is 0.01. Harness:
`scratchpad/r3/photon_sensitivity.py` (central difference, ±2%, monkeypatched
coefficients on the Python path, which is parity-checked against Rust).

Measured ∂ln μ/∂ln(x_c coefficient), analytic μ-era response:

| z_h | x_inj = x_c | x_inj = 0.1 | x_inj = 1 | x_inj = 5 |
|---|---|---|---|---|
| 10⁴ | **−1.034** (BR) | −0.452 | −0.061 | +0.022 |
| 3×10⁴ | **−1.013** (BR) | −0.214 | −0.029 | +0.011 |
| 2×10⁵ | −0.823 (BR) | −0.054 | −0.007 | +0.003 |
| 10⁶ | **−0.910** (DC) | −0.060 | −0.008 | +0.003 |
| 2×10⁶ | **−0.987** (DC) | −0.088 | −0.012 | +0.004 |

Analytically, with u ≡ P_s x₀/x_inj: ∂ln μ/∂ln P_s = −u/(1−u) and
∂ln P_s/∂ln x_c = −x_c/x_inj, so ∂ln μ/∂ln x_c → −x_c/x_inj for u ≫ 1 — O(1)
exactly at x_inj ≈ x_c, and the two coefficients separate cleanly (BR below
z ~ 10⁵, DC above). At x_inj ≥ 1, where the rest of the photon suite sits, a
50% coefficient error moves μ by <3%: the same blind spot that let K_DC be
wrong by 1.535×.

**Gap this closed.** Every pre-existing x_c test asserted an *ordering*
(`x_c_dc > x_c_br` at high z) or a *bound* (`P_s > 0.99`); none asserted a
value, so 8.60×10⁻³, 1.23×10⁻³, 0.5 and −0.672 were unpinned — consistent with
the mutation campaign finding these survivors.

Three tests added to `tests/physics_identities.rs`:

| ID | Class | Check |
|---|---|---|
| T-PS-1 | (i) | `P_s(x_c, z) = 1/e` to <10⁻¹⁴ at six redshifts — ties `photon_survival_probability` to `x_c` (Chluba 2015 Eq. 24) |
| T-PS-2 | (ii) | `x_c^DC`, `x_c^BR`, `x_c` vs Eqs. 25a/25b re-evaluated in the test, rel <10⁻¹², at five redshifts |
| T-PS-3 | (i)+(ii) | μ/(ΔN/N) for injection at the *literature* x_c against the closed form α_ρ x_c (3/κ_c) J_bb* J_μ (1 − x₀/(e x_c)), plus the μ < 0 sign guard, at z_h = 10⁴ (BR) and 2×10⁶ (DC) |

**Kill-power verified, not assumed.** Planting +10% on `x_c_dc` and separately
on `x_c_br` in `src/greens.rs`: T-PS-2 and T-PS-3 both FAIL for both plants;
T-PS-1 correctly stays green (a pure rescaling of x_c preserves P_s(x_c) = 1/e
by construction — that is its role). Source restored bit-identical (md5
verified); `--lib`, `greens_function_checks`, `cosmotherm_comparison`,
`coverage_gaps`, `physics_identities` all green; clippy clean.

**Next target from the same map (not yet done).** The Arsenadze broadening
helpers and y_γ are O(1)-observable on the full photon Green's function at
z_h = 3×10⁴, x_inj = 1: ∂ln μ/∂ln = **−2.03** (`_y_compton`), **−2.33**
(`_beta_cs`), +0.55 (`_alpha_cs`), −0.37 (`_f_cs`). These are the least-pinned
functions per the mutation campaign, and that (z_h, x_inj) point is where a
value anchor would bite. Caveat for whoever picks this up: the decomposition
band is x ∈ [0.5, 18], so sensitivities computed with x_inj ≪ 0.5 are
meaningless (P1-7 out-of-span L² best fit), which is what produced the
spurious |∂ln y/∂ln x_c| ~ 10²–10³ entries in the first run of the harness.

## Rejected / already covered (do not re-propose)

- BE stationarity under Kompaneets → `test_kompaneets_preserves_bose_einstein`.
- Planck is a fixed point → `test_pde_planck_is_stable_equilibrium`.
- Pure temperature shift is the zero mode → `science_high_z_thermalization_is_temperature_shift`.
- Photon-number conservation under pure Compton → `fuzz_photon_number_conservation_pure_compton`.
- x₀ = 3.60 balanced-injection zero μ → `test_photon_gf_balanced_injection_zero_mu`, `test_x_balanced_from_first_principles`.
- σ²(ln x) = 2y_γ broadening variance → `greens.rs::test_compton_broadening_identities` (R2 P3). Its ⟨x⟩ = x′·f_int assertion is internal self-consistency, not physics — that gap is F-PC-1/T-PC-3.
- DC/BR detailed balance, τ_ff anchors, DC relativistic correction, He-epoch X_e, `interp_2d`, `distortion_from_*` → R2 close-out P4/P2/B1/B4/B2/B3, all ✅ in the working tree.
- Y_SZ zero crossing x = 3.831 → `test_y_sz_zero_crossing_from_transcendental_equation`; T-PC-3 reaches the same root from the drift side.
- **Reading y_c off the bump broadening** (original T-PC-5) → invalid, see F-PC-4.
- **Two-run y-self-similarity** to pin the dτ↔z mapping → dropped. The plan claimed no test can see a mapping error; that was too strong. The CosmoTherm comparisons (class iii) do bound it, since thermalization depth depends on ∫dτ, and T-PC-1 now anchors the same integrand against z_*. A self-similarity run would also be broken by the adiabatic term, which is not self-similar in y.

## Environment / gotchas

- 7 GB box: never run heavy Rust builds concurrently (`env-7gb-oom-serialize-builds`).
  Release-only tests, detached via `run_in_background` (`detached-jobs-run-in-background`).
- One deep-μ-era PDE run (z_start = 4.5×10⁶ → 500, production grid) takes
  **153 s**. Budget T-PC-4 at ~7 min.
- Rust format strings have no `f` trait: `{x:.3f}` fails to compile, use `{x:.3}`.
- `kompaneets_rhs` / `kompaneets_tridiagonal` are `#[cfg(test)]` (crate-internal).
  The public one-step entry usable from `tests/` is
  `kompaneets_step_nonlinear(grid, dn, θ_e, θ_z, dτ)` — pass θ_e = θ_z to freeze
  the electron temperature and get the clean linearised operator.
- Python: `/home/bakerem/miniforge3/bin/python`, numpy 1.26.4 / scipy 1.15.0;
  `sys.path.insert(0,'python')`. `greens._y_compton` and
  `_photon_survival_probability_numerical` take a **dict** — use
  `Cosmology.default().to_dict()`.
- Working tree carries the *uncommitted* R2 close-out tests (P1–P4, B1–B4).
  `git status` before assuming any test is absent.

## Status: all nine checks implemented and green (2026-07-27)

`cargo clippy --all-targets -- -D warnings` clean. New file
`tests/physics_identities.rs` (8 tests + 1 ignored) plus one characterisation
test in `src/greens.rs::tests` and the F-PC-2 one-line fix in
`tests/heat_injection.rs`. CLAUDE.md test inventory updated.

## Next action

1. **Flag F-PC-3 to EB** (the in-text DC/BR crossover claim is x-dependent:
   3.07×10⁵ at x = 0.1 but 2.14×10⁵ at x = 1). No paper edit was made, matching
   how R4-1 and R1-A were handled.
2. Optional tightening, in value order:
   - **T-PC-1 → class (iii):** re-run z_* with a like-for-like HyRec-2
     recombination history (`dev/audit/xe_hyrec_comparison.md` has the setup) to
     replace the ±12 convention band with a real ±1 comparison. This is the
     single biggest available upgrade to the τ/y_γ integrand behind coverage
     rows 6/7.
   - **T-PC-4 lever arm:** add z_h = 3.5×10⁶ (τ ≈ 4.3) to halve the α_th
     uncertainty; +~3 min.
3. Still open from the earlier plan, unchanged by this audit: R3 refsolver
   (`ROUND2_STATUS.md`) remains the only planned independent-code anchor for the
   photon channel, and R1 CLASS Cases B–D for the heat channel.
