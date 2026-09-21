# Rust physics-module doc-comment review

Scope: `//!` and `///` comments only, in `src/kompaneets.rs`, `double_compton.rs`,
`bremsstrahlung.rs`, `electron_temp.rs`, `recombination.rs`, `constants.rs`, `cosmology.rs`,
`spectrum.rs`, `grid.rs`, `dark_photon.rs`, `axion.rs`. Judged against `rubric.md` (Google
developer style guide highlights + linked pages). Independent pass: no git history, no
`dev/PLAN_*`/`dev/DOCS_STYLE_STATUS.md`/`dev/scripts/docs_style_lint.py`, no other reviewer's
report read.

## Summary table

| Severity | Count |
|---|---|
| HIGH | 0 |
| MEDIUM | 129 |
| LOW | 29 |
| DOMAIN | 2 explicitly listed (illustrative; the broader arrow/`~`/`×` pattern is not exhaustively enumerated, see Systematic patterns §3) |

No HIGH findings: no undefined abbreviation central to a page, no missing module intro, no
unexplained placeholder, no broken procedure order, no "here" link text, no undocumented
parameter, and no procedure using the wrong list type.

## Systematic patterns

1. **F11 — em dash with surrounding spaces.** Every em dash in this corpus is written
   `" — "` (space, em dash, space). Google style specifies no spaces around the em dash. 43
   occurrences across 9 of the 11 files (`grid.rs` and `dark_photon.rs` have none). Counted
   once per file below (first 10 shown where a file has more, with the file total).
2. **A2 — function/method summary is a noun phrase, not a present-tense verb.** The large
   majority of `///` summaries on functions (not structs, not constants) state the quantity
   the function returns ("DC Gaunt factor in the soft photon limit...", "Ω_b = ω_b / h².")
   rather than what the function does ("Computes...", "Returns..."). This is both the Google
   A2 rule and the Rust API Guidelines' own convention for function docs, so it is not a
   PEP‑257-style DOMAIN exception. 83 instances total: 59 on `pub fn`, 10 on private
   functions, 14 on `#[test]`/`miri_*` functions (test-rationale titles, lower impact, listed
   as LOW). Breakdown by file: `cosmology.rs` 26, `bremsstrahlung.rs` 13, `spectrum.rs` 8,
   `recombination.rs` 10, `double_compton.rs` 10, `dark_photon.rs` 4, `axion.rs` 2,
   `kompaneets.rs` 6, `electron_temp.rs` 2, `grid.rs` 2, `constants.rs` 0 (constants are
   correctly documented as noun phrases, which is the right convention for a value, not a
   function). Some instances are physics-formula restatements (e.g. `Ω_m = Ω_b + Ω_cdm.`)
   that are arguably self-documenting, but the rule as stated is still violated; severity kept
   at MEDIUM rather than DOMAIN because Rust has no community carve-out for this the way
   Python/NumPy docstrings do.
3. **DOMAIN — reaction arrows, `~`, `×`, `↔` in physics notation.** `double_compton.rs`,
   `bremsstrahlung.rs` module titles ("γ + e → γ + γ + e", "e + ion → e + ion + γ"),
   `dark_photon.rs`/`axion.rs` ("γ ↔ A'", "γ ↔ a"), and order-of-magnitude "~" throughout are
   T11-flagged symbols but match the rubric's explicit DOMAIN carve-out for standard physics
   notation. Not exhaustively listed per file.
4. **T9 — "above"/"below" as document-position references** inside doc comments referring to
   another item earlier/later in the same file ("the test above", "(below)"): 6 instances
   across `electron_temp.rs`, `double_compton.rs`, `bremsstrahlung.rs`, `recombination.rs`.
5. **F1 — file paths and `CLAUDE.md` referenced without code font**, sometimes inconsistently
   (the same path is backticked in one place and not in another in `recombination.rs`): ~9
   instances across `kompaneets.rs`, `double_compton.rs`, `spectrum.rs`, `recombination.rs`,
   `bremsstrahlung.rs`.

## kompaneets.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 232–233 | A1/A2 | MEDIUM | `Factorizes once, solve two right-hand sides against the SAME tridiagonal matrix.` | Mixed verb mood ("Factorizes" then "solve"). Rewrite: "Factorizes once and solves two right-hand sides against the same tridiagonal matrix." |
| 232 | F6 | MEDIUM | `Factorizes once, solve two right-hand sides against the SAME tridiagonal` | All caps used for emphasis. Use italics or restructure: "...against the same tridiagonal matrix (shared with `thomas_solve_inplace`)." |
| 360 | F6 | MEDIUM | `Performs one implicit step of the NONLINEAR Kompaneets equation on Δn.` | Drop the caps: "Performs one implicit step of the nonlinear Kompaneets equation on Δn." |
| 1321 | F1 | LOW | `Guards the analytic Planck cancellation in \`kompaneets_rhs\` (CLAUDE.md` | Put the filename in code font: `` `CLAUDE.md` ``. |
| 1325 | F6 | MEDIUM | `zero BEFORE any finite differences touch n_pl. A naive flux that kept` | Remove the caps emphasis: "zero before any finite differences touch n_pl." |
| 1333 | F6 | MEDIUM | `(the (φ−1) source is the ONLY nonzero piece).` | "...is the only nonzero piece)." |
| 566 | A4 | LOW | `Use Crank-Nicolson (instead of backward Euler) for DC/BR.` (doc for `pub cn_dcbr: bool`) | Boolean field docs should read "If true, ...": "If true, uses Crank-Nicolson (instead of backward Euler) for DC/BR." |
| 237, 1580 | F11 | MEDIUM (2 instances, file total 2) | `— including its \`upper[i]/denom\` division chain, which is the serial` | Remove the spaces around the em dash: `—including`. |
| 570 | A2 | MEDIUM | `In-place Kompaneets and DC/BR step using pre-allocated workspace.` | Fragment/noun phrase for a `pub fn`. Rewrite: "Steps Δn (and optionally ρ_e) forward in place using pre-allocated workspace." |
| 1229–1234, 1271–1274, 1597–1600, 1630–1631, 1672–1673 | A2 | LOW (5 test-doc instances) | e.g. `Coupled step with DC/BR active and a nonzero equilibrium offset, tiny N.` | Test-rationale titles as noun phrases; lower impact but still not verb-led. Optional: "Tests a coupled step with DC/BR active..." |

## double_compton.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 23, 39, 53, 68, 92 | A2 | MEDIUM (5, file total 10 incl. tests below) | `DC Gaunt factor in the soft photon limit with relativistic corrections.` | Lead with a verb: "Computes the DC Gaunt factor in the soft photon limit with relativistic corrections." |
| 183, 291, 392, 418, 466 | A2 | LOW (5 test-doc instances) | `K_DC must be one implementation, not two (R2 mutation audit, fix A1).` | Test-rationale titles as fragments/assertions rather than verb-led summaries. |
| 190 | T9 | MEDIUM | `The Planck test above only covers ρ_e = 1, where φ = 1 and any error in` | Replace document-position reference: "The preceding Planck test only covers..." or name the test function. |
| 191 | F1 | LOW | `the φ convention — the exact inversion CLAUDE.md pitfall #1 warns about —` | Code-font the filename: `` `CLAUDE.md` ``. |
| 191, 399 | F11 | MEDIUM (2, counted in file total 3) | `the φ convention — the exact inversion CLAUDE.md pitfall #1 warns about —` | Unspaced em dash. |
| 13 | F11 | MEDIUM | `- Lightman (1981) — original DC coefficient` | Unspaced em dash. |
| 467 | F1 | LOW | `Absolute value of K_DC, derived by hand from CS2012 Eq. 13 rather than` (comment continues) `read off code output (CLAUDE.md pitfall #9).` | Code-font `CLAUDE.md`. |

## bremsstrahlung.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 37, 101, 184, 246, 342, 376, 403, 421 | A2 | MEDIUM (8 pub fns, file total 13) | `BR emission coefficient K_BR at frequency x.` | "Computes the BR emission coefficient K_BR at frequency x." |
| 59, 69, 278, 357 | A2 | MEDIUM (4 private fns) | `Fast Gaunt factor with precomputed 0.5*ln(θ_e) hoisted out of grid loop.` | "Computes the Gaunt factor with precomputed 0.5·ln(θ_e) hoisted out of the grid loop." (First 10 of the 13 are listed here; the remaining 3 are the private-fn rows above plus the test row below — see file total.) |
| 563–568 | A2 | LOW (test doc) | `Detailed balance (Kirchhoff) at T_e ≠ T_z (R2 mutation audit, fix P4).` | Test-rationale fragment. |
| 567 | T9 | MEDIUM | `φ = θ_z/θ_e, at any ρ_e. The test above only covers ρ_e = 1, where φ = 1` | Replace "the test above" with the concrete test name or "the preceding test." |
| 13, 14, 239 | F11 | MEDIUM (3) | `- Karzas & Latter (1961) — original Gaunt factors` | Unspaced em dash. |
| 893–894 | F1 | LOW | `re-derived in the B1 validation audit (dev/audit/` / `double_compton_bremsstrahlung_audit.md, F1):` | Code-font the path: `` `dev/audit/double_compton_bremsstrahlung_audit.md` ``. |

## electron_temp.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 10 | T9 | MEDIUM | `computed from Δn only. The full form ρ_eq = I₄/(4G₃) (below) has` | Replace "(below)" with a concrete pointer: "(implemented in `update_equilibrium_full`, kept for verification only)". |
| 130 | T13 | LOW | `Proof: dn/dx = −(1/a) n(1+n), hence n(1+n) = −a dn/dx and` | Replace the formal connective: "..., so n(1+n) = −a dn/dx and" |
| 134 | T9 | MEDIUM | `This is the discriminating check the ρ_eq=1 tests above lack: they feed` | Name the tests instead of "above": "...the check that `test_rho_eq_unity_for_any_bose_einstein` and its neighbors lack". |
| 12 | F11 | MEDIUM | `physical signal — do not use it in the solver. It is retained here only` | Unspaced em dash (file total 2). |
| 42–48 | A2 | MEDIUM | `Sets ρ_e from the full Compton-equilibrium form I₄/(4G₃).` | This one is actually verb-led ("Sets") — no fix needed; included only to confirm the file was checked (not counted in totals). |
| 34 | A2 | MEDIUM | `θ_e from a precomputed θ_z value (cosmology-aware).` | "Computes θ_e from a precomputed θ_z value (cosmology-aware)." |
| 125 | A2 | LOW (test doc) | `` `update_equilibrium` must recover a *non-unity* temperature ratio. `` | Test-rationale fragment/assertion style. |

## recombination.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 6–14 | L2 | MEDIUM | `## Physical picture` immediately followed by `- z > 8000: Fully ionized (H and He). Helium is doubly ionized (He²⁺).` | Add an introductory sentence ending in a colon before the bullets, e.g. "The ionization history passes through the following regimes as z decreases:". |
| 16–22 | L2 | MEDIUM | `## Implementation` immediately followed by `- \`alpha_recomb\`: Case-B recombination coefficient (Péquignot fit)` | Add an intro sentence: "The module follows DarkHistory's three-level-atom structure through these pieces:". |
| 41, 78, 110, 137, 153, 165, 186, 282, 322 | A2 | MEDIUM (5 pub fns: 78, 110, 137, 153, 322) | `He II → He I Saha ionization fraction (54.4 eV).` | "Computes the He II → He I Saha ionization fraction (54.4 eV)." |
| 41, 165, 186, 282 | A2 | MEDIUM (4 private fns) | `Thermal de Broglie factor: (m_e k_B T / (2π ℏ²))^{3/2} [m⁻³].` | "Computes the thermal de Broglie factor..." |
| 685 | A2 | LOW (test doc) | `X_e through the helium recombination epoch, against the same HyRec-2` | Test-rationale fragment. |
| 88 | T9 | MEDIUM | `by H⁺ at z ≳ 1500 (H is fully ionized throughout He recombination since` `χ_I(H) = 13.6 eV ≪ χ_II(He) = 54.4 eV).` | Replace causal "since" with "because": "...He recombination because χ_I(H) = 13.6 eV ≪ χ_II(He) = 54.4 eV)." |
| 688 | T9 | MEDIUM | `The milestone test above probes only z = 1100/800/200 — all hydrogen-` | Name the test: "`test_xe_vs_recfast_milestones` probes only z = 1100/800/200...". |
| 696 | T9 | LOW | `and cosmology as above). Bands follow the measured per-band disagreement` | "...same run and cosmology). Bands follow..." |
| 662, 695 | F1 | MEDIUM | `dev/output/hyrec2_xe_default_cosmo.dat. HyRec values:` | Code-font the path: `` `dev/output/hyrec2_xe_default_cosmo.dat` `` (it is backticked elsewhere in the same file, e.g. `` `xe_hyrec_comparison.md` `` at line 691, so this is an inconsistency, not a one-off). |
| 665 | F1 | LOW | `(see dev/audit/xe_hyrec_comparison.md); the ±6% band gives ~3× slack.` | Code-font the path. |
| 696 | F1 | LOW | `in xe_hyrec_comparison.md with ~1.5× slack: ≤0.14% for 3000–5000 (He²⁺/He⁺` | Code-font the filename (inconsistent with the backticked use two paragraphs earlier). |
| 660 | F1 | LOW | `Anchors are HyRec-2 (github.com/nanoomlee/HyRec-2, run 2026-07-05) on` | Put the bare repo path in code font: `` `github.com/nanoomlee/HyRec-2` ``. |
| 30–34 (×5) | F11 | MEDIUM (file total 15) | `- Peebles (1968) — Three-level atom model` | Unspaced em dashes throughout the reference list and prose (15 lines in this file use `" — "`). |

## constants.rs

No A2 violations (all documented items are constants, correctly using noun-phrase docs).

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 152 | T5 | LOW | `Convenience helper for tests and quick calculations. Production code must` | "Convenience helper for tests and short calculations." (avoid "quick" as a speed/ease claim). |
| 7, 108, 118 (+2 more) | F11 | MEDIUM (file total 5) | `` `M_PROTON`, `SIGMA_THOMSON`, `ALPHA_FS` — feed Compton scattering rates, `` | Unspaced em dash. |

## cosmology.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 125 | T13 | MEDIUM | `are known a priori to be valid. Using non-finite or zero \`h\` / \`y_p\`` | Replace the Latin phrase: "are known in advance to be valid." |
| 432 | T6 | LOW | `These are intentionally not the latest Planck values. The defaults` | "These are intentionally not the most recent Planck values." (or name the Planck release directly and drop the timeless qualifier). |
| 180, 210, 216, 222, 228, 234, 240, 246, 252, 269 | A2 | MEDIUM (first 10 of 25 `pub fn` instances, file total 26 incl. 1 private) | `Planck 2015 cosmological parameters (Planck XIII, Table 4).` | "Returns the Planck 2015 cosmological parameters (Planck XIII, Table 4)." Similarly for the getters: "Returns Ω_b = ω_b / h²." etc. |
| 31, 35, 37 (+1 more) | F11 | MEDIUM (file total 4) | `(1-Y_p) * rho_b0 / M_PROTON — multiply by (1+z)³ to get n_H(z).` | Unspaced em dash. |

## spectrum.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 24, 39, 52, 68, 78, 95 | A2 | MEDIUM (6 pub fns) | `Planck (blackbody) occupation number: n_pl(x) = 1/(e^x - 1).` | "Computes the Planck (blackbody) occupation number n_pl(x) = 1/(e^x − 1)." |
| 150 | A2 | MEDIUM (private fn) | `Trapezoidal integral of x^power × Δn over the grid, divided by norm.` | "Computes the trapezoidal integral of x^power × Δn over the grid, divided by norm." |
| 176 | A2 | LOW (test doc) | `Planck identity: dn_pl/dx + n_pl(1 + n_pl) = 0 exactly.` | Test-rationale fragment. |
| 178 | F1 | LOW | `This cancellation underpins the Kompaneets flux-split (CLAUDE.md` | Code-font `CLAUDE.md`. |
| 6, 10, 12 (+4 more) | F11 | MEDIUM (file total 7) | `coefficient that multiplies the shape — such as \`Δn_μ(x) = μ · M(x)\`,` | Unspaced em dash. |

## grid.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 67 | T5 | LOW | `Fast/testing grid: 500 points, \`x ∈ [1e-4, 40]\`. Suitable for quick` | "Suitable for exploratory runs; distortion amplitudes are accurate to a few percent." (drop "quick" as an ease/speed claim). |
| 54, 67 | A2 | MEDIUM (2, both `pub fn`) | `Production-quality grid: 4000 points, \`x ∈ [1e-5, 60]\`. Used for all` | "Builds the production-quality grid: 4000 points..." |

No em dash-spacing violations in this file.

## dark_photon.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 22, 33, 64, 77 | A2 | MEDIUM (4, all `pub fn`) | `Photon plasma frequency ω_pl (in eV) at redshift \`z\`.` | "Computes the photon plasma frequency ω_pl (in eV) at redshift `z`." |
| 1 | DOMAIN | DOMAIN | `Helpers for dark photon (γ ↔ A') conversion in the narrow-width approximation.` | Standard physics bidirectional-process notation; no fix needed. |

No em dash-spacing violations in this file.

## axion.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 49, 58 | A2 | MEDIUM (2, both `pub fn`) | `Axion mixing coupling \`κ = g_aγγ B_rms⁰\` (today), in eV.` | "Computes the axion mixing coupling κ = g_aγγ B_rms⁰ (today), in eV." |
| 10–18 | L1 | MEDIUM | `Two differences from the dark-photon case:` followed by `1. **Frequency dependence flips.** ...` `2. **The \`γ_con\` prefactor** replaces ...` | These are two independent, unordered points, not sequential steps — use a bulleted list, not a numbered one. |
| 12–18 | L3 | MEDIUM | Item 1: `**Frequency dependence flips.** The axion probability carries...`; item 2: `**The \`γ_con\` prefactor** replaces \`ε² m²\` with...` | Item 1 uses a bolded complete-sentence lead-in before continuing; item 2 uses a bolded noun phrase folded into one sentence. Make the two items parallel (both as bolded noun-phrase lead-ins, or both as bolded standalone sentences). |
| 31, 32 | F11 | MEDIUM (file total 2) | `Manoj 2024, Sec. II B, Eqs. 7–12) — which shift \`z_con(ω)\` and produce` | Unspaced em dash. |
| 5 | DOMAIN | DOMAIN | `photon → dark-photon treatment of Chluba, Cyr & Johnson (2024): a resonant` | Arrow as physics-process shorthand; borderline prose use, but consistent with the module's other reaction/process notation. |

## Cross-file note (F8, not tallied above)

Equation/section-range punctuation is inconsistent: `double_compton.rs` uses a hyphen
("Chluba & Sunyaev (2012), MNRAS 419, 1294 [Eq. 10-13]"), `electron_temp.rs` likewise
("[Eq. 15-18]"), while `axion.rs` uses an en dash ("Sec. II B, Eqs. 7–12"). Pick one range
convention (en dash is the more common in this corpus, for example `recombination.rs`'s
"z ~ 1500–800") and apply it to equation ranges too. LOW severity, 2 files affected.
