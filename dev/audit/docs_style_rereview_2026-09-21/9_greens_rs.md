## src/greens.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 569–593 (doc for `greens_function_photon`) | A4 | HIGH | `/// Green's function for monochromatic photon injection.` ... doc ends at `///   Chluba (2015), arXiv:1506.06582` with no panic note, but the function body (line 601) calls `assert_photon_gf_regime(z_h)`, which panics when `z_h` is in the μ-y transition band | Add a `# Panics` section: "Panics if `z_h` falls in the μ-y transition band (see `assert_photon_gf_regime`); callers must check `in_photon_gf_transition_band` first or use the PDE solver." |
| 674–686 (doc for `mu_from_photon_injection`) | A4 | HIGH | `/// Computes μ from monochromatic photon injection at frequency x_inj.` ... doc ends at `/// Reference: Chluba (2015), Eq. C7` with no panic note, but line 688 calls `assert_photon_gf_regime(z_h)` | Same fix: document the panic condition on `z_h`. |
| 41–42 | A2 | MEDIUM | `Thermalization visibility: probability that energy injection at z is fully thermalized into a blackbody (temperature shift).` | Rewrite as a verb-first sentence, e.g. "Computes the thermalization visibility J_bb(z): the probability that..." |
| 57 | A2 | MEDIUM | `Improved thermalization visibility with correction factor.` | "Computes the improved thermalization visibility J_bb*(z), with correction factor." |
| 73 | A2 | MEDIUM | `y-distortion branching ratio: fraction of energy going into y-type distortion.` | "Computes the y-distortion branching ratio J_y(z): the fraction..." |
| 89 | A2 | MEDIUM | `μ-distortion branching ratio: fraction of energy going into μ-type distortion.` | "Computes the μ-distortion branching ratio J_μ(z): the fraction..." |
| 100 | A2 | MEDIUM | `Temperature shift branching: fraction of energy going into temperature shift.` | "Computes the temperature-shift branching J_T(z): the fraction..." |
| 302 | A2 | MEDIUM | `Critical frequency for double Compton absorption.` | "Computes the critical frequency for double-Compton absorption." |
| 311 | A2 | MEDIUM | `Critical frequency for bremsstrahlung absorption.` | "Computes the critical frequency for bremsstrahlung absorption." |
| 320 | A2 | MEDIUM | `Combined critical frequency for photon absorption.` | "Computes the combined critical frequency for photon absorption." |
| 333 | A2 | MEDIUM | `Photon survival probability P_s(x, z).` | "Computes the photon survival probability P_s(x, z)." |
| 350 | A2 | MEDIUM | `Photon survival probability: numerical τ_ff in the y-era, analytic at higher z.` | "Computes the photon survival probability, using a numerical τ_ff integral in the y-era and the analytic form at higher z." (3 more instances at lines 465, 472, 481 — total 13) |
| 26 | T11 | MEDIUM | `Chluba (2013) §3 notes that the "missing" energy in this ansatz stays within` | Spell out: "Chluba (2013), Section 3, notes that..." |
| 248 | T9 | MEDIUM | `separately, as it evaluates the visibility functions only once per z-step.` | Replace causal "as" with "because": "...separately, because it evaluates the visibility functions only once per z-step." |
| 917 | T9 | MEDIUM | `and the DC term carries the same x⁻² since H_dc(x→0) → 1.` | Replace causal "since" with "because": "...carries the same x⁻² because H_dc(x→0) → 1." |
| 78 | F11 | LOW | `Eq. 5, which reads (1+z)/(6.0×10⁴) with exponent 2.58 — verified against` | Remove spaces around the em dash: "2.58—verified against". |
| 909 | F11 | LOW | ``tau_ff_survival` — the branch that sets the` | Remove spaces around the em dash. |
| 1054 | F11 | LOW | `≈8×10⁻⁵ — comfortably inside the 5×10⁻⁴ band. (At σ = 0.02 that term is` | Remove spaces around the em dash. |
| 1106 | F11 | LOW | `` `tests/` — they were verified once by hand against Arsenadze et al. `` | Remove spaces around the em dash. |

### Systematic patterns

- **A2 (noun-phrase summaries instead of verb-first sentences):** 13 functions/consts (lines 41, 57, 73, 89, 100, 302, 311, 320, 333, 350, 465, 472, 481) open their doc comment with a colon-definition noun phrase ("X: fraction of..."/"Critical frequency for...") rather than a present-tense verb ("Computes...", "Returns..."). The file is otherwise consistent with the verb-first convention (`greens_function`, `distortion_from_heating`, `mu_from_heating`, etc. all start with "Computes"/"Extracts"), so this is an internal inconsistency rather than a deliberate alternate style — not a DOMAIN carve-out.
- **A4 (undocumented panics):** the two public entry points into the photon-injection Green's function (`greens_function_photon`, `mu_from_photon_injection`) both call `assert_photon_gf_regime`, which panics outside its own doc comment's knowledge — the panic condition is never stated in either function's doc comment. `distortion_from_photon_injection`, by contrast, correctly guards against the same panic internally and needs no such note.
- **F11 (em dash spacing):** 4 instances of `" — "` (space on both sides) instead of Google style's unspaced em dash; low-severity polish, listed in full above (fewer than 10 total).
- Citations and abbreviations are otherwise handled well: PDE, DC, BR, GF, and FIRAS are each spelled out at first use before being abbreviated, and cross-references (e.g., "see `visibility_j_bb`") name the specific item rather than using "see above/below".
