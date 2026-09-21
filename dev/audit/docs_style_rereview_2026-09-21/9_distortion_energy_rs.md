## src/distortion.rs

| line | rule | severity | quotation | suggested fix |
|---|---|---|---|---|
| 42 | T9 | MEDIUM | `the supplied grid should extend beyond [x_min, x_max] on both` | Use "must extend" if this is a real precondition (the function has undefined behavior otherwise), or rephrase as a factual statement. |
| 221 | T9 | MEDIUM | `which spans the same 3-D subspace as the CJ2014 basis (Y_SZ, M, G) since` | Replace "since" (causal) with "because": "...(Y_SZ, M, G) because M(x) = ...". |
| 391-392 | F11 | LOW | `μ is the physical chemical potential (matching FIRAS-convention fits) —` | Remove the spaces around the em dash: "fits)—not Chluba's". |
| 391 | F7 | LOW | `Note: B&F absorbs μ inside the Bose-Einstein exponential` | Use the file's own bold-notice convention (`**Note:**`) as used elsewhere in this doc set (e.g. energy_injection.rs line 112), for consistency. |
| 417 | T7 | MEDIUM | `a footgun for callers who set a custom (too-narrow)` | Replace the programmer slang "footgun" with plain language, e.g. "an easy mistake" or "a common source of silent errors," for a global audience. |
| 418 | T9 | MEDIUM | `Solvers should sample this once at startup and surface` | If this is a real requirement for correct use, use "must sample"; otherwise state it as guidance without "should". |

Systematic patterns:
- Struct-field and enum-variant doc comments in this file are consistently noun phrases (e.g. lines 26-36), which is normal for Rust field docs and not flagged.
- No missing-abbreviation issues: PIXIE and FIRAS are both spelled out at first use in the module header (lines 5-6) before being used as acronyms.
- Function summaries generally follow the present-tense-verb pattern correctly (A2 compliant): "Collects..." (40), "Decomposes..." (381), "Converts..." (441), "Checks..." (432).

## src/energy_injection.rs

| line | rule | severity | quotation | suggested fix |
|---|---|---|---|---|
| 11 | T2 | MEDIUM | `All heating rates are expressed as d(Δρ_γ/ρ_γ)/dt in units of 1/s.` | Rephrase active: "This module expresses all heating rates as d(Δρ_γ/ρ_γ)/dt, in units of 1/s." |
| 216-217 | A4 | MEDIUM | `/// Custom heating function.` | State what the closure's two arguments mean, e.g. "Custom heating function: takes the current redshift and the cosmology, and returns the heating rate in 1/s." |
| 246-247 | A2 | MEDIUM | `/// Bilinear interpolation on a 2D table (z ascending, x ascending).` | Start with a present-tense verb: "Interpolates bilinearly on a 2D table..." |
| 462 | A2 | MEDIUM | `/// CLI-friendly name for this injection scenario.` | "Returns a CLI-friendly name for this injection scenario." |
| 880 | A2 | MEDIUM | `/// Frequency-dependent photon injection or removal rate.` | "Returns the frequency-dependent photon injection or removal rate." |
| 1039 | A2 | MEDIUM | `/// Dark-photon NWA parameters (γ_con, z_res), if applicable.` | "Returns the dark-photon NWA parameters (γ_con, z_res), if applicable." |
| 1053 | A2 | MEDIUM | `/// Axion NWA parameters (γ_con, z_res), if applicable.` | "Returns the axion NWA parameters (γ_con, z_res), if applicable." |
| 1069 | A2 | MEDIUM | `/// Axion NWA parameters — always \`None\` without the \`axion\` feature.` | "Returns the axion NWA parameters; always `None` without the `axion` feature." |
| 1091 | A2 | MEDIUM | `/// Impulsive-resonance NWA parameters (γ_con, z_res) for whichever resonant` | "Returns the impulsive-resonance NWA parameters (γ_con, z_res) for whichever resonant..." |
| 1102 | A2 | MEDIUM | `/// Initial-condition perturbation Δn(x) to be installed at \`z_start\`.` | "Returns the initial-condition perturbation Δn(x) to install at `z_start`." |
| 1141 | A2 | MEDIUM | `/// Suggest a lower \`x_min\` for the frequency grid when needed.` | Match the file's third-person convention used everywhere else: "Suggests a lower `x_min` for the frequency grid when needed." |
| 1170 | A2 | MEDIUM | `/// Checks for strong distortion regime and return warnings.` | Fix the internal verb-form mismatch: "Checks for a strong-distortion regime and returns warnings." |
| 1025 | F11 | LOW | `` `None` — injection happens at all z. `` | Remove spaces around the em dash. |
| 1069 | F11 | LOW | `Axion NWA parameters — always \`None\` without the \`axion\` feature.` | Remove spaces around the em dash. |
| 1317-1318 | F11 | LOW | `` `interp_log_z` / `interp_2d` return 0.0 outside the table — a silent `` | Remove spaces around the em dash. |
| 1614 | F11 | LOW | `` Previous version asserted `rate > 1e-20` — 26 orders of magnitude `` | Remove spaces around the em dash. |
| 1871 | F11 | LOW | `tolerance check — which is what makes it a tight anchor.` | Remove spaces around the em dash. |
| 1614-1615 | T12 | LOW | `A 12-OOM formula bug would have passed.` | "OOM" is never spelled out; the preceding sentence says "orders of magnitude" but does not tie it to the abbreviation. Write "A 12-orders-of-magnitude (OOM) formula bug..." at first use, or avoid the abbreviation entirely since it is used only once more. |

Systematic patterns:
- A2 (function summaries as noun phrases instead of present-tense verbs) recurs 10 times across the accessor methods on `InjectionScenario` (lines 246, 462, 880, 1039, 1053, 1069, 1091, 1102, 1141, 1170); most of the file's `fn`/`pub fn` docs (e.g. `Loads...`, `Warns...`, `Computes...`, `Returns...`) do follow the correct present-tense-verb pattern, so this is a localized inconsistency in the accessor cluster around lines 1039-1141, not a file-wide style choice.
- Em dashes with surrounding spaces recur 5 times in this file (lines 1025, 1069, 1317, 1614, 1871) plus 1 in distortion.rs (line 391); Google style is a spaceless em dash, so this is a minor but repeated formatting slip.
- Citation-style ampersands ("Chluba & Sunyaev", "Bolliet & Chluba", "B&C 2021") are classified DOMAIN, not flagged: these are author-list conventions in physics citations, not prose symbol substitution.
- Reaction arrows (X → γγ, γ ↔ A′, γ ↔ a) are classified DOMAIN per the rubric's own example (γe → γγe).
- Physical-quantity units are stated consistently and well throughout (eV, 1/s, GeV⁻¹, nG, [1/s]) — this file does A4's units requirement better than most.
