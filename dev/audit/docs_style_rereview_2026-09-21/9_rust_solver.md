# Documentation review: Rust solver core (Google developer style guide)

Scope: `src/solver.rs`, `src/greens.rs`, `src/distortion.rs`, `src/energy_injection.rs`,
`src/output.rs`, `src/cli.rs` (doc comments and CLI help-text strings), `src/lib.rs`, `src/main.rs`,
`src/bin/check_adiabatic.rs`, and the `//!` headers of the five files in `examples/`. Only
`//!`/`///` doc comments and, for `cli.rs`, the printed CLI help text were reviewed, per the brief.
Rule IDs and severities follow `rubric.md`.

## Summary table (counts by severity)

| File | HIGH | MEDIUM | LOW | DOMAIN |
|---|---|---|---|---|
| src/solver.rs | 0 | 4 | 0 | 0 |
| src/greens.rs | 2 | 16 | 4 | 0 |
| src/distortion.rs | 0 | 4 | 2 | 0 |
| src/energy_injection.rs | 0 | 12 | 6 | 0 |
| src/output.rs | 0 | 1 | 2 | 0 |
| src/cli.rs | 2 | 6 | 1 | 3 |
| src/lib.rs | 0 | 4 | 0 | 2 |
| src/main.rs | 0 | 3 | 1 | 0 |
| src/bin/check_adiabatic.rs | 0 | 0 | 0 | 0 |
| examples/*.rs headers | 0 | 3 | 2 | 0 |
| **Total** | **4** | **53** | **18** | **5** |

## Systematic patterns

1. **A2 — noun-phrase function summaries instead of present-tense verbs.** The largest recurring
   issue by instance count. Concentrated in `src/greens.rs` (13 instances: visibility/branching-ratio
   functions, critical-frequency functions) and `src/energy_injection.rs` (10 instances, mostly the
   `InjectionScenario` accessor cluster around lines 1039-1170), with isolated cases in
   `src/distortion.rs`-adjacent files and one verb-agreement slip each in `src/solver.rs` and
   `src/energy_injection.rs` ("Checks... and return", "Replaces... and reset"). Each affected file is
   otherwise internally consistent with the verb-first convention used elsewhere in the same file
   (`greens_function` → "Computes...", `decompose_distortion` → "Decomposes..."), so this reads as a
   localized inconsistency, not a deliberate alternate convention — not eligible for the DOMAIN/PEP
   257 carve-out.
2. **T12 — undefined abbreviations, isolated to `src/cli.rs`'s user-facing help text.** Every
   `//!`/`///` doc comment reviewed, in every file, correctly expands PDE, CMB, DC, BR, ODE, IMEX,
   CLI, GF, NWA, IC, and FIRAS at first use. The one exception is the actual `println!` text a CLI
   user sees (`spectroxide help`, `spectroxide <sub> --help`): "CMB" (line 852) and "PDE" (lines 859,
   860, 861, 863, 934, 977, 1000, 1034, 1036, ...) are used repeatedly and never expanded anywhere in
   the printed help output — a separate "page" from the source comments by the rubric's own
   document-scope logic, and the one that fails T12 where the source comments pass. Both are HIGH:
   the top-line banner and the subcommand list are core to the page.
3. **F11 — em dash with surrounding spaces.** At least 9 confirmed instances across `greens.rs` (4),
   `energy_injection.rs` (5), plus 1 each in `distortion.rs` and `examples/energy_budget.rs` — all LOW
   severity, all the same fix (drop the space on each side of the em dash).
4. **T9 — causal "since"/"as", and "above"/"below" for document position.** Confirmed in
   `greens.rs` (×2), `solver.rs` (×1), `distortion.rs` (×2), `cli.rs` (×3, including one "vs" and two
   "above"/"below" instances). Each is replaceable with "because" or "preceding"/"following" without
   other changes.
5. **A4 — undocumented panics, isolated but real.** `greens.rs`'s two public photon-injection entry
   points (`greens_function_photon`, `mu_from_photon_injection`) call `assert_photon_gf_regime`,
   which panics in the μ-y transition band, with no `# Panics` note in either doc comment (HIGH — the
   sibling function `distortion_from_photon_injection` correctly guards against the same panic and
   needs no note, so the omission is an inconsistency, not a deliberate simplification).
   `solver.rs::set_initial_delta_n` documents one of its two panic conditions but not the
   length-mismatch `assert_eq!` (MEDIUM). `energy_injection.rs`'s `Custom` heating-closure doc doesn't
   state what the closure's two arguments mean (MEDIUM).
6. **T2 — passive "Used to ..." construction.** Recurs in `output.rs` (line 747) and `cli.rs`
   (line 1346) with near-identical phrasing ("Used to catch...", "Used to compress..."), suggesting a
   house idiom rather than one-off passive voice; still flagged per-instance since the actor (the
   function itself) is trivial to name.
7. Outside these patterns, the doc comments are generally well-formed: `# Errors` sections are
   present and specific (though two multi-condition ones in `cli.rs` run semicolon-joined conditions
   into one sentence rather than a list — F11/L1 borderline, listed under `cli.rs` below),
   cross-references name specific items rather than "see above/below" (with the two noted
   exceptions), reaction arrows and citation ampersands are correctly treated as DOMAIN throughout,
   and `src/bin/check_adiabatic.rs`'s single module doc comment has no confirmed violations.

---

## src/solver.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 46 | T9 | MEDIUM | `with 5–6× fewer steps, since \`dtau_max\` takes over as the binding` | Replace causal "since" with "because": "...fewer steps, because `dtau_max` takes over as the binding constraint..." |
| 297–298 | T6 | MEDIUM | `Many fields are currently \`pub\` for use by examples, tests, and diagnostic tooling. These are not part of the stable public API and may become \`pub(crate)\` in a future release.` | Drop the timeless-text markers: "Many fields are `pub` for use by examples, tests, and diagnostic tooling. These are not part of the stable public API and may move to `pub(crate)`." |
| 692–693 | A2 | MEDIUM | `Replaces the solver configuration and reset the current redshift to` | Fix verb agreement to keep both verbs present-tense third-person: "Replaces the solver configuration and resets the current redshift to..." |
| 707–717 | A4 | MEDIUM | doc reads `Panics if any entry is non-finite — silently passing NaN/Inf into the solver causes a deep panic...` but the function (`set_initial_delta_n`) also panics via `assert_eq!` at line 708 on a length mismatch (`"initial_delta_n length {} != grid size {}"`), which the doc comment never mentions | Extend the `Panics` note to cover both conditions: "Panics if `delta_n.len()` does not match the grid size, or if any entry is non-finite." |

### Systematic patterns

- Module and struct-level doc comments (module header, `SolverConfig`, `SolverDiagnostics`,
  `ThermalizationSolver`, `SolverBuilder`) consistently define abbreviations at first use (PDE, DC,
  BR, ODE, IMEX, CLI, CMB) and are otherwise well-formed: first sentences are complete, references
  are specific (`see [\`Self::reset\`]`, not "see above"), and panics/errors are documented for most
  public methods.
- Several method docs use passive constructions ("is taken from," "is set to," e.g. line 1672) where
  the actor (the solver / the method) is clear and nameable (T2). Low-density (a handful of instances
  across ~360 doc-comment lines), not itemized individually.
- No "e.g.", "i.e.", "etc.", "via", "vs.", "utilize", "leverage", or Latin phrases found.
- Not a rubric-mappable finding, flagged for awareness only: lines 1070-1084 read as a
  duplicated/leftover doc sentence — `subtract_temperature_shift`'s summary appears twice ("Subtracts
  the temperature shift component..." and, after the algorithm list, "Subtract the number-conserving
  temperature shift from Δn.") — looks like an editing artifact, not a style-rule violation per se.

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
| 350 (+ 465, 472, 481; 4 total) | A2 | MEDIUM | `Photon survival probability: numerical τ_ff in the y-era, analytic at higher z.` | "Computes the photon survival probability, using a numerical τ_ff integral in the y-era and the analytic form at higher z." |
| 26 | T11 | MEDIUM | `Chluba (2013) §3 notes that the "missing" energy in this ansatz stays within` | Spell out: "Chluba (2013), Section 3, notes that..." |
| 248 | T9 | MEDIUM | `separately, as it evaluates the visibility functions only once per z-step.` | Replace causal "as" with "because": "...separately, because it evaluates the visibility functions only once per z-step." |
| 917 | T9 | MEDIUM | `and the DC term carries the same x⁻² since H_dc(x→0) → 1.` | Replace causal "since" with "because": "...carries the same x⁻² because H_dc(x→0) → 1." |
| 78 | F11 | LOW | `Eq. 5, which reads (1+z)/(6.0×10⁴) with exponent 2.58 — verified against` | Remove spaces around the em dash: "2.58—verified against". |
| 909 | F11 | LOW | ``tau_ff_survival` — the branch that sets the` | Remove spaces around the em dash. |
| 1054 | F11 | LOW | `≈8×10⁻⁵ — comfortably inside the 5×10⁻⁴ band. (At σ = 0.02 that term is` | Remove spaces around the em dash. |
| 1106 | F11 | LOW | `` `tests/` — they were verified once by hand against Arsenadze et al. `` | Remove spaces around the em dash. |

### Systematic patterns

- A2 (noun-phrase summaries instead of verb-first sentences): 13 functions/consts (lines 41, 57, 73,
  89, 100, 302, 311, 320, 333, 350, 465, 472, 481) open with a colon-definition noun phrase rather than
  a present-tense verb. The file is otherwise consistent with the verb-first convention
  (`greens_function`, `distortion_from_heating`, `mu_from_heating` all start with "Computes"/
  "Extracts"), so this is an internal inconsistency, not a deliberate alternate style.
- A4 (undocumented panics): the two public entry points into the photon-injection Green's function
  both call `assert_photon_gf_regime` without stating the panic condition in their own doc comments;
  `distortion_from_photon_injection` correctly guards against the same panic and needs no such note.
- F11 (em dash spacing): 4 instances of `" — "` instead of Google style's unspaced em dash.
- Citations and abbreviations are otherwise handled well: PDE, DC, BR, GF, and FIRAS are each spelled
  out at first use, and cross-references name the specific item rather than "see above/below".

## src/distortion.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 42 | T9 | MEDIUM | `the supplied grid should extend beyond [x_min, x_max] on both` | Use "must extend" if this is a real precondition (undefined behavior otherwise), or rephrase as a factual statement. |
| 221 | T9 | MEDIUM | `which spans the same 3-D subspace as the CJ2014 basis (Y_SZ, M, G) since` | Replace causal "since" with "because": "...(Y_SZ, M, G) because M(x) = ...". |
| 417 | T7 | MEDIUM | `a footgun for callers who set a custom (too-narrow)` | Replace the programmer slang "footgun" with plain language, e.g. "an easy mistake" or "a common source of silent errors," for a global audience. |
| 418 | T9 | MEDIUM | `Solvers should sample this once at startup and surface` | If this is a real requirement for correct use, use "must sample"; otherwise state it as guidance without "should". |
| 391–392 | F11 | LOW | `μ is the physical chemical potential (matching FIRAS-convention fits) —` | Remove the spaces around the em dash: "fits)—not Chluba's". |
| 391 | F7 | LOW | `Note: B&F absorbs μ inside the Bose-Einstein exponential` | Use the file's own bold-notice convention (`**Note:**`), as used elsewhere in this doc set (for example, `energy_injection.rs` line 112), for consistency. |

### Systematic patterns

- Struct-field and enum-variant doc comments are consistently noun phrases (for example, lines
  26-36), normal for Rust field docs and not flagged.
- No missing-abbreviation issues: PIXIE and FIRAS are both spelled out at first use in the module
  header (lines 5-6) before use as acronyms.
- Function summaries generally follow the present-tense-verb pattern correctly (A2 compliant):
  "Collects..." (40), "Decomposes..." (381), "Converts..." (441), "Checks..." (432).

## src/energy_injection.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 11 | T2 | MEDIUM | `All heating rates are expressed as d(Δρ_γ/ρ_γ)/dt in units of 1/s.` | Rephrase active: "This module expresses all heating rates as d(Δρ_γ/ρ_γ)/dt, in units of 1/s." |
| 216-217 | A4 | MEDIUM | `/// Custom heating function.` | State what the closure's two arguments mean, for example "Custom heating function: takes the current redshift and the cosmology, and returns the heating rate in 1/s." |
| 246-247 | A2 | MEDIUM | `/// Bilinear interpolation on a 2D table (z ascending, x ascending).` | Start with a present-tense verb: "Interpolates bilinearly on a 2D table..." |
| 462 | A2 | MEDIUM | `/// CLI-friendly name for this injection scenario.` | "Returns a CLI-friendly name for this injection scenario." |
| 880 | A2 | MEDIUM | `/// Frequency-dependent photon injection or removal rate.` | "Returns the frequency-dependent photon injection or removal rate." |
| 1039 | A2 | MEDIUM | `/// Dark-photon NWA parameters (γ_con, z_res), if applicable.` | "Returns the dark-photon NWA parameters (γ_con, z_res), if applicable." |
| 1053 | A2 | MEDIUM | `/// Axion NWA parameters (γ_con, z_res), if applicable.` | "Returns the axion NWA parameters (γ_con, z_res), if applicable." |
| 1069 | A2 | MEDIUM | `/// Axion NWA parameters — always \`None\` without the \`axion\` feature.` | "Returns the axion NWA parameters; always `None` without the `axion` feature." |
| 1091 | A2 | MEDIUM | `/// Impulsive-resonance NWA parameters (γ_con, z_res) for whichever resonant` | "Returns the impulsive-resonance NWA parameters (γ_con, z_res) for whichever resonant..." |
| 1102 | A2 | MEDIUM | `/// Initial-condition perturbation Δn(x) to be installed at \`z_start\`.` | "Returns the initial-condition perturbation Δn(x) to install at `z_start`." |
| 1141 | A2 | MEDIUM | `/// Suggest a lower \`x_min\` for the frequency grid when needed.` | Match the file's third-person convention used elsewhere: "Suggests a lower `x_min` for the frequency grid when needed." |
| 1170 | A2 | MEDIUM | `/// Checks for strong distortion regime and return warnings.` | Fix the internal verb-form mismatch: "Checks for a strong-distortion regime and returns warnings." |
| 1025 | F11 | LOW | `` `None` — injection happens at all z. `` | Remove spaces around the em dash. |
| 1069 | F11 | LOW | `Axion NWA parameters — always \`None\` without the \`axion\` feature.` | Remove spaces around the em dash. |
| 1317-1318 | F11 | LOW | `` `interp_log_z` / `interp_2d` return 0.0 outside the table — a silent `` | Remove spaces around the em dash. |
| 1614 | F11 | LOW | `` Previous version asserted `rate > 1e-20` — 26 orders of magnitude `` | Remove spaces around the em dash. |
| 1871 | F11 | LOW | `tolerance check — which is what makes it a tight anchor.` | Remove spaces around the em dash. |
| 1614-1615 | T12 | LOW | `A 12-OOM formula bug would have passed.` | "OOM" is never spelled out; the preceding sentence says "orders of magnitude" but doesn't tie it to the abbreviation. Write "A 12-orders-of-magnitude (OOM) formula bug..." at first use, or drop the abbreviation since it is used only once more. |

### Systematic patterns

- A2 (function summaries as noun phrases) recurs 10 times across the `InjectionScenario` accessor
  cluster (lines 246, 462, 880, 1039, 1053, 1069, 1091, 1102, 1141, 1170); most of the file's other
  `fn`/`pub fn` docs ("Loads...", "Warns...", "Computes...", "Returns...") follow the correct
  present-tense-verb pattern, so this is localized, not file-wide.
- Em dashes with surrounding spaces recur 5 times (lines 1025, 1069, 1317, 1614, 1871).
- Citation-style ampersands ("Chluba & Sunyaev", "Bolliet & Chluba", "B&C 2021") are DOMAIN, not
  flagged: author-list conventions in physics citations, not prose symbol substitution.
- Reaction arrows (X → γγ, γ ↔ A′, γ ↔ a) are DOMAIN per the rubric's own example (γe → γγe).
- Physical-quantity units are stated consistently and well throughout (eV, 1/s, GeV⁻¹, nG, [1/s]) —
  this file handles A4's units requirement better than most.

## src/output.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 747-749 | T2 | MEDIUM | `Used to catch the audit H7 class of bug (\`}}]}\` emitting one too many closing braces) without` | Name the actor: "Catches the audit H7 class of bug..." |
| 396 | T6 | LOW | `Pre-warnings format was a bare JSON array.` | Rephrase without the temporal framing, for example "Without warnings, the output is a bare JSON array." |
| 750 | F1 | LOW | `taking a serde_json dev-dependency.` | Put the crate name in code font: `` `serde_json` ``. |

### Systematic patterns

- Only one passive "Used to ..." instance in this file, matching the same construction in `cli.rs`
  line 1346 (see cross-file pattern above).
- Abbreviations (PDE, GF) are never contracted in this file — always spelled out ("partial
  differential equation", "Green's function") even on repeat use. Not a rubric violation, just an
  observed consistency choice.

## src/cli.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 852 (help text, `print_help`) | T12 | HIGH | `println!("spectroxide: CMB spectral distortion solver");` | Expand at first use in the printed banner: "spectroxide: cosmic microwave background (CMB) spectral distortion solver". |
| 859 (help text, `print_help`) | T12 | HIGH | `println!("  solve <injection-type>  Single PDE solve for one injection scenario");` | Expand at first use in the printed help: "Single partial differential equation (PDE) solve..."; PDE otherwise appears unexplained throughout the printed help (lines 860, 861, 863, 934, 977, 1000, 1034, 1036, ...). |
| 772-775 | F11 | MEDIUM | `Returns \`Err\` if the preset name is not \`default\`, \`planck2015\`, or \`planck2018\`; if only\n/// one of \`omega_b\` and \`omega_m\` is given; if either lies outside [0, 1] or Ω_m < Ω_b; or\n/// if [\`crate::cosmology::Cosmology::new\`] rejects the resulting parameters.` (`# Errors`) | Convert the semicolon-joined conditions into a bulleted list, one condition per item. |
| 1346 | T2 | MEDIUM | `Used to compress repeated per-worker warnings (for example, one identical` | Name the actor: "Compresses repeated per-worker warnings..." |
| 1465-1470 | F11 | MEDIUM | `Returns \`Err\` if the cosmology options are invalid (see [\`build_cosmology\`]); if\n/// \`--delta-rho\` or the \`--dn-planck\` amplitude is not a finite number; if the injection type\n/// is unknown or its parameters fail \`InjectionScenario::validate\`; if a resonant-conversion\n/// scenario has no resonance redshift in [50, 3e6]; if the solver or grid configuration fails\n/// validation; or if \`z_start\` lies below, or \`z_end\` above, the upper edge of the injection\n/// window, so that the solve would miss the injection.` (`# Errors`) | Convert the five semicolon-joined conditions into a bulleted list, one condition per item. |
| 215 | T9 | MEDIUM | `Individual flags above override preset values.` (doc comment) | Replace "above" with "preceding" or restructure: "The individual flags in this struct override preset values." |
| 912 | T9 | MEDIUM | `  --cosmology <preset>  default, planck2015, planck2018 (flags below override it)` (help text) | Replace "below" with "following" or name the specific flags. |
| 1036 | T9 | MEDIUM | `Accuracy vs the PDE: 2-5% for mu, ~5% for y; ~8-13% shape error in the` (help text) | Replace "vs" with "compared with" or "against". |
| 1413 | T9 | LOW | `returning the soft warnings that should surface to the caller.` (doc comment) | If surfacing is mandatory, use "must surface"; if optional, say so explicitly. |
| 1589 | T11 | DOMAIN | `cost in the sweeps spans ~128 to ~80,000 solver steps (z_start ≈ z_h + 7σ,` (doc comment) | Order-of-magnitude computational-cost estimate, analogous to a physics approximation; judgment call — leave as is or spell out "approximately". |
| 870 | T11 | DOMAIN | `  z        redshift (dimensionless); mu-era z > ~5e4, y-era z < ~1e4` (help text) | "~" marks order-of-magnitude redshift regimes, standard in this domain; judgment call — leave as is. |
| 51, 55, 323, 855, 856, 859, 885, 887, 891, 898 (~44 total) | F3 | DOMAIN | for example `println!("  spectroxide <subcommand> [options]");` / `--z-start <z>         Starting redshift. Default: 5e6 for solve...` | Angle-bracket placeholders (`<subcommand>`, `<z>`, `<val>`, `<preset>`, `<path>`, `<injection-type>`, etc.) appear roughly 44 times, following the standard POSIX/GNU CLI usage-synopsis convention, each explained inline. Recorded once as a systematic pattern rather than itemized individually; judgment call, DOMAIN. |

### Systematic patterns

- T12: CMB and PDE are used throughout the printed CLI help text (not just source doc comments) and
  are never expanded anywhere in that output — a distinct, user-facing "page" from the source
  comments, and the two HIGH findings above.
- Angle-bracket placeholders are the file's dominant "violation" by raw count but read as standard
  CLI usage syntax (F3 DOMAIN row above) — not itemized further.
- `# Errors` sections with more than two conditions are written as semicolon-joined run-on sentences
  (2 instances flagged above); two shorter `# Errors` blocks (lines 1656-1662, 1759-1767) use commas
  instead and are not flagged.
- "above"/"below" for document position appears twice (doc comment line 215, help text line 912).

## src/lib.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 40 | H1 | MEDIUM | `## Green's Function Mode` | Sentence case: "## Green's function mode". |
| 28 | T9 | MEDIUM | `// Set energy injection (e.g., delta-function burst at z=2×10⁵)` | Replace "e.g." with "for example": "(for example, a delta-function burst at z=2×10⁵)". |
| 54 | T2 | MEDIUM | `Used to invalidate cached Green's function tables when the underlying` | Name the actor: "Invalidates cached Green's function tables when the underlying physics code changes." |
| 60 | T2 | MEDIUM | `the physics is implemented and tested, but it is not part` | Name the actor: "the code implements and tests the physics, but excludes it from the released feature set." |
| 15 | T11 | DOMAIN | `**Double Compton emission**: photon-number changing, γe → γγe` | Reaction-notation arrow; standard physics shorthand, per the rubric's own example. |
| 16 | T11 | DOMAIN | `**Bremsstrahlung**: photon-number changing, e+ion → e+ion+γ` | Same as above. |

## src/main.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 3 | F1 | MEDIUM | `All logic lives in the library (cli.rs, output.rs).` | Code font the file names: "(\`cli.rs\`, \`output.rs\`)". |
| 89 | F1 | MEDIUM | `Gets the output writer: file if --output specified, stdout otherwise.` | Code font the flag: "file if \`--output\` specified, stdout otherwise." |
| 100 | F1 | MEDIUM | `Writes any result type to the configured output destination and format.` (paired with the un-coded `write_output`/`execute_*` references in the module header at line 4) | Code font the function-name pattern in the module header: "calls \`execute_*\`". |
| 4 | T5 | LOW | `This binary just parses args, calls execute_*, and writes output.` | Drop "just": "This binary parses args, calls \`execute_*\`, and writes output." |

## src/bin/check_adiabatic.rs

No confirmed violations in the module's `//!` doc comment (lines 1-5).

## examples/*.rs headers

| File:Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| examples/temporal_error_check.rs:1 | F6 | MEDIUM | `//! One-off: measure the temporal discretization error of μ at the DEFAULT` | Replace all-caps emphasis with code font or plain wording: "at the default \`dy_max = 0.02\`". |
| examples/temporal_error_check.rs:3 | F1 | MEDIUM | `//! reference. Scenario matches tests/convergence_order.rs::run_full_physics.` | Code font the path/identifier: "\`tests/convergence_order.rs::run_full_physics\`". |
| examples/generate_parity_fixtures.rs:1 | A3 | MEDIUM | `//! Generate Rust-to-Python parity fixtures (validation-audit Part B2).` | Match the noun-phrase/descriptive convention used in the other four example headers (`custom_injection.rs`: "Example: ..."; `energy_budget.rs`: "Energy-conservation budget for..."): "Rust-to-Python parity fixture generator (validation-audit Part B2)." |
| examples/temporal_error_check.rs:1 | T14 | LOW | `//! One-off: measure the temporal discretization error of μ at the DEFAULT` | Sentence fragment; write as a complete sentence: "This one-off example measures the temporal discretization error of μ..." |
| examples/energy_budget.rs:7-8 | F11 | LOW | `contribute ≲10⁻⁴ —` (line-wraps to `the deviation is not a bookkeeping error;`) | Remove the space before the em dash so it reads unspaced when the doc-comment lines join: "≲10⁻⁴—the deviation...". |

### Systematic patterns

- `custom_injection.rs`, `photon_diag.rs`, and `check_adiabatic.rs` have no confirmed violations in
  their headers.
- Style is inconsistent across the five example headers on whether the opening line is a noun phrase
  ("Example: ...", "Diagnostic for...", "Energy-conservation budget for...") or an imperative /
  fragment ("Generate...", "One-off: measure..."); flagged individually above (A3, T14).
