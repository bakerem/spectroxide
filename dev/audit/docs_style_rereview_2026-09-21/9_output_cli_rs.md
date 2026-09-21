# Doc-comment / help-text review: src/output.rs and src/cli.rs

## src/output.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 396 | T6 | LOW | "Pre-warnings format was a bare JSON array." | Rephrase without the temporal framing, for example "Without warnings, the output is a bare JSON array." |
| 747-749 | T2 | MEDIUM | "Used to catch the audit H7\n/// class of bug (`}}]}` emitting one too many closing braces) without" | Name the actor: "Catches the audit H7 class of bug..." |
| 750 | F1 | LOW | "taking a serde_json dev-dependency." | Put the crate name in code font: `` `serde_json` ``. |

Systematic patterns:
- Only one passive "Used to ..." instance in this file, but the same construction recurs in `src/cli.rs` line 1346 — worth a single project-wide pass if the phrasing is intentional house style.
- Abbreviations (PDE, GF) are never contracted in this file — always spelled out ("partial differential equation", "Green's function") even on repeat use. Not a rubric violation (T12 only requires expansion at first use), just an observed consistency choice.

## src/cli.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 215 | T9 | MEDIUM | "Individual flags above override preset values." (doc comment) | Replace "above" with "preceding" or restructure: "The individual flags in this struct override preset values." |
| 772-775 | F11 | MEDIUM | "Returns `Err` if the preset name is not `default`, `planck2015`, or `planck2018`; if only\n/// one of `omega_b` and `omega_m` is given; if either lies outside [0, 1] or Ω_m < Ω_b; or\n/// if [`crate::cosmology::Cosmology::new`] rejects the resulting parameters." (doc comment, `# Errors`) | Convert the semicolon-joined conditions into a bulleted list, one condition per item. |
| 1346 | T2 | MEDIUM | "Used to compress repeated per-worker warnings (for example, one identical" (doc comment) | Name the actor: "Compresses repeated per-worker warnings..." |
| 1413 | T9 | LOW | "returning the soft warnings that should surface to the caller." (doc comment) | If surfacing is mandatory, use "must surface"; if optional, say so explicitly. |
| 1465-1470 | F11 | MEDIUM | "Returns `Err` if the cosmology options are invalid (see [`build_cosmology`]); if\n/// `--delta-rho` or the `--dn-planck` amplitude is not a finite number; if the injection type\n/// is unknown or its parameters fail `InjectionScenario::validate`; if a resonant-conversion\n/// scenario has no resonance redshift in [50, 3e6]; if the solver or grid configuration fails\n/// validation; or if `z_start` lies below, or `z_end` above, the upper edge of the injection\n/// window, so that the solve would miss the injection." (doc comment, `# Errors`) | Convert the five semicolon-joined conditions into a bulleted list, one condition per item. |
| 912 | T9 | MEDIUM | "  --cosmology <preset>  default, planck2015, planck2018 (flags below override it)" (help text) | Replace "below" with "following" or name the specific flags. |
| 1036 | T9 | MEDIUM | "Accuracy vs the PDE: 2-5% for mu, ~5% for y; ~8-13% shape error in the" (help text) | Replace "vs" with "compared with" or "against". |
| 1589 | T11 | DOMAIN | "cost in the sweeps spans ~128 to ~80,000 solver steps (z_start ≈ z_h + 7σ," (doc comment) | Judgment call: order-of-magnitude computational-cost estimate, analogous to a physics approximation; leave as is or spell out "approximately" if flagged for cleanup. |
| 870 | T11 | DOMAIN | "  z        redshift (dimensionless); mu-era z > ~5e4, y-era z < ~1e4" (help text) | Judgment call: "~" marks order-of-magnitude redshift regimes, standard in this domain; leave as is. |
| 51, 55, 323, 855, 856, 859, 885, 887, 891, 898 | F3 | DOMAIN | e.g. "Print detailed help for one subcommand (`spectroxide <sub> --help`)." / "  spectroxide <subcommand> [options]" / "  --z-start <z>         Starting redshift. Default: 5e6 for solve (or z_res for" | Angle-bracket placeholders (`<subcommand>`, `<z>`, `<val>`, `<preset>`, `<path>`, `<injection-type>`, etc.) appear roughly 44 times across doc comments and help text. This is the standard POSIX/GNU command-line usage-synopsis convention, not free-form documentation prose; each placeholder is explained inline in the same line. Flagging every instance would be noise — recorded once as a systematic pattern, severity DOMAIN (judgment call, per rubric). Total instances: approximately 44 (3 in doc comments, ~41 in the help-text block). |

Systematic patterns:
- Angle-bracket placeholders (`<val>`, `<z>`, `<preset>`, `<subcommand>`, ...) are the file's dominant "violation" by raw count but read as standard CLI usage syntax (see F3 DOMAIN row above) — not itemized further.
- `# Errors` sections with more than two conditions are written as semicolon-joined run-on sentences (2 instances flagged: lines 772-775 and 1465-1470; two shorter `# Errors` blocks at lines 1656-1662 and 1759-1767 use commas instead and are not flagged).
- "above"/"below" for document position appears twice (doc comment line 215, help text line 912) — both flagged individually above.
