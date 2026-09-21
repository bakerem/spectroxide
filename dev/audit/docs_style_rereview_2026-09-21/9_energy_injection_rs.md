## src/energy_injection.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 1170 | A2 | MEDIUM | `Checks for strong distortion regime and return warnings.` | Fix verb agreement: "Checks for the strong-distortion regime and returns warnings." |
| 190 | F11 | LOW | `because it follows the solver's d/dz < 0 (time-forward) convention —` | Remove the space before the em dash (Google style: no surrounding spaces). |
| 1025 | F11 | LOW | `` `None` — injection happens at all z. `` | Remove spaces around the em dash. |
| 1069 | F11 | LOW | `Axion NWA parameters — always \`None\` without the \`axion\` feature.` | Remove spaces around the em dash. |
| 1317 | F11 | LOW | `` `interp_log_z` / `interp_2d` return 0.0 outside the table — a silent `` | Remove spaces around the em dash. |
| 1614 | F11 | LOW | `` Previous version asserted `rate > 1e-20` — 26 orders of magnitude `` | Remove spaces around the em dash. |
| 1871 | F11 | LOW | `tolerance check — which is what makes it a tight anchor. The pre-audit` | Remove spaces around the em dash. |

### Systematic patterns

- Abbreviation discipline is good throughout: CLI, NWA, IC, DC, BR, PDE, CL are each spelled out at first use before being abbreviated, and the module header (lines 1–17) gives a clear scope statement naming all scenario kinds.
- Function docs overwhelmingly lead with a present-tense verb ("Computes...", "Interpolates...", "Loads...", "Warns...", "Suggest..." — the last one is itself a minor tense slip, "Suggests," at line 1141, not tabled above as it's a single low-impact instance). The 1170 instance above is the one clear internal-parallelism break worth flagging.
- `# Errors` sections (`load_heating_table`, `load_photon_source_table`) are complete and well-formed: each documents every distinct failure mode as a bulleted list introduced by a colon sentence, satisfying A4 and L2.
- 6 instances of em dash with surrounding spaces (Google style prefers no surrounding spaces); all 6 listed above (fewer than 10 total).
- No "we", "e.g.", "i.e.", "etc.", "via", "utilize", or causal "since"/"as" misuse found in this file's doc comments.
