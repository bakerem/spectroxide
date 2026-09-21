## src/distortion.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 69 | A2 | MEDIUM | `CJ2014 Appendix A: Gram-Schmidt decomposition over a frequency band.` | Lead with a verb: "Performs the CJ2014 Appendix A Gram-Schmidt decomposition over a frequency band." |
| 203 | A2 | MEDIUM | `Bianchini & Fabbian (2022) nonlinear fit: μ inside the BE exponential.` | Lead with a verb: "Fits μ inside the BE exponential, following Bianchini & Fabbian (2022)." |
| 406 | A2 | MEDIUM | `Convenience wrapper: returns (mu, y, delta_t_over_t) tuple.` | Lead with a verb: "Returns (mu, y, delta_t_over_t) as a convenience-wrapper tuple around `decompose_distortion`." |
| 412–413 | A2 | MEDIUM | `Number of grid points falling inside the default μ/y decomposition band` | Lead with a verb: "Counts the grid points falling inside the default μ/y decomposition band..." |
| 221 | T9 | MEDIUM | `which spans the same 3-D subspace as the CJ2014 basis (Y_SZ, M, G) since M(x) = G(x)/β_μ − G(x)/x.` | Replace causal "since" with "because": "...(Y_SZ, M, G) because M(x) = G(x)/β_μ − G(x)/x." |
| 391–392 | F11 | LOW | `so the returned μ is the physical chemical potential (matching FIRAS-convention fits) — not Chluba's orthogonalized "M-shape" μ.` | Remove spaces around the em dash: "...fits)—not Chluba's..." |
| 397–398 | F11 | LOW | `for spectra with support outside span{M, Y_SZ, G_bb} — for example, frozen or locked-in photon-injection bumps` | Remove spaces around the em dash. |

### Systematic patterns

- Abbreviations are handled correctly: PIXIE and FIRAS are spelled out at first use in the module header (lines 5–6) and then used freely; "BE" (Bose-Einstein) and "CL" (confidence level) are likewise expanded at first use (lines 391, 425).
- Four of the file's function-level doc comments (lines 69, 203, 406, 412) open with a colon-definition noun phrase instead of a present-tense verb, inconsistent with the majority of the file's functions (`decompose_distortion`, `firas_fraction`, `intensity_from_delta_n`, `band_weights`), which all correctly start "Decomposes...", "Checks...", "Converts...", "Collects...".
- Only two low-severity em-dash spacing instances found; no "we", "e.g.", "etc.", "via", or other T9/T13 word-list violations detected in this file's doc comments.
