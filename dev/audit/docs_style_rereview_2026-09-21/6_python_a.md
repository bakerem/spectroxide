# Python docstring/warning review: `python/spectroxide/{__init__,solver,greens,_validation}.py`

Reviewed against `/tmp/.../scratchpad/rubric.md` (Google developer documentation style guide). Scope per brief: docstrings, plus user-facing warning and error strings. Code comments are out of scope and excluded below even where grep initially matched them.

## Summary of counts by severity

| Severity | Count |
|---|---|
| HIGH | 2 |
| MEDIUM | 31 |
| LOW | 21 |
| DOMAIN | 8 |
| **Total** | **62** |

## Systematic patterns

1. **Undefined "GF" / late-defined "PDE"** — `__init__.py` uses "GF" in its flagship Quick-start comment without ever spelling out "Green's function (GF)" anywhere in the file (HIGH). `_validation.py` uses "PDE" in a user-facing warning before the term is spelled out anywhere in that file (HIGH). `solver.py` and `greens.py` each have one further undefined/underdefined "GF" instance (MEDIUM). `_validation.py` also uses "DC/BR" in a warning without local definition.
2. **Noun-phrase function summaries instead of a present-tense verb (rule A2)** — the vast majority of public function docstrings in `greens.py` (20 of ~22) open with a bare noun phrase describing what the return value *is* ("μ-distortion branching ratio.") rather than what the function *does* ("Compute the μ-distortion branching ratio."). This is inconsistent even with the file's own two exceptions (`decompose_distortion`, `delta_n_to_delta_I`, which are verb-led), so it does not qualify for the rubric's NumPy/PEP257 DOMAIN carve-out. `solver.py` has 4 more instances of the same problem (`run_photon_sweep`, `run_photon_sweep_batch`, `run_single`, `solve`) against a much larger fraction of verb-led functions.
3. **Custom prose section headers with no lead-in sentence (rule L2)** — "Conventions" (solver.py, greens.py), "Two-tier validation" (_validation.py), "Two modes of operation", "Accuracy (compared with PDE)", "Sign behavior" (greens.py) each drop straight into a bulleted list with no introductory sentence.
4. **Em dash with surrounding spaces instead of Google's no-space style (rule F11)** — 15 instances across all four files, none exceeding 10 in a single file.
5. **Author-list citations using "&" ("Chluba & Jeong", "Danese & de Zotti", "Hu & Silk", "Bianchini & Fabbian")** — 8 instances in `greens.py`, standard academic citation convention, classified DOMAIN.
6. **Non-rubric aside (not counted in totals):** `_validation.py:279` documents `validate_dq_dz_callable` as sampling "five log-spaced redshifts," but the implementation (line 305) samples 32 points plus the two endpoints. This is a factual/accuracy defect, not a style-rubric violation, so it is not scored, but it is worth fixing alongside the style items.

---

## `__init__.py`

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 16 | T12 | HIGH | `# Unified entry point (handles both GF and PDE modes)` | Spell out "Green's function (GF)" at first use in this file (PDE is defined at line 9; GF never is), or write "Green's function and PDE modes". |
| 12 | T5 | LOW | `Quick start::` | Rename to a neutral label such as "Usage example::" — "quick" reads as an ease claim. |

---

## `solver.py`

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 5 | T5 | LOW | `(PDE) solver from Python (subprocess and JSON over stdout) and for quick` | Drop "quick"; state what the function does ("single-injection calculations"). |
| 862 | A2 | MEDIUM | `"""Photon-injection sweep over multiple ``z_h`` at fixed ``x_inj``.` | Lead with a present-tense verb: "Sweep photon injection over multiple ``z_h`` at fixed ``x_inj``." |
| 997 | A2 | MEDIUM | `"""Batch photon-injection sweep over multiple ``x_inj`` values.` | "Sweep photon injection over multiple ``x_inj`` values in one batch." |
| 1137 | A2 | MEDIUM | `"""Quick calculation using the pure-Python Green's function.` | "Compute a spectral distortion using the pure-Python Green's function." |
| 1137 | T5 | LOW | `"""Quick calculation using the pure-Python Green's function.` | Same fix as above removes "Quick" as a side effect. |
| 1334 | A2 | MEDIUM | `"""Unified entry point for spectral-distortion calculations.` | "Dispatch a spectral-distortion calculation to the PDE solver, Green's function, or table." |
| 1366 | T4 | MEDIUM | `Note that ``delta_rho`` is a top-level argument, not an injection` | Delete "Note that" and state the fact directly: "``delta_rho`` is a top-level argument, not an injection key." |
| 1372 | F5 | MEDIUM | ```axion`` feature, see above)::` | Name the target explicitly instead of "see above", for example "see the Cargo-feature note earlier in this docstring". |
| 1414 | T9 | MEDIUM | `Precomputed table for ``method="table"``.  May be a table object,` | "Precomputed table for ``method=\"table\"``. Can be a table object, ..." |
| 1418 | T12 | MEDIUM | `Lower integration bound for ``dq_dz`` in table/GF mode` | Spell out "Green's function (GF)" or use the already-established `method="greens_function"` name instead of the shorthand "GF". |
| 1531 | T9 | MEDIUM | `"e.g. solve(method='greens_function', z_h=2e5)."` | Replace "e.g." with "for example" in this raised-`ValueError` message. |
| 877 | F11 | LOW | `Injection redshifts.  Default *None* — Rust uses 150 log-spaced` | Remove the spaces around the em dash: "Default *None*—Rust uses...". |
| 1141 | F11 | LOW | `**Single burst** (default) — provide ``z_h`` and ``delta_rho`` for a` | Remove spaces around the em dash. |
| 1148 | F11 | LOW | `**Custom heating** — provide ``dq_dz``, a callable returning` | Remove spaces around the em dash. |
| 1586 | T2 | LOW | `` "`dark_photon_depletion=γ_con` was removed. Use " `` | Active voice: "spectroxide removed `dark_photon_depletion=γ_con`. Use ..." |
| 8–10 | L2 | MEDIUM | `Conventions` / `-----------` / ```` - ``Δρ/ρ`` is the fractional energy injection...` `` | Add a lead sentence before the list, for example "This module uses the following conventions:". |
| 1139–1141 | L2 | MEDIUM | `Two modes of operation` / `----------------------` / `**Single burst** (default) — provide...` | Add a lead sentence, for example "``run_single`` supports two modes of operation:". |

---

## `greens.py`

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 60 | T9 | MEDIUM | `Function to evaluate. Should accept a float or array_like and return` | Requirement, not suggestion: "Must accept a float or array_like and return ...". |
| 471 | T9 | MEDIUM | `convention: positive for heating.  Should accept either a scalar` | "...positive for heating.  Must accept either a scalar ...". |
| 358 | T4 | MEDIUM | `Note that ``J_y ≠ 1 − J_μ`` in the transition region; using the` | Delete "Note that": "``J_y ≠ 1 − J_μ`` in the transition region; the independent fit gives better...". |
| 1108 | F5 | MEDIUM | `the line is still drawn with a minimal width — see above).` | Name the referent instead of "see above", for example "falls back to a narrow ``0.005 x_inj`` Gaussian (below)." or restate the value inline. |
| 1337 | T12 | MEDIUM | `"land in the band are skipped (the photon GF is undefined "` | Spell out "Green's function (GF)" in this warning text — it is shown to users independently of the rest of the module. |
| 1507 | T12 | LOW | ` """Three-component GF ansatz per unit Δρ/ρ.""" ` | Same undefined-abbreviation issue, repeat occurrence; fix once "GF" is defined per finding above. |
| 17 | F11 | LOW | `module) — or any user dict with the same keys.` | Remove spaces around the em dash. |
| 1702 | T11 | LOW | `B&F parameterization using ``δ_BF = δ_GS + μ/β_μ``).  The LM iteration` | Spell out "the Bianchini & Fabbian (B&F) parameterization" at first use of the shorthand, or write "the B&F parameterization" only after introducing "(B&F)" explicitly. |
| 1876 | F6 | LOW | `Cosmic microwave background temperature today, in **K**.  Default` | Drop the bold; code/units don't need emphasis here: "...today, in K. Default ...". |
| 1882 | F6 | LOW | `Frequency in **GHz**.` | "Frequency in GHz." |
| 1884 | F6 | LOW | `Intensity distortion in **Jy/sr** (= 10⁻²⁶ W m⁻² Hz⁻¹ sr⁻¹).` | "Intensity distortion in Jy/sr (= 10⁻²⁶ W m⁻² Hz⁻¹ sr⁻¹)." |
| 133 | A2 | MEDIUM | ``` """Planck (blackbody) occupation number ``n_pl(x) = 1 / (e^x − 1)``.``` | "Compute the Planck (blackbody) occupation number ``n_pl(x) = 1 / (e^x − 1)``." |
| 160 | A2 | MEDIUM | ``` """Blackbody derivative ``G_bb(x) = x e^x / (e^x − 1)^2``.``` | "Compute the blackbody derivative ``G_bb(x) = ...``." |
| 188 | A2 | MEDIUM | ``` """μ-distortion spectral shape ``M(x) = (x/β_μ − 1) · G_bb(x) / x``.``` | "Compute the μ-distortion spectral shape ``M(x) = ...``." |
| 207 | A2 | MEDIUM | `"""y-distortion (Sunyaev–Zel'dovich) spectral shape.` | "Compute the y-distortion (Sunyaev–Zel'dovich) spectral shape." |
| 238 | A2 | MEDIUM | ``` """Temperature shift spectral shape ``G(x) = x e^x / (e^x − 1)^2``.``` | "Compute the temperature-shift spectral shape ``G(x) = ...``." |
| 263 | A2 | MEDIUM | ``` """Thermalization visibility ``J_bb(z) = exp(−(z/z_μ)^{5/2})``.``` | "Compute the thermalization visibility ``J_bb(z) = ...``." |
| 290 | A2 | MEDIUM | `"""Improved thermalization visibility with the Chluba (2015) correction.` | "Compute the improved thermalization visibility with the Chluba (2015) correction." |
| 323 | A2 | MEDIUM | `"""μ-distortion branching ratio.` | "Compute the μ-distortion branching ratio." |
| 347 | A2 | MEDIUM | `"""y-distortion branching ratio.` | "Compute the y-distortion branching ratio." |
| 381 | A2 | MEDIUM | ``` """Three-component Green's function ``G_th(x, z_h)`` (Chluba 2013).``` | "Compute the three-component Green's function ``G_th(x, z_h)`` (Chluba 2013)." |
| — | A2 | MEDIUM | (10 further instances: lines 455, 545, 603, 695, 716, 737, 760, 1077, 1212, 1280 — total 20 in this file) | Apply the same fix pattern (lead with "Compute"/"Return"/similar) throughout. |
| 22 | T11 | DOMAIN | `- Chluba & Jeong (2014), MNRAS 438, 2065 [arXiv:1306.5751].` | Standard citation convention; no change needed. |
| 270 | T11 | DOMAIN | `photon production rate to the Hubble rate (Chluba & Sunyaev 2012), and` | Standard citation convention; no change needed. |
| 271 | T11 | DOMAIN | `5/2 from the DC opacity scaling (Danese & de Zotti 1982; Hu & Silk 1993).` | Standard citation convention; no change needed. |
| 1401 | T11 | DOMAIN | `Default method: Bianchini & Fabbian (2022) nonlinear Bose–Einstein fit` | Standard citation convention; no change needed. |
| 1410 | T11 | DOMAIN | `(Y_SZ, M, G) over the same band (Chluba & Jeong 2014, Appendix A).` | Standard citation convention; no change needed. |
| 1598 | T11 | DOMAIN | `Reference: Chluba & Jeong (2014), arXiv:1306.5751, Appendix A.` | Standard citation convention; no change needed. |
| 1694 | T11 | DOMAIN | `"""Bianchini & Fabbian (2022) nonlinear Bose–Einstein fit.` | Standard citation convention; no change needed. |
| 1707 | T11 | DOMAIN | `Reference: Bianchini & Fabbian (2022), arXiv:2206.02762, Eqs. (1)–(4).` | Standard citation convention; no change needed. |
| 11, 13 | L2 | MEDIUM | `Conventions` / `-----------` / `- Frequency variable: x = h ν / (k_B T_z), dimensionless.` | Add a lead sentence, for example "This module follows the conventions:". |
| 396, 398 | L2 | MEDIUM | `Accuracy (compared with PDE)` / `----------------------------` / `- Deep μ-era (z_h > 2 × 10⁵): spectral shape accurate to <5%.` | Add a lead sentence, for example "Accuracy compared with the PDE solver, by regime:". |
| 1224, 1226 | L2 | MEDIUM | `Sign behavior` / `--------------` / ```` - ``x_inj > x₀`` and ``P_s ≈ 1``: ``μ > 0`` (energy-dominated).` `` | Add a lead sentence, for example "The sign of μ depends on regime:". |

---

## `_validation.py`

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 335–336 | T12 | HIGH | `"regime (J_bb* < 1e-2). Use the PDE solver: "` / `"solve(injection={'type': ..., 'z_h': ...}, delta_rho=...)."` | Spell out "partial differential equation (PDE)" the first time this abbreviation reaches a user, in this warning (currently first spelled out only later, at line 385, in a different function). |
| 369 | T12 | MEDIUM | `f"x_inj={x_inj:.2e}: DC/BR absorption extremely strong at this "` | Spell out "double Compton (DC) and bremsstrahlung (BR)" on first use in this file. |
| 3, 5 | L2 | MEDIUM | `Two-tier validation` / `-------------------` / `- **ERROR** (:class:``ValueError``) — nonsensical inputs that cannot` | Add a lead sentence, for example "Validation in this module has two tiers:". |
| 5 | F11 | LOW | `- **ERROR** (:class:``ValueError``) — nonsensical inputs that cannot` | Remove spaces around the em dash. |
| 7 | F11 | LOW | `- **WARNING** (:func:``warnings.warn``) — inputs in untested or unreliable` | Remove spaces around the em dash. |
| 258 | F11 | LOW | ```` If ``|Δρ/ρ| > 0.01`` — the linearized Kompaneets equation is no` `` | Remove spaces around the em dash. |
| 267 | T3 | LOW | `will be inaccurate for large energy injections.` | Present tense: "results are inaccurate for large energy injections." |
| 387 | F11 | LOW | `so the GF should warn well before that — this function mirrors the PDE` | Remove spaces around the em dash. |
| 418 | F11 | LOW | ```` ``5 × 10⁴ < z_h < 2 × 10⁵`` — residual r-type contributions become` `` | Remove spaces around the em dash. |
| 591 | F11 | LOW | `this path — passing anything but their defaults silently produced the` | Remove spaces around the em dash. |
| 513 | T9 | LOW | `f"Only {z.size} z-injection point(s). Need >= 50 for "` | Avoid the "(s)" plural: "Only {z.size} z-injection points." (the count already disambiguates singular/plural). |

---

**Reply to parent (per brief): counts and HIGH findings only, given below.**
