# Documentation review — Python group B

Files reviewed: `python/spectroxide/cosmology.py`, `cosmotherm.py`, `dark_photon.py`, `axion.py`,
`firas.py`, `greens_table.py`, `plot_params.py`, `style.py`. Scope per brief: docstrings (module,
class, function/method) and user-facing warning/error strings (`raise ...`, `warnings.warn(...)`).
Plain `#` code comments are out of scope and excluded; Sphinx `#:` attribute doc-comments are
in scope (they render as API docs) and are included.

Rubric: `/tmp/claude-1000/-home-bakerem-spectroxide/7ed413ee-d36c-42e3-b01d-100d13e6aac3/scratchpad/rubric.md`.

## Summary table

| Severity | Count |
|---|---|
| HIGH | 2 |
| MEDIUM | 9 |
| LOW | 82 |
| DOMAIN | 5 |

LOW breakdown: 18 spaced-em-dash instances (F11), 59 double-space-after-period instances (T15),
2 slash-for-"or" instances (T11), 1 "simply" (T5), 1 inconsistent boolean-literal markup (A6),
1 inconsistent abbreviation introduction (T12), plus a handful of docstring lines over 80
characters (F2, not separately tallied — see Systematic patterns).

## Systematic patterns

1. **Double space after a sentence-ending period (T15 — "one space after periods"), severity LOW.**
   Present in six of the eight files' docstrings/`#:` comments. Counts: `cosmology.py` 15,
   `cosmotherm.py` 8, `dark_photon.py` 5, `axion.py` 0, `firas.py` 10, `greens_table.py` 18,
   `plot_params.py` 0, `style.py` 3. This is the classic Sphinx/reST two-space convention, applied
   consistently, so it reads as a project-wide formatting choice rather than a series of typos —
   still a literal T15 violation if the house style is Google's one-space rule. Not itemized as
   individual findings per file beyond the first 10 rows (see each file's table); fix by a
   project-wide search/replace of `.  ` → `. ` in docstrings if the one-space rule is adopted.

2. **Em dash surrounded by spaces (F11 — Google style wants an unspaced em dash), severity LOW.**
   Present in six of eight files: `cosmology.py` 1, `cosmotherm.py` 7, `dark_photon.py` 0,
   `axion.py` 1, `firas.py` 2, `greens_table.py` 6, `plot_params.py` 0, `style.py` 1. Listed in full
   per file below (none exceeds 10).

3. **One-line docstring summaries as noun-phrase fragments (A2/T14), severity DOMAIN.**
   Every file mixes noun-phrase fragments ("Photon density parameter.", "Hubble rate H(z) in
   1/s.") with imperative-verb summaries ("Solve X²/(1−X) = S for X.", "Find redshift where Saha H
   drops below 0.99."). Neither form is "present-tense third-person" (A2's literal preference), but
   this is the standard NumPy/PEP 257 scientific-docstring idiom, applied uniformly across all
   eight files. Per the rubric's DOMAIN carve-out for this exact conflict, not itemized as
   individual findings — flagging every one-line summary (well over 100 instances) would be
   padding, not signal.

4. **Boolean and `None` defaults use numpydoc italics (`*True*`, `*False*`, `*None*`), not code
   font.** This is the numpydoc style-guide convention (distinguishing Python literals from
   generic prose values) and is applied consistently in `cosmology.py`, `cosmotherm.py`,
   `dark_photon.py`, `firas.py`, and `greens_table.py`. Not a violation (DOMAIN), noted only because
   `style.py` breaks the pattern (see its table, A6 finding).

5. **Private (`_`-prefixed) helper functions frequently have a one-line docstring with no
   Parameters/Returns section** (for example `cosmology.py`'s `_cosmo_h0`, `_thermal_de_broglie`,
   `_solve_saha_quadratic`; `cosmotherm.py`'s `_compute_g_bb_jy`, `_get_cosmotherm_cosmo`). These are
   internal implementation helpers, not part of the public API surface documented for users, and
   the summaries are self-explanatory from the formula/name. Not itemized under A4 — the two A4
   findings below (both in `greens_table.py`) were kept because they affect *public* callers
   (documented `**kwargs` forwarding and a parameter reachable only through it).

## cosmology.py

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 110 | T15 | LOW | ``Cosmology::planck2018`` exactly.  Note Ω_m here is ≈ 0.31377, | Single space after the period. |
| 113 | T15 | LOW | not model.  Anchoring ω_b/ω_cdm/h (the early-universe densities | Single space. |
| 365 | T15 | LOW | (audit M1 / recomb).  Steps from z_prev = z_new + dz down to z_new. | Single space. |
| 446 | T15 | LOW | with fudge factor F = 1.125 (Chluba & Thomas 2011).  The ODE table is | Single space. |
| 454 | T15 | LOW | Cosmological parameters.  Defaults to :data:`DEFAULT_COSMO`. | Single space (recurs identically at lines 506, 534, 558, 582 — 6 of the file's 15 instances are this exact template sentence). |
| 459 | T15 | LOW | Total free-electron fraction (H and He) per hydrogen atom.  Returns | Single space. |
| 160 | F11 | LOW | the Planck paper value 2.7255 K — use | Remove the spaces around the em dash: `K—use`. |

Total T15 double-space instances in this file: 15 (first 6 distinct contexts shown; the remainder
repeat the "Cosmological parameters.  Defaults to :data:\`DEFAULT_COSMO\`." template at lines 506,
534, 558, 582, plus one more). Total F11 spaced-em-dash instances: 1 (shown).

No HIGH or MEDIUM findings in this file. `ionization_fraction`, `hubble`, `n_hydrogen`,
`n_electron`, `omega_gamma`, `rho_gamma`, `cosmic_time`, `baryon_photon_ratio`, and the
`Cosmology` dataclass all have complete Parameters/Returns sections, units stated, and correct
`:data:`/`:func:` cross-references (no "see above/below").

## cosmotherm.py

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 357 | T11 | LOW | Metadata missing Tgin/Tglast/rho arrays. | Spell out the list: "Metadata missing the Tgin, Tglast, or rho arrays." |
| 20 | T12 | LOW | **Green's function database** — precomputed exact GF (large | Unlike the sibling bullets ("Distortion intensity (DI)", "Dark matter (DM)"), this one never writes out "(GF)"; add the parenthetical for consistency: "Green's function (GF) database". |
| 61 | T15 | LOW | Full path to the DI file.  Either \`\`path\`\` or \`\`name\`\` must be | Single space. |
| 182 | T15 | LOW | If the database file does not exist at the resolved path.  The | Single space. |
| 534 | T15 | LOW | ΔT/T is unobservable.  CosmoTherm therefore defines the *distortion* | Single space. |
| 602 | T15 | LOW | Cosmological parameters.  Defaults to | Single space (recurs at 644, 682). |
| 753 | T15 | LOW | :func:\`load_greens_database\`).  The database is loaded before | Single space. |
| 758 | T15 | LOW | If \`\`params\`\` lacks a key that the scenario needs.  This surfaces | Single space. |
| 18 | F11 | LOW | tortion intensity (DI) files\*\* — predicted cosmic microwave | Unspaced em dash. |
| 20 | F11 | LOW | \*\*Green's function database\*\* — precomputed exact GF (large | Unspaced em dash. |
| 22 | F11 | LOW | er (DM) heating-rate helpers\*\* — CosmoTherm-convention | Unspaced em dash. |
| 169 | F11 | LOW | Green's function G_th(x, z_h) — the residual mu+y spectral | Unspaced em dash. |
| 174 | F11 | LOW | \`\`tgin\`\`: ndarray, shape (N_z,) — initial blackbody temperature | Unspaced em dash. |
| 175 | F11 | LOW | \`\`tglast\`\`: ndarray, shape (N_z,) — last blackbody temperature [K] | Unspaced em dash. |
| 177 | F11 | LOW | \`\`rho\`\`: ndarray, shape (N_z,) — Delta-rho/rho used for each entry | Unspaced em dash. |

Total T15 double-space instances: 8 (first 6 distinct contexts shown; 602's template repeats at
644 and 682). Total F11 spaced-em-dash instances: 7 (all shown).

No HIGH findings. No other MEDIUM findings: `load_di_file`, `di_to_delta_n`,
`load_greens_database`, `reconstruct_full_gf`, `cosmotherm_gf_to_delta_n`,
`convolve_cosmotherm_gf`, `strip_gbb`, the `ct_heating_rate_*` functions, and
`cosmotherm_gf_distortion` all document parameters (including the grouped-parameter numpydoc
form, e.g. "z_h, x, g_th : ndarray, optional"), units, and raised exceptions correctly. The
module-level `.. warning::` box correctly uses one non-stacked notice at the right severity (F7).

## dark_photon.py

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 56 | T15 | LOW | Cosmological parameters.  Defaults to | Single space (same template recurs at lines 84, 120, 160, 195 — all 5 instances in the file). |

No HIGH or MEDIUM findings. `plasma_frequency_ev`, `resonance_redshift`, `dln_omega_pl_sq_dlna`,
`gamma_con`, and `gc_per_epsilon_sq` all have complete Parameters/Returns sections with units and
correctly formatted equations (each display equation is glossed by adjoining prose, satisfying the
math–prose pairing convention). No spaced em dashes in this file.

## axion.py

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 7 | T9 / F5 | MEDIUM | equation path below only works if the binary was built with | "Below" refers to document position; replace with a direct reference, e.g. "the partial differential equation path (see `gamma_con_axion`) only works if...". |
| 23 | F11 | LOW | convert preferentially — opposite to the dark photon's | Unspaced em dash. |

No HIGH findings. `kappa_ev` and `gamma_con_axion` document parameters, units, and the return
tuple correctly, with the governing equation glossed by the surrounding text.

## firas.py

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 11 | F4 | MEDIUM | From https://lambda.gsfc.nasa.gov/product/cobe/firas_products.html: | Use descriptive text with the URL as a proper link, e.g. "From the LAMBDA FIRAS products page:" with the URL attached to that phrase, not written out bare in prose. |
| 556 | F5 / T9 | MEDIUM | literature limit. See the warning below. | Vague positional reference; either fold the warning content into this sentence or omit the pointer — the `.. warning::` block already immediately follows in the rendered docs. |
| 96 | T3 | MEDIUM | '{old}' is deprecated and will be removed in the next | Present it without "will": "is deprecated; removed in the next minor release." |
| 533 | T3 | MEDIUM | Emits a \`\`DeprecationWarning\`\`.  The alias will be removed in the | Same fix as line 96 (also has the T15 double-space issue on this line). |
| 537 | T3 | MEDIUM | 'fit_amplitude_marginalised' is deprecated and will be removed | Same fix as line 96. |
| 76 | T15 | LOW | Emits \`\`DeprecationWarning\`\` for each deprecated name.  Raises | Single space. |
| 77 | T15 | LOW | \`\`TypeError\`\` if a call passes both spellings of one argument.  Only | Single space. |
| 230 | T15 | LOW | cosmological fit.  The absolute normalization here is arbitrary | Single space. |
| 452 | T15 | LOW | level.  For a two-sided bound at 95% CL, \`\`z_{cl} ≈ 1.96\`\`. | Single space. |
| 463 | T15 | LOW | Confidence level (default 0.95).  Must lie in \`\`(0, 1)\`\`. | Single space. |
| 492 | T15 | LOW | Nuisance templates to marginalize over.  Default | Single space. |
| 668 | T15 | LOW | frequencies.  If array_like, shape (43,), \`\`Δn\`\` at the FIRAS | Single space. |
| 31 | F11 | LOW | Two distinct — and mutually inconsistent — limit conventions coexist in | Two unspaced em dashes needed here. |
| 493 | F11 | LOW | \`\`[G_bb]\`\` — the temperature shift is always | Unspaced em dash. |
| 37 | T11 | DOMAIN | separately); the joint-marginalization defaults are ~1.8× looser. | Standard physics shorthand for an approximate ratio; leave as is. |
| 554 | T11 | DOMAIN | is ~1.8× looser than the module constant :data:\`MU_FIRAS_95\` = 9e-5 | Same as above. |
| 567 | T11 | DOMAIN | by ~82% under joint marginalization, giving a ~1.8× looser | Same as above. |
| 614 | T11 | DOMAIN | *separately*, and the μ–y degeneracy inflates σ_y by ~82% | Same as above. |

Total T15 double-space instances: 10 (first 7 shown; remaining at lines 669, 672, 768). Total F11
spaced-em-dash instances: 2 (both shown). Total T11 "~" approximation instances: 4 (all shown,
DOMAIN — standard physics usage for an order-of-magnitude/approximate factor, not flagged as
requiring a fix).

`FIRASData` and its methods otherwise document parameters, return dicts (with every key named),
booleans ("If *True* (default), ..."), and raised/emitted warnings correctly; the two competing
95%-CL conventions are flagged to the reader explicitly in the module docstring, which is good
practice, not a violation.

## greens_table.py

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 1117-1118 | A4/A5 | HIGH | Forwarded to the private \`\`_build_greens_table\`\` builder when a new table needs to be generated (for example, \`\`z_h_grid\`\`, \`\`n_threads\`\`). | Neither \`z_h_grid\` nor \`n_threads\` is a parameter of \`_build_greens_table\` (its signature has \`z_injections\`, \`delta_rho\`, \`n_points\`, \`x_min\`, \`x_max\`, \`n_x\`, \`z_end\`, \`cosmo_params\`, \`number_conserving\`, \`cache_path\`, \`timeout\`, \`progress\`, \`checkpoint\`, \`dy_max\` — no thread-count parameter exists at all). Replace the example with real parameter names, e.g. "(for example, \`z_injections\`, \`dy_max\`)". |
| 1163-1165 | A4/A5 | HIGH | Forwarded to the private \`\`_build_photon_greens_table\`\` builder when a new table needs to be generated (for example, \`\`x_inj_grid\`\`, \`\`z_h_grid\`\`, \`\`n_threads\`\`). | Same problem: the actual parameters are \`x_inj_values\`, \`z_injections\`, \`delta_n_over_n\`, \`n_points\`, \`x_min\`, \`x_max\`, \`n_x\`, \`z_end\`, \`cosmo_params\`, \`number_conserving\`, \`cache_path\`, \`timeout\`, \`progress\`, \`checkpoint\`. Fix the example names to match. |
| 672-728 | A4 | MEDIUM | checkpoint : bool, optional / Enable checkpointing (default *True*). / (Returns) | \`_build_greens_table\`'s signature (line 686) has a \`dy_max: float \| None = None\` parameter that is never documented in the Parameters section, and is not even named in the (also-wrong) \`**kwargs\` examples above — a caller has no way to discover it from the docs. Add a \`dy_max : float, optional\` entry. |
| 56 | T2 | MEDIUM | Warned when a cached Green's function table was built by a different | Passive construction with no stated agent; rewrite active, e.g. "Warns that a cached Green's function table was built by a different physics-code version...". |
| 5-9 | L2 | MEDIUM | Two table classes / ----------------- / - :class:\`GreensTable\` — 2-D heating Green's function \`\`G_th(x, z_h)\`\`. | The bulleted list is introduced only by a section heading, not a complete sentence ending in a colon. Add an intro sentence, e.g. "This module provides two table classes:". |
| 429-430 | T11 | LOW | warning if not.  Disable for synthetic / test tables that | Spell out: "Disable this for synthetic or test tables that...". (Also has the T15 double-space issue on the same line.) |
| 13 | T15 | LOW | convolution of arbitrary injection histories.  This eliminates the | Single space. |
| 160 | T15 | LOW | each frequency point.  Callers wanting a number-conserving result | Single space. |
| 391 | T15 | LOW | Output path.  Default | Single space (recurs at line 604 — same template). |
| 424 | T15 | LOW | Input path.  Default \`\`~/.spectroxide/greens_table.npz\`\`. | Single space (recurs at line 635). |
| 696 | T15 | LOW | each chunk.  If interrupted, resumes from the last completed chunk | Single space. |
| 702 | T15 | LOW | Injection redshifts.  Default *None* — uses 150 log-spaced | Single space (also contains the F11 spaced-em-dash pattern). |
| 717 | T15 | LOW | Cosmological parameters.  Default *None* (Rust defaults). | Single space. |
| 7 | F11 | LOW | - :class:\`GreensTable\` — 2-D heating Green's function \`\`G_th(x, z_h)\`\`. | Unspaced em dash. |
| 8 | F11 | LOW | - :class:\`PhotonGreensTable\` — 3-D photon-injection Green's function | Unspaced em dash. |
| 187 | F11 | LOW | Green's function is consulted — the result depends | Unspaced em dash. |
| 702 | F11 | LOW | Injection redshifts.  Default *None* — uses 150 log-spaced | Unspaced em dash. |
| 920 | F11 | LOW | frequencies.  Default *None* — 10 log-spaced points | Unspaced em dash. |
| 923 | F11 | LOW | redshifts.  Default *None* — 150 log-spaced points | Unspaced em dash. |
| 14 | T11 | DOMAIN | ~8–13% shape errors of the analytic Green's function in the μ-to-y | Standard physics shorthand for an approximate range; leave as is. |

Total T15 double-space instances: 18 (first 7 shown; remainder repeat the "Output path./Input
path. Default ..." templates and similar across the two table classes' `save`/`load` methods).
Total F11 spaced-em-dash instances: 6 (all shown).

No other findings: `GreensTable`, `PhotonGreensTable`, and the build/load functions otherwise
document every parameter, return value, and warning correctly, including grouped boolean
defaults and the checkpoint/resume behavior.

## plot_params.py

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 11 | T5 | LOW | Or simply:: | "Simply" claims ease; drop it: "Or, equivalently::" or just "Alternatively::". |

No HIGH or MEDIUM findings. No double-space or spaced-em-dash instances in this file's short
docstring.

## style.py

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 32-34 | A6 | LOW | usetex : bool / If True (default), use LaTeX for all text rendering with the / custom preamble.  Set False to fall back to mathtext (faster | Every other file in this group italicizes Python literals in numpydoc Parameters sections (\`*True*\`, \`*False*\`, \`*None*\`); this docstring writes them as plain text. For consistency, use \`*True*\`/\`*False*\`. |
| 34 | T15 | LOW | custom preamble.  Set False to fall back to mathtext (faster | Single space. |
| 40 | T15 | LOW | If Matplotlib is not installed.  It is an optional dependency: | Single space. |
| 47 | T15 | LOW | (dvipng is needed too but is not checked here).  The error message | Single space. |
| 43 | F11 | LOW | without it — \`\`spectroxide/__init__.py\`\` re-exports this module | Unspaced em dash. |

Total T15 double-space instances: 3 (all shown). Total F11 spaced-em-dash instances: 1 (shown).

No HIGH or MEDIUM findings. `apply_style`'s `RuntimeError` message is a good example of correct
T8 ordering ("To skip LaTeX rendering, use: apply_style(usetex=False)") and gives three concrete,
platform-specific install commands rather than a vague pointer — no F5/F4 issues.
