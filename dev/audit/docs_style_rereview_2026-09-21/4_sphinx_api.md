# Review: docs/api/*.rst against the Google developer documentation style guide

Scope: `axion.rst`, `cosmology.rst`, `dark_photon.rst`, `firas.rst`, `greens.rst`, `greens_table.rst`,
`index.rst`, `solver.rst`, `style.rst` (1,279 lines total). Read-only review; rule IDs and severities
from the rubric.

## Summary table (counts by severity)

| Severity | Count |
|---|---|
| HIGH | 0 |
| MEDIUM | 32 |
| LOW | 42 |
| DOMAIN | 11 |
| **Total** | **85** |

No finding reached HIGH: every page has an introductory paragraph, every abbreviation central to a
page is spelled out at first use (several link `:term:`), no list is a broken procedure, no
placeholder is left unexplained, and the manually documented callback parameters in `solver.rst` are
complete. The issues found are consistent, repeated style deviations rather than comprehension
blockers.

## Systematic patterns

1. **Code font in page titles (H3).** Every page title except `index.rst` embeds the Python module
   path in double backticks, e.g. `Axion helpers (``spectroxide.axion``)`. H3 says to avoid code font
   in headings. 8 instances (one per file: `axion.rst:1`, `cosmology.rst:1`, `dark_photon.rst:1`,
   `firas.rst:1`, `greens.rst:1`, `greens_table.rst:1`, `solver.rst:1`, `style.rst:1`). Two headings in
   `greens_table.rst` (`` ``GreensTable`` `` at line 47, `` ``PhotonGreensTable`` `` at line 70) are
   *entirely* code font, which is the stronger form of the same problem.
2. **Spaced em dashes (F11).** The docs consistently write `word — word` instead of Google's unspaced
   `word—word`. 30 instances across 6 of 9 files: `axion.rst` (4), `dark_photon.rst` (1), `greens.rst`
   (5), `index.rst` (9), `solver.rst` (10), `style.rst` (1). `cosmology.rst`, `firas.rst`, and
   `greens_table.rst` have none. Severity LOW per the rubric's own example, but the volume makes it
   worth a single sweep-wide fix rather than 30 individual edits.
3. **Dropped grammatical subject in module-intro sentences (T14).** `axion.rst:14-16` and
   `dark_photon.rst:6-9` both open with a fragment ("Pure-Python … helpers … conversion.") followed by
   a second sentence that starts with a bare verb but later switches to indicative mood without ever
   naming a subject ("Mirror the Rust `src/axion.rs` routines and reuse …" / "… and are the documented
   route …"). The two files read as copy-paste variants of the same broken construction.
4. **"This page documents …" openings (T4).** `axion.rst:6` and `solver.rst:12` both pre-announce with
   "This page documents …". The other seven files state their content directly instead.

## axion.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | MEDIUM | ``Axion helpers (``spectroxide.axion``)`` | Drop the code-font module path from the heading; keep it in the `.. currentmodule::` directive or the intro sentence. |
| 6 | T4 | MEDIUM | "This page documents pure-Python helpers for resonant axion-photon conversion." | State it directly: "Pure-Python helpers implement resonant axion-photon conversion." |
| 14-16 | T14 | MEDIUM | "Mirror the Rust ``src/axion.rs`` routines and reuse the plasma-frequency machinery of :mod:`spectroxide.dark_photon`" | Restore the subject: "These helpers mirror the Rust ``src/axion.rs`` routines and reuse …". |
| 15 | T11 | LOW | "Cyr, Chluba & Manoj" | Write "Cyr, Chluba, and Manoj". |
| 30, 33 | L1 | MEDIUM | "1. The conversion probability carries ``x`` in the numerator," | These two items are not sequential steps; use a bulleted list, not a numbered one. |
| 19 | F11 | LOW | "the top level — import explicitly:" | Remove the spaces: "top level—import explicitly:". |
| 32 | F11 | LOW | "photons convert preferentially — opposite to the dark photon's ``1/x``." | Remove the spaces around the dash. |
| 41-42 | F11 | LOW | "Eqs. 7–12) — which shift the conversion redshift and produce multiple crossings near recombination —" | Remove the spaces around both dashes. |
| 43 | T2 | MEDIUM | "are not modeled." | Name the actor: "This module does not model them." |
| 15 | DOMAIN | DOMAIN | "``γ ↔ a`` axion–photon conversion" | Standard reaction notation; no fix needed. |

## cosmology.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | MEDIUM | ``Cosmology (``spectroxide.cosmology``)`` | Drop the code-font module path from the heading. |
| 116 | T11 | LOW | "fudge factor ``F = 1.125`` (Chluba & Thomas 2011)." | Write "Chluba and Thomas". |
| 7 | DOMAIN | DOMAIN | "Flat ΛCDM background quantities" | Standard cosmology abbreviation, not in the project glossary but universally recognized; no fix needed. |

This file is otherwise clean: it has a proper intro paragraph, active voice throughout, and no
word-list or symbol violations beyond the one ampersand.

## dark_photon.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | MEDIUM | ``Dark-photon helpers (``spectroxide.dark_photon``)`` | Drop the code-font module path from the heading. |
| 6-9 | T14 | MEDIUM | "Mirror the Rust ``src/dark_photon.rs`` routines and are the documented route to reproduce the dark-photon constraint numbers." | Restore the subject: "These helpers mirror the Rust ``src/dark_photon.rs`` routines and are the documented route …". |
| 9 | F11 | LOW | "Not re-exported at the top level — import explicitly:" | Remove the spaces around the dash. |
| 7 | DOMAIN | DOMAIN | "``γ ↔ A'`` conversion" | Standard reaction notation; no fix needed. |

## firas.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | MEDIUM | ``FIRAS data (``spectroxide.firas``)`` | Drop the code-font module path from the heading. |
| 42 | T5 | LOW | "constants for quick use without constructing a :class:`FIRASData`" | Drop the ease claim: "constants you can use directly, without constructing a :class:`FIRASData` object." |
| 43 | T2 | MEDIUM | "Limits are quoted at the confidence level (:term:`CL`) noted in the table." | Name the actor: "The table quotes limits at the confidence level (:term:`CL`) noted." |
| 9 | DOMAIN | DOMAIN | "the full 43 × 43 frequency-frequency covariance matrix" | Standard matrix-dimension notation; no fix needed. |

`FIRAS` and `CL` are both defined at first use with `:term:` links (lines 7 and 43) — good practice,
not a finding.

## greens.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | MEDIUM | ``Analytic Green's function (``spectroxide.greens``)`` | Drop the code-font module path from the heading. |
| 12 | F11 | LOW | "solver — for production work prefer the :doc:`PDE solver <solver>`." | Remove the spaces around the dash. |
| 17-18 | F11 | LOW | "into three channels — μ, y, and a temperature shift — weighted by" | Remove the spaces around both dashes. |
| 18 | T11 | MEDIUM | "weighted by redshift-dependent visibility / branching functions:" | Drop the slash: "visibility and branching functions". |
| 33 | F11 | LOW | "**Accuracy versus PDE** — ``<5%`` deep μ-era (z_h > 2 × 10⁵)" | Remove the spaces around the dash. |
| 177 | T9 | MEDIUM | "∫dQ/dz dz ≈ Δρ/ρ ~ 1e-5, i.e. the linear regime the Green's function assumes" | Replace "i.e." with "that is". |
| 177 | DOMAIN | DOMAIN | "Δρ/ρ ~ 1e-5" | "~" for order-of-magnitude in a physics statement; no fix needed. |
| 263 | T9 | MEDIUM | "is the entry point — it dispatches via a ``method=`` keyword" | Replace "via" with "through". |
| 263 | F11 | LOW | (same line) "is the entry point — it dispatches via a" | Remove the spaces around the dash. |
| 265 | F11 | LOW | "the linear Gram-Schmidt fit (``\"gs\"``).  Both" | Collapse the double space to one. |
| 265-266 | T11 | LOW | "private (``_decompose_nonlinear_be`` / ``_decompose_gram_schmidt``)" | Replace the slash with "or". |
| 34 | DOMAIN | DOMAIN | "``<1%`` y-era" region described via "μ↔y transition (3 × 10⁴–10⁵)" | Standard regime-transition notation; no fix needed. |
| 300-301 | T2 | MEDIUM | "It is documented on the :doc:`PDE-solver page <solver>` for proximity with the other ``solver`` entry points" | Active: "The PDE-solver page documents it for proximity with the other ``solver`` entry points." |

## greens_table.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | MEDIUM | ``PDE-based numerical Green's function (``spectroxide.greens_table``)`` | Drop the code-font module path from the heading. |
| 47 | H3 | MEDIUM | "``GreensTable``" | Heading is entirely code font; write "The GreensTable class" as prose with the identifier in inline code font, not as the whole heading. |
| 70 | H3 | MEDIUM | "``PhotonGreensTable``" | Same fix. |
| 122 | T11 | LOW | "Builders / loaders" | Write "Builders and loaders". |
| 79 | T5 | LOW | "this is often slower than just running the PDE directly." | Drop "just": "slower than running the PDE directly." |
| 11 | DOMAIN | DOMAIN | "especially in the μ↔y transition region" | Standard regime-transition notation; no fix needed. |

## index.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 43 | F5 | MEDIUM | "the science targets are computed by the PDE solver above." | Replace the backward reference: "the science targets are computed by the PDE solver (see :doc:`solver`)." |
| 43 | T2 | MEDIUM | "the science targets are computed by the PDE solver above." | Active: "The PDE solver computes the science targets described in the preceding section." |
| 26, 49, 59, 68, 76, 86, 94, 103, 113 | F11 | LOW | "PDE solver — :doc:`solver`" (line 26; representative of 9 instances at the lines listed) | Remove the spaces around each dash throughout the grid-card list. |
| 90 | DOMAIN | DOMAIN | "resonant γ↔A' conversion" | Standard reaction notation; no fix needed. |
| 98 | DOMAIN | DOMAIN | "resonant γ↔a conversion" | Standard reaction notation; no fix needed. |
| 64 | DOMAIN | DOMAIN | "the μ↔y transition region" | Standard regime-transition notation; no fix needed. |
| 72 | DOMAIN | DOMAIN | "the full 43 × 43 covariance matrix" | Standard matrix-dimension notation; no fix needed. |

## solver.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | MEDIUM | ``PDE solver (``spectroxide.solver``)`` | Drop the code-font module path from the heading. |
| 12 | T4 | MEDIUM | "This page documents the PDE solver only." | State it directly, or fold into the surrounding cross-reference sentence, e.g. "For the analytic Green's function approximation, see :doc:`greens`; this page covers the PDE solver." |
| 89 | T11 | MEDIUM | "such as ``f_x → --f-x`` and ``delta_n_over_n → --delta-n-over-n``" | Replace the arrow with words: "``f_x`` becomes ``--f-x``". |
| 89 | F3 | MEDIUM | "Rust command-line interface flag ``--<kebab-case>``" | Angle-bracket placeholder, not the UPPER_SNAKE_CASE convention, and the pattern name itself is undefined; write "flag `--PARAMETER-NAME` (its kebab-case form)" and explain PARAMETER-NAME. |
| 112-124 | L1 | MEDIUM | "* **Signature**: ``dq_dz(z) -> float`` (or array)." | This is a term/definition list (Signature, Quantity, Sign, Tabulation grid, Mode); use a description list, not a bulleted list of bold run-in terms. |
| 137-149 | L1 | MEDIUM | "* **Signature**: ``photon_source(x, z) -> float``." | Same fix. |
| 160-162 | F2 | MEDIUM | ".. tab-item:: Single PDE solve\n\n      .. code-block:: python" | Add an introductory sentence before the code block, e.g. "Run a single decaying-particle injection:". |
| 177-179 | F2 | MEDIUM | ".. tab-item:: Redshift sweep\n\n      .. code-block:: python" | Same fix. |
| 191-193 | F2 | MEDIUM | ".. tab-item:: Photon injection\n\n      .. code-block:: python" | Same fix. |
| 204-206 | F2 | MEDIUM | ".. tab-item:: Tabulated heating\n\n      .. code-block:: python" | Same fix. |
| 214-216 | F2 | MEDIUM | ".. tab-item:: Parameter scan\n\n      .. code-block:: python" | Same fix. |
| 98 | F11 | LOW | "``injection`` dict.  The wrapper tabulates both" | Collapse the double space to one. |
| 141 | F11 | LOW | "perturbation.  Dimensionless." | Collapse the double space to one. |
| 175 | F12 | LOW | "x, dn = result.x, result.delta_n     # numpy arrays" | Capitalize the product name in the comment: "# NumPy arrays" (the `import numpy as np` statement itself is correct as-is). |
| 298 | T5 | LOW | "switch to the faster ``DEBUG`` preset for quick checks." | Drop the ease claim: "for faster checks." |
| 38, 41, 45, 50, 102, 114, 126, 142, 221, 286 | F11 | LOW | "One PDE solve for a custom injection scenario, custom heating" (line 38; representative of 10 instances at the lines listed) | Remove the spaces around each dash. |

## style.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | MEDIUM | ``Plotting utilities (``spectroxide.style``, ``spectroxide.plot_params``)`` | Drop the code-font module paths from the heading; two module names in one title is also a readability concern independent of H3. |
| 5 | F11 | LOW | "Not re-exported at the top level — import explicitly:" | Remove the spaces around the dash. |

This file is otherwise clean.
