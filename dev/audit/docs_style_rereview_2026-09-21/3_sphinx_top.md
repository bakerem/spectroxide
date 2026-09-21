# Documentation review: docs/index.rst, docs/installation.rst, docs/cli.rst, docs/rust_api.rst, docs/tutorials/index.rst, docs/glossary.rst

Reviewed against `/tmp/claude-1000/-home-bakerem-spectroxide/7ed413ee-d36c-42e3-b01d-100d13e6aac3/scratchpad/rubric.md`
(Google developer documentation style guide highlights). Read-only; no files edited; no other reviewer's notes consulted.

## Summary

| Severity | Count |
|---|---|
| HIGH | 1 |
| MEDIUM | 35 |
| LOW | 12 |
| DOMAIN | 2 |
| **Total** | **50** |

## Systematic patterns

- **Passive voice without a stated actor (T2)** — 5 instances, one per occurrence in `index.rst` (×2), `installation.rst`, `cli.rst`, `rust_api.rst`. The docs otherwise favor active voice and second person; these read as leftover "textbook" phrasing.
- **"above/below" for document position (T9)** — 2 instances (`cli.rst:12`, `tutorials/index.rst:18`), both fixable by "following."
- **Em dash with surrounding spaces (F11)** — 8 instances: a code comment in `index.rst`, one prose sentence in `cli.rst`, and the `---` separator repeated in all 6 rows of the `tutorials/index.rst` list-table.
- **Code font used for entire heading text (H3)** — all 6 subcommand headings in `cli.rst` (` ``solve`` `, ` ``sweep`` `, etc.) are backticked in full; code font belongs on the flag/command names in body text, not as the heading itself.
- **Package names given as plain lowercase text instead of code font (F1)** — the "Adds" column of the extras table in `installation.rst` (4 rows: `matplotlib`, `matplotlib, jupyter`, `matplotlib, jupyter, pytest, mutmut`, `sphinx, pydata-sphinx-theme, nbsphinx, nbsphinx-link, sphinx-copybutton, ipython`), inconsistent with the same page's own use of `` ``numpy`` `` / `` ``scipy`` `` code font two lines above the table.
- **Bold used for emphasis, not a UI element (F6)** — the `tutorials/index.rst` list-table bolds both the notebook number and title in every row (12 instances across 6 rows); this pattern repeats more than 10 times, so the table below lists the first 10 and gives the total.

## docs/index.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 11–12 | T2 | MEDIUM | "The core solver is implemented in Rust and integrated through Python" | Name the actor: "Rust implements the core solver; Python integrates it." |
| 12–13 | T2 | MEDIUM | "an analytic Green's-function approximation (Chluba 2013) is also provided for fast estimates" | "spectroxide also provides an analytic Green's-function approximation (Chluba 2013) for fast estimates." |
| 28 | F11 | LOW | "# Full Rust PDE — single burst at z_h = 2e5, Δρ/ρ = 1e-5" | Remove the spaces around the em dash: "PDE—single burst." |
| 29 | F2 | LOW | `result = solve(injection={"type": "single_burst", "z_h": 2e5}, delta_rho=1e-5)` | Wrap the call onto two lines to stay at or under 80 characters. |

## docs/installation.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 7 | H2 | MEDIUM | "Quick install (recommended)" | Use an imperative task heading, for example "Install with the script." |
| 23 | T2 | MEDIUM | "``numpy`` and ``scipy`` are required by the Python package itself and are always installed." | "The Python package always installs numpy and scipy." |
| 34 | F1 | MEDIUM | "matplotlib" | Put the package name in code font: `` ``matplotlib`` ``. |
| 37 | F1 | MEDIUM | "matplotlib, jupyter" | Code-font each package name: `` ``matplotlib``, ``jupyter`` ``. |
| 40 | F1 | MEDIUM | "matplotlib, jupyter, pytest, mutmut" | Code-font each package name. |
| 43 | F1 | MEDIUM | "sphinx, pydata-sphinx-theme, nbsphinx, nbsphinx-link, sphinx-copybutton, ipython" | Code-font each package name. |
| 49 | H2 | MEDIUM | "Manual installation" | Use an imperative task heading, for example "Install manually" or "Install from source." |
| 116 | F2 | LOW | `python -c "from spectroxide import run_single; print(run_single(z_h=2e5, delta_rho=1e-5))"` | Shorten the inline script or wrap it across lines. |

## docs/cli.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | LOW | "CLI reference" | The abbreviation is not yet defined when the heading appears (definition follows in the first sentence); consider deferring the abbreviation to the body or accepting it as a page-title exception. |
| 12 | T9 | MEDIUM | "Replace ``SUBCOMMAND`` with one of the subcommands below, and ``OPTIONS`` with" | "...one of the following subcommands..." |
| 19 | H3 | MEDIUM | "``solve``" | Drop the code font from the heading text itself; keep it in the body prose. |
| 59 | H3 | MEDIUM | "``sweep``" | Same fix. |
| 68 | H3 | MEDIUM | "``photon-sweep``" | Same fix. |
| 78 | H3 | MEDIUM | "``photon-sweep-batch``" | Same fix. |
| 87 | H3 | MEDIUM | "``greens``" | Same fix. |
| 96 | H3 | MEDIUM | "``info``" | Same fix. |
| 132 | F8 | MEDIUM | "Use the high-resolution production grid preset (4000 points)." | "4,000 points." |
| 147 | T12 | MEDIUM | "Operator-split DC/BR instead of coupled Newton iteration." | Link the abbreviations (`:term:`DC``, `:term:`BR``) or spell them out at their first bare use in this document; the row two above spells out "double Compton and bremsstrahlung" but never introduces the shorthand "DC/BR" itself. |
| 169 | F2 | LOW | `   --omega-b 0.044  --omega-m 0.26  --h 0.71  --y-p 0.24  --t-cmb 2.726  --n-eff 3.046` | Wrap onto two lines to stay at or under 80 characters. |
| 179 | T2 | MEDIUM | "``--omega-b`` and ``--omega-m`` must be supplied together — the CLI derives the CDM" | "You must supply --omega-b and --omega-m together; the CLI derives the CDM density from their difference." |
| 179 | F11 | LOW | "must be supplied together — the CLI derives the CDM" | Remove the spaces around the em dash. |

## docs/rust_api.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 26 | T2 | MEDIUM | "Built from the current source tree and served alongside this site." | Name the actor: "``make -C docs html`` builds this from the current source tree and serves it alongside this site." |

## docs/tutorials/index.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 18 | T9 | MEDIUM | "The table below covers the partial differential equation (:term:`PDE`)" | "The following table covers..." |
| 25 | L5 | MEDIUM | ":header-rows: 0" | Add a header row (for example "Notebook" / "Covers") instead of suppressing it. |
| 27 | F6 | MEDIUM | "**01**" | Do not bold table-cell text for emphasis; use plain text (the code number is already distinguished by column position). |
| 28 | F6 | MEDIUM | "**Getting started**" | Use plain text. |
| 29 | F6 | MEDIUM | "**02**" | Use plain text. |
| 30 | F6 | MEDIUM | "**Energy injection**" | Use plain text. |
| 31 | F6 | MEDIUM | "**03**" | Use plain text. |
| 32 | F6 | MEDIUM | "**New physics**" | Use plain text. |
| 33 | F6 | MEDIUM | "**04**" | Use plain text. |
| 34 | F6 | MEDIUM | "**Custom scenarios**" | Use plain text. |
| 35 | F6 | MEDIUM | "**05**" | Use plain text. |
| 36 | F6 | MEDIUM | "**Observational constraints**" | Use plain text. (Pattern continues at lines 37–38; total 12 bold-for-emphasis instances in this table.) |
| 28 | F11 | LOW | "**Getting started** --- Green's function basics, first PDE runs, PDE versus GF comparison." | Use an unspaced em dash, or a colon. |
| 30 | F11 | LOW | "**Energy injection** --- Decaying particles, DM annihilation (s-wave, p-wave), amplitude scaling." | Same fix. |
| 32 | F11 | LOW | "**New physics** --- Dark photon oscillation, monochromatic photon injection, :math:`\mu` sign flip." | Same fix. |
| 34 | F11 | LOW | "**Custom scenarios** --- Tabulated heating histories, custom injection closures." | Same fix. |
| 36 | F11 | LOW | "**Observational constraints** --- FIRAS and PIXIE limits, :math:`\mu`--:math:`y` exclusion plane, mock PIXIE observation." | Same fix. |
| 38 | F11 | LOW | "**Green's function tables** --- Precomputed PDE-based tables for fast convolution." | Same fix. |
| 28 | T12 | MEDIUM | "PDE versus GF comparison." | "GF" is never linked or spelled out in this document (only PDE, DM, FIRAS, and PIXIE get `:term:` links at line 18–21); add a `:term:`GF`` link or spell out "Green's function (GF)" at this first bare use. |

## docs/glossary.rst

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 46 | T12 | HIGH | "Far Infrared Absolute Spectrophotometer. The instrument on the COBE" | Spell out "COBE" (Cosmic Background Explorer) or add it as its own glossary entry; the glossary's stated purpose is to define every abbreviation used elsewhere, and this one is left undefined with no contextual clue. |
| 16–17 | T11 | DOMAIN | ":math:`e + \mathrm{ion} \to e + \mathrm{ion} + \gamma`" | Standard reaction-equation notation; no change needed. |
| 32 | T11 | DOMAIN | ":math:`\gamma + e \to \gamma + \gamma + e`" | Standard reaction-equation notation; no change needed. |
