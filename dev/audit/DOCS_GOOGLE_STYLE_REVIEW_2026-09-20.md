# Documentation review against the Google developer documentation style guide

Date: 2026-09-20. Commit reviewed: `2662954` (branch `main`). No repository file was changed.

Reference: [Google developer documentation style guide highlights](https://developers.google.com/style/highlights)
and the pages it links to (word list, headings, lists, code samples, placeholders, link text,
API reference code comments, inclusive documentation).

## Scope and method

Reviewed, line by line:

1. `README.md`, `data/cosmotherm/README.md`
2. `CONTRIBUTING.md`, `CONTRIBUTING_CLAUDE.md`
3. Sphinx top-level pages: `docs/index.rst`, `docs/installation.rst`, `docs/cli.rst`, `docs/rust_api.rst`, `docs/tutorials/index.rst`
4. Sphinx API pages: all nine files in `docs/api/`
5. The six tutorial notebooks in `notebooks/tutorials/` (Markdown cells, and whether each code cell is introduced)
6. Python docstrings in all 12 modules of `python/spectroxide/` (sections 6 and 7), plus user-facing warning and error strings
7. Rust doc comments (`//!`, `///`) in all 19 files of `src/`, `src/bin/check_adiabatic.rs`, the headers of the five `examples/` files, and the CLI help text (sections 8 and 9)

Not reviewed: `CLAUDE.md`, `.claude/`, and `dev/` (internal working files), the `notebooks/physics/`,
`notebooks/observational/`, and `notebooks/paper_figures/` notebooks, and the paper.

Nine reviewers (one per section) each applied the same rubric of rule IDs, which is reproduced in the appendix.
Every finding carries a line number and a quotation. A script then searched for each quotation within five lines of the cited
line: 518 of 591 tabulated quotations matched. The 73 that did not match are of three kinds: structural findings that
describe a layout ("heading followed directly by a code block") instead of quoting text, rows that join quotations from
several distant lines, and three rows that recorded "no violation," which were removed. The three highest-impact claims below
(abbreviations never expanded, the undocumented `cosmo` parameter, the undocumented `BrPrecomputed` fields) and the
`docs/api/axion.rst` page order were checked by hand against the source. The
tutorial-notebook findings cite `cell N, line M` and were not machine-checked.

Severity: HIGH impedes use or comprehension; MEDIUM is a clear rule violation; LOW is polish; DOMAIN means the text breaks
the Google rule but follows a standard scientific or language-community convention, so a fix is a judgment call.

## Overall result

The documentation does not conform. Roughly 900 findings, about 70 of them HIGH. Coverage of parameters, defaults, and units in
the API reference is the strong point; nearly every problem is one of twelve repeated patterns, and most are mechanical to fix.

| Section | HIGH | MEDIUM | LOW and DOMAIN |
|---|---|---|---|
| 1. README files | 6 | 57 | 6 |
| 2. Contributor guides | 5 | about 75 | about 10 |
| 3. Sphinx top-level pages | about 14 | about 55 | about 15 |
| 4. Sphinx API pages | 6 | 74 | 60 |
| 5. Tutorial notebooks | 17 | about 38 | about 20 |
| 6. Python docstrings, group A | 11 | about 45 | about 20 |
| 7. Python docstrings, group B | 3 | about 15 | about 8 |
| 8. Rust doc comments, physics and infrastructure | 9 | about 165 | about 16 |
| 9. Rust doc comments and CLI help, solver and entry points | 3 | about 135 | about 8 |

## Cross-cutting patterns, in order of impact

1. **Abbreviations are never spelled out (rule T12, HIGH).** "Partial differential equation" appears nowhere in the README,
   contributor guides, Sphinx pages, `python/spectroxide/__init__.py`, or `src/lib.rs`. "Cosmic microwave background" appears once,
   inside a citation title (`README.md:291`). PDE, CMB, DC (double Compton), BR (bremsstrahlung), GF (Green's function), DM (dark
   matter), NWA (narrow-width approximation), IC (initial condition), CLI, FIRAS, PIXIE, IMEX, ODE, and DI are used as if known. Google requires
   the expansion at first use on every page, because readers arrive from search, not from page 1. This is the single
   largest barrier for a reader outside the spectral-distortion field.
2. **Code blocks and tables follow headings with no introductory sentence (F2, L2, H4).** About 35 instances: `README.md` quick start (6),
   `CONTRIBUTING_CLAUDE.md` (3), `docs/cli.rst` and `docs/rust_api.rst` (5), `docs/api/solver.rst` (7 of the 11 in `docs/api/`), and tutorial code cells.
3. **Procedures are not numbered lists (L1, L4, HIGH where it affects installation).** The `README.md` manual installation is labeled
   "step-by-step" but uses bold run-in labels; one block bundles build, test, and two alternative `pip` commands. Tutorial 05 packs a
   five-step procedure into one semicolon-chained sentence. Conversely, `CONTRIBUTING.md:66` numbers a non-sequential requirement set.
4. **Placeholders (F3).** `docs/cli.rst` and the CLI help use lowercase angle brackets (`<z>`, `<val>`, `<subcommand>`) mixed with
   `PATH`, and no placeholder is ever explained with a "Replace X with ..." sentence. Angle brackets in `--help` output are a
   command-line convention (DOMAIN); in `docs/cli.rst` they are a plain violation.
5. **Rust function summaries use the imperative (A2, about 180 instances, MEDIUM).** "Compute", "Build", "Solve" instead of "Computes",
   "Builds", "Solves". The Rust standard library and Google both use the third person, so there is no competing convention to cite. The Python
   docstrings do the same (about 60 reviewed), but PEP 257 and the NumPy docstring standard prescribe the imperative, so that is DOMAIN:
   leave Python alone.
6. **Real API documentation gaps (A4, HIGH).** `br_emission_coefficient` omits `cosmo` from `# Arguments` (`src/bremsstrahlung.rs:121`);
   `BrPrecomputed` has three undocumented public fields (`src/bremsstrahlung.rs:221`); `kompaneets_step_coupled_inplace` documents 2 of
   10 parameters; 10 `Result`-returning Rust functions never say what produces `Err` (two table loaders in `src/energy_injection.rs` matter most);
   nine `warn_*` helpers in `_validation.py` have no Parameters section; `load_greens_database` and `cosmotherm_gf_distortion` raise
   without a Raises section; `examples/photon_diag.rs` has no file-level doc. 38 one-line Rust docs lack a final period (A1).
7. **Spelling is mixed American and British (T15).** A repository-wide search finds about 99 American `-ize`/`behavior` forms and about 71 British
   `-ise`/`-our` forms. The section reviewers undercounted this: `python/spectroxide/firas.py` alone has 26, and they include **public API
   names** (`marginalise_y`, `marginalise_mu`, `marginalise_galactic`, `fit_amplitude_marginalised`). Fixing the prose is mechanical; renaming the
   keywords breaks callers, so that needs a deprecation alias or a decision to leave them.
8. **Word-list violations (T9), about 90 instances.** "e.g." (about 25), "via" (about 25), "etc." (about 12), "vs"/"vs." (about 8), "i.e.", "and/or", "(s)"
   plurals in error messages, "above"/"below" for position, "since" for "because", "should" where "must" is meant.
9. **First person and passive voice (T1, T2, T3).** "we"/"our" about 20 times (contributor review process, five Rust modules, one `firas.py`
   docstring); "the user"/"users" 3 times; about 50 agentless passives; "will" for present behavior 6 times.
10. **Headings (H1, H2, H3).** Title case on 5 of 6 tutorial titles, 3 README headings, 2 toctree captions, and `CONTRIBUTING_CLAUDE.md`. Gerund task
    headings ("Adding a new injection scenario") 12 times. Code font inside headings about 20 times, including all six subcommand headings
    in `docs/cli.rst` (arguably the clearest choice there, but it is a violation). `docs/api/axion.rst` places a warning before the page title,
    and `docs/installation.rst` has no introductory sentence (H5).
11. **Emphasis (F6).** ALL CAPS ("NEVER", "WITHOUT", "MUST") and mid-sentence bold are used for emphasis about 35 times. Google reserves bold for
    user-interface elements and notice labels, and uses Note, Caution, and Warning notices for emphasis.
12. **Dashes and symbols (F11, T11).** About 150 em dashes have a space on each side; Google uses none. This is uniform enough to be a deliberate house
    style, so decide once instead of fixing piecemeal. Separately, `README.md` and two notebooks type `---` or `--`, which GitHub and Jupyter
    render literally (Sphinx converts them, so the `.rst` files are fine). Arrows, "~", "+", and "/" stand in for words in prose about 30 times
    outside math; inside reactions and order-of-magnitude statements this is DOMAIN.

One inclusive-language hit (T10): "sanity check" in `src/bin/check_adiabatic.rs:1` (three more in test comments); Google's word list asks for
"quick check" or "consistency check". "native" appears once in `greens_table.py`.

## Where the rubric and this project legitimately disagree

- Imperative docstring summaries in Python (PEP 257, NumPy docstring standard).
- Physics notation in prose: "z ~ 10⁶", "γe → γγe", "γ ↔ A′", "×".
- Angle-bracket placeholders inside `--help` output.
- Code font in a heading whose entire subject is one identifier (`solve`, `run_sweep`).

Treat these as recorded exceptions. Everything else in the report is a plain non-conformance.

## Suggested order of work

1. Add first-use expansions on every page and module docstring (pattern 1). A short glossary page in `docs/` that each page links to covers the long tail.
2. Close the A4 gaps in pattern 6; these are documentation defects under any style guide.
3. Restructure the `README.md` installation and the tutorial 05 procedure as numbered lists; add one introductory sentence before each bare code block.
4. Fix `docs/cli.rst` placeholders and explain them.
5. Run the mechanical substitutions: "e.g." to "for example", "via" to "through" or "with", Rust summaries to third person, final periods, American spelling in prose, sentence-case headings.
6. Decide the two house-style questions (spaced em dashes; British keyword names in `firas.py`) and record the decision in `CONTRIBUTING.md`.

---

## 1. README files

Files reviewed: `README.md` (315 lines), `data/cosmotherm/README.md` (49 lines)

### Summary

The dominant problem is undefined abbreviations: CMB, PDE, DC/BR, and DM are used in `README.md` before any reader outside the field could know what they mean, and DI (the prefix on every data file in `data/cosmotherm/README.md`) is never expanded at all — this is the single highest-impact fix. Second, the install and quick-start sections that the task asked to scrutinize fail core procedure rules: the "Manual installation" block is explicitly labeled "step-by-step instructions" but is not a numbered list, four of the "Quick start" subsections drop a bare code block straight under a heading with no explanatory sentence, and one code block bundles build, test, and two alternative installs with no per-step separation. Third, formatting is inconsistent in mechanical, easy-to-fix ways: `---`/`--` are used as a manual em dash with surrounding spaces in ten places, three headings in the CosmoTherm README use title case instead of sentence case, and package names go from backticked (`numpy`, `scipy`) to plain text (`matplotlib`, `jupyter`, …) one paragraph later in the same table.

| Rule | Count | Severity |
|---|---|---|
| T12 Undefined abbreviations (CMB, PDE, DC/BR, DM, DI, CLI, GF, FIRAS) | 9 | 5 HIGH, 4 MEDIUM |
| F11 Dash/semicolon punctuation (`---`/`--` with spaces, semicolon chain, hyphen-as-dash) | 12 | 11 MEDIUM, 1 LOW |
| F6 Bold/caps used for emphasis, not UI/notice | 5 | MEDIUM |
| F2 Bare code block after heading (2 also missing a language tag) | 6 | MEDIUM |
| F4 Link text is a raw path | 5 | MEDIUM |
| T2 Passive voice, agent omitted | 4 | MEDIUM |
| F1 Package names missing code font | 4 (1 pattern) | MEDIUM |
| H1 Title-case headings | 3 | MEDIUM |
| L2 List/table with no introductory sentence | 3 | MEDIUM |
| T9 Word list (`etc.`, `via`) | 3 | MEDIUM |
| T11 Symbols standing in for words (`/` for "or", `~` for "about") | 3 | MEDIUM |
| T5 "please"/"Quick" as ease/politeness claims | 3 | LOW |
| H4 Heading stacked directly on heading, no text between | 2 | MEDIUM |
| L1 Procedure/parameters not in the required list type | 2 | 1 HIGH, 1 MEDIUM |
| H2 Procedure heading is a noun phrase, not imperative | 2 | LOW |
| T1 "our" instead of second person | 1 | MEDIUM |
| L4 Multiple actions bundled in one unlabeled step | 1 | MEDIUM |
| H5 Intro paragraph does not state the audience | 1 | MEDIUM |

### Systematic patterns

**T12 — undefined abbreviations.** Grep: `grep -noE 'CMB|PDE|DC/BR|DM|CLI|GF|FIRAS|DI'`. Instances:
- `README.md:7` — "Numerical solver for **CMB spectral distortions** from energy and photon injection" — CMB never expanded to "cosmic microwave background" anywhere in the file's prose. Fix: "cosmic microwave background (CMB) spectral distortions".
- `README.md:11` — "...It provides both a **full PDE solver** (Rust) and a **fast Green's function approximation**" — PDE (partial differential equation) never expanded.
- `README.md:15` — "- **Full PDE solver** in Rust: implicit Kompaneets + coupled DC/BR with adaptive stepping" — DC/BR never tied to "double Compton"/"bremsstrahlung" as an abbreviation.
- `README.md:17` — "...DM annihilation (s-wave/p-wave), dark photon oscillation..." — DM (dark matter) never expanded.
- `README.md:57` — "**Rust** (required for the PDE solver and CLI):" — CLI never expanded.
- `README.md:187` — "Green's function basics, first PDE runs, PDE vs GF comparison" — GF used with no antecedent "(GF)" tag.
- `README.md:191` — "FIRAS/PIXIE limits, $\mu$-$y$ plane, mock PIXIE observation" — FIRAS never expanded.
- `data/cosmotherm/README.md:6` — "## DI Files (included in repo)" — DI is the prefix on every data file in the table and is never defined in the file.
- `data/cosmotherm/README.md:41` — "The GF database uses our default cosmology (h=0.71, Omega_b=0.044) matching" — GF used without a defining "(GF)" tag in this file.

**F11 — manual em dash with surrounding spaces, instead of `—` with no surrounding space.** Grep: `grep -n -- ' --- \| -- '`. 10 instances, all listed (≤10):
- `README.md:9` — "distortions --- $\mu$-type" and "(Compton) --- encode" (two on one line)
- `README.md:25` — "everything --- Rust toolchain"
- `README.md:291` — "2604.24838)) --- this paper"
- `README.md:292` — "...19786.x)) ---" (before "CosmoTherm thermalization solver")
- `README.md:293` — "stt1025)) --- Green's function I"
- `README.md:294` — "problem -- II. Effect" (double hyphen variant)
- `README.md:300` — "Contributions --- new injection scenarios" and "bug fixes ---" (two on one line)
- `README.md:310` — "prompt --- it encodes"
Fix: replace with a real em dash and no surrounding space, e.g. "distortions—μ-type".

**F2 — heading immediately followed by a bare code block, no introductory sentence.** 6 instances:
- `README.md:86`–`88` — "### Python: PDE solver" → ` ```python ` with no lead-in sentence.
- `README.md:128`–`130` — "### Python: Green's function (fast approximate, no Rust needed)" → ` ```python ` with no lead-in sentence.
- `README.md:144`–`146` — "### Rust API" → ` ```rust ` with no lead-in sentence.
- `README.md:164`–`166` — "### CLI" → ` ```bash ` with no lead-in sentence.
- `README.md:225`–`227` — "## Architecture" → ` ``` ` (also missing a language tag).
- `data/cosmotherm/README.md:18`–`20` — "## Cosmology (Planck 2015)" → ` ``` ` (also missing a language tag).
Fix: add one sentence ending in a colon before each block, e.g. "The Python API mirrors the Rust one:".

**T12/F1/F4/F6/H1 systematic entries** are covered item-by-item in the file tables below (counts small enough to list every instance there, per the accuracy rule).

### Findings by file

#### `README.md`

| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 7 | T12 | HIGH | `Numerical solver for **CMB spectral distortions**` | Expand: "cosmic microwave background (CMB) spectral distortions" |
| 7 | F6 | MEDIUM | `**CMB spectral distortions**` | Bold used for emphasis in flowing prose, not a UI element; remove bold or use italics for the first-use term |
| 7 | H5 | MEDIUM | `Numerical solver for **CMB spectral distortions** from energy and photon injection in the early Universe.` | State who the page is for, e.g. add "for cosmologists modeling early-universe energy injection" |
| 9 | F11 | MEDIUM | `distortions --- $\mu$-type` | Use `—` (em dash, no surrounding space): "distortions—$\mu$-type" |
| 9 | F11 | MEDIUM | `(Compton) --- encode` | "(Compton)—encode" |
| 11 | T12 | HIGH | `both a **full PDE solver** (Rust)` | Expand: "partial differential equation (PDE) solver" at first use |
| 11 | F6 | MEDIUM | `**full PDE solver**` | Drop bold; not a UI element |
| 11 | F6 | MEDIUM | `**fast Green's function approximation**` | Drop bold; not a UI element |
| 13–15 | L2 | MEDIUM | `## Features` (heading directly followed by the bullet list at line 15) | Add an introductory sentence ending in a colon before the list |
| 15 | T12 | HIGH | `implicit Kompaneets + coupled DC/BR with adaptive stepping` | Expand at first use: "double Compton (DC) and bremsstrahlung (BR)" |
| 17 | T12 | HIGH | `DM annihilation (s-wave/p-wave), dark photon oscillation` | Expand: "dark matter (DM) annihilation" |
| 17 | T11 | MEDIUM | `DM annihilation (s-wave/p-wave)` | "/" stands for "or"; write "s-wave or p-wave" |
| 17 | T9 | MEDIUM | `plus custom heating via Rust API` | Replace "via" with "using" or "through" |
| 17 | F8 | MEDIUM | `**9 built-in injection scenarios**` | Spell out: "Nine built-in injection scenarios" (not a measurement/version) |
| 21–23 | H4 | MEDIUM | `## Installation` immediately followed by `### Quick install (recommended)` | Add a sentence of text under `## Installation` before the first subheading |
| 23 | H2 | LOW | `### Quick install (recommended)` | Prefer an imperative form, e.g. "Install with the script (recommended)" |
| 23 | T5 | LOW | `### Quick install (recommended)` | "Quick" is an ease/speed claim; consider "Install with the script" |
| 25 | F11 | MEDIUM | `everything --- Rust toolchain` | "everything—Rust toolchain" |
| 40–41 | — | — | `Every install pulls in the base dependencies \`numpy\` and \`scipy\`; the` | (context for the F1 table inconsistency below) |
| 45–48 | F1 | MEDIUM | `\| \`plot\`     \| matplotlib                        \| Scripts and plotting (default)    \|` | Package names in the "Adds" column (matplotlib, jupyter, pytest, mutmut, sphinx, nbsphinx, pydata-sphinx-theme) are not in code font, unlike `numpy`/`scipy` two lines above; wrap each in backticks |
| 50 | T9 | MEDIUM | `to see all options (skip steps, verbose output, etc.).` | Replace "etc." with a specific example or "and more" |
| 52–82 | L1 | HIGH | `<summary>Click to expand step-by-step instructions</summary>` (procedure delivered as bold labels + code, not a numbered list) | Convert the Rust / Python / Build-and-install steps into a numbered list |
| 75–80 | L4 | MEDIUM | `**Build and install:**` block runs `cargo build`, `cargo test`, and two alternative `pip install` lines with no per-step separation | Split into separate numbered steps: build, test (optional), then one `pip install` line, marking the second `pip install` "Optional:" |
| 84–86 | H4 | MEDIUM | `## Quick start` immediately followed by `### Python: PDE solver` | Add an introductory sentence under `## Quick start` |
| 86–88 | F2 | MEDIUM | `### Python: PDE solver` → bare ` ```python ` block | Add a lead-in sentence ending in a colon |
| 128–130 | F2 | MEDIUM | `### Python: Green's function (fast approximate, no Rust needed)` → bare ` ```python ` block | Add a lead-in sentence ending in a colon |
| 144–146 | F2 | MEDIUM | `### Rust API` → bare ` ```rust ` block | Add a lead-in sentence ending in a colon |
| 164–166 | F2 | MEDIUM | `### CLI` → bare ` ```bash ` block | Add a lead-in sentence ending in a colon |
| 181 | T2 | MEDIUM | `Output is written to stdout as JSON (pipe to a file with \`> output.json\`).` | "The CLI writes output to stdout as JSON." |
| 183–187 | L2 | MEDIUM | `## Example notebooks` (heading directly followed by the table at line 185) | Add a one-sentence intro before the table |
| 187 | T12 | MEDIUM | `Green's function basics, first PDE runs, PDE vs GF comparison` | Tie "GF" to "Green's function" on first table use, e.g. define GF once in the section's intro sentence |
| 191 | T12 | MEDIUM | `FIRAS/PIXIE limits, $\mu$-$y$ plane, mock PIXIE observation` | Expand FIRAS at first use |
| 194 | F4 | MEDIUM | `` [`notebooks/physics/`](notebooks/physics/) `` | Link text is a raw path; use descriptive text, e.g. "physics notebooks" |
| 194 | F4 | MEDIUM | `` [`notebooks/observational/`](notebooks/observational/) `` | Use "observational notebooks" as link text |
| 194 | F4 | MEDIUM | `` [`dev/notebooks/`](dev/notebooks/) `` | Use "development notebooks" as link text |
| 200 | T2 | MEDIUM | `Arbitrary sources are passed as callables instead of an \`injection\` dict.` | "Pass arbitrary sources as callables instead of an `injection` dict." |
| 215 | F6 | MEDIUM | `**per baryon**` | Bold used for emphasis mid-sentence; drop bold or use italics |
| 215–217 | F11 | MEDIUM | `decay rate); $f_{\rm ann}$ [eV/s] is energy per baryon per second;` (two semicolons chaining three clauses) | Split into two or three sentences |
| 221 | T11 | MEDIUM | `` as `result.mu` / `result.y`). `` | "/" stands for "or"; write "as `result.mu` or `result.y`" |
| 225–227 | F2 | MEDIUM | `## Architecture` → bare ` ``` ` block, no language tag | Add a lead-in sentence and tag the fence, e.g. ` ```text ` |
| 287 | T5 | LOW | `please cite the accompanying paper` | Drop "please": "cite the accompanying paper" |
| 288 | T5 | LOW | `also cite Chluba (2013) and Chluba (2015)` (continuation of "...mode, please") | Drop "please" |
| 291–294 | F11 | MEDIUM | `2604.24838)) --- this paper` / `...19786.x)) ---` / `stt1025)) --- Green's function I` / `problem -- II. Effect` | Replace each with a real em dash, no surrounding space |
| 296 | T2 | MEDIUM | `A machine-readable \`CITATION.cff\` is also included in the repository root.` | "The repository root also includes a machine-readable `CITATION.cff` file." |
| 300 | F11 | MEDIUM | `Contributions --- new injection scenarios` / `bug fixes ---` | Replace with em dash, no surrounding space |
| 302 | T2 | MEDIUM | `codebase) are written with LLM assistance, and the workflow is built around that:` | "...an LLM writes most contributions...and the workflow builds around that:" |
| 308 | F4 | MEDIUM | `[CONTRIBUTING.md](CONTRIBUTING.md)` | Link text is a raw filename; consider "the contributing guide" |
| 309 | F4 | MEDIUM | `[CONTRIBUTING_CLAUDE.md](CONTRIBUTING_CLAUDE.md)` | Consider "the Claude-specific contributing notes" |
| 310 | F11 | MEDIUM | `prompt --- it encodes` | "prompt—it encodes" |

#### `data/cosmotherm/README.md`

| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 1 | H1 | MEDIUM | `# CosmoTherm Reference Data` | Sentence case: "CosmoTherm reference data" |
| 6 | H1 | MEDIUM | `## DI Files (included in repo)` | Sentence case: "DI files (included in repo)" |
| 6 | T12 | HIGH | `## DI Files (included in repo)` | Define "DI" (e.g., "distortion intensity") at first use; it prefixes every file in the table below and is never expanded |
| 18–20 | F2 | MEDIUM | `## Cosmology (Planck 2015)` → bare ` ``` ` block, no language tag | Add a lead-in sentence, e.g. "CosmoTherm's reference run uses:", and tag the fence |
| 18–28 | L1 | MEDIUM | Six term = value cosmological parameters (`Y_p`, `T_CMB`, `Omega_cdm`, …) presented as a bare code block rather than a table or description list | Convert to a description list or two-column table of parameter : value |
| 30 | H1 | MEDIUM | `## Green's Function Database (NOT included - too large)` | Sentence case: "Green's function database (not included — too large)" |
| 30 | F6 | MEDIUM | `(NOT included - too large)` | All caps used for emphasis; use plain text |
| 30 | F11 | LOW | `(NOT included - too large)` | Single hyphen with spaces used as a dash; use an em dash: "(not included—too large)" |
| 32 | T11 | MEDIUM | `` `Greens_data.dat` (~12 MB) is available `` | "~" stands for "approximately" outside math; write "about 12 MB" |
| 41 | T1 | MEDIUM | `The GF database uses our default cosmology (h=0.71, Omega_b=0.044) matching` | Second person / no "our": "spectroxide's default cosmology" |
| 41 | T12 | MEDIUM | `The GF database uses our default cosmology` | "GF" used with no defining tag in this file |
| 44–46 | L2 | MEDIUM | `## References` (heading directly followed by the bullet list at line 46) | Add a one-sentence intro, e.g. "This data draws on:" |

### What conforms

- Serial (Oxford) comma used consistently in lists (`README.md:11,17`).
- Condition-before-instruction order followed correctly (`README.md:59`, `219`, `287`).
- No inclusive-language violations, gendered pronouns, "click here"/"see above" link text, or curly-quote inconsistency found in either file.
- No future tense ("will") used for product behavior.
- Headings in `README.md` are in sentence case throughout (unlike the CosmoTherm README).
- No heading level is skipped, and each file has exactly one H1.
- Table header rows are in sentence case with no unexplained empty cells (`README.md:43-48`, `185-192`, `202-212`; `data/cosmotherm/README.md:10-16`).
- `data/cosmotherm/README.md`'s opening paragraph (lines 3–4) states what the page covers.
- Author-list ampersands in citation entries and math-mode symbols ($\times$, $\sim$, $\Delta$, etc.) were treated as domain convention and not flagged, per the rubric's DOMAIN carve-out.


---

## 2. Contributor guides

Files reviewed: `CONTRIBUTING.md` (140 lines), `CONTRIBUTING_CLAUDE.md` (190 lines)

### Summary

Both files are otherwise well-organized but share four dominant, highly systematic problems. First, central abbreviations (PDE, CMB, DC, BR, GF, CLI, IMEX) are used from the first sentence onward and never spelled out (T12), which is most damaging in `CONTRIBUTING_CLAUDE.md`, whose opening sentence defines the whole project using two undefined acronyms. Second, `CONTRIBUTING_CLAUDE.md` opens three code blocks directly under a heading with no introductory sentence (F2), and its title is in title case rather than sentence case (H1). Third, both files use gerund section headings ("Setting up...", "Submitting...", "Modifying...") where Google style calls for the bare imperative (H2). Fourth, `CONTRIBUTING.md` numbers a non-sequential set of PR requirements as a 1-4 list instead of a bullet list (L1), and combines two actions into single procedure steps (L4). Minor, high-volume issues include spaced em dashes throughout (F11, LOW) and heavy use of bold for run-in list headers and mid-sentence emphasis (F6).

| Rule | Count | Severity |
|---|---|---|
| T12 Undefined abbreviations (PDE, CMB, DC, BR, GF, CLI, IMEX) | 7 abbreviations, 20+ uses | HIGH/MEDIUM |
| F2 Bare code block directly after a heading | 3 | HIGH |
| L1 Numbered list used for a non-sequential requirement set | 1 (4 items) | HIGH |
| T1 "we" instead of "you" | 4 | MEDIUM |
| T2 Passive voice, agent unstated or de-emphasized | 4 | MEDIUM |
| T3 Future tense "will" for product/process behavior | 4 | MEDIUM |
| T8 Instruction stated before its condition | 3 | MEDIUM |
| T9 Word list ("e.g.", "etc.", "vs", "via") | 8 | MEDIUM |
| H1 Title-case document title | 1 | MEDIUM |
| H2 Gerund section headings instead of imperative | 7 | MEDIUM |
| H3 Code font inside headings | 2 | MEDIUM |
| F2 Fenced code block with no language tag | 1 | MEDIUM |
| F6 ALL CAPS used for emphasis in headings | 5 | MEDIUM |
| F6 Bold used for mid-sentence/table-cell emphasis | 3 | MEDIUM |
| L4 Two actions combined into one procedure step | 2 | MEDIUM |
| T11 ASCII arrows / slash-for-or in prose | 3 | MEDIUM/DOMAIN |
| F11 Em dash with surrounding spaces | 11 (`CONTRIBUTING.md`) + 14 (`CONTRIBUTING_CLAUDE.md`) | LOW |
| F6 Bold overused as list run-in headers | 32 + 29 lines | LOW (pattern note) |
| T13 Latin phrase "ad hoc" | 1 | LOW |
| L4 Missing "Optional:" prefix on a conditional step | 1 | LOW |
| F12 Tool name "clippy" not capitalized/code-font | 1 | LOW |
| F2 Code-sample line over 80 characters | 1 | LOW |
| F1 Inconsistent code-font treatment of the same formula | 1 | LOW |

### Systematic patterns

**T12 — abbreviations used without being spelled out.** Neither file defines PDE, CMB, DC, BR, GF, CLI, or IMEX at first use, even though `CONTRIBUTING_CLAUDE.md`'s first sentence is "spectroxide is a Rust PDE solver for CMB spectral distortions." Grep: `grep -n -E "\bPDE\b|\bCMB\b|\bDC\b|\bBR\b|\bGF\b|\bCLI\b|\bIMEX\b" CONTRIBUTING.md CONTRIBUTING_CLAUDE.md`.
- `CONTRIBUTING_CLAUDE.md:13` — "spectroxide is a Rust PDE solver for CMB spectral distortions." — spell out on first use: "partial differential equation (PDE)" and "cosmic microwave background (CMB)".
- `CONTRIBUTING_CLAUDE.md:35` — "IMEX integrator coupling all processes." — expand "implicit-explicit (IMEX)".
- `CONTRIBUTING_CLAUDE.md:63` — "string identifier for CLI/output" — expand "command-line interface (CLI)" at first use.
- `CONTRIBUTING_CLAUDE.md:127` — "**DC/BR divergence at low x**: Emission rate ~ 1/x^3." — expand "double Compton (DC)" and "bremsstrahlung (BR)".
- `CONTRIBUTING_CLAUDE.md:115` — "For simple injection histories, the PDE and GF should agree within ~5%." — GF ("Green's function") is spelled out elsewhere but the abbreviation itself is never tied to its expansion.
- `CONTRIBUTING.md:38` — "3. Wire into the CLI in `src/main.rs`" — CLI undefined in this file.
- `CONTRIBUTING.md:49` — "backward Euler for DC/BR instead of Crank-Nicolson" — DC/BR undefined in this file.
- `CONTRIBUTING.md:78` — "**Cross-validation of PDE vs Green's function**" — PDE undefined in this file.

**F2 — bare code block immediately after a heading, no introductory sentence.** Grep: `awk '/^#/{h=NR} /^```/{print FILENAME":"NR" (heading at "h")"}' CONTRIBUTING_CLAUDE.md`.
- `CONTRIBUTING_CLAUDE.md:17-19` — heading "## Build and test" is followed by a blank line then directly by the fenced block — add a sentence such as "Build and test the Rust crate with the following commands:".
- `CONTRIBUTING_CLAUDE.md:47-49` — heading "### Step 1: Add the variant..." is followed directly by a ```rust block with no lead-in sentence.
- `CONTRIBUTING_CLAUDE.md:88-90` — heading "### 1. NEVER calibrate test targets from the code itself" is followed directly by a ```rust block.

**H2 — gerund section headings where Google style expects the bare imperative for task-oriented sections.** Grep: `grep -n "^#" CONTRIBUTING.md CONTRIBUTING_CLAUDE.md`.
- `CONTRIBUTING.md:1` — "# Contributing to spectroxide" — "Contribute to spectroxide".
- `CONTRIBUTING.md:22` — "### Setting up your LLM" — "Set up your LLM".
- `CONTRIBUTING.md:32` — "### Adding a new energy injection scenario (the most common contribution)" — "Add a new energy injection scenario".
- `CONTRIBUTING.md:45` — "### Modifying solver physics" — "Modify solver physics".
- `CONTRIBUTING.md:53` — "## Submitting a pull request" — "Submit a pull request".
- `CONTRIBUTING_CLAUDE.md:43` — "## How to add a new energy injection scenario" — drop "How to": "Add a new energy injection scenario".
- `CONTRIBUTING_CLAUDE.md:165` — "## Example: adding a simple scenario" — "Example: add a simple scenario".

**F11 — em dash written with surrounding spaces (" — "), where Google style uses no surrounding spaces.** Count: 11 in `CONTRIBUTING.md`, 14 in `CONTRIBUTING_CLAUDE.md`. Grep: `grep -n "—" CONTRIBUTING.md` / `grep -n "—" CONTRIBUTING_CLAUDE.md`. First 10 in `CONTRIBUTING.md`:
- `CONTRIBUTING.md:3` — "spectroxide welcomes contributions — new energy injection scenarios" — "contributions—new energy".
- `CONTRIBUTING.md:7` — "verified the code against *itself* — asserting whatever" — "*itself*—asserting".
- `CONTRIBUTING.md:43` — "you must be actively involved — the test targets" — "involved—the test targets".
- `CONTRIBUTING.md:113` — "**Physical correctness** — Are the equations right?" — "correctness—Are".
- `CONTRIBUTING.md:114` — "**Numerical soundness** — Does the implementation respect" — "soundness—Does".
- `CONTRIBUTING.md:115` — "**Code quality** — Does it follow existing patterns?" — "quality—Does".
- `CONTRIBUTING.md:128` — "fix the issue — do not ask for the check to be skipped." — "issue—do not".
- `CONTRIBUTING.md:132` — "Use a short `type: subject` style — typical types are" — "style—typical".
- `CONTRIBUTING.md:137` — "`notebooks/tutorials/` — start with `01_getting_started.ipynb`" — "tutorials/`—start".
- `CONTRIBUTING.md:139` — "**Chluba & Sunyaev (2012)**, MNRAS 419, 1294 — primary reference" — "1294—primary".
(1 more instance at line 140.)

### Findings by file

#### `CONTRIBUTING.md`

| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 1 | H2 | MEDIUM | "# Contributing to spectroxide" | "Contribute to spectroxide" |
| 15 | F6 | MEDIUM | "\| `CONTRIBUTING.md` (this file) \| **You**, the human contributor \|" | Drop bold; bold is not for prose emphasis outside UI elements. |
| 16 | F6 | MEDIUM | "\| `CONTRIBUTING_CLAUDE.md` \| **Your LLM** \|" | Drop bold. |
| 18 | T1 | MEDIUM | "we do things a certain way" | "why the project works a certain way" |
| 22 | H2 | MEDIUM | "### Setting up your LLM" | "Set up your LLM" |
| 24 | T9 | MEDIUM | "For other tools (ChatGPT, Copilot, Cursor, etc.), paste the contents" | "For other tools, such as ChatGPT, Copilot, or Cursor, paste the contents" |
| 24 | T9 | MEDIUM | "this happens automatically via `CLAUDE.md`" | "this happens automatically through `CLAUDE.md`" |
| 32 | H2 | MEDIUM | "### Adding a new energy injection scenario (the most common contribution)" | "Add a new energy injection scenario" |
| 38 | T12 | MEDIUM | "3. Wire into the CLI in `src/main.rs`" | Spell out "command-line interface (CLI)" at first use in this file. |
| 45 | H2 | MEDIUM | "### Modifying solver physics" | "Modify solver physics" |
| 49 | T9 | MEDIUM | "These modules encode subtle numerical choices (e.g., backward Euler" | "such as backward Euler" |
| 49 | T12 | MEDIUM | "backward Euler for DC/BR instead of Crank-Nicolson" | Spell out "double Compton (DC)" and "bremsstrahlung (BR)" at first use. |
| 53 | H2 | MEDIUM | "## Submitting a pull request" | "Submit a pull request" |
| 59 | L4 | MEDIUM | "1. **Fork the repository** and create a feature branch (e.g., `add-pbh-evaporation`)." | Split into two steps: fork, then create a branch. |
| 59 | T9 | MEDIUM | "create a feature branch (e.g., `add-pbh-evaporation`)" | "create a feature branch, for example `add-pbh-evaporation`" |
| 61 | L4 | MEDIUM | "3. **Run formatting and linting**: `cargo fmt` and `cargo clippy --all-targets -- -D warnings`." | Split `cargo fmt` and `cargo clippy` into two steps, or make the combined action explicit ("Run `cargo fmt` and `cargo clippy` together"). |
| 61 | T3 | MEDIUM | "CI will reject unformatted code." | "CI rejects unformatted code." |
| 62 | L4 | LOW | "4. **If you modified Python code**: from `python/`, run `black spectroxide/` and `pytest tests/`." | Prefix with "Optional:" since this step only applies conditionally. |
| 66-74 | L1 | HIGH | "Every PR that adds or modifies physics code must include:" (followed by items 1-4) | Convert the numbered list to a bulleted list — the four requirements have no sequential order. |
| 68 | T3 | MEDIUM | "will be asked to add one during review" | "is asked to add one during review" |
| 68 | T9 | MEDIUM | "the expected value comes from (e.g., \"Eq. 15 of Chluba 2015\"" | "for example \"Eq. 15 of Chluba 2015\"" |
| 68 | T9 | MEDIUM | "Each test comment should state where the expected value comes from" | "Each test comment must state where the expected value comes from" (this is a hard requirement per line 117). |
| 78 | T9 | MEDIUM | "**Cross-validation of PDE vs Green's function**" | "Cross-validation of the PDE against the Green's function" |
| 78 | T12 | HIGH | "**Cross-validation of PDE vs Green's function**... where the GF is applicable" | Spell out "partial differential equation (PDE)" and tie "GF" to "Green's function" at first use. |
| 78 | T2 | MEDIUM | "Agreement within ~5% is expected." | "We expect agreement within 5%." or state who expects it. |
| 111 | T2 | MEDIUM | "PRs are reviewed for:" | "Reviewers check PRs for:" |
| 113 | T2 | LOW | "1. **Physical correctness** — Are the equations right?" | Minor; em-dash spacing separately flagged. |
| 117 | T1 | MEDIUM | "We will not merge code where test targets cannot be traced" | "This project does not merge code where test targets cannot be traced" |
| 117 | T1 | MEDIUM | "the one rule we will not bend on" | "the one rule that is never relaxed" |
| 117 | T3 | MEDIUM | "We will not merge code where test targets cannot be traced to an independent source." | "This project does not merge code..." |
| 117 | T3 | MEDIUM | "This is the one rule we will not bend on" | "This is the one rule that is never relaxed" |
| 123 | F12 | LOW | "unit tests, science suite, convergence tests, doc tests, clippy, format check" | "Clippy" (tool's proper name) or `` `cargo clippy` `` in code font. |
| 138 | T1 | MEDIUM | "the reference implementation we validate against" | "the reference implementation used for validation" |

#### `CONTRIBUTING_CLAUDE.md`

| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 1 | H1 | MEDIUM | "# LLM Context File for spectroxide Contributors" | "LLM context file for spectroxide contributors" (sentence case). |
| 3 | T2 | MEDIUM | "This file is designed to be included as context" | "Include this file as context" (active, addresses "you"). |
| 3 | T9 | MEDIUM | "when using an LLM (Claude, GPT, etc.) to develop features" | "such as Claude or GPT" |
| 13 | T12 | HIGH | "spectroxide is a Rust PDE solver for CMB spectral distortions." | Spell out "partial differential equation (PDE)" and "cosmic microwave background (CMB)" at first use. |
| 17-19 | F2 | HIGH | "## Build and test" \[blank line\] "```bash" | Add an introductory sentence ending in a colon before the fenced block. |
| 21 | F2 | LOW | "cargo test --release                      # Always use --release; some tests are slow in debug" (94 chars) | Shorten the trailing comment or wrap it. |
| 35 | T12 | MEDIUM | "**Solver** (`src/solver.rs`): IMEX integrator coupling all processes." | Spell out "implicit-explicit (IMEX)" at first use. |
| 43 | H2 | MEDIUM | "## How to add a new energy injection scenario" | "Add a new energy injection scenario" |
| 47 | H3 | MEDIUM | "### Step 1: Add the variant to `InjectionScenario` in `src/energy_injection.rs`" | Move the code-font identifiers out of the heading text into the body. |
| 47-49 | F2 | HIGH | "### Step 1: Add the variant..." \[blank line\] "```rust" | Add a sentence introducing the snippet, e.g. "Add the variant like this:". |
| 63 | T12 | MEDIUM | "`name(&self)` — string identifier for CLI/output" | Spell out "command-line interface (CLI)" at first use. |
| 65 | T8 | MEDIUM | "Return 0.0 if your scenario injects photons rather than heat." | "If your scenario injects photons rather than heat, return 0.0." |
| 68 | T8 | MEDIUM | "Return empty vec if not needed." | "If not needed, return an empty vector." |
| 69 | T8 | MEDIUM | "Return 0.0 unless your scenario removes photons." | "Unless your scenario removes photons, return 0.0." |
| 72 | H3 | MEDIUM | "### Step 3: Wire into CLI (`src/cli.rs`)" | Move the file-path code font out of the heading. |
| 74 | T11 | MEDIUM | "New scenarios usually plug into `solve` and `sweep` via the existing `--scenario` / `--params` machinery" | "the existing `--scenario` or `--params` machinery" |
| 74 | T9 | MEDIUM | "New scenarios usually plug into `solve` and `sweep` via the existing" | "through the existing" |
| 76 | F6 | MEDIUM | "### Step 4: Write tests with INDEPENDENT targets" | "Step 4: Write tests with independent targets" (use italics if emphasis is needed). |
| 84-86 | F2/L1 | LOW | "## Critical rules for writing tests" followed by "### 1. NEVER calibrate..." | These are five non-sequential rules formatted as numbered headings; consider a bulleted list or unnumbered subheadings. |
| 88 | F6 | MEDIUM | "### 1. NEVER calibrate test targets from the code itself" | "Never calibrate test targets from the code itself" |
| 88-90 | F2 | HIGH | "### 1. NEVER calibrate test targets from the code itself" \[blank line\] "```rust" | Add an introductory sentence before the code block. |
| 100 | F6 | MEDIUM | "### 2. Always check dimensions FIRST" | "Always check dimensions first" |
| 113 | T12 | MEDIUM | "### 4. Cross-validate PDE against Green's function" | Rely on the PDE definition given at first use (line 13) once it is added; currently still undefined at this point. |
| 121 | F6 | MEDIUM | "## Numerical pitfalls you MUST know about" | "Numerical pitfalls you must know about" |
| 123 | F6 | MEDIUM | "`CLAUDE.md` enumerates **10 critical numerical pitfalls** in full" | Drop bold; not a UI element or run-in label. |
| 133 | T11 | MEDIUM/DOMAIN | "Injected photon energy absorbed by DC/BR must flow through T_e -> Kompaneets -> mu/y." | "flows from T_e to Kompaneets to mu/y" (informal workflow arrow, not a physics reaction notation). |
| 147 | F6 | MEDIUM | "## What NOT to do" | "## What not to do" |
| 150 | T11 | MEDIUM/DOMAIN | "Photon injection must go through `photon_source_rate()` -> DC/BR absorption -> T_e -> Kompaneets." | Replace "->" with "to" in prose. |
| 151 | T13 | LOW | "**Do not use ad hoc numerical fixes.**" | "improvised numerical fixes" |

### What conforms

- Both files begin with an introductory sentence/paragraph stating purpose and audience (H5).
- All Markdown headings already use sentence case except the `CONTRIBUTING_CLAUDE.md` title (H1) — no other capitalization violations found.
- No heading levels are skipped and each file has exactly one H1 (H4).
- No exclamation marks, marketing language, jokes, or idioms in prose (T7).
- No inclusive-language violations found (T10): no "sanity check," "dummy," "blacklist/whitelist," "master/slave," "kill," "hang," "blind," "crazy," "native," "first-class," "man-hours," or gendered pronouns.
- No bare URLs or "click here"/"this link" style link text (F4) — neither file contains hyperlinks at all.
- No "see above"/"see below" vague cross-references (F5).
- No `<placeholder>`, `your-thing`, `foo/bar`, or `path/to/...` style unexplained placeholders (F3); the PR-template's bracketed prose placeholders (e.g. "[1-3 sentences]") are self-explanatory in context.
- The single table in `CONTRIBUTING.md` (lines 13-16) is genuinely two-dimensional, has a sentence-case header row, and no empty cells (L5).
- Serial commas are used consistently in enumerations (T15), e.g. `CONTRIBUTING.md:3` and `CONTRIBUTING_CLAUDE.md:41`.
- `CONTRIBUTING_CLAUDE.md` consistently addresses the reader as "you" and never uses "we" (T1) — a clean contrast with `CONTRIBUTING.md`.
- Numeral usage for counts of four and above is correct (e.g. "four serious bugs," "430+ Rust tests," "10 critical numerical pitfalls") (F8).
- Fenced Rust/bash code blocks in `CONTRIBUTING_CLAUDE.md` all specify a language tag except none — every block has one; the one bare-block-after-heading issue is about missing prose lead-in, not a missing language tag.


---

## 3. Sphinx site: top-level pages

Files reviewed: `docs/index.rst` (54 lines), `docs/installation.rst` (86 lines), `docs/cli.rst` (209 lines), `docs/rust_api.rst` (60 lines), `docs/tutorials/index.rst` (33 lines). `docs/conf.py` glanced at for title/description strings only (no findings; `project = "spectroxide"`, no rendered description string present).

### Summary

The dominant problem is undefined abbreviations: PDE, CMB, CLI, DC, BR, and DM are used throughout all five pages without ever being spelled out, and this is the project's own central vocabulary. The second most common issue is structural: three of five pages open a section heading directly onto a code block or table with no introductory sentence (F2/L2), and `cli.rst` stacks an H2 directly onto an H3 with no body text. Formatting is mostly clean — code font is applied correctly to flags and identifiers almost everywhere — but placeholders in `cli.rst` mix an unexplained `<angle-bracket>` style with a correct `PATH`-style convention, and two H2 headings use gerund/adjective openers instead of imperatives. Minor issues: bold used for emphasis rather than UI/notice labels, a real em dash with surrounding spaces, an un-spaced em-dash-rendering `---` repeated six times in a table, and one table with seven unexplained empty cells.

| Rule | Count | Severity |
|---|---|---|
| T12 Undefined abbreviations (PDE/CMB/CLI/DC/BR/DM/FIRAS/PIXIE) | 11 | HIGH/MEDIUM |
| F2 Bare code block or table after heading | 5 | MEDIUM |
| F3 Placeholder formatting/explanation | 13 | MEDIUM |
| L5 Empty table cells without explanation | 7 | MEDIUM |
| H3 Code font in headings | 6 | MEDIUM |
| T2 Passive voice, no agent stated | 5 | MEDIUM |
| F11 Em dash spacing | 7 | LOW/MEDIUM |
| F6 Bold used for emphasis | 3 | LOW |
| H2 Gerund/non-imperative task heading | 3 | MEDIUM |
| H4 Stacked headings with no text between | 1 | MEDIUM |
| T9 Word list (`etc.`, `via`) | 2 | MEDIUM/LOW |
| T11 Symbol `/` standing in for a word in prose | 1 | LOW |
| T14 Sentence fragment (dropped subject) | 1 | LOW |
| T5 "quick" as an ease/speed claim | 1 | LOW |
| H1 Title-case toctree caption | 2 | MEDIUM |
| F1 Package names not in code font | 4 | LOW |
| H5 Missing page-intro paragraph | 1 | HIGH |

### Systematic patterns

**T12 — undefined abbreviations.** PDE, CMB, CLI, DC, BR, and DM are used across all five files and never spelled out anywhere in the reviewed set. Grep: `grep -n -E "\bCLI\b|\bPDE\b|\bCMB\b|\bDC\b|\bBR\b|\bDM\b|\bFIRAS\b" docs/*.rst docs/tutorials/index.rst`. Instances (11, all ≤10 so all listed):
- `docs/index.rst:4` — "A PDE solver for CMB :math:`\mu`- and :math:`y`-type spectral" — PDE and CMB both first use, undefined, central to the page. HIGH.
- `docs/installation.rst:49` — "**Rust** (required for the PDE solver and CLI):" — PDE and CLI first use, undefined. MEDIUM (single incidental mention).
- `docs/cli.rst:4` — "The ``spectroxide`` binary provides a command-line interface to the PDE solver." — PDE undefined; central to page. HIGH.
- `docs/cli.rst:1` — "CLI reference" — CLI used in the title before "command-line interface" is spelled out in body text on line 4 (wrong order). LOW.
- `docs/cli.rst:137` — "Operator-split DC/BR instead of coupled Newton iteration." — DC/BR undefined. MEDIUM.
- `docs/rust_api.rst:4` — "The Rust crate ``spectroxide`` is the PDE solver itself; the Python package is a" — PDE undefined, central. HIGH.
- `docs/rust_api.rst:5` — "thin wrapper that calls the Rust CLI for heavy computations and provides a" — CLI undefined. MEDIUM.
- `docs/tutorials/index.rst:23` — "- **Getting started** --- Green's function basics, first PDE runs, PDE vs GF comparison." — PDE undefined. MEDIUM.
- `docs/tutorials/index.rst:25` — "- **Energy injection** --- Decaying particles, DM annihilation (s-wave, p-wave), amplitude scaling." — DM undefined. MEDIUM.
- `docs/tutorials/index.rst:31` — "- **Observational constraints** --- FIRAS/PIXIE limits, :math:\`\mu\`--:math:\`y\` exclusion plane, mock PIXIE observation." — FIRAS and PIXIE undefined. MEDIUM/LOW (PIXIE not in the rubric's example list; lower confidence).

Suggested rewrite (first use in each doc): "the PDE (partial differential equation) solver", "CMB (cosmic microwave background)", "CLI (command-line interface)", "DC (double Compton) and BR (bremsstrahlung)", "DM (dark matter)".

**F2 — bare code block/table immediately after a heading, no introductory sentence.** Grep: manual inspection of every heading→next-block transition. 5 instances:
- `docs/index.rst:18-21` — heading "Quick example" (line 18) directly followed by `.. code-block:: python` (line 21).
- `docs/rust_api.rst:35-38` — heading "Quick Rust example" (line 35) directly followed by `.. code-block:: rust` (line 38).
- `docs/cli.rst:151-154` — heading "Cosmology options" (line 151) directly followed by `.. code-block:: bash` (line 154).
- `docs/cli.rst:186-189` — heading "Examples" (line 186) directly followed by `.. code-block:: bash` (line 189).
- `docs/cli.rst:171-174` — heading "Output options" (line 171) directly followed by `.. list-table::` (line 174), no intro sentence (L2, same pattern applied to a table rather than a code block).

Fix: add one sentence before each block, for example "The following snippet imports the prelude and runs a single-burst injection:" before the rust_api.rst example.

**F3 — placeholder formatting.** cli.rst mixes two placeholder styles. Grep: `grep -n '<[a-zA-Z_-]*>' docs/cli.rst` returns 11 hits; `grep -n 'PATH' docs/cli.rst` returns 2. Total 13 instances, first 10 of the angle-bracket set:
- `docs/cli.rst:9` — "cargo run --release --bin spectroxide -- <subcommand> [options]" — lower-case angle-bracket placeholder, never explained.
- `docs/cli.rst:22` — "spectroxide solve <injection-type> [options]" — same issue.
- `docs/cli.rst:111` — "* - \`\`--z-start <z>\`\`" — placeholder `<z>` in code font but wrong case/style and not explained beyond the "Description" column.
- `docs/cli.rst:114` — "* - \`\`--z-end <z>\`\`"
- `docs/cli.rst:117` — "* - \`\`--n-points <n>\`\`"
- `docs/cli.rst:123` — "* - \`\`--dy-max <val>\`\`"
- `docs/cli.rst:126` — "* - \`\`--dtau-max <val>\`\`"
- `docs/cli.rst:129` — "* - \`\`--dtau-max-photon-source <val>\`\`"
- `docs/cli.rst:141` — "* - \`\`--nc-z-min <z>\`\`"
- `docs/cli.rst:147` — "* - \`\`--threads <n>\`\`" (11th instance, `docs/cli.rst:182 * - \`\`--output <path>\`\``, omitted per the ≤10 cap)

Separately, `docs/cli.rst:47` — "- \`\`--heating-table PATH\`\` (CSV: \`\`z,dq_dz\`\`)" and `docs/cli.rst:49` — "- \`\`--photon-table PATH\`\` (CSV: \`\`z,x1,...,xN\`\`)" use the Google-preferred UPPER_SNAKE_CASE style for `PATH` but never explain it with a "Replace PATH with..." sentence. Fix: standardize on `Z`, `N`, `VAL` (upper snake case) throughout the table and add one sentence after the table explaining the placeholder convention, e.g. "Replace `<value>`-style placeholders with a redshift, integer, or float as shown in the flag name."

**L5 — empty table cells with no explanation.** Grep: `grep -n '^\s*-\s*$' docs/cli.rst`. 7 instances, all listed (the "Default" column is left blank for boolean flags with no stated meaning for the blank):
- `docs/cli.rst:121` (row for `--production-grid`)
- `docs/cli.rst:124` (row for `--dy-max <val>`)
- `docs/cli.rst:133` (row for `--no-dcbr`)
- `docs/cli.rst:136` (row for `--split-dcbr`)
- `docs/cli.rst:139` (row for `--no-number-conserving`)
- `docs/cli.rst:145` (row for `--no-auto-refine`)
- `docs/cli.rst:148` (row for `--threads <n>`)

Fix: fill each with "off" / "unset" / "(none)" plus a one-line note under the table explaining what a blank Default means.

**H3 — code font in headings.** Grep: `grep -n '^\`\`' docs/cli.rst`. 6 instances (the subcommand headings under "Subcommands"), all listed:
- `docs/cli.rst:15` — "\`\`solve\`\`"
- `docs/cli.rst:51` — "\`\`sweep\`\`"
- `docs/cli.rst:60` — "\`\`photon-sweep\`\`"
- `docs/cli.rst:70` — "\`\`photon-sweep-batch\`\`"
- `docs/cli.rst:79` — "\`\`greens\`\`"
- `docs/cli.rst:88` — "\`\`info\`\`"

Fix: drop the backticks in the heading text itself (keep code font for the command in body prose), e.g. "Solve".

### Findings by file

#### `docs/index.rst`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 4 | T12 | HIGH | "A PDE solver for CMB :math:\`\mu\`- and :math:\`y\`-type spectral" | Spell out "PDE (partial differential equation)" and "CMB (cosmic microwave background)" at first use. |
| 10 | T2 | MEDIUM | "and a user-specified injection source. The core solver is implemented" | "Rust implements the core solver and Python integrates it" (state the agent). |
| 12 | T2 | MEDIUM | "approximation (Chluba 2013) is also provided for fast estimates." | "spectroxide also provides an analytic Green's-function approximation ... for fast estimates." |
| 36 | H1 | MEDIUM | ":caption: Getting Started" | Sentence case: "Getting started". |
| 43 | H1 | MEDIUM | ":caption: User Guide" | Sentence case: "User guide". |
| 18 / 21 | F2 | MEDIUM | "Quick example" / ".. code-block:: python" | Add a sentence before the code block, e.g. "The following example runs a single-burst injection and prints μ and y:". |

#### `docs/installation.rst`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| — | H5 | HIGH | (no quote — page has no text) | Add one introductory sentence after the "Installation" title stating what the page covers and for whom, before the first "Quick install" heading. |
| 4 | H2 | MEDIUM | "Quick install (recommended)" | Use a bare imperative: "Install quickly (recommended)" or split the parenthetical into prose. |
| 20 | T2 | MEDIUM | "\`\`numpy\`\` and \`\`scipy\`\` are required by the Python package itself and are" | "The Python package always installs \`\`numpy\`\` and \`\`scipy\`\`." |
| 43 | T9 | MEDIUM | "Run \`\`./install.sh --help\`\` for all options (skip steps, verbose output, etc.)." | Replace "etc." with an explicit list or "for example, to skip steps or enable verbose output." |
| 49 | T12 | MEDIUM | "**Rust** (required for the PDE solver and CLI):" | Spell out PDE and CLI at first use in this document. |
| 51 | T9 | MEDIUM | "If you don't have Rust installed, the easiest way is via \`rustup <https://rustup.rs/>\`_:" | "...the easiest way is to use \`rustup\`..." |
| 58 | F2 | MEDIUM | "**Python 3.9+** (required for the Python package and notebooks):" | This bold fragment directly precedes a code block with no sentence; add "Create and activate a conda environment:" |
| 65 | F2 | MEDIUM | "**Build and install:**" | Replace the fragment with a full sentence, e.g. "Build the Rust solver and install the Python package:" |
| 75 | H2 | MEDIUM | "Verifying the installation" | Use the imperative: "Verify the installation". |
| 31 / 34 / 37 / 40 | F1 | LOW | "matplotlib" (31); "matplotlib, jupyter" (34); "matplotlib, jupyter, pytest, mutmut" (37); "sphinx, pydata-sphinx-theme, nbsphinx, nbsphinx-link, sphinx-copybutton, ipython" (40) | Wrap each package name in double backticks, consistent with the "Extra" column in the same table (\`\`plot\`\`, \`\`notebook\`\`, etc.). |

#### `docs/cli.rst`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 4 | T12 | HIGH | "The \`\`spectroxide\`\` binary provides a command-line interface to the PDE solver." | Spell out PDE at first use. |
| 5 | T2 | MEDIUM | "Output is JSON by default and written to stdout." | "The CLI writes JSON output to stdout by default." |
| 12 / 15 | H4 | MEDIUM | "Subcommands" (12) directly followed by "\`\`solve\`\`" (15) | Add one sentence under "Subcommands" before the first subcommand heading, e.g. "spectroxide provides the following subcommands:" |
| 15,51,60,70,79,88 | H3 | MEDIUM | "\`\`solve\`\`" and 5 siblings (see Systematic patterns) | Remove code font from heading text. |
| 24 | L2 | MEDIUM | "**Injection types:**" | Replace the bold fragment with a full sentence: "The \`\`solve\`\` subcommand supports the following injection types:" |
| 9 | F3 | HIGH | "cargo run --release --bin spectroxide -- <subcommand> [options]" | Use \`\`SUBCOMMAND\`\` (upper snake case) and add "Replace SUBCOMMAND with one of the subcommands listed below." |
| 22 | F3 | HIGH | "spectroxide solve <injection-type> [options]" | Use \`\`INJECTION_TYPE\`\` and explain it below the block. |
| 111,114,117,123,126,129,141,147 | F3 | MEDIUM | "\`\`--z-start <z>\`\`" and 7 siblings (see Systematic patterns) | Standardize on \`\`Z\`\`, \`\`N\`\`, \`\`VAL\`\` and add one explanatory sentence after the table. |
| 47 | F3 | MEDIUM | "\`\`--heating-table PATH\`\` (CSV: \`\`z,dq_dz\`\`)" | Add "Replace PATH with the path to a CSV file with columns z, dq_dz." |
| 49 | F3 | MEDIUM | "\`\`--photon-table PATH\`\` (CSV: \`\`z,x1,...,xN\`\`)" | Add "Replace PATH with the path to a CSV file with columns z, x1, ..., xN." |
| 121,124,133,136,139,145,148 | L5 | MEDIUM | empty "-" table cell in the Default column (see Systematic patterns) | Fill with "off" or add a footnote explaining blank defaults. |
| 137 | T12 | MEDIUM | "- Operator-split DC/BR instead of coupled Newton iteration." | Spell out DC (double Compton) and BR (bremsstrahlung) at first use in this document. |
| 122 | F8 | MEDIUM | "Use the high-resolution production grid preset (4000 points)." | "4,000 points" — comma for numbers ≥ 1,000 in prose. |
| 159-160 | F6 | LOW | "**same convention** as the Python API: \`\`--omega-b\`\` is the fractional" | Remove bold; if emphasis is needed use italics, e.g. "*same convention*". |
| 167 | F11 | MEDIUM | "\`\`--omega-b\`\` and \`\`--omega-m\`\` must be supplied together — the CDM" | Remove the spaces around the em dash: "together—the CDM". |
| 168 | T2 | MEDIUM | "density is derived from their difference." | "the CLI derives the CDM density from their difference." |
| 171 / 174 | L2 | MEDIUM | "Output options" (171) directly followed by ".. list-table::" (174) | Add "The following flags control output format and destination:" |
| 151 / 154 | F2 | MEDIUM | "Cosmology options" (151) / ".. code-block:: bash" (154) | Add "Select a cosmology preset or override individual parameters:" |
| 186 / 189 | F2 | MEDIUM | "Examples" (186) / ".. code-block:: bash" (189) | Add "The following commands illustrate common usage:" |
| 1 | T12 | LOW | "CLI reference" | Acronym used in the title before being spelled out on line 4; consider "Command-line interface (CLI) reference". |

#### `docs/rust_api.rst`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 4 | T12 | HIGH | "The Rust crate \`\`spectroxide\`\` is the PDE solver itself; the Python package is a" | Spell out PDE at first use. |
| 5 | T12 | MEDIUM | "thin wrapper that calls the Rust CLI for heavy computations and provides a" | Spell out CLI at first use. |
| 14 | H2 | MEDIUM | "Browsing the crate documentation" | Use the imperative: "Browse the crate documentation". |
| 25 | T14 | LOW | "Built from the current source tree and served alongside this site." | "This build uses the current source tree and is served alongside this site." (state the subject; avoid the dropped-subject passive fragment). |
| 32-33 | T11 | LOW | "\`\`InjectionScenario\`\`, \`\`RefinementZone\`\`, and the \`\`SolverBuilder\`\` /" / "\`\`SolverDiagnostics\`\` / output types." | Replace "/" with commas or "and": "...and the \`\`SolverBuilder\`\`, \`\`SolverDiagnostics\`\`, and output types." |
| 35 / 38 | F2 | MEDIUM | "Quick Rust example" (35) / ".. code-block:: rust" (38) | Add "The following snippet builds a solver and runs it to a single redshift:" |
| 56 | H3 | MEDIUM | "CLI" | Undefined abbreviation used bare as a heading; spell out or link with explanatory text, e.g. "Command-line interface". |
| 6 | T5 | LOW | "pure-Python Green's function for quick estimates. Most users only touch Python," | "quick" is a claims-about-ease/speed word per the word list; consider "for fast estimates" only if quantified, or drop the adjective. |

#### `docs/tutorials/index.rst`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 23 | T12 | MEDIUM | "- **Getting started** --- Green's function basics, first PDE runs, PDE vs GF comparison." | Spell out PDE at first use in this document. |
| 25 | T12 | MEDIUM | "- **Energy injection** --- Decaying particles, DM annihilation (s-wave, p-wave), amplitude scaling." | Spell out DM (dark matter) at first use. |
| 31 | T12 | LOW | "- **Observational constraints** --- FIRAS/PIXIE limits, :math:\`\mu\`--:math:\`y\` exclusion plane, mock PIXIE observation." | Spell out FIRAS and PIXIE at first use (mission/instrument names). |
| 31 | T11 | LOW | "- **Observational constraints** --- FIRAS/PIXIE limits, :math:\`\mu\`--:math:\`y\` exclusion plane, mock PIXIE observation." | "/" stands in for "and"; write "FIRAS and PIXIE limits". |
| 22,23,24,25,26,27,28,29,30,31,32,33 | F6 | LOW | "**01**", "**Getting started**" and 10 similar (>10 instances; grep: \`grep -n '\*\*' docs/tutorials/index.rst\`) | Bold used purely for table-cell emphasis, not a UI element or notice label. Use plain text or a definition list instead of a headerless list-table. |
| 23,25,27,29,31,33 | F11 | MEDIUM | "- **Getting started** --- Green's function basics, first PDE runs, PDE vs GF comparison." and 5 siblings | The em-dash-rendering "---" carries spaces on both sides; Google style uses an unspaced em dash. Change "text --- text" to "text—text". |
| 18-33 | L5 | MEDIUM | ".. list-table::" with ":header-rows: 0" (line 20) | Table has no header row, so the two columns ("number", "description") are unlabeled. Add a header row or convert to an RST definition list, since this is a term–definition pairing. |

### What conforms

- T1 (second person / no "we", "one", "let's"): none found in any of the five files.
- T3 / T6 (present tense, no "will", no "currently"/"now"/"latest"/"soon"/"recently" in prose): clean across all five files.
- T7 (no exclamation marks, no marketing language, no idioms): clean.
- T8 (condition before instruction): `docs/installation.rst:51` — "If you don't have Rust installed, the easiest way is..." — correct order.
- T10 (inclusive language): no flagged terms found.
- T15 (serial comma, American spelling, one space after periods): serial comma used consistently, e.g. `docs/index.rst:29-30` "injection scenarios, photon injection, and tabulated sources"; no double-space-after-period instances found.
- F1 (code font for flags and identifiers): flags, file names, and identifiers are correctly double-backtick-fenced almost everywhere, including every command-line example (`docs/cli.rst`, `docs/installation.rst`).
- F4 (link text): no "click here" / "this link" / bare-URL link text found.
- F7 (notices): no `.. note::`/`.. warning::` directives present, so no misuse to flag.
- F12 (`cargo` vs `Cargo`): `docs/rust_api.rst:27` correctly lowercases the command "\`\`cargo\`\` on \`\`PATH\`\`" while `docs/installation.rst` and `docs/cli.rst` capitalize "Rust" and "Python" correctly as product names throughout.
- H4 (one H1 per page): each of the five files has exactly one title-level heading.


---

## 4. Sphinx site: API pages

Files reviewed: `docs/api/index.rst` (124 lines), `docs/api/solver.rst` (304 lines), `docs/api/greens.rst` (296 lines), `docs/api/greens_table.rst` (131 lines), `docs/api/firas.rst` (150 lines), `docs/api/cosmology.rst` (122 lines), `docs/api/dark_photon.rst` (39 lines), `docs/api/axion.rst` (50 lines), `docs/api/style.rst` (38 lines). Total 1,254 lines. Autodoc/autosummary/autoclass/automethod/automodule directive blocks and their target docstrings were skipped (owned by the docstring reviewer); only hand-written prose, headings, lists, tables, and code samples were checked.

### Summary

The dominant problem is an undefined-abbreviation gap: "PDE" is the subject of the entire site (it is literally the title of three pages) and is never spelled out as "partial differential equation" anywhere in `docs/api/`; "FIRAS" is never expanded on its own dedicated page either. The second-largest issue is a mechanical formatting habit repeated on every page: em dashes are typed with a space on each side (`word — word`) in 29 places, and every page title puts the Python module path in code font inside the H1/H2 heading text (a `` `` code-span in the heading itself``), which Google style disallows. Two structural problems stand out: `docs/api/axion.rst` opens with a `.. warning::` block *before* its own page title, so the page has no introductory sentence at all, and `docs/api/solver.rst` runs two subsection headings and five tabbed code examples straight into bare `.. code-block::` directives with no introducing sentence. Passive voice ("is cached", "is tagged", "is mapped") appears through the infrastructure-description prose in `solver.rst` and `greens_table.rst`, and bold is used for plain emphasis (not UI elements or notice labels) in about a dozen places, most concentrated in `greens.rst` and `solver.rst`. Word-choice issues ("e.g.", "vs.", "etc.", "above") and passive-inflected "users will" language are present but comparatively minor and confined to a handful of lines.

| Rule | Count | Severity |
|---|---|---|
| T12 Undefined abbreviation "PDE" (page-central use) | 4 | HIGH |
| T12 Undefined abbreviation "PDE" (peripheral use) | 2 | MEDIUM |
| T12 Undefined abbreviation "FIRAS" (own dedicated page) | 1 | HIGH |
| T12 Undefined abbreviation "CMB" | 2 | MEDIUM |
| T12 Undefined abbreviation "CLI" | 1 | MEDIUM |
| T12 Undefined abbreviation "DC/BR" | 1 | MEDIUM |
| T12 Undefined abbreviation "ODE" | 1 | LOW |
| T12 "GF" used without a formal definition | 1 | LOW |
| T12 "NWA" used in `index.rst` without local definition | 1 | MEDIUM |
| T12 "NWA" defined but the short form is never reused | 2 | LOW |
| H5 Page opens with a notice before its own title | 1 | HIGH |
| H5 Page opens with a notice, no lead paragraph first | 1 | MEDIUM |
| H3 Code font inside the page-title heading | 9 | LOW |
| H3 Heading that is entirely a code span | 4 | MEDIUM |
| L1 Numbered list used for non-sequential items | 1 | MEDIUM |
| L1 Term/definition pairs formatted as a bullet list | 2 | MEDIUM |
| L2 List/table not introduced by a colon-terminated sentence | 4 | MEDIUM |
| F1 Bare identifier not in code font | 2 | MEDIUM |
| F1 Code font used on prose percentages, not code | 3 | MEDIUM |
| F1 Class name in a heading without code font | 1 | MEDIUM |
| F2 Bare code block immediately after a heading | 11 | MEDIUM |
| F3 Angle-bracket placeholder notation | 1 | LOW |
| F4 Non-descriptive link text ("paper") | 1 | LOW |
| F6 Bold used for emphasis (phrase) | 5 | MEDIUM |
| F6 Bold used for emphasis (single word) | 8 | LOW |
| F6 Italics used for emphasis, not term introduction | 2 | LOW |
| F11 Em dash with a space on both sides | 29 | LOW |
| F12 Product name miscapitalized ("numpy") | 1 | MEDIUM |
| T1 Third person ("users") instead of "you" | 2 | MEDIUM |
| T2 Passive voice, actor unstated | 12 | MEDIUM |
| T3 Future tense ("will") for product behavior | 1 | MEDIUM |
| T4 Pre-announce pattern ("Note X is Y") | 1 | MEDIUM |
| T7 Marketing/opinion phrasing | 1 | LOW |
| T8 Instruction stated before its condition/goal | 3 | MEDIUM |
| T9 "e.g." in prose | 2 | MEDIUM |
| T9 "vs." in prose | 1 | MEDIUM |
| T9 "etc." in prose | 1 | MEDIUM |
| T9 "above" for document position | 1 | MEDIUM |
| T9 "/" standing in for "and"/"or" | 2 | LOW |
| T14 Telegraphic sentence fragments (card blurbs) | 8 | MEDIUM |
| T15 Double space after a period | 3 | LOW |

**Totals: 6 HIGH, 74 MEDIUM, 60 LOW — 140 findings.** Symbol use that is standard physics/scientific notation (`×` in "3 × 10⁴", "43 × 43"; `↔` in "γ ↔ A'", "γ ↔ a"; `~` in "~8–13%") was checked and classified DOMAIN — not counted above, not flagged individually.

### Systematic patterns

**F11 — em dash typed with a space on each side.** Google style sets em dashes without surrounding spaces. Pattern: `grep -n ' — ' *.rst`. 29 instances across 6 of 9 files (`index.rst` 9, `solver.rst` 10, `greens.rst` 5, `axion.rst` 3, `dark_photon.rst` 1, `style.rst` 1). First 10:
- `index.rst:25` — "PDE solver — :doc:`solver`"
- `index.rst:50` — "``spectroxide.greens`` — pure-Python implementation of the"
- `index.rst:54` — "approximation** — accuracy is documented on that page."
- `index.rst:60` — "``spectroxide.greens_table`` — precomputed numerical Green's"
- `index.rst:69` — "``spectroxide.firas`` — load the COBE/FIRAS monopole, residuals,"
- `index.rst:77` — "``spectroxide.cosmology`` — flat ΛCDM background quantities"
- `index.rst:87` — "``spectroxide.dark_photon`` — NWA helpers (ω_pl, z_res, γ_con) for"
- `index.rst:95` — "``spectroxide.axion`` — NWA helpers for resonant γ↔a conversion."
- `index.rst:103` — "``spectroxide.style`` and ``spectroxide.plot_params`` — Matplotlib"
- `solver.rst:37` — "- :func:`solve` — returns a structured :class:`SolverResult`."
Suggested rewrite: close up the spaces (`word—word`) or, in headings and card titles, drop the dash and use a colon or separate sentence.

**H3 — module path in code font inside a heading.** Every page title repeats the pattern `Name (``spectroxide.module``)`, and two files title a subsection with nothing but a code span. Pattern: `grep -n -B1 -E '^(=+|-+|~+)\s*$' *.rst` then inspect the heading line for double backticks. 13 instances (listed under Findings by file below). Suggested rewrite: move the module path out of the heading text into the first sentence of body text (most pages already restate it there via `.. currentmodule::` and the opening paragraph), or keep it but accept as a deliberate site convention — the reviewer's severity split (LOW for the title line, MEDIUM for `` `GreensTable` `` / `` `PhotonGreensTable` `` / `` `dq_dz` `` / `` `photon_source` `` subheadings that are pure code) reflects that the full-title case at least carries surrounding words while the subsection case does not.

**F2 — bare code block right after a heading.** Google style requires a sentence (usually ending in a colon) before every code sample; four `Quick example` sections and two `solver.rst` subsections jump straight from the underlined heading to `.. code-block::`, and all five `solver.rst` tab-items do the same. Pattern: heading line, blank line, `.. code-block::` with no prose paragraph between. 11 instances: `greens.rst:37-40`, `greens_table.rst:17-20`, `firas.rst:14-17`, `cosmology.rst:14-17`, `solver.rst:101-104` (`` `dq_dz` `` heading), `solver.rst:123-126` (`` `photon_source` `` heading), and the five tab-items in `solver.rst:155-157/172-174/186-188/199-201/209-211`. Suggested rewrite: add one sentence stating what the example shows, e.g. "Load a table from cache or build one via the Rust PDE:" before each block.

**T2 — passive voice describing what the package/module does.** The actor (the wrapper, the loader, the cache) is grammatically absent throughout the infrastructure-description paragraphs. 12 instances across `index.rst`, `cosmology.rst`, `greens.rst`, `greens_table.rst`, `solver.rst` (see Findings by file). Suggested rewrite pattern: "X is mapped/cached/tagged" → "The wrapper maps X" / "`load_or_build_*` caches X" / "the loader tags X".

### Findings by file

#### `docs/api/index.rst`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 4 | T12 | HIGH | "The Python package ``spectroxide`` wraps the Rust PDE solver and provides" | Spell out on first use: "...the Rust partial differential equation (PDE) solver..." |
| 5 | T1 | MEDIUM | "a pure-Python analytic Green's-function implementation. Most users will" | "You need only the top-level import" |
| 5 | T3 | MEDIUM | "Most users will" | Drop "will": "Most users need only the top-level import." |
| 12 | T2 | MEDIUM | "Plot styling lives in submodules and must be imported explicitly:" | "Import plot styling explicitly from its submodule:" |
| 25 | F11 | LOW | "PDE solver — :doc:`solver`" | "PDE solver—:doc:`solver`" or restructure without the dash |
| 29 | T14 | MEDIUM | "``spectroxide.solver``. The full photon-Boltzmann PDE" | Combine into one sentence with a verb: "``spectroxide.solver`` runs the full photon-Boltzmann PDE..." |
| 32-33 | F6 | MEDIUM | "you almost certainly want.**" | Remove bold; state plainly: "Start here for most workflows." |
| 32-33 | T7 | LOW | "you almost certainly want.**" | Replace subjective framing with a factual statement of scope |
| 41 | T2 | MEDIUM | "the science targets are computed by the PDE solver above." | "the PDE solver computes the science targets shown preceding." |
| 41 | T9 | MEDIUM | "the science targets are computed by the PDE solver above." | Replace "above" with "preceding" or a direct `:doc:` link |
| 50 | F11 | LOW | "``spectroxide.greens`` — pure-Python implementation of the" | Close up dash spacing |
| 50 | T14 | MEDIUM | "``spectroxide.greens`` — pure-Python implementation of the" | Add a subject and verb: "``spectroxide.greens`` provides a pure-Python implementation..." |
| 52 | T9 | LOW | "436, 2232). Spectral shapes, μ/y/T branching functions," | "μ, y, and T branching functions" |
| 54 | F11 | LOW | "approximation** — accuracy is documented on that page." | Close up dash spacing |
| 54 | F6 | MEDIUM | "approximation** — accuracy is documented on that page." | Remove bold from "An approximation"; state it as plain prose |
| 54 | T2 | LOW | "approximation** — accuracy is documented on that page." | "see the accuracy figures on this page" |
| 60 | F11 | LOW | "``spectroxide.greens_table`` — precomputed numerical Green's" | Close up dash spacing |
| 60 | T14 | MEDIUM | "``spectroxide.greens_table`` — precomputed numerical Green's" | Add a verb: "``spectroxide.greens_table`` provides a precomputed numerical Green's function..." |
| 62 | T12 | LOW | "accurate than the analytic GF in the μ↔y transition region" | Spell out or drop the abbreviation: "the analytic Green's function" |
| 69 | F11 | LOW | "``spectroxide.firas`` — load the COBE/FIRAS monopole, residuals," | Close up dash spacing |
| 69 | T14 | MEDIUM | "``spectroxide.firas`` — load the COBE/FIRAS monopole, residuals," | Add a subject: "``spectroxide.firas`` loads the COBE/FIRAS monopole..." |
| 77 | F11 | LOW | "``spectroxide.cosmology`` — flat ΛCDM background quantities" | Close up dash spacing |
| 77 | T14 | MEDIUM | "``spectroxide.cosmology`` — flat ΛCDM background quantities" | Add a verb: "``spectroxide.cosmology`` provides flat ΛCDM background quantities..." |
| 87 | F11 | LOW | "``spectroxide.dark_photon`` — NWA helpers (ω_pl, z_res, γ_con) for" | Close up dash spacing |
| 87 | T12 | MEDIUM | "``spectroxide.dark_photon`` — NWA helpers (ω_pl, z_res, γ_con) for" | Expand on first use: "narrow-width-approximation (NWA) helpers" |
| 87 | T14 | MEDIUM | "``spectroxide.dark_photon`` — NWA helpers (ω_pl, z_res, γ_con) for" | Add a subject and verb |
| 95 | F11 | LOW | "``spectroxide.axion`` — NWA helpers for resonant γ↔a conversion." | Close up dash spacing |
| 95 | T14 | MEDIUM | "``spectroxide.axion`` — NWA helpers for resonant γ↔a conversion." | Add a subject and verb |
| 96 | F6 | LOW | "**Experimental**; the PDE path needs a binary built with" | Move status to a `.. note::`/`.. warning::` label rather than inline bold |
| 103 | F11 | LOW | "``spectroxide.style`` and ``spectroxide.plot_params`` — Matplotlib" | Close up dash spacing |
| 103 | T14 | MEDIUM | "``spectroxide.style`` and ``spectroxide.plot_params`` — Matplotlib" | Add a subject and verb |

#### `docs/api/solver.rst`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | LOW | "PDE solver (``spectroxide.solver``)" | Move the module path into the opening sentence, keep the title plain |
| 11 | F6 | MEDIUM | "This page documents the **PDE solver only**. For the analytic Green's" | Remove bold; the sentence carries the emphasis on its own |
| 17 | T12 | (see index/greens rows) | — | — |
| 26 | T13/T1 (conforms) | — | "Pick by *what you're scanning over*:" | (contraction is acceptable under Google style; flagged separately below for italics) |
| 26 | F6 | LOW | "Pick by *what you're scanning over*:" | Use plain text, not italics, for emphasis: "Pick by what you're scanning over:" |
| 37 | F11 | LOW | "- :func:`solve` — returns a structured :class:`SolverResult`." | Close up dash spacing |
| 40 | F11 | LOW | "- :func:`run_sweep` — one Rust process loops over ``z_injections``" | Close up dash spacing |
| 44 | F11 | LOW | "- :func:`run_photon_sweep` — same idea for the photon-sweep case." | Close up dash spacing |
| 49 | F11 | LOW | "``m_ev``, ``f_x``, …), call :func:`solve` in a Python loop — there is" | Close up dash spacing |
| 58 | F6 | LOW | "``delta_rho`` is always a **top-level** argument (not an injection key)." | Remove bold; not a UI element or notice label |
| 60 | F4 | LOW | "` \`paper <https://arxiv.org/abs/2604.24838>\`_.`" | Use a descriptive link text, e.g. "the paper (arXiv:2604.24838)" |
| 61-82 | L2 | MEDIUM | "`paper <https://arxiv.org/abs/2604.24838>`_." (line 60, precedes list-table at 62) | Add a colon-terminated lead sentence directly before the table, e.g. "The built-in scenarios and their required keys are:" |
| 87 | T2 | MEDIUM | "Each parameter name is mapped to the corresponding Rust CLI flag" | "The wrapper maps each parameter name to the corresponding Rust CLI flag" |
| 87 | T12 | MEDIUM | "Each parameter name is mapped to the corresponding Rust CLI flag" | Expand on first use: "command-line interface (CLI) flag" |
| 88 | T9 | MEDIUM | "``--<kebab-case>`` (e.g. ``f_x → --f-x``, ``delta_n_over_n →" | "for example" instead of "e.g." |
| 88 | F3 | LOW | "``--<kebab-case>`` (e.g. ``f_x → --f-x``, ``delta_n_over_n →" | Angle-bracket placeholder notation; acceptable here since two worked examples follow immediately, but consider `UPPER_SNAKE_CASE`-style placeholder convention for consistency |
| 97 | T2 | MEDIUM | "``injection`` dict.  Both are tabulated on a log-spaced grid and" | "The wrapper tabulates both on a log-spaced grid and dispatches them..." |
| 97 | T15 | LOW | "``injection`` dict.  Both are tabulated on a log-spaced grid and" | Single space after the period |
| 99 | T2 | LOW | "outside the integration range are treated as zero." | "the solver treats values outside the integration range as zero." |
| 99 | F11 | (below) | — | — |
| 101 | H3 | MEDIUM | "``dq_dz`` — energy-injection history" | Rewrite as a plain-text subheading, e.g. "Energy-injection history (`dq_dz`)" |
| 101 | F11 | LOW | "``dq_dz`` — energy-injection history" | Close up dash spacing |
| 101-104 | F2 | MEDIUM | "``dq_dz`` — energy-injection history" (heading), code block follows at line 104 with no prose between | Add an introducing sentence ending in a colon before the code block |
| 109 | L1 | MEDIUM | "* **Signature**: ``dq_dz(z) -> float`` (or array). The wrapper attempts a" | Use an RST definition list (`Signature\n    dq_dz(z) -> float`) instead of a bulleted list with bold pseudo-terms |
| 111 | F11 | LOW | "evaluation. Vectorise where you can — the tabulation grid has 5000" | Close up dash spacing |
| 119 | T2 | LOW | "* **Mode**: with ``method=\"pde\"`` the callable is tabulated and the Rust" | "the wrapper tabulates the callable and the Rust PDE solver integrates it" |
| 123 | H3 | MEDIUM | "``photon_source`` — frequency-dependent photon injection" | Rewrite as plain-text subheading |
| 123 | F11 | LOW | "``photon_source`` — frequency-dependent photon injection" | Close up dash spacing |
| 123-126 | F2 | MEDIUM | "``photon_source`` — frequency-dependent photon injection" (heading), code block follows at line 126 with no prose between | Add an introducing sentence ending in a colon |
| 132 | L1 | MEDIUM | "* **Signature**: ``photon_source(x, z) -> float``. Called scalar-by-scalar" | Same as line 109 — use a description list |
| 137 | F11 | LOW | "* **Frequency**: :math:`x = h\\nu/(k_{\\rm B} T_z)` — the same dimensionless" | Close up dash spacing |
| 150-157 | F2 | MEDIUM | "`.. tab-item:: Single PDE solve`" (155), code block follows at 157 with no prose between | Add one sentence inside each tab-item before the code block |
| 172-174 | F2 | MEDIUM | "`.. tab-item:: Redshift sweep`" (172), code block follows at 174 with no prose between | Same fix |
| 186-188 | F2 | MEDIUM | "`.. tab-item:: Photon injection`" (186), code block follows at 188 with no prose between | Same fix |
| 199-201 | F2 | MEDIUM | "`.. tab-item:: Tabulated heating`" (199), code block follows at 201 with no prose between | Same fix |
| 209-211 | F2 | MEDIUM | "`.. tab-item:: Parameter scan`" (209), code block follows at 211 with no prose between | Same fix |
| 216 | F11 | LOW | "# Scan decaying-particle gamma_x — call solve() in a Python loop" | (inside code-block comment — informational only, not counted; code-block text is exempt from prose rules) |
| 230-231 | (conforms) | — | "``run_sweep`` accept either a :class:`~spectroxide.cosmology.Cosmology`" | Correct use of `:class:` cross-reference |
| 234-235 | (conforms) | — | "Reference" | Sentence-case heading, no issue |
| 256 | T4 | MEDIUM | "converting to intensity units. Note ``accumulated_delta_t`` is a" | "``accumulated_delta_t`` is a PDE-only diagnostic..." (drop "Note") |
| 258 | F6 | LOW | "non-conservation), **not** a full ΔT/T fit component and typically 0.0;" | Remove bold from "not" |
| 279 | F6 | LOW | "around the **analytic** Green's function in :mod:`spectroxide.greens`." | Remove bold |
| 280 | F6 | LOW | "PDE — it bundles single-burst and custom-heating calculations into a" | Remove bold from "not" (line 280 continuation) |
| 281 | F11 | LOW | "PDE — it bundles single-burst and custom-heating calculations into a" | Close up dash spacing |
| 293 | F8 (conforms) | — | "Two preset dicts control grid resolution and timestep caps for" | "Two" spelled out correctly |

#### `docs/api/greens.rst`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | LOW | "Analytic Green's function (``spectroxide.greens``)" | Move module path out of the heading text |
| 7-14 | H5 | MEDIUM | "This module is the **analytic three-component approximation** of" (first content block is a `.. important::`, no lead paragraph precedes it) | Add one introductory sentence stating what `spectroxide.greens` is, before the `.. important::` box |
| 9 | F6 | MEDIUM | "This module is the **analytic three-component approximation** of" | Remove bold, or use italics only for the first introduction of the term |
| 11 | F6 | LOW | "Python. It is *not* the PDE solver — for production work prefer the" | Remove italics used purely for emphasis |
| 11 | F11 | LOW | "Python. It is *not* the PDE solver — for production work prefer the" | Close up dash spacing |
| 16-17 | F11 | LOW | "The Chluba ansatz decomposes the distortion into three channels — μ, y," | Close up dash spacing (both dashes on lines 16 and 17) |
| 32 | F1 | MEDIUM | "**Accuracy vs. PDE** — ``<5%`` deep μ-era (z_h > 2 × 10⁵), ``<1%``" | Drop the code font from `<5%`/`<1%` — these are prose values, not code |
| 32 | F1 | MEDIUM | "**Accuracy vs. PDE** — ``<5%`` deep μ-era (z_h > 2 × 10⁵), ``<1%``" | Put the bare "z_h" in code font: "``z_h``" |
| 32 | F9/T9 | MEDIUM | "**Accuracy vs. PDE** — ``<5%`` deep μ-era (z_h > 2 × 10⁵), ``<1%``" | Replace "vs." with "compared to" or "versus" |
| 32 | F11 | LOW | "**Accuracy vs. PDE** — ``<5%`` deep μ-era (z_h > 2 × 10⁵), ``<1%``" | Close up dash spacing |
| 33 | F1 | MEDIUM | "y-era (z_h < 10⁴), ``~8–13%`` shape error in the μ↔y transition" | Put "z_h" in code font; drop code font from "~8–13%" |
| 37-40 | F2 | MEDIUM | "Quick example" (37), code block follows at 40 with no prose between | Add a sentence: "Compute the distortion for a delta-function injection:" |
| 79 | T2 | LOW | "Three presets are provided:" | "The module provides three presets:" |
| 174 | (code block, exempt) | — | "# The prefactor is chosen so that ∫dQ/dz dz ≈ Δρ/ρ ~ 1e-5, i.e. the" | Not flagged — inside a code-block comment |
| 182 | T9 (conforms) | — | "Compton visibility into the y-channel automatically (via the bundled" | "via" here is acceptable technical usage, borderline; not flagged |
| 195 | T12 | MEDIUM | "``z_h`` (Chluba 2015). Includes critical frequencies for DC/BR" | Spell out on first use: "double Compton (DC) / bremsstrahlung (BR) absorption" |
| 234-238 | (conforms) | — | "Decompose an arbitrary Δn(x) into (μ, y, ΔT/T) components and convert" | Clear intro sentence before the list |
| 241 | F6 | LOW | "The Python :func:`delta_n_to_delta_I` returns **Jy/sr**, whereas the" | Remove bold; state the contrast in plain text |
| 242 | F6 | LOW | "Rust ``distortion.rs`` converter returns **MJy/sr** (a factor of 10⁶)." | Remove bold |
| 259 | F11 | LOW | ":func:`decompose_distortion` is the entry point — it dispatches via a" | Close up dash spacing |
| 261 | T15 | LOW | "(``\"be\"``, default) or the linear Gram-Schmidt fit (``\"gs\"``).  Both" | Single space after the period |
| 269-277 | (conforms) | — | "FIRAS measures the CMB spectrum with the absolute temperature as a free" | — |
| 272 | T12 | MEDIUM | "FIRAS measures the CMB spectrum with the absolute temperature as a free" | Spell out on first use: "cosmic microwave background (CMB) spectrum" |
| 277 | T2 | LOW | "perturbation is absorbed into :math:`\\alpha \\cdot G_{bb}(x)`." | "CosmoTherm absorbs any nonzero photon-number perturbation into..." |
| 288-296 | (conforms) | — | "Convenience wrapper" section | Clear, active-ish prose |
| 293 | F11 | LOW | "It is documented on the" | Close up dash spacing (part of "`:func:`spectroxide.solver.run_single`... is documented on the`") — see next row |
| 295 | F6 | LOW | "PDE. It is documented on the" (continues) "despite living there, it does **not** invoke the Rust" | Remove bold from "not" |

#### `docs/api/greens_table.rst`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | LOW | "PDE-based numerical Green's function (``spectroxide.greens_table``)" | Move module path out of the heading |
| 7-15 | T12 | HIGH | "Precomputed numerical Green's function from the Rust PDE, tabulated for" | Spell out "PDE" on first use in this file too (each document must define it independently) |
| 13-15 | T8 | MEDIUM | "Use this when you need fast convolution but want PDE-quality results" | "When you need fast convolution..., use this table." |
| 14 | T9 | MEDIUM | "(e.g. parameter scans where running the full PDE per point is too" | "for example" instead of "e.g." |
| 17-20 | F2 | MEDIUM | "Quick example" (17), code block follows at 20 with no prose between | Add a sentence introducing the example |
| 44-45 | H3 | MEDIUM | "``GreensTable``" | Rewrite as plain text: "The `GreensTable` class" |
| 67-68 | H3 | MEDIUM | "``PhotonGreensTable``" | Rewrite as plain text: "The `PhotonGreensTable` class" |
| 70-77 | (conforms) | — | ".. note:: The photon Green's function depends on both..." | Appropriate, single, non-stacked note |
| 99-100 | T2 (conforms — active) | — | "Tables are expensive to build (one PDE solve per ``z_h``), so the" | Active voice, fine |
| 101 | T2 | MEDIUM | "across sessions. Each cached table is tagged with a hash of the physics" | "the loader tags each cached table with a hash of the physics configuration" |
| 102 | T2 | MEDIUM | "configuration used to generate it; on load, the hash is checked against" | "on load, the loader checks the hash against the current code" |
| 102-104 | T8 | MEDIUM | "the current code and a :class:`GreensTableHashMismatch` warning is" | State the condition first: "If they disagree, the loader emits a `GreensTableHashMismatch` warning" |
| 105-106 | T8 | MEDIUM | "not silently shadow updated physics. Pass ``rebuild=True`` to force a" | "To force a fresh build, pass ``rebuild=True``" |
| 108 | (conforms) | — | "The hash itself comes from :func:`~spectroxide.solver.get_physics_hash`," | Clear cross-reference |
| 118-122 | (conforms) | — | "Cache-aware constructors that load from disk or trigger a Rust PDE" | — |

#### `docs/api/firas.rst`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | LOW | "FIRAS data (``spectroxide.firas``)" | Move module path out of heading |
| 1-9 | T12 | HIGH | "FIRAS data (``spectroxide.firas``)" | Expand "FIRAS" on first use — this is its own dedicated page and the acronym is never spelled out anywhere in `docs/api/` |
| 9 | F12 | MEDIUM | "numpy arrays. A :class:`FIRASData` instance is the primary handle: its" | Capitalize the product name: "NumPy arrays" |
| 12 | T9 | MEDIUM | "fits over a free CMB temperature, etc.)." | Drop "etc." and name the remaining items, or end the list explicitly |
| 12 | T12 | MEDIUM | "fits over a free CMB temperature, etc.).” | Spell out "CMB" on first use in this file |
| 14-17 | F2 | MEDIUM | "Quick example" (14), code block follows at 17 with no prose between | Add an introducing sentence |
| 35-41 | L2 | MEDIUM | "without constructing a :class:`FIRASData` object." (39, precedes list-table at 41) | Add a colon-terminated lead sentence: "The precomputed constants are:" |
| 80-86 | L2 | MEDIUM | "``sigma_kJy`` is the diagonal only." (84, precedes list-table at 86) | Add a colon-terminated lead sentence: "Construction populates these attributes:" |
| 128-150 | (conforms) | — | "Full-covariance χ² primitive plus the headline upper-limit and" | Clear intro before the autosummary/automethod list |

#### `docs/api/cosmology.rst`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | LOW | "Cosmology (``spectroxide.cosmology``)" | Move module path out of heading |
| 8-10 | T12 | MEDIUM | ":class:`Cosmology` dataclass. The Green's-function and PDE-table" | Spell out "PDE" on first use in this file |
| 10 | T1 | MEDIUM | "themselves; users can either pass a :class:`Cosmology` instance or a" | "you can either pass a `Cosmology` instance" |
| 14-17 | F2 | MEDIUM | "Quick example" (14), code block follows at 17 with no prose between | Add an introducing sentence |
| 61-67 | L2 | MEDIUM | "``omega_m``, ``y_p``, ``t_cmb``, ``n_eff``." (65, precedes list-table at 67) | Add a colon-terminated lead sentence: "The presets are:" |
| 110-115 | (conforms) | — | "Free-electron fraction ``X_e(z)``: Saha for helium, Peebles three-level" | — |
| 115 | T12 | LOW | "2011). The ODE table is cached per cosmology." | Spell out "ODE" on first use: "ordinary differential equation (ODE)" |
| 115 | T2 | LOW | "2011). The ODE table is cached per cosmology." | "the module caches the ODE table per cosmology" |

#### `docs/api/dark_photon.rst`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | LOW | "Dark-photon helpers (``spectroxide.dark_photon``)" | Move module path out of heading |
| 6-9 | (conforms) | — | "Pure-Python narrow-width-approximation (NWA) helpers for resonant" | NWA correctly expanded at first use |
| 9 | F11 | LOW | "numbers. Not re-exported at the top level — import explicitly:" | Close up dash spacing |
| 9 | T14 | (not flagged) | "numbers. Not re-exported at the top level — import explicitly:" | Fragment, but functions as a standard "not re-exported" convention repeated verbatim across files — left unflagged for consistency with `style.rst`/`axion.rst`, which use the identical phrase |
| — | T12 | LOW | "narrow-width-approximation (NWA) helpers for resonant" (line 6) | "NWA" is defined but the short form is never used again in this file; either drop the parenthetical or reuse "NWA" later |

#### `docs/api/axion.rst`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 1-6 | H5 | HIGH | ".. warning::" (line 1, precedes the page title at line 7) | Move the page title above the warning, and add a one-sentence introduction before the warning block: "`spectroxide.axion` provides NWA helpers for resonant γ↔a conversion." then the warning |
| 12 | (conforms) | — | "Pure-Python narrow-width-approximation (NWA) helpers for resonant" | NWA correctly expanded at first use |
| — | T12 | LOW | "narrow-width-approximation (NWA) helpers for resonant" (line 12) | "NWA" is defined but never reused in this file |
| 13 | F11 (conforms) | — | "``γ ↔ a`` axion–photon conversion, following Cyr, Chluba & Manoj" | En dash in "axion–photon" is correct range/compound punctuation, not flagged |
| 17 | F11 | LOW | "the top level — import explicitly:" | Close up dash spacing |
| 27-30 | L1 | MEDIUM | "Two differences from the dark-photon case:" (27), followed by a numbered list of two non-sequential facts | Use a bulleted list, not a numbered list — the two items are not sequential steps |
| 30 | F11 | LOW | "photons convert preferentially — opposite to the dark photon's ``1/x``." | Close up dash spacing |
| 36 | F6 | MEDIUM | "This is the **monopole / plasma-frequency** treatment" | Use italics (or no markup) rather than bold for this term introduction |
| 39 | F11 | LOW | "the photon mass (paper Sec. II B, Eqs. 7–12) — which shift the" | Close up dash spacing |
| 41 | F6 | LOW | "are **not** modeled." | Remove bold |

#### `docs/api/style.rst`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 1 | H3 | LOW | "Plotting utilities (``spectroxide.style``, ``spectroxide.plot_params``)" | Move module paths out of heading |
| 5 | F11 | LOW | "publication-quality figures. Not re-exported at the top level — import" | Close up dash spacing |
| 32-33 | H3 | LOW | "Plot parameters (``spectroxide.plot_params``)" | Move module path out of heading |

### What conforms

- Every page is introduced by `.. currentmodule::` plus (except `axion.rst` and `greens.rst`, flagged above) a plain-prose lead paragraph — H5 generally observed.
- No heading level is skipped anywhere (`=` → `-` → `~` order is consistent in every file), and each file has exactly one H1 — H4 fully observed.
- Headings are already in sentence case; no title-case or ALL-CAPS headings were found — H1 observed.
- No task heading is phrased as a gerund ("Installing...") — H2 observed.
- Rust and Python are capitalized correctly everywhere in prose (only lowercase inside `code-block:: python`/`cargo` command syntax, both correct) — F12 mostly observed (one `numpy` exception, flagged).
- No curly quotes anywhere; straight quotes/apostrophes used consistently — part of F11 observed.
- No semicolon chains (never more than one semicolon per sentence) — F11 observed.
- List-tables never have empty cells and are always genuinely two-dimensional (no single-dimension data forced into a table) — L5 fully observed.
- Numbers are handled correctly: zero–nine spelled out in prose ("Three presets", "Four PDE entry points", "Two preset dicts"), numerals used for measurements and 10+, no sentence begins with a bare numeral — F8 fully observed.
- No inclusive-language violations (no "sanity check", "dummy", "blacklist/whitelist", "master/slave", gendered pronouns, etc.) — T10 fully observed.
- No Latin phrases or overly formal connectors ("thus", "hence", "whilst", "a priori", etc.) — T13 fully observed.
- No dates, images, or figures appear on these pages, so F9 and F10 do not apply.
- No exclamation marks, jokes, or idioms anywhere in the reviewed prose — most of T7 observed (one marketing-flavored phrase flagged).
- Cross-references consistently use proper `:doc:`/`:func:`/`:class:` targets with descriptive surrounding text rather than "see above/below" or "the docs" — F5 fully observed.
- Contractions ("you're", "you can") are used in a way consistent with Google's conversational-tone guidance and are not flagged.


---

## 5. Tutorial notebooks

Files reviewed: `notebooks/tutorials/01_getting_started.ipynb` (17 cells), `notebooks/tutorials/02_energy_injection.ipynb` (8 cells), `notebooks/tutorials/03_new_physics.ipynb` (9 cells), `notebooks/tutorials/04_custom_scenarios.ipynb` (15 cells), `notebooks/tutorials/05_observational_constraints.ipynb` (8 cells), `notebooks/tutorials/06_greens_table.ipynb` (13 cells).

### Summary

The dominant problem is undefined abbreviations: PDE, DC, BR, GF, DM, NWA, IC, CMB, and CL are used from each notebook's first cell onward and never spelled out in any of the six files, including three page titles ("PDE Solver"). Second, five of six H1 page titles use title case instead of sentence case; only `05_observational_constraints.ipynb` conforms. Third, the notebooks are inconsistent in dash usage — a triple hyphen (`---`) stands in for an em dash in two files, while the rest use a correctly-glyphed but incorrectly spaced em dash (`" — "` instead of `"—"`), and math-adjacent prose leans on bare arrows (`→`), tildes (`~`), and plus signs (`+`) as stand-ins for words ("and", "approximately", "et al."). A fifth pattern: several sequential procedures and enumerated concept lists use the wrong list type (numbered lists for non-sequential concept sets in `01`/`02`/`03`/`04`; a five-step procedure buried in one semicolon-chained sentence in `05`).

| Rule | Count | Severity |
|---|---|---|
| T12 Undefined abbreviations (PDE/DC/BR/GF/DM/NWA/IC/CMB/CL) | 17 first-use instances across 6 files | HIGH |
| H1 Title-case page titles | 5 | MEDIUM |
| F11 Dash errors (`---` for em dash; spaced `—`) | 2 + 15 | MEDIUM |
| T11 Symbols standing in for words (`→`, `~`, `+`, `Fixsen+`) | 13 | MEDIUM/LOW |
| L1 Wrong list type (numbered for non-sequential; prose for sequential) | 4 | MEDIUM/HIGH |
| T9 Word-list violations (`vs`, `e.g.`, `via`) | 8 | MEDIUM |
| T1 "we" / first person | 2 | MEDIUM |
| T15 British spelling (`normalisation`, `parallelised`, `normalises`) | 3 | LOW/MEDIUM |
| T14 Sentence fragments / dropped articles | 9 | LOW |
| H3 Code font in headings | 2 | MEDIUM |
| F6 Bold used for emphasis | 1 | LOW |
| L3 Non-parallel list construction | 1 | MEDIUM |
| F3 Unexplained single-letter placeholder | 1 | LOW |
| L5 Header row not sentence case / empty corner cell | 2 | LOW/MEDIUM |
| F8 Numeral under 10 for a count | 1 | LOW |
| T4 Pre-announcing opening sentence | 1 | LOW |

### Systematic patterns

**T12 — undefined abbreviations, HIGH.** No file ever pairs an abbreviation with its expansion (grepped for `(GF)`, `(PDE)`, `(DC)`, `(BR)`, `(DM)`, `(NWA)`, `(IC)`, `(CL)`, `(FIRAS)`, `(CMB)`, and full phrases "partial differential equation" / "double Compton" / "narrow-width approximation" / "confidence level" — none found paired with the abbreviation). First-use instance per file:
- `01_getting_started.ipynb`, cell 0, line 7 — "full Kompaneets + DC + BR evolution via the Rust binary" (PDE, DC, BR undefined; PDE first appears same cell, line 6).
- `02_energy_injection.ipynb`, cell 0, line 1 — "# Energy Injection Scenarios (PDE Solver)" (PDE undefined, in the title).
- `03_new_physics.ipynb`, cell 0, line 1 — "# New Physics: Dark Photons and Photon Injection (PDE Solver)" (PDE undefined, in the title); cell 2, line 3 — "the solver uses NWA to set the IC at $z_{\rm res}$" (NWA, IC undefined).
- `04_custom_scenarios.ipynb`, cell 0, line 5 — "Works in GF and PDE modes." (GF, PDE undefined).
- `05_observational_constraints.ipynb`, cell 0, line 1 — "# FIRAS limits on monochromatic photon injection and dark photon mixing" (FIRAS undefined, in the title); cell 0, line 10 — "| Limit | CL | Source |" (CL undefined).
- `06_greens_table.ipynb`, cell 0, line 3 — "The analytic GF has 30-70% shape errors..." (GF, PDE undefined; PDE appears later same line).

**F11 — dash errors, MEDIUM.** Two files use a literal triple hyphen where an em dash belongs:
- `01_getting_started.ipynb`, cell 0, lines 6–7 — "(`method="greens_function"`) --- pure-Python" / "(`method="pde"`) --- full Kompaneets".
- `02_energy_injection.ipynb`, cell 0, lines 5–7 — "**Decaying particles** --- exponential decay", "**DM annihilation (s-wave)** --- ...", "**DM annihilation (p-wave)** --- ...".

Separately, every genuine em dash in the six files is spaced (`" — "`), which Google style writes unspaced. 15 instances, grep `" — "` on `nb_0*.txt`, first 10:
`01_getting_started.ipynb:21` "...chemical potential. Compton..." (cell 2, line 5) — **$\mu$-distortion** ... — Bose-Einstein; `01:22` (cell 2, line 6); `01:23` (cell 2, line 7); `01:61` (cell 8, line 5); `01:108–112` (cell 16, lines 10–14, five instances); `02_energy_injection.ipynb:57` (cell 7, line 11); `03_new_physics.ipynb:6–7` (cell 0, lines 5–6); `04_custom_scenarios.ipynb:6–7` (cell 0, lines 5–6). Suggested fix: replace `" — "` with `"—"` throughout, and replace `"---"` with `"—"` in the two files above.

**T11 — symbols for words, MEDIUM/LOW.** Bare arrows in prose (outside `$...$`): `02_energy_injection.ipynb` cell 2, line 3 (three arrows) — "$z_X > 10^5$ → $\mu$-type, $z_X \sim 5\times 10^4$ → mixed, $z_X < 10^4$ → $y$-type"; `03_new_physics.ipynb` cell 4, line 5 — "rapidly absorbed by DC/BR → equivalent to heat injection"; `03_new_physics.ipynb` cell 6, line 3 (two arrows) — "$x_i < x_0$ → negative $\mu$ ... $x_i > x_0$ → positive $\mu$"; `04_custom_scenarios.ipynb` cell 7, line 3 — "Higher $n$ concentrates injection at higher $z$ → more $\mu$, less $y$." Tilde for "approximately" outside math: `01_getting_started.ipynb` cell 0, line 7 — "takes ~5-20 s per solve" and cell 6, line 5 — "takes ~5–20 s"; `04_custom_scenarios.ipynb` cell 5, line 3 — "PDE/GF agree to ~5%." Plus sign for "and" in prose: `01_getting_started.ipynb` cell 0, line 7 — "Kompaneets + DC + BR evolution"; `03_new_physics.ipynb` cell 0, line 9 — "DC/BR + Compton thermalization". "Fixsen+" for "et al.": `05_observational_constraints.ipynb` cell 0, lines 12 and 13 — "Fixsen+ 1996" (both rows).

**L1 — wrong list type, MEDIUM/HIGH.** Numbered lists used for non-sequential concept sets (should be bulleted): `02_energy_injection.ipynb` cell 0, lines 5–7 (three injection-history types, not steps); `03_new_physics.ipynb` cell 0, lines 5–6 (two unrelated scenarios); `04_custom_scenarios.ipynb` cell 0, lines 5–6 (two API hooks). Conversely, a genuine 5-step sequential procedure is embedded as parenthesized roman numerals inside one prose sentence instead of a numbered list: `05_observational_constraints.ipynb` cell 5, line 9 — "**Pipeline.** For each mass: (i) find $z_{\rm res}$ ...; (ii) compute ...; (iii) run the PDE ...; (iv) build ...; (v) translate ..." — this is also an F11 semicolon chain (four semicolons joining five clauses). Suggested fix: convert to a numbered list, one step per line.

**T9 — word list, MEDIUM.** `vs` for "versus" in headings/prose: `01_getting_started.ipynb` cell 10, line 1 — "## 5. PDE vs Green's function"; `02_energy_injection.ipynb` cell 4, line 1 — "## 2. DM annihilation: s-wave vs p-wave"; `05_observational_constraints.ipynb` cell 0, lines 6–7 (both Fig. captions use "vs"); `06_greens_table.ipynb` cell 6, line 1 — "## 3. Analytic vs table vs PDE in the transition" (two instances). "e.g." instead of "for example": `03_new_physics.ipynb` cell 0, line 6 — "(e.g., $X\to\gamma\gamma$)". "via" in prose (avoid per word list): `01_getting_started.ipynb` cell 0, line 7; cell 12, line 3; cell 14, line 3; `05_observational_constraints.ipynb` cell 5, line 9 (twice).

### Findings by file

#### `notebooks/tutorials/01_getting_started.ipynb`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| cell 0, line 1 | H1 | MEDIUM | "# Getting Started with spectroxide" | "# Getting started with spectroxide" |
| cell 0, line 6 | F11 | MEDIUM | "(\`method="greens_function"\`) --- pure-Python, vectorized" | Replace `---` with an unspaced em dash. |
| cell 0, line 7 | F11 | MEDIUM | "(\`method="pde"\`) --- full Kompaneets" | Replace `---` with an unspaced em dash. |
| cell 0, line 7 | T12 | HIGH | "full Kompaneets + DC + BR evolution via the Rust binary" | Spell out "double Compton (DC) and bremsstrahlung (BR)" at first use; also define PDE (introduced cell 0 line 6). |
| cell 0, line 7 | T11 | LOW | "full Kompaneets + DC + BR evolution" | "Kompaneets, DC, and BR evolution". |
| cell 0, line 7 | T11 | LOW | "takes ~5-20 s per solve" | "takes 5–20 s per solve". |
| cell 0, line 7 | T9 | MEDIUM | "evolution via the Rust binary" | "evolution, run by the Rust binary". |
| cell 0, line 8 | T14 | LOW | "Walks through distortion shapes, single-burst injection, parallel parameter sweeps, intensity conversion, and cosmology presets." | "This tutorial walks through..." or restore a subject. |
| cell 2, lines 5–7 | F11 | MEDIUM | "— Bose-Einstein with chemical potential", "— frequency redistribution...", "— degenerate with $T_{CMB}$..." | Unspace the em dashes. |
| cell 8, line 1 | H3 | MEDIUM | "## 4. Parallel sweeps with \`run_sweep\`" | "## 4. Parallel sweeps" (drop code font from heading; introduce `run_sweep` in body). |
| cell 8, line 3 | T2 | MEDIUM | "the entire $z_h$ list is shipped to the Rust binary, which fans out across CPU cores. By default all available cores are used" | "spectroxide ships the entire $z_h$ list to the Rust binary... By default, the sweep uses all available cores". |
| cell 8, line 3 | F3 | LOW | "pass \`n_threads=N\` to cap" | "pass \`n_threads=N\`, where N is the maximum thread count, to cap it". |
| cell 8, line 5 | F11 | MEDIUM | "one entry per injection redshift carrying \`pde_mu\`..." (preceding em dash) | Unspace the em dash. |
| cell 10, line 1 | T9 | MEDIUM | "## 5. PDE vs Green's function" | "## 5. PDE and Green's function compared". |
| cell 12, line 3 | T9 | MEDIUM | "in Jy/sr via the \`delta_I\` property on \`SolverResult\`" | "...through the \`delta_I\` property..." |
| cell 12, line 3 | T14 | LOW | "Useful for plotting against FIRAS/PIXIE bands." | "Use it to plot against FIRAS/PIXIE bands." |
| cell 12, line 3 | T12 | MEDIUM | "Useful for plotting against FIRAS/PIXIE bands." | Spell out FIRAS and PIXIE at first use. |
| cell 14, line 3 | T9 | MEDIUM | "Pass a \`Cosmology\` dataclass (or dict) via \`cosmo=...\`." | "...by passing \`cosmo=...\`." |
| cell 16, lines 10–14 | F11 | MEDIUM | "— built-in scenarios...", "— dark photon, monochromatic...", "— user-defined \`dq_dz\`...", "— FIRAS limits...", "— PDE-accurate interpolation..." | Unspace all five em dashes. |
| cell 16, line 9 | L2 | LOW | "Next:" | Use a complete sentence, e.g., "Continue with:". |

#### `notebooks/tutorials/02_energy_injection.ipynb`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| cell 0, line 1 | H1 | MEDIUM | "# Energy Injection Scenarios (PDE Solver)" | "# Energy injection scenarios (PDE solver)". |
| cell 0, line 1 | T12 | HIGH | "# Energy Injection Scenarios (PDE Solver)" | Spell out "partial differential equation (PDE)" at first use. |
| cell 0, lines 5–7 | L1 | MEDIUM | "1. **Decaying particles** --- exponential decay...", "2. **DM annihilation (s-wave)** ---...", "3. **DM annihilation (p-wave)** ---..." | Use a bulleted list — the three items are not sequential steps. |
| cell 0, lines 5–7 | F11 | MEDIUM | "--- exponential decay of a long-lived relic", "--- $\langle\sigma v\rangle$ constant", "--- $\langle\sigma v\rangle \propto v^2 \propto (1+z)$" | Replace `---` with an unspaced em dash. |
| cell 0, line 6 | T12 | HIGH | "**DM annihilation (s-wave)** --- $\langle\sigma v\rangle$ constant" | Spell out "dark matter (DM)" at first use. |
| cell 0, line 9 | T12 | HIGH | "the distortion at each $z$ feeds back into DC/BR and $T_e$" | Spell out DC and BR at first use in this document. |
| cell 4, line 1 | T9 | MEDIUM | "## 2. DM annihilation: s-wave vs p-wave" | "## 2. DM annihilation: s-wave and p-wave". |
| cell 4, line 3 | L5 | LOW | "| | s-wave | p-wave |" | Label the corner cell, e.g., "| Quantity | s-wave | p-wave |". |
| cell 7, line 5 | L5 | MEDIUM | "| Scenario | injection dict |" | "| Scenario | Injection dict |" (sentence-case header). |
| cell 7, line 11 | F11 | MEDIUM | "**Next:** \`03_new_physics.ipynb\` — dark photon oscillation, monochromatic photon injection." | Unspace the em dash. |

#### `notebooks/tutorials/03_new_physics.ipynb`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| cell 0, line 1 | H1 | MEDIUM | "# New Physics: Dark Photons and Photon Injection (PDE Solver)" | "# New physics: dark photons and photon injection (PDE solver)". |
| cell 0, line 1 | T12 | HIGH | "# New Physics: Dark Photons and Photon Injection (PDE Solver)" | Spell out PDE at first use. |
| cell 0, lines 5–6 | L1 | MEDIUM | "1. **Dark photon depletion** — ...", "2. **Monochromatic photon injection** — ..." | Use a bulleted list — these are two unrelated scenarios, not sequential steps. |
| cell 0, line 6 | T12 | MEDIUM | "1. **Dark photon depletion** — $\gamma\to A'$ resonant conversion removes CMB photons" | Spell out "cosmic microwave background (CMB)" at first use. |
| cell 0, line 7 | T9 | MEDIUM | "(e.g., $X\to\gamma\gamma$)" | "(for example, $X\to\gamma\gamma$)". |
| cell 0, line 9 | T11 | LOW | "the PDE handles the subsequent DC/BR + Compton thermalization" | "DC/BR and Compton thermalization". |
| cell 0, line 9 | T12 | HIGH | "the PDE handles the subsequent DC/BR + Compton thermalization self-consistently" | Spell out DC and BR at first use in this document. |
| cell 2, line 3 | T12 | HIGH | "the solver uses NWA to set the IC at $z_{\rm res}$" | Spell out "narrow-width approximation (NWA)" and "initial condition (IC)". |
| cell 4, line 5 | T11 | LOW | "rapidly absorbed by DC/BR → equivalent to heat injection" | "rapidly absorbed by DC/BR, equivalent to heat injection". |
| cell 6, line 3 | T11 | LOW | "$x_i < x_0$ → negative $\mu$ (extra cold photons); $x_i > x_0$ → positive $\mu$" | "$x_i < x_0$ gives negative $\mu$...; $x_i > x_0$ gives positive $\mu$...". |
| cell 8, line 3 | T9 | MEDIUM | "| Scenario | injection dict / call |" | "| Scenario | Injection dict or call |" (also sentence-case). |
| cell 8, line 7 | T12 | MEDIUM | "| Photon injection (GF) | \`greens_function_photon(x, x_inj, z_h)\`..." | Spell out "Green's function (GF)" at first use. |
| cell 8, line 9 | T12 | MEDIUM | "**Next:** \`04_custom_scenarios.ipynb\` (user-defined injection), \`05_observational_constraints.ipynb\` (FIRAS limits)." | Spell out FIRAS at first use in this document. |

#### `notebooks/tutorials/04_custom_scenarios.ipynb`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| cell 0, line 1 | H1 | MEDIUM | "# Custom Injection Scenarios" | "# Custom injection scenarios". |
| cell 0, line 3 | L2 | MEDIUM | "Two custom-API hooks for user-defined injection physics:" | "spectroxide exposes two custom-API hooks for user-defined injection physics:". |
| cell 0, lines 5–6 | L1 | MEDIUM | "1. \`dq_dz(z)\` — heating rate...", "2. \`photon_source(x, z)\` — frequency-resolved..." | Use a bulleted list — the two hooks are not sequential steps. |
| cell 0, line 5 | T12 | HIGH | "Works in GF and PDE modes." | Spell out GF and PDE at first use in this document. |
| cell 0, line 6 | T14 | LOW | "PDE only." | "Available in PDE mode only." |
| cell 5, line 3 | T14 | LOW | "Same Gaussian, full nonlinear PDE. Solver tabulates \`dq_dz\` on a redshift grid internally." | "The solver runs the same Gaussian burst through the full nonlinear PDE, tabulating \`dq_dz\` on a redshift grid internally." |
| cell 5, line 3 | T11 | LOW | "PDE/GF agree to ~5%." | "The PDE and GF results agree to 5%." |
| cell 3, line 8 | T15 | LOW (code comment) | "# normalisation check" | "# normalization check". |
| cell 7, line 3 | T11 | LOW | "Higher $n$ concentrates injection at higher $z$ → more $\mu$, less $y$." | "Higher $n$ concentrates injection at higher $z$, giving more $\mu$ and less $y$." |
| cell 9, line 3 | T14 | LOW | "Two Gaussians: 70% in $\mu$-era ($z=3\times10^5$), 30% in $y$-era ($z=5\times10^3$)." | "This example superposes two Gaussians: 70% in the $\mu$-era..., 30% in the $y$-era...". |
| cell 11, line 3 | T2/T14 | LOW | "Internally converted to the Boltzmann source $S = ...$." | "The solver converts this internally to the Boltzmann source $S = ...$." |
| cell 14, lines 9–12 | L3 | MEDIUM | "- \`dq_dz(z)\` returns...", "- \`photon_source(x, z)\` returns...", "- Set \`z_start = ...\`...", "- Always verify normalisation..." | Make all four bullets imperative instructions or all four declarative statements, not a mix. |
| cell 14, line 12 | T15 | MEDIUM | "Always verify normalisation with \`np.trapezoid\`" | "Always verify normalization with \`np.trapezoid\`". |
| cell 14, line 14 | T12 | MEDIUM | "**Next:** \`05_observational_constraints.ipynb\` (FIRAS / PIXIE)." | Spell out FIRAS and PIXIE at first use in this document. |
| cell 14, line 14 | T9 | LOW | "(FIRAS / PIXIE)" | "(FIRAS and PIXIE)". |

#### `notebooks/tutorials/05_observational_constraints.ipynb`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| cell 0, line 1 | T12 | HIGH | "# FIRAS limits on monochromatic photon injection and dark photon mixing" | Spell out "Far-Infrared Absolute Spectrophotometer (FIRAS)" at first use. |
| cell 0, line 3 | T4 | LOW | "This tutorial reproduces a few representative points from two FIRAS constraint figures in the paper:" | State the content directly, e.g., "Two FIRAS constraint figures from the paper each get a few representative points here:". |
| cell 0, line 9 | T1 | MEDIUM | "The paper figures sweep many points; here we run **3 points each** to demonstrate the workflow." | "The paper figures sweep many points; this tutorial runs 3 points each to demonstrate the workflow." |
| cell 0, line 9 | F6 | LOW | "here we run **3 points each** to demonstrate the workflow" | Remove the bold; it is not a UI element. |
| cell 0, line 9 | F8 | LOW | "here we run **3 points each**" | "here we run three points each". |
| cell 0, line 9 | T12 | MEDIUM | "run the PDE solver with the relevant injection scenario" | Spell out PDE at first use in this document. |
| cell 0, line 10 | T12 | HIGH | "| Limit | CL | Source |" | Spell out "confidence level (CL)" at first use. |
| cell 0, lines 12–13 | T11 | LOW | "Fixsen+ 1996" (both rows) | "Fixsen et al. 1996". |
| cell 2, line 3 | T12 | (secondary use, no new severity) | "Inverting the FIRAS bound at 68% CL," | — |
| cell 2, line 7 | T1 | MEDIUM | "We pick three illustrative points: a low-$x_i$ point at $z_h=2\times10^6$..." | "This example picks three illustrative points: ..." |
| cell 2, line 7 | F11 | MEDIUM | "(Chluba 2015) — energy- and number-injection cancel — so the limit is weakest there" | Unspace both em dashes. |
| cell 5, line 3 | T12 | (secondary use) | "depletes the CMB blackbody by" | — |
| cell 5, line 7 | T11 | LOW | "(narrow-width approximation, Mirizzi+2009; Chluba & Cyr 2024)" | "(narrow-width approximation, Mirizzi et al. 2009; Chluba and Cyr 2024)". |
| cell 5, line 9 | L1 | HIGH | "**Pipeline.** For each mass: (i) find $z_{\rm res}$...; (ii) compute...; (iii) run the PDE...; (iv) build...; (v) translate..." | Convert to a numbered list, one step per line. |
| cell 5, line 9 | F11 | MEDIUM | same quote as above | Semicolon chain of five clauses; break into list items (see L1 fix). |
| cell 5, line 9 | T9 | MEDIUM | "fit it to FIRAS via \`firas.profile_limit_floating_T\`; (v) translate the bound on $\gamma_{\rm con}$ back to $\epsilon$ via $\epsilon = ...$" | "fit it to FIRAS with \`firas.profile_limit_floating_T\`... translate the bound... using $\epsilon = ...$". |

#### `notebooks/tutorials/06_greens_table.ipynb`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| cell 0, line 1 | H1 | MEDIUM | "# Precomputed Green's Function Tables" | "# Precomputed Green's function tables". |
| cell 0, line 3 | T12 | HIGH | "The analytic GF has 30-70% shape errors in the $\mu\to y$ transition" | Spell out "Green's function (GF)" at first use; also spell out PDE (same line, later). |
| cell 0, line 3 | F8 | LOW | "has 30-70% shape errors" | "30–70%" (en dash, consistent with the range style used elsewhere in the series, e.g. "2–5%"). |
| cell 0, line 3 | T14 | LOW | "CosmoTherm-style fix: precompute $G_{\rm th}(x, z_h)$ from PDE runs, interpolate." | "The CosmoTherm-style fix precomputes $G_{\rm th}(x, z_h)$ from PDE runs and interpolates." |
| cell 2, line 3 | T15 | MEDIUM | "runs the PDE at each $z_h$ (parallelised in the Rust binary), normalises by $\Delta\rho/\rho$" | "parallelized"; "normalizes". |
| cell 2, line 3 | T14 | LOW | "Demo uses 30 redshifts; production tables use 150+." | "The demo uses 30 redshifts; production tables use 150 or more." |
| cell 8, line 1 | T12 | MEDIUM | "## 4. Convolve with a DM-decay heating rate" | Spell out "dark matter (DM)" at first use in this document. |
| cell 10, line 1 | H3 | MEDIUM | "## 5. Caching and \`solve(method="table")\`" | "## 5. Caching" (move the code example into the body). |

### What conforms

- `05_observational_constraints.ipynb`'s H1 title is correctly sentence-cased — the only one of the six.
- `06_greens_table.ipynb`'s task-oriented `##` headings ("Build a heating GF table", "Inspect spectra...", "Convolve with...") correctly use bare imperative verbs (H2); `01`–`04`'s conceptual headings are correctly noun phrases.
- No skipped heading levels and no stacked headings with empty content between them anywhere in the six files (H4).
- Code font is applied consistently to function names, parameters, dict keys, and file names throughout (F1).
- No exclamation marks, marketing language, jokes, or culture-specific idioms found in any file (T7).
- No `e.g./i.e./etc.` outside the single `03_new_physics.ipynb` instance found; Latin phrases ("a priori", "per se", "vice versa") and overly formal connectives ("thus", "hence", "whilst") do not appear (T13).
- List/table intro sentences generally end in a colon where a list or table follows (L2), apart from the one fragment noted in `04_custom_scenarios.ipynb`.


---

## 6. Python docstrings: `__init__`, `solver`, `greens`, `_validation`

Files reviewed: `python/spectroxide/__init__.py` (198 lines), `python/spectroxide/solver.py` (1662 lines), `python/spectroxide/greens.py` (1890 lines), `python/spectroxide/_validation.py` (625 lines).

### Summary

The docstrings are unusually well-formed for A4 (parameter/return/raise completeness): every public `validate_*`/`solve`/`run_*`/`greens.py` function has a full NumPy-style Parameters/Returns/Raises block, and a mechanical signature-vs-docstring parameter check found no missing or extra documented parameters among them. The dominant problems are instead (1) core abbreviations central to the package — PDE, CMB, DC, BR, CLI, GF — are never spelled out at first use anywhere in these four files, including the first line of the package docstring; (2) nine `warn_*` helpers in `_validation.py` have parameters but no Parameters section at all, breaking from the rigor of the neighboring `validate_*` functions in the same file; (3) a consistent word-list problem (`e.g.`, `i.e.`, `etc.`, `via`, `we`, `/` for "or", `(s)` plurals) scattered through prose docstrings and two raised-error/warning messages; and (4) systematic British spelling (`-ise`/`-isation`) against the required American spelling. Docstring summaries consistently use imperative mood ("Validate...", "Return...") rather than third-person present tense, which is flagged as A2 but is a standard, self-consistent PEP 257/NumPy convention and is scored DOMAIN throughout.

| Rule | Count | Severity |
|---|---|---|
| T12 undefined abbreviations (PDE, CMB) | 2 (representative; pervasive) | HIGH |
| T12 undefined abbreviations (DC, BR, CLI, GF) | 4 (representative; pervasive) | MEDIUM |
| A4 `warn_*` helpers missing Parameters section | 9 | HIGH |
| T15 British spelling (`-ise`/`-isation`/`behaviour`) | 11 | MEDIUM |
| A6 bare numeric/identifier literals not in code font | 4 | MEDIUM |
| T9 "e.g." in prose/messages | 3 | MEDIUM |
| T9 "etc." in prose | 3 | MEDIUM |
| T9 "i.e." in prose | 1 | MEDIUM |
| T9 "via" in prose | 7 | MEDIUM |
| T9 "/" used for "or" | 6 | LOW |
| T9 "(s)" plural in messages | 3 | LOW |
| T9 / F5 "see above" / "above" (document position) | 3 | MEDIUM |
| T1 "we" in docstring | 1 | MEDIUM |
| T4 pre-announcing "Note that" | 2 | MEDIUM |
| F11 em dash with surrounding spaces | 9 | LOW |
| A4 nested closures with no docstring (not true public API) | 7 | LOW |
| A2 imperative-mood first sentence (consistent, whole codebase) | ~60 reviewed items | DOMAIN |

### Systematic patterns

**T12 — abbreviations never spelled out.** PDE, CMB, DC, BR, CLI, and GF are used throughout all four files without a spelled-out definition at first use, including on the very first line of the package's own module docstring. Grep pattern: `grep -nE "\bPDE\b|\bCMB\b|\bDC\b|\bBR\b|\bCLI\b|\bGF\b"`.
- `__init__.py:2` — "spectroxide: Python API for CMB spectral distortion calculations." — first line of the package; CMB never defined. Fix: "...for cosmic microwave background (CMB) spectral distortion calculations."
- `__init__.py:8` — "``solver``: Wrapper that calls the Rust PDE solver binary." — PDE never defined. Fix: define as "partial differential equation (PDE)" at this first use.
- `solver.py:176` — "Translate a cosmology dict to Rust CLI flags." — CLI never defined in this file.
- `greens.py:268` — "``z_μ`` from equating the DC+BR photon production rate to the Hubble" — DC ("double Compton") and BR ("bremsstrahlung") never defined in this file.
- `_validation.py:369` — "Warn if integration extends beyond reliable GF regime." — GF ("Green's function") never spelled out in this file (the module does write "Green's function" elsewhere, but "GF" itself is used unexplained starting here).

**A4 — `warn_*` helpers in `_validation.py` lack a Parameters section.** Verified via AST: these functions have 1–3 parameters each but their docstring is a one- or few-line summary with no `Parameters`/`Raises`/`Warns` block, unlike every `validate_*` function in the same file. Grep pattern: `grep -n "^def warn_" python/spectroxide/_validation.py`.
- `_validation.py:323` — `def warn_z_h_regime(z_h):` — docstring: "Warn if z_h is outside the reliable Green's function regime." (no Parameters section)
- `_validation.py:351` — `def warn_x_inj_regime(x_inj):` — "Warn if injection frequency is in extreme regime."
- `_validation.py:368` — `def warn_z_max_regime(z_max):` — has explanatory prose but no Parameters section
- `_validation.py:424` — `def warn_x_grid_narrow(x_grid):` — "Warn if frequency grid is too narrow for decomposition."
- `_validation.py:440` — `def warn_analytic_gf_heating(z_min, z_max):` — has explanatory prose, no Parameters section, 2 undocumented params
- `_validation.py:460` — `def warn_table_z_density(z_injections):` — has explanatory prose, no Parameters section
- `_validation.py:505` — `def warn_convolution_resolution(n_z, z_min, z_max):` — has explanatory prose, no Parameters section, 3 undocumented params
- `_validation.py:576` — `def warn_grid_resolution_photon(n_points, injection_type):` — has explanatory prose, no Parameters section, 2 undocumented params
- `_validation.py:603` — `def warn_table_z_coverage(z_injections, z_min_query, z_max_query):` — has explanatory prose, no Parameters section, 3 undocumented params

Suggested fix: give each a NumPy `Parameters` block matching the style already used by every `validate_*` function directly above it in the same file.

**T15 — British spelling instead of American.** Grep pattern: `grep -nE "\b\w*(is(e|ed|es|ing)|isation)\b" *.py | grep -E "paralleli|linearis|normalis|orthogonalis|minimis|optimis|parameteris|behaviour"`. All 11 instances found (≤10 threshold — listing all since only marginally over):
- `__init__.py:23` — "injection); the Rust binary loops internally and parallelises across cores." → "parallelizes"
- `_validation.py:258` — "If ``|Δρ/ρ| > 0.01`` — the linearised Kompaneets equation is no" → "linearized"
- `solver.py:738` — "(parallelised via ``n_threads``)." → "parallelized"
- `solver.py:862` — "Calls the Rust ``photon-sweep`` subcommand, which parallelises across" → "parallelizes"
- `solver.py:997` — "Calls the Rust ``photon-sweep-batch`` subcommand, which parallelises" → "parallelizes"
- `greens.py:116` — `"""Mu-distortion normalisation :math:...` → "normalization"
- `greens.py:1221` — "Sign behaviour" (sub-heading) → "Sign behavior"
- `greens.py:1406` — "**``method="gs"``:** Linear Gram-Schmidt orthogonalisation of" → "orthogonalization"
- `greens.py:1414` — "J_μ and J_bb* by minimising the x³-weighted residual. Use this for" → "minimizing"
- `greens.py:1444` — "If the ``gf_fit`` L-BFGS-B optimisation fails to converge." → "optimization"
- `greens.py:1697` — "B&F parameterisation via ``δ_BF = δ_GS + μ/β_μ``).  The LM iteration" → "parameterization"

**T9 word list — "e.g." / "i.e." / "etc." / "via".** Grep patterns: `grep -n "e\.g\.\|i\.e\.\|etc\.\| via "`.
- `solver.py:1360` — "Remaining keys are scenario parameters, e.g.::" → "for example:"
- `solver.py:1444` — "If incompatible arguments are supplied (e.g. ``method="pde"``" → "for example ``method=\"pde\"``"
- `solver.py:1528` — `"e.g. solve(method='greens_function', z_h=2e5)."` (a `ValueError` message raised to the user) → `"for example, solve(method='greens_function', z_h=2e5)."`
- `_validation.py:146` — `"""Validate that a scalar is finite (i.e. neither NaN nor ±Inf).` → "(that is, neither NaN nor ±Inf)"
- `greens.py:50` — `("cannot broadcast", "only integer scalar arrays can be converted", etc.)` → spell out the remaining error substrings or say "and similar broadcasting errors"
- `solver.py:402` — "rho_e clamping, x_inj-out-of-grid, untested-regime soft warnings, etc.)." → list the remaining warning kinds or drop "etc."
- `_validation.py:200` — "``omega_m < omega_b``, ``y_p ∉ [0, 1)``, etc.)." → same
- `solver.py:397` — `"""Re-emit Rust solver diagnostic warnings via warnings.warn.` → "using `warnings.warn`"
- `solver.py:738` — "(parallelised via ``n_threads``)." → "parallelized using ``n_threads``"
- `solver.py:1148` — "via :func:`spectroxide.greens.distortion_from_heating` and ``μ``/``y``" → "using"
- `greens.py:1587` — "``(Y_SZ, M, G_bb)`` via Gram–Schmidt under the trapezoidal inner" → "using Gram–Schmidt"
- `greens.py:1697` — "B&F parameterisation via ``δ_BF = δ_GS + μ/β_μ``).  The LM iteration" → "using"
- `__init__.py:30` — "CosmoTherm comparison utilities are available via submodule import::" → "through a submodule import:"
- `__init__.py:34` — "Plot parameter constants are available via submodule import::" → "through a submodule import:"

**T9 — "/" standing in for "or".** These join distinct identifiers/types with a slash rather than spelling "or"; distinguish from the many legitimate physical-ratio slashes (`Δρ/ρ`, `ΔN/N`, `T_e/T_z`), which are DOMAIN notation and not flagged.
- `solver.py:399` — "The Rust ``SolverResult`` / ``SweepResult`` / ``PhotonSweepResult`` /" → "one of ``SolverResult``, ``SweepResult``, ``PhotonSweepResult``, or ``GreensResult``"
- `solver.py:1053` — "entries, or if ``sigma_x``/``dy_max`` are out of range." → "``sigma_x`` or ``dy_max``"
- `solver.py:1445` — "but neither ``injection`` nor ``dq_dz``/``photon_source``)." → "``dq_dz`` nor ``photon_source``"
- `_validation.py:535` — "extent from :mod:`grid`, point count from ``n_points``/``production_grid``)." → "``n_points`` or ``production_grid``"
- `_validation.py:555` — "If ``x`` is not *None*, or ``x_min``/``x_max``/``n_x`` differ from" → "``x_min``, ``x_max``, or ``n_x``"
- `greens.py:657` (comment, cited for pattern completeness only, not scored) — `` `from spectroxide.greens import hubble` / `cosmic_time` / `DEFAULT_COSMO` ``

**F11 — em dash with surrounding spaces** (Google style: no spaces around the em dash). All 9 docstring instances:
- `_validation.py:5` — "- **ERROR** (:class:`ValueError`) — nonsensical inputs that cannot"
- `_validation.py:7` — "- **WARNING** (:func:`warnings.warn`) — inputs in untested or unreliable"
- `_validation.py:258` — "If ``|Δρ/ρ| > 0.01`` — the linearised Kompaneets equation is no"
- `_validation.py:397` — "``5 × 10⁴ < z_h < 2 × 10⁵`` — residual r-type contributions become"
- `_validation.py:537` — "``x_max``, and ``n_x`` have zero effect on this path — passing anything"
- `greens.py:17` — "module) — or any user dict with the same keys."
- `greens.py:1105` — "the line is still drawn with a minimal width — see above)."
- `solver.py:875` — "Injection redshifts.  Default *None* — Rust uses 150 log-spaced"
- `solver.py:1139` — "**Single burst** (default) — provide ``z_h`` and ``delta_rho`` for a"

### Findings by file

#### `python/spectroxide/__init__.py`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 2 | T12 | HIGH | `spectroxide: Python API for CMB spectral distortion calculations.` | Spell out "cosmic microwave background (CMB)" at this first use. |
| 8 | T12 | HIGH | ` ``solver``: Wrapper that calls the Rust PDE solver binary.` | Spell out "partial differential equation (PDE)". |
| 23 | T15 | MEDIUM | `injection); the Rust binary loops internally and parallelises across cores.` | "parallelizes" |
| 30 | T9 | MEDIUM | `CosmoTherm comparison utilities are available via submodule import::` | "through a submodule import:" |
| 34 | T9 | MEDIUM | `Plot parameter constants are available via submodule import::` | "through a submodule import:" |

#### `python/spectroxide/solver.py`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 176 | T12 | MEDIUM | `"""Translate a cosmology dict to Rust CLI flags.` | Spell out "command-line interface (CLI)" at first use in this file. |
| 397 | T9 | MEDIUM | `"""Re-emit Rust solver diagnostic warnings via warnings.warn.` | "using ``warnings.warn``" |
| 399 | T9 | LOW | `The Rust ``SolverResult`` / ``SweepResult`` / ``PhotonSweepResult`` /` | "one of ``SolverResult``, ``SweepResult``, ``PhotonSweepResult``, or ``GreensResult``" |
| 402 | T9 | MEDIUM | `rho_e clamping, x_inj-out-of-grid, untested-regime soft warnings, etc.).` | Enumerate the remaining cases or drop "etc." |
| 738 | T9 | MEDIUM | `(parallelised via ``n_threads``).` | "parallelized using ``n_threads``" |
| 738 | T15 | MEDIUM | `(parallelised via ``n_threads``).` | "parallelized" |
| 862 | T15 | MEDIUM | `Calls the Rust ``photon-sweep`` subcommand, which parallelises across` | "parallelizes" |
| 997 | T15 | MEDIUM | `Calls the Rust ``photon-sweep-batch`` subcommand, which parallelises` | "parallelizes" |
| 1053 | T9 | LOW | `entries, or if ``sigma_x``/``dy_max`` are out of range.` | "``sigma_x`` or ``dy_max``" |
| 1148 | T9 | MEDIUM | `via :func:`spectroxide.greens.distortion_from_heating` and ``μ``/``y``` | "using" |
| 1360 | T9 | MEDIUM | `Remaining keys are scenario parameters, e.g.::` | "for example:" |
| 1364 | T4 | MEDIUM | `Note that ``delta_rho`` is a top-level argument, not an injection` | Drop "Note that" — state the fact directly. |
| 1370 | T9 / F5 | MEDIUM | ` ``axion`` feature, see above)::` | Reference the specific parameter by name instead of "see above". |
| 1444 | T9 | MEDIUM | `If incompatible arguments are supplied (e.g. ``method="pde"``` | "for example ``method=\"pde\"``" |
| 1445 | T9 | LOW | `but neither ``injection`` nor ``dq_dz``/``photon_source``).` | "``dq_dz`` nor ``photon_source``" |
| 1528 | T9 | MEDIUM | `"e.g. solve(method='greens_function', z_h=2e5)."` | "for example, ``solve(method='greens_function', z_h=2e5)``." (user-facing `ValueError` message) |

#### `python/spectroxide/greens.py`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 17 | F11 | LOW | `module) — or any user dict with the same keys.` | Remove spaces around the em dash: "module)—or any..." |
| 50 | T9 | MEDIUM | `("cannot broadcast", "only integer scalar arrays can be converted", etc.)` | Spell out the remaining error substrings. |
| 116 | T15 | MEDIUM | `"""Mu-distortion normalisation :math:` | "normalization" |
| 268 | T12 | MEDIUM | ` ``z_μ`` from equating the DC+BR photon production rate to the Hubble` | Spell out "double Compton (DC)" and "bremsstrahlung (BR)" at first use. |
| 355 | T4 | MEDIUM | `Note that ``J_y ≠ 1 − J_μ`` in the transition region; using the` | Drop "Note that". |
| 393 | T9 | MEDIUM | `Accuracy (vs. PDE)` | "Accuracy versus the PDE solver" |
| 1105 | F11 | LOW | `the line is still drawn with a minimal width — see above).` | Remove spaces around the dash; name the specific parameter instead of "see above". |
| 1105 | T9 / F5 | MEDIUM | `the line is still drawn with a minimal width — see above).` | Reference the parameter directly rather than "see above". |
| 1221 | T15 | MEDIUM | `Sign behaviour` | "Sign behavior" |
| 1406 | T15 | MEDIUM | `**``method="gs"``:** Linear Gram-Schmidt orthogonalisation of` | "orthogonalization" |
| 1414 | T15 | MEDIUM | `J_μ and J_bb* by minimising the x³-weighted residual. Use this for` | "minimizing" |
| 1444 | T15 | MEDIUM | `If the ``gf_fit`` L-BFGS-B optimisation fails to converge.` | "optimization" |
| 1587 | T9 | MEDIUM | ` ``(Y_SZ, M, G_bb)`` via Gram–Schmidt under the trapezoidal inner` | "using Gram–Schmidt" |
| 1628 | T9 | LOW | `f"_decompose_gram_schmidt: only {len(xb)} grid point(s) fall in the "` | "grid points" (or "at least one grid point") — `RuntimeError` message |
| 1697 | T15 | MEDIUM | `B&F parameterisation via ``δ_BF = δ_GS + μ/β_μ``).  The LM iteration` | "parameterization using" |
| 1742 | T9 | LOW | `f"_decompose_nonlinear_be: only {len(xb)} grid point(s) fall in the "` | "grid points" — `RuntimeError` message |

#### `python/spectroxide/_validation.py`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 5 | F11 | LOW | `- **ERROR** (:class:`ValueError`) — nonsensical inputs that cannot` | Remove spaces around the em dash. |
| 7 | F11 | LOW | `- **WARNING** (:func:`warnings.warn`) — inputs in untested or unreliable` | Remove spaces around the em dash. |
| 108 | T9 / F5 | MEDIUM | `If any of the conditions above are violated.` | "If any of the preceding conditions are violated." |
| 146 | T9 | MEDIUM | `"""Validate that a scalar is finite (i.e. neither NaN nor ±Inf).` | "(that is, neither NaN nor ±Inf)" |
| 200 | T9 | MEDIUM | ` ``omega_m < omega_b``, ``y_p ∉ [0, 1)``, etc.).` | Enumerate the remaining conditions or drop "etc." |
| 258 | T15 | MEDIUM | `If ``|Δρ/ρ| > 0.01`` — the linearised Kompaneets equation is no` | "linearized" |
| 258 | F11 | LOW | `If ``|Δρ/ρ| > 0.01`` — the linearised Kompaneets equation is no` | Remove spaces around the em dash. |
| 323 | A4 | HIGH | `"""Warn if z_h is outside the reliable Green's function regime."""` | Add a Parameters section documenting `z_h`. |
| 351 | A4 | HIGH | `"""Warn if injection frequency is in extreme regime."""` | Add a Parameters section documenting `x_inj`. |
| 368 | A4 | HIGH | `"""Warn if integration extends beyond reliable GF regime.` | Add a Parameters section documenting `z_max`. |
| 369 | T12 | MEDIUM | `"""Warn if integration extends beyond reliable GF regime.` | Spell out "Green's function (GF)" at first use in this file. |
| 372 | A6 | MEDIUM | `invalid for theta_e > 0.005), so the GF should warn well before that —` | Wrap the literal in code font: `` `theta_e > 0.005` ``, matching the backtick treatment given to `` `z_start > 1e7` `` one line above. |
| 373 | T1 | MEDIUM | `we mirror the PDE soft-warning threshold (``5e6``) and escalate at ``1e7``.` | "The threshold mirrors the PDE solver's soft-warning value..." |
| 397 | F11 | LOW | ` ``5 × 10⁴ < z_h < 2 × 10⁵`` — residual r-type contributions become` | Remove spaces around the em dash. |
| 424 | A4 | HIGH | `"""Warn if frequency grid is too narrow for decomposition."""` | Add a Parameters section documenting `x_grid`. |
| 440 | A4 | HIGH | `"""Warn when analytic GF covers the mu-y transition region.` | Add a Parameters section documenting `z_min`, `z_max`. |
| 444 | A6 | MEDIUM | `transition era (3e4 < z < 2e5) because it decomposes into pure mu +` | Wrap in code font: `` `3e4 < z < 2e5` ``, and use `` `μ` ``/`` `y` `` consistent with the rest of the module. |
| 460 | A4 | HIGH | `"""Warn if z-injection grid is too sparse for accurate interpolation.` | Add a Parameters section documenting `z_injections`. |
| 464 | A6 | MEDIUM | `especially in the transition region (3e4-2e5) where the Green's` | Wrap in code font: `` `3e4–2e5` ``. |
| 470 | T9 | LOW | `f"Only {z.size} z-injection point(s). Need >= 50 for "` | "points" — `UserWarning` message. |
| 505 | A4 | HIGH | `"""Warn if the convolution integral has too few redshift points.` | Add a Parameters section documenting `n_z`, `z_min`, `z_max`. |
| 530 | T12 | LOW | `"""Reject frequency-grid kwargs that the PDE injection path silently ignores.` | (Same PDE-definition issue; grouped under the `__init__.py:8` fix.) |
| 576 | A4 | HIGH | `"""Warn if PDE grid is too coarse for photon injection scenarios.` | Add a Parameters section documenting `n_points`, `injection_type`. |
| 581 | A6 | MEDIUM | `gives ~10% errors at injection peaks.` | Wrap "DEBUG" in code font two lines above (`` `DEBUG` `` preset) for consistency with `:data:`DEBUG`` used elsewhere. |
| 603 | A4 | HIGH | `"""Warn if convolution bounds extend beyond the table's z-injection range.` | Add a Parameters section documenting `z_injections`, `z_min_query`, `z_max_query`. |

### What conforms

- A4 (parameter/return/raise completeness) is strong for every `validate_*` function in `_validation.py` and essentially all public functions in `solver.py` and `greens.py`: a mechanical AST-based signature-vs-docstring check found the `Parameters` block always lists exactly the function's real parameters, with defaults and units stated (for example, `t_cmb : float, optional ... in **K**. Default 2.726.` at `greens.py:1870`).
- A5: defaults and valid ranges are consistently stated ("Default ``1e-5``", "Must lie in ``(0, 0.1]``", "must satisfy ``0 ≤ z_min < z_max``").
- A6: the large majority of docstrings wrap identifiers, code literals, and inline math in double backticks or `:math:`/`:func:`/`:class:`/`:data:` roles — the exceptions are the small `warn_*`-helper cluster flagged above.
- Boolean parameters are documented with clear "If *True*, ..." phrasing in several places (for example `debug : bool, optional` in `solve`, `solver.py:1424`).
- No contractions, exclamation marks, or Latin phrases (a priori, ad hoc, vice versa, per se) found in any docstring or message text.
- No curly quotes; straight quotes used consistently throughout.
- The one Sphinx `.. note::` directive found (`solver.py:1296`, on `SolverResult.delta_I`) is used for a genuine non-obvious caveat (an allocating property), not overused, and not stacked with other notices.
- Cross-references use `:func:`, `:class:`, `:data:`, `:mod:` roles with the identifier as link text — never "here" or a bare URL.


---

## 7. Python docstrings: remaining modules

Files reviewed: `python/spectroxide/cosmology.py` (690 lines), `python/spectroxide/cosmotherm.py` (769 lines), `python/spectroxide/dark_photon.py` (203 lines), `python/spectroxide/axion.py` (132 lines), `python/spectroxide/firas.py` (1070 lines), `python/spectroxide/greens_table.py` (1185 lines), `python/spectroxide/plot_params.py` (123 lines), `python/spectroxide/style.py` (68 lines).

### Summary

These eight files are in strong shape for A4 (parameter documentation): every public function's signature was cross-checked against its NumPy-style `Parameters` block by an AST dump, and no mismatches or wholly undocumented public functions turned up. The two real defects are structural, not cosmetic: two public functions raise an exception with no `Raises` section (A4), and the abbreviation "PDE" — the module's central subject — is never spelled out in any of the six files that use it (T12). Below that, the findings are the usual Google-style prose gaps that show up in a scientific-register codebase: "e.g."/"i.e."/"via" in prose, three "we" sentences in one docstring, spaced em dashes instead of Google's unspaced style, a "see ... below" cross-reference, and a scattering of passive-voice sentences. None of these block use of the API; they are polish and consistency items.

| Rule | Count | Severity |
|---|---|---|
| A4 Missing `Raises` section for a documented public function | 2 | HIGH |
| T12 "PDE" never spelled out (6 files) | 22 uses | HIGH |
| T12 "CMB" never spelled out (5 files) | 13 uses | MEDIUM |
| T12 "FIRAS" never spelled out | 1 (module docstring, first use) | MEDIUM |
| T12 "DM" never spelled out (cosmotherm.py) | 3 | MEDIUM |
| T9 "e.g." / "i.e." in docstring prose | 3 | MEDIUM |
| T9 "via" in docstring prose | 2 | MEDIUM/LOW |
| F6 ALL CAPS used for emphasis ("WITHOUT") | 2 | MEDIUM |
| T1 "we"/"We" in API docstring | 3 (one docstring) | MEDIUM |
| F5 "See the warning below" | 1 | MEDIUM |
| F11 Em dash with surrounding spaces (Google style wants none) | 17 | MEDIUM |
| F11 Semicolon-chained independent clauses | 2 | MEDIUM |
| T2 Passive voice ("is/are + past participle") | ~20 | LOW/MEDIUM |
| T6 "currently" | 2 | LOW |
| T9 "should" for what reads as a requirement | 2 | LOW |
| T5 "simply" as an ease claim | 1 | LOW |
| T11 ASCII arrow `->` in a type-alias doc comment | 1 | LOW/DOMAIN |
| T12 "CT" used as an undefined shorthand for CosmoTherm | 1 | LOW |
| T10 "native" as a feature descriptor | 1 | LOW |

### Systematic patterns

**T12 — "PDE" (partial differential equation) never spelled out.** The term is the load-bearing subject of the whole package ("the PDE solver") but is never expanded in any docstring. Grep: `grep -n '\bPDE\b' *.py`. Count: 22 uses across `dark_photon.py` (2), `axion.py` (2), `cosmotherm.py` (2), `greens_table.py` (14), `firas.py` (1), `cosmology.py` (1). First 10 instances:
- `dark_photon.py:3` — "The PDE solver handles dark-photon oscillations through the initial-condition" — spell out "partial differential equation (PDE)" at first use in the module docstring.
- `dark_photon.py:9` — "diagnostics that need the conversion probability without running the PDE."
- `axion.py:6` — "off-by-default \`\`axion\`\` Cargo feature, so the PDE path below only works if"
- `axion.py:11` — "Mirrors :mod:\`spectroxide.dark_photon\`. The PDE solver handles axion"
- `cosmotherm.py:11` — "   should use :mod:\`spectroxide.solver\` (PDE),"
- `cosmotherm.py:13` — "   :mod:\`spectroxide.greens_table\` (precomputed PDE Green's function)"
- `greens_table.py:10` — "Tables are built by running the PDE solver at many injection redshifts,"
- `greens_table.py:120` — "(frequency, injection redshift) grid point. Built from PDE solver"
- `greens_table.py:178` — "        \"\"\"Evaluate the tabulated PDE Green's function \`\`G_th(x, z_h)\`\`."
- `greens_table.py:186` — "        only on the cached PDE table."
Suggested fix: expand once, e.g. in `greens_table.py`'s module docstring ("...running the PDE (partial differential equation) solver..."), and rely on that definition for the package; or add the expansion to each module's docstring independently, matching the rubric's per-document first-use rule.

**T12 — "CMB", "FIRAS", "DM" also never spelled out.** Grep: `grep -n '\bCMB\b'`, `'\bFIRAS\b'`, `'\bDM\b'` per file.
- CMB (13 uses): `cosmology.py:618` "Photon energy density \`\`ρ_γ(z) = ... T(z))⁴...\`\`" region, `cosmotherm.py:283` region, `axion.py:98`, `firas.py:1` "FIRAS spectral-distortion constraints...", `greens_table.py:2` "Precomputed Green's-function tables for CMB spectral distortions." First use is `greens_table.py:2` — none of the 5 files expands "cosmic microwave background."
- FIRAS (module-defining term, `firas.py:1`): "\`\`\`FIRAS spectral-distortion constraints with full covariance matrix." — never expanded to "Far Infrared Absolute Spectrophotometer" anywhere in the file.
- DM (`cosmotherm.py:574,615,703`): "Heating rate \`\`d(Δρ/ρ)/dz\`\` for s-wave DM annihilation." / "...for p-wave DM annihilation." / "Convenience wrapper: load GF database + convolve for a DM scenario." — "dark matter" is never spelled out in this file.
Suggested fix: expand each at first use per file, e.g. "dark matter (DM)" in `cosmotherm.py:574`.

**F11 — em dash written with surrounding spaces.** Google style specifies an em dash with no surrounding spaces; these docstrings consistently use " — " (paper/scientific convention). Grep: `grep -n ' — ' *.py` filtered to docstring text (code comments excluded). Count: 17. First 10 instances:
- `cosmology.py:157` — "files (Fixsen 1996 = 2.726 K), not the Planck paper value 2.7255 K — use" (Sphinx `#:` attribute doc, rendered by autodoc)
- `cosmotherm.py:18` — "- **DI files** — predicted ΛCDM spectral distortions (ASCII, two columns)."
- `cosmotherm.py:19` — "- **Green's function database** — precomputed exact GF (large"
- `cosmotherm.py:166` — "Green's function G_th(x, z_h) — the residual mu+y spectral"
- `cosmotherm.py:171` — "- \`\`tgin\`\`: ndarray, shape (N_z,) — initial blackbody temperature [K]"
- `cosmotherm.py:172` — "- \`\`tglast\`\`: ndarray, shape (N_z,) — last blackbody temperature [K]"
- `cosmotherm.py:174` — "- \`\`rho\`\`: ndarray, shape (N_z,) — Delta-rho/rho used for each entry"
- `axion.py:23` — "convert preferentially — opposite to the dark photon's \`\`1/x\`\`."
- `firas.py:29` — "Two distinct — and mutually inconsistent — limit conventions coexist in"
- `firas.py:445` — "\`\`[G_bb]\`\` — the temperature shift is always unobservable."
(also `greens_table.py:6,7,185,700,918,921` and `style.py:43`). Suggested fix: remove the surrounding spaces ("2.7255 K—use") if strict Google style is required; flag as a house-style decision otherwise, since it is applied consistently.

**T2 — passive voice.** The register favors passive constructions ("is stored," "are listed," "is tracked") where an active rewrite naming the actor (the function, the file, the code) is available. Grep: `grep -n -E '\b(is|are|was|were|be|been|being) [a-z]+ed\b' *.py`. Count: ~20 (excluding 2 matches that are code comments, not docstrings). First 10 instances:
- `cosmology.py:87` — "Density fractions are derived from the paper's physical densities" → "The paper's physical densities derive the density fractions" / "This classmethod derives the density fractions from..."
- `cosmology.py:106` — "Density fractions are derived from the paper's physical densities" (duplicate wording in a sibling classmethod)
- `cosmotherm.py:136` — "The injection redshifts z_h are listed in the header." → "The header lists the injection redshifts z_h."
- `cosmotherm.py:142` — "Greens.cpp). The temperature shift is tracked separately via" → "...the Tgin/Tglast metadata tracks the temperature shift separately"
- `cosmotherm.py:416` — "is applied analytically (matching CosmoTherm's Greens.cpp)."
- `style.py:13` — "All plot parameters are defined in \`\`plot_params.py\`\`." → "\`\`plot_params.py\`\` defines all plot parameters."
- `firas.py:14` — "The per-pixel covariance is converted to a monopole covariance using the"
- `firas.py:408` — "\`\`|A| > A_limit\`\` is excluded."
- `firas.py:596` — "- If \`\`delta_n\`\` is provided, it is converted to kJy/sr and"
- `firas.py:598` — "- If \`\`model_kJy\`\` is provided, it is also subtracted."
This is a scientific-prose convention as much as a defect; report at LOW–MEDIUM depending on how strictly Google style is enforced against the codebase's established register.

### Findings by file

#### `python/spectroxide/cosmology.py`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 664 | T9 | MEDIUM | "early-Universe applications (e.g., neutrino decoupling)." | "for example, neutrino decoupling" |
| 158 | T12 | LOW | "this for CT comparisons and \`\`Cosmology.planck2015()\`\` otherwise." | Write "CosmoTherm (CT) comparisons" at first use of the shorthand, two lines above "CosmoTherm" is spelled out but "CT" itself is never introduced as its abbreviation |
| 87, 106 | T2 | LOW | "Density fractions are derived from the paper's physical densities" | "This classmethod derives the density fractions from the paper's physical densities" |

#### `python/spectroxide/cosmotherm.py`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 131–175 | A4 | HIGH | `def load_greens_database(...)` raises `FileNotFoundError` at line 185 with no `Raises` section in its docstring (132–175) | Add a `Raises`/`FileNotFoundError` block documenting when the database file cannot be found |
| 690–734 | A4 | HIGH | `def cosmotherm_gf_distortion(...)` raises `ValueError` at line 749 ("Unknown scenario: {scenario!r}...") with no `Raises` section | Add `Raises`/`ValueError` documenting the accepted `scenario` values |
| 703 | T11 | MEDIUM | "\"\"\"Convenience wrapper: load GF database + convolve for a DM scenario." | "Load the GF database and convolve it for a DM scenario." |
| 61 | T9 | MEDIUM | "Filename (e.g. \`\`\"DI_damping.dat\"\`\`) looked up in the bundled" | "for example, \`\`\"DI_damping.dat\"\`\`" |
| 142 | T9 | MEDIUM | "Greens.cpp). The temperature shift is tracked separately via" | "...tracked separately using the Tgin/Tglast metadata" |
| 140–141, 167 | F6 | MEDIUM | "The database stores Green's function entries WITHOUT the G_bb" / "distortion, WITHOUT the G_bb temperature shift." | Use italics or rephrase: "does not store the G_bb temperature shift component" |
| 574, 615, 703 | T12 | MEDIUM | "\`\`\`Heating rate \`\`d(Δρ/ρ)/dz\`\` for s-wave DM annihilation." | Spell out "dark matter (DM)" at first use (line 574) |
| 18, 19, 166, 171, 172, 174 | F11 | MEDIUM | e.g. "- **DI files** — predicted ΛCDM spectral distortions (ASCII, two columns)." | Close up the spaces around the em dash for strict Google style |
| 136, 142, 416 | T2 | LOW | "The injection redshifts z_h are listed in the header." | "The header lists the injection redshifts z_h." |

#### `python/spectroxide/dark_photon.py`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 3, 9 | T12 | HIGH | "The PDE solver handles dark-photon oscillations through the initial-condition" | Spell out "partial differential equation (PDE)" at first use |

#### `python/spectroxide/axion.py`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 6, 11 | T12 | HIGH | "Mirrors :mod:\`spectroxide.dark_photon\`. The PDE solver handles axion" | Spell out "PDE" at first use in this module too (each file needs its own first-use definition per T12) |
| 23 | F11 | MEDIUM | "convert preferentially — opposite to the dark photon's \`\`1/x\`\`." | Close up the em-dash spacing for strict Google style |

#### `python/spectroxide/firas.py`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 1 | T12 | MEDIUM | "\"\"\"FIRAS spectral-distortion constraints with full covariance matrix." | Spell out "Far Infrared Absolute Spectrophotometer (FIRAS)" at first use |
| 436 | T9 | MEDIUM | "returns the marginalised constraint on \`\`A\`\` (i.e. \`\`A\`\` after" | "that is, \`\`A\`\` after" |
| 493 | F5 | MEDIUM | "literature limit. See the warning below." | Name the target directly, e.g. "See the Warning in this docstring" or restructure so the warning precedes the reference |
| 865, 872, 875 | T1 | MEDIUM | "shapes, so when \`\`marginalise_galactic=True\`\` (default) we" / "\`\`x(T) = hν/(k_B T)\`\` depend nonlinearly on \`\`T\`\`.  We therefore" / "model is linear in both), and we take the \`\`T\`\` that minimises" | Rewrite in passive or third person about the function: "...the function marginalises over a fixed-shape template..." / "...the fit therefore profiles over \`\`T\`\`..." |
| 29, 445 | F11 | MEDIUM | "Two distinct — and mutually inconsistent — limit conventions coexist in" / "\`\`[G_bb]\`\` — the temperature shift is always unobservable." | Close up em-dash spacing |
| 14, 408, 596, 598, 762, 874 | T2 | LOW | "The per-pixel covariance is converted to a monopole covariance using the" | "We convert the per-pixel covariance..." (or name the actor) |

#### `python/spectroxide/greens_table.py`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 10, 120, 178, 186, 686, 688, 705, 713, 907, 908, 926, 934 (+2 more) | T12 | HIGH | "Tables are built by running the PDE solver at many injection redshifts," | Spell out "PDE" at first use (module docstring, line 10) |
| 2 | T12 | MEDIUM | "Precomputed Green's-function tables for CMB spectral distortions." | Spell out "cosmic microwave background (CMB)" at first use |
| 55, 439 | T6 | LOW | "physics-code version than the currently-installed Rust binary." / "currently installed binary." | Drop "currently" — "the installed Rust binary" |
| 159, 237 | T9 | LOW | "should apply :func:\`~spectroxide.cosmotherm.strip_gbb\` themselves." / "number-conserving Δn should call :func:\`~spectroxide.cosmotherm.strip_gbb\`." | If this is a requirement for a correct number-conserving result, use "must"; if genuinely optional, "should" is acceptable — confirm intent |
| 182–183 | F11 | MEDIUM | "Both axes are clipped to the stored range; queries outside the" | Split into two sentences: "Both axes are clipped to the stored range. Queries outside the range are pinned to the nearest edge (no extrapolation)." |
| 236 | F11 | MEDIUM | "\`\`x_grid\`\`.  No NC strip is applied; callers that want the" | Split into two sentences |
| 185 | F11 | MEDIUM | "No analytic Green's function is consulted — the result depends" | Close up em-dash spacing |
| 49 | T11 | LOW/DOMAIN | "#: Type alias for a heating-rate callable \`\`z -> dQ/dz\`\`." | Use \`\`z → dQ/dz\`\` (Unicode arrow, consistent with the rest of the package) or spell "as a function of" |
| 235 | T10 | LOW | "cache's native \`\`self.x\`\` grid, then linearly interpolated to" | "the cache's own \`\`self.x\`\` grid" |
| 700, 918, 921 | F11 | MEDIUM | "Injection redshifts.  Default *None* — uses 150 log-spaced" | Close up em-dash spacing |

#### `python/spectroxide/plot_params.py`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 11 | T5 | LOW | "Or simply::" | "Alternatively:" |

#### `python/spectroxide/style.py`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 11 | T9 | MEDIUM | "\`\`\$\\Te\$\`\`, etc." | List the remaining commands explicitly, or drop "etc." — for example, "...and similar commands defined in the LaTeX preamble." |
| 43 | F11 | MEDIUM | "without it — \`\`spectroxide/__init__.py\`\`\` re-exports this module" | Close up em-dash spacing |
| 56 | T9 | LOW | "\"Install via conda:\\n\"" | "Install with conda:" (this is a user-facing `RuntimeError` message) |
| 13 | T2 | LOW | "All plot parameters are defined in \`\`plot_params.py\`\`." | "\`\`plot_params.py\`\` defines all plot parameters." |

### What conforms

- **A4 (parameter documentation):** every public function's signature (checked by AST against its NumPy-style `Parameters` block) matches the documented parameters — no missing or extra entries found across ~35 public functions/methods sampled.
- **A4 (docstring presence):** no public function or class lacks a docstring; the only undocumented items are dunder methods (`__init__`, `__post_init__`, `__repr__`), a nested closure, and private helpers, none of which are part of the rendered public API.
- **A5 (defaults/ranges):** defaults are consistently stated ("default \`\`0.95\`\`", "Default \`\`False\`\`"), and valid ranges are given where relevant (e.g., `cl` "must lie in (0, 1)").
- **A6 (code font):** identifiers, literals, and cross-references consistently use single or double backticks (`\`\`x\`\``, `:func:`, `:class:`, `:data:`).
- **A4 (boolean parameters):** boolean parameters consistently follow the "If True (default), ..." pattern the rubric asks for.
- **F8/F9 (numbers, units, dates):** no glued number+unit pairs or ambiguous dates found; units are consistently bracketed in bold within `Returns`/`Parameters` blocks.
- **T7/T10 (tone, inclusive language):** no exclamation marks, marketing language, idioms, or gendered pronouns found; only one borderline "native" usage (see findings).
- **T3 (tense):** no future-tense "will" describing product behavior.


---

## 8. Rust doc comments: physics and infrastructure modules

Files reviewed: `src/kompaneets.rs` (1718 lines, doc-comment scope ends at line 1066, the `mod tests` boundary), `src/double_compton.rs` (538 lines, scope ends 138), `src/bremsstrahlung.rs` (923 lines, scope ends 457), `src/electron_temp.rs` (156 lines, scope ends 55), `src/recombination.rs` (737 lines, scope ends 475), `src/constants.rs` (300 lines, scope ends 161), `src/cosmology.rs` (723 lines, scope ends 455), `src/spectrum.rs` (394 lines, scope ends 173), `src/grid.rs` (472 lines, scope ends 354), `src/dark_photon.rs` (178 lines, scope ends 94), `src/axion.rs` (136 lines, scope ends 89). Individual `#[cfg(test)]`-gated `pub fn` helpers outside the `mod tests` block (`kompaneets_rhs`, `kompaneets_tridiagonal`, `kompaneets_step`, `dc_rhs`, `br_rhs`) were excluded: they never render in a normal `cargo doc` build.

### Summary

The dominant problem is a house style split from Google's API-reference conventions on two axes that repeat in nearly every file: function summaries almost never open with a present-tense third-person verb (imperative verbs like "Solve", "Build", "Validate" or bare noun phrases like "H₀ in 1/s" instead of "Returns H₀ in 1/s"), and terse single-line docs for constants and simple accessors routinely drop the terminal period. A third pervasive, purely mechanical pattern is em dashes written with surrounding spaces (`word — word`) instead of Google's tight `word—word`. Against that backdrop, three findings are substantive rather than stylistic: `bremsstrahlung::br_emission_coefficient` documents 7 of its 8 parameters and silently omits `cosmo`; `BrPrecomputed` has three undocumented public fields (`n_hii`, `n_heiii`, `n_heii`) sandwiched between documented ones; and `kompaneets_step_coupled_inplace`, the solver's core coupled step function, documents 2 of its 10 parameters. The abbreviation "NWA" is used as the lead noun of a function summary in both `dark_photon.rs` and `axion.rs` without ever being spelled out in either file, and `bremsstrahlung.rs` uses "BR" as its central term throughout without ever writing "Bremsstrahlung (BR)" the way `double_compton.rs` writes "Double Compton (DC)".

| Rule | Count | Severity |
|---|---|---|
| A2 Function summary not present-tense 3rd person verb | 82 | MEDIUM |
| A1 Single-line doc missing terminal period | 38 | MEDIUM |
| F11 Em dash with surrounding spaces | 33 | MEDIUM |
| A4 Undocumented/under-documented parameters or fields | 6 findings (covering 1 param, 3 fields, 8 params) | HIGH |
| T12 Undefined abbreviation ("NWA", "BR") | 3 | HIGH |
| T1 "we"/"We" in prose | 5 | MEDIUM |
| T9 "via" in prose | 4 | LOW |
| T9 "e.g." / "i.e." | 3 | MEDIUM |
| T15 British spelling ("honoured", "normalisations") | 3 | MEDIUM |
| T11 Arrow (→) substituting a word in prose | 4 | LOW–MEDIUM |
| A4 Missing/inconsistent unit or "(dimensionless)" tag | 4 | LOW–MEDIUM |
| T9 "since" meaning "because" | 1 | MEDIUM |
| T9 "vs" in prose | 1 | MEDIUM |
| T9 "above/below" as document position | 1 | LOW |
| T13 Latin phrase "a priori" | 1 | LOW |
| T12 "DC/BR" used without local definition (downstream file) | 1 | MEDIUM |

### Systematic patterns

**A2 — function summaries are imperative or noun-phrase, not present-tense third person (82 instances).** Google (and Rust's own API guidelines) want "Computes the …", "Returns …"; this codebase overwhelmingly uses imperative verbs ("Solve", "Build", "Create", "Validate", "Construct", "Find", "Precompute", "Set") or bare noun phrases with no verb at all ("H₀ in 1/s", "BR emission coefficient K_BR at frequency x"). Zero of the ~84 `pub fn` summaries surveyed use the "Computes"/"Returns" form. Grep used: manual AST-style walk (Python script over `pub fn` + preceding doc block, excluding `#[cfg(test)]`). First 10:
- `src/kompaneets.rs:178` — "Solve a tridiagonal system Ax = d using the Thomas algorithm." — rewrite: "Solves a tridiagonal system Ax = d using the Thomas algorithm."
- `src/kompaneets.rs:242` — "Factorize once, solve two right-hand sides against the SAME tridiagonal" — rewrite: "Factorizes once and solves two right-hand sides against the same tridiagonal matrix."
- `src/cosmology.rs:45` — "Validate cosmological parameters." — rewrite: "Validates cosmological parameters."
- `src/cosmology.rs:87` — "Construct a Cosmology from dimensionless parameters, validating the inputs." — rewrite: "Constructs a `Cosmology` from dimensionless parameters, validating the inputs."
- `src/cosmology.rs:210` — "H₀ in 1/s" — rewrite: "Returns H₀ in 1/s."
- `src/cosmology.rs:269` — "Hubble rate H(z) in 1/s" — rewrite: "Returns the Hubble rate H(z) in 1/s."
- `src/bremsstrahlung.rs:120` — "BR emission coefficient K_BR at frequency x." — rewrite: "Computes the BR emission coefficient K_BR at frequency x."
- `src/spectrum.rs:127` — "Compute the Compton equilibrium temperature ratio T_e^eq / T_z" — rewrite: "Computes the Compton equilibrium temperature ratio T_e^eq / T_z."
- `src/grid.rs:163` — "Build a FrequencyGrid from a sorted vector of grid points." — rewrite: "Builds a `FrequencyGrid` from a sorted vector of grid points."
- `src/dark_photon.rs:76` — "NWA dark-photon conversion parameter γ_con (dimensionless)." — rewrite: "Computes the NWA dark-photon conversion parameter γ_con (dimensionless)."

**A1 — single-line const/accessor docs drop the terminal period (38 instances).** Every multi-line prose doc block in these files correctly ends its last sentence with a period; it is specifically the one-line "definition-style" docs on `pub const` and simple `pub fn` accessors that omit it, concentrated in `constants.rs` (17), `cosmology.rs` (16), and `spectrum.rs` (5). Grep used: Python script comparing each `pub` item's immediately preceding doc line against `[.:!?]$`. First 10:
- `src/constants.rs:42` — "1 eV in Joules (exact by 2019 SI redefinition)" — rewrite: add trailing period.
- `src/constants.rs:54` — "2s→1s two-photon decay rate [s⁻¹]" — rewrite: add trailing period.
- `src/constants.rs:68` — "Electron Compton wavelength: h / (m_e * c)" — rewrite: add trailing period (also missing a stated output unit, see A4 below).
- `src/constants.rs:71` — "m_e c^2 in Joules" — rewrite: add trailing period.
- `src/constants.rs:83` — "Helium mass fraction" — rewrite: "Helium mass fraction (dimensionless)."
- `src/constants.rs:86` — "Helium number fraction relative to hydrogen: f_He = Y_p / (4*(1-Y_p))" — rewrite: add trailing period.
- `src/constants.rs:89` — "Effective number of neutrino species" — rewrite: add trailing period.
- `src/constants.rs:92` — "km/s/Mpc → 1/s" — rewrite: "Converts km/s/Mpc to 1/s." (also fixes the T11 arrow, see below).
- `src/constants.rs:96` — "G_1 = ∫₀^∞ x n_pl(x) dx = π²/6 = ζ(2)" — rewrite: add trailing period.
- `src/cosmology.rs:210` — "H₀ in 1/s" — rewrite: add trailing period.

**F11 — em dash written with surrounding spaces (33 instances found, 0 counter-examples of the tight form).** Google style uses `word—word` with no surrounding spaces; every em dash found in these files instead uses `word — word`. Grep used: `grep -E ' — '` vs. `grep -oE '[^ ]—[^ ]'` (0 hits) over the extracted doc-comment corpus. First 10:
- `src/kompaneets.rs:234` — "— including its `upper[i]/denom` division chain, which is the serial" — rewrite: close up the spaces around the dash.
- `src/double_compton.rs:13` — "- Lightman (1981) — original DC coefficient" — rewrite: "Lightman (1981)—original DC coefficient".
- `src/bremsstrahlung.rs:13` — "- Karzas & Latter (1961) — original Gaunt factors"
- `src/bremsstrahlung.rs:14` — "- Itoh et al. (2000) — relativistic thermal BR fits"
- `src/bremsstrahlung.rs:226` — "`exp(S3_PI·(ln 2.25 + ½lnθ_e) + 1.425)` — the x-independent half of"
- `src/electron_temp.rs:12` — "physical signal — do not use it in the solver. It is retained here only"
- `src/recombination.rs:30` — "- Peebles (1968) — Three-level atom model"
- `src/recombination.rs:31` — "- Péquignot, Petitjean & Boisson (1991) — Case-B recombination fit"
- `src/recombination.rs:32` — "- Seager, Sasselov & Scott (1999) — RECFAST"
- `src/recombination.rs:33` — "- Chluba & Thomas (2011, arXiv:1011.3758) — Updated fudge factor"

(This pattern is so uniform across all 11 files — including the paper-citation lines — that it reads as a deliberate scientific-prose convention rather than an oversight; flagged as MEDIUM rather than HIGH for that reason, but it is a genuine, code-wide deviation from the cited style guide.)

### Findings by file

#### `src/kompaneets.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 15 | T11 | MEDIUM | `with Crank-Nicolson time stepping → tridiagonal system.` | "…with Crank-Nicolson time stepping, producing a tridiagonal system." |
| 436 | T9 | LOW | `becomes bordered tridiagonal, solved in O(N) via two Thomas solves.` | "…solved in O(N) using two Thomas solves." |
| 526 | T12 | MEDIUM | `DC/BR coupling data for implicit backward Euler within the Kompaneets step.` | Neither "DC" nor "BR" is spelled out anywhere in this file; add a one-time gloss or a `[double_compton]`/`[bremsstrahlung]` doc link at first use. |
| 529 | T1 | MEDIUM | `we pass the emission rates and equilibrium targets so DC/BR is solved implicitly` | "…the caller passes the emission rates and equilibrium targets so DC/BR is solved implicitly…" |
| 571 | T9 | LOW | `DC/BR is handled via backward Euler within the Newton iteration:` | "DC/BR is handled with backward Euler within the Newton iteration:" |
| 583 | T9 | LOW | `` loop exits via `max_newton_iter`, `last_correction` is the final step `` | "…loop exits after reaching `max_newton_iter`…" |
| 587 | A4 | HIGH | `pub fn kompaneets_step_coupled_inplace(` | 10 parameters (`grid`, `delta_n`, `theta_e`, `theta_z`, `dtau`, `dcbr`, `rho_coupling`, `ws`, `max_dn_abs`, `max_newton_iter`); only `delta_n` and `max_dn_abs` are described in prose. Add an `# Arguments` list (following the pattern already used in `bremsstrahlung.rs::br_emission_coefficient`) covering the remaining 8, especially `dtau` (units) and `max_newton_iter` (valid range/default). |

#### `src/double_compton.rs`
No findings beyond the A2/A1/F11 patterns above (`dc_prefactor`, `dc_relativistic_correction`, etc. contribute to the A2 tally). This file is otherwise a good example of the abbreviation rule done right: "Double Compton (DC) scattering" is defined at first use in the module doc (line 1).

#### `src/bremsstrahlung.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 6 | T12 | HIGH | `The BR emission coefficient:` | "BR" is the module's central term and is never spelled out anywhere in the file (contrast `double_compton.rs`, which writes "Double Compton (DC)"). Change the module summary at line 1 to "Bremsstrahlung (BR) (free-free) emission and absorption." or similar. |
| 113–120 | A4 | HIGH | `` * `x_e_frac` - ionization fraction X_e = N_e/N_H `` (last of 7 listed args) | `br_emission_coefficient` takes 8 parameters; `cosmo: &crate::cosmology::Cosmology` is not listed in the `# Arguments` block. Add `` * `cosmo` - cosmology used to derive the radiation temperature and helium ionization fractions ``. |
| 221–223 | A4 | HIGH | `pub n_hii: f64,` / `pub n_heiii: f64,` / `pub n_heii: f64,` | Three public fields of `BrPrecomputed` have no doc comment, sandwiched between documented `phi` and `half_ln_theta_e`. Add one-line docs, e.g. "Precomputed H⁺ number density [1/m³]." |
| 248 | T9 | MEDIUM | `` Returns `0.0` for `ln_x < −69` (i.e. `x < 1e-30`), which propagates through `` | "…(that is, `x < 1e-30`)…" |
| 284–285 | A4/A5 | MEDIUM | `Precompute x-independent BR factors.` | `br_precompute` returns `Option<BrPrecomputed>` and returns `None` when `theta_e < 1e-30 \|\| n_e < 1e-30` (checked in the body), but the doc never mentions the `None` case. Add "Returns `None` if `theta_e` or `n_e` is effectively zero." |

#### `src/electron_temp.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 10 | T9 | LOW | `computed from Δn only. The full form ρ_eq = I₄/(4G₃) (below) has` | Replace "(below)" with a concrete forward reference, e.g. "(see `update_equilibrium`)". |
| 36 | T15 | MEDIUM | `` Pass `cosmo.theta_z(z)` so a non-default T_CMB is honoured. `` | American spelling: "…is honored." |

#### `src/recombination.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 88 | T9 | MEDIUM | `by H⁺ at z ≳ 1500 (H is fully ionized throughout He recombination since` | "since" here means "because": "…throughout He recombination because χ_I(H) = 13.6 eV ≪ χ_II(He) = 54.4 eV)." |
| 90 | T9 | MEDIUM | `(y_II = 1 limit) introduces ≲7% error in n_e vs the fully self-consistent` | "…error in n_e compared with the fully self-consistent…" |
| 121 | T1 | MEDIUM | `n_e ≈ n_H + y_I·n_He; we approximate y_I = 1 to keep the equation linear,` | "…the code approximates y_I = 1 to keep the equation linear," |
| 207 | T9 | LOW | `` - `rate_lya_escape`: Lyman-α escape via Sobolev approximation. `` | "…Lyman-α escape using the Sobolev approximation." |
| 309 | T1 | MEDIUM | `This is where the Peebles correction becomes significant and we` | Rewrite to avoid first person, e.g. "…significant; the solver switches from the Saha equation to the TLA ODE here." |
| 337 | T1 | MEDIUM | `Using the Saha relation β_B = α_B X_S² n_H / (1−X_S), we rewrite:` | "…the ODE is rewritten as:" |
| 393 | T11 | MEDIUM | `Redshift where Saha → Peebles switch occurs` | "Redshift where the switch from the Saha to the Peebles equation occurs" |

#### `src/constants.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 8 | T15 | MEDIUM | `Planck normalisations, and DC/BR emission prefactors.` | American spelling: "Planck normalizations, and DC/BR emission prefactors." |
| 68 | A4 | MEDIUM | `Electron Compton wavelength: h / (m_e * c)` | Every sibling constant states its output unit (e.g. `LAMBDA_LYA`: "…, in m."); `LAMBDA_ELECTRON` does not. Add ", in m." |
| 83 | A4 | LOW | `Helium mass fraction` | `ALPHA_FS` explicitly states "(dimensionless)" two blocks above; `Y_P` should too for consistency. |
| 86 | A4 | LOW | `Helium number fraction relative to hydrogen: f_He = Y_p / (4*(1-Y_p))` | Add "(dimensionless)." |
| 89 | A4 | LOW | `Effective number of neutrino species` | Add "(dimensionless)." |
| 92 | T11 | LOW | `km/s/Mpc → 1/s` | "Converts km/s/Mpc to 1/s." |
| 153 | T9 / T15 | MEDIUM | `(e.g. Planck 2018's 2.7255 K) is honoured.` | "(for example, Planck 2018's 2.7255 K) is honored." |

#### `src/cosmology.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 125 | T13 | LOW | `are known a priori to be valid. Using non-finite or zero \`h\` / \`y_p\`` | "…are known in advance to be valid…" |

(16 A1 instances and roughly 30 A2 instances in this file are covered by the Systematic patterns section above and not re-listed per-line here.)

#### `src/spectrum.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 6 | T9 | MEDIUM | `` coefficient that multiplies the shape — e.g. `Δn_μ(x) = μ · M(x)`, `` | "…for example, `Δn_μ(x) = μ · M(x)`," |

#### `src/grid.rs`
No findings beyond the A2 pattern (`validate`, `from_points`, `new`, `log_uniform`, `uniform`, `find_index` all use imperative verbs). This file otherwise conforms well: `# Panics` sections are present on every function that can panic (`from_points`, `log_uniform`, `uniform`), defaults are stated (`GridConfig::x_min`: "Default 1e-4."), and units are given throughout.

#### `src/dark_photon.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 76 | T12 | HIGH | `NWA dark-photon conversion parameter γ_con (dimensionless).` | "NWA" (narrow-width approximation) is used as the lead word of this summary but is never spelled out anywhere in the file — the module doc (line 1) writes out "narrow-width approximation" but never introduces the acronym "(NWA)". Add "(NWA)" the first time "narrow-width approximation" appears in the module doc. |

#### `src/axion.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 58 | T12 | HIGH | `` NWA axion–photon conversion parameter `γ_con` (dimensionless). `` | Same issue as `dark_photon.rs:76`: the module doc (lines 1–2) writes "narrow-width (Landau–Zener) approximation" but never defines "NWA". |

### What conforms

- Reference lists (`Kompaneets (1957), JETP`; `Chluba & Sunyaev (2012), MNRAS 419, 1294 [Eq. 4]`, etc.) are present and consistently formatted in every module's `//!` header, with arXiv IDs where applicable.
- `double_compton.rs` and `recombination.rs` correctly spell out abbreviations at first use in the file ("Double Compton (DC)", "Peebles three-level atom (TLA)").
- `grid.rs` consistently uses `# Panics` sections on every function that can panic, and states defaults and valid ranges (`GridConfig` field docs).
- `bremsstrahlung::br_emission_coefficient` uses a proper `# Arguments` list with stated units (`[1/m³]`, etc.) for 7 of its 8 parameters — the right pattern, just incompletely applied (see A4 finding above).
- Physical-constant docs in `constants.rs` state units for the large majority of entries (`m/s`, `J·s`, `kg`, `m²`), and explicitly flag dimensionless quantities where the file is at its best (`ALPHA_FS`: "(dimensionless)").
- No public function in scope that calls `.unwrap()`/`.expect()`/`panic!`/`unreachable!` was found lacking a `# Panics` section (checked programmatically across all 11 files).
- No entirely undocumented `pub fn`, `pub struct`, `pub enum`, or `pub const` item was found in any of the 11 files — the only missing-doc gaps are the three `BrPrecomputed` fields noted above.


---

## 9. Rust doc comments and CLI help: solver and entry points

Files reviewed: `src/lib.rs` (95 lines), `src/main.rs` (119 lines), `src/solver.rs` (doc comments through line 2000, before `#[cfg(test)]`), `src/greens.rs` (through line 763), `src/distortion.rs` (through line 450), `src/energy_injection.rs` (through line 1369), `src/output.rs` (through line 664), `src/cli.rs` (through line 1974, plus its `println!`/`eprintln!` CLI help/error strings), `src/bin/check_adiabatic.rs` (51 lines), `examples/custom_injection.rs`, `examples/energy_budget.rs`, `examples/generate_parity_fixtures.rs`, `examples/photon_diag.rs`, `examples/temporal_error_check.rs` (header comments only).

### Summary

The dominant, highest-impact problem is **A2**: essentially every function/method doc comment across `solver.rs`, `greens.rs`, `distortion.rs`, `energy_injection.rs`, `output.rs`, and `cli.rs` opens with an imperative verb ("Compute", "Set", "Write", "Parse", "Build", "Execute") instead of third-person present tense ("Computes", "Sets", "Writes"). This is not a PEP 257/NumPy carryover (these are Rust files, and the Rust standard library itself uses third person), so it is graded as a genuine, systemic MEDIUM finding rather than DOMAIN — roughly 100 instances across the six files. Second, several `Result`-returning public functions (`load_heating_table`, `load_photon_source_table`, `OutputFormat::from_str`, `parse_command`, `build_cosmology`, `print_info`, and the four `execute_*` entry points in `cli.rs`) never state what triggers `Err` (A4). Third, the prose inside doc comments carries the word-list violations CLAUDE.md itself would flag as unwanted register drift: "e.g." (11+ instances), "via" (7 instances), "since"/"should" used causally/as requirements, one explicit first-person "we" (`output.rs:394`) and one "our" (`distortion.rs:78`), and "currently" (T6). The CLI help text is comparatively clean (backticked flags, consistent `Default: …` phrasing) but has one capitalization break in the subcommand list (`cli.rs:834`) and one "vs" in prose (`cli.rs:1008`). `examples/photon_diag.rs` has no header doc comment at all.

| Rule | Count | Severity |
|---|---|---|
| A2 Imperative-mood function summaries | ~100 (29 solver.rs, 9 greens.rs, 4 distortion.rs, 16 energy_injection.rs, 18 output.rs, 22 cli.rs) | MEDIUM |
| A4 `Result`-returning fns with undocumented error conditions | 10 | MEDIUM–HIGH |
| A1 Duplicate/redundant summary sentence in one doc block | 1 | LOW |
| A3 Function/const doc restates the name | 1 | LOW |
| A4 `pub mod` declarations in `lib.rs` with no doc comment on the declaration line | 16 | LOW (module itself is documented via its own `//!`) |
| T9 "e.g." | 11 | MEDIUM |
| T9 "via" | 7 | MEDIUM |
| T9 "since" meaning "because" | 2 | MEDIUM |
| T9 "should" for a requirement | 3 | MEDIUM |
| T9 "etc." | 2 | MEDIUM |
| T9 "and/or" | 1 | MEDIUM |
| T9 "above" for document position | 1 | MEDIUM |
| T1 "we"/"our" | 2 | MEDIUM |
| T1 "the user" | 1 | MEDIUM |
| T6 "currently" | 1 | LOW |
| T10 "sanity check" | 1 | MEDIUM |
| T11 "+" standing in for "and" | 2 | MEDIUM |
| T11 "↔" arrow in a prose title | 1 | MEDIUM |
| T3 "will" for near-future product/user behavior | 1 | LOW |
| H1 Title-case heading in a `//!` doc block | 1 | MEDIUM |
| F1 File paths / flags / commands not in code font | 3 | MEDIUM |
| F6 Bold used for term introduction instead of italics | 1 | LOW |
| F11 Em dash with surrounding spaces | 1 | LOW |
| H5 Missing file-level summary (`examples/photon_diag.rs`) | 1 | HIGH |
| L3 Inconsistent capitalization in a parallel CLI list | 2 | MEDIUM |
| F3 Angle-bracket CLI placeholders (`<z>`, `<val>`, …) | pervasive | DOMAIN |

### Systematic patterns

**A2 — imperative-mood function summaries.** Grep used: a Python AST-light scan pairing each `fn`/`pub fn` with the first line of its preceding `///` block (regex `^\s*(fn|pub fn)\s`). Representative first-10 per file:
- `src/solver.rs:76` — "Validate solver configuration parameters." — "Validates solver configuration parameters."
- `src/solver.rs:406` — "Compute DC+BR heating integral and optionally its analytic derivative" — "Computes…"
- `src/solver.rs:526` — "Construct a solver with the given cosmology and frequency grid." — "Constructs…"
- `src/solver.rs:640` — "Attach an energy-injection scenario, validating it first." — "Attaches…"
- `src/solver.rs:688` — "Replace the solver configuration and reset the current redshift to" — "Replaces… resets…"
- `src/solver.rs:695` — "Set an initial photon perturbation Δn(x) for the next PDE run." — "Sets…"
- `src/solver.rs:721` — "Reset solver state for reuse, keeping grid and recombination cache." — "Resets…"
- `src/solver.rs:843` — "Update ρ_e from distortion feedback + injection." — "Updates ρ_e from distortion feedback and injection."
- `src/solver.rs:1066` — "Subtract the temperature shift component from Δn to enforce" — "Subtracts…"
- `src/solver.rs:1105` — "Advance the solver by a single adaptively-chosen timestep." — "Advances…"
(19 more in `solver.rs`; also `greens.rs:109,146,213,226,243,365,488,672,696`; `distortion.rs:38,379,430,439`; `energy_injection.rs:21,216,301,359,457,759,957,994,1117,1146,1192,1233,1243,1290,1329,1353`; `output.rs:55,94,112,160,208,230,282,326,346,392,414,437,468,493,510,587,615,651`; `cli.rs:359,409,623,710,752,818,900,1040,1065,1184,1246,1270,1288,1298,1310,1358,1377,1427,1542,1613,1710,1822`.)

**A4 — `Result`-returning public functions with no stated error condition.** All 10 instances (≤10, listed in full):
- `src/energy_injection.rs:307` `load_heating_table` — doc says only "Returns `TabulatedHeating` variant," never what makes it `Err` (malformed CSV, non-numeric field, unsorted z). HIGH — this function does file I/O and is a documented CLI entry path.
- `src/energy_injection.rs:366` `load_photon_source_table` — same gap. HIGH.
- `src/output.rs:652` `from_str` — doc is "Parse an output format from a string (`json`, `csv`, `table`)."; no mention that any other string is `Err`. MEDIUM.
- `src/cli.rs:409` `parse_command` — "Parse CLI arguments into a Command." No error conditions. MEDIUM.
- `src/cli.rs:752` `build_cosmology` — "Build a Cosmology from CosmoOpts." No error conditions. MEDIUM.
- `src/cli.rs:1040` `print_info` — "Print cosmology info." No error conditions. MEDIUM.
- `src/cli.rs:1184` `execute_greens` — "Execute a Green's function calculation. Returns result without doing I/O." — return value is described, not the error trigger. MEDIUM.
- `src/cli.rs:1427` `execute_solve` — same pattern. MEDIUM.
- `src/cli.rs:1613` `execute_sweep` — same pattern. MEDIUM.
- `src/cli.rs:1710` `execute_photon_sweep` — same pattern. MEDIUM.

**T9 word list.** Grep pattern: `\be\.g\.|i\.e\.|etc\.|\bvia\b|\bvs\.?\b|and/or|\bsince\b|\bshould\b|\babove\b` applied to the extracted doc-comment files.
- "e.g." (11, first 10): `solver.rs:133` "Collect non-fatal validation warnings (e.g. regimes where the"; `solver.rs:342` "strong, e.g. at z ≳ 10⁶ or during a photon-injection burst)."; `solver.rs:642` "Returns `Err` if the scenario parameters are unphysical (e.g. negative"; `solver.rs:850` "nonlinear denominator. Necessary for strong depletions (e.g., dark photon"; `distortion.rs:396` "e.g. frozen/locked-in photon-injection bumps from z < 1100 that never"; `energy_injection.rs:54` "This matches the CosmoTherm convention (e.g. f_ann = 1e-22 eV/s for s-wave)."; `energy_injection.rs:1334` "mode at x_inj); for single-photon channels (e.g. X → γ X') the"; `cli.rs:53` "Injection-scenario tag (positional, e.g. `single-burst`,"; `cli.rs:1311` "Used to compress repeated per-worker warnings (e.g. one identical"; `lib.rs:28` is inside a fenced `rust,no_run` code example and is excluded per the accuracy rule against flagging code-block text.
  - Suggested rewrite pattern: "e.g." → "for example".
- "via" (7): `solver.rs:314` "Active injection scenario, if any. Set via [`Self::set_injection`]."; `solver.rs:1707` "Extract `(μ, y)` from the current `Δn(x)` via the default joint"; `distortion.rs:72` "subspace spanned by (Y_SZ, M, G) via Gram-Schmidt in the order"; `distortion.rs:82` "⟨Δn, e_T⟩) are mapped back to (μ, y, ΔT/T) via exact back-substitution of"; `distortion.rs:213` "Initial guess: bootstrap from `decompose_gram_schmidt` (converted via"; `energy_injection.rs:80` "The frequency-dependent source is applied separately via"; `energy_injection.rs:1197` "the hard error for that case lives in the solver-builder path via".
  - Suggested rewrite: "via" → "with"/"by"/"through" depending on context, or restructure.
- "since" meaning "because" (2): `solver.rs:44` "with 5–6× fewer steps, since `dtau_max` takes over as the binding"; `distortion.rs:219` "which spans the same 3-D subspace as the CJ2014 basis (Y_SZ, M, G) since". Rewrite: "since" → "because".
- "should" for a requirement (3): `distortion.rs:40` "Precondition: the supplied grid should extend beyond [x_min, x_max] on both"; `energy_injection.rs:95` "Gaussian width in frequency (should match grid resolution)"; `cli.rs:1378` "`SolverBuilder::build` does, returning the soft warnings that should". Rewrite: "should" → "must".
- "etc." (2): `solver.rs:1668` "The solver state (delta_n, rho_e, etc.) is taken from the current state,"; `energy_injection.rs:1359` "**WARNING**: The Green's function routines (`mu_from_heating`, etc.)". Rewrite: name the remaining items or drop "etc.".
- "and/or" (1): `cli.rs:62` "Cosmology overrides (preset and/or individual parameters)." Rewrite: "preset, individual parameters, or both".
- "above" for document position (1): `cli.rs:210` "`planck2018`. Individual flags above override preset values." Rewrite: "The preceding individual flags override preset values."

**T1 first person.**
- `output.rs:394` "Pre-warnings format was a bare JSON array. With warnings we wrap into" — explicit "we". Rewrite: "…the JSON array is wrapped …" (passive is acceptable here since the agent, a past format-migration decision, is not a specific actor) or "the format wraps into…".
- `distortion.rs:78` "uniform-channel flat sum to our non-uniform x-grid and reduces to it in" — "our" possessive. Rewrite: "the solver's non-uniform x-grid".
- `cli.rs:1378` "…returning the soft warnings that should" (continues) "surface to the user." — "the user" instead of "you". Rewrite: "…the warnings that must surface to you" or, better for an internal API doc, "…the warnings the caller must surface".

**T11 symbols standing in for words in prose.**
- `solver.rs:843` "Update ρ_e from distortion feedback + injection." — "+" means "and", not addition of two ρ_e-valued quantities in a formula. Rewrite: "Updates ρ_e from distortion feedback and injection."
- `cli.rs:871` `println!("  --no-dcbr             Disable double-Compton + bremsstrahlung");` — same. Rewrite: "Disable double-Compton and bremsstrahlung".
- `examples/generate_parity_fixtures.rs:1` "Generate Rust↔Python parity fixtures (validation-audit Part B2)." — "↔" arrow standing in for "to and from"/"and" in a title, not a reaction equation. Rewrite: "Generate Rust-to-Python parity fixtures".

**A1 duplicate summary.** `src/solver.rs:1066-1080`: the doc comment for `subtract_temperature_shift` opens with "Subtract the temperature shift component from Δn to enforce photon number conservation: ∫x² Δn dx = 0.", gives a 4-step algorithm, then restates "Subtract the number-conserving temperature shift from Δn." at line 1079 before "Returns the δT/T that was subtracted." at 1080. The second summary sentence is redundant with the first and reads like an unremoved edit artifact.

**L3 parallel construction in CLI help.**
- `src/cli.rs:831-838` (subcommand list in `print_help`): every description starts capitalized ("Single PDE solve…", "PDE sweep…", "Analytic Green's…", "Print the parameters…") except `cli.rs:834`: `println!("  photon-sweep-batch      photon-sweep for several x_inj values in parallel");` — starts lowercase. Fix: capitalize to "Photon-sweep for several x_inj values in parallel" (or reword to avoid starting with the subcommand's own name).
- `src/cli.rs:884` (`print_cosmo_options_help`): `println!("  --cosmology <preset>  default, planck2015, planck2018");` starts lowercase while every sibling line in the same block ("Fractional baryon density…", "Reduced Hubble parameter…", "Helium mass fraction") starts capitalized. Fix: "Preset: `default`, `planck2015`, `planck2018`".

### Findings by file

#### `src/lib.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 14-17 | F6 | LOW | \`- **Compton scattering** (Kompaneets equation): frequency redistribution\` | Use italics for term introduction, not bold: `*Compton scattering*`. |
| 40 | H1 | MEDIUM | `//! ## Green's Function Mode` | Sentence case: `## Green's function mode`. |
| 64-79 | A4 | LOW | `pub mod bremsstrahlung;` (through `pub mod spectrum;`, 15 lines) | Each of these 15 `pub mod` declarations has no doc comment on the declaration line itself; rustdoc will pull the one-line summary from each module's own top `//!` instead, so this is not a hard violation, but it means `lib.rs`'s own text gives the crate-root module index no differentiation from the `axion`/`prelude` entries that do carry inline docs. Consider a short `///` on each for a more informative crate-root page. |

#### `src/main.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 3 | F1 | MEDIUM | `//! All logic lives in the library (cli.rs, output.rs). This binary just` | Backtick the file names: `` `cli.rs` ``, `` `output.rs` ``. |
| 89 | A2 | MEDIUM | `/// Get the output writer: file if --output specified, stdout otherwise.` | "Gets the output writer…"; also backtick `` `--output` ``. |
| 100 | A2 | MEDIUM | `/// Write any result type to the configured output destination and format.` | "Writes any result type…" |

#### `src/bin/check_adiabatic.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 1 | T10 | MEDIUM | `//! Adiabatic cooling sanity check.` | "Adiabatic cooling consistency check." |

#### `src/solver.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 40 | T3 | LOW | `/// runs. The fields that most users will want to change are [\`Self::z_start\`],` | "The fields most users want to change are…" |
| 44 | T9 | MEDIUM | `` /// with 5–6× fewer steps, since `dtau_max` takes over as the binding `` | "…because `dtau_max` takes over…" |
| 76 | A2 | MEDIUM | `/// Validate solver configuration parameters.` | "Validates solver configuration parameters." |
| 133 | A2, T9 | MEDIUM | `/// Collect non-fatal validation warnings (e.g. regimes where the` | "Collects…"; "for example, regimes where the". |
| 294 | T6 | LOW | `` /// Many fields are currently `pub` for use by examples, tests, and diagnostic `` | Drop "currently": "Many fields are `pub`…" |
| 314 | T9 | MEDIUM | `` /// Active injection scenario, if any. Set via [`Self::set_injection`]. `` | "Set with [`Self::set_injection`]." |
| 342 | T9 | MEDIUM | `/// strong, e.g. at z ≳ 10⁶ or during a photon-injection burst).` | "for example at z ≳ 10⁶ …" |
| 406 | A2 | MEDIUM | `/// Compute DC+BR heating integral and optionally its analytic derivative` | "Computes DC+BR heating integral…" |
| 526 | A2 | MEDIUM | `/// Construct a solver with the given cosmology and frequency grid.` | "Constructs a solver…" |
| 640 | A2 | MEDIUM | `/// Attach an energy-injection scenario, validating it first.` | "Attaches an energy-injection scenario…" |
| 642 | T9 | MEDIUM | `` /// Returns `Err` if the scenario parameters are unphysical (e.g. negative `` | "for example negative". |
| 688 | A2 | MEDIUM | `/// Replace the solver configuration and reset the current redshift to` | "Replaces the solver configuration and resets…" |
| 695 | A2 | MEDIUM | `/// Set an initial photon perturbation Δn(x) for the next PDE run.` | "Sets an initial photon perturbation…" |
| 700 | A4 | LOW | `` /// Panics if any entry is non-finite — silently passing NaN/Inf into the `` | Move under an explicit `# Panics` heading for rustdoc's standard rendering. |
| 721 | A2 | MEDIUM | `/// Reset solver state for reuse, keeping grid and recombination cache.` | "Resets solver state…" |
| 843 | A2, T11 | MEDIUM | `/// Update ρ_e from distortion feedback + injection.` | "Updates ρ_e from distortion feedback and injection." |
| 850 | T9 | MEDIUM | `/// nonlinear denominator. Necessary for strong depletions (e.g., dark photon` | "for example, dark photon". |
| 1066 / 1079 | A1 | LOW | `/// Subtract the temperature shift component from Δn to enforce` … `/// Subtract the number-conserving temperature shift from Δn.` | Delete the redundant restated sentence at line 1079; keep the return-value sentence ("Returns the δT/T that was subtracted."). |
| 1105 | A2 | MEDIUM | `/// Advance the solver by a single adaptively-chosen timestep.` | "Advances the solver…" |
| 1114 | A2 | MEDIUM | `/// Take a single timestep with a specified dz (instead of the adaptive choice).` | "Takes a single timestep…" |
| 1479 | A2 | MEDIUM | `` /// Integrate from `z_start` to `z_end`, recording a snapshot at each `` | "Integrates from…" |
| 1651 | A2 | MEDIUM | `` /// Run the solver with `n_snapshots` log-spaced snapshot redshifts between `` | "Runs the solver…" |
| 1667 | A2 | MEDIUM | `/// Save a snapshot with a specific redshift label.` | "Saves a snapshot…" |
| 1668 | T9 | MEDIUM | `/// The solver state (delta_n, rho_e, etc.) is taken from the current state,` | Name the remaining fields or drop "etc." |
| 1707 | A2, T9 | MEDIUM | `` /// Extract `(μ, y)` from the current `Δn(x)` via the default joint `` | "Extracts…"; "…using the default joint". |
| 1715 | A2 | MEDIUM | `` /// Run the solver and return an owned [`crate::output::SolverResult`] `` | "Runs the solver and returns…" |
| 1739 | A2 | MEDIUM | `/// Create a builder for configuring a solver with a fluent API.` | "Creates a builder…" |
| 1803 | A2 | MEDIUM | `/// Set the frequency grid configuration.` | "Sets the frequency grid configuration." |
| 1809 | A2 | MEDIUM | `/// Use the fast (500-point) grid for quick tests.` | "Uses the fast (500-point) grid…" |
| 1815 | A2 | MEDIUM | `/// Set the energy injection scenario.` | "Sets the energy injection scenario." |
| 1821 | A2 | MEDIUM | `/// Set the redshift range (z_start, z_end).` | "Sets the redshift range…" |
| 1828 | A2 | MEDIUM | `/// Set a complete solver config, overriding individual z/dy/dtau settings.` | "Sets a complete solver config…" |
| 1842 | A2 | MEDIUM | `/// Set the maximum fractional change in ln(1+z) per step (the` | "Sets the maximum fractional change…" |
| 1849 | A2 | MEDIUM | `/// Set the maximum Compton optical depth per step.` | "Sets the maximum Compton optical depth per step." |
| 1855 | A2 | MEDIUM | `/// Disable DC/BR processes (Kompaneets only).` | "Disables DC/BR processes…" |
| 1861 | A2 | MEDIUM | `/// Use operator-split DC/BR instead of coupled IMEX.` | "Uses operator-split DC/BR…" |
| 1867 | A2 | MEDIUM | `/// Disable number-conserving T-shift subtraction.` | "Disables number-conserving T-shift subtraction." |
| 1873 | A2 | MEDIUM | `/// Set the maximum number of Newton iterations per Kompaneets step.` | "Sets the maximum number of Newton iterations…" |
| 1879 | A2 | MEDIUM | `/// Build the configured solver.` | "Builds the configured solver." |

#### `src/greens.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 20 | T5 | LOW | `//! which is NOT simply (1 − J_μ) × J_bb*. As a result, the three branching` | "which is not equal to (1 − J_μ) × J_bb*." |
| 109 | A2 | MEDIUM | `/// Compute the Green's function G_th(x, z_h) for a delta-function energy injection` | "Computes the Green's function…" |
| 146 | A2 | MEDIUM | `/// Compute the spectral distortion from an arbitrary energy release history.` | "Computes the spectral distortion…" |
| 213 | A2 | MEDIUM | `/// Extract μ parameter from the Green's function approximation.` | "Extracts the μ parameter…" |
| 226 | A2 | MEDIUM | `/// Extract y parameter from the Green's function approximation.` | "Extracts the y parameter…" |
| 243 | A2 | MEDIUM | `/// Compute both μ and y from an arbitrary energy release history in a single pass.` | "Computes both μ and y…" |
| 331 | A3 | LOW | `/// Photon survival probability P_s(x, z).` | Restates the function name `photon_survival_probability`; add what it means physically, e.g. "Fraction of injected photons that survive DC/BR absorption to redshift 0." |
| 365 | A2 | MEDIUM | `/// Compute P_s = exp(−τ_ff) from the integrated DC+BR absorption optical depth.` | "Computes P_s…" |
| 488 | A2 | MEDIUM | `/// Compute the Compton-broadened photon bump and its energy integral f_int.` | "Computes the Compton-broadened photon bump…" |
| 672 | A2 | MEDIUM | `/// Compute μ from monochromatic photon injection at frequency x_inj.` | "Computes μ…" |
| 696 | A2 | MEDIUM | `/// Compute spectral distortion from an arbitrary photon injection history.` | "Computes spectral distortion…" |

#### `src/distortion.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 38 | A2 | MEDIUM | `/// Collect trapezoidal weights and indices for grid points within [x_min, x_max].` | "Collects trapezoidal weights…" |
| 40 | T9 | MEDIUM | `/// Precondition: the supplied grid should extend beyond [x_min, x_max] on both` | "…the supplied grid must extend beyond…" |
| 72 | T9 | MEDIUM | `/// subspace spanned by (Y_SZ, M, G) via Gram-Schmidt in the order` | "…spanned by (Y_SZ, M, G) using Gram-Schmidt…" |
| 78 | T1 | MEDIUM | `/// uniform-channel flat sum to our non-uniform x-grid and reduces to it in` | "…to the solver's non-uniform x-grid…" |
| 82 | T9 | MEDIUM | `` /// ⟨Δn, e_T⟩) are mapped back to (μ, y, ΔT/T) via exact back-substitution of `` | "…mapped back… by exact back-substitution of". |
| 213 | T9 | MEDIUM | `` /// Initial guess: bootstrap from `decompose_gram_schmidt` (converted via `` | "(converted with δ_BF = …)". |
| 219 | T9 | MEDIUM | `/// which spans the same 3-D subspace as the CJ2014 basis (Y_SZ, M, G) since` | "…(Y_SZ, M, G), because". |
| 379 | A2 | MEDIUM | `/// Decompose a spectral distortion into μ, y, and temperature shift components.` | "Decomposes a spectral distortion…" |
| 396 | T9 | MEDIUM | `/// e.g. frozen/locked-in photon-injection bumps from z < 1100 that never` | "for example, frozen (locked-in) photon-injection bumps…" |
| 416 | T9 | MEDIUM | `` /// `x_min`/`x_max`. Solvers should sample this once at startup and surface `` | "Solvers must sample this once at startup…" |
| 430 | A2 | MEDIUM | `/// Check distortion parameters against FIRAS limits.` | "Checks distortion parameters…" |
| 439 | A2 | MEDIUM | `/// Convert distortion Δn(x) to specific intensity ΔI_ν in MJy/sr.` | "Converts distortion Δn(x)…" |

#### `src/energy_injection.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 21 | A2 | MEDIUM | `/// Compute vacuum survival fraction for decaying particle photon injection.` | "Computes vacuum survival fraction…" |
| 54 | T9 | MEDIUM | `/// This matches the CosmoTherm convention (e.g. f_ann = 1e-22 eV/s for s-wave).` | "for example, f_ann = 1e-22 eV/s". |
| 80 | T9 | MEDIUM | `/// The frequency-dependent source is applied separately via` | "…applied separately with/by". |
| 95 | T9 | MEDIUM | `/// Gaussian width in frequency (should match grid resolution)` | "(must match grid resolution)". |
| 216 | A2 | MEDIUM | `/// Interpolate a value from a table sorted ascending in z, using linear` | "Interpolates a value…" |
| 301 | A2, A4 | MEDIUM/HIGH | `/// Load a tabulated heating rate from a CSV file.` | "Loads a tabulated heating rate…"; add an `# Errors` note on malformed rows/unsorted z. |
| 359 | A2, A4 | MEDIUM/HIGH | `/// Load a tabulated photon source from a CSV file.` | "Loads a tabulated photon source…"; document error triggers. |
| 457 | A2 | MEDIUM | `/// Validate parameters, returning an error message if invalid.` | "Validates parameters…" |
| 759 | A2 | MEDIUM | `/// Compute the heating rate d(Δρ_γ/ρ_γ)/dt at redshift z.` | "Computes the heating rate…" |
| 957 | A2 | MEDIUM | `/// Return refinement zones for adaptive grid resolution near injection features.` | "Returns refinement zones…" |
| 994 | A2, T9 | MEDIUM | `/// Return the characteristic injection redshift(s) for this scenario.` | "Returns the characteristic injection redshift or redshifts…" (avoid "(s)"). |
| 1117 | A2 | MEDIUM | `` /// Suggest a lower `x_min` for the frequency grid when needed. `` | "Suggests a lower `x_min`…" |
| 1146 | A2 | MEDIUM | `/// Check for strong distortion regime and return warnings.` | "Checks for a strong distortion regime and returns warnings." |
| 1192 | A2 | MEDIUM | `/// Warn when the dark-photon NWA resonance falls outside the validated` | "Warns when the dark-photon NWA resonance…" |
| 1197 | T9 | MEDIUM | `/// the hard error for that case lives in the solver-builder path via` | "…lives in the solver-builder path through/by". |
| 1233 / 1243 | A2 | MEDIUM | `/// Warn when the axion NWA resonance falls outside the validated redshift` | "Warns when the axion NWA resonance…" (both `cfg` variants). |
| 1290 | A2 | MEDIUM | `/// Warn when a tabulated-source table doesn't cover the solver's` | "Warns when a tabulated-source table doesn't cover…" |
| 1329 | A2 | MEDIUM | `/// Warn if stimulated emission (Bose enhancement) is missing for photon decay.` | "Warns if stimulated emission…" |
| 1334 | T9 | MEDIUM | `/// mode at x_inj); for single-photon channels (e.g. X → γ X') the` | "for example, X → γ X'". |
| 1353 | A2 | MEDIUM | `/// Compute the physical d(Δρ/ρ)/dz.` | "Computes the physical d(Δρ/ρ)/dz." |
| 1359 | T9 | MEDIUM | `` /// **WARNING**: The Green's function routines (`mu_from_heating`, etc.) `` | Name the routines or drop "etc." |

#### `src/output.rs`
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 55 | A2 | MEDIUM | `/// Serialize to a JSON string (zero dependencies).` | "Serializes to a JSON string…" |
| 94 | A2 | MEDIUM | `/// Write CSV (frequency, delta_n) to a writer.` | "Writes CSV…" |
| 112 | A2 | MEDIUM | `/// Write a human-readable summary table to a writer.` | "Writes a human-readable summary table…" |
| 160 | A2 | MEDIUM | `/// Serialize to a JSON string.` | "Serializes to a JSON string." |
| 208 | A2 | MEDIUM | `/// Write CSV summary to a writer.` | "Writes CSV summary…" |
| 230 | A2 | MEDIUM | `/// Write a human-readable summary table to stderr-style output.` | "Writes a human-readable summary table…" |
| 282 | A2 | MEDIUM | `/// Serialize to a JSON string.` | "Serializes to a JSON string." |
| 326 | A2 | MEDIUM | `/// Write CSV summary to a writer.` | "Writes CSV summary…" |
| 346 | A2 | MEDIUM | `/// Write a human-readable summary table.` | "Writes a human-readable summary table." |
| 392 | A2 | MEDIUM | `/// Serialize to a JSON object containing per-x_inj results and aggregated warnings.` | "Serializes to a JSON object…" |
| 394 | T1 | MEDIUM | `/// Pre-warnings format was a bare JSON array. With warnings we wrap into` | "…the format now wraps into…" (drop "we"). |
| 414 | A2 | MEDIUM | `/// Write combined CSV summary to a writer.` | "Writes combined CSV summary…" |
| 437 | A2 | MEDIUM | `/// Write a human-readable summary table.` | "Writes a human-readable summary table." |
| 468 | A2 | MEDIUM | `/// Serialize to a JSON string.` | "Serializes to a JSON string." |
| 493 | A2 | MEDIUM | `/// Write CSV to a writer.` | "Writes CSV to a writer." |
| 510 | A2 | MEDIUM | `/// Write a human-readable summary.` | "Writes a human-readable summary." |
| 587 | A2 | MEDIUM | `/// Write a JSON-safe float: NaN and Inf become null (valid JSON).` | "Writes a JSON-safe float…" |
| 615 | A2 | MEDIUM | `` /// Write a JSON array of strings, escaping `"`, `\`, and control characters. `` | "Writes a JSON array of strings…" |
| 651 | A2, A4 | MEDIUM | `` /// Parse an output format from a string (`json`, `csv`, `table`). `` | "Parses an output format…"; state the `Err` case ("`Err` if `s` is none of `json`, `csv`, `table`."). |

#### `src/cli.rs` (doc comments)
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| 53 | T9 | MEDIUM | `` /// Injection-scenario tag (positional, e.g. `single-burst`, `` | "for example `single-burst`". |
| 62 | T9 | MEDIUM | `/// Cosmology overrides (preset and/or individual parameters).` | "Cosmology overrides (a preset, individual parameters, or both)." |
| 210 | T9 | MEDIUM | `` /// `planck2018`. Individual flags above override preset values. `` | "The preceding individual flags override preset values." |
| 359 | A2 | MEDIUM | `` /// Reject any parsed `--flag` that no group of `allowed` contains. `` | "Rejects any parsed `--flag`…" |
| 409 | A2, A4 | MEDIUM | `/// Parse CLI arguments into a Command.` | "Parses CLI arguments into a Command."; state what makes it `Err` (unknown subcommand, missing required flag). |
| 623 | A2 | MEDIUM | `/// Parse flat --key value args into a HashMap (shared by all subcommands and legacy mode).` | "Parses flat `--key value` args…" (also backtick `--key value`). |
| 710 | A2 | MEDIUM | `/// Parse an optional float CLI argument, returning a clear error on invalid values.` | "Parses an optional float CLI argument…" |
| 752 | A2, A4 | MEDIUM | `/// Build a Cosmology from CosmoOpts.` | "Builds a Cosmology…"; state error triggers. |
| 818 | A2 | MEDIUM | `/// Print the general help overview to stdout.` | "Prints the general help overview to stdout." |
| 900 | A2 | MEDIUM | `/// Print detailed help for one subcommand to stdout.` | "Prints detailed help…" |
| 1040 | A2, A4 | MEDIUM | `/// Print cosmology info.` | "Prints cosmology info."; state error triggers. |
| 1065 | A2 | MEDIUM | `/// Build an InjectionScenario from CLI arguments.` | "Builds an InjectionScenario…" |
| 1184 | A2, A4 | MEDIUM | `` /// Execute a Green's function calculation. Returns result without doing I/O. `` | "Executes a Green's function calculation…"; state error triggers. |
| 1246 | A2 | MEDIUM | `/// Build a GridConfig from CLI options.` | "Builds a GridConfig…" |
| 1270 | A2 | MEDIUM | `/// Build a SolverConfig from CLI solver options with the given z_start.` | "Builds a SolverConfig…" |
| 1288 | A2 | MEDIUM | `/// Extract a human-readable message from a thread panic payload.` | "Extracts a human-readable message…" |
| 1298 | A2 | MEDIUM | `/// Generate the default log-spaced redshift array for photon sweep (150 points from 1e3 to 5e6).` | "Generates the default log-spaced redshift array…" |
| 1310 | A2 | MEDIUM | `` /// Deduplicate a `Vec<String>` while preserving first-occurrence order. `` | "Deduplicates a `Vec<String>`…" |
| 1358 | A2 | MEDIUM | `/// Apply common solver flags from CLI options to a solver instance.` | "Applies common solver flags…" |
| 1377 | A2 | MEDIUM | `/// Validate a (config, grid, injection) combination the same way` | "Validates a (config, grid, injection) combination…" |
| 1378 | T9, T1 | MEDIUM | `` /// `SolverBuilder::build` does, returning the soft warnings that should `` | "…the warnings that surface to you" or "…the warnings the caller must surface". |
| 1427 | A2, A4 | MEDIUM | `` /// Execute a single PDE solve. Returns result without doing I/O. `` | "Executes a single PDE solve…"; state error triggers. |
| 1542 | A2 | MEDIUM | `` /// Run `worker` over `items` on `n_threads` long-lived scoped threads pulling `` | "Runs `worker` over `items`…" |
| 1613 | A2, A4 | MEDIUM | `` /// Execute a sweep over multiple injection redshifts. Returns result without doing I/O. `` | "Executes a sweep…"; state error triggers. |
| 1710 | A2, A4 | MEDIUM | `/// Execute a photon injection sweep over multiple injection redshifts at a fixed x_inj.` | "Executes a photon injection sweep…"; state error triggers. |
| 1822 | A2 | MEDIUM | `/// Execute a batch photon injection sweep over multiple x_inj values.` | "Executes a batch photon injection sweep…" |

#### `src/cli.rs` and `src/main.rs` (CLI help/usage/error strings)
| Line | Rule | Severity | Quote | Suggested fix |
|---|---|---|---|---|
| `cli.rs:834` | L3 | MEDIUM | `println!("  photon-sweep-batch      photon-sweep for several x_inj values in parallel");` | Capitalize to match its siblings: "Photon-sweep for several x_inj values in parallel". |
| `cli.rs:884` | L3 | MEDIUM | `println!("  --cosmology <preset>  default, planck2015, planck2018");` | "Preset: `default`, `planck2015`, `planck2018`" to match capitalized siblings. |
| `cli.rs:1008` | T9 | MEDIUM | `println!("Accuracy vs the PDE: 2-5% for mu, ~5% for y; ~8-13% shape error in the");` | "Accuracy versus the PDE: …" |

Angle-bracket placeholders throughout `print_help`, `print_solver_options_help`, `print_cosmo_options_help`, `print_output_options_help`, and `print_subcommand_help` (`<z>`, `<val>`, `<n>`, `<path>`, `<preset>`, `<K>`, `<eV>`, `<x1,x2,...>`, and similar) are rated **DOMAIN** per the rubric: angle brackets are the standard CLI usage-string convention, and each placeholder is explained in the adjacent description on the same or following line, so F3's "unexplained placeholder" clause is satisfied even though the tokens are lowercase rather than UPPER_SNAKE_CASE.

### What conforms

- Every reviewed file states defaults, units, and valid ranges consistently for tunable fields (A5) — e.g. `solver.rs:40-64` (`SolverConfig` fields) and the CLI `COSMOLOGY`/`SOLVER OPTIONS` blocks (`cli.rs:882-897`) name the default value and, where relevant, the physical unit (kelvin, km/s/Mpc) for every parameter.
- Mechanical A4/A1 check: every `pub fn`, `pub struct`, `pub enum`, `pub const`, and `pub trait` in `solver.rs`, `greens.rs`, `distortion.rs`, `energy_injection.rs`, `output.rs`, and `cli.rs` has an attached `///` doc comment; the only undocumented `pub` declarations found are the 15 `pub mod` lines in `lib.rs`, each of which is documented instead via its own file's `//!` header (rustdoc pulls that as the module-index summary).
- Backticked code font is used correctly and pervasively for identifiers, types, and flags inside doc comments (e.g. `` [`Self::z_start`] ``, `` `dtau_max` ``, `` `--nc-z-min` ``).
- Math notation (θ_e, ρ_e, Δn, arrows in reaction equations like `X → γ X'`, "~"/"≲" for order-of-magnitude) is used correctly and consistently, and is out of scope for T9/T11 per the rubric's DOMAIN exception.
- CLI error messages (`cli.rs:379-403`, `validate_known_flags`) are exemplary: second-person-implicit imperative voice, backticked flag names, a concrete "Did you mean" suggestion, and a pointer to `--help` for more information — this matches F5's "For more information, see…" spirit even in a non-hyperlinked terminal context.
- No inclusive-language (T10) violations beyond the one "sanity check"; no gendered pronouns found.
- Example headers in `energy_budget.rs` and `temporal_error_check.rs` use backticked, properly fenced-style `Usage:` lines; only `custom_injection.rs` and `generate_parity_fixtures.rs` have bare (non-backticked) shell commands.


---

## Appendix: review rubric


Reference: https://developers.google.com/style/highlights and the linked pages under
https://developers.google.com/style/. If WebFetch is available, fetch the highlights page once to
confirm the rules. If you are unsure whether the guide has a rule, fetch the specific page
(for example https://developers.google.com/style/word-list, /headings, /lists, /code-samples,
/placeholders, /link-text, /api-reference-comments, /inclusive-documentation). Do not invent rules.
If you cannot confirm a rule, do not report it.

### Rule IDs (use these IDs in the report)

Tone and content
- T1 Second person: address the reader as "you". Avoid "we", "the user", "one", "let's".
- T2 Active voice; say who performs the action.
- T3 Present tense. Avoid "will" for product behavior.
- T4 Do not pre-announce ("This section describes...", "Note that", "It is important to", "As mentioned").
- T5 No "please", "simply", "just", "easy", "easily", "obviously", "of course", "quick(ly)" as claims about ease.
- T6 Timeless text: avoid "currently", "now", "new", "latest", "soon", "in the future", "recently", "as of this writing".
- T7 No exclamation marks, no marketing or buzzword language, no jokes, no idioms or culture-specific phrasing (global audience).
- T8 Condition before instruction ("If X, do Y"; "To do X, run Y"), not after.
- T9 Word list: "for example" not "e.g."; "that is" not "i.e."; avoid "etc.", "via", "vs." in prose, "and/or", "in order to",
     "allows you to" (use "lets you"), "leverage", "utilize", "may" for ability (use "can"), "should" for requirements (use "must"),
     "above"/"below" for document position (use "preceding"/"following" or a link), "(s)" plurals, "once" meaning "after", "since" meaning "because",
     "as" meaning "because", "while" meaning "although".
- T10 Inclusive language: "sanity check", "dummy", "blacklist/whitelist", "master/slave", "kill", "hang", "cripple", "blind", "crazy", "native" (as feature), "first-class", "man-hours", gendered pronouns.
- T11 Symbols standing in for words in prose: "&", "+", "~" (for approximately, outside math), arrows ("→", "->", "=>") in prose, "w/", "/" for "or", "#" for "number", "×" outside math. If the symbol is inside a math expression or is a standard physics notation, classify as DOMAIN (see severity).
- T12 Abbreviations and acronyms: spell out at first use in each document (PDE, CMB, DC, BR, GF, NWA, FIRAS, IMEX, ODE, DM, CLI, IC, etc.). Do not define abbreviations that are then never used.
- T13 Contractions and register: avoid overly formal ("thus", "hence", "whilst", "aforementioned") and overly casual text. Avoid Latin phrases ("a priori", "ad hoc", "vice versa", "per se") where plain English exists.
- T14 Sentence fragments and dropped articles in prose (telegraphic style: "Returns dict with keys", "Requires Rust toolchain").
- T15 American spelling. Serial (Oxford) comma. One space after periods.

Headings and structure
- H1 Sentence case for headings and titles (capitalize only first word and proper nouns).
- H2 Task headings start with a bare infinitive/imperative ("Install the package"), not a gerund ("Installing") ; conceptual headings are noun phrases.
- H3 No end punctuation in headings; avoid code font, links, abbreviations undefined, and ampersands in headings where avoidable.
- H4 Do not skip heading levels; one H1 per page; do not stack headings with no text between them.
- H5 Each page begins with an introductory sentence or paragraph saying what the page covers and for whom.

Lists, tables, procedures
- L1 Numbered lists for sequential steps; bulleted for non-sequential sets; description lists (term: definition) for term–value pairs.
- L2 Introduce every list and table with a complete sentence (usually ending in a colon).
- L3 Parallel construction in list items; consistent capitalization and end punctuation (start each item with a capital letter; period if items are sentences).
- L4 Procedures: one action per step; imperative verbs; state the goal or location before the action; "Optional:" prefix for optional steps; results not as separate steps.
- L5 Tables: header row in sentence case; no empty cells without explanation; no table for a single-dimension list.

Formatting
- F1 Code font for code, commands, flags, file names, paths, class/function/parameter names, environment variables, HTTP verbs, literal values. Missing code font, or code font on things that are not code (product names, emphasis).
- F2 Code samples: introduce each code block with a sentence (usually ending in a colon); no bare code blocks after headings; specify language on fenced blocks; lines ≤ 80 chars where possible; do not include shell prompt characters that break copy-paste unless output is shown; separate command from output.
- F3 Placeholders: UPPER_SNAKE_CASE (in code font, italic where supported), and explain each placeholder after the block ("Replace OUTPUT_FILE with ..."). Flag `<angle-bracket>`, `your-thing`, `foo/bar`, `path/to/...` placeholders and unexplained placeholders.
- F4 Link text: descriptive. Flag "here", "this link", "this page", bare URLs used as link text in prose, "click here", and link text that is a raw path when a title exists.
- F5 Cross-references: "For more information, see LINK." Flag "see above/below", vague references ("the docs", "elsewhere").
- F6 Bold only for UI elements and (sparingly) run-in headings/notice labels; italics for term introduction and placeholders. Flag bold used for emphasis, ALL CAPS for emphasis.
- F7 Notices: Note / Caution / Warning (and Success/Key Point etc.) used with the right severity; no stacked notices; no overuse. In RST these are `.. note::`, `.. warning::` directives; in Markdown, `**Note:**`.
- F8 Numbers and units: space between number and unit ("10 GB", "5 min"); spell out zero–nine in prose unless a measurement, version, or technical value; numerals for 10 and up; commas in numbers ≥ 1,000 in prose; do not start a sentence with a numeral; ranges written consistently.
- F9 Dates unambiguous (2026-09-20 or "September 20, 2026"), never 9/20/26.
- F10 Images and figures: alt text present and descriptive; figures introduced by a sentence.
- F11 Punctuation: em dashes without surrounding spaces (Google style), not " - " or " -- "; no semicolon chains; parentheses not overused for essential content; straight vs curly quotes consistent; commas and periods inside quotation marks (American style).
- F12 Product, language, and tool names capitalized officially (Rust, Python, NumPy, Matplotlib, Jupyter, GitHub, macOS, Cargo vs `cargo` the command).

API reference comments (Python docstrings, Rust doc comments) — https://developers.google.com/style/api-reference-comments
- A1 First sentence is a short, complete summary of what the item does or is; it ends with a period.
- A2 Method/function descriptions start with a present-tense third-person verb ("Gets", "Computes", "Returns"), not imperative ("Compute") and not "This function ...". Note: PEP 257/NumPy docstring style prefers imperative, which conflicts; report as A2 but mark severity DOMAIN if the file consistently follows NumPy/PEP 257 style.
- A3 Class/struct/module descriptions say what the thing represents, as a noun phrase or sentence; do not restate the name.
- A4 Every parameter, return value, error/exception/panic documented; parameter descriptions are noun phrases starting lowercase/uppercase consistently; units stated for physical quantities; boolean params: "True if ...; False otherwise" / "If true, ...".
- A5 Deprecations, defaults, and valid ranges stated.
- A6 Code font (backticks; in RST double backticks) for identifiers and literals inside docstrings.
- A7 All T*, F* prose rules also apply inside docstrings/doc comments (abbreviations, "e.g.", symbols, fragments, etc.).

### Severity

- HIGH: impedes comprehension or use (undefined abbreviation central to the page, missing intro, unexplained placeholder, broken step order, link text "here", wrong list type for a procedure, undocumented parameter).
- MEDIUM: clear rule violation that does not block use (title-case headings, passive voice, "e.g.", "we", future tense, bare code block, missing serial comma).
- LOW: polish (one "simply", spacing around em dash, a single "currently").
- DOMAIN: technically non-conforming, but it is standard scientific notation or a language-community docstring convention, so fixing it is a judgment call ("~" for order of magnitude in a physics statement, "→" in a reaction such as γe → γγe, imperative docstring summary per PEP 257, "we" is NOT domain in user docs).

