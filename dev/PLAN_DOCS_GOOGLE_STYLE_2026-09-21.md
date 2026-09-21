# Plan: bring the documentation into line with the Google developer documentation style guide

Date: 2026-09-21. Input: `dev/audit/DOCS_GOOGLE_STYLE_REVIEW_2026-09-20.md` (about 900 findings, about 74 high severity,
twelve repeated patterns). Status file to create at the start of work: `dev/DOCS_STYLE_STATUS.md`.

This is a plan only. Nothing in it is implemented.

## Goal and non-goals

Goal: close every high-severity finding and every mechanical medium-severity pattern, and record the exceptions, so that a
second review pass finds no high-severity items and fewer than 50 medium items.

Non-goals:

- No change to physics, numerics, or public behavior. Doc comments, docstrings, help strings, and prose only.
- No rewrite of Python docstring summaries to the third person (PEP 257 and the NumPy standard prescribe the imperative).
- No change to physics notation inside math, reactions, or order-of-magnitude statements.
- No edits to `CLAUDE.md`, `dev/`, the paper, or notebooks outside `notebooks/tutorials/`.

## Decisions needed from EB before phase 5 and phase 7

The plan proceeds without these through phase 4. Recommendations are given so work does not stall.

- **D1. Spaced em dashes (about 150).** Recommendation: keep the spaced form as a recorded house style. It is uniform, it matches
  the paper, and converting it buys nothing for the reader. Fix only the literal `---` and `--` in `README.md` and two notebooks,
  which render wrongly today.
- **D2. British keyword names in `firas.py`** (`marginalise_y`, `marginalise_mu`, `marginalise_galactic`,
  `fit_amplitude_marginalised`). Callers found: `python/tests/test_firas.py`,
  `notebooks/paper_figures/dark_photon_constraints.ipynb`, one `dev/scripts/` diagnostic. Recommendation: add American-spelled
  names as the primary API, keep the British names as aliases that emit `DeprecationWarning`, and remove them in the next minor
  release. Alternative: leave the names, fix only prose. Either is defensible; mixed spelling in prose is not.
- **D3. Code font in headings whose whole subject is one identifier** (`solve`, `run_sweep`). Recommendation: keep, record as an exception.
- **D4. Angle-bracket placeholders in `--help` output.** Recommendation: keep in `--help` (command-line convention), convert in `docs/cli.rst`.

## Hazards found while planning

- **H-1. Tests match on strings.** `tests/cli_integration.rs` has 20 lines that inspect stdout, stderr, or `contains(...)`, and four
  Python test files use `pytest.raises(match=...)` or `pytest.warns`. Any edit to help text, warning text, or error text can break
  them. Before editing a message, grep the tests for a distinctive substring of it.
- **H-2. Rust doc comments contain doctests** (3 pass today). Editing `///` blocks can break them. Run `cargo test --release --doc`
  after every Rust phase.
- **H-3. Both feature configurations must build.** `src/axion.rs` doc edits need `cargo doc --features axion` as well as the default.
- **H-4. Tutorial notebooks carry executed outputs.** Edit Markdown cells with a JSON-level script that touches only
  `cell["source"]` of Markdown cells. Do not re-execute. Never open the `.ipynb` files with a text reader. Confirm afterward that the
  diff touches no `outputs` key.
- **H-5. `notebooks/paper_figures/` is generated** by `_generate_notebooks.py`. If D2 renames keywords, change the generator's source,
  not the generated notebook.
- **H-6. Commit messages.** Phases that touch only `.md`, `.rst`, or notebook Markdown take `[skip ci]`. Phases that touch `.rs` or
  `.py` files must not, even though only comments change, because doctests and `black` run in CI.
- **H-7. The review's line numbers go stale** as soon as the first edit lands in a file. Work file by file, top to bottom within a
  file is not enough; re-locate each finding by its quotation, not its line number.
- **H-8. Reviewer error rate.** 73 of 591 quotations did not machine-match, and the section reviewers undercounted British spelling
  (they missed 26 in `firas.py`). Treat the report as a lead list, not as ground truth; each phase below starts with its own grep.

## Phases

Each phase is one commit (or one per file group where noted), independently revertible.

### Phase 0. Set up the guardrails

- [ ] Create `dev/DOCS_STYLE_STATUS.md` with the phase checklist.
- [ ] Record baselines: `python -m sphinx -b html docs docs/_build/html` warning count; `cargo doc --no-deps` warning count (default and
      `--features axion`); `cargo test --release --doc`; `pytest python/tests -q`.
- [ ] Write `dev/scripts/docs_style_lint.py`: a grep-based checker for the mechanical rules (undefined-abbreviation first use per
      file, "e.g.", "i.e.", "etc.", "via", "and/or", British `-ise`/`-our` forms, `///` summary lines that start with a bare
      imperative verb, one-line `///` docs with no final period, headings followed directly by a code block in `.md` and `.rst`,
      title-case headings). It prints counts per rule per file. Record the baseline counts in the status file. This is the
      measure of progress and the regression check; it imports nothing from spectroxide.
- [ ] Decide whether to add the lint to CI as non-blocking. Recommendation: not yet; add after phase 6.

### Phase 1. Abbreviations and a glossary (pattern 1, high severity)

- [ ] Add `docs/glossary.rst` using the Sphinx `glossary` directive: PDE, CMB, DC, BR, GF, DM, NWA, IC, CLI, FIRAS, PIXIE, IMEX,
      ODE, DI, CL, μ and y distortion, x, θ_e, ρ_e. Link it from `docs/index.rst`.
- [ ] Spell out each abbreviation at first use on every page: `README.md`, `data/cosmotherm/README.md`, both contributor guides,
      five top-level `.rst` pages, nine `docs/api/*.rst` pages, six tutorial notebooks (see H-4).
- [ ] Spell out at first use in each module docstring: 12 Python modules, 19 Rust `//!` headers, `src/bin/check_adiabatic.rs`.
- [ ] Remove definitions of abbreviations that are then never used (two "NWA" cases in `docs/api/`).
- Verification: lint rule "first use" reports zero; Sphinx build has no new warnings; `:term:` links resolve.

### Phase 2. Close the API reference gaps (pattern 6, high severity)

These are defects under any style guide. Re-verify each against the source before writing; only the first two are hand-confirmed.

- [ ] `src/bremsstrahlung.rs`: document `cosmo` on `br_emission_coefficient`; document `n_hii`, `n_heiii`, `n_heii` on `BrPrecomputed`
      with units [1/m³]; state when `br_precompute` returns `None`.
- [ ] `src/kompaneets.rs`: full `# Arguments` for `kompaneets_step_coupled_inplace` (reported 2 of 10 documented). Add a `# Panics`
      section that names the entry `assert!` guards (Pitfall #10).
- [ ] `# Errors` sections on the 10 `Result`-returning functions (`src/energy_injection.rs` table loaders first, then `src/cli.rs`,
      `src/output.rs`).
- [ ] Units or "dimensionless" on the 4 physical quantities the review lists in section 8.
- [ ] `python/spectroxide/_validation.py`: Parameters sections for the nine `warn_*` helpers.
- [ ] `python/spectroxide/cosmotherm.py`: Raises sections for `load_greens_database` and `cosmotherm_gf_distortion`.
- [ ] `examples/photon_diag.rs`: file-level `//!` summary.
- [ ] Final period on the 38 one-line Rust docs (rule A1).
- Verification: `cargo doc` (both configurations) and `cargo test --release --doc` clean; `cargo clippy -- -D warnings`; `black --check`;
  Sphinx build. Physics statements added to docs (units, panic conditions) get a physics-inquisitor pass, because a wrong unit
  in a doc comment is worse than a missing one.

### Phase 3. Procedures, code-block introductions, page openings (patterns 2, 3, 10)

- [ ] `README.md`: rewrite manual installation as a numbered list, one action per step, "Optional:" prefix where it applies; split
      the build, test, and two alternative `pip install` commands. Add one introductory sentence before each of the 6 bare blocks and
      3 bare lists or tables. Add language tags to the 2 untagged fences.
- [ ] `docs/installation.rst`: add the missing introductory paragraph; mirror the README procedure.
- [ ] `docs/api/axion.rst`: move the warning below the title and a one-sentence introduction.
- [ ] `docs/api/solver.rst` (7), other `docs/api/` pages (4), `docs/cli.rst` and `docs/rust_api.rst` (5), `CONTRIBUTING_CLAUDE.md` (3):
      one introductory sentence per bare code block or table.
- [ ] `CONTRIBUTING.md:66`: numbered list to bulleted list. Tutorial 05: the semicolon-chained five-step sentence to a numbered list.
- [ ] Tutorial notebooks: a Markdown sentence before each code cell that lacks one (H-4).
- [ ] `docs/cli.rst`: explain the 7 empty table cells or fill them ("none").
- Verification: lint rule "heading then code block" reports zero; render the README on GitHub preview; Sphinx build.

### Phase 4. Placeholders in `docs/cli.rst` (pattern 4)

- [ ] Convert `<z>`, `<val>`, `<subcommand>` and the other 8 to UPPER_SNAKE_CASE; after each block add "Replace `Z_START` with ...".
- [ ] Apply D4 to the `--help` strings. If they change, update `tests/cli_integration.rs` in the same commit (H-1).

### Phase 5. Mechanical prose substitutions (patterns 5, 7, 8, 9, 11, 12)

Do these with reviewed `sed`-style scripts, one rule per commit, so each diff is readable. Never run a blind global replace:
"via", "since", "should", and "above" each have legitimate uses.

- [ ] Rust `///` function summaries to third person (about 180). Script the first-word change from a verb table, then read the diff.
- [ ] "e.g." to "for example", "i.e." to "that is", drop "etc." by naming the items or using "such as", "via" to "through" or "with",
      "vs" to "versus" or "compared with", "and/or", "(s)" plurals in messages (H-1), position words "above"/"below" to links.
- [ ] American spelling in prose and comments (about 71 forms). Identifiers wait for D2.
- [ ] "we"/"our" to "you" or an active subject (about 20); "the user" to "you" (3); "will" to present tense (6). Passive voice: fix the
      about 50 flagged cases only where the agent matters to the reader; do not chase the rest.
- [ ] Sentence-case headings (11) and imperative task headings (12). Check `:ref:` targets and README anchor links, because heading
      text changes the generated anchors.
- [ ] Emphasis: ALL CAPS and mid-sentence bold (about 35) to plain text, or to a `.. note::` / `.. warning::` / `**Caution:**` notice
      where the emphasis carries a real hazard (the "NEVER calibrate test targets from the code" rules qualify).
- [ ] Literal `---` and `--` in `README.md` and two notebooks to "—". Apply D1 to the rest.
- [ ] Symbols for words in prose outside math (about 30): "+" to "and", "/" to "or", "~" to "about", arrows to words.
- [ ] "sanity check" to "consistency check" in `src/bin/check_adiabatic.rs:1`; "native" in `greens_table.py`.
- [ ] Link text: the 5 raw-path links in `README.md`, "paper" in `docs/api/solver.rst`.
- [ ] `numpy` to NumPy, Clippy capitalization, package names in code font (8).
- Verification after each commit: lint count for that rule is zero or equals the recorded exception count; test suites as in phase 2
  for commits that touch `.rs` or `.py`.

### Phase 6. Record the house style

- [ ] Add a "Documentation style" section to `CONTRIBUTING.md`: Google guide as the base, the four recorded exceptions (imperative
      Python summaries, physics notation, angle brackets in `--help`, code font in single-identifier headings), D1 outcome, the
      glossary rule, and how to run `dev/scripts/docs_style_lint.py`.
- [ ] Update `CLAUDE.md` `dev/scripts/` list for the new script (count goes from 20 to 21).
- [ ] Optionally wire the lint into CI as a non-blocking job.

### Phase 7. `firas.py` keyword names (only if D2 says rename)

- [ ] Add American-spelled keywords and method; keep British names as deprecated aliases with `DeprecationWarning`; passing both raises `TypeError`.
- [ ] Update `python/tests/test_firas.py` (add one test for the alias path), the `_generate_notebooks.py` source for
      `dark_photon_constraints.ipynb` (H-5), `docs/api/firas.rst`, the README if it shows the call.
- [ ] Production-code-reviewer pass before commit; this is the only phase that changes behavior.

### Phase 8. Independent re-review

- [ ] Rerun the nine-reviewer pass with the same rubric against the new commit, with fresh context and no access to this plan or
      the first report. Success: zero high-severity findings, fewer than 50 medium, every remaining DOMAIN item on the recorded
      exception list.
- [ ] Write the outcome into `dev/DOCS_STYLE_STATUS.md` and close it.

## Order and size

| Phase | Files touched | CI | Rough size |
|---|---|---|---|
| 0 | `dev/` only | skip | small |
| 1 | about 50 | runs | medium; the highest reader value |
| 2 | about 8 source files | runs | medium; needs care, not volume |
| 3 | about 15 docs files and 6 notebooks | skip | medium |
| 4 | 1 to 3 | runs if help text changes | small |
| 5 | nearly all | runs | large but mechanical; about 12 commits |
| 6 | 2 | skip | small |
| 7 | about 6 | runs | small, behavior-changing |
| 8 | none | none | review only |

Phases 1 to 3 remove about 60 of the 74 high-severity findings. If time is short, stop after phase 3 and record the rest as open.

Parallelism: phases 1, 3, and 5 fan out by file group (the same nine groups as the review), one subagent per group, because
groups share no files. Phase 2 stays in one context, since it needs the physics. Mechanical groups go to a small model; phase 2
and phase 7 do not.
