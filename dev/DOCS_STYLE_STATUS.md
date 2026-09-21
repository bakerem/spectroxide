# Documentation style work: status and resume file

Plan: `dev/PLAN_DOCS_GOOGLE_STYLE_2026-09-21.md`. Input review: `dev/audit/DOCS_GOOGLE_STYLE_REVIEW_2026-09-20.md`.
A fresh session resumes from the first unchecked box below.

## Phase checklist

- [x] Phase 0. Guardrails (2026-09-21): this file, baselines, `dev/scripts/docs_style_lint.py`. Lint is not in CI (revisit after phase 6).
- [x] Phase 1. Abbreviations and `docs/glossary.rst` (2026-09-21): `abbrev-first-use` 160 to 0
- [x] Phase 2. API reference gaps (2026-09-21): all plan items; physics-inquisitor pass done, 17 corrections applied
- [x] Phase 3. Procedures, code-block introductions, page openings (2026-09-21): `heading-then-code` 28 to 0, `nb-code-no-intro` 6 to 0
- [x] Phase 4. Placeholders in `docs/cli.rst` (2026-09-21): 11 placeholders to UPPER_SNAKE_CASE with "Replace ..." sentences. `--help` left as is (D4 recommendation; reversible if EB decides otherwise)
- [ ] Phase 5. Mechanical prose substitutions (blocked on D1 for dashes only)
- [ ] Phase 6. Record the house style in `CONTRIBUTING.md`; add the script to the `CLAUDE.md` list (20 to 22; see the log)
- [x] Phase 7. `firas.py` keyword names (2026-09-21): American names, British names are deprecated aliases
- [ ] Phase 8. Independent re-review

## Open decisions (EB)

Decided by EB on 2026-09-21: D1 keep spaced em dashes (house style; fix only literal `---` and `--`); D2 add American keyword
names in `firas.py`, keep British names as deprecated aliases; D3 keep code font in single-identifier headings. D4 (angle
brackets stay in `--help`) follows the plan's recommendation; EB did not object. Do not push; commit only.

## Baselines, 2026-09-21, commit 2662954

| Check | Command | Result |
|---|---|---|
| Sphinx | `python -m sphinx -b html docs OUT` | 0 warnings |
| rustdoc, default | `cargo doc --no-deps` | 2 warnings |
| rustdoc, axion | `cargo doc --no-deps --features axion` | 2 warnings (the same two) |
| Doctests | `cargo test --release --doc` | 3 passed |
| Python tests | `python -m pytest python/tests -q` | 331 passed, 1 skipped |

The two rustdoc warnings predate this work: the public doc of `gaunt_expc_factor` (`src/bremsstrahlung.rs:249-250`) links to the
private items `gaunt_from_expc` and `gaunt_ff_nr_fast_preln`. Phase 2 touches that file; fix them there (plain code font instead
of an intra-doc link).

## Lint baseline, 2026-09-21

Run `python dev/scripts/docs_style_lint.py` for the per-file table, `--list RULE` for each hit.

| Rule | Hits | Review's estimate | Phase |
|---|---|---|---|
| `abbrev-first-use` | 160 | not counted | 1 |
| `eg` | 27 | about 25 | 5 |
| `ie` | 6 | | 5 |
| `etc` | 14 | about 12 | 5 |
| `via` | 41 | about 25 | 5 |
| `and-or` | 1 | | 5 |
| `vs` | 21 | about 8 | 5 |
| `paren-s` | 2 | | 5 |
| `british` | 72 (25 in `firas.py`) | about 71 | 5, 7 |
| `rs-imperative-summary` | 136 | about 180 | 5 |
| `rs-oneline-no-period` | 86 | 38 | 2 |
| `heading-then-code` | 28 | about 35 | 3 |
| `nb-code-no-intro` | 6 | | 3 |
| `title-case-heading` | 9 | 11 | 5 |

Notes on the lint, so that nobody reads a count as a verdict:

- `rs-oneline-no-period` counts every one-line `///` block, including struct-field docs that end in a formula or a unit. The
  review counted 38, probably functions only. All 86 are A1 violations by the letter.
- `rs-imperative-summary` works from a verb table. A summary that starts with a verb missing from the table is not counted, which
  is the likely reason for 136 against the review's 180. Extend `IMPERATIVE_VERBS` when phase 5 finds misses.
- `abbrev-first-use` passes if the expansion is within 120 characters of the first use, or if the first use is a `:term:` role.
- `via`, `vs`, and `etc` hits include legitimate uses; phase 5 records the exception count per rule here.
- Python docstrings are excluded from the imperative rule on purpose (PEP 257; recorded exception).

## Findings for EB (behavior, not style; nothing changed in code)

- **F-DS-1 (FIXED 2026-09-21, EB chose the code fix). `--cosmology PRESET` silently dropped individual cosmology flags.**
  Now the preset is the base and each given flag overrides it; a preset with no override is returned bit-identical. Checked end
  to end: `--cosmology planck2018 --t-cmb 3.0` gave the same μ and y as no override under the old code, and a different μ
  (1.3878e-5 against 1.3968e-5) under the new code. Original finding:
  `--cosmology PRESET` silently drops individual cosmology flags. `build_cosmology` (`src/cli.rs`) returns the preset
  before it reads `--h`, `--t-cmb`, and the rest. `docs/cli.rst` says "Individual parameters override the selected preset",
  which is false. Found by reading the code, confirmed by the physics-inquisitor; not run. Decide: fix the code (apply
  overrides on top of the preset, or reject the combination) or fix the sentence. The new `build_cosmology` doc comment states
  the real behavior.
- **F-DS-2. `warn_table_z_density` stays silent for a table with no point in `3e4 <= z <= 2e5`**, the worst case it exists for
  (`if in_transition.size > 0` guard). The docstring now says so.
- **F-DS-3. `warn_convolution_resolution` skips the check when `z_min == 0`**, which `validate_z_range` accepts, so nothing reports
  a coarse grid there. The inline comment "Will be caught by validate_z_range" is wrong for that case.
- **F-DS-4. `cosmotherm_gf_distortion` loads the database before it checks `scenario`**, so a bad scenario name with no database
  raises `FileNotFoundError`, not `ValueError`. A missing `params` key raises `KeyError` from inside the convolution.
- **F-DS-5. Stale doc fixed:** `DcbrCoupling::dem_drho_eq` told callers to pass an empty slice for the legacy behavior; an empty
  slice panics at the entry asserts. The doc now says to pass zeros. The claim that zeros reproduce the legacy Picard behavior
  follows from how the term enters the Jacobian; not run.

- **F-DS-6. `dy_max` was documented as the wrong quantity (fixed).** Four sites said "maximum fractional change in ln(1+z) per
  step". The code gives dz = dy_max · t_C H (1+z)/θ_e, so θ_e Δτ = dy_max: it caps the Compton-y increment. Δln(1+z) per
  step is 120 × dy_max at z = 1e4 and 0.01 × dy_max at z = 1e6 (the two agree only near z = 1e5). Verified by a fresh-context
  claim-verifier (algebra plus an independent Python table). Fixed in `src/solver.rs` (two doc comments, one code comment) and
  the `--dy-max` help line in `src/cli.rs`. `docs/cli.rst` and `python/spectroxide/solver.py` were already right. No
  numerical behavior changes. If the paper describes `dy_max`, check its wording.

## Recorded exceptions

None yet. Add one line per exception with the rule, the file, and the reason.

## Log

- 2026-09-21: phase 7 done. `marginalize_y`, `marginalize_mu`, `marginalize_gbb` (the plan missed this one),
  `marginalize_galactic`, and `fit_amplitude_marginalized` are the API. A decorator maps the British keyword names with a
  `DeprecationWarning` and raises `TypeError` if both spellings are passed; it maps only names the wrapped method accepts.
  `fit_amplitude_marginalised` is a warning alias. 14 new tests (alias values equal, warning points at the caller, names stay
  out of signatures). pytest 345 passed. The production-code-reviewer found five defects (all fixed), no numerical change.
  H-5 was wrong for `dark_photon_constraints.ipynb`: it is maintained by hand, not generated, so its one call was changed in
  the raw file (cell 9 source only; outputs identical). Stale: `dev/audit/TEST_PROVENANCE.md` lists three renamed tests;
  regenerate with `dev/scripts/build_test_provenance.py` when its fragments are next built. Remove the aliases in the next
  minor release.
- 2026-09-21: phase 5, first two rules committed: third-person Rust summaries (141, `rs-imperative-summary` 136 to 0 after
  adding five verbs to the table) and American spelling in prose (`british` 72 to 0).

- 2026-09-21: CHECKPOINT after phase 4. Branch `docs-google-style`, five phase commits plus the `dy_max` doc fix. Next: phase 5
  (about 12 mechanical commits; start with `python dev/scripts/docs_style_lint.py --list rs-imperative-summary`), which needs
  only D1 for the dash item. Phase 7 needs D2. Safe to clear context here.

- 2026-09-21: phase 4 done. `docs/cli.rst` only; Sphinx 0 warnings. `src/cli.rs` help text and `tests/cli_integration.rs` untouched.

- 2026-09-21: phase 3 done. README manual installation is a numbered procedure (a clone step added; `docs/installation.rst`
  mirrors it and now says `cargo test --release`). Introductory sentences before every bare block in the README, the
  contributor guides, the `.rst` pages, and the notebooks (seven new Markdown cells, inserted with the new `--insert` mode of
  `nb_md_replace.py`; code cells and outputs proven identical to HEAD). `docs/api/axion.rst` reordered, `CONTRIBUTING.md`
  requirement list bulleted, seven empty `docs/cli.rst` default cells filled from `src/cli.rs`. One agent sentence described a
  plot wrongly (notebook 05, dark-photon plot) and was corrected by hand after reading the code cell. Known limit of
  `nb_md_replace.py`: a cell whose `source` is one JSON string, not a line list (tutorial 06 Summary), breaks multi-line
  replace; the script refuses to write in that case. Not checked: the README render on GitHub.
- 2026-09-21: phase 2 done. Checks: rustdoc now 0 warnings in both configurations (the two baseline warnings are fixed), Clippy
  clean in both, 3 doctests, pytest 331, fmt and black clean, zero changed Rust code lines. The physics-inquisitor found 17
  wrong or incomplete statements in the first draft (K_BR called a rate, which is wrong by x³; `base_factor` formula missing
  exp(−xφ); He⁺ density formula; "all per Thomson time" on `DcbrCoupling`; θ_e time-centering; photon conservation only up to
  edge flux; NaN return on a degenerate bordered system; several missing error paths). All applied. The review's own guesses
  were also wrong in places (it said the table loaders reject unsorted z; they sort), as H-8 predicted.
  `rs-oneline-no-period` 86 to 0.

- 2026-09-21: phase 1 done. Eight subagents, one per file group, then a review pass by hand. Checks equal the baseline: Sphinx 0
  warnings (so every `:term:` link resolves), rustdoc 2 warnings in both configurations, Clippy clean in both, 3 doctests,
  pytest 331 passed, `cargo fmt --check` and `black --check` clean. Proof that no code changed: zero changed Rust lines outside
  `///` and `//!`; the Python AST with docstrings stripped is identical to HEAD for all 12 modules; in the notebooks only
  Markdown cells differ (checked cell by cell against HEAD).
  - Convention used: an abbreviation that a file uses once or twice is spelled out and never introduced; otherwise
    "long form (ABBR)" at first use, with `:term:` in `docs/*.rst`. Headings are left alone and the first body sentence carries
    the expansion.
  - Three lint defects found and fixed during the phase: expansion patterns did not match across line breaks; `:term:` roles
    were blanked with inline code; headings counted as first use. `docs/glossary.rst` is excluded from the lint.
  - New helper `dev/scripts/nb_md_replace.py` edits one Markdown cell at the raw-text level and proves nothing else changed
    (notebook 05 mixes escaped and raw non-ASCII, so a JSON load-and-dump rewrites output lines). Phase 6 must add two scripts
    to the `CLAUDE.md` list, not one (20 to 22).
  - One heading changed: tutorial 06, "4. Convolve with a DM-decay heating rate" to "... dark matter decay ...". No source
    links to its anchor.

- 2026-09-21: phase 0 done. No source or documentation file changed.
