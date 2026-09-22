# Documentation style work: status and resume file

Plan: `dev/PLAN_DOCS_GOOGLE_STYLE_2026-09-21.md`. Input review: `dev/audit/DOCS_GOOGLE_STYLE_REVIEW_2026-09-20.md`.
A fresh session resumes from the first unchecked box below.

## Phase checklist

- [x] Phase 0. Guardrails (2026-09-21): this file, baselines, `dev/scripts/docs_style_lint.py`. Lint is not in CI (revisit after phase 6).
- [x] Phase 1. Abbreviations and `docs/glossary.rst` (2026-09-21): `abbrev-first-use` 160 to 0
- [x] Phase 2. API reference gaps (2026-09-21): all plan items; physics-inquisitor pass done, 17 corrections applied
- [x] Phase 3. Procedures, code-block introductions, page openings (2026-09-21): `heading-then-code` 28 to 0, `nb-code-no-intro` 6 to 0
- [x] Phase 4. Placeholders in `docs/cli.rst` (2026-09-21): 11 placeholders to UPPER_SNAKE_CASE with "Replace ..." sentences. `--help` left as is (D4 recommendation; reversible if EB decides otherwise)
- [x] Phase 5. Mechanical prose substitutions (2026-09-21): five commits; every lint rule reads zero
- [x] Phase 6. House style recorded in `CONTRIBUTING.md`; `CLAUDE.md` script list updated (2026-09-21). CI wiring left for EB: `docs_style_lint.py --check` exits 1 on any hit
- [x] Phase 7. `firas.py` keyword names (2026-09-21): American names, British names are deprecated aliases
- [x] Phase 8. Independent re-review (2026-09-21): run; HIGH criterion met after fixes, MEDIUM criterion NOT met (see the log)
- [x] Phase 9. MEDIUM residue (2026-09-22): lint rules `rs-noun-summary` (145 to 0) and `position-word` (9 to 0); F6, T14, L1, L2 rows of the re-review applied in Sphinx pages, Python, Rust, and the tutorial notebooks

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

The five project-wide exceptions are listed in `CONTRIBUTING.md`, section "Documentation style". Left on purpose in phase 5:

- Citation ampersands ("Chluba & Sunyaev"), "COBE/FIRAS", "RECFAST/Seager", "key/value", "NaN/Inf", "2x2/3x3": names or fixed pairs.
- Arrows for limits, reactions, atomic transitions, and swaps ("P_s→1", "He II→He I", "z↔x"); order-of-magnitude "~".
- Bold run-in labels and bold unit markers in parameter docs; `NOTE` and `DEPRECATED` comment tags; `PRODUCTION`, `DEBUG` identifiers.
- Agentless passives where no reader action depends on the agent (about 40; the plan said not to chase these).
- The title "Contributing to spectroxide" and the README heading "Contributing" (conventional).
- One British spelling inside a notebook code-cell comment (tutorial 04); code cells are not edited.
- `//` (non-doc) Rust comments are out of scope and not linted.

## CI toolchain drift (2026-09-21)

The two red runs on `main` (2026-09-16 Clippy `needless_late_init`; 2026-09-17 `black --check` on `greens.py`) were both
fixed by the next commit, and the latest `main` run (2662954) is green on all seven jobs. The cause is that `ci.yml` uses the
floating `stable` Rust toolchain and an unpinned `black`, while the local machine had Rust 1.93 and black 26.3. Before pushing
this branch, run the checks with the CI versions: Rust 1.98 (`rustup toolchain install 1.98.0` plus the `rustfmt` and `clippy`
components; `cargo +1.98.0 clippy --all-targets -- -D warnings`, with and without `--features axion`; `cargo +1.98.0 fmt
--check`) and the newest black in a scratch venv (`pip install -U black; python -m black --check python/spectroxide/`). All of
these passed on this branch at 9670179. Pinning `black` in `ci.yml` and `pyproject.toml` would remove the second failure class;
left for EB.

## Log

- 2026-09-22: phase 9 done. Two lint rules added to `docs_style_lint.py`: `rs-noun-summary` (a `fn` summary whose first word
  is not a third-person form of a verb in `IMPERATIVE_VERBS`; 145 hits, now 0) and `position-word` ("above"/"below" after a
  pointer word, after "the", after "table"/"figure"/..., or closing a clause; 9 hits, now 0; a comparison such as "below
  z = 1e4" is not matched). Verbs added to the table: suggest. The 145 summaries were rewritten by four subagents and by hand
  after the subagents stalled on API errors; every rewrite keeps the original wording after the verb ("Returns the ...",
  "Computes the ...", "Checks that ..." for tests, "Runs the ..." for Miri kernels). From the re-review: F6 (bold or caps
  for emphasis) in `docs/tutorials/index.rst` table, `kompaneets.rs`, `temporal_error_check.rs`, and the notebooks; T14
  fragments in `docs/api/axion.rst`, `docs/api/dark_photon.rst`, and the notebooks; L1 numbered-set lists in `axion.rst`,
  `src/axion.rs`, and `docs/api/solver.rst` (the two callable-property lists are now reST definition lists); L2 lead
  sentences in `_validation.py`, `greens.py` (3), `solver.py` (2), `greens_table.py`, `src/recombination.rs` (2). One new
  house rule in `CONTRIBUTING.md` (no "above"/"below" as page positions). Not done, on purpose: agentless passives (T2),
  which the plan excluded. Checks: lint all zero; Clippy, rustdoc (`-D warnings`), fmt and 3 doctests with Rust 1.98 in both
  configurations; black 26.3 and 26.5 clean; Sphinx 0 warnings; pytest 345; zero non-doc Rust lines changed; Python AST with
  docstrings stripped identical for the three package modules; notebook code cells and outputs identical to HEAD (see the
  per-notebook proof in the commit message). No second fresh re-review has been run after phase 9.

- 2026-09-21: phase 8 done. Nine fresh reviewers (no access to the plan, the first review, the lint, or git history) applied the
  same rubric. Reports: `dev/audit/docs_style_rereview_2026-09-21/`.

  | Section | HIGH | MEDIUM | LOW | DOMAIN |
  |---|---|---|---|---|
  | 1 README files | 1 | 9 | 9 | 2 |
  | 2 Contributor guides | 1 | 38 | 22 | 4 |
  | 3 Sphinx top-level | 1 | 35 | 12 | 2 |
  | 4 Sphinx API pages | 0 | 32 | 42 | 11 |
  | 5 Tutorial notebooks | 1 | 54 | 28 | 2 |
  | 6 Python group A | 2 | 31 | 21 | 8 |
  | 7 Python group B | 2 | 9 | 82 | 5 |
  | 8 Rust physics | 0 | 129 | 29 | 2 |
  | 9 Rust solver and CLI | 4 | 53 | 18 | 5 |
  | Total | 12 (first review: about 74) | 390 | 263 | 41 |

  **HIGH: all 12 fixed after the review**, so the criterion "zero HIGH" holds for the committed tree, but no second fresh pass
  has confirmed it. They were: NWA and IMEX undefined inside README code blocks; COBE undefined in the glossary; CI undefined in
  `CONTRIBUTING.md`; GF undefined in the package quick start; PDE undefined in one warning string and in the CLI help header
  (CMB too); a second five-step procedure in tutorial 05 written as one sentence (phase 3 had fixed a different sentence);
  two `greens_table.py` docstrings that named keyword arguments the builders do not have (`z_h_grid`, `n_threads`; now
  `z_injections`, `n_points`); two `src/greens.rs` functions with an undocumented panic in the μ-y transition band. The lint
  missed the abbreviation cases because it does not read code blocks or message strings.

  **MEDIUM: criterion "fewer than 50" NOT met (390).** About 130 of these are recorded exceptions (spaced em dash about 105,
  imperative Python summaries about 15, code font in headings about 16). The real residue, largest first:
  1. Rust function summaries that are noun phrases ("Photon survival probability.") instead of a verb phrase ("Returns the
     ..."): about 135. The first review counted only imperative summaries, so phase 5 did not touch these. Mechanical but needs
     reading: a getter takes "Returns", a computation takes "Computes".
  2. Bold run-in labels and bold terms in notebooks and Sphinx pages (F6): about 30. Google wants italics for a new term.
  3. Dropped-subject fragments in notebooks ("Captures the ...") (T14): about 16.
  4. Position words "above" and "below" in Rust and Python doc comments (T9): about 25. Six that phase 3 agents had introduced
     in the notebooks and docs ("The table below") were fixed.
  5. Agentless passives (T2): about 25. The plan said not to chase these.
  6. Lists in Python docstrings with no introductory sentence (L2): about 8; unnumbered-set lists written as numbered (L1): 6.
  A phase 9 that takes items 1 to 4 would bring MEDIUM under 50 outside the exceptions. Add lint rules for items 1 and 4 first.

  Checks at close: full `cargo test --release` 482 passed, 0 failed, 3 ignored (run before the phase 8 fixes; after them:
  Clippy clean in both configurations, doctests 3, `cli_integration` 4, CLI unit tests 11); pytest 345; Sphinx 0 warnings;
  rustdoc 0 warnings; lint all zero.

- 2026-09-21: phases 5 and 6 done. Phase 5 commits: third-person summaries; American spelling; word list; headings, real em
  dashes, word choices; person, tense, emphasis, symbols, link text. Subagents did the judgment rules by file group; I read
  every diff and corrected about ten slips, mostly "/" turned into "or" where "and" or an apposition was meant, and one
  sentence that lost its subject. After each commit: Sphinx 0 warnings, rustdoc 0, Clippy clean in both configurations,
  3 doctests, pytest 345, zero changed Rust code lines, Python AST identical, notebook code cells and outputs identical.
  Remaining: phase 8 (independent re-review).

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
