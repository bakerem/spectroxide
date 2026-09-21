# Documentation style work: status and resume file

Plan: `dev/PLAN_DOCS_GOOGLE_STYLE_2026-09-21.md`. Input review: `dev/audit/DOCS_GOOGLE_STYLE_REVIEW_2026-09-20.md`.
A fresh session resumes from the first unchecked box below.

## Phase checklist

- [x] Phase 0. Guardrails (2026-09-21): this file, baselines, `dev/scripts/docs_style_lint.py`. Lint is not in CI (revisit after phase 6).
- [ ] Phase 1. Abbreviations and `docs/glossary.rst`
- [ ] Phase 2. API reference gaps (one context, physics-inquisitor pass on added units and panic conditions)
- [ ] Phase 3. Procedures, code-block introductions, page openings
- [ ] Phase 4. Placeholders in `docs/cli.rst`
- [ ] Phase 5. Mechanical prose substitutions (blocked on D1 for dashes only)
- [ ] Phase 6. Record the house style in `CONTRIBUTING.md`; add the script to the `CLAUDE.md` list (20 to 21)
- [ ] Phase 7. `firas.py` keyword names (blocked on D2)
- [ ] Phase 8. Independent re-review

## Open decisions (EB)

D1 spaced em dashes, D2 British keyword names in `firas.py`, D3 code font in single-identifier headings, D4 angle brackets in
`--help`. None decided yet. Phases 1 to 4 do not depend on them.

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

## Recorded exceptions

None yet. Add one line per exception with the rule, the file, and the reason.

## Log

- 2026-09-21: phase 0 done. No source or documentation file changed.
