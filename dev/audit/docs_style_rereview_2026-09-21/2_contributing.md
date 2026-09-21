# Documentation review: CONTRIBUTING.md and CONTRIBUTING_CLAUDE.md

Reviewed against the Google developer documentation style guide (fetched and confirmed
live: highlights, headings, word-list, code-samples, placeholders, link-text, lists,
dashes, abbreviations, text-formatting). Rules not confirmed on the live guide (for
example, a mandatory language tag on fenced code blocks) are excluded rather than
reported, per the review brief.

## Summary table

| File | HIGH | MEDIUM | LOW | DOMAIN | Total |
|---|---|---|---|---|---|
| CONTRIBUTING.md | 1 | 19 | 8 | 2 | 30 |
| CONTRIBUTING_CLAUDE.md | 0 | 19 | 14 | 2 | 35 |
| **Combined** | **1** | **38** | **22** | **4** | **65** |

## Systematic patterns

1. **Spaced em dash, confirmed non-conforming.** Google's dash page states "Don't put a
   space before or after" an em dash. Both files use `" — "` throughout (11 instances in
   CONTRIBUTING.md, 14 in CONTRIBUTING_CLAUDE.md — 25 total). CONTRIBUTING.md itself
   documents this as one of five deliberate departures from the guide ("The em dash has a
   space on each side, to match the paper," line 167), so the pattern is intentional
   project policy, not an oversight. Reported anyway because the brief asks for real
   violations of the confirmed guide; the deliberate-exception status is a mitigating
   fact, not a reason to omit it.
2. **"CI" never spelled out.** Both files use "CI" (continuous integration) repeatedly —
   6 times in CONTRIBUTING.md (including a whole "CI pipeline" heading), 4 in
   CONTRIBUTING_CLAUDE.md — without ever writing "continuous integration (CI)." This is a
   direct inconsistency with the house rule both files declare: CONTRIBUTING.md's own
   "Documentation style" section says "Spell out each abbreviation at its first use in
   every file" (line 144).
3. **Ampersand in citations.** "Baker, Liu & Mishra-Sharma," "Chluba & Sunyaev," "Chluba,
   Ravenni & Bolliet" appear 5 times combined. Google's text-formatting page: "Don't use
   ampersands (&) as conjunctions or shorthand for and. Use and instead," with the only
   exception being UI elements/menu names — citations don't qualify.
4. **Procedural steps bundle more than one action.** "Fork the repository and create a
   feature branch" (CONTRIBUTING.md:61), "Run `cargo fmt` and `cargo clippy`"
   (CONTRIBUTING.md:63), "run `black` ... and `pytest`" (CONTRIBUTING.md:64). This
   conflicts with the same "Documentation style" section's own stated rule: "numbered
   lists for procedures, with one action per step" (line 147).
5. Physics notation in prose (`z ~ 10^6`, `~0.003`, `γe → γγe`-style reaction arrows
   elsewhere in the corpus) is classified DOMAIN per the rubric's own worked example and
   CONTRIBUTING.md's declared exception list (line 163); it is not double-counted as a
   plain violation.

---

## Findings: CONTRIBUTING.md

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 63 | T12 | HIGH | "CI rejects unformatted code." | Spell out at first use: "Continuous integration (CI) rejects unformatted code." Recurs undefined at lines 117, 121, 123, 130, 134. |
| 3 | T2 | MEDIUM | "which is how most contributions (including the original codebase) are likely to be written" | Name the actor: "which most contributors write with LLM assistance, including the original codebase." |
| 7 | T2 | MEDIUM | "Every one was caught by a human applying physical reasoning, not by automated tests." | "A human applying physical reasoning caught every one; no automated test did." |
| 70 | T2 | MEDIUM | "A test that asserts a value without justification is asked to add one during review." | A test cannot be "asked" anything — name the reviewer: "During review, add a justification for any test that asserts a value without one." |
| 113 | T2 | MEDIUM | "PRs are reviewed for:" | "Reviewers check PRs for:" |
| 80 | T12 | MEDIUM | "versus Green's function** for new injection scenarios where the GF is applicable" | "Green's function" is spelled out but never tagged "(GF)" before "GF" is used later in the same sentence. Add the parenthetical at first use: "versus the Green's function (GF)." |
| 61 | L4 | MEDIUM | "**Fork the repository** and create a feature branch (for example, `add-pbh-evaporation`)." | Split into two steps: one to fork, one to create the branch. |
| 63 | L4 | MEDIUM | "**Run formatting and linting**: `cargo fmt` and `cargo clippy --all-targets -- -D warnings`." | Split into two steps, one per command. |
| 64 | L4 | MEDIUM | "**If you modified Python code**: from `python/`, run `black spectroxide/` and `pytest tests/`." | Split into two steps, one per command. |
| 3 | F11 | MEDIUM | "spectroxide welcomes contributions — new energy injection scenarios" | Remove the spaces around the dash: "contributions—new energy". |
| 7 | F11 | MEDIUM | "asserting whatever the (buggy) code produced." (preceded by "verified the code against *itself* — asserting") | Remove surrounding spaces. |
| 45 | F11 | MEDIUM | "Step 4 is where you must be actively involved — the test targets come from your physics knowledge" | Remove surrounding spaces. |
| 115 | F11 | MEDIUM | "**Physical correctness** — Are the equations right?" | Remove surrounding spaces. |
| 116 | F11 | MEDIUM | "**Numerical soundness** — Does the implementation respect the solver's conventions" | Remove surrounding spaces. |
| 117 | F11 | MEDIUM | "**Code quality** — Does it follow existing patterns?" | Remove surrounding spaces. |
| 130 | F11 | MEDIUM | "If CI fails, fix the issue — do not ask for the check to be skipped." | Remove surrounding spaces. |
| 134 | F11 | MEDIUM | "Use a short `type: subject` style — typical types are `fix:`, `docs:`" | Remove surrounding spaces. |
| 184 | F11 | MEDIUM | "**Tutorial notebooks**: `notebooks/tutorials/` — start with `01_getting_started.ipynb`" | Remove surrounding spaces. |
| 186 | F11 | MEDIUM | "**Chluba & Sunyaev (2012)**, MNRAS 419, 1294 — primary reference for the equations" | Remove surrounding spaces. |
| — | F11 | MEDIUM | (1 more instance not listed above: line 187. Total spaced em dashes in this file: 11.) | Same fix throughout. |
| 82 | T12 | LOW | "**Updated docstrings** on any new public API (enum variants, methods)." | Spell out at first use: "public application programming interface (API)." |
| 90 | F3 | LOW | "[1-3 sentences]" | Bracket-style placeholders in the PR template aren't UPPER_SNAKE_CASE. Acceptable as a fill-in-the-blank template convention, but if treated as a formal placeholder, rename and explain, for example `SUMMARY` — "Replace `SUMMARY` with 1–3 sentences." |
| 93 | F3 | LOW | "[Reference to the paper/equation this implements. For new scenarios: what is the" | Same as above. |
| 97 | F3 | LOW | "[List each new test and where its expected value comes from. Example:" | Same as above. |
| 125 | T11 | LOW | "format check (Ubuntu + macOS)" | "+" stands in for "and": "(Ubuntu and macOS)." |
| 161 | T9 | LOW | "as PEP 257 and the" (full sentence spans lines 161–162: "Python docstring summaries use the imperative (\"Compute the ...\"), as PEP 257 and the NumPy docstring standard prescribe.") | If causal, use "because": "...because PEP 257 and the NumPy docstring standard prescribe it." |
| 183 | T11 | LOW | "**Paper**: Baker, Liu & Mishra-Sharma (2026), Sec. 6 documents the AI development process and failure modes" | "Baker, Liu, and Mishra-Sharma (2026)." |
| 186 | T11 | LOW | "**Chluba & Sunyaev (2012)**, MNRAS 419, 1294" | "Chluba and Sunyaev (2012)." |
| 98 | DOMAIN | DOMAIN | "from Eq. 12 of Acharya+ (2020) with M = 1e13 g" | Astrophysics "et al." shorthand ("Acharya+"); standard field convention, judgment call whether to spell out "Acharya et al." |
| 163 | DOMAIN | DOMAIN | "Physics notation stays in prose: \"z ~ 10⁶\", \"γe → γγe\", \"DC/BR\", \"μ/y\", \"×\"." | Self-declared, matches the rubric's own worked DOMAIN example; no fix needed. |

## Findings: CONTRIBUTING_CLAUDE.md

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 45 | H2 | MEDIUM | "## How to add a new energy injection scenario" | Google: "For a task-based heading, start with a bare infinitive" (for example "Create an instance," not a "how to" phrasing). Rename to "Add a new energy injection scenario." |
| 3 | T2 | MEDIUM | "This file is designed to be included as context when using an LLM (including Claude and GPT) to develop features for spectroxide." | Name the actor: "Include this file as context when you use an LLM ... to develop features for spectroxide." |
| 5 | F5 | MEDIUM | "See Sec. 6 of the paper (Baker, Liu & Mishra-Sharma, 2026) for the full story." | "The paper" is a vague, unlinked reference. Add a link or the paper's title: "For more information, see Sec. 6 of [paper title]." |
| 25 | T12 | MEDIUM | "cargo clippy --all-targets -- -D warnings # CI rejects warnings" | Spell out at first use: "continuous integration (CI)." Recurs undefined at lines 26, 162, 164. |
| 29 | F2 / L2 | MEDIUM | "Python package:" | Not a complete sentence introducing the code block. Google: precede a code sample with "an introductory sentence." Fix: "Install the Python package with:" |
| 36 | F11 | MEDIUM | "Each physical process has its own file — `kompaneets.rs`, `double_compton.rs`" | Remove surrounding spaces around the dash. |
| 67 | F11 | MEDIUM | "`name(&self)` — string identifier for CLI/output" | Remove surrounding spaces. |
| 68 | F11 | MEDIUM | "`validate(&self)` — parameter validation, return `Err(String)` for invalid inputs" | Remove surrounding spaces. |
| 69 | F11 | MEDIUM | "`heating_rate(&self, z, cosmo)` — energy injection rate d(Deltarho/rho)/dt [1/s]." | Remove surrounding spaces. |
| 70 | F11 | MEDIUM | "`photon_source_rate(&self, x, z, cosmo)` — frequency-dependent photon source dn/dt [1/s]." | Remove surrounding spaces. |
| 71 | F11 | MEDIUM | "`has_photon_source(&self)` — return true if `photon_source_rate` is non-zero" | Remove surrounding spaces. |
| 72 | F11 | MEDIUM | "`refinement_zones(&self)` — return `Vec<RefinementZone>` for grid adaptation" | Remove surrounding spaces. |
| 73 | F11 | MEDIUM | "`depletion_rate(&self, x, z, cosmo)` — photon removal rate [1/s]." | Remove surrounding spaces. |
| 74 | F11 | MEDIUM | "`heating_rate_per_redshift(&self, z, cosmo)` — rate per unit redshift (used by Green's function)." | Remove surrounding spaces. |
| 131 | F11 | MEDIUM | "is the canonical source — do not paraphrase it from memory." | Remove surrounding spaces. |
| — | F11 | MEDIUM | (4 more instances not listed above: lines 168, 169, 170, 171. Total spaced em dashes in this file: 14.) | Same fix throughout. |
| 5 | F5 mitigation / T11 | LOW | "See Sec. 6 of the paper (Baker, Liu & Mishra-Sharma, 2026) for the full story." | Ampersand stands for "and": "Baker, Liu, and Mishra-Sharma." |
| 5 | F7 | LOW | "> **Why this file exists.** spectroxide is a numerical physics code..." | Blockquote-with-bold-lead is used as an ad hoc notice; the rest of the corpus's convention (per CONTRIBUTING.md's own rule) is `**Note:**`/`**Caution:**`/`**Warning:**`. Use one of those labels or drop the blockquote. |
| 7 | F7 | LOW | "> **Canonical sources.** `CLAUDE.md` (loaded automatically by Claude Code) is the authoritative reference..." | Same as above. |
| 23 | F2 | LOW | "cargo test --release                      # Always use --release; some tests are slow in debug" | 94 characters; Google's code-samples page: "Wrap lines at 80 characters." Shorten the comment or move it above the line. |
| 43 | F8 | LOW | "430+ Rust tests across 8 files" | Spell out the single-digit count in prose: "eight files." |
| 43 | T11 | LOW | "430+ Rust tests across 8 files" | "+" stands in for "or more" (lower-confidence rule: Google's symbols guidance doesn't explicitly cover "+", included per the rubric's T11 list). Consider "more than 430 Rust tests." |
| 49 | H3 | LOW | "### Step 1: Add the variant to `InjectionScenario` in `src/energy_injection.rs`" | Two code-font identifiers in one heading, beyond the narrower exception CONTRIBUTING.md declares ("a heading whose whole subject is one identifier," line 166). Google's own rule permits code font in a heading if paired with a descriptive noun, which this heading already has, so this is a judgment call rather than a clear violation. |
| 76 | H3 | LOW | "### Step 3: Wire into CLI (`src/cli.rs`)" | Same judgment call as above. |
| 123 | T12 | LOW | "the PDE and the Green's function (GF) should agree within about 5%." | "GF" is defined here but never used again in the file. Either drop the parenthetical or use "GF" consistently afterward. |
| 133 | F8 | LOW | "Finite-difference error (~0.003) is 1000x the physical signal (~1e-5)." | Number ≥1,000 lacks the required comma: "1,000x" or "1,000 times." |
| 159 | T13 | LOW | "**Do not use ad hoc numerical fixes.**" | Latin phrase where plain English exists: "Do not use improvised numerical fixes." |
| 129 | L1 | LOW | "## Numerical pitfalls you must know about" (numbered list, items 1–5, none of which is sequential — order doesn't matter) | Google: numbered lists are for "ordered steps, phases, or priorities" where sequence matters; bulleted lists for non-sequential sets. Judgment call, since the numbers double as cross-reference IDs matching CLAUDE.md's own numbered pitfall list. |
| 168 | T11 | LOW | "Chluba & Sunyaev (2012), MNRAS 419, 1294 — CosmoTherm paper" | "Chluba and Sunyaev (2012)." |
| 171 | T11 | LOW | "Chluba, Ravenni & Bolliet (2020), MNRAS 492, 177 — BR Gaunt factor (BRpack)" | "Chluba, Ravenni, and Bolliet (2020)." |
| 13 | DOMAIN | DOMAIN | "from z ~ 10^6 to z ~ 100" | Order-of-magnitude tilde in a physics statement; matches the rubric's own DOMAIN example. |
| 133 | DOMAIN | DOMAIN | "Finite-difference error (~0.003) is 1000x the physical signal (~1e-5)." | Same tilde convention. |
