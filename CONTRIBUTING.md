# Contributing to spectroxide

spectroxide welcomes contributions — new energy injection scenarios, improved physics, better validation, and bug fixes. This guide explains how to contribute effectively, with particular attention to AI-assisted development, which is how most contributions (including the original codebase) are likely to be written.

## The problem this guide solves

spectroxide is a numerical physics code. Unlike typical software, where "it compiles and tests pass" is a strong signal of correctness, physics code can be *silently wrong*. During the development of spectroxide, four serious bugs passed a large automated test suite. Every one was caught by a human applying physical reasoning, not by automated tests. The most dangerous failure mode was an LLM writing tests that verified the code against *itself* — asserting whatever the (buggy) code produced.

This guide and its companion file exist to prevent you from repeating those mistakes.

## Two files, two audiences

Use the following table to find the right file for your role:

| File | Audience | Purpose |
|------|----------|---------|
| `CONTRIBUTING.md` (this file) | You, the human contributor | Explains the philosophy, workflow, and expectations |
| `CONTRIBUTING_CLAUDE.md` | Your LLM | Technical context to include in your LLM's system prompt |

The separation is deliberate. You need to understand *why* the project works a certain way. Your LLM needs to know *what* to do and *what not to do*. These are different documents for different readers.

## Workflow

### Set up your LLM

1. **Include `CONTRIBUTING_CLAUDE.md` as context.** In Claude Code, this happens automatically through `CLAUDE.md`. For other tools (including ChatGPT, Copilot, and Cursor), paste the contents of `CONTRIBUTING_CLAUDE.md` into your system prompt or project instructions.

2. **Tell the LLM what you're building and what the correct answer is.** For example: *"Add a scenario for evaporating primordial black holes. The heating rate is given by .... . In the y-era, I expect y = f(M_PBH) to match their Figure 3."*

3. **Review the LLM's test targets.** Before accepting any test the LLM writes, ask yourself: *"Where does this expected value come from? Could I defend it in a paper?"* If the answer is "the LLM ran the code and used the output," reject the test.

4. **Run the full test suite.** `cargo test --release` must pass.

### Add a new energy injection scenario (the most common contribution)

This is the contribution most likely to benefit from LLM assistance. The mechanical steps are:

1. Add an enum variant to `InjectionScenario` in `src/energy_injection.rs`
2. Implement match arms for all required methods (see `CONTRIBUTING_CLAUDE.md` for the full list)
3. Wire into the command-line interface (CLI) in `src/main.rs`
4. Write integration tests with independently derived targets
5. Add Python support in `python/spectroxide/solver.py`
6. Add or update a tutorial notebook demonstrating the scenario

Your LLM can handle steps 1-3 and 5 reliably. Step 4 is where you must be actively involved — the test targets come from your physics knowledge, not from the code. Step 6 benefits from human judgment about what's pedagogically useful.

### Modify solver physics

Changes to the core solver (`kompaneets.rs`, `double_compton.rs`, `bremsstrahlung.rs`, `electron_temp.rs`, `solver.rs`) require extra care:

- **Read the existing code first.** These modules encode subtle numerical choices (for example, backward Euler for double Compton (DC) and bremsstrahlung (BR) instead of Crank-Nicolson to avoid amplification instability). Ask your LLM to explain the existing approach before modifying it.
- **Check limiting cases.** Does your change preserve mu = 1.401 * Delta_rho/rho in the deep mu-era? Does it preserve energy conservation? Does it maintain stability at z > 10^6?
- **Run convergence tests.** `cargo test --release convergence` exercises grid and timestep convergence.

## Submit a pull request

All contributions go through pull requests to `main`. Here's the process:

### Before you open a PR

1. **Fork the repository** and create a feature branch (for example, `add-pbh-evaporation`).
2. **Run the full test suite locally**: `cargo test --release`. All existing tests must pass. Do not skip tests or mark them `#[ignore]` to get a green build.
3. **Run formatting and linting**: `cargo fmt` and `cargo clippy --all-targets -- -D warnings`. Continuous integration (CI) rejects unformatted code.
4. **If you modified Python code**: from `python/`, run `black spectroxide/` and `pytest tests/`.

### What your PR must include

Every PR that adds or modifies physics code must include:

- **Tests with independently justified targets.** Each test comment should state where the expected value comes from (such as "Eq. 15 of Chluba 2015", "y-era limit: y = drho/(4*rho)", or "dimensional analysis: K_BR is dimensionless"). A test that asserts a value without justification is asked to add one during review.

- **A dimensional analysis check** for any new rate coefficient or physical formula. This can be a comment in the code or a note in the PR description showing the units work out.

- **Energy conservation verification** for new injection scenarios: `mu/1.401 + 4y + 4*DeltaT/T = Delta_rho/rho` to within a few percent.

- **No new crate dependencies.** The zero-dependency constraint is a hard rule, not a preference. If you think an exception is warranted, open an issue to discuss before implementing.

### What your PR should include (when applicable)

- **Cross-validation of partial differential equation (PDE) versus Green's function** for new injection scenarios where the GF is applicable (simple injection histories). Agreement within about 5% is expected.
- **A notebook or script** demonstrating the new feature, especially for new injection scenarios.
- **Updated docstrings** on any new public API (enum variants, methods).

### PR description template

Your PR description should include:

```
## What this adds/changes
[1-3 sentences]

## Physics basis
[Reference to the paper/equation this implements. For new scenarios: what is the
heating rate or source term, and what regime is it valid in?]

## Test targets and their sources
[List each new test and where its expected value comes from. Example:
- test_pbh_y_era: y = 2.5e-5, from Eq. 12 of Acharya+ (2020) with M = 1e13 g
- test_pbh_mu_era: mu/drho = 1.401, analytic mu-era limit
- test_pbh_energy: energy conservation < 3%]

## Checklist
- [ ] `cargo test --release` passes
- [ ] `cargo fmt` and `cargo clippy` clean
- [ ] No new crate dependencies
- [ ] Test targets justified from independent sources
- [ ] Dimensional analysis of new rate coefficients
- [ ] Energy conservation verified
```

### Review process

PRs are reviewed for:

1. **Physical correctness** — Are the equations right? Are the test targets independently justified? Do dimensions check out?
2. **Numerical soundness** — Does the implementation respect the solver's conventions (Thomson time normalization, perturbative T_e, backward Euler for stiff terms)?
3. **Code quality** — Does it follow existing patterns? Is it tested? Does CI pass?

This project does not merge code where test targets cannot be traced to an independent source. This is the one rule that is never relaxed, because it is the one that would have prevented every major bug in the project's history.

### CI pipeline

The GitHub Actions CI runs automatically on every PR:

- **Rust**: build, unit tests, science suite, convergence tests, doc tests, Clippy, format check (Ubuntu + macOS)
- **Python**: install, import tests, pytest, black format check
- **Docs**: Sphinx and rustdoc build
- **Coverage**: uploaded to Codecov

All checks must pass before merge. If CI fails, fix the issue — do not ask for the check to be skipped.

### Commit messages

Use a short `type: subject` style — typical types are `fix:`, `docs:`, `polish:`, `chore:`, `feat:`. Append `[skip ci]` to commits that touch only documentation, the paper, or other non-code files.

## Documentation style

The documentation follows the
[Google developer documentation style guide](https://developers.google.com/style/highlights).
This covers the README files, the Sphinx pages in `docs/`, the tutorial notebooks, Python
docstrings, Rust doc comments, and help text. The main rules are:

- Address the reader as "you". Use the active voice and the present tense.
- Spell out each abbreviation at its first use in every file, then use the abbreviation.
  In `docs/*.rst`, link the first use to the glossary: ``partial differential equation
  (:term:`PDE`)``. If you add an abbreviation, add it to `docs/glossary.rst`.
- Introduce every code block, table, and list with a sentence. Use numbered lists for
  procedures, with one action per step.
- Use sentence case for headings. Start a task heading with a verb ("Install the package").
- Write "for example", "that is", "through", and "versus", not `e.g.`, `i.e.`, `via`, and
  `vs`. Do not write `etc.`; name the items.
- Use American spelling.
- Write placeholders in UPPER_SNAKE_CASE and explain each one after the code block.
- Do not use ALL CAPS or bold for emphasis. For a real hazard, use a **Note:**, **Caution:**,
  or **Warning:** notice.
- Start a Rust function summary with a third-person verb ("Computes", "Returns"). State the
  units of every physical quantity, and document every `Err`, `None`, and panic condition.
- Do not use "above" or "below" to point at a place on the page. Name the thing instead:
  "see `update_equilibrium`", "the following table".

The project departs from the guide in five places, on purpose:

- Python docstring summaries use the imperative ("Compute the ..."), as PEP 257 and the
  NumPy docstring standard prescribe.
- Physics notation stays in prose: "z ~ 10⁶", "γe → γγe", "DC/BR", "μ/y", "×".
- The `--help` output keeps angle-bracket placeholders (`--z-start <z>`), the
  command-line convention. `docs/cli.rst` uses UPPER_SNAKE_CASE.
- A heading whose whole subject is one identifier keeps the code font (`solve`, `run_sweep`).
- The em dash has a space on each side, to match the paper.

To check your changes, run the style checker. It counts hits per rule per file, and every
count must stay zero (`--check` exits with status 1 otherwise):

```bash
python dev/scripts/docs_style_lint.py
```

To list the hits for one rule, add `--list RULE`. Replace `RULE` with a rule name from the
table header, for example `abbrev-first-use`. The checker is a set of heuristics, so a hit is a lead and not a
verdict. To edit a Markdown cell of an executed notebook without touching its outputs, use
`python dev/scripts/nb_md_replace.py`.

## Resources

- **Paper**: Baker, Liu & Mishra-Sharma (2026), Sec. 6 documents the AI development process and failure modes
- **Tutorial notebooks**: `notebooks/tutorials/` — start with `01_getting_started.ipynb`
- **CosmoTherm**: Chluba (2012), the reference implementation used for validation
- **Chluba & Sunyaev (2012)**, MNRAS 419, 1294 — primary reference for the equations
- **Chluba (2013)**, MNRAS 434, 352 — Green's function formalism
