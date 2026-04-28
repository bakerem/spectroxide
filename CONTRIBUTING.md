# Contributing to spectroxide

spectroxide welcomes contributions — new energy injection scenarios, improved physics, better validation, and bug fixes. This guide explains how to contribute effectively, with particular attention to AI-assisted development, which is how most contributions (including the original codebase) are likely to be written.

## The problem this guide solves

spectroxide is a numerical physics code. Unlike typical software, where "it compiles and tests pass" is a strong signal of correctness, physics code can be *silently wrong*. During the development of spectroxide, four serious bugs passed a large automated test suite. Every one was caught by a human applying physical reasoning, not by automated tests. The most dangerous failure mode was an LLM writing tests that verified the code against *itself* — asserting whatever the (buggy) code produced.

This guide and its companion file exist to prevent you from repeating those mistakes.

## Two files, two audiences

| File | Audience | Purpose |
|------|----------|---------|
| `CONTRIBUTING.md` (this file) | **You**, the human contributor | Explains the philosophy, workflow, and expectations |
| `CONTRIBUTING_CLAUDE.md` | **Your LLM** | Technical context to include in your LLM's system prompt |

The separation is deliberate. You need to understand *why* we do things a certain way. Your LLM needs to know *what* to do and *what not to do*. These are different documents for different readers.

## Workflow

### Setting up your LLM

1. **Include `CONTRIBUTING_CLAUDE.md` as context.** In Claude Code, this happens automatically via `CLAUDE.md`. For other tools (ChatGPT, Copilot, Cursor, etc.), paste the contents of `CONTRIBUTING_CLAUDE.md` into your system prompt or project instructions.

2. **Tell the LLM what you're building and what the correct answer is.** For example: *"Add a scenario for evaporating primordial black holes. The heating rate is given by .... . In the y-era, I expect y = f(M_PBH) to match their Figure 3."*

3. **Review the LLM's test targets.** Before accepting any test the LLM writes, ask yourself: *"Where does this expected value come from? Could I defend it in a paper?"* If the answer is "the LLM ran the code and used the output," reject the test.

4. **Run the full test suite.** `cargo test --release` must pass.

### Adding a new energy injection scenario (the most common contribution)

This is the contribution most likely to benefit from LLM assistance. The mechanical steps are:

1. Add an enum variant to `InjectionScenario` in `src/energy_injection.rs`
2. Implement match arms for all required methods (see `CONTRIBUTING_CLAUDE.md` for the full list)
3. Wire into the CLI in `src/main.rs`
4. Write integration tests with independently derived targets
5. Add Python support in `python/spectroxide/solver.py`
6. Add or update a tutorial notebook demonstrating the scenario

Your LLM can handle steps 1-3 and 5 reliably. Step 4 is where you must be actively involved — the test targets come from your physics knowledge, not from the code. Step 6 benefits from human judgment about what's pedagogically useful.

### Modifying solver physics

Changes to the core solver (`kompaneets.rs`, `double_compton.rs`, `bremsstrahlung.rs`, `electron_temp.rs`, `solver.rs`) require extra care:

- **Read the existing code first.** These modules encode subtle numerical choices (e.g., backward Euler for DC/BR instead of Crank-Nicolson to avoid amplification instability). Ask your LLM to explain the existing approach before modifying it.
- **Check limiting cases.** Does your change preserve mu = 1.401 * Delta_rho/rho in the deep mu-era? Does it preserve energy conservation? Does it maintain stability at z > 10^6?
- **Run convergence tests.** `cargo test --release convergence` exercises grid and timestep convergence.

## Submitting a pull request

All contributions go through pull requests to `main`. Here's the process:

### Before you open a PR

1. **Fork the repository** and create a feature branch (e.g., `add-pbh-evaporation`).
2. **Run the full test suite locally**: `cargo test --release`. All existing tests must pass. Do not skip tests or mark them `#[ignore]` to get a green build.
3. **Run formatting and linting**: `cargo fmt` and `cargo clippy --all-targets -- -D warnings`. CI will reject unformatted code.
4. **If you modified Python code**: from `python/`, run `black spectroxide/` and `pytest tests/`.

### What your PR must include

Every PR that adds or modifies physics code must include:

1. **Tests with independently justified targets.** Each test comment should state where the expected value comes from (e.g., "Eq. 15 of Chluba 2015", "y-era limit: y = drho/(4*rho)", "dimensional analysis: K_BR is dimensionless"). A test that asserts a value without justification will be asked to add one during review.

2. **A dimensional analysis check** for any new rate coefficient or physical formula. This can be a comment in the code or a note in the PR description showing the units work out.

3. **Energy conservation verification** for new injection scenarios: `mu/1.401 + 4y + 4*DeltaT/T = Delta_rho/rho` to within a few percent.

4. **No new crate dependencies.** The zero-dependency constraint is a hard rule, not a preference. If you think an exception is warranted, open an issue to discuss before implementing.

### What your PR should include (when applicable)

- **Cross-validation of PDE vs Green's function** for new injection scenarios where the GF is applicable (simple injection histories). Agreement within ~5% is expected.
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

We will not merge code where test targets cannot be traced to an independent source. This is the one rule we will not bend on, because it is the one that would have prevented every major bug in the project's history.

### CI pipeline

The GitHub Actions CI runs automatically on every PR:

- **Rust**: build, unit tests, science suite, convergence tests, doc tests, clippy, format check (Ubuntu + macOS)
- **Python**: install, import tests, pytest, black format check
- **Docs**: Sphinx + rustdoc build
- **Coverage**: uploaded to Codecov

All checks must pass before merge. If CI fails, fix the issue — do not ask for the check to be skipped.

### Commit messages

Use a short `type: subject` style — typical types are `fix:`, `docs:`, `polish:`, `chore:`, `feat:`. Append `[skip ci]` to commits that touch only documentation, the paper, or other non-code files.

## Resources

- **Paper**: Baker, Liu & Mishra-Sharma (2026), Sec. 6 documents the AI development process and failure modes
- **Tutorial notebooks**: `notebooks/tutorials/` — start with `01_getting_started.ipynb`
- **CosmoTherm**: Chluba (2012), the reference implementation we validate against
- **Chluba & Sunyaev (2012)**, MNRAS 419, 1294 — primary reference for the equations
- **Chluba (2013)**, MNRAS 436, 2232 — Green's function formalism
