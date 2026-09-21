## src/solver.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 46 | T9 | MEDIUM | `with 5–6× fewer steps, since \`dtau_max\` takes over as the binding` | Replace causal "since" with "because": "...fewer steps, because `dtau_max` takes over as the binding constraint..." |
| 297–298 | T6 | MEDIUM | `Many fields are currently \`pub\` for use by examples, tests, and diagnostic tooling. These are not part of the stable public API and may become \`pub(crate)\` in a future release.` | Drop the timeless-text markers: "Many fields are `pub` for use by examples, tests, and diagnostic tooling. These are not part of the stable public API and may move to `pub(crate)`." |
| 692–693 | A2 | MEDIUM | `Replaces the solver configuration and reset the current redshift to` | Fix verb agreement to keep both verbs present-tense third-person: "Replaces the solver configuration and resets the current redshift to..." |
| 707–717 | A4 | MEDIUM | doc reads `Panics if any entry is non-finite — silently passing NaN/Inf into the solver causes a deep panic...` but the function (`set_initial_delta_n`) also panics via `assert_eq!` at line 708 on a length mismatch (`"initial_delta_n length {} != grid size {}"`), which the doc comment never mentions | Extend the `Panics` note to cover both conditions: "Panics if `delta_n.len()` does not match the grid size, or if any entry is non-finite." |

### Systematic patterns

- The module and struct-level doc comments (module header, `SolverConfig`, `SolverDiagnostics`, `ThermalizationSolver`, `SolverBuilder`) consistently define abbreviations at first use (PDE, DC, BR, ODE, IMEX, CLI, CMB) and are otherwise well-formed: first sentences are complete, references are specific (`see [\`Self::reset\`]`, not "see above"), and panics/errors are documented for most public methods (`validate`, `set_injection`, `set_initial_delta_n`, `build`).
- Several method docs use passive constructions ("is taken from," "is set to," e.g. line 1672 "The solver state ... is taken from the current state, but the snapshot's z field is set to the requested value") where the actor (the solver / the method) is clear and could be named directly (T2). This is a recurring but low-density pattern (a handful of instances across ~360 doc-comment lines), not severe enough to itemize individually.
- No instances of "e.g.", "i.e.", "etc.", "via", "vs.", "utilize", "leverage", or Latin phrases were found in the doc comments — the file's word choice is otherwise compliant with T9/T13.
- Note (not a rubric-mappable finding, flagged for awareness only): lines 1070–1084 contain what reads as a duplicated/leftover doc sentence — the doc comment for `subtract_temperature_shift` states its one-line summary twice ("Subtracts the temperature shift component..." at the top, and "Subtract the number-conserving temperature shift from Δn." appended after the algorithm list), which looks like an editing artifact rather than a style-rule violation.
