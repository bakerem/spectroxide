## src/output.rs

No confirmed rubric violations found. All 81 doc-comment lines were checked (module header, trait doc, and every public struct/method summary).

### Systematic patterns

- Function/method summaries consistently lead with a present-tense third-person verb ("Serializes...", "Writes...", "Parses...", "Counts..."), satisfying A2 throughout.
- `# Errors` sections are present and complete where a function returns `Result` (for example, `OutputFormat::parse`, lines 653–657).
- No em dashes, no T9 word-list violations (no "e.g.", "i.e.", "etc.", "via", "we", "the user", "simply", "just", "currently"), and rustdoc cross-references (`[SolverResult]`, `main.rs::write_output`) are specific rather than "see above/below".
- The module-level `# JSON field naming` section (lines 10–17) is a good example of documenting a historical/legacy naming inconsistency (`drho` vs. `delta_rho_inj` vs. `delta_rho_over_rho`) without hedging or apologizing — it states the fact and the reason plainly.
