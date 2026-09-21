## src/lib.rs

| Line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| 28 | T9 | MEDIUM | `// Set energy injection (e.g., delta-function burst at z=2×10⁵)` | Replace with "for example": "// Set energy injection (for example, a delta-function burst at z=2×10⁵)". |
| 40 | H1 | MEDIUM | `//! ## Green's Function Mode` | Sentence case: "## Green's function mode". |
| 54 | T2 | LOW | `/// Used to invalidate cached Green's function tables when the underlying` | Name the actor: "Invalidates cached Green's function tables when the underlying physics code changes." |

### Notes

- PDE and CMB are both spelled out at first use in the crate-level `//!` doc (lines 3–4, 9–10), and CLI is spelled out at line 56 — good T12 compliance for the library's own top-level documentation.

## src/main.rs

No confirmed violations in the 6 doc-comment lines. `Gets the output writer...` (line 89) and `Writes any result type...` (line 100) both lead with a present-tense verb (A2-compliant).

## src/bin/check_adiabatic.rs

No confirmed violations in the 5 doc-comment lines.

## examples/*.rs (`//!` headers only)

| File / line | Rule | Severity | Quotation | Suggested fix |
|---|---|---|---|---|
| examples/energy_budget.rs:7–15 | L3 | MEDIUM | `* the x-quadrature and the analytic G₃ normalization contribute ≲10⁻⁴ —` / `* the heat-injection deficit is the first-order-in-Δτ temporal error of` / `* the photon-injection "1%" is mostly the finite width of the Gaussian` | Capitalize the first letter of each list item per L3 ("The x-quadrature...", "The heat-injection...", "The photon-injection..."), or restructure the lead-in so the colon is dropped and each bullet is a self-contained capitalized sentence. |
| examples/energy_budget.rs:7 | F11 | LOW | `the x-quadrature and the analytic G₃ normalization contribute ≲10⁻⁴ —` (em dash preceded by a space, continued on the next line) | Remove the space before the em dash. |
| examples/temporal_error_check.rs:1 | F6 | MEDIUM | `One-off: measure the temporal discretization error of μ at the DEFAULT` | Do not use all caps for emphasis; use code font or plain text instead: "at the default `dy_max = 0.02`". |

### Notes

- examples/custom_injection.rs, examples/generate_parity_fixtures.rs, and examples/photon_diag.rs headers show no confirmed rubric violations: each opens with a summary sentence, introduces its numbered/itemized content with a colon sentence (L2), and gives a runnable `cargo run` command in code font.
