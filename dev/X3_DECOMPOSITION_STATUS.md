# x³-weighted decomposition (ADR 0006): status, 2026-09-24

Branch `review/x3-decomposition` (worktree `.claude/worktrees/x3-decomposition`), from `59df43b`.
Committed on the branch; not merged, not pushed.

## EB decisions (2026-09-24)

1. The appendix fit and the Chluba & Jeong (2014) Gram–Schmidt fit use the x³ (intensity) weight,
   with ΔT/T free. This is the default `decompose_distortion` and the μ, y the solver reports.
2. Tests with visibility-function targets and paper Fig. 1 use a second estimator with ΔT/T fixed
   by photon-number conservation: Rust `decompose_number_conserving`, Python `method="nc"`.

## Done

- Code: `src/distortion.rs`, `python/spectroxide/greens.py`; new Rust and Python unit tests.
- Nine failing tests switched to the `nc` estimator (list in ADR 0006). Eight pass.
- Full release suite: 452 pass, 1 fail (`test_solver_respects_cosmology_parameters`, deleted at EB's request:
  underived 1e-3 threshold, measured 8.4e-4). Clippy clean with and without `axion`.
  Python `test_greens.py` and `test_adversarial_inputs.py` pass; black clean.
- Fig. 1: `notebooks/paper_figures/mu_y_vs_injection_redshift.ipynb` in the MAIN tree (on top of
  EB's uncommitted edits) now calls `method="nc"`; run with `PYTHONPATH=<worktree>/python`;
  `notebooks/figures/pde_mu_y_vs_zh.pdf` regenerated. Old PDF in the session scratchpad.
- Paper (`~/cosmoxide/paper/paper.tex`, uncommitted, `\radd`/`\rdel`): Fig. 1 paragraph,
  Appendix decomposition. Test-compiles cleanly.
- Comparison plot: `dev/scripts/y_estimator/plot_full_range.py` (main tree, untracked),
  low-z spectra `lowz_sweep.py` and `dev/data/visibility_table_lowz.npz`.

## Open

- Claim verifier: all eight claims confirmed (J_μ bound corrected 0.030 to 0.031). Code review done; fixes applied.
- Paper: EB trimmed my appendix additions at 18:46 (kept only the x³ sentence); paper text is EB's.
- Decide the cosmology-sensitivity test.
- `docs/api/greens.rst` method list; CLAUDE.md test counts (EB's CLAUDE.md is dirty in main).
- Commit the branch; merge needs care: `src/solver.rs` test module is also dirty in EB's main tree.
- EB (2026-09-24): keep the paper as written. Flagged and declined: the appendix does not
  define how the temperature shift is removed for Fig. 1, and it keeps the Bianchini & Fabbian
  "avoid this problem" rationale (nonlinear = linear to within 3e-5 at drho = 1e-5). Do not re-raise.
- Final run (2026-09-24): 453 pass, 0 fail, 4 ignored (196 unit + 254 integration + 3 doc); clippy
  clean with and without `axion`; Python decomposition tests and black clean.
- CLAUDE.md not edited on this branch (EB's main copy has uncommitted edits on the same lines).
  After merge: unit count 192 → 196; `pde_heat.rs` 44 → 43 tests (cosmology test deleted), so
  integration total drops by one from whatever EB's copy says.
- Fig. 1 notebook edit lives in the main tree, uncommitted, on top of EB's edits; it needs this
  branch merged (method="nc").
