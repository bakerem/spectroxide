# Plan: code attribution and paper updates after the 2026-09-22 review (2026-09-23)

Source of the items: `dev/audit/paper_impact_2026-09-23.md`. Paper: `~/cosmoxide/paper/paper.tex`.
Paper edits use the referee-round markup (`\radd`, `\rdel`, `\rtodo`) and are left uncommitted
for EB to review.

## EB decisions (2026-09-23)

- Plan approved. Order: steps 1 and 2 now; steps 4 and 5 after Fix A (ADR 0004) merges, so
  figures are regenerated once.
- Gaunt fit: cite Draine (2011), flagged with `\rtodo` in the paper for EB to review.
- Photon stimulated-emission wording (paper lines 494, 514): leave as is.
- Pathological heating: keep "sub-percent"; do not quote the new RMS values.

## Steps

1. **Gaunt attribution in code** (no change to numbers). The fit is Draine (2011), Ch. 10
   interpolation g_ff ≈ ln{exp[5.960 − (√3/π) ln(Z ν₉ T₄^{−3/2})] + e}, rewritten in (x_e, θ_e);
   agreement 4.6e-5 over ν = 1e6–1e15 Hz, T = 3e3–1e7 K, Z = 1, 2. Equation number (10.8?)
   unconfirmed. Update `src/bremsstrahlung.rs`, `python/spectroxide/greens.py`, `CLAUDE.md`,
   `tests/heat_injection.rs` (`g_crb20`), add a test against Draine's formula with typed
   constants, close F4 in `dev/audit/double_compton_bremsstrahlung_audit.md`.
2. **Paper edits independent of Fix A:** Eq. `kbr` argument x_e; Draine citation (flagged) at the
   two Gaunt mentions; grid defaults; z_end and z_start defaults; line 964 X_e accuracy; Appendix C
   `dq_dz` example amplitude; referee-1 response line 124 (Fig. 3 compares with the CosmoTherm
   table).
3. **Fix A:** implementation report, physics-inquisitor, code review, merge into local `main`;
   then narrow `ENERGY_CHECK_Z_LATE` / `ENERGY_CHECK_MAX_LATE_FRACTION` separately.
4. **Regenerate figures after Fix A** (fix the Fig. 8 notebook import first): energy
   conservation (both panels), `pde_visibility_fit` + Table 1, `pde_cosmotherm_comparison`,
   `pathological_heating_validation`, `pde_mu_y_vs_zh`, `photon_injection_spectra`,
   `convergence_study`, `firas_photon_limits_paper`. Diff each against the current PDF.
5. **Paper edits after Fix A:** Crank–Nicolson old-half sentence (lines 1399–1404); rewrite the
   energy-conservation paragraph (1533–1539); update Table 1 and the quoted numbers (lines 701,
   703, 731, 1072, 1642; ref1 125–127, 162); rerun `injection_width_resolution.ipynb` (ref2 69).
6. **Verification:** claim-verifier on every changed paper number against notebook output;
   `/refcheck` on the new citation; `/clean-prose` on rewritten paragraphs.

## Status

| Step | Status |
|---|---|
| 1 | Done and merged 2026-09-23 (`c3c176c`): code, docs, CLAUDE.md cite Draine (2011) Ch. 10; new Rust and Python tests against his formula (max deviation 4.6e-5); F4 resolved |
| 2 | Citation audit 2026-09-23: only `Draine2011` missing (already flagged); no other wrong citations. Done 2026-09-23, uncommitted in `cosmoxide` for EB review (Eq. kbr x_e; Draine citation with `\rtodo`; grid defaults; z_end; X_e accuracy with `\rtodo`; `dq_dz` amplitude 1e-25; ref1 line 124 LaTeX comment). Draine2011 bib entry pending: Zotero was not running |
| 3 | Fix A merged 2026-09-23 (`69f1cba`); energy-check narrowing merged 2026-09-24 (`d40e349`) |
| 4 | Done 2026-09-23 23:01 (uncommitted, in the main working tree; backups in the session scratchpad `refig_cache_backup/`). Fig. 3 not run (missing `solve` import; the fix was blocked by the permission system, left for EB). Fig. 4 regenerated but WRONG (notebook f_ann /n_H0 bug, under investigation on `review/fig4-dm-notebook`); do not commit its PDF. Fig. 2 PDE points are the analytic start values (stuck L-BFGS-B; fix on `review/gf-fit-optimizer`). Convergence notebook ran zero tests (fixed on `review/convergence-nb`, `9bc9feb`). Width study now 0.125% (ref2 "at most 0.2%" holds). MMS unchanged |
| 5 | Applied (uncommitted): Crank–Nicolson sentence; energy paragraph (mechanism, heat 0.01% to 0.2%, photon x=0.5 0.5%/0.85%); Table 1 (α_y +1.9, α_μ +2.9, B 0.043 +12, β 2.29 +0.1). "Within 3%", "0.8% rms", "0.022" still hold. Pending: Fig. 3 and Fig. 4 numbers |
| 6 | claim-verifier running on the step 5 numbers |

## Branches waiting to merge (2026-09-23, night)

- `review/grid-floor-100` (`7d318b8`): floor 10 to 100 points, reviewed. Held until the figure
  queue finishes, so every figure comes from one binary. After merge: rebuild, run the full
  release suite once, and confirm the integration-test count in `CLAUDE.md` (the branch adds one
  test; the agent reported 322, the same as before).
- `review/grid-floor-100` and `review/energy-check-late`: merged into local `main` 2026-09-24 (`3550406`, `f511c36`); post-merge suite running.
- `review/convergence-nb` (`9bc9feb`): notebook-only fix. Merge after EB decides on the
  regenerated notebook outputs in the main working tree (same file).
- `review/gf-fit-optimizer` (`77f03b0`, `0d2fb73`): gf_fit now a linear least-squares solve for
  P = J_mu*J_bb* (the old optimizer never moved); Fig. 2 notebook plots PDE mu and y. Code review
  done, follow-ups in `2644a79` (114 greens tests pass, black clean). Held: it touches the Fig. 2
  notebook, which the main working tree also has modified; its PDF waits for EB's y decision. Fig. 2 exposes a paper problem: 4y/drho = 1.031 at z_h = 9e3 and 1.84 at 4.8e4, so the
  caption's "sub-percent for z_h <~ 1e4" and the text's "y-era (z_h <~ 5e4)" fail; physics-inquisitor
  checking whether the excess is a decomposition artifact. Decision for EB: which y the figure shows.
- `review/fig4-dm-notebook` (`370713a`): /n_H0 confirmed as a bug from the start (5.28x over-injection);
  paper values restored (3.758e-20, 5.789e-26 eV/s, f_X = 7.757e5 eV); GF-table cache keyed on all
  parameters; `_generate_notebooks.py` no longer regenerates it. The cosmoxide copy is the regressed
  version and needs the same fix (EB). With the 4000-point table: RMS decay 0.07%, s-wave 0.16%,
  p-wave 0.45% (worst −0.81%); paper "≲ 2%" holds, could tighten to "≲ 0.5% RMS". 8000-point
  table built (34 min); Fig. 4 PDF regenerated and committed on the branch (`36819b3`), same RMS
  values; s-wave residual smaller than in the committed PDF. `dev/notebooks/pde_greens_function.ipynb`
  fixed too (`9460243`), but it still saves to the paper figure path with its N = 2000 method, so
  running it overwrites Fig. 4. Held: same notebook is modified
  in the main working tree.

## TODO for EB (2026-09-24)

Figures (the paper now reads spectroxide's `notebooks/figures/` through a symlink at
`~/cosmoxide/notebooks/figures`; the old directory is `figures.pre-symlink-2026-09-24`):
1. Decide what to commit from the figure run in the spectroxide working tree; then merge
   `review/convergence-nb`, `review/gf-fit-optimizer`, `review/fig4-dm-notebook`.
2. Fig. 3: add `solve` to the imports (both notebook copies), rerun, recheck ref1 line 162 and the
   ref1 line 124 comment.
3. Fig. 2: pick the y option in the paper `\rtodo`; regenerate on `review/gf-fit-optimizer`.
4. Fig. 4: the working tree shows the 2026-08-10 PDF (correct); the branch has the 8000-point
   regeneration. Fix the cosmoxide notebook copy (n_H0). Optionally tighten "≲ 2%" to "≲ 0.5% RMS"
   (paper and ref1 line 127).
5. Fig. 8: review the pre-existing notebook edits now visible in `dp_firas_pde_constraints.pdf`;
   the PDE curve is from the 2026-08-10 cache.
6. cosmoxide git now shows the tracked figure PDFs deleted plus a new symlink: commit or revert.
   Files only in the old directory (posters, `dp_firas_statistic_ladder.pdf`, untracked
   `helium_kink_gamma_con.*`) are in the backup directory.

Paper:
7. Review the uncommitted `\radd`/`\rdel` edits; add `Draine2011` to refs.bib through Zotero;
   confirm the Draine equation number; resolve the X_e accuracy `\rtodo`.

Code and physics (not blocking the paper):
8. `dev/notebooks/pde_greens_function.ipynb` still saves to the Fig. 4 path with N = 2000.
9. Late heat hits the T_e cap at small amplitudes (z_h = 800, Δρ/ρ ≥ 1e-7): runs warn, results wrong.
10. Recheck R1-A′ with identical decompositions; the 0.6–0.8% y floor at z_h ≤ 4e3 is unexplained.
11. Table 1 generator scripts live only in `~/cosmoxide/dev/scripts`.
12. Nothing is pushed.

## 2026-09-24 session (after EB's go-ahead)

Done: figures committed (`799f5a2`, Fig. 3 `c896e73`); Fig. 4 and convergence branches merged;
dev notebooks no longer overwrite paper figures (`a108960`); Table 1 scripts and input tables
merged (`0650af8`, JSON reproduces byte for byte). cosmoxide notebooks are dead (EB); paper edits
are EB's.

Stopped by an out-of-memory kill, resume one at a time:
1. Fig. 2 (`review/gf-fit-optimizer`, uncommitted edits in its worktree): free J_y in `gf_fit`,
   Table 1 range and weighting, CosmoTherm overlay. Resumed first.
2. Item 10 (R1-A′ recheck and the low-z y floor): partial results in the session scratchpad `r1a/`.
3. Fig. 8 PDE limits rerun (`review/fig8-dp-pde-rerun` not created yet): nothing done.
