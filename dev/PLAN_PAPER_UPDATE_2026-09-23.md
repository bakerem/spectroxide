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
| 1 | Running (agent, branch `review/gaunt-draine`) |
| 2 | Done 2026-09-23, uncommitted in `cosmoxide` for EB review (Eq. kbr x_e; Draine citation with `\rtodo`; grid defaults; z_end; X_e accuracy with `\rtodo`; `dq_dz` amplitude 1e-25; ref1 line 124 LaTeX comment). Draine2011 bib entry pending: Zotero was not running |
| 3 | Implementation agent running on `review/adr4-fix-a` |
| 4–6 | Not started |
