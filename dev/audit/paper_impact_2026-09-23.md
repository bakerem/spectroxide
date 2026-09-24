# Paper impact of the 2026-09-22/23 review changes

Checklist for `~/cosmoxide/paper/paper.tex` (line numbers as of 2026-09-23) and
the two referee responses (`ref1`, `ref2`). Compiled from a read-only audit
agent and a manual pass over the appendices. Nothing in the paper has been
edited. "Fix A" is ADR 0004 (being implemented on `review/adr4-fix-a`).

## Wrong now

- [ ] **Line 1316, Eq. `kbr`:** `g_ff(x, θ_e, Z_i)` should be `g_ff(x_e, θ_e, Z_i)` (P-1).
- [ ] **Lines 1319, 1725:** the softplus Gaunt interpolation is attributed to Chluba, Ravenni &
  Bolliet (2020). `src/bremsstrahlung.rs:44-60` now records that it came from J. Chluba by
  private communication and is not in that paper (P-6).
- [ ] **Lines 1505–1506, grid defaults:** no code switches to N = 4000 at z_h > 1e6. Python always
  uses the production grid (N = 4000, x ∈ [1e-5, 60], x_t = 0.5, 35% log); the CLI and Rust
  default is 2000 points. The Table 2 caption (line 1682) already says Python uses N = 4000.
- [ ] **Line 1527:** "down to z = 1" — the CLI and Python default is z_end = 500. The
  z_h + 7σ_z start is now true for every front end (ADR 0001). The CLI starts continuous
  scenarios at 5e6, Python at 3e6.

## Wrong once Fix A lands

- [ ] **Lines 1399–1404 and 1420:** say that the old Crank–Nicolson half uses the step's
  electron temperature, so the heating term is backward Euler like Eq. `te_be` and the heat
  the electrons lose equals the heat the photons gain. The second-order claim holds for Δn,
  not in θ_e.
- [ ] **Lines 1533, 1536–1537:** "the implicit time-stepping conserves energy to high accuracy
  by construction" and "the dominant source of leakage is the first-order backward-Euler DC/BR
  coupling" are both wrong. For z_h ≤ 2e5 the main leak was the ρ_e time-centering mismatch
  (6% of the heat at z_h = 1000). What remains is a μ-era residual first order in Δτ.
- [ ] **Line 1539:** "deviations of comparable magnitude at the low-z end" — the low-z end
  closes (0.99998 at z_h = 3e3). Near 1e6–2e6 the gap stays at 0.4–0.6%.

## Rerun a number or regenerate a figure (mostly after Fix A)

- [ ] `pde_energy_conservation`, `pde_energy_conservation_photon` (lines 1539–1541).
- [ ] `pde_visibility_fit` and Table 1 (lines 693–731; "within 3%", "0.8% rms", "0.022";
  ref1:125–126).
- [ ] `pde_cosmotherm_comparison` (line 743; ref1:162): the amplitude offset should shrink.
- [ ] `pde_gf_dm_comparison` (lines 763, 771; ref1:127): "≲ 2%" survives.
- [ ] `pathological_heating_validation` (lines 786–788): RMS 0.20/0.07/0.49% after Fix A,
  could be quoted.
- [ ] `pde_mu_y_vs_zh` (lines 666–677): text should survive.
- [ ] `photon_injection_spectra` (notebook runs to z_end = 50): P-1, the P-8 guard, and Fix A
  all act at low x; probably invisible, unchecked.
- [ ] `firas_photon_limits_paper`: at most a 0.5% μ shift; low priority.
- [ ] `convergence_study` (line 1598): probably survives.
- [ ] Line 1642, end-to-end MMS error "2.1e-4": runs through the full solver; recheck.
- [ ] Line 1072, μ/Δρ "≈ 1.16" at z = 1e6: may become 1.17; low priority.
- [ ] ref2:69, width study "at most 0.2%": recheck `injection_width_resolution.ipynb`.

## Found in passing (not caused by these changes)

- [ ] **Line 964:** "Peebles … ∼5–10% in X_e at z ∼ 1000–1200" is overstated;
  `dev/audit/xe_hyrec_comparison.md` measured at most 1.9% against HyRec-2 before P-4 and 0.79%
  at the P-4 spot points after it.
- [ ] **Lines 494, 514:** "(1+n_pl)² … both photons occupy the same state". Review item P-3 found
  the net factor is 1 + 2n and that the photons travel back to back. The code docs were left
  unchanged by decision; the paper states it as physics.
- [ ] **ref1:124:** says Fig. 3 compares PDE and analytic Green's functions; Fig. 3 compares the
  PDE with the CosmoTherm table directly.
- [ ] **Appendix C, Python example:** `dq_dz = 1e-12 (1+z)**2` gives μ ≈ 3.5e6; use a
  physically sized rate.
- [ ] **Fig. 8 notebook:** `notebooks/paper_figures/dark_photon_constraints.ipynb` imports the
  removed `_cosmo_omega_gamma` (being fixed in phase 9).

## Checked and unaffected

Per-n_H wording (line 396, ref2:76), the p-wave T_χ = T_γ condition (lines 426–432), the timing
table (measured through Python), the DC/BR target text (ADR 0002 made the code match it), the
dark-photon limits (≤ 0.15%), the MMS and moment checks (Kompaneets operator alone), and the
Appendix C module list, Rust example, and exported Python names (verified to run on `main`).
