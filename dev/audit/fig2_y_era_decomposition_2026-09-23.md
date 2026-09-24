# Fig. 2 y-era: the fitted y stops measuring energy (2026-09-23)

Found while fixing the stuck `gf_fit` optimizer (`review/gf-fit-optimizer`). The old Fig. 2 never
plotted PDE values: the left panel was the fitter's analytic start, the right panel was J_y itself.

## Definition of pde_y

`src/distortion.rs:402` `decompose_nonlinear_be`: Δn = [n_BE(x+μ) − n_pl] + δ G_bb + y Y_SZ, all
three free, Levenberg–Marquardt, x in [0.5, 18], unweighted L² with trapezoid weights (dominated by
the low-x tail).

## Measurements (4000-point production grid, Δρ/ρ = 1e-5)

| z_h | y_γ | energy in Δn / Δρ | PDE 4y/Δρ | toy linear Kompaneets | CosmoTherm GF, same fit | x³-weighted fit |
|---|---|---|---|---|---|---|
| 5e3 | 7e-4 | 0.9999 | 1.0098 | 1.0073 | 1.0099 | 0.996 |
| 9e3 | 2.9e-3 | 0.9998 | 1.0318 | 1.0301 | 1.0339 | 0.985 |
| 2e4 | 1.7e-2 | 0.9996 | 1.171 | 1.170 | 1.171 | 0.912 |
| 5e4 | 0.113 | 0.9993 | 1.875 | 1.873 | 1.864 | 0.505 |
| 7e4 | 0.225 | 0.9991 | 2.100 | 2.099 | 2.093 | 0.271 |

y_γ = ∫θ_z dτ from z_h to today (spectroxide cosmology).

## Mechanism

Late-time spectrum = Y + y_γ K(Y) + O(y_γ²). K(Y) carries no energy, but its projection on Y_SZ
under this fit is 10.4, so 4y/Δρ ≈ 1 + 10.4 y_γ. The fitted (μ, δ, y) then carries more energy than
Δn; the balance sits in an in-band residual near x ≈ 7 that L² barely sees. The sign of the excess
depends on the weighting (x³ weighting gives < 1). Chluba's J_y is a different quantity, so plotting
J_y against this y compares different things.

Open: the 0.6–0.8% floor at z_h ≤ 4.2e3 (with μ < 0) is larger than the toy predicts (~0.1%);
CosmoTherm shows it too (1.0069 at z_h = 2514). Cause not pinned.

## Consequences

- Paper lines 666–678 and the Fig. 2 caption: "sub-percent for z_h ≲ 1e4" and "y-era (z_h ≲ 5e4)"
  are wrong; flagged with `\rtodo` for EB. PDE agrees with CosmoTherm to 0.3%, so the solver is fine.
- R1-A′ (open y-excess finding, `dev/audit/ROUND2_STATUS.md`) compares fitted y between codes; it
  should be rechecked with identical decompositions on both sides. Not done.

Scripts: session scratchpad `yera/` (`sweep.pkl`, `ana.py`, `y2b.py`, `ygam.py`). The CosmoTherm
column came from an inline script that was not saved; reproduce with `spectroxide.cosmotherm`
loaders and the same fit before quoting it in the paper.

## Update 2026-09-24: the two estimators side by side

Scripts: `dev/scripts/y_estimator/compare.py` (both estimators on the 118 stored PDE spectra in
`dev/data/visibility_table.npz` and on the CosmoTherm GF database) and `plot.py`
(`dev/figures/mu_y_estimator_comparison.pdf`, git-ignored). No PDE runs.

- **Visibility fit, per spectrum** (the Table 1 recipe with P = J_μJ_bb* and J_y free): the PDE's
  J_y matches Chluba's formula to 0.005 at every z_h (0.544 vs 0.539 at 5.65e4, 0.302 vs 0.300 at
  8.3e4), and CosmoTherm through the same fit agrees. This is the estimator J_y is defined by.
- **Paper appendix fit (Bianchini–Fabbian)**: 4y/Δρ peaks near 2.0 at z_h ≈ 7e4 for both the PDE and
  CosmoTherm, and its μ rises about a factor 2 later in z_h than J_μJ_bb*. Not a solver bug.
- **Implementation check**: an independent linear fit on span{G/x, G, Y} (identical to span{M, G, Y})
  reproduces the Rust nonlinear BF values to 0.5% (e.g. 1.871 vs 1.875 at 5e4). Keeping μ inside
  the exponential changes nothing at μ ~ 1e-5, so the appendix's stated reason for preferring it
  (breaking the μ–T degeneracy of the linear fit) does not hold for small distortions. Condition
  number of the normalised basis on [0.5, 18]: 20.
- `visibility_table.npz` columns `pde_mu`/`pde_y` are the old stuck `gf_fit` output: `pde_y` equals
  the J_y formula exactly. They are not the Rust decomposition.
- CosmoTherm entries above z_h ≈ 4e5 are stored without the exp(−(z/2e6)^{5/2}) factor, so they
  are left off the plot.
- **Appendix fit with x³ weighting** (same model, fitted to x³Δn over [0.5, 18]): y lands close to
  the visibility fit (largest gap −0.135 in 4y/Δρ, at z_h ≈ 6.9e4), but μ overshoots to 1.75 Δρ/ρ
  there (+0.56 above the visibility fit), well above the 1.401 energy limit. With ΔT free, the
  weighting moves the transition-era residual from y into μ instead of removing it. PDE and
  CosmoTherm agree through every estimator. Plot panels: estimators on top, differences from the
  visibility fit below.
- **Appendix fit with the temperature shift removed first** (strip G_bb from data and shapes by
  photon-number conservation, then fit μ and y unweighted on [0.5, 18]): the y excess halves (peak
  4y/Δρ ≈ 1.3, +0.87 above the visibility fit at z_h ≈ 7.2e4) but μ is unchanged (−0.65). This
  variant spans the same shapes as the visibility fit and differs only in weight and band, so the
  remaining gap is the weighting.
