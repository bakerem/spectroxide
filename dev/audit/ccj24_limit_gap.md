# Why our dark-photon FIRAS limit is stronger than CCJ24's

Investigation I-2 of `dev/REVIEW_2026-09-22.md`, 2026-09-22. A second agent verified it independently: it wrote its own statistics code, loaded the raw FIRAS file, and did not reuse the first agent's scripts. Scripts, outputs, and the four PDE templates are in `dev/scripts/ccj24_gap/`.

## Result

The paper's dark-photon limit (Fig. 8, `FIRASData.profile_limit_floating_T`) is 13–16% stronger in ε than the CCJ24 curve (Chluba, Cyr & Johnson 2024, arXiv:2409.12115, Fig. 8). The size depends on mass, and the median over the cached mass grid is 0.850. Almost all of the gap comes from one convention: the paper does not floor the best-fit amplitude at zero. The FIRAS data fluctuate low along the dark-photon template, so the unfloored one-sided limit comes out tighter than the experiment's sensitivity.

This corrects the earlier attribution to profiling a floating temperature. That attribution appeared in the method-comparison notebook `notebooks/observational/dp_firas_method_comparison.ipynb` and in the notes for the SciPost response. Profiling the temperature changes nothing.

## Mechanism

Both statistics are generalized least-squares fits of the FIRAS spectrum to three templates: the dark-photon distortion S, the temperature derivative G_bb, and dust. Let â be the best-fit amplitude of S and σ_a its error after the nuisance templates are marginalized. Let k = â/σ_a and z = 1.645.

- **Paper:** the limit is u = â + zσ_a, with â unfloored.
- **Floored:** the limit is u = max(â, 0) + zσ_a.

The amplitude scales as ε², so for k < 0 the ratio of limits in ε is

  ε_paper/ε_floor = √[(k + z)/z].

For the paper's own statistic (full covariance, column 2 minus B(T), floating T), FIRAS gives k = −0.471, −0.471, −0.452, and −0.395 at masses 1.5e-12, 3.4e-9, 1.6e-7, and 3.6e-6 eV. The ratio is then 0.845, 0.845, 0.852, and 0.872, and it matches the measured ratio exactly. If we floor the paper's own statistic, the result is 0.994, 0.998, 1.005, and 1.002 of the CCJ24 curve that the authors supplied.

## What does not matter

- **Floating T against fixed T₀:** we compared the paper's temperature profiled as a truly nonlinear parameter with T fixed at T₀ = 2.725 K and G_bb projected out linearly. The two limits differ by a factor of 0.99998 to 1.00003. The best-fit T is 2.725023–2.725027 K, so the linear treatment is exact. For a Gaussian likelihood that is linear in its nuisance parameters, profiling and projection are the same operation.
- **Projecting from the model only, or from data and model:** the Δχ² curves agree to 3e-14, because d·S⊥ = d⊥·S⊥.
- **Covariance and data column:** without a floor, full against diagonal covariance gives 0.993–0.997, and FIRAS column 2 minus B(T) against the residual column 3 gives 0.977–0.980. Column 2 differs from column 3 by rounding: up to 7.6 kJy/sr, 0.30σ RMS. With a floor, σ_a does not depend on the data, so the column has no effect, and full covariance moves the limit by +1.6%. These factors are not a decomposition of the paper-to-CCJ24 ratio once a floor is applied.
- **Templates:** with the same statistic, our PDE templates and the CCJ24 analytic Eq. 25 templates give limits that agree to 0.3% for masses up to 1.6e-7 eV. At 3.6e-6 eV, deep in the μ era, the PDE limit is 9% stronger.

## What CCJ24 did (not settled)

Their Sect. 5 (source lines 626–645) says only "simple Gaussian likelihood". Two readings fit the published curve:

- **A floored best fit,** with diagonal covariance and column 3: 0.978–0.989 of the author-supplied curve.
- **A Bayesian flat prior on γ_con ≥ 0,** with a 95% upper limit U = â + σΦ⁻¹(1 − 0.05Φ(k)): 0.998–1.005 of the hand-digitized curve, and 1.004–1.026 of the author-supplied curve. For a Gaussian likelihood this equals the asymptotic CLs limit. It gives Δρ/ρ < 5.27e-5, close to the 5.3e-5 in their text.

Our two copies of the CCJ24 curve disagree by up to 3.3% below 1.6e-7 eV and 4.0% up to 3e-5 eV, on a dense grid (the author-supplied `dev/AxionLimits/limit_data/DarkPhoton/COBEFIRAS_Chluba.txt` against the hand-digitized `dev/data/cosmotherm_dp_lims.csv`). That is more than the gap between the two readings, so the evidence we have cannot pick one.

## Which limit is correct

Both limits are valid, but they answer different questions. The unfloored one-sided limit has correct frequentist coverage. When the data fluctuate low, it excludes more than the experiment's sensitivity supports. For a parameter that is physically non-negative, the usual choice is CLs (Read 2002, J. Phys. G 28, 2693) or a limit that caps this effect (Cowan, Cranmer, Gross & Vitells 2011, arXiv:1105.3166). Feldman & Cousins (1998, Phys. Rev. D 57, 3873) handle the same boundary with unified intervals.

## Actions

- Paper Sect. 6 and the method-comparison notebook must stop attributing the gap to floating-T profiling or to the "statistical power" of the likelihood ratio. Either state that the limit has no floor and that FIRAS fluctuates about 0.45σ low along the dark-photon shape, or switch Fig. 8 to CLs or a floor. That choice is yours.
- The CCJ24 overlay notebook (`notebooks/observational/dp_firas_ccj24_overlay.ipynb`) links to arXiv:2409.13818, which is a different paper. CCJ24 is arXiv:2409.12115.

## Notebook

`notebooks/observational/dp_firas_limit_conventions.ipynb` (`d110653`) plots every statistic above against both CCJ24 curves, with the step table and a goodness-of-fit (GoF) limit. The earlier GoF code had the right form (absolute χ² at fixed amplitude against χ²₄₁) but projected the nuisance templates out of the model and not the data, and used diagonal errors. The corrected GoF limit is 1.21 of CCJ24, not 1.35, so CCJ24 did not use a GoF test. FIRAS χ²_min/dof = 48.6/40 (p = 0.17) with full covariance.
