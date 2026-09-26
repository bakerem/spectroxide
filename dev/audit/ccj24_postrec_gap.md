# The post-recombination part of the dark-photon gap to CCJ24

Follow-up to `ccj24_limit_gap.md`, 2026-09-23. That record traced the overall 15% gap to the unfloored statistic. This one asks why, at the CCJ24-literal statistic, our limit is still low at masses that resonate after recombination (z_res ≲ 1100, m ≲ 2e-10 eV). The script is `dev/scripts/ccj24_gap/postrec_gap.py` and its output is `postrec_gap_out.txt`. It runs no PDE: it uses the 30 templates in `grid_templates.npz` and changes one input of the ε → γ_con mapping at a time.

## Result

The gap is 1.6% in ε, not 5%, and two inputs explain most of it. Swapping in CCJ24's choices closes it to +0.1% for z_res < 800 and to −1.0% for 800 < z_res < 1450.

Statistic: diagonal errors, residual column, G_bb and dust projected out of the model, and the limit at Δχ² = 4 above the no-distortion model. We report each ratio ε_ours/ε_CCJ24 divided by its median over 3e3 < z_res ≤ 5e4. This division removes the overall level, which depends on the threshold. With Δχ² = 3.84 the absolute post-recombination ratio is 0.960–0.977. This is the "2–4% low" in the conventions notebook, and presumably the source of the 5% figure.

| z_res band | 250–800 | 800–1100 | 1100–1450 | 1450–3000 |
|---|---|---|---|---|
| ours: Peebles X_e, default cosmology, AxionLimits file | 0.986 | 0.978 | 0.975 | 0.986 |
| same, against the Fig. 8 vector curve | 0.994 | 0.982 | 0.982 | 0.982 |
| HyRec X_e, Planck 2018, against the vector curve | 1.001 | 0.991 | 0.990 | 0.994 |

The y-era normalization is 0.995 against the AxionLimits file and 1.000–1.001 against the vector curve. The gap is flat below z_res ≈ 800. It deepens to about 2.5% at 800–1450, where X_e falls and the mapping steepens.

## Causes, each tested by a swap

1. **The AxionLimits curve is 1.0–1.4% too high at z_res < 1450.** We read CCJ24's Fig. 8 FIRAS curve (solid blue, 170 vertices) straight from the vector paths of `eps/eps_limits_new.pdf` in the arXiv source (`ccj24_fig8_firas_vector.txt`). The AxionLimits file divided by the vector curve gives 1.014, 1.011, and 1.009 in the three bands below z_res = 1450. It gives 1.001 in the y era and 1.000 in the μ era, so the file matches everywhere except the low-mass end. Switching reference curves moves the post-recombination ratio by +0.8% (250–800) and +0.4–0.7% (800–1450).
2. **Cosmology: +0.4% (250–800) and +1.0% (800–1450).** The change from our default (h = 0.71, Ω_m = 0.26) to Planck 2018 enters through Ω_m h² (0.131 against 0.142). An earlier scratch run showed that setting Ω_m h² alone reproduces the full Planck 2018 shift. γ_con ∝ 1/H(z_res). H follows √(Ω_m h²) fully after recombination but only partly in the y era, where radiation still contributes, so the ratio between the eras moves. Planck 2015 does the same job to within 0.4%.

## What does not matter

- **Ionization history.** HyRec-2020 and RecfastCLASS X_e(z), from CLASS for our cosmology, move the bands by −0.3% to +0.5% relative to our Peebles solution. HyRec and Recfast agree with each other to 0.1% (0.5% at 1450–3000). This includes the d ln ω_pl²/dt factor, which we take from the same X_e spline. Our mapping matches `spectroxide.dark_photon.gc_per_epsilon_sq` to 0.1%.
- **T₀ = 2.7255 against 2.726 K:** ≤0.05%.
- **Template shape.** Every PDE template with z_res < 1.5e4 gives the same γ_con limit as the analytic CCJ24 Eq. 25 template to 0.2%. The deviation is 0.3% at 2e4, and k = −0.348 at every mass. No Comptonization or BR reprocessing after the resonance affects the FIRAS limit. This holds whatever CosmoTherm does after conversion, as long as its template follows Eq. 25.
- **Plasma-frequency definition.** CCJ24 Eq. 3 uses free electrons, as we do. Rescaling n_e by 1.02 would close the 800–1450 band, but it moves the y-era normalization by −0.9%. Counting all electrons (×1.079) overshoots by 2–3%. The paper does not support either rescaling.
- **Coarse grids.** Interpolating a 100–200-point log mass grid biases the curve by ≤0.05% outside the He recombination kink. Inside the kink (1450 < z_res < 1700), the bias is −1% on median and up to 4% at single masses (scratch run). Reading the Δχ² crossing off a 100–200-point ε grid biases the limit by 0.1–0.5%. The bias is the same in both eras, so it cancels in the ratio.
- **CCJ24's own γ*_con contour figure** (`gamma_con_contour.pdf`) cannot serve as a mapping reference. Their Fig. 8 divided by their γ* = 1e-4 contour should be flat in mass, but it ranges from 0.95 to 1.09 and is not smooth, even in the y era.

## Unexplained

After both swaps, a −1.0% residual remains at 800 < z_res < 1450, and −0.6% at 1450–3000. The Recfast/HyRec spread (0.3%) and the ε-grid bias (≤0.5%) are of this size. The residual could come from CCJ24's exact cosmology, from their Recfast++ or CosmoRec settings, or from how their 20,000 grid models were interpolated. We cannot separate these without their inputs. At 1% this residual does not affect Fig. 8.

## Actions

- If Fig. 8 or the notebooks quote agreement with CCJ24 below 2e-10 eV, they should use the vector curve `ccj24_fig8_firas_vector.txt` or state that the AxionLimits file is 1–1.4% high there.
- A like-for-like comparison below 2e-9 eV should use Planck cosmology in the ε → γ_con mapping. The default cosmology costs 0.4–1% there.

## 2026-09-25 follow-up: two errors on our side

Script `dev/scripts/ccj24_gap/postrec_followup.py`, output `postrec_followup_out.txt`. Same statistic as above (Δχ² = 4, normalized to z_res 3e3–5e4), Planck 2018 throughout, current X_e (ADR 0009 and 0011), against the Fig. 8 vector curve, on a 120-point mass grid instead of band medians. The band medians above hid both effects below.

**He I recombination (our error, −4% to +12% in ε at m = 1.1e-9 to 2.6e-9 eV).** Our helium is Saha at T_z (`src/recombination.rs`), so He I recombines too early and too steeply. Seager, Sasselov & Scott (2000) found that He I recombination lags Saha. Our X_e is 2.3%, 4.8%, and 5.9% below HyRec-2020 at z = 1900, 2100, and 2300, while RECFAST and HyRec agree to 0.1% there and ours matches HyRec to 1e-4 at z ≤ 1700. The slope, not the level, drives the error. The mapping uses d = 3 + d ln X_e / d ln(1+z) at z_res. At 1.32e-9 eV our d is 3.00 against HyRec's 3.32, and at 2.04e-9 eV it is 3.79 against 3.09. That accounts for 0.951 of the 0.958 dip and 1.108 of the 1.124 peak. The level difference shifts z_res by at most 1.7% and ε by 1–2%. So a fix must reproduce the timing of He I recombination; rescaling the level will not work. On a dense grid, ε_ours/ε_HyRec runs from 0.958 at 1.32e-9 eV to 1.124 at 2.04e-9 eV. A fresh-context verifier reproduced these numbers with its own mapping code, which agrees with `gc_per_epsilon_sq` to 0.05%. (The script's splice "ours + HyRec above z = 1550" is identical to HyRec for m ≥ 1e-9 eV, because z_res > 1550 there. It shows nothing beyond the X_e comparison.) The notebook `dp_firas_limit_conventions.ipynb` attributes its dip at 1.3e-9 eV to "the difference between our template and CosmoTherm's". That is wrong: the templates agree to 0.2% there. Paper Fig. 8 uses the same mapping. Its cached curve (`dev/data/dp_firas_pde_limits.npz`, 70 masses), divided by the vector curve and normalized to z_res 3e3–5e4, gives 0.974 at 9.0e-10 eV, 0.940 at 1.34e-9, 1.101 at 2.0e-9, and 0.990 at 3.0e-9 eV. So the figure carries the error at two of its nodes.

**Low-z hydrogen (our error, −0.1% to −0.7% in ε at z_res < 800).** Our standard X_e table evolves hydrogen with the gas at T_z. Below z ≈ 800 our X_e runs 0.3–1.9% above HyRec (z = 800 to 200), and s is low by 0.01–0.03. Splicing HyRec below z = 800 removes the offset. That the cause is T_m < T_z is likely but not tested directly: the sign and growth toward low z match, and `tests/ionization_coupling.rs` shows the coupled (ADR 0009) history matches HyRec where the fixed one is off by 2–10% at z ≤ 200.

**CCJ24 Fig. 2 is not their Fig. 8 mapping.** We read the z_con(m) curve from the vector paths of `eps/con_plot.pdf` (`ccj24_zcon_vector.txt`). It agrees with our z_res to 0.3% above 1e-8 eV but sits 2–7% lower in z below 1e-9 eV, for every X_e history and cosmology we tried, and it is a staircase in z with 1.5% steps. Used as the mapping, it would make our limit 5–14% weaker than CCJ24 below 1e-9 eV. The measured gap has the opposite sign and is under 1%, so Fig. 8 did not use the Fig. 2 curve. Its matter–radiation equality marker sits at z_con = 3418. Planck 2018 gives z_eq = 3403 and our default cosmology gives 3130. If the marker was computed rather than placed by hand, this is consistent with CCJ24 using Planck, which supports the swap in the table above.

**What remains.** With HyRec at Planck 2018, the ratio is 1.001 at z_res 250–800, 0.991 at 800–1550, 0.993 at 1550–2700, 0.997 at 2700–7000, and 1.001 above. The −0.9% at 800–1550 appears for every X_e history, so it is not recombination physics we can vary. Single-mass features at 1.1e-9 eV (−1.5%) and 7.4e-9 eV (+2.9%) are identical for all histories. The template γ limits change by less than 0.02% across them. The CCJ24 vector curve is not monotone there: ε = 3.64e-8, 3.70e-8, 3.81e-8, and 3.45e-8 at m = 5.9e-9, 6.9e-9, 8.0e-9, and 9.3e-9 eV. So these features sit in their curve, probably from contouring across the H and He II steps. We do not know the source of the −0.9%.

**Budget, current code against the AxionLimits file (what the conventions notebook plots).** At z_res 250–800: AxionLimits file high by 0.6–1.5%, default against Planck cosmology 0.4%, low-z X_e 0.1–0.7%, residual ≈ 0. At 800–1550: file 1.0%, cosmology 1.1%, residual 0.9%. At 1.1e-9 to 2.6e-9 eV: He I, −4% to +12%, per mass.

**Fixes (not made; each changes a physics model or default and needs an ADR).** (1) A He I recombination ODE in `recombination.rs` (RECFAST 1.5.2's He I treatment, as in Seager et al. 2000 and Wong, Moss & Scott 2008), replacing Saha. It also changes n_e, and so the Thomson rate, by up to 6% at z = 1900–2300 in every run. (2) A standard X_e table that evolves T_m with Compton coupling, as RECFAST does, rather than T_m = T_z. Or: build the dark-photon mapping from the coupled no-injection history.

### The plateau offset is a mass-axis (n_e) signature

Between z_res ≈ 850 and 1450 the resonance redshift barely changes with mass, so ε ∝ 1/m, and any offset in the m ↔ z_res relation shows up there and almost nowhere else. We regressed ln(ε_ours/ε_CCJ24) on the local slope d ln ε / d ln m. We used RECFAST at Planck 2018 and 200 masses, excluding the He I kink and m > 5e-8 eV. Against the Fig. 8 vector curve: ln ratio = 0.0027 + 0.0103 × slope, and the rms drops from 0.58% to 0.38%, the scatter floor of the reference. Against the AxionLimits file: 0.0155 × slope, with rms 0.85% → 0.54%. Scaling n_e by (1 + f) gives ln ratio = f/2 × (1 + slope), so the fit reads f ≈ 2.1% against the figure and 3.1% against the file. CCJ24 behave as if their n_e(z) near recombination is about 2% above ours, and the file carries another 1% of the same signature. So the file-to-figure difference is probably a different run or mapping, not digitization.

`postrec_candidates.py` tests the physical inputs, with ratios normalized to the y era (target 1.000):

| case | z_res 250–800 | z_res 850–1450 |
|---|---|---|
| HyRec, Planck 2018 | 1.002 | 0.992 |
| RECFAST 1.5 (escape correction on) | 1.002 | 0.992 |
| RECFAST 1.4 (F = 1.14, no correction) | 1.007 | 0.995 |
| HyRec, Y_p = 0.24 | 1.004 | 0.995 |
| Y_p = 0.24 + RECFAST 1.4 | 1.009 | 0.998 |
| HyRec, Planck 2015 | 0.999 | 0.988 |
| HyRec, our default cosmology | 1.000 | 0.983 |
| HyRec, n_e × 1.02 | 1.004 | 1.001 |

Each plausible CosmoTherm-era choice (Y_p = 0.24, RECFAST without the 1.5 correction) moves the plateau by about +0.3%. Together they close it to 0.2%, but overshoot the 250–800 band by 0.9%. No documented combination fits both bands better than a flat 2% n_e rescaling. We have no physical reason for such a rescaling. The inputs that could decide it are CCJ24's Y_p, their recombination code and version, and their X_e table.
