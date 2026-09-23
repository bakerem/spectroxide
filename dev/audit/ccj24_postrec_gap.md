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
