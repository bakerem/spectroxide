# Add RECFAST's Lyman-α escape correction to the three-level atom

## Status

Accepted, 2026-09-25 (EB).

Extends the atomic model that
[ADR 0009](0009-evolve-hydrogen-ionization-with-electron-temperature.md) evolves with T_e.

## Context

`src/recombination.rs` and its Python mirror `python/spectroxide/cosmology.py` solve the Peebles
(1968) three-level atom with the Péquignot et al. (1991) case-B coefficient times a fudge factor
F = 1.125. The code credited F = 1.125 to "Chluba & Thomas (2011, arXiv:1011.3758)". That arXiv
number is the HyRec paper (Ali-Haïmoud & Hirata 2011), which finds a best constant fudge factor of
1.126 for its effective multi-level atom without radiative transfer. Neither paper is the source of
the value the code uses.

F = 1.125 is RECFAST 1.5.2's value, and RECFAST uses it together with a correction to the Sobolev
parameter K_H = λ_Lyα³/(8πH), which divides the Lyman-α escape rate:

    1 + Δ(z) = 1 − 0.14 exp{−[(ln(1+z) − 7.28)/0.18]²} + 0.079 exp{−[(ln(1+z) − 6.73)/0.33]²}.

Lee & Ali-Haïmoud (2020, HyRec-2, arXiv:2007.14114, App. B2) describe both: F ≈ 1.125 follows
Rubiño-Martín et al. (2010, arXiv:0910.4383), and "the function Δ(z) is a sum of two Gaussians,
whose amplitudes and widths were chosen to best mimick detailed calculations of HYREC and
COSMOREC." CLASS stores the pairing explicitly (`include/precisions.h`: `recfast_fudge_H` = 1.14
and `recfast_delta_fudge_H` = −0.015 when the correction is switched on, plus `recfast_AGauss1` to
`recfast_wGauss2` with the values above). DarkHistory 1.1.2 applies both in `physics.peebles_C`.
Spectroxide used the fudge factor without the correction it was fitted with.

Measured against HyRec-2 on the default cosmology, with no heating, with DarkHistory's TLA
integrator on a matched background and binding energy standing in for each model:

| z | with Δ(z) | without Δ(z) (spectroxide before this change) |
|---|---|---|
| 1300 | −0.01% | +1.36% |
| 1200 | −0.12% | +1.18% |
| 900 | +0.08% | −1.24% |
| 800 | +0.26% | −0.82% |
| 600 | +0.35% | −0.01% |
| 200 | −0.11% | −0.24% |

Without the correction the error reaches 1.4% during recombination; with it the error stays
below 0.35% from z = 1400 to 100.

The DarkHistory comparison (`dev/notebooks/xe_darkhistory_comparison.ipynb`, paper Sec. 2) had
to strip DarkHistory's correction so the two codes solved the same equations.

## Decision

Divide the Lyman-α escape rate by 1 + Δ(z) in `peebles_c` (Rust) and `_peebles_c` (Python),
through a new `lya_escape_correction(z)` / `_lya_escape_correction(z)`. Keep F = 1.125. Cite
RECFAST 1.5.2 through Lee & Ali-Haïmoud (2020) and Rubiño-Martín et al. (2010) instead of
Chluba & Thomas (2011).

The correction depends on z alone, so it applies equally to the standard table (the Green's
function, `--fixed-ionization`, dark-photon resonances) and to the coupled X_H of ADR 0009. Above
the Saha switch (z ≈ 1575) nothing changes.

The independent oracles in `dev/scripts/heatloss/` get the same correction, since their job is to
reproduce the model the solver claims to solve.

## Alternatives considered

- **Keep the model and fix only the citation.** Leaves a 1.4% error that a four-line change removes,
  and keeps a fudge factor paired with the wrong model.
- **F = 1.14 without the correction** (the original RECFAST pairing). Consistent, but less accurate
  than the RECFAST 1.5.2 pair, which was fitted to modern multi-level codes.
- **Replace the three-level atom with HyRec-2's effective four-level atom.** Removes the fudges
  altogether, at the cost of a larger port. Not needed at the accuracy distortions require.

## Consequences

- X_e below z ≈ 1575 changes by up to 1.4%. The Python X_e table matches DarkHistory's
  unmodified `peebles_C` to 1e-5 for z ≥ 1000.
- Heat-delivery targets move within their tolerances: the burst at z_h = 1000 from 0.9998547 to
  0.9998562, the decay from 0.99417 to 0.99418; the hot burst at z_h = 600 gives X_e(200) =
  8.902e-4 (was 8.893e-4) and a coupling gain of 6.0721e-3 (was 6.0865e-3).
- The DarkHistory notebook runs DarkHistory's own `peebles_C`.
- `PHYSICS_HASH` changes, and the Python/Rust parity fixtures are regenerated.
- Dark-photon limits read the standard table, so resonances below z ≈ 1575 may shift; not yet
  measured. The ADR 0008
  test `test_cell_average_piece_between_two_tangencies` fails: the new history puts a tangency at
  z ≈ 324.6, where two crossings merge within Δz = 0.01, below the Δz = 0.1 finite-difference step
  of dX_e/dz, and the test's 64-node reference is off by 0.4%. That is a weakness of the tangency
  treatment, open under ADR 0008.
  *Correction, 2026-09-26:* the failure came from the test's reference, not from
  `cell_average`. The 64-node value (3.095e-8) is an outlier: 32, 128, 256 and 512 nodes, and
  8–128 composite panels, give 3.079e-8 to 3.081e-8, and `cell_average` is within 0.15% of
  them. The test now uses a composite reference with a 0.3% tolerance and passes. The Δz step
  concern above is not tested by this fix.
- The paper's recombination accuracy statement tightens from 2% to below 0.4% against HyRec-2.
