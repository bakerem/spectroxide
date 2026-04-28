# CosmoTherm Reference Data

Reference spectral distortion data from Jens Chluba's CosmoTherm code, used for
validation of spectroxide predictions.

## DI Files (included in repo)

ASCII two-column files: `nu [GHz]` and `DI [Jy/sr]`. Lines starting with `#` are comments.

| File | Description | Reference |
|------|------------|-----------|
| `DI_damping.dat` | Acoustic damping + adiabatic cooling | Chluba (2016), Fig. 1 |
| `DI_cooling.dat` | Adiabatic cooling only | Chluba (2016), Fig. 1 |
| `DI_CRR.dat` | Cosmological recombination radiation (CosmoSpec) | Chluba & Ali-Haimoud (2016) |
| `DI_y_late.dat` | Late-time y-distortion (y = 2e-6) | Chluba (2016) |
| `DI_rel_corr.dat` | Relativistic temperature corrections | Chluba (2016) |

## Cosmology (Planck 2015)

```
Y_p = 0.2467, T_CMB = 2.726 K
Omega_m = 0.264737, Omega_b = 0.049169, h = 0.6727
N_eff = 3.046, Omega_k = 0
A_s = 2.207e-9, n_s = 0.9645, n_run = 0
```

## Green's Function Database (NOT included - too large)

The precomputed Green's function database `Greens_data.dat` (~12 MB) is available
from Chluba's website. Use the download script:

```bash
./download_greens.sh
```

Source: https://www.jb.man.ac.uk/~jchluba/Science/CosmoTherm/Download.html

The GF database uses our default cosmology (h=0.71, Omega_b=0.044) matching
Chluba (2013), MNRAS 436, 2232.

## References

- Chluba & Sunyaev (2012), MNRAS 419, 1294 [arXiv:1109.6552]
- Chluba, Khatri & Sunyaev (2012), MNRAS 425, 1129 [arXiv:1202.0057]
- Chluba (2013), MNRAS 436, 2232 [arXiv:1304.6120]
- Chluba (2016), MNRAS 460, 227 [arXiv:1603.02496]
