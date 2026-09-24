"""Constants, background, and X_e(z) source shared by the heat-loss expectation scripts.

The constants are CODATA 2018, typed here. The cosmology is the spectroxide
default (Omega_b = 0.044, Omega_m = 0.26, h = 0.71, Y_p = 0.24, N_eff = 3.046)
with one exception: T_0 = 2.7255 K, where the solver uses 2.726 K. Switching
to 2.726 K moves a delivered fraction by at most 4e-6, but an absolute energy
(the decay's injected Delta rho/rho, the cooling baseline) by 5e-4 to 7e-4.

The only solver-side input is the ionization history X_e(z). It comes from one
of two sources:

- ``python`` (default): ``spectroxide.ionization_fraction`` from this
  checkout's ``python/`` package, a port of ``src/recombination.rs`` run with
  the default cosmology (T_0 = 2.726 K, as in the solver). It matches the X_e
  column of the solver's per-step ledger to 4e-6 at 200 < z < 3000.
- ``--ledger FILE``: the "LEDGER z dz dtau x_e ..." lines that an instrumented
  build prints (``ledger_and_fix_a.patch``, env ``SPX_LEDGER=1``). The ledger
  holds one row per solver step, only about 60 rows between z = 3000 and 200,
  so linear interpolation between them misses X_e by up to 1.2% near z = 1400.
"""

import sys
from pathlib import Path

import numpy as np

# CODATA 2018
SIGMA_T = 6.6524587321e-29  # m^2
K_B = 1.380649e-23  # J/K
M_E = 9.1093837015e-31  # kg
C = 299792458.0  # m/s
M_P = 1.67262192369e-27  # kg
G_N = 6.67430e-11  # m^3 kg^-1 s^-2
A_RAD = 7.565733250e-16  # J m^-3 K^-4
MPC = 3.0856775814913673e22  # m
EV = 1.602176634e-19  # J

T0, OMB, OMM, H, YP, NEFF = 2.7255, 0.044, 0.26, 0.71, 0.24, 3.046
H0 = 100e3 * H / MPC
RHO_CRIT = 3 * H0**2 / (8 * np.pi * G_N)
OMG = A_RAD * T0**4 / C**2 / RHO_CRIT
OMR = OMG * (1 + 7 / 8 * (4 / 11) ** (4 / 3) * NEFF)
OML = 1 - OMM - OMR
NH0 = (1 - YP) * OMB * RHO_CRIT / M_P
FHE = YP / (4 * (1 - YP))  # He-4 mass taken as 4 m_p


def hubble(z):
    return H0 * np.sqrt(OMR * (1 + z) ** 4 + OMM * (1 + z) ** 3 + OML)


def add_xe_argument(parser):
    parser.add_argument(
        "--ledger",
        metavar="FILE",
        help="take X_e(z) from the LEDGER lines of an instrumented build "
        "instead of spectroxide.ionization_fraction",
    )


def _python_ionization_fraction():
    repo_python = Path(__file__).resolve().parents[3] / "python"
    sys.path.insert(0, str(repo_python))
    from spectroxide.cosmology import ionization_fraction

    return ionization_fraction


def _ledger_table(path):
    rows = np.array(
        [l.split()[1:5] for l in open(path) if l.startswith("LEDGER ")], float
    )
    zt, xt = rows[:, 0][::-1], rows[:, 3][::-1]
    zt, idx = np.unique(zt, return_index=True)
    return zt, np.log(xt[idx])


def xe_function(ledger, z_lo, z_hi):
    """Returns X_e(z) on z_lo <= z <= z_hi, interpolated linearly in (z, ln X_e).

    Without a ledger the table is 20000 points of ``ionization_fraction``,
    log-spaced in z. With a ledger, the table is the ledger's own steps. Where
    the requested range reaches above the ledger, ``ionization_fraction``
    fills in.
    """
    lo, hi = 0.99 * (1 + z_lo) - 1, 1.01 * (1 + z_hi) - 1
    if ledger is None:
        zt = np.geomspace(lo, hi, 20000)
        lxt = np.log(_python_ionization_fraction()(zt))
    else:
        zt, lxt = _ledger_table(ledger)
        if hi > zt.max() * 1.001:
            zhi = np.geomspace(zt.max() * 1.0005, hi, 400)
            xhi = _python_ionization_fraction()(zhi)
            zt, lxt = np.concatenate([zt, zhi]), np.concatenate([lxt, np.log(xhi)])

    def xe(z):
        return np.exp(np.interp(z, zt, lxt))

    return xe


def compton_rates(z, xe):
    """Returns (n_H, n_tot, T_r, Gamma_C, rho_gamma) at redshift z, SI units.

    Gamma_C = 8 sigma_T a T_r^4 n_e / (3 m_e c n_tot) is the rate at which
    Compton scattering relaxes the gas temperature toward T_r.
    """
    nh = NH0 * (1 + z) ** 3
    ne = xe(z) * nh
    ntot = ne + nh * (1 + FHE)
    tr = T0 * (1 + z)
    gam = 8 * SIGMA_T * A_RAD * tr**4 * ne / (3 * M_E * C * ntot)
    rho_g = A_RAD * tr**4
    return nh, ntot, tr, gam, rho_g
