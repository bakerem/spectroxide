"""Independent expectation for the fraction of burst heat that reaches the CMB photons.

Integrates the gas-temperature excess dT (over the no-injection history) for a
Gaussian heat burst in redshift, with Compton exchange, adiabatic cooling of the
gas, and the injected heat, and integrates the Compton power into the photons:

    d(dT)/dt = Q(t) / (3/2 k n_tot) - Gamma_C dT - 2 H dT
    Gamma_C  = 8 sigma_T a T_r^4 n_e / (3 m_e c n_tot),  n_tot = n_e + n_H + n_He
    P_gamma  = (3/2) k n_tot Gamma_C dT                   (photon energy gain rate)

Everything here comes from CODATA constants typed below and the default
spectroxide cosmology parameters (Omega_b = 0.044, Omega_m = 0.26, h = 0.71,
Y_p = 0.24, N_eff = 3.046), except T_0 = 2.7255 K where the solver uses
2.726 K. A delivered fraction does not depend on that choice, but an absolute
injected energy does. The only input taken from the
solver is its ionization history X_e(z), read from the per-step ledger lines
("LEDGER z dz dtau x_e ...") printed by an instrumented build. Nothing else is
imported from spectroxide.

Usage: python heat_delivery_expectation.py LEDGER_FILE Z_H [DELTA_RHO] [Z_END]
"""

import sys

import numpy as np
from scipy.integrate import solve_ivp

# CODATA 2018
SIGMA_T = 6.6524587321e-29  # m^2
K_B = 1.380649e-23  # J/K
M_E = 9.1093837015e-31  # kg
C = 299792458.0  # m/s
M_P = 1.67262192369e-27  # kg
G_N = 6.67430e-11  # m^3 kg^-1 s^-2
A_RAD = 7.565733250e-16  # J m^-3 K^-4
MPC = 3.0856775814913673e22  # m

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


def main():
    f, zh = sys.argv[1], float(sys.argv[2])
    drho = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
    zend = float(sys.argv[4]) if len(sys.argv) > 4 else 200.0
    sig = max(0.04 * zh, 100.0)
    rows = np.array([l.split()[1:5] for l in open(f) if l.startswith("LEDGER ")], float)
    zt, xt = rows[:, 0][::-1], rows[:, 3][::-1]
    zt, idx = np.unique(zt, return_index=True)
    lxt = np.log(xt[idx])

    def xe(z):
        return np.exp(np.interp(z, zt, lxt))

    def rates(z):
        nh = NH0 * (1 + z) ** 3
        ne = xe(z) * nh
        ntot = ne + nh * (1 + FHE)
        tr = T0 * (1 + z)
        gam = 8 * SIGMA_T * A_RAD * tr**4 * ne / (3 * M_E * C * ntot)
        rho_g = A_RAD * tr**4
        return nh, ntot, tr, gam, rho_g

    def rhs(z, y):
        # y = [dT (K), E_gamma/rho_gamma delivered, E_adiabatic lost]; integrate in z (dt = -dz/(H(1+z)))
        dT = y[0]
        nh, ntot, tr, gam, rho_g = rates(z)
        hz = hubble(z)
        dtdz = -1.0 / (hz * (1 + z))
        q_rel = (
            drho
            * np.exp(-((z - zh) ** 2) / (2 * sig**2))
            / np.sqrt(2 * np.pi)
            / sig
            * hz
            * (1 + z)
        )
        cap = 1.5 * K_B * ntot / rho_g  # gas heat capacity per photon energy density
        d0 = (q_rel / cap - gam * dT - 2 * hz * dT) * dtdz
        d1 = cap * gam * dT * dtdz
        d2 = (
            cap * hz * dT * dtdz
        )  # energy the excess loses to expansion beyond T_r scaling
        return [d0, d1, d2]

    zs = zh + 7 * sig
    sol = solve_ivp(
        rhs,
        (zs, zend),
        [0.0, 0.0, 0.0],
        method="Radau",
        rtol=1e-10,
        atol=[1e-14, 1e-22, 1e-22],
        dense_output=False,
    )
    dT, eg, ead = sol.y[:, -1]
    nh, ntot, tr, gam, rho_g = rates(zend)
    stored = 1.5 * K_B * ntot / rho_g * dT
    print(
        f"z_h={zh:g} sigma={sig:g} z_end={zend:g}: delivered/injected = {eg / drho:.6f}; "
        f"adiabatic-excess loss = {ead / drho:.2e}; still in gas at z_end = {stored / drho:.2e}"
    )


if __name__ == "__main__":
    main()
