"""Independent expectation for the fraction of burst heat that reaches the CMB photons.

Integrates the gas-temperature excess dT (over the no-injection history) for a
Gaussian heat burst in redshift, with Compton exchange, adiabatic cooling of the
gas, and the injected heat, and integrates the Compton power into the photons:

    d(dT)/dt = Q(t) / (3/2 k n_tot) - Gamma_C dT - 2 H dT
    Gamma_C  = 8 sigma_T a T_r^4 n_e / (3 m_e c n_tot),  n_tot = n_e + n_H + n_He
    P_gamma  = (3/2) k n_tot Gamma_C dT                   (photon energy gain rate)

Constants, cosmology, and the X_e(z) source are in heatloss_common.py. Nothing
but X_e(z) is taken from spectroxide. The burst width is the CLI default,
sigma_z = max(0.04 z_h, 100); ``--sigma`` overrides it.

Target of tests/heat_delivery.rs::burst_at_recombination_delivers_independent_fraction:
    python heat_delivery_expectation.py 1000        ->  0.999863

Usage: python heat_delivery_expectation.py Z_H [--drho 1e-8] [--z-end 200] [--sigma S] [--ledger FILE]
"""

import argparse

import numpy as np
from scipy.integrate import solve_ivp

from heatloss_common import K_B, add_xe_argument, compton_rates, hubble, xe_function


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("z_h", type=float, help="burst redshift")
    p.add_argument("--drho", type=float, default=1e-8, help="injected Delta rho/rho")
    p.add_argument("--z-end", type=float, default=200.0, help="final redshift")
    p.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="burst width in z (default: the CLI's max(0.04 z_h, 100))",
    )
    add_xe_argument(p)
    args = p.parse_args()
    zh, drho, zend = args.z_h, args.drho, args.z_end
    sig = max(0.04 * zh, 100.0) if args.sigma is None else args.sigma
    if not sig > 0:
        p.error(f"--sigma must be positive, got {sig}")
    zs = zh + 7 * sig
    xe = xe_function(args.ledger, zend, zs)

    def rhs(z, y):
        # y = [dT (K), E_gamma/rho_gamma delivered, E_adiabatic lost]; integrate in z (dt = -dz/(H(1+z)))
        dT = y[0]
        nh, ntot, tr, gam, rho_g = compton_rates(z, xe)
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
        # energy the excess loses to expansion beyond T_r scaling
        d2 = cap * hz * dT * dtdz
        return [d0, d1, d2]

    sol = solve_ivp(
        rhs,
        (zs, zend),
        [0.0, 0.0, 0.0],
        method="Radau",
        rtol=1e-10,
        atol=[1e-14, 1e-22, 1e-22],
    )
    dT, eg, ead = sol.y[:, -1]
    nh, ntot, tr, gam, rho_g = compton_rates(zend, xe)
    stored = 1.5 * K_B * ntot / rho_g * dT
    print(
        f"z_h={zh:g} sigma={sig:g} z_end={zend:g}: delivered/injected = {eg / drho:.6f}; "
        f"adiabatic-excess loss = {ead / drho:.2e}; still in gas at z_end = {stored / drho:.2e}"
    )


if __name__ == "__main__":
    main()
