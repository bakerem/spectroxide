"""Independent expectation for the fraction of decaying-particle heat that reaches the CMB photons.

Same gas-excess integration as heat_delivery_expectation.py, with the burst
replaced by the solver's DecayingParticle heating rate
f_X Gamma N_H exp(-Gamma t) per unit volume:

    d(dT)/dt = Q(t) / (3/2 k n_tot) - Gamma_C dT - 2 H dT
    P_gamma  = (3/2) k n_tot Gamma_C dT

The injected Delta rho/rho_gamma is the integral of Q / rho_gamma over the run.
Here rho_gamma uses T_0 = 2.7255 K and the solver 2.726 K, so the injected
energy printed here is 7e-4 below the solver's; the delivered fraction does
not depend on it. The age t(z) is integrated from H(z). Constants, cosmology,
and the X_e(z) source are in heatloss_common.py.

Target of tests/heat_delivery.rs::decay_at_recombination_delivers_independent_fraction
(lifetime at z = 1000):
    python decay_delivery_expectation.py 10 7.1838e-14 5e4 200   ->  0.99417

Usage: python decay_delivery_expectation.py F_X_EV GAMMA Z_START Z_END [--ledger FILE]
"""

import argparse

import numpy as np
from scipy.integrate import quad, solve_ivp

from heatloss_common import (
    A_RAD,
    EV,
    K_B,
    NH0,
    T0,
    add_xe_argument,
    compton_rates,
    hubble,
    xe_function,
)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("f_x", type=float, help="energy per hydrogen atom, eV")
    p.add_argument("gamma", type=float, help="decay rate, 1/s")
    p.add_argument("z_start", type=float)
    p.add_argument("z_end", type=float)
    add_xe_argument(p)
    args = p.parse_args()
    fx, gamx, zs, zend = args.f_x, args.gamma, args.z_start, args.z_end
    xe = xe_function(args.ledger, zend, zs)

    def t_of(z):
        return quad(
            lambda lz: 1 / hubble(np.exp(lz) - 1),
            np.log(1 + z),
            np.log(1 + 1e12),
            limit=500,
        )[0]

    lzg = np.linspace(np.log(1 + zend), np.log(1 + zs), 4000)
    tg = np.array([t_of(np.exp(v) - 1) for v in lzg])

    def tfun(z):
        return np.interp(np.log(1 + z), lzg, tg)

    def rhs(z, y):
        # y = [dT (K), E_gamma/rho_gamma delivered, E_adiabatic lost]; integrate in z (dt = -dz/(H(1+z)))
        dT = y[0]
        nh, ntot, tr, gam, rho_g = compton_rates(z, xe)
        hz = hubble(z)
        dtdz = -1.0 / (hz * (1 + z))
        # heating rate per unit time, relative to rho_gamma
        q_rel = fx * EV * nh * gamx * np.exp(-gamx * tfun(z)) / rho_g
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

    def dinj_dlz(lz):
        z = np.exp(lz) - 1
        q = fx * EV * NH0 * (1 + z) ** 3 * gamx * np.exp(-gamx * tfun(z))
        return q / (A_RAD * (T0 * (1 + z)) ** 4) / hubble(z)

    inj = quad(dinj_dlz, np.log(1 + zend), np.log(1 + zs), limit=400, epsrel=1e-10)[0]
    print(f"injected Delta rho/rho_gamma = {inj:.6e}")
    print(
        f"decay z_start={zs:g} z_end={zend:g}: delivered/injected = {eg / inj:.6f}; "
        f"adiabatic-excess loss = {ead / inj:.2e}; still in gas at z_end = {stored / inj:.2e}"
    )


if __name__ == "__main__":
    main()
