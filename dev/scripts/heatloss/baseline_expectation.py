"""Independent expectation for the no-injection adiabatic-cooling photon energy loss.

The gas temperature T feels Compton coupling and adiabatic cooling
(monatomic gas, T proportional to a^-2):

    dT/dt = -2 H T + Gamma_C (T_r - T),  Gamma_C = 8 sigma_T a T_r^4 n_e / (3 m_e c n_tot)
    d(Delta rho/rho_gamma)/dt = (3/2) k n_tot Gamma_C (T - T_r) / rho_gamma

Above z = 3000, where Gamma_C/H > 1e5, the excess is quasi-stationary,
T - T_r = -H T_r / (Gamma_C + 2H), and the loss is integrated directly. DC/BR
energy exchange is neglected. Constants, cosmology, and the X_e(z) source are in
heatloss_common.py. Unlike a delivered fraction, this absolute Delta rho/rho
depends on T_0 through n_tot/rho_gamma.

Baseline of the burst test in tests/heat_delivery.rs (from z = 1700):
    python baseline_expectation.py 1700 200                  ->  -6.109e-10
    python baseline_expectation.py 1700 200 --ledger FILE    ->  -6.105e-10
The two differ by 7e-4 because the ledger has only its step points in X_e and
linear interpolation between them misses X_e by up to 1.2%. The default is the
better value; dev/audit/fix_a_cn_old_half_ab.md quotes the ledger one.

Usage: python baseline_expectation.py Z_START Z_END [--ledger FILE]
"""

import argparse

import numpy as np
from scipy.integrate import solve_ivp

from heatloss_common import K_B, add_xe_argument, compton_rates, hubble, xe_function

ZQS = 3000.0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("z_start", type=float)
    p.add_argument("z_end", type=float)
    add_xe_argument(p)
    args = p.parse_args()
    zs, ze = args.z_start, args.z_end
    xe = xe_function(args.ledger, ze, zs)

    def rhs(lz, y):
        # y = [T - T_r (K), Delta rho/rho_gamma]; integrate in ln(1+z)
        z = np.exp(lz) - 1
        dT = y[0]
        nh, ntot, tr, gam, rho_g = compton_rates(z, xe)
        hz = hubble(z)
        dtdlz = -1.0 / hz
        cap = 1.5 * K_B * ntot / rho_g
        d0 = (-2 * hz * dT - hz * tr - gam * dT) * dtdlz
        d1 = cap * gam * dT * dtdlz
        return [d0, d1]

    def quasi_stationary(z):
        # Returns (T - T_r, d(Delta rho/rho)/d ln(1+z)).
        nh, ntot, tr, gam, rho_g = compton_rates(z, xe)
        hz = hubble(z)
        cap = 1.5 * K_B * ntot / rho_g
        dT = -hz * tr / (gam + 2 * hz)
        return dT, cap * gam * dT / hz

    e0, z0, dT0 = 0.0, zs, 0.0
    if zs > ZQS:
        lz = np.linspace(np.log(1 + ZQS), np.log(1 + zs), 200001)
        vals = quasi_stationary(np.exp(lz) - 1)[1]
        # d(Delta rho) = rate dt with dt = -d ln(1+z) / H, from zs down to ZQS
        e0 = np.sum(0.5 * (vals[1:] + vals[:-1]) * np.diff(lz))
        z0 = ZQS
        dT0 = quasi_stationary(z0)[0]
    sol = solve_ivp(
        rhs,
        (np.log(1 + z0), np.log(1 + ze)),
        [dT0, e0],
        method="Radau",
        rtol=1e-10,
        atol=[1e-16, 1e-24],
    )
    rho_end = 1 + sol.y[0, -1] / (compton_rates(ze, xe)[2])
    print(
        f"z_start={zs:g} z_end={ze:g}: expected baseline drho/rho = {sol.y[1, -1]:.6e}  "
        f"(T - T_r at z_end = {sol.y[0, -1]:.4e} K, T/T_r = {rho_end:.4f})"
    )


if __name__ == "__main__":
    main()
