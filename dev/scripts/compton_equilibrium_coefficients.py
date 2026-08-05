#!/usr/bin/env python3
"""Analytic Compton-equilibrium temperature coefficients (Part II §II.2).

Derives the O(y) and O(mu) coefficients of the perturbative Compton-equilibrium
temperature response the solver uses,

    rho_e = I4 / (4 G3),   I4 = int x^4 n(1+n) dx,   G3 = int x^3 n dx,

linearized about a Planck spectrum n = n_pl + dn:

    d rho_eq = [int x^4 (1 + 2 n_pl) dn dx] / (4 G3)  -  [int x^3 dn dx] / G3

(CLAUDE.md Pitfall #4; src/electron_temp.rs -> spectrum::compton_equilibrium_ratio).

For dn = y * Y_SZ(x) and dn = mu * M(x) the response is d rho_eq = coeff * y (resp. mu).
This script computes those coefficients from the ANALYTIC shapes ONLY (nothing
imported from spectroxide) to >=12 digits via mpmath quadrature, and verifies the
number-conserving / energy sanity identities. Paste the printed constants (with
this derivation) into tests/compton_equilibrium_analytic.rs.

Run: python dev/scripts/compton_equilibrium_coefficients.py
"""

import mpmath as mp

mp.mp.dps = 40  # 40 decimal digits of working precision

# --- Analytic shapes (hardcoded; import nothing from spectroxide) -----------

# beta_mu = 3 zeta(3) / zeta(2) = 18 zeta(3) / pi^2  (src/constants.rs: 3 ZETA_3 / G1_PLANCK)
BETA_MU = 3 * mp.zeta(3) / mp.zeta(2)


def n_pl(x):
    """Planck occupation n_pl(x) = 1/(e^x - 1)."""
    return 1 / mp.expm1(x)


def g_bb(x):
    """Blackbody derivative G_bb(x) = x e^x / (e^x - 1)^2 = x n_pl (1 + n_pl)."""
    em1 = mp.expm1(x)
    return x * (1 + em1) / (em1 * em1)


def y_shape(x):
    """Zeldovich-Sunyaev y-distortion shape Y_SZ(x) = G_bb(x) [x coth(x/2) - 4]."""
    return g_bb(x) * (x * mp.coth(x / 2) - 4)


def mu_shape(x):
    """mu-distortion shape M(x) = (x/beta_mu - 1) G_bb(x) / x  (src/spectrum.rs::mu_shape)."""
    return (x / BETA_MU - 1) * g_bb(x) / x


def nn1(x):
    """n_pl (1 + n_pl) = G_bb(x)/x."""
    return g_bb(x) / x


# --- Quadrature over (0, inf), split to help mpmath near the small-x cusps ---

_SPLIT = [0, mp.mpf("0.5"), 2, 8, 30, mp.inf]


def integrate(f):
    return mp.quad(f, _SPLIT)


G3 = mp.pi ** 4 / 15  # int x^3 n_pl dx


def rho_eq_coeff(shape):
    """d rho_eq per unit amplitude for dn = amp * shape(x)."""
    dI4 = integrate(lambda x: x ** 4 * (1 + 2 * n_pl(x)) * shape(x))
    dG3 = integrate(lambda x: x ** 3 * shape(x))
    return dI4 / (4 * G3) - dG3 / G3


def main():
    mp.nprint  # noqa
    print("# Compton-equilibrium coefficients (mpmath, dps=%d)" % mp.mp.dps)
    print(f"beta_mu           = {mp.nstr(BETA_MU, 20)}")
    print(f"G3 = pi^4/15      = {mp.nstr(G3, 20)}")
    print()

    # Sanity identities for Y_SZ (must hold analytically).
    num_ysz = integrate(lambda x: x ** 2 * y_shape(x))          # expect 0
    en_ysz = integrate(lambda x: x ** 3 * y_shape(x))           # expect 4 G3
    print("# Y_SZ sanity identities")
    print(f"int x^2 Y_SZ dx   = {mp.nstr(num_ysz, 8)}   (expect 0, number-conserving)")
    print(f"int x^3 Y_SZ dx   = {mp.nstr(en_ysz, 16)}")
    print(f"           4 G3   = {mp.nstr(4 * G3, 16)}   (expect equal: d rho/rho = 4y)")
    print(f"  rel err (num)   = {mp.nstr(abs(num_ysz) / (4 * G3), 3)}")
    print(f"  rel err (energy)= {mp.nstr(abs(en_ysz - 4 * G3) / (4 * G3), 3)}")
    print()

    c_y = rho_eq_coeff(y_shape)
    c_mu = rho_eq_coeff(mu_shape)

    # mu-shape sanity: number and energy moments (informational).
    num_mu = integrate(lambda x: x ** 2 * mu_shape(x))
    en_mu = integrate(lambda x: x ** 3 * mu_shape(x))
    print("# mu-shape moments (informational)")
    print(f"int x^2 M dx      = {mp.nstr(num_mu, 12)}")
    print(f"int x^3 M dx      = {mp.nstr(en_mu, 12)}")
    print()

    print("# === Perturbative Compton-equilibrium response coefficients ===")
    print("# d rho_eq = COEFF_Y * y   for dn = y * Y_SZ(x)")
    print("# d rho_eq = COEFF_MU * mu for dn = mu * M(x)")
    print(f"COEFF_Y  = {mp.nstr(c_y, 20)}")
    print(f"COEFF_MU = {mp.nstr(c_mu, 20)}")
    print()
    print("# Constants to paste into tests/compton_equilibrium_analytic.rs (f64):")
    print(f"const COEFF_Y: f64 = {mp.nstr(c_y, 17)};")
    print(f"const COEFF_MU: f64 = {mp.nstr(c_mu, 17)};")


if __name__ == "__main__":
    main()
