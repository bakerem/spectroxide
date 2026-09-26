"""Independent expectation for burst-heat delivery when X_e responds to the gas temperature.

heat_delivery_expectation.py takes X_e(z) as a fixed input. This script instead
evolves the hydrogen ionization fraction with a Peebles three-level atom whose
recombination coefficient is evaluated at the gas temperature T_m, so a hot gas
recombines more slowly, keeps more free electrons, and couples to the photons
more strongly. Nothing is taken from spectroxide except the constants and
cosmology in heatloss_common.py (which contain no solver code).

State, integrated in z with dt = -dz / (H (1+z)):  [X_H, D = T_m - T_r, E].

Hydrogen (Peebles 1968; Seager, Sasselov & Scott 1999, RECFAST):

    dX_H/dt = -C [ alpha_B(T_m) n_H X_e X_H - beta_B(T_r) (1 - X_H) exp(-E_Lya / k T_r) ]
    alpha_B = F 1e-19 * 4.309 t^-0.6166 / (1 + 0.6703 t^0.5300) m^3/s,  t = T/1e4 K
              (Pequignot, Petitjean & Boisson 1991), F = 1.125 with the RECFAST 1.5.2 escape-rate correction (ADR 0011)
    beta_B  = alpha_B(T_r) (m_e k T_r / 2 pi hbar^2)^{3/2} exp(-E_2 / k T_r),  E_2 = E_H / 4
    C       = (K_esc + L_2s1s) / (K_esc + L_2s1s + beta_B)
    K_esc   = 8 pi H / (n_H (1 - X_H) lambda_Lya^3),  L_2s1s = 8.2246 s^-1
    E_H     = Ry / (1 + m_e/m_p) = 13.5983 eV,  E_Lya = 3 E_H / 4  (lambda_Lya = 121.567 nm)

X_H is the ionized (free-proton) fraction per hydrogen nucleus. With T_m = T_r
the bracket vanishes at hydrogen Saha equilibrium, since E_2 + E_Lya = E_H.

Helium: Saha equilibrium at T_r, solved jointly with the total electron
density (RECFAST style), with n_e (He III/He II) = lambda^-3 exp(-54.418 eV/kT)
and n_e (He II/He I) = 4 lambda^-3 exp(-24.587 eV/kT), lambda^-3 = (2 pi m_e k T/h^2)^{3/2}.
Saha helium is neutral to < 1e-4 below z = 2000, so it only enters above that.
Real He I recombination lags Saha (HyRec-2 has X_e = 1.037 at z = 2000); this
changes n_e at z > 1700 only, where Gamma_C/H > 1e5 anyway.

Gas temperature (the full T_m, written for D = T_m - T_r because the
adiabatic term makes T_m - T_r a near-cancellation):

    dT_m/dt = Q / (3/2 k n_tot) + Gamma_C (T_r - T_m) - 2 H T_m
    dD/dt   = Q / (3/2 k n_tot) - Gamma_C D - 2 H D - H T_r
    Gamma_C = 8 sigma_T a T_r^4 n_e / (3 m_e c n_tot),  n_tot = n_e + n_H + n_He,  n_e = X_e n_H
    Q / rho_gamma = drho * Gauss(z; z_h, sigma) * H (1+z)      (so int Q/rho_gamma dt = drho)
    dE/dt   = (3/2) k n_tot Gamma_C D / rho_gamma              (photon energy gain per rho_gamma)

Delivered fraction = [E(Q) - E(Q = 0)] / drho. The Q = 0 run carries the
adiabatic-cooling baseline; both runs are integrated as one 6-component system
so they share the same steps. With --fixed-xe the heated run uses the Q = 0
run's X_H (alpha_B at the no-injection T_m), which separates the coupling.
With --table-xe both runs use an X_H history evolved with alpha_B at T_r, the
standard table that spectroxide's --fixed-ionization mode reads; compare that
mode against this one.

Conventions: T_0 = 2.726 K (as in the solver; heatloss_common uses 2.7255 K,
the background here is rebuilt with 2.726 K). Burst width sigma_z = max(0.04 z_h, 100)
unless --sigma is given; the run starts at z_start = max(z_h + 7 sigma, 2000)
with X_H, helium in Saha at T_r and T_m = T_r; the default z_end is 200.

Reference numbers (z_h = 1000, sigma = 100, z_end = 200, linear limit):
    --fixed-xe  -> 0.9998635  (heat_delivery_expectation.py: 0.999863)
    --table-xe  -> 0.9998635
    coupled     -> 0.9998562 at drho = 1e-8 (with the ADR 0011 escape-rate correction)

Limitation: the three-level atom has no collisional ionization or collisional
excitation cooling. Both matter once T_m exceeds about 1e4 K; along the heated
trajectories with peak T_m/T_r >~ 5 the omitted collisional ionization rate
(Cen 1992) exceeds the Hubble rate by 1e2 to 1e7. Results there are the answer
for this model, not for the full physics.

Usage:
    python3 coupled_xe_expectation.py Z_H [--drho 1e-8] [--z-end 200] [--sigma S]
                                          [--fixed-xe | --table-xe] [--rtol 1e-10]
                                          [--report-z 1100,800,500,200,100] [--hyrec FILE]
"""

import argparse

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from heatloss_common import (
    A_RAD,
    C,
    EV,
    FHE,
    G_N,
    H,
    K_B,
    M_E,
    M_P,
    NEFF,
    NH0,
    OMM,
    SIGMA_T,
    H0,
)

T0 = 2.726
RHO_CRIT = 3 * H0**2 / (8 * np.pi * G_N)
OMG = A_RAD * T0**4 / C**2 / RHO_CRIT
OMR = OMG * (1 + 7 / 8 * (4 / 11) ** (4 / 3) * NEFF)
OML = 1 - OMM - OMR

H_PL = 6.62607015e-34  # J s (exact)
RYD_EV = 13.605693122994  # R_inf h c, CODATA 2018
E_H = RYD_EV * EV / (1 + M_E / M_P)  # reduced-mass hydrogen ground state
E_2 = E_H / 4
E_LYA = 0.75 * E_H
LAM_LYA = 121.567e-9  # m, as in RECFAST; hc/E_LYA = 121.568 nm (E_LYA = E_H - E_2 keeps Saha the fixed point)
L2S1S = 8.2246  # s^-1
FUDGE = 1.125
CHI_HEI = 24.587387 * EV
CHI_HEII = 54.417760 * EV


def hubble(z):
    return H0 * np.sqrt(OMR * (1 + z) ** 4 + OMM * (1 + z) ** 3 + OML)


def alpha_b(t_k):
    t = t_k / 1e4
    return FUDGE * 1e-19 * 4.309 * t**-0.6166 / (1 + 0.6703 * t**0.5300)


def lam3inv(t_k):
    """(2 pi m_e k T / h^2)^{3/2} in m^-3."""
    return (2 * np.pi * M_E * K_B * t_k / H_PL**2) ** 1.5


def helium_electrons(ne, tr):
    """Free electrons per helium nucleus from Saha at T_r for electron density ne."""
    l3 = lam3inv(tr)
    kt = K_B * tr
    r1 = 4 * l3 * np.exp(-CHI_HEI / kt) / ne  # HeII/HeI
    r2 = l3 * np.exp(-CHI_HEII / kt) / ne  # HeIII/HeII
    y1 = 1 / (1 + r1 + r1 * r2)
    return y1 * (r1 + 2 * r1 * r2)


def helium_per_h(xh, z):
    """Helium electrons per H nucleus, Saha at T_r, consistent with n_e = n_H (X_H + that)."""
    tr = T0 * (1 + z)
    nh = NH0 * (1 + z) ** 3
    if tr < 4000:  # z < 1467: Saha He electrons < 1e-30 per H
        return 0.0
    f = lambda xe: xe - xh - FHE * helium_electrons(xe * nh, tr)
    return brentq(f, xh, xh + 2 * FHE + 1e-300, xtol=1e-16, rtol=1e-15) - xh


def saha_xh(z):
    tr = T0 * (1 + z)
    nh = NH0 * (1 + z) ** 3
    s = lam3inv(tr) * np.exp(-E_H / (K_B * tr))

    def f(xe):
        xh = s / (s + xe * nh)
        return xe - xh - FHE * helium_electrons(xe * nh, tr)

    xe = brentq(f, 1e-6, 1 + 2 * FHE, xtol=1e-16, rtol=1e-15)
    return s / (s + xe * nh)


def rates(z, xh, d):
    """Returns (dX_H/dt, dD/dt without Q, dE/dt, cap, x_e)."""
    nh = NH0 * (1 + z) ** 3
    tr = T0 * (1 + z)
    tm = tr + d
    hz = hubble(z)
    xe = xh + helium_per_h(xh, z)
    ne = xe * nh
    ntot = ne + nh * (1 + FHE)
    # Peebles
    ab_m = alpha_b(tm)
    bb = alpha_b(tr) * lam3inv(tr) * np.exp(-E_2 / (K_B * tr))
    x1s = max(1 - xh, 1e-300)
    # RECFAST 1.5.2 escape-rate correction, divides the Sobolev rate (ADR 0011)
    lnz = np.log(1 + z)
    gauss = 1 - 0.14 * np.exp(-(((lnz - 7.28) / 0.18) ** 2)) + 0.079 * np.exp(-(((lnz - 6.73) / 0.33) ** 2))
    kesc = 8 * np.pi * hz / (nh * x1s * LAM_LYA**3) / gauss
    cp = (kesc + L2S1S) / (kesc + L2S1S + bb)
    dxh = -cp * (ab_m * nh * xe * xh - bb * x1s * np.exp(-E_LYA / (K_B * tr)))
    # temperature
    rho_g = A_RAD * tr**4
    gam = 8 * SIGMA_T * rho_g * ne / (3 * M_E * C * ntot)
    cap = 1.5 * K_B * ntot / rho_g
    dd = -gam * d - 2 * hz * d - hz * tr
    de = cap * gam * d
    return dxh, dd, de, cap, xe


def solve(zh, drho, sig, zend, fixed_xe, rtol, t_eval=None, table_xe=False):
    zs = max(zh + 7 * sig, 2000.0)
    xh0 = saha_xh(zs)
    shared = fixed_xe or table_xe

    def rhs(z, y):
        hz = hubble(z)
        dtdz = -1.0 / (hz * (1 + z))
        b = rates(z, y[0], y[1])
        # table_xe: the shared X_H recombines with alpha_B at T_r (D = 0)
        dxh_b = rates(z, y[0], 0.0)[0] if table_xe else b[0]
        xh_h = y[0] if shared else y[3]
        # with a shared X_H the heated run's electrons are the baseline's; alpha_B(T_m) is then irrelevant
        h = rates(z, xh_h, y[4])
        q_rel = drho * np.exp(-((z - zh) ** 2) / (2 * sig**2)) / np.sqrt(2 * np.pi) / sig * hz * (1 + z)
        dxh_h = dxh_b if shared else h[0]
        return np.array(
            [dxh_b, b[1], b[2], dxh_h, h[1] + q_rel / h[3], h[2]]
        ) * dtdz

    y0 = [xh0, 0.0, 0.0, xh0, 0.0, 0.0]
    atol = [1e-16, 1e-12, 1e-24, 1e-16, 1e-12, 1e-24]
    # np.errstate: scipy's finite-difference Jacobian overflows its step factor on
    # columns nothing depends on (E); the warning is harmless.
    with np.errstate(over="ignore"):
        sol = solve_ivp(rhs, (zs, zend), y0, method="Radau", rtol=rtol, atol=atol,
                        t_eval=t_eval, dense_output=False)
    if not sol.success:
        raise RuntimeError(sol.message)
    return zs, sol


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("z_h", type=float, help="burst redshift")
    p.add_argument("--drho", type=float, default=1e-8, help="injected Delta rho/rho (0 allowed)")
    p.add_argument("--z-end", type=float, default=200.0)
    p.add_argument("--sigma", type=float, default=None,
                   help="burst width in z (default: max(0.04 z_h, 100))")
    p.add_argument("--fixed-xe", action="store_true",
                   help="heated run uses the Q = 0 run's X_H history")
    p.add_argument("--table-xe", action="store_true",
                   help="both runs use X_H evolved with alpha_B at T_r (the standard table)")
    p.add_argument("--rtol", type=float, default=1e-10)
    p.add_argument("--report-z", default="", help="comma list of z to print X_e, T_m/T_r (Q = 0 and heated)")
    p.add_argument("--hyrec", metavar="FILE", help="HyRec-2 table (z, X_e, T_m) to compare the Q = 0 run at --report-z")
    args = p.parse_args()
    zh, drho, zend = args.z_h, args.drho, args.z_end
    sig = max(0.04 * zh, 100.0) if args.sigma is None else args.sigma
    if not sig > 0:
        p.error("--sigma must be positive")
    zs = max(zh + 7 * sig, 2000.0)
    zr = [float(s) for s in args.report_z.split(",") if s]
    grid = np.unique(np.concatenate([np.linspace(zs, zend, 4001), zr]))[::-1]
    if args.fixed_xe and args.table_xe:
        p.error("--fixed-xe and --table-xe are exclusive")
    zs, sol = solve(zh, drho, sig, zend, args.fixed_xe, args.rtol, t_eval=grid, table_xe=args.table_xe)
    y = sol.y
    xe_b = np.array([y[0, i] + helium_per_h(y[0, i], z) for i, z in enumerate(sol.t)])
    xh_h = y[0] if (args.fixed_xe or args.table_xe) else y[3]
    xe_h = np.array([xh_h[i] + helium_per_h(xh_h[i], z) for i, z in enumerate(sol.t)])
    tr = T0 * (1 + sol.t)
    peak = np.max(1 + y[4] / tr)
    frac = (y[5, -1] - y[2, -1]) / drho if drho != 0 else float("nan")
    mode = "fixed-xe" if args.fixed_xe else "table-xe" if args.table_xe else "coupled"
    print(
        f"z_h={zh:g} sigma={sig:g} drho={drho:g} z_start={zs:g} z_end={zend:g} [{mode}, rtol={args.rtol:g}]: "
        f"delivered/injected = {frac:.7f}; X_e(z_end) heated = {xe_h[-1]:.5e}, Q=0 = {xe_b[-1]:.5e}; "
        f"peak T_m/T_r = {peak:.5g}; E(Q=0) = {y[2, -1]:.4e}"
    )
    if zr:
        ref = np.loadtxt(args.hyrec) if args.hyrec else None
        for z in zr:
            i = int(np.argmin(np.abs(sol.t - z)))
            line = (f"  z={z:g}: X_e Q=0 {xe_b[i]:.5e}, T_m/T_r Q=0 {1 + y[1, i] / tr[i]:.5f}; "
                    f"heated X_e {xe_h[i]:.5e}, T_m/T_r {1 + y[4, i] / tr[i]:.5f}")
            if ref is not None:
                j = int(np.argmin(np.abs(ref[:, 0] - z)))
                line += (f" | HyRec-2 X_e {ref[j, 1]:.5e} (ratio {xe_b[i] / ref[j, 1]:.4f}), "
                         f"T_m/T_r {ref[j, 2] / tr[i]:.5f}")
            print(line)


if __name__ == "__main__":
    main()
