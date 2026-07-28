#!/usr/bin/env python3
r"""
Clean-room reference solver for CMB spectral distortions  (workstream R3).

Independent N-version cross-check.  Implements a Chang-Cooper (1970)
finite-volume discretisation of the Kompaneets operator for the FULL photon
occupation number n(x, z) (not a distortion), following ONLY

  * dev/refsolver/contract.md
  * the raw arXiv LaTeX source of Chluba & Sunyaev 2012 (arXiv:1109.6552)

Isolation: no spectroxide solver source (src/*.rs, python/spectroxide/*.py,
tests/, notebooks/, dev/audit/) was read.  See README.md.

------------------------------------------------------------------------------
PHYSICS
------------------------------------------------------------------------------
x = h nu / (k T_gamma),  T_gamma = T_cmb (1+z)  exactly (so free expansion
leaves n(x) invariant and there is no explicit redshift term).

Kompaneets (CS2012 Eq. 9):
    dn/dtau|_C = (theta_e / x^2) d/dx [ x^4 ( dn/dx + phi n(1+n) ) ]
    phi = T_gamma/T_e,  theta_e = k T_e/(m_e c^2),  dtau = sigma_T n_e c dt.

Discretisation.  Change variables to  g = n/(1+n)  (so n = g/(1-g)):
    dn/dx = (1+n)^2 dg/dx    =>    F = x^4 (1+n)^2 [ dg/dx + phi g ].
F is then a *linear* Fokker-Planck flux in g with unit diffusion and constant
drift phi, whose exact zero is g = exp(-phi x), i.e. n = 1/(exp(phi x) - 1).
Chang-Cooper weighting of the drift term,
    delta(w) = 1/w - 1/(e^w - 1),   w = phi * (x_{j+1} - x_j),
makes the *discrete* interface flux vanish identically for that g, so the
discrete Bose-Einstein spectrum at T_e is an exact fixed point of the scheme
(verified to machine precision by self_test_equilibrium()).

Emission / absorption (CS2012 Eq. 12):
    dn/dtau|_em = (K_DC + K_BR)/x^3 * [ 1 - n (e^{phi x} - 1) ]

  K_DC (CS2012 Eqs. 13-16):
      K_DC = (4 alpha / 3 pi) theta_g^2 * I4pl/(1 + 14.16 theta_g) * H_dc(x)
      H_dc(x) = e^{-2x} (1 + 3x/2 + 29x^2/24 + 11x^3/16 + 5x^4/12)
      I4pl = 4 pi^4/15

  K_BR (CS2012 Eq. 17, verbatim):
      K_BR = alpha lambda_e^3 / (2 pi sqrt(6 pi)) * theta_e^{-7/2} e^{-phi x}
             / phi^3 * sum_i Z_i^2 N_i g_ff(Z_i, x, theta_e)
      with g_ff taken as the *Born-limit* thermally averaged free-free Gaunt
      factor,  g_ff = (sqrt(3)/pi) e^{u/2} K_0(u/2),  u = phi x = h nu/k T_e
      (Karzas & Latter 1961 Born limit; Rybicki & Lightman 1979 Sect. 5.2).
      This is ~10-20% accurate at small x compared with the Itoh (2000) fits
      that CS2012 actually use -- see README.

Electron temperature (CS2012 Eq. 21 / 22), integrated with backward Euler
rather than assumed quasi-stationary:
    d rho_e/dtau = beta_C (rho_eq - rho_e) - H t_C rho_e + beta_C * dri(z)
    rho_e = T_e/T_gamma,  phi = 1/rho_e
    rho_eq = I_4/(4 G_3),  I_4 = int x^4 n(1+n) dx,  G_3 = int x^3 n dx
    beta_C = 4 rho_gamma_tilde / alpha_h,  rho_gamma_tilde = kappa_g theta_g^4 G_3
    kappa_g = 8 pi / lambda_e^3,  alpha_h = (3/2) N_H (1 + f_He + X_e)
    dri = t_C Qdot / (4 rho_gamma_tilde theta_g)   [the heat-injection term]
The DC/BR matter cooling integral H_DC,BR of CS2012 Eq. 23 is *omitted*; its
size is measured and reported as a diagnostic (see README).

Energy bookkeeping identity used for the injection normalisation (exact):
    d ln G_3 / dtau = 4 theta_g (rho_e - rho_eq)
so  int 4 theta_g dri dtau  is the fractional photon energy release.
"""

import json
import os
import sys
import time

import numpy as np
from scipy.integrate import simpson
from scipy.linalg import solve_banded
from scipy.special import kv

# --------------------------------------------------------------------------
# constants (CODATA 2018, SI)
# --------------------------------------------------------------------------
K_B = 1.380649e-23           # J/K
H_PL = 6.62607015e-34        # J s
C_L = 2.99792458e8           # m/s
M_E = 9.1093837015e-31       # kg
MEC2 = M_E * C_L ** 2        # J
ALPHA_FS = 7.2973525693e-3
SIGMA_T = 6.6524587321e-29   # m^2
LAMBDA_E = H_PL / (M_E * C_L)          # Compton wavelength [m]
KAPPA_G = 8.0 * np.pi / LAMBDA_E ** 3  # [1/m^3]  ~ 1.760e36

ZETA3 = 1.2020569031595943
G3_PL = np.pi ** 4 / 15.0        # int x^3 n_pl dx
I4_PL = 4.0 * np.pi ** 4 / 15.0  # int x^4 n_pl(1+n_pl) dx
N2_PL = 2.0 * ZETA3              # int x^2 n_pl dx
BETA_MU = 3.0 * ZETA3 / (np.pi ** 2 / 6.0)   # 3 zeta(3)/zeta(2) = 2.192289

T_CMB = 2.726
Y_P = 0.24
F_HE = (Y_P / 4.0) / (1.0 - Y_P)     # n_He/n_H = 0.078947

BR_PREF = ALPHA_FS * LAMBDA_E ** 3 / (2.0 * np.pi * np.sqrt(6.0 * np.pi))
SQRT3_OVER_PI = np.sqrt(3.0) / np.pi

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY_CSV = os.path.join(HERE, "inputs", "history.csv")


# --------------------------------------------------------------------------
# frozen background history
# --------------------------------------------------------------------------
class History:
    """Log-z interpolation of the frozen ingredient table (used as given)."""

    def __init__(self, path=HISTORY_CSV):
        d = np.genfromtxt(path, delimiter=",", names=True)
        o = np.argsort(d["z"])
        self.lz = np.log(d["z"][o])
        self._x_e = d["x_e"][o]
        self._log = {
            "H": np.log(d["H_z_per_s"][o]),
            "n_e": np.log(d["n_e_per_m3"][o]),
            "n_H": np.log(d["n_H_per_m3"][o]),
            "Tg": np.log(d["T_gamma_K"][o]),
            "tC": np.log(d["t_C_s"][o]),
        }
        self.z_min = float(np.exp(self.lz[0]))
        self.z_max = float(np.exp(self.lz[-1]))

    def at(self, z):
        lz = np.log(z)
        out = {k: float(np.exp(np.interp(lz, self.lz, v))) for k, v in self._log.items()}
        out["x_e"] = float(np.interp(lz, self.lz, self._x_e))
        out["z"] = float(z)
        out["theta_g"] = K_B * out["Tg"] / MEC2
        # sum_i Z_i^2 N_i, reconstructed from x_e alone by the standard
        # recombination ladder (He++ -> He+ near z ~ 6e3, He+ -> He near
        # z ~ 2.5e3, H+ -> H near z ~ 1.3e3), i.e. electrons are given up in
        # order of decreasing binding energy.  With x_e = n_e/N_H:
        #   x_e > 1 + f_He :  H fully ionised, He a He++/He+ mix with He++
        #                     fraction a = (x_e-1)/f_He - 1, so
        #                     sum Z^2 N/N_H = 1 + f_He(1+3a) = 3 x_e - 2 - 2 f_He
        #   x_e <= 1 + f_He:  at most singly ionised He (and then partially
        #                     ionised H), for which every ion has Z = 1 and
        #                     sum Z^2 N/N_H = x_e.
        # At full ionisation this gives N_H(1+4 f_He) = 1.3157 N_H, the nucleon
        # number density -- the simplification CS2012 quote as sum ~ g_ff N_b.
        # The naive "He always He++" form, N_H(x_e + 2 f_He), overestimates by
        # up to 15% for 1500 < z < 5500 and 28% at z ~ 1300.
        if out["x_e"] > 1.0 + F_HE:
            out["sumZ2N"] = out["n_H"] * (3.0 * out["x_e"] - 2.0 - 2.0 * F_HE)
        else:
            out["sumZ2N"] = out["n_H"] * out["x_e"]
        out["alpha_h"] = 1.5 * out["n_H"] * (1.0 + F_HE + out["x_e"])
        out["HtC"] = out["H"] * out["tC"]
        return out


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------
class Grid:
    """Logarithmic node grid with finite-volume cells (edges = geometric means)."""

    def __init__(self, N=2049, xmin=1e-4, xmax=40.0):
        if N % 2 == 0:
            N += 1                      # odd N -> even #intervals, clean Simpson
        self.N = N
        self.xmin, self.xmax = xmin, xmax
        self.x = np.logspace(np.log10(xmin), np.log10(xmax), N)
        self.u = np.log(self.x)
        self.xh = np.sqrt(self.x[:-1] * self.x[1:])      # interfaces, len N-1
        edges = np.empty(N + 1)
        edges[1:-1] = self.xh
        edges[0] = self.x[0] ** 2 / self.xh[0]
        edges[-1] = self.x[-1] ** 2 / self.xh[-1]
        self.dxc = edges[1:] - edges[:-1]                # cell widths, len N
        self.dxn = self.x[1:] - self.x[:-1]              # node spacing, len N-1
        self.xh4 = self.xh ** 4
        self.x2 = self.x ** 2
        self.x3 = self.x ** 3
        self.x4 = self.x ** 4
        self.n_pl = 1.0 / np.expm1(self.x)
        # DC polynomial part of H_dc, exponential factored out
        self.dc_poly = (1.0 + 1.5 * self.x + (29.0 / 24.0) * self.x ** 2
                        + (11.0 / 16.0) * self.x ** 3 + (5.0 / 12.0) * self.x ** 4)
        self.emx2 = np.exp(-2.0 * self.x)

    # ---- moments.  Simpson in u = ln x  (int f dx = int f x du) ----
    def G3(self, n):
        return simpson(self.x4 * n, x=self.u)

    def I4(self, n):
        return simpson(self.x ** 5 * n * (1.0 + n), x=self.u)

    def Nph_simpson(self, n):
        return simpson(self.x3 * n, x=self.u)

    def Nph_cell(self, n):
        """Cell-rule photon number: EXACTLY conserved by the discrete Compton
        operator with zero-flux boundaries (see module docstring)."""
        return float(np.sum(self.x2 * n * self.dxc))


# --------------------------------------------------------------------------
# emission coefficients
# --------------------------------------------------------------------------
def cc_delta(w):
    """Chang-Cooper weight  delta = 1/w - 1/(e^w - 1)  (limit 1/2 at w -> 0)."""
    out = np.empty_like(w)
    small = np.abs(w) < 1e-8
    big = w > 500.0
    mid = ~small & ~big
    out[small] = 0.5 - w[small] / 12.0
    out[big] = 1.0 / w[big]
    ws = w[mid]
    out[mid] = 1.0 / ws - 1.0 / np.expm1(ws)
    return out


def _psi(n):
    """psi = ln[ n/(1+n) ], evaluated without cancellation for all n."""
    ns = np.maximum(n, 1e-280)
    out = np.empty_like(ns)
    big = ns >= 1.0
    out[big] = -np.log1p(1.0 / ns[big])
    sm = ~big
    out[sm] = np.log(ns[sm]) - np.log1p(ns[sm])
    return out


def cc_flux(g, n, phi):
    r"""Chang-Cooper interface flux of the Kompaneets operator, in the exact
    cancellation-free form

        F_j = P_j g_j (phi / expm1(w_j)) expm1(Delta psi_j + w_j),
        w_j = phi (x_{j+1}-x_j),  psi = ln[n/(1+n)],
        Delta psi_j = psi_{j+1} - psi_j,  P_j = x_{j+1/2}^4 (1+n_j)(1+n_{j+1}).

    This is algebraically identical to
        F_j = P_j [ (gg_{j+1}-gg_j)/dx + phi((1-delta)gg_{j+1} + delta gg_j) ]
    with delta = 1/w - 1/(e^w-1)  (substitute delta and factor out gg_j), but it
    vanishes to machine precision at the discrete equilibrium psi = -phi x for
    ALL x: writing the flux via gg = n/(1+n) loses ~11 digits to cancellation at
    small x (gg -> 1) and via 1/(1+n) at large x, whereas Delta psi + w is a
    difference of two O(w) quantities only.
    Returns (F, P, hh) with hh = 1/(1+n) for the Jacobian.
    """
    opn = 1.0 + n
    hh = 1.0 / opn
    gg = n * hh
    psi = _psi(n)
    w = phi * g.dxn
    P = g.xh4 * (opn[:-1] * opn[1:])
    A = np.clip((psi[1:] - psi[:-1]) + w, -600.0, 600.0)
    F = P * gg[:-1] * (phi / np.expm1(w)) * np.expm1(A)
    return F, P, hh, w


def flux_moments(g, n, phi):
    r"""The moments 4*G3 and I4 in the discretisation induced by the flux sum.

    Summation by parts on the discrete Compton operator with zero-flux
    boundaries gives, EXACTLY,
        Delta G3(cell rule) = -theta_e dtau sum_k (x_{k+1}-x_k) F_k
                            = 4 theta_g G3d dtau (rho_e - I4d/(4 G3d))
    with
        4 G3d = -sum_k P_k (gg_{k+1}-gg_k) = sum_k x_{k+1/2}^4 (n_k - n_{k+1})
        I4d   =  sum_k (x_{k+1}-x_k) P_k ggbar_k
              =  sum_k dx_k x_{k+1/2}^4 [(1-d)n_{k+1}(1+n_k) + d n_k(1+n_{k+1})]
    (both cancellation-free).  Using rho_eq = I4d/(4 G3d) as the Compton
    equilibrium temperature therefore makes the *discrete* Compton energy
    transfer vanish identically at rho_e = rho_eq, and makes the energy
    delivered by a heat source exactly 4 theta_g dri dtau at ANY step size.
    Using instead the Simpson-quadrature I4/(4 G3) leaves a residual mismatch
    which, multiplied by the huge theta_g*dtau of the mu era, truncates the
    injected energy badly (measured: only 20% delivered for z_h = 2e6).
    Consistency: for a discrete Planck at T_e the Chang-Cooper flux vanishes
    interface by interface, so rho_eq = T_e/T_gamma exactly (to round-off).
    """
    opn = 1.0 + n
    dl = cc_delta(phi * g.dxn)
    G3d4 = float(np.sum(g.xh4 * (n[:-1] - n[1:])))
    I4d = float(np.sum(g.dxn * g.xh4 *
                       ((1.0 - dl) * n[1:] * opn[:-1] + dl * n[:-1] * opn[1:])))
    return 0.25 * G3d4, I4d


def gaunt_ff_born(u):
    """Born-limit thermally averaged free-free Gaunt factor,
    g_ff = (sqrt3/pi) e^{u/2} K_0(u/2),  u = h nu / k T_e.
    Small-u limit (sqrt3/pi) ln(4 e^{-gamma_E}/u) used below u = 1e-8 for
    numerical safety; e^{u/2} K_0(u/2) is evaluated via scipy.special.kve."""
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    tiny = u < 1e-10
    out[tiny] = SQRT3_OVER_PI * (np.log(4.0 / np.maximum(u[tiny], 1e-300)) - 0.5772156649015329)
    v = 0.5 * u[~tiny]
    # kve(0, v) = exp(v) K_0(v) : exponentially scaled, no overflow
    from scipy.special import kve
    out[~tiny] = SQRT3_OVER_PI * kve(0, v)
    return out


def emission_terms(g, phi, theta_g, theta_e, sumZ2N, use_dc=True, use_br=True):
    """Return (S, Gamma) with  dn/dtau|_em = S - Gamma * n.

    Exponentials are factored analytically so nothing overflows:
      DC:  (c/x^3)[e^{-2x} - n(e^{(phi-2)x} - e^{-2x})]
      BR:  (c/x^3)[e^{-phi x} - n(1 - e^{-phi x})]
    """
    x = g.x
    S = np.zeros(g.N)
    Gam = np.zeros(g.N)
    # Validity gate on DC.  The absorption rate implied by Eq. 12 is
    # Gamma_DC = (K_DC/x^3)(e^{phi x} - 1) with K_DC ~ e^{-2x}, so for phi > 2 it
    # GROWS exponentially with x.  That is an artefact: the DC Gaunt factor
    # g_dc was derived for a blackbody ambient field with T_e ~ T_gamma
    # (CS2012 Sect. 2.2.1), which fails once the electrons decouple thermally
    # (phi > 2 below z ~ 70, phi = 60 by z = 1).  Measured: without the gate,
    # Gamma_DC*dtau > 1 for x > 1 by z ~ 7, which would erase the spectrum
    # inside the fit window, whereas the physical DC rate there is
    # K_DC/x^3 * dtau <~ 1e-15.  BR needs no gate: its factor
    # e^{-phi x}(e^{phi x} - 1) = 1 - e^{-phi x} <= 1 is bounded.
    # Irrelevant for the contract's z_end = 200, where phi ~ 1.15.
    if phi > 2.0:
        use_dc = False
    if use_dc:
        c = ((4.0 * ALPHA_FS / (3.0 * np.pi)) * theta_g ** 2
             * (I4_PL / (1.0 + 14.16 * theta_g)) * g.dc_poly / g.x3)
        e2 = g.emx2
        ep = np.exp(np.clip((phi - 2.0) * x, -745.0, 600.0))
        S += c * e2
        Gam += c * (ep - e2)
    if use_br:
        u = phi * x
        gff = gaunt_ff_born(u)
        c = (BR_PREF * theta_e ** (-3.5) / phi ** 3 * sumZ2N * gff / g.x3)
        emu = np.exp(-np.clip(u, 0.0, 745.0))
        S += c * emu
        Gam += c * (1.0 - emu)
    return S, Gam


# --------------------------------------------------------------------------
# one implicit-Euler step of (Compton + emission), coupled to rho_e
# --------------------------------------------------------------------------
def implicit_step(n_old, g, dtau, hz, rho_e_old, dri=0.0, src=0.0, cooling=True,
                  use_dc=True, use_br=True, rho_eq_mode="direct",
                  max_iter=60, tol=1e-13, diag=None):
    """Advance n and rho_e by one backward-Euler step of dtau.

    Photons:
        (n - n_old)/dtau = Compton[n; phi] + (S - Gamma n) + src
    Electrons (CS2012 Eq. 21, backward Euler):
        (rho_e - rho_e_old)/dtau
            = beta_C(rho_eq(n) + dri - H_dcbr - rho_e) - HtC rho_e
    with rho_eq from flux_moments() and H_dcbr the CS2012 Eq. 23 DC/BR matter
    cooling integral.  rho_e is re-solved inside the iteration from the current
    n iterate, so the photon and electron equations converge together and end up
    consistent at a single phi.  `dri` is the heat-injection term
    t_C Qdot/(4 rho_gamma_tilde theta_g); `src` is a photon source in dn/dtau.
    """
    x, dxn, dxc, N = g.x, g.dxn, g.dxc, g.N
    theta_g = hz["theta_g"]
    alpha_h = hz["alpha_h"]
    HtC = hz["HtC"] if cooling else 0.0
    sumZ2N = hz["sumZ2N"]

    n = n_old.copy()
    rho_e = rho_e_old
    it = 0
    prev = np.inf
    conv = np.inf
    capped = False
    H_dcbr = 0.0
    rho_eq = rho_e_old
    beta_C = 0.0
    for it in range(1, max_iter + 1):
        # Picard: everything in this iteration uses phi from the current rho_e;
        # rho_e is refreshed at the end from the updated n.  On convergence the
        # photon and electron equations are consistent at a single phi.
        phi = 1.0 / rho_e
        theta_e = theta_g * rho_e

        # ---- emission (linear in n) ------------------------------------
        S, Gam = emission_terms(g, phi, theta_g, theta_e, sumZ2N,
                                use_dc=use_dc, use_br=use_br)

        # ---- Compton flux (Chang-Cooper, cancellation-free form) --------
        F, P, hh, w = cc_flux(g, n, phi)

        divF = np.zeros(N)
        divF[:-1] += F
        divF[1:] -= F
        divF /= dxc
        pref = theta_e / g.x2
        comp = pref * divF

        R = (n - n_old) / dtau - comp - (S - Gam * n) - src

        # ---- quasi-Newton Jacobian (P and rho_e frozen) ----------------
        # Exact derivatives of the cc_flux form (see cc_flux docstring):
        #   dF_j/dn_j     = -C_j h_j^2
        #   dF_j/dn_{j+1} = +C_j h_{j+1}^2 e^{w_j},   C_j = P_j phi/expm1(w_j)
        # (identical to  P(-1/dx + phi delta) h^2  and  P(1/dx + phi(1-delta)) h^2 )
        Cj = P * (phi / np.expm1(w))
        dgdn = hh * hh
        dFdnj = -Cj * dgdn[:-1]
        dFdnjp1 = Cj * np.exp(np.minimum(w, 600.0)) * dgdn[1:]

        dd = np.zeros(N)
        dlo = np.zeros(N)
        dup = np.zeros(N)
        dd[:-1] += dFdnj / dxc[:-1]
        dup[:-1] += dFdnjp1 / dxc[:-1]
        dd[1:] += -dFdnjp1 / dxc[1:]
        dlo[1:] += -dFdnj / dxc[1:]

        diag_ = 1.0 / dtau - pref * dd + Gam
        lo = -pref * dlo
        up = -pref * dup

        ab = np.zeros((3, N))
        ab[0, 1:] = up[:-1]
        ab[1, :] = diag_
        ab[2, :-1] = lo[1:]
        dn = solve_banded((1, 1), ab, -R, overwrite_ab=True, check_finite=False)
        n = n + dn
        np.maximum(n, 0.0, out=n)

        # ---- refresh the electron temperature from the new n ------------
        # rho_eq from the flux-consistent moments (see flux_moments docstring);
        # H_dcbr is the CS2012 Eq. 23 DC/BR matter cooling integral, in the same
        # units, which makes the TOTAL photon energy gain per step exactly
        # 4 theta_g G3d dri dtau (Compton + emission channels together).
        G3d, I4d = flux_moments(g, n, phi)
        rho_eq = I4d / (4.0 * G3d)
        H_dcbr = (float(np.sum(g.x3 * (S - Gam * n) * dxc))
                  / (4.0 * theta_g * G3d))
        rho_g = KAPPA_G * theta_g ** 4 * G3d
        beta_C = 4.0 * rho_g / alpha_h
        rho_e_new = ((rho_e_old / dtau + beta_C * (rho_eq + dri - H_dcbr))
                     / (1.0 / dtau + beta_C + HtC))
        drho_e = abs(rho_e_new - rho_e) / rho_e
        rho_e = rho_e_new

        conv = max(float(np.max(np.abs(dn) / (n + 1e-300))), drho_e)
        if conv < tol:
            break
        # Stagnation escape: when the total change over the step is itself tiny
        # the Newton correction bottoms out on linear-solve round-off and a
        # purely relative tolerance is unreachable.  Only allow this once the
        # correction is already absolutely small -- the (n, rho_e) Picard
        # coupling is only weakly contracting (rate ~ theta_g*dtau) and would
        # otherwise be cut off long before the injected energy is delivered.
        if it >= 4 and conv > 0.5 * prev and conv < 1e-9:
            break
        prev = conv
    else:
        capped = True

    if diag is not None:
        diag["conv"] = conv
        diag["capped"] = capped
        diag["newton_iters"] = it
        diag["rho_e"] = rho_e
        diag["rho_eq"] = rho_eq
        diag["beta_C"] = beta_C
        diag["HtC"] = HtC
        diag["H_dcbr"] = H_dcbr
    return n, rho_e


# --------------------------------------------------------------------------
# integrator
# --------------------------------------------------------------------------
def build_schedule(hist, z_start, z_end, dlnz_max=0.01, dy_max=0.05,
                   z_h=None, sigma_z=None, pts_per_sigma=20.0):
    r"""Deterministic redshift schedule.  NOT adaptive: every cap below depends
    only on the frozen background table and the (fixed) injection parameters,
    never on the solution, so the schedule is reproducible and is refined by
    simply scaling dlnz_max / dy_max / pts_per_sigma.

    Caps, per step:
      * dlnz <= dlnz_max                                  (global floor)
      * theta_g * dtau <= dy_max                           (Comptonisation per
        step).  Required: the energy a heat source delivers in one step is
        4 theta_g dri dtau, i.e. a temperature rise theta_g*dri*dtau, while the
        instantaneous electron offset driving it is only dri.  For
        theta_g*dtau >> 1 the (n, rho_e) fixed-point iteration is therefore
        only weakly contracting and the delivered energy is truncated.
      * dlnz <= sigma_z/(z * pts_per_sigma)  within 8 sigma_z of z_h, so the
        injection Gaussian is resolved.
    """
    zs = [float(z_start)]
    z = float(z_start)
    while z > z_end:
        h = hist.at(z)
        dtau_dlnz = z / ((1.0 + z) * h["H"] * h["tC"])
        d = min(dlnz_max, dy_max / (h["theta_g"] * dtau_dlnz))
        if z_h is not None and abs(z - z_h) < 8.0 * sigma_z:
            d = min(d, sigma_z / (z * pts_per_sigma))
        z = z * np.exp(-d)
        if z <= z_end * (1.0 + 1e-12):
            z = float(z_end)
        zs.append(z)
        if len(zs) > 2_000_000:
            raise RuntimeError("schedule did not terminate")
    return np.array(zs)


def integrate(g, hist, z_start, z_end, zgrid, drho=0.0, z_h=None,
              sigma_frac=0.04, photon=None, cooling=True, use_dc=True,
              use_br=True, verbose=False, tol=1e-13):
    """Integrate from z_start down to z_end on the given (precomputed) zgrid.

    drho  : total fractional photon energy release of the heat burst
    z_h   : burst centre (heat) or injection redshift (photon)
    photon: dict(x_inj=..., sigma_x_frac=..., sigma_z_frac=..., dNoverN=...)
    """
    zg = np.asarray(zgrid, dtype=float)
    nz = len(zg) - 1
    zc = np.sqrt(zg[:-1] * zg[1:])                       # geometric midpoints
    hzs = [hist.at(zz) for zz in zc]
    dtaus = np.array([abs(zg[i] - zg[i + 1])
                      / ((1.0 + zc[i]) * hzs[i]["H"] * hzs[i]["tC"])
                      for i in range(nz)])

    # ---- heat-burst normalisation on the *discrete* schedule ----------
    dri = np.zeros(nz)
    if drho != 0.0:
        sig = sigma_frac * z_h
        shape = np.exp(-0.5 * ((zc - z_h) / sig) ** 2)
        th = np.array([h["theta_g"] for h in hzs])
        acc = float(np.sum(4.0 * th * shape * dtaus))
        dri = (drho / acc) * shape

    # ---- photon source: Gaussian in x, Gaussian in z ------------------
    n = g.n_pl.copy()
    ledger = {}
    src_shape = None
    src_z = np.zeros(nz)
    if photon is not None:
        x_inj = photon["x_inj"]
        sx = photon.get("sigma_x_frac", 0.05) * x_inj
        bump = np.exp(-0.5 * ((g.x - x_inj) / sx) ** 2)
        if photon.get("instant", False):
            src_shape = None
            amp = photon["dNoverN"] * N2_PL / g.Nph_cell(bump)
            n = n + amp * bump
        else:
            sigz = photon.get("sigma_z_frac", 0.04) * photon["z_h"]
            sz = np.exp(-0.5 * ((zc - photon["z_h"]) / sigz) ** 2)
            # normalise with the CELL rule (the quadrature the discrete Compton
            # operator conserves exactly), then measure with uniform trapz.
            amp = (photon["dNoverN"] * N2_PL
                   / (g.Nph_cell(bump) * float(np.sum(sz * dtaus))))
            src_shape = bump
            src_z = amp * sz
        ledger["bump_amp"] = float(amp)
        ledger["pts_per_sigma_x"] = float(sx / (x_inj * (g.u[1] - g.u[0])))
        ledger["dN_nominal"] = photon["dNoverN"]

    # ---- rho_e initial value: quasi-stationary ------------------------
    h0 = hist.at(z_start)
    G3d, I4d = flux_moments(g, n, 1.0)
    rho_g = KAPPA_G * h0["theta_g"] ** 4 * G3d
    beta_C = 4.0 * rho_g / h0["alpha_h"]
    rho_e = (I4d / (4.0 * G3d)) / (1.0 + (h0["HtC"] if cooling else 0.0) / beta_C)

    diag = {}
    max_iters = 0
    max_H = 0.0
    n_capped = 0
    worst_conv = 0.0
    rho_e_max = rho_e
    tau = 0.0
    t0 = time.time()
    z_after_window = None
    for i in range(nz):
        src = (src_shape * src_z[i]) if src_shape is not None else 0.0
        n, rho_e = implicit_step(n, g, dtaus[i], hzs[i], rho_e, dri=dri[i],
                                 src=src, cooling=cooling, use_dc=use_dc,
                                 use_br=use_br, tol=tol, diag=diag)
        tau += dtaus[i]
        if photon is not None and z_after_window is None and \
                zg[i + 1] < photon["z_h"] - 7.0 * photon.get("sigma_z_frac", 0.04) * photon["z_h"]:
            z_after_window = float(zg[i + 1])
            ledger["z_after_window"] = z_after_window
            ledger["dN_after_window_trapz"] = float(
                np.trapz(g.x2 * (n - g.n_pl), g.x) / N2_PL)
            ledger["dN_after_window_cell"] = float(
                np.sum(g.x2 * (n - g.n_pl) * g.dxc) / N2_PL)
        max_iters = max(max_iters, diag["newton_iters"])
        max_H = max(max_H, abs(diag["H_dcbr"]))
        n_capped += int(diag["capped"])
        worst_conv = max(worst_conv, diag["conv"])
        rho_e_max = max(rho_e_max, rho_e)
        if verbose and (i % max(1, nz // 12) == 0):
            print(f"   z={zc[i]:.4g} dtau={dtaus[i]:.3g} rho_e-1={rho_e-1:+.4e} "
                  f"it={diag['newton_iters']}", flush=True)
        if not np.all(np.isfinite(n)):
            raise FloatingPointError(f"non-finite spectrum at z={zc[i]:.5g}")

    info = dict(tau_total=float(tau), max_newton_iters=int(max_iters),
                max_abs_H_dcbr=float(max_H), rho_e_final=float(rho_e),
                rho_e_max=float(rho_e_max), dri_max=float(np.max(dri)),
                n_newton_capped=int(n_capped), worst_newton_conv=float(worst_conv),
                max_dy_per_step=float(np.max([hzs[i]["theta_g"] * dtaus[i]
                                              for i in range(nz)])),
                wall_s=round(time.time() - t0, 2), nz=nz, N=g.N)
    if photon is not None:
        ledger["dN_final_trapz"] = float(np.trapz(g.x2 * (n - g.n_pl), g.x) / N2_PL)
        ledger["dN_final_cell"] = float(np.sum(g.x2 * (n - g.n_pl) * g.dxc) / N2_PL)
        ledger["dN_final_simpson"] = float(
            (g.Nph_simpson(n) - g.Nph_simpson(g.n_pl)) / N2_PL)
        info["ledger"] = ledger
    info["drho_final_measured"] = float(g.G3(n) / G3_PL - 1.0)
    G3c = float(np.sum(g.x3 * n * g.dxc))
    info["drho_final_cellrule"] = float(G3c / np.sum(g.x3 * g.n_pl * g.dxc) - 1.0)
    return n, info


# --------------------------------------------------------------------------
# distortion decomposition  (contract section 5, corrected templates)
# --------------------------------------------------------------------------
def templates(x):
    ex = np.exp(x)
    G_bb = x * ex / (ex - 1.0) ** 2
    G = G_bb
    Y = G_bb * (x * (ex + 1.0) / (ex - 1.0) - 4.0)
    M = G_bb * (1.0 / BETA_MU - 1.0 / x)
    return G, Y, M


def decompose(dn, x, xlo=0.5, xhi=18.0, weight=None):
    """Joint linear least squares  dn ~ dT*G + y*Y + mu*M  on x in [xlo, xhi].

    weight=None : uniform weights on the grid nodes in range (contract literal).
    weight='dx' : cell-width weights (sensitivity diagnostic; makes the fit
                  independent of the node density).
    """
    m = (x >= xlo) & (x <= xhi)
    xf, d = x[m], dn[m]
    G, Y, M = templates(xf)
    A = np.vstack([G, Y, M]).T
    if weight == "dx":
        xh = np.sqrt(xf[:-1] * xf[1:])
        e = np.empty(len(xf) + 1)
        e[1:-1] = xh
        e[0] = xf[0] ** 2 / xh[0]
        e[-1] = xf[-1] ** 2 / xh[-1]
        wt = np.sqrt(e[1:] - e[:-1])
        A = A * wt[:, None]
        d = d * wt
    coef, res, rank, sv = np.linalg.lstsq(A, d, rcond=None)
    fit = A @ coef
    rms = float(np.sqrt(np.mean((d - fit) ** 2)))
    peak = float(np.max(np.abs(d))) if len(d) else 0.0
    return dict(dT=float(coef[0]), y=float(coef[1]), mu=float(coef[2]),
                resid_rms=rms, resid_rel=(rms / peak if peak > 0 else 0.0))


# --------------------------------------------------------------------------
# self tests
# --------------------------------------------------------------------------
def self_test_roundtrip(g=None, verbose=True):
    """Pure T-shift / pure y / pure mu synthetic inputs must round-trip."""
    x = (g.x if g is not None else np.logspace(-4, np.log10(40.0), 2049))
    G, Y, M = templates(x)
    out = {}
    for name, shape, amp in (("dT", G, 1e-5), ("y", Y, 1e-5), ("mu", M, 1e-5)):
        r = decompose(amp * shape, x)
        out[name] = r
        if verbose:
            print(f"[roundtrip] pure {name}={amp:.1e}: dT={r['dT']:+.6e} "
                  f"y={r['y']:+.6e} mu={r['mu']:+.6e} resid_rms={r['resid_rms']:.2e}")
    return out


def self_test_moments(g, verbose=True):
    """G3, I4, N of the discrete Planck vs closed forms."""
    n = g.n_pl
    r = dict(G3_rel=g.G3(n) / G3_PL - 1.0,
             I4_rel=g.I4(n) / I4_PL - 1.0,
             ratio_rel=g.I4(n) / (4.0 * g.G3(n)) - 1.0,
             N_simpson_rel=g.Nph_simpson(n) / N2_PL - 1.0,
             N_cell_rel=g.Nph_cell(n) / N2_PL - 1.0)
    if verbose:
        print("[moments] " + "  ".join(f"{k}={v:+.3e}" for k, v in r.items()))
    return r


def self_test_equilibrium(g, rho_e=0.997, verbose=True):
    """Planck at T_e must be an exact fixed point of the full discrete step."""
    phi = 1.0 / rho_e
    n_eq = 1.0 / np.expm1(phi * g.x)
    theta_g = 1e-3
    hz = dict(theta_g=theta_g, alpha_h=1e18, HtC=0.0, sumZ2N=1e19, n_H=1e19,
              x_e=1.1579)
    x, dxn, dxc, N = g.x, g.dxn, g.dxc, g.N
    F, P, hh, w = cc_flux(g, n_eq, phi)
    gg = n_eq / (1.0 + n_eq)
    S, Gam = emission_terms(g, phi, theta_g, theta_g * rho_e, hz["sumZ2N"])
    em = S - Gam * n_eq
    fl = float(np.max(np.abs(F) / (P * phi * gg[:-1] + 1e-300)))
    er = float(np.max(np.abs(em) / (np.abs(S) + 1e-300)))
    G3d, I4d = flux_moments(g, n_eq, phi)
    rho_eq_err = I4d / (4.0 * G3d) / rho_e - 1.0
    # induced spurious relative drift  |dn/dtau|/n  from Compton alone
    divF = np.zeros(N)
    divF[:-1] += F
    divF[1:] -= F
    divF /= dxc
    drift = float(np.max(np.abs(theta_g * rho_e / g.x2 * divF) / n_eq))
    # ... and a full step of the coupled solver from exact equilibrium
    hz2 = dict(hz)
    n2, _ = implicit_step(n_eq.copy(), g, 1e4, hz2, rho_e)
    step_rel = float(np.max(np.abs(n2 - n_eq) / n_eq))
    if verbose:
        print(f"[equilibrium] rho_e={rho_e}: max rel Compton flux={fl:.3e}  "
              f"max rel emission residual={er:.3e}  max |dn/dtau|/n={drift:.3e}  "
              f"rho_eq(flux)/rho_e-1={rho_eq_err:+.3e}  "
              f"drift over dtau=1e4 step={step_rel:.3e}")
    return dict(compton_flux_rel=fl, emission_rel=er, spurious_dndtau_rel=drift,
                rho_eq_flux_rel_err=float(rho_eq_err), step_drift_rel=step_rel)


def self_test_number_conservation(g, verbose=True):
    """Pure Compton (no emission) must conserve the cell-rule photon number."""
    n0 = g.n_pl * (1.0 + 1e-2 * np.exp(-0.5 * ((g.x - 3.0) / 0.5) ** 2))
    hz = dict(theta_g=1e-3, alpha_h=1e18, HtC=0.0, sumZ2N=1e19)
    N0 = g.Nph_cell(n0)
    n = n0.copy()
    rho = 1.0
    for _ in range(20):
        n, rho = implicit_step(n, g, 5.0, hz, rho, use_dc=False, use_br=False)
    rel = g.Nph_cell(n) / N0 - 1.0
    if verbose:
        print(f"[number conservation] pure Compton, 20 x dtau=5: dN/N={rel:+.3e}")
    return float(rel)


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------
def case_defs(drho=1e-3, z_end=200.0):
    return {
        "heat_z2e6": dict(kind="heat", z_h=2e6, drho=drho),
        "heat_z2e5": dict(kind="heat", z_h=2e5, drho=drho),
        "heat_z5e3": dict(kind="heat", z_h=5e3, drho=drho),
        "adiabatic": dict(kind="adiabatic"),
        "photon_x0.1_z3e5": dict(kind="photon", z_h=3e5, x_inj=0.1,
                                 sigma_x_frac=0.05, sigma_z_frac=0.04,
                                 dNoverN=1e-3),
    }


def schedule(cd, z_end):
    if cd["kind"] == "adiabatic":
        return 3e6, z_end
    # heat burst and photon injection both use z_start = z_h + 7 sigma_z
    return cd["z_h"] * (1.0 + 7.0 * 0.04), z_end


def run_case(cid, cd, hist, N=2049, refine=1.0, z_end=200.0, use_br=True,
             verbose=False, dlnz_max=0.01, dy_max=0.05, pts_per_sigma=20.0):
    """Run one case plus its numerical-drift control on the identical schedule.

    control:  same grid, same steps, no injection.  For the heat/photon cases
    the control keeps the Hubble cooling term on, so subtracting it removes the
    (common-mode) quadrature drift.  For the adiabatic case the control has the
    cooling term OFF, in which case the exact solution is a pure Planck for all
    time, so the control spectrum IS the numerical drift and the difference is
    the physical cooling signal.
    """
    g = Grid(N=N)
    z_start, z_e = schedule(cd, z_end)
    z_h = cd.get("z_h")
    sig_z = 0.04 * z_h if z_h is not None else None
    zgrid = build_schedule(hist, z_start, z_e, dlnz_max=dlnz_max / refine,
                           dy_max=dy_max / refine, z_h=z_h, sigma_z=sig_z,
                           pts_per_sigma=pts_per_sigma * refine)
    kw = dict(cooling=True, use_dc=True, use_br=use_br, verbose=verbose)
    if cd["kind"] == "heat":
        n, info = integrate(g, hist, z_start, z_e, zgrid, drho=cd["drho"],
                            z_h=cd["z_h"], **kw)
        nc, ic = integrate(g, hist, z_start, z_e, zgrid, **kw)
    elif cd["kind"] == "photon":
        n, info = integrate(g, hist, z_start, z_e, zgrid,
                            photon=dict(x_inj=cd["x_inj"], z_h=cd["z_h"],
                                        sigma_x_frac=cd["sigma_x_frac"],
                                        sigma_z_frac=cd["sigma_z_frac"],
                                        dNoverN=cd["dNoverN"]), **kw)
        nc, ic = integrate(g, hist, z_start, z_e, zgrid, **kw)
    else:
        n, info = integrate(g, hist, z_start, z_e, zgrid, **kw)
        kw0 = dict(kw)
        kw0["cooling"] = False
        nc, ic = integrate(g, hist, z_start, z_e, zgrid, **kw0)

    dn_raw = n - g.n_pl
    dn_ctl = nc - g.n_pl
    dn = dn_raw - dn_ctl
    res = dict(case=cid, N=g.N, nz=info["nz"], refine=refine,
               z_start=z_start, z_end=z_e,
               max_dy_per_step=info["max_dy_per_step"],
               tau_total=info["tau_total"],
               max_newton_iters=info["max_newton_iters"],
               max_abs_H_dcbr=info["max_abs_H_dcbr"],
               rho_e_final=info["rho_e_final"], rho_e_max=info["rho_e_max"],
               dri_max=info["dri_max"],
               n_newton_capped=info["n_newton_capped"],
               worst_newton_conv=info["worst_newton_conv"],
               wall_s=info["wall_s"] + ic["wall_s"],
               drho_final_measured=info["drho_final_measured"],
               drho_final_cellrule=info["drho_final_cellrule"],
               drho_control_measured=ic["drho_final_measured"],
               xmin=g.xmin, xmax=g.xmax)
    res["fit"] = decompose(dn, g.x)
    res["fit_raw_nocontrol"] = decompose(dn_raw, g.x)
    res["fit_control_only"] = decompose(dn_ctl, g.x)
    res["fit_dxweight"] = decompose(dn, g.x, weight="dx")
    if "ledger" in info:
        led = dict(info["ledger"])
        led["dN_control_final_trapz"] = float(np.trapz(g.x2 * dn_ctl, g.x) / N2_PL)
        led["dN_final_net_trapz"] = led["dN_final_trapz"] - led["dN_control_final_trapz"]
        led["surviving_fraction_vs_after_window"] = (
            led["dN_final_net_trapz"] / led["dN_after_window_trapz"])
        led["surviving_fraction_vs_nominal"] = (
            led["dN_final_net_trapz"] / led["dN_nominal"])
        res["ledger"] = led
        f = res["fit"]
        res["normalised_by_measured_dN"] = dict(
            mu_over_dN_after_window=f["mu"] / led["dN_after_window_trapz"],
            y_over_dN_after_window=f["y"] / led["dN_after_window_trapz"],
            dT_over_dN_after_window=f["dT"] / led["dN_after_window_trapz"],
            mu_over_dN_nominal=f["mu"] / led["dN_nominal"])
    return res, g.x, dn, dn_raw


# --------------------------------------------------------------------------
def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=2049)
    p.add_argument("--refine", type=float, default=1.0,
                   help="scale factor on the step schedule (2 = twice as fine)")
    p.add_argument("--z-end", type=float, default=200.0)
    p.add_argument("--drho", type=float, default=1e-3)
    p.add_argument("--cases", default="all")
    p.add_argument("--no-br", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--outdir", default=os.path.join(HERE, "outputs"))
    p.add_argument("--write-spectra", action="store_true")
    p.add_argument("--selftest-only", action="store_true")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)

    os.makedirs(a.outdir, exist_ok=True)
    g = Grid(N=a.N)
    st = dict(roundtrip=self_test_roundtrip(g),
              moments=self_test_moments(g),
              equilibrium=self_test_equilibrium(g),
              number_conservation=self_test_number_conservation(g))
    if a.selftest_only:
        print(json.dumps(st, indent=2, default=float))
        return 0

    hist = History()
    defs = case_defs(drho=a.drho, z_end=a.z_end)
    want = list(defs) if a.cases == "all" else a.cases.split(",")
    out = dict(config=dict(N=g.N, refine=a.refine, z_end=a.z_end, drho=a.drho,
                           xmin=g.xmin, xmax=g.xmax, use_br=not a.no_br,
                           tag=a.tag),
               self_tests=st, cases={})
    for cid in want:
        print(f"=== {cid}  N={g.N} refine={a.refine} z_end={a.z_end} "
              f"drho={a.drho} BR={'off' if a.no_br else 'on'}", flush=True)
        r, x, dn, dn_raw = run_case(cid, defs[cid], hist, N=a.N,
                                    refine=a.refine, z_end=a.z_end,
                                    use_br=not a.no_br, verbose=a.verbose)
        out["cases"][cid] = r
        f = r["fit"]
        print(f"    mu={f['mu']:+.6e}  y={f['y']:+.6e}  dT/T={f['dT']:+.6e}  "
              f"resid_rel={f['resid_rel']:.2e}  nz={r['nz']} "
              f"drho_meas={r['drho_final_measured']:.5e} ({r['wall_s']}s)",
              flush=True)
        if a.write_spectra:
            fn = os.path.join(a.outdir, f"spectrum_{cid}{a.tag}.csv")
            np.savetxt(fn, np.column_stack([x, dn]), delimiter=",",
                       header="x,delta_n", comments="")
    fn = os.path.join(a.outdir, f"results{a.tag}.json")
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("wrote", fn)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
