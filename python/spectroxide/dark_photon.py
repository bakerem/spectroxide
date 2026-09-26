"""Helpers for dark-photon (γ ↔ A') conversion in the narrow-width approximation.

The partial differential equation solver handles dark-photon oscillations
through the initial-condition path: pass
``injection={"type": "dark_photon_resonance", "epsilon": ε, "m_ev": m}`` to
:func:`spectroxide.solve` and the Rust solver applies
``Δn(x) = −[1 − exp(−γ_con/x)] × n_pl(x)`` at ``z_start = z_res``, with the
photon mass equal to the plasma frequency, as in Chluba, Cyr & Johnson (2024),
and evolves forward in time.

With ``"neutral_hydrogen": True`` in the injection dict, the solver instead
applies ``Δn(x) = −⟨1 − exp(−τ(x))⟩ × n_pl(x)`` at the same ``z_res`` (the
latest redshift at which any frequency converts), where ``⟨·⟩`` is a grid-cell
average (:func:`cell_average`). :func:`conversion_probability` computes the
point values of ``τ(x)`` without running the solver. This option adds neutral
hydrogen to the photon effective mass (Caputo, Liu, Mishra-Sharma & Ruderman
2020, PRD 102, 103533, Eq. 1):

.. math::

    m_\\gamma^2(z, \\omega) = \\omega_{pl}^2(z) - 4\\pi \\alpha_H n_{HI}(z)\\, \\omega^2,
    \\qquad \\alpha_H = 4.5\\, a_0^3,

so 4π α_H = 8.38×10⁻²⁴ cm³ (helium neglected, as in Caputo et al.). Every
crossing ``m_γ²(z, x) = m²`` at fixed ``x = ω/(k T_γ)`` converts with the
Landau–Zener probability ``P_i = π ε² m² / (ω_i H_i |d ln m_γ²/d ln a|_i)``
(Mirizzi, Redondo & Sigl 2009, Eq. 18), and ``τ(x) = Σ_i P_i``. Without
neutral hydrogen, ``τ(x) = γ_con/x`` with the plasma-only :func:`gamma_con`
(Chluba, Cyr & Johnson 2024, Eq. 6), which stays the nominal value used for
reporting. After recombination the neutral term can exceed the plasma term
(1.9× at x = 4 for m = 10⁻¹¹ eV at its plasma-only z_res ≈ 668), which moves
the crossings of high-x photons (ADR 0008).

References
----------
- Mirizzi, Redondo & Sigl (2009), JCAP 0903, 026.
- Caputo, Liu, Mishra-Sharma & Ruderman (2020), PRD 102, 103533 [arXiv:2004.06733].
- Chluba, Cyr & Johnson (2024), MNRAS 535, 1874 [arXiv:2409.12115].
"""

from __future__ import annotations

from typing import List, Mapping, NamedTuple, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from .cosmology import (
    DEFAULT_COSMO,
    _C_LIGHT,
    _E_H_ION,
    _E_HE_I_ION,
    _E_HE_II_ION,
    _HBAR,
    _K_BOLTZMANN,
    _cosmo_hubble,
    _cosmo_n_e,
    _cosmo_n_h,
    _get_recomb_table,
    _thermal_de_broglie,
    ionization_fraction,
)

#: Type alias for cosmology mappings accepted by these helpers.
CosmoLike = Mapping[str, float]

_ALPHA_FS = 7.297_352_5693e-3
_M_ELECTRON = 9.109_383_7015e-31  # kg
_EV_IN_JOULES = 1.602_176_634e-19
_HBAR_EV_S = _HBAR / _EV_IN_JOULES

#: Bohr radius a₀ in m (CODATA 2018).
BOHR_RADIUS = 5.291_772_109_03e-11
#: Static polarizability of ground-state hydrogen in units of a₀³.
HYDROGEN_POLARIZABILITY_A0_CUBED = 4.5

_NEUTRAL_COEFF_M3 = 4.0 * np.pi * HYDROGEN_POLARIZABILITY_A0_CUBED * BOHR_RADIUS**3
_Z_SCAN_MIN = 10.0
_Z_SCAN_MAX = 3.0e7
_N_Z_SCAN = 3000  # log-spaced; Δln z ≈ 0.005, as in the Rust scan


def plasma_frequency_ev(z: float, cosmo: CosmoLike | None = None) -> float:
    """Photon plasma frequency ``ω_pl`` at redshift ``z``.

    .. math::

        \\omega_{pl}^2 = \\frac{4 \\pi \\alpha\\, n_e\\, \\hbar c}{m_e},
        \\qquad n_e = X_e(z) \\, n_H(z).

    Parameters
    ----------
    z : float
        Redshift.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to
        :data:`spectroxide.DEFAULT_COSMO`.

    Returns
    -------
    float
        Plasma frequency in **eV**.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    n_e = _cosmo_n_e(z, cosmo)
    factor = 4.0 * np.pi * _ALPHA_FS * _HBAR * _C_LIGHT / _M_ELECTRON
    return _HBAR_EV_S * np.sqrt(n_e * factor)


def resonance_redshift(
    m_ev: float,
    cosmo: CosmoLike | None = None,
    z_min: float = 10.0,
    z_max: float = 3.0e7,
) -> float | None:
    """Solve ``ω_pl(z_res) = m`` for ``z_res`` by Brent bisection.

    Parameters
    ----------
    m_ev : float
        Dark-photon mass in **eV**.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to
        :data:`spectroxide.DEFAULT_COSMO`.
    z_min : float, optional
        Lower edge of the search bracket (default ``10.0``).
    z_max : float, optional
        Upper edge of the search bracket (default ``3.0e7``).

    Returns
    -------
    float or None
        Resonance redshift, or *None* if no sign change of
        ``ω_pl(z) − m`` exists in ``[z_min, z_max]``.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO

    def f(z):
        return plasma_frequency_ev(z, cosmo) - m_ev

    if f(z_min) * f(z_max) > 0:
        return None
    return brentq(f, z_min, z_max)


def dln_omega_pl_sq_dlna(z: float, cosmo: CosmoLike | None = None) -> float:
    """Compute ``|d ln ω_pl² / d ln a|`` at redshift ``z``.

    Uses a centered finite difference on ``X_e(z)`` with relative step
    ``max(0.1, 1e-4 z)``.  Falls back to the matter-era value 3 when
    ``X_e`` is too small to differentiate reliably.

    Parameters
    ----------
    z : float
        Redshift.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to
        :data:`spectroxide.DEFAULT_COSMO`.

    Returns
    -------
    float
        Magnitude of the logarithmic derivative (dimensionless).
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    dz = max(0.1, z * 1e-4)
    x_e = ionization_fraction(z, cosmo)
    if x_e <= 1e-30:
        return 3.0
    x_e_p = ionization_fraction(z + dz, cosmo)
    x_e_m = ionization_fraction(z - dz, cosmo)
    dlnxe_dz = (x_e_p - x_e_m) / (2.0 * dz * x_e)
    return abs((1.0 + z) * dlnxe_dz + 3.0)


def gamma_con(
    epsilon: float, m_ev: float, cosmo: CosmoLike | None = None
) -> Tuple[float | None, float | None]:
    """Narrow-width approximation conversion parameter ``γ_con``.

    .. math::

        \\gamma_{con} = \\frac{\\pi\\, \\epsilon^2 m^2}
                            {\\bigl|d\\ln \\omega_{pl}^2 / d\\ln a\\bigr|_{z_{res}}
                             T_\\gamma(z_{res})\\, H(z_{res})},

    following Chluba, Cyr & Johnson (2024), MNRAS 535, 1874, Eq. 6.

    This is the plasma-only value (``m_γ = ω_pl``), kept as the nominal
    conversion parameter, and the solver depletion uses it by default. With
    the ``neutral_hydrogen`` option the solver depletion uses the full photon
    mass (:func:`conversion_probability`), which reduces to ``τ(x) = γ_con/x``
    wherever neutral hydrogen is negligible at the crossing.

    Parameters
    ----------
    epsilon : float
        Kinetic-mixing parameter ``ε`` (dimensionless).
    m_ev : float
        Dark-photon mass in **eV**.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to
        :data:`spectroxide.DEFAULT_COSMO`.

    Returns
    -------
    tuple of (float, float) or (None, None)
        ``(γ_con, z_res)`` if a resonance exists in the search bracket,
        otherwise ``(None, None)``.  ``γ_con`` is dimensionless;
        ``z_res`` is the resonance redshift.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    z_res = resonance_redshift(m_ev, cosmo)
    if z_res is None:
        return None, None
    t_cmb_ev = _K_BOLTZMANN * cosmo["t_cmb"] * (1.0 + z_res) / _EV_IN_JOULES
    h_ev = _HBAR_EV_S * _cosmo_hubble(z_res, cosmo)
    d = dln_omega_pl_sq_dlna(z_res, cosmo)
    gc = np.pi * epsilon**2 * m_ev**2 / (d * t_cmb_ev * h_ev)
    return gc, z_res


def gc_per_epsilon_sq(
    m_ev: float, cosmo: CosmoLike | None = None
) -> Tuple[float | None, float | None]:
    """Return ``(γ_con/ε², z_res)`` at the resonance for mass ``m``.

    Convenience scaling factor: re-fit constraints on ``ε`` without
    recomputing ``z_res`` for each candidate.

    Parameters
    ----------
    m_ev : float
        Dark-photon mass in **eV**.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to
        :data:`spectroxide.DEFAULT_COSMO`.

    Returns
    -------
    tuple of (float, float) or (None, None)
        ``(γ_con / ε² , z_res)``.  Returns ``(None, None)`` when no
        resonance exists.
    """
    return gamma_con(1.0, m_ev, cosmo)


# ---------------------------------------------------------------------------
# Full photon mass with neutral hydrogen (ADR 0008)
# ---------------------------------------------------------------------------


def _ionization_state(z, cosmo):
    """Vectorized ``(X_e, x_He)`` at redshifts ``z``.

    Same history as :func:`spectroxide.cosmology.ionization_fraction` (Saha
    helium, Saha hydrogen above the switch, cached Peebles table below), but
    evaluated on arrays without a Python loop. ``x_He`` is the helium
    contribution to ``X_e``, so ``X_H = X_e − x_He``.
    """
    z_in = np.asarray(z, dtype=np.float64)
    z = np.atleast_1d(z_in)
    t = cosmo["t_cmb"] * (1.0 + z)
    kt = _K_BOLTZMANN * t
    n_h = _cosmo_n_h(z, cosmo)
    f_he = cosmo["y_p"] / (4.0 * (1.0 - cosmo["y_p"]))
    n_he = f_he * n_h
    tdb = _thermal_de_broglie(t)
    s2 = tdb * np.exp(-_E_HE_II_ION / kt) / (n_h + 2.0 * n_he)
    s1 = 4.0 * tdb * np.exp(-_E_HE_I_ION / kt) / (n_h + n_he)
    x_he = f_he * (s1 / (1.0 + s1) + s2 / (1.0 + s2))

    z_ode, x_h_ode, z_switch = _get_recomb_table(cosmo)
    x_h = np.interp(z, z_ode, x_h_ode)  # clamps to x_h_ode[0] below the table
    saha = (z > z_switch) & (z <= 8000.0)
    if np.any(saha):
        s_h = tdb[saha] * np.exp(-_E_H_ION / kt[saha]) / n_h[saha]
        # Stable root of X²/(1−X) = s.
        x_h[saha] = np.minimum(2.0 * s_h / (s_h + np.sqrt(s_h * s_h + 4.0 * s_h)), 1.0)
    x_h[z > 8000.0] = 1.0
    x_e = x_h + x_he
    if z_in.ndim == 0:
        return float(x_e[0]), float(x_he[0])
    return x_e, x_he


def _background(z, cosmo):
    """``(ω_pl², N)`` in eV², with the neutral term ``N x²``; ``N`` = 4π α_H n_HI (k T_γ)²."""
    x_e, x_he = _ionization_state(z, cosmo)
    n_h = _cosmo_n_h(z, cosmo)
    omega_pl_sq = (
        _HBAR_EV_S**2
        * x_e
        * n_h
        * (4.0 * np.pi * _ALPHA_FS * _HBAR * _C_LIGHT / _M_ELECTRON)
    )
    f_n = 1.0 - np.clip(x_e - x_he, 0.0, 1.0)
    t_ev = _K_BOLTZMANN * cosmo["t_cmb"] * (1.0 + np.asarray(z)) / _EV_IN_JOULES
    return omega_pl_sq, _NEUTRAL_COEFF_M3 * f_n * n_h * t_ev**2, x_e, f_n


def photon_mass_sq_ev2(
    z: ArrayLike, x: ArrayLike, cosmo: CosmoLike | None = None
) -> np.ndarray | float:
    """Photon effective mass squared ``m_γ²(z, x)`` in eV².

    .. math::

        m_\\gamma^2 = \\omega_{pl}^2 - 4\\pi \\alpha_H n_{HI} \\omega^2,
        \\qquad \\omega = x\\, k T_\\gamma(z),\\quad n_{HI} = (1 - X_H)\\, n_H,

    following Caputo et al. (2020), PRD 102, 103533, Eq. 1. Negative where
    neutral hydrogen dominates. Mirrors ``photon_mass_sq_ev2`` in
    ``src/dark_photon.rs``.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    x : float or array_like
        Dimensionless frequency ``hν/(k T_γ)``; broadcasts against ``z``.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to
        :data:`spectroxide.DEFAULT_COSMO`.

    Returns
    -------
    float or ndarray
        ``m_γ²`` in **eV²**.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    omega_pl_sq, neutral, _, _ = _background(z_arr, cosmo)
    out = omega_pl_sq - neutral * np.asarray(x, dtype=np.float64) ** 2
    if np.ndim(z) == 0 and np.ndim(x) == 0:
        return float(out.ravel()[0])
    return out


class DarkPhotonConversion(NamedTuple):
    """Result of :func:`conversion_probability`."""

    #: Conversion depth ``τ(x) = Σ_i P_i(x)`` (dimensionless).
    tau: np.ndarray
    #: Total conversion probability ``1 − exp(−τ)``; ``Δn = −probability × n_pl``.
    probability: np.ndarray
    #: Crossing redshifts for each ``x``, in decreasing order (empty if none).
    crossings: List[np.ndarray]


def _dln_mass_sq_dlna(z, x, m_sq, cosmo):
    """``|d ln m_γ²/d ln a|`` at crossings ``z`` (fixed ``x``); mirrors the Rust slope."""
    dz = np.maximum(0.1, 1.0e-4 * z)
    omega_pl_sq, neutral_c, x_e, f_n = _background(z, cosmo)
    _, _, x_e_p, f_n_p = _background(z + dz, cosmo)
    _, _, x_e_m, f_n_m = _background(z - dz, cosmo)
    opz = 1.0 + z
    plasma = omega_pl_sq * (3.0 + opz * (x_e_p - x_e_m) / (2.0 * dz) / x_e)
    t_ev = _K_BOLTZMANN * cosmo["t_cmb"] * opz / _EV_IN_JOULES
    neutral_per_fn = _NEUTRAL_COEFF_M3 * _cosmo_n_h(z, cosmo) * t_ev**2 * x**2
    neutral = neutral_c * x**2
    neutral_slope = 5.0 * neutral + neutral_per_fn * opz * (f_n_p - f_n_m) / (2.0 * dz)
    return np.abs((plasma - neutral_slope) / m_sq)


def _find_crossings(m_sq, x, cosmo, z_scan, omega_pl_sq, neutral):
    """All crossings of ``m_γ² = m²`` for each ``x``: returns ``(ix, z)`` arrays.

    Sign changes on the scan give most crossings; a scan extremum whose
    neighbors share its sign and that points towards zero is refined by
    golden-section search, and if it crosses zero both roots are kept (pairs
    closer than one scan step). Mirrors ``Scanner::crossings`` in Rust.
    """
    lo_list = [np.empty(0)]
    hi_list = [np.empty(0)]
    ix_list = [np.empty(0, dtype=np.intp)]
    ext_ix, ext_iz = [np.empty(0, dtype=np.intp)], [np.empty(0, dtype=np.intp)]
    chunk = 256  # bounds the (chunk × 3000) work array
    for start in range(0, x.size, chunk):
        xs = x[start : start + chunk]
        f = omega_pl_sq[None, :] - neutral[None, :] * xs[:, None] ** 2 - m_sq
        below = f < 0.0
        ix, iz = np.nonzero(below[:, :-1] != below[:, 1:])
        ix_list.append(ix + start)
        lo_list.append(z_scan[iz])
        hi_list.append(z_scan[iz + 1])
        a, b, c = f[:, :-2], f[:, 1:-1], f[:, 2:]
        same = ((a < 0) == (b < 0)) & ((b < 0) == (c < 0))
        toward = np.where(b > 0, (b <= a) & (b <= c), (b >= a) & (b >= c))
        jx, jz = np.nonzero(same & toward)
        ext_ix.append(jx + start)
        ext_iz.append(jz + 1)
    ext_ix = np.concatenate(ext_ix)
    ext_iz = np.concatenate(ext_iz)

    if ext_ix.size:
        xe = x[ext_ix]
        z0, z1 = z_scan[ext_iz - 1], z_scan[ext_iz + 1]
        w2, nc, _, _ = _background(z_scan[ext_iz], cosmo)
        sgn = np.where(w2 - nc * xe**2 - m_sq > 0, 1.0, -1.0)

        def g(zz):
            w2, nc, _, _ = _background(zz, cosmo)
            return sgn * (w2 - nc * xe**2 - m_sq)

        r = 0.5 * (np.sqrt(5.0) - 1.0)
        lo, hi = z0.copy(), z1.copy()
        c1, c2 = hi - r * (hi - lo), lo + r * (hi - lo)
        g1, g2 = g(c1), g(c2)
        for _ in range(100):
            left = g1 < g2
            hi = np.where(left, c2, hi)
            lo = np.where(left, lo, c1)
            new_c1 = np.where(left, hi - r * (hi - lo), c2)
            new_c2 = np.where(left, c1, lo + r * (hi - lo))
            new_g1 = np.where(left, np.nan, g2)
            new_g2 = np.where(left, g1, np.nan)
            c1, c2 = new_c1, new_c2
            ev = g(np.where(left, c1, c2))
            g1 = np.where(left, ev, new_g1)
            g2 = np.where(left, new_g2, ev)
            if np.all(hi - lo <= 1e-12 * hi):
                break
        z_e = np.where(g1 < g2, c1, c2)
        g_e = np.minimum(g1, g2)
        hit = g_e < 0.0
        for a_, b_ in ((z0, z_e), (z_e, z1)):
            ix_list.append(ext_ix[hit])
            lo_list.append(a_[hit])
            hi_list.append(b_[hit])

    ix = np.concatenate(ix_list)
    lo = np.concatenate(lo_list)
    hi = np.concatenate(hi_list)
    xc = x[ix]

    def f(z):
        w2, nc, _, _ = _background(z, cosmo)
        return w2 - nc * xc**2 - m_sq

    if ix.size:
        f_lo = f(lo)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            same = (f_lo < 0.0) == (f_mid < 0.0)
            lo = np.where(same, mid, lo)
            f_lo = np.where(same, f_mid, f_lo)
            hi = np.where(same, hi, mid)
    return ix, 0.5 * (lo + hi)


class _Scanner:
    """Crossing finder for one ``(ε, m)``; the redshift scan is built once."""

    def __init__(self, epsilon, m_ev, cosmo):
        self.cosmo = cosmo
        self.m_sq = m_ev * m_ev
        self.eps_sq = epsilon * epsilon
        self.z_scan = np.geomspace(_Z_SCAN_MIN, _Z_SCAN_MAX, _N_Z_SCAN)
        self.omega_pl_sq, self.neutral, _, _ = _background(self.z_scan, cosmo)

    def crossings(self, x):
        return _find_crossings(
            self.m_sq, x, self.cosmo, self.z_scan, self.omega_pl_sq, self.neutral
        )

    def bisect_count(self, lo, hi, c_lo):
        """Bisect ``x`` in each bracket to where the crossing count leaves ``c_lo``."""
        lo, hi = np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64)
        if lo.size == 0:
            return lo
        for _ in range(100):
            if np.all(hi - lo <= 1e-13 * hi):
                break
            mid = 0.5 * (lo + hi)
            keep = self.tau_count(mid)[1] == c_lo
            lo = np.where(keep, mid, lo)
            hi = np.where(keep, hi, mid)
        return 0.5 * (lo + hi)

    def tangency_estimates(self):
        """Grid-independent estimates of tangency frequencies ``x_t``.

        A tangency has ``f = ∂f/∂z = 0`` with ``f = P − N x² − m²``, so
        ``x² = P'/N'`` and ``h = P − N P'/N' − m² = 0``; roots of ``h`` on the
        scan give ``x_t``. Mirrors ``Scanner::tangency_estimates`` in Rust.
        """
        P, N = self.omega_pl_sq, self.neutral
        dp, dn = P[2:] - P[:-2], N[2:] - N[:-2]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(dn != 0.0, dp / dn, np.nan)
        valid = np.isfinite(r) & (r > 0.0)
        h = np.where(valid, P[1:-1] - N[1:-1] * r - self.m_sq, np.nan)
        k = np.nonzero(valid[:-1] & valid[1:] & ((h[:-1] < 0) != (h[1:] < 0)))[0]
        t = h[k] / (h[k] - h[k + 1])
        return np.sort(np.sqrt(r[k] + t * (r[k + 1] - r[k])))

    def tau_count(self, x):
        """``(τ, number of crossings)`` at each ``x``."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        ix, zc = self.crossings(x)
        if zc.size:
            xc = x[ix]
            omega = xc * _K_BOLTZMANN * self.cosmo["t_cmb"] * (1.0 + zc) / _EV_IN_JOULES
            h_ev = _HBAR_EV_S * _cosmo_hubble(zc, self.cosmo)
            d = _dln_mass_sq_dlna(zc, xc, self.m_sq, self.cosmo)
            p_i = np.pi * self.eps_sq * self.m_sq / (omega * h_ev * d)
        else:
            p_i = zc
        tau = np.bincount(ix, weights=p_i, minlength=x.size).astype(np.float64)
        count = np.bincount(ix, minlength=x.size)
        return tau, count, ix, zc


def conversion_probability(
    epsilon: float,
    m_ev: float,
    x: ArrayLike,
    cosmo: CosmoLike | None = None,
) -> DarkPhotonConversion:
    """Point values of the dark-photon conversion with the full photon mass.

    For each ``x``, finds every crossing ``z_i`` of ``m_γ²(z, x) = m²`` in
    ``z ∈ [10, 3×10⁷]`` and sums the Landau–Zener probabilities

    .. math::

        P_i = \\frac{\\pi \\epsilon^2 m^2}
                   {\\omega_i H_i \\left|d\\ln m_\\gamma^2/d\\ln a\\right|_i},
        \\qquad \\omega_i = x\\, k T_\\gamma(z_i),

    with the derivative at fixed ``x`` (Mirizzi et al. 2009, Eq. 18; Caputo
    et al. 2020, PRL 125, 221303, Eq. 3). Crossings come from 3000
    log-spaced redshifts refined by bisection, plus a golden-section check
    of every scan extremum for pairs closer than one step.

    Near a tangency ``x_t``, where two crossings merge, the narrow-width
    ``P_i`` diverges as ``(x_t − x)^(−1/2)``, so point values on a grid depend
    on where the points fall; use :func:`cell_average` for grids. Mirrors
    ``conversion_probability`` in ``src/dark_photon.rs``.

    Parameters
    ----------
    epsilon : float
        Kinetic-mixing parameter ``ε``. ``τ ∝ ε²``, so ``epsilon=1`` returns
        ``τ/ε²``.
    m_ev : float
        Dark-photon mass in **eV**.
    x : float or array_like
        Dimensionless frequencies ``hν/(k T_γ)``.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to
        :data:`spectroxide.DEFAULT_COSMO`.

    Returns
    -------
    DarkPhotonConversion
        ``(tau, probability, crossings)``.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    tau, _, ix, zc = _Scanner(epsilon, m_ev, cosmo).tau_count(x)
    crossings = [np.sort(zc[ix == i])[::-1] for i in range(x.size)]
    return DarkPhotonConversion(tau, -np.expm1(-tau), crossings)


def cell_average(
    epsilon: float,
    m_ev: float,
    x: ArrayLike,
    cosmo: CosmoLike | None = None,
    n_sub_cell: int = 4,
    n_sub_tangent: int = 32,
    tangent_reach: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cell averages of ``τ(x)`` and ``1 − exp(−τ(x))`` on an ascending grid.

    Cell ``i`` spans the midpoints to its neighbors (half cells at the two
    ends); the grid may be non-uniform. Away from tangencies the average is
    the mean of ``n_sub_cell`` midpoint sub-samples. A tangency ``x_t`` (where
    the number of crossings changes) is located by bisection between
    sub-samples, and every cell within ``tangent_reach`` cell widths of one is
    split at ``x_t`` and integrated with ``n_sub_tangent`` midpoints in
    ``u = |x − x_t|^(1/2)``, which removes the ``(x_t − x)^(−1/2)`` divergence
    of the narrow-width approximation. The result is independent of where
    grid points fall (ADR 0008) but remains the narrow-width value, which
    overstates the true, finite conversion at a tangency. Tangencies also
    come from the grid-independent tangency curve (``x² = P'/N'`` where
    ``f = ∂f/∂z = 0``), so a band of extra crossings narrower than a
    sub-sample is still found unless its ends lie closer than the curve's
    finite-difference accuracy. A piece between two tangencies is split at
    its midpoint and each half integrated toward its own end. The two end
    half cells are not centered on their nodes, so a sloped ``τ`` is biased
    there by about ``Δx/(4x)``; keep the grid ends outside the band of
    interest. Mirrors
    ``cell_averaged_probability`` in ``src/dark_photon.rs``.

    Parameters
    ----------
    epsilon : float
        Kinetic-mixing parameter ``ε``.
    m_ev : float
        Dark-photon mass in **eV**.
    x : array_like
        Ascending dimensionless frequencies.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to
        :data:`spectroxide.DEFAULT_COSMO`.
    n_sub_cell, n_sub_tangent : int, optional
        Sub-samples per ordinary cell and per piece near a tangency.
    tangent_reach : float, optional
        Cells within this many of their widths of a tangency use the
        ``u`` integration.

    Returns
    -------
    tuple of ndarray
        ``(tau_avg, probability_avg)``; the depletion is
        ``−probability_avg × n_pl(x)``.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    if n_sub_cell < 1 or n_sub_tangent < 1:
        raise ValueError("cell_average: n_sub_cell and n_sub_tangent must be >= 1")
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    sc = _Scanner(epsilon, m_ev, cosmo)
    if x.size < 2:
        tau = sc.tau_count(x)[0]
        return tau, -np.expm1(-tau)
    if np.any(np.diff(x) < 0):
        raise ValueError("cell_average: x must be ascending")

    edges = np.concatenate([[x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]]])
    width = np.diff(edges)
    k = (np.arange(n_sub_cell) + 0.5) / n_sub_cell
    xs = (edges[:-1, None] + width[:, None] * k[None, :]).ravel()
    tau_s, cnt_s, _, _ = sc.tau_count(xs)

    # Tangencies from count changes between sub-samples, plus the
    # grid-independent tangency curve, which catches bands narrower than a
    # sub-sample.
    j = np.nonzero((cnt_s[:-1] != cnt_s[1:]) & (xs[1:] > xs[:-1]))[0]
    tangencies = list(sc.bisect_count(xs[j], xs[j + 1], cnt_s[j]))
    est = sc.tangency_estimates()
    est = est[(est > x[0]) & (est < x[-1])]
    if est.size:
        half = 1e-3 * est
        gaps = np.diff(est)
        half[1:] = np.minimum(half[1:], 0.4 * gaps)
        half[:-1] = np.minimum(half[:-1], 0.4 * gaps)
        lo, hi = est - half, est + half
        c_lo, c_hi = sc.tau_count(lo)[1], sc.tau_count(hi)[1]
        ok = c_lo != c_hi
        for x_t in sc.bisect_count(lo[ok], hi[ok], c_lo[ok]):
            if all(abs(t - x_t) > 1e-9 * x_t for t in tangencies):
                tangencies.append(x_t)
    tangencies = np.sort(np.array(tangencies, dtype=np.float64))

    tau_avg = tau_s.reshape(x.size, n_sub_cell).mean(axis=1)
    p_avg = (-np.expm1(-tau_s)).reshape(x.size, n_sub_cell).mean(axis=1)
    if tangencies.size == 0:
        return tau_avg, p_avg

    a, b = edges[:-1], edges[1:]
    dist = np.maximum(np.maximum(a[:, None] - tangencies, tangencies - b[:, None]), 0.0)
    near = np.nonzero(
        (width > 0) & np.any(dist <= tangent_reach * width[:, None], axis=1)
    )[0]
    kk = (np.arange(n_sub_tangent) + 0.5) / n_sub_tangent
    pts, wts, owner = [], [], []

    def segment(i, s0, s1, x_o):
        if x_o is None:
            du = (s1 - s0) / n_sub_tangent
            pts.append(s0 + kk * (s1 - s0))
            wts.append(np.full(n_sub_tangent, du))
        else:
            if x_o <= s0:
                u0, u1, sgn = np.sqrt(s0 - x_o), np.sqrt(s1 - x_o), 1.0
            else:
                u0, u1, sgn = np.sqrt(x_o - s1), np.sqrt(x_o - s0), -1.0
            u = u0 + kk * (u1 - u0)
            pts.append(x_o + sgn * u**2)
            wts.append(2.0 * u * (u1 - u0) / n_sub_tangent)
        owner.append(np.full(n_sub_tangent, i))

    for i in near:
        reach = tangent_reach * width[i]
        cuts = np.concatenate(
            [
                [a[i]],
                tangencies[(tangencies > a[i]) & (tangencies < b[i])],
                [b[i]],
            ]
        )
        for p, q in zip(cuts[:-1], cuts[1:]):
            # Each end may diverge: integrate each half toward its own side.
            lt = tangencies[(tangencies <= p) & (p - tangencies <= reach)]
            rt = tangencies[(tangencies >= q) & (tangencies - q <= reach)]
            left = lt.max() if lt.size else None
            right = rt.min() if rt.size else None
            if left is not None and right is not None:
                mid = 0.5 * (p + q)
                segment(i, p, mid, left)
                segment(i, mid, q, right)
            else:
                segment(i, p, q, left if left is not None else right)
    pts, wts, owner = np.concatenate(pts), np.concatenate(wts), np.concatenate(owner)
    t = sc.tau_count(pts)[0]
    int_tau = np.bincount(owner, weights=t * wts, minlength=x.size)
    int_p = np.bincount(owner, weights=-np.expm1(-t) * wts, minlength=x.size)
    tau_avg[near] = int_tau[near] / width[near]
    p_avg[near] = int_p[near] / width[near]
    return tau_avg, p_avg


def tau_per_epsilon_sq(
    m_ev: float,
    x: ArrayLike,
    cosmo: CosmoLike | None = None,
    cell_averaged: bool = True,
) -> np.ndarray:
    """Conversion depth per unit ``ε²``, ``τ(x)/ε²``, with the full photon mass.

    In the linear regime the depletion is ``Δn = −ε² (τ/ε²) n_pl(x)``. This is
    the frequency template matching the solver's ``neutral_hydrogen`` option;
    it reduces to ``(γ_con/ε²)/x``, the default, where neutral hydrogen is
    negligible.

    Parameters
    ----------
    m_ev : float
        Dark-photon mass in **eV**.
    x : float or array_like
        Dimensionless frequencies; ascending when ``cell_averaged``.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to
        :data:`spectroxide.DEFAULT_COSMO`.
    cell_averaged : bool, optional
        If *True* (default) and ``x`` has at least two points, return cell
        averages (:func:`cell_average`), which stay finite and grid-independent
        at tangent crossings; otherwise return point values.

    Returns
    -------
    ndarray
        ``τ/ε²`` at each ``x`` (dimensionless).
    """
    if cell_averaged:
        return cell_average(1.0, m_ev, x, cosmo)[0]
    return conversion_probability(1.0, m_ev, x, cosmo).tau
