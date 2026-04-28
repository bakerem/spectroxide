"""Helpers for dark-photon (γ ↔ A') conversion in the narrow-width approximation.

The PDE solver handles dark-photon oscillations through the initial-condition
path: pass
``injection={"type": "dark_photon_resonance", "epsilon": ε, "m_ev": m}`` to
:func:`spectroxide.solve` and the Rust solver computes ``γ_con``/``z_res``
itself, applying ``Δn(x) = −[1 − exp(−γ_con/x)] × n_pl(x)`` at ``z_start =
z_res`` and evolving forward in time. Use :func:`gamma_con` for standalone
diagnostics that need the conversion probability without running the PDE.

References
----------
- Mirizzi, Redondo & Sigl (2009), JCAP 0903, 026.
- Chluba, Cyr & Johnson (2024), MNRAS 535, 1874 [arXiv:2409.12115].
"""

from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np
from scipy.optimize import brentq

from .greens import (
    _C_LIGHT,
    _HBAR,
    _K_BOLTZMANN,
    _cosmo_hubble,
    _cosmo_n_e,
)
from . import DEFAULT_COSMO, ionization_fraction

#: Type alias for cosmology mappings accepted by these helpers.
CosmoLike = Mapping[str, float]

_ALPHA_FS = 7.297_352_5693e-3
_M_ELECTRON = 9.109_383_7015e-31  # kg
_EV_IN_JOULES = 1.602_176_634e-19
_HBAR_EV_S = _HBAR / _EV_IN_JOULES


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

    following Chluba & Cyr (2024) Eq. 6.

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
