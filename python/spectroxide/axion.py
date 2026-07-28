"""Helpers for resonant axion-photon (γ ↔ a) conversion in the narrow-width
(Landau-Zener) approximation.

.. warning::
   **Not part of the released feature set.** The Rust side lives behind the
   off-by-default ``axion`` Cargo feature, so the PDE path below only works if
   the binary was built with ``cargo build --release --features axion``. These
   pure-Python helpers are importable regardless (they call no Rust), but the
   accompanying replication study has an unresolved 3-10x discrepancy against
   Cyr, Chluba & Manoj (2024) at m_a < 1e-11 eV. Treat as experimental.

Mirrors :mod:`spectroxide.dark_photon`. The PDE solver handles axion
conversions through the initial-condition path: pass
``injection={"type": "axion_resonance", "g_agamma": g, "b_rms": B, "m_ev": m}``
to :func:`spectroxide.solve` and the Rust solver computes ``γ_con``/``z_res``
itself, installing the impulsive depletion
``Δn(x) = −[1 − exp(−γ_con·x)] × n_pl(x)`` at ``z_start = z_res`` and evolving
forward in time. Use :func:`gamma_con_axion` for standalone diagnostics.

Two differences from the dark-photon case (Cyr, Chluba & Manoj 2024):

1. The conversion probability carries ``x`` in the numerator,
   ``P = 1 − exp(−γ_con·x)`` (Eq. 2), so high-frequency (Wien-tail) photons
   convert preferentially — opposite to the dark photon's ``1/x``.
2. The ``γ_con`` prefactor replaces ``ε² m²`` with ``κ² (1+z)⁴ T_CMB(z)``,
   where ``κ = g_aγγ B_rms`` (Eq. 3b).

This is the monopole / plasma-frequency treatment (``m_γ² ≈ ω_pl²``); the
frequency-dependent HI/HeI/HeII corrections (paper Sec. II B) are not modeled.

References
----------
- Cyr, Chluba & Manoj (2024), arXiv:2411.13701.
- Chluba, Cyr & Johnson (2024), MNRAS 535, 1874 [arXiv:2409.12115].
"""

from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np

from .dark_photon import (
    CosmoLike,
    _EV_IN_JOULES,
    _HBAR_EV_S,
    dln_omega_pl_sq_dlna,
    plasma_frequency_ev,
    resonance_redshift,
)
from .greens import _K_BOLTZMANN, _cosmo_hubble
from . import DEFAULT_COSMO

#: Normalization of κ = g_aγγ B_rms in eV per unit ε = (g/10⁻¹⁰GeV⁻¹)(B/nG).
#: Cyr, Chluba & Manoj (2024), Eq. 3b.
KAPPA_PER_EPSILON_EV = 1.95e-30

# Re-export the reused ω_pl helpers so callers can ``from spectroxide.axion
# import plasma_frequency_ev`` without reaching into dark_photon.
__all__ = [
    "KAPPA_PER_EPSILON_EV",
    "kappa_ev",
    "gamma_con_axion",
    "plasma_frequency_ev",
    "resonance_redshift",
    "dln_omega_pl_sq_dlna",
]


def kappa_ev(g_agamma: float, b_rms: float) -> float:
    """Axion mixing coupling ``κ = g_aγγ B_rms⁰`` (today), in eV.

    Parameters
    ----------
    g_agamma : float
        Axion-photon coupling ``g_aγγ`` in GeV⁻¹.
    b_rms : float
        Comoving RMS transverse magnetic field today in nG.

    Returns
    -------
    float
        ``κ`` in eV, using Eq. 3b/3c:
        ``κ = 1.95×10⁻³⁰ eV × (g_aγγ/10⁻¹⁰GeV⁻¹)(B_rms/nG)``.
    """
    return KAPPA_PER_EPSILON_EV * (g_agamma / 1.0e-10) * b_rms


def gamma_con_axion(
    g_agamma: float,
    b_rms: float,
    m_ev: float,
    cosmo: CosmoLike | None = None,
) -> Tuple[float | None, float | None]:
    """Narrow-width axion-photon conversion parameter ``γ_con``.

    .. math::

        \\gamma_{con} = \\frac{\\pi\\, \\kappa^2 (1+z_{res})^4 T_{CMB}(z_{res})}
            {m_a^2\\, H(z_{res})\\,
             \\bigl|d\\ln \\omega_{pl}^2 / d\\ln a\\bigr|_{z_{res}}},

    following Cyr, Chluba & Manoj (2024), Eq. 3a in the monopole limit
    ``m_γ² ≈ ω_pl²``.

    Parameters
    ----------
    g_agamma : float
        Axion-photon coupling ``g_aγγ`` in GeV⁻¹.
    b_rms : float
        Comoving RMS transverse magnetic field today in nG.
    m_ev : float
        Axion mass in eV.
    cosmo : Mapping, optional
        Cosmological parameters. Defaults to :data:`spectroxide.DEFAULT_COSMO`.

    Returns
    -------
    tuple of (float, float) or (None, None)
        ``(γ_con, z_res)`` if a resonance exists in the search bracket,
        otherwise ``(None, None)``.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    z_res = resonance_redshift(m_ev, cosmo)
    if z_res is None:
        return None, None
    kappa = kappa_ev(g_agamma, b_rms)
    t_cmb_ev = _K_BOLTZMANN * cosmo["t_cmb"] * (1.0 + z_res) / _EV_IN_JOULES
    h_ev = _HBAR_EV_S * _cosmo_hubble(z_res, cosmo)
    d = dln_omega_pl_sq_dlna(z_res, cosmo)
    gc = (
        np.pi
        * kappa**2
        * (1.0 + z_res) ** 4
        * t_cmb_ev
        / (m_ev**2 * h_ev * d)
    )
    return gc, z_res
