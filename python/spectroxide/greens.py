"""
Green's function for the cosmological thermalization problem.

Provides a fast, approximate method for computing spectral distortions
from arbitrary energy release histories. The distortion from a delta-function
energy injection at redshift ``z_h`` is decomposed into mu, y, and temperature
shift components using visibility or branching functions.

Ported from ``src/greens.rs`` and ``src/spectrum.rs``.

Conventions
-----------
This module uses the following conventions:

- Frequency variable: x = h ν / (k_B T_z), dimensionless.
- Redshift z is dimensionless; ``z_h`` denotes the *injection* redshift.
- All cosmology routines accept either ``DEFAULT_COSMO`` (Chluba 2013),
  ``PLANCK2015_COSMO``, or ``PLANCK2018_COSMO`` (re-exported from this
  module) — or any user dict with the same keys.

References
----------
- Chluba (2013), MNRAS 434, 352 [arXiv:1304.6120].
- Chluba & Jeong (2014), MNRAS 438, 2065 [arXiv:1306.5751].
- Chluba (2015), MNRAS 454, 4182 [arXiv:1506.06582].
"""

from __future__ import annotations

from typing import Callable, Mapping, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _validation as _val

#: Type alias for a cosmology parameter mapping (or any mapping accepting the
#: required keys ``h``, ``omega_b``, ``omega_m``, ``y_p``, ``t_cmb``,
#: ``n_eff``).
CosmoLike = Mapping[str, float]

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

#: Type alias for a scalar or NumPy array of float64 values.
FloatOrArray = Union[float, NDArray[np.float64]]


def _call_vectorized(
    func: Callable[..., ArrayLike], z_arr: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Call a user-provided callable with an array, falling back to scalar loop.

    Tries ``func(z_arr)`` first. Only specific broadcasting failure modes,
    such as "cannot broadcast" or "only integer scalar arrays can be converted",
    and shape mismatches trigger the scalar-loop fallback. Other
    TypeError or ValueError exceptions are assumed to be genuine bugs in
    ``func`` and are re-raised, preventing audit I3 (silently running
    a buggy user callable point-by-point and producing a misleading
    traceback).

    Parameters
    ----------
    func : callable
        Function to evaluate. Should accept a float or array_like and return
        a float or array_like of the same shape.
    z_arr : ndarray
        1-D array of input values.

    Returns
    -------
    ndarray
        Result array with the same shape as *z_arr*.
    """
    _ARRAY_FALLBACK_HINTS = (
        "broadcast",
        "only integer scalar arrays",
        "only size-1 arrays",
        "could not be coerced",
        "setting an array element",
        "ambiguous",
    )
    try:
        result = func(z_arr)
    except (TypeError, ValueError) as e:
        msg = str(e).lower()
        if not any(hint in msg for hint in _ARRAY_FALLBACK_HINTS):
            # Not a vectorization issue — let the real bug surface.
            raise
        return np.array([func(float(z)) for z in z_arr], dtype=np.float64)

    result = np.asarray(result, dtype=np.float64)
    if result.shape == z_arr.shape:
        return result
    # Scalar return or shape mismatch: fall back to per-element calls.
    # A downstream shape mismatch in the per-element result would raise a
    # clear error, which is the intended behavior.
    return np.array([func(float(z)) for z in z_arr], dtype=np.float64)


# ---------------------------------------------------------------------------
# Constants (from src/constants.rs)
# ---------------------------------------------------------------------------

Z_MU: float = 1.98e6
"""Mu-era thermalization redshift."""

G1_PLANCK: float = np.pi**2 / 6.0
r"""Planck integral :math:`G_1 = \int_0^\infty x\, n_{pl}(x)\, dx = \pi^2/6`."""

G2_PLANCK: float = 2 * 1.2020569031595943
r"""Planck integral :math:`G_2 = \int_0^\infty x^2\, n_{pl}(x)\, dx = 2\,\zeta(3)`."""

G3_PLANCK: float = np.pi**4 / 15.0
r"""Planck integral :math:`G_3 = \int_0^\infty x^3\, n_{pl}(x)\, dx = \pi^4/15`."""

BETA_MU: float = 2.192_288_908_204_316
"""Mu-distortion zero crossing :math:`\\beta_\\mu = 3\\,\\zeta(3)/\\zeta(2)`."""

KAPPA_C: float = 12.0 / BETA_MU - 9.0 * G2_PLANCK / G3_PLANCK
"""Mu-distortion normalisation :math:`\\kappa_c = 12/\\beta_\\mu - 9\\,G_2/G_3`."""

ALPHA_RHO: float = G2_PLANCK / G3_PLANCK
"""Photon-number-to-energy ratio :math:`\\alpha_\\rho = G_2/G_3 \\approx 0.3702`."""

X_BALANCED: float = 4.0 / (3.0 * ALPHA_RHO)
"""Balanced injection frequency :math:`x_0 = 4/(3\\,\\alpha_\\rho) \\approx 3.60`.

Photon injection at :math:`x = x_0` produces zero net :math:`\\mu`-distortion."""


# ---------------------------------------------------------------------------
# Spectral shapes (from src/spectrum.rs)
# ---------------------------------------------------------------------------


def planck(x: ArrayLike) -> NDArray[np.float64]:
    """Planck (blackbody) occupation number ``n_pl(x) = 1 / (e^x − 1)``.

    Uses series expansions for ``x < 1e-6`` and ``x > 500`` to avoid
    catastrophic cancellation and overflow.

    Parameters
    ----------
    x : array_like
        Dimensionless frequency ``h ν / (k_B T_z)``.

    Returns
    -------
    ndarray of float64
        Planck occupation number.
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    small = x < 1e-6
    large = x > 500.0
    mid = ~small & ~large
    result[small] = 1.0 / x[small] - 0.5 + x[small] / 12.0
    result[large] = np.exp(-x[large])
    result[mid] = 1.0 / (np.exp(x[mid]) - 1.0)
    return result


def g_bb(x: ArrayLike) -> NDArray[np.float64]:
    """Blackbody derivative ``G_bb(x) = x e^x / (e^x − 1)^2``.

    Equal to ``-x · dn_pl/dx`` and represents the spectral response of
    a blackbody to a small temperature shift ΔT/T.

    Parameters
    ----------
    x : array_like
        Dimensionless frequency ``h ν / (k_B T_z)``.

    Returns
    -------
    ndarray of float64
        ``G_bb(x)`` evaluated pointwise.
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    small = x < 1e-6
    large = x > 500.0
    mid = ~small & ~large
    result[small] = 1.0 / x[small] - x[small] / 12.0
    result[large] = x[large] * np.exp(-x[large])
    ex = np.exp(x[mid])
    result[mid] = x[mid] * ex / (ex - 1.0) ** 2
    return result


def mu_shape(x: ArrayLike) -> NDArray[np.float64]:
    """μ-distortion spectral shape ``M(x) = (x/β_μ − 1) · G_bb(x) / x``.

    Crosses zero at ``x = β_μ ≈ 2.19`` (frequency of the mu-distortion null).

    Parameters
    ----------
    x : array_like
        Dimensionless frequency.

    Returns
    -------
    ndarray of float64
        ``M(x)`` evaluated pointwise.
    """
    x = np.asarray(x, dtype=np.float64)
    return (x / BETA_MU - 1.0) * g_bb(x) / x


def y_shape(x: ArrayLike) -> NDArray[np.float64]:
    """y-distortion (Sunyaev–Zel'dovich) spectral shape.

    ``Y_SZ(x) = G_bb(x) · [x coth(x/2) − 4]``.

    Crosses zero at ``x ≈ 3.83`` (the SZ null in the cosmic microwave
    background intensity spectrum).

    Parameters
    ----------
    x : array_like
        Dimensionless frequency.

    Returns
    -------
    ndarray of float64
        ``Y_SZ(x)`` evaluated pointwise.
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    small = x < 1e-6
    mid = ~small
    # Small-x: G_bb ~ 1/x - x/12, (x*coth(x/2) - 4) ~ -2 + x^2/6
    # Product: (1/x)(-2) + (1/x)(x^2/6) + (-x/12)(-2) = -2/x + x/6 + x/6 = -2/x + x/3
    result[small] = -2.0 / x[small] + x[small] / 3.0
    result[mid] = g_bb(x[mid]) * (
        x[mid] * np.cosh(x[mid] / 2.0) / np.sinh(x[mid] / 2.0) - 4.0
    )
    return result


def temperature_shift_shape(x: ArrayLike) -> NDArray[np.float64]:
    """Temperature shift spectral shape ``G(x) = x e^x / (e^x − 1)^2``.

    Identical to :func:`g_bb`; provided as a separate name to make
    decomposition expressions read more clearly.  Represents the response
    ΔI ∝ dB/dT to a small temperature perturbation.

    Parameters
    ----------
    x : array_like
        Dimensionless frequency.

    Returns
    -------
    ndarray of float64
        ``G(x)`` evaluated pointwise.
    """
    return g_bb(x)


# ---------------------------------------------------------------------------
# Visibility / branching functions (from src/greens.rs)
# ---------------------------------------------------------------------------


def j_bb(z: ArrayLike) -> NDArray[np.float64]:
    """Thermalization visibility ``J_bb(z) = exp(−(z/z_μ)^{5/2})``.

    Probability that injected energy at redshift ``z`` is *fully*
    thermalized into a blackbody by the present epoch.

    Both ``z_μ`` and the exponent 5/2 are analytically derived:
    ``z_μ`` from equating the double Compton (DC) and bremsstrahlung (BR)
    photon production rate to the Hubble rate (Chluba & Sunyaev 2012), and
    5/2 from the DC opacity scaling (Danese & de Zotti 1982; Hu & Silk 1993).
    These are *not* fit parameters.

    Parameters
    ----------
    z : float or array_like
        Injection redshift.

    Returns
    -------
    ndarray of float64
        ``J_bb(z)`` ∈ [0, 1].
    """
    z = np.asarray(z, dtype=np.float64)
    ratio = z / Z_MU
    return np.exp(-(ratio**2.5))


def j_bb_star(z: ArrayLike) -> NDArray[np.float64]:
    """Improved thermalization visibility with the Chluba (2015) correction.

    ``J_bb*(z) = 0.983 · J_bb(z) · (1 − 0.0381 · (z/z_μ)^{2.29})``.

    Reference: Chluba (2015), arXiv:1506.06582, Eq. 13.  Valid for
    3 × 10⁵ ≲ z ≲ 6 × 10⁶ in the standard cosmology (Chluba 2014 fit,
    neglecting relativistic temperature corrections that become noticeable
    at z_i ≳ 4 × 10⁶).  The prefactor 0.983 absorbs the small residual
    blackbody mismatch during the mu-era.  The base :func:`j_bb` exponent
    (5/2) and ``z_μ`` are analytically derived.

    The result is clamped at 0 because the empirical correction factor
    becomes negative for ``z/z_μ ≳ 3.9``, outside the fit's range of
    validity.

    Parameters
    ----------
    z : float or array_like
        Injection redshift.

    Returns
    -------
    ndarray of float64
        ``J_bb*(z)`` ∈ [0, 1].
    """
    z = np.asarray(z, dtype=np.float64)
    ratio = z / Z_MU
    # Clamp at 0: correction factor goes negative for z/z_mu >~ 3.9,
    # outside the fit's range of validity. Physically J_bb* in [0, 1].
    return np.maximum(0.983 * j_bb(z) * (1.0 - 0.0381 * ratio**2.29), 0.0)


def j_mu(z: ArrayLike) -> NDArray[np.float64]:
    """μ-distortion branching ratio.

    ``J_mu(z) = 1 − exp(−((1+z)/5.8 × 10⁴)^{1.88})``.

    Reference: Chluba (2013), arXiv:1304.6120, Eq. 5.  Approaches 0 for
    z ≪ 5.8 × 10⁴ (no μ) and 1 for z ≫ 5.8 × 10⁴ (pure μ).  The transition
    scale is physically motivated by y_γ(z) ~ 1, but the precise value
    and exponent are fit parameters.

    Parameters
    ----------
    z : float or array_like
        Injection redshift.

    Returns
    -------
    ndarray of float64
        ``J_μ(z)`` ∈ [0, 1].
    """
    z = np.asarray(z, dtype=np.float64)
    return 1.0 - np.exp(-(((1.0 + z) / 5.8e4) ** 1.88))


def j_y(z: ArrayLike) -> NDArray[np.float64]:
    """y-distortion branching ratio.

    ``J_y(z) = 1 / (1 + ((1+z)/6.0 × 10⁴)^{2.58})``.

    Reference: Chluba (2013), arXiv:1304.6120, Eq. 5.  Least-squares fit
    to the partial differential equation (PDE) Green's function in the
    μ–y transition era.  Approaches 1 for z ≪ 6 × 10⁴ (pure y-era) and 0
    for z ≫ 6 × 10⁴.  The transition scale z ~ 6 × 10⁴ is physically
    motivated by y_γ(z) ~ 1, but the precise value and exponent are fit
    parameters.

    Note that ``J_y ≠ 1 − J_μ`` in the transition region; using the
    independent fit gives better spectral agreement with PDE results.

    Parameters
    ----------
    z : float or array_like
        Injection redshift.

    Returns
    -------
    ndarray of float64
        ``J_y(z)`` ∈ [0, 1].
    """
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + ((1.0 + z) / 6.0e4) ** 2.58)


# ---------------------------------------------------------------------------
# Green's function
# ---------------------------------------------------------------------------


def greens_function(x: ArrayLike, z_h: float) -> NDArray[np.float64]:
    """Three-component Green's function ``G_th(x, z_h)`` (Chluba 2013).

    Spectral distortion observed at ``z = 0`` per unit ``Δρ/ρ`` injected
    as a delta function at redshift ``z_h``:

    .. math::

        G_{th}(x, z_h) = \\frac{3}{\\kappa_c} J_\\mu J_{bb}^* M(x)
            + \\frac{1}{4} J_y \\, Y_{SZ}(x)
            + \\frac{1}{4} (1 - J_{bb}^*) G_{bb}(x),

    where ``J_mu``, ``J_y``, and ``J_bb*`` are independently fitted
    visibility functions.  The temperature-shift weight ``(1 − J_bb*)/4``
    follows the Chluba (2013) convention.

    Accuracy (compared with PDE)
    ----------------------------
    Compared with the PDE solver, the accuracy is:

    - Deep μ-era (z_h > 3 × 10⁵): <17% per-point shape error; <5% on
      integrated μ.
    - y-era (z_h < 10⁴): <5% per-point shape error; <1% on integrated y.
    - Transition era (z_h ~ 10⁴–3 × 10⁵): 8–17% per-point shape error.

    Parameters
    ----------
    x : float or array_like
        Dimensionless frequency ``h ν / (k_B T_z)``.
    z_h : float
        Injection redshift (must be positive and finite).

    Returns
    -------
    ndarray of float64
        Spectral distortion ``Δn(x)`` per unit ``Δρ/ρ``.

    Raises
    ------
    ValueError
        If ``z_h ≤ 0`` or ``x`` contains non-positive entries.

    See Also
    --------
    distortion_from_heating : convolution over a heating history.

    References
    ----------
    - Chluba (2013), MNRAS 434, 352 [arXiv:1304.6120].
    """
    _val.validate_z_h(z_h)
    _val.validate_x_positive(x)
    _val.warn_z_h_regime(z_h)
    x = np.asarray(x, dtype=np.float64)

    _j_mu = j_mu(z_h)
    _j_bb_star = j_bb_star(z_h)
    _j_y = j_y(z_h)

    mu_part = (3.0 / KAPPA_C) * _j_mu * _j_bb_star * mu_shape(x)
    y_part = 0.25 * _j_y * y_shape(x)
    t_part = 0.25 * (1.0 - _j_bb_star) * temperature_shift_shape(x)

    return mu_part + y_part + t_part


# ---------------------------------------------------------------------------
# Integrated distortion parameters
# ---------------------------------------------------------------------------


@_val.renamed_kwargs(x_grid="x")
def distortion_from_heating(
    x: ArrayLike,
    dq_dz: Callable[[ArrayLike], ArrayLike],
    z_min: float,
    z_max: float,
    n_z: int = 5000,
) -> NDArray[np.float64]:
    """Spectral distortion from an arbitrary energy release history.

    .. math::

        \\Delta n(x) = \\int_{z_{min}}^{z_{max}} G_{th}(x, z')
                       \\frac{d(\\Delta\\rho/\\rho_\\gamma)}{dz'} \\, dz'.

    Integration is performed in ``ln(1 + z)`` for numerical stability,
    using the trapezoidal rule.

    Parameters
    ----------
    x : array_like
        Frequency grid (must be positive and finite).
    dq_dz : callable
        Heating rate ``d(Δρ/ρ_γ)/dz`` as a function of redshift.  Sign
        convention: positive for heating.  Should accept either a scalar
        or an array; vectorized calls are attempted first and fall back
        to scalar evaluation on broadcasting failure.
    z_min : float
        Minimum integration redshift (must satisfy ``0 ≤ z_min < z_max``).
    z_max : float
        Maximum integration redshift.
    n_z : int, optional
        Number of redshift integration points.  Default 5000; recommended
        ≥ 2000 per ``log10(1+z)`` decade for broad ranges.

    Returns
    -------
    ndarray of float64
        Distortion ``Δn(x)`` evaluated on ``x``.

    Raises
    ------
    ValueError
        If ``x`` contains non-positive entries or the redshift range
        is invalid.
    """
    _val.validate_x_positive(x)
    _val.validate_z_range(z_min, z_max, n_z)
    _val.warn_z_max_regime(z_max)
    _val.warn_analytic_gf_heating(z_min, z_max)
    _val.warn_convolution_resolution(n_z, z_min, z_max)
    x = np.asarray(x, dtype=np.float64)
    ln_min = np.log(1.0 + z_min)
    ln_max = np.log(1.0 + z_max)
    dln = (ln_max - ln_min) / max(n_z - 1, 1)

    # Build redshift array and weights
    j_arr = np.arange(n_z)
    ln_1pz = ln_min + j_arr * dln
    z_arr = np.exp(ln_1pz) - 1.0
    dz_dln = 1.0 + z_arr

    heating = _call_vectorized(dq_dz, z_arr) * dz_dln  # shape (n_z,)

    w = np.full(n_z, dln)
    w[0] = 0.5 * dln
    w[-1] = 0.5 * dln

    hw = heating * w  # shape (n_z,)

    # Decompose greens_function into shape * coefficient:
    #   G(x, z) = c_mu(z)*M(x) + c_y(z)*Y(x) + c_t(z)*G_bb(x)
    jmu = j_mu(z_arr)  # shape (n_z,)
    jbb = j_bb_star(z_arr)  # shape (n_z,)
    jyv = j_y(z_arr)  # shape (n_z,)

    c_mu = (3.0 / KAPPA_C) * jmu * jbb  # shape (n_z,)
    c_y = 0.25 * jyv  # shape (n_z,)
    c_t = 0.25 * (1.0 - jbb)  # shape (n_z,)

    # Precompute spectral shapes once: shape (n_x,)
    m_x = mu_shape(x)
    y_x = y_shape(x)
    g_x = g_bb(x)

    # Weighted sum using outer products: delta_n = sum_z hw(z) * G(x, z)
    # = M(x) * sum(c_mu * hw) + Y(x) * sum(c_y * hw) + G(x) * sum(c_t * hw)
    delta_n = m_x * np.dot(c_mu, hw) + y_x * np.dot(c_y, hw) + g_x * np.dot(c_t, hw)

    return delta_n


def mu_from_heating(
    dq_dz: Callable[[ArrayLike], ArrayLike],
    z_min: float,
    z_max: float,
    n_z: int = 5000,
) -> float:
    """μ parameter from the Green's function approximation.

    .. math::

        \\mu = \\frac{3}{\\kappa_c} \\int_{z_{min}}^{z_{max}}
               J_{bb}^*(z) \\, J_\\mu(z)
               \\frac{d(\\Delta\\rho/\\rho)}{dz} \\, dz.

    Parameters
    ----------
    dq_dz : callable
        Heating rate ``d(Δρ/ρ_γ)/dz`` (positive for heating).
    z_min : float
        Minimum integration redshift.
    z_max : float
        Maximum integration redshift.
    n_z : int, optional
        Number of redshift integration points (default 5000).

    Returns
    -------
    float
        Chemical-potential parameter ``μ`` (dimensionless).
    """
    _val.validate_z_range(z_min, z_max, n_z)
    _val.warn_z_max_regime(z_max)
    _val.warn_convolution_resolution(n_z, z_min, z_max)
    ln_min = np.log(1.0 + z_min)
    ln_max = np.log(1.0 + z_max)
    dln = (ln_max - ln_min) / max(n_z - 1, 1)

    # Build redshift array
    j_arr = np.arange(n_z)
    ln_1pz = ln_min + j_arr * dln
    z_arr = np.exp(ln_1pz) - 1.0
    dz_dln = 1.0 + z_arr

    # Vectorized heating rate
    heating = _call_vectorized(dq_dz, z_arr) * dz_dln

    # Trapezoidal weights
    w = np.full(n_z, dln)
    w[0] = 0.5 * dln
    w[-1] = 0.5 * dln

    # Vectorized visibility functions (already array-safe)
    result = float(
        np.dot((3.0 / KAPPA_C) * j_bb_star(z_arr) * j_mu(z_arr) * heating, w)
    )
    return result


def y_from_heating(
    dq_dz: Callable[[ArrayLike], ArrayLike],
    z_min: float,
    z_max: float,
    n_z: int = 5000,
) -> float:
    """Compton-y parameter from the Green's function approximation.

    .. math::

        y = \\frac{1}{4} \\int_{z_{min}}^{z_{max}} J_y(z)
            \\frac{d(\\Delta\\rho/\\rho)}{dz} \\, dz.

    Uses the independently fitted ``J_y`` (see :func:`j_y`), which gives
    better agreement with PDE results than ``(1 − J_μ)``.

    Parameters
    ----------
    dq_dz : callable
        Heating rate ``d(Δρ/ρ_γ)/dz`` (positive for heating).
    z_min : float
        Minimum integration redshift.
    z_max : float
        Maximum integration redshift.
    n_z : int, optional
        Number of redshift integration points (default 5000).

    Returns
    -------
    float
        Compton y-parameter (dimensionless).
    """
    _val.validate_z_range(z_min, z_max, n_z)
    _val.warn_z_max_regime(z_max)
    _val.warn_convolution_resolution(n_z, z_min, z_max)
    ln_min = np.log(1.0 + z_min)
    ln_max = np.log(1.0 + z_max)
    dln = (ln_max - ln_min) / max(n_z - 1, 1)

    # Build redshift array
    j_arr = np.arange(n_z)
    ln_1pz = ln_min + j_arr * dln
    z_arr = np.exp(ln_1pz) - 1.0
    dz_dln = 1.0 + z_arr

    # Vectorized heating rate
    heating = _call_vectorized(dq_dz, z_arr) * dz_dln

    # Trapezoidal weights
    w = np.full(n_z, dln)
    w[0] = 0.5 * dln
    w[-1] = 0.5 * dln

    result = float(np.dot(0.25 * j_y(z_arr) * heating, w))
    return result


# ---------------------------------------------------------------------------
# Silk damping and ΛCDM distortions
# ---------------------------------------------------------------------------

# Cosmology background, presets, recombination history, and physical
# constants live in ``spectroxide.cosmology``. Re-imported here so legacy imports
# such as ``from spectroxide.greens import hubble``, ``cosmic_time``,
# ``DEFAULT_COSMO``, or ``_C_LIGHT`` keep working.
# DEPRECATED back-compat shim: prefer the canonical path
# ``from spectroxide import cosmic_time`` (or ``spectroxide.cosmology``).
# Remove once no in-repo docs/notebooks import cosmology names through
# ``spectroxide.greens`` (grep: `spectroxide.greens import .*cosmic_time`).
from .cosmology import (  # noqa: E402,F401
    DEFAULT_COSMO,
    COSMOTHERM_GF_COSMO,
    _C_LIGHT,
    _K_BOLTZMANN,
    _HBAR,
    _M_ELECTRON,
    _SIGMA_THOMSON,
    _cosmo_hubble,
    _cosmo_n_h,
    _cosmo_n_e,
    _saha_he_i,
    _saha_he_ii,
    ionization_fraction,
    baryon_photon_ratio,
    hubble,
    n_hydrogen,
    n_electron,
    omega_gamma,
    rho_gamma,
    cosmic_time,
)

# ---------------------------------------------------------------------------
# Photon injection Green's function
# ---------------------------------------------------------------------------


def x_c_dc(z: ArrayLike) -> NDArray[np.float64]:
    """Critical frequency for double-Compton absorption.

    ``x_c^{DC}(z) = 8.60 × 10⁻³ · ((1+z)/2 × 10⁶)^{1/2}``.

    Reference: Chluba (2015), arXiv:1506.06582, Eq. 25a.

    Parameters
    ----------
    z : float or array_like
        Redshift.

    Returns
    -------
    ndarray of float64
        Dimensionless critical frequency.
    """
    z = np.asarray(z, dtype=np.float64)
    return 8.60e-3 * ((1.0 + z) / 2.0e6) ** 0.5


def x_c_br(z: ArrayLike) -> NDArray[np.float64]:
    """Critical frequency for bremsstrahlung absorption.

    ``x_c^{BR}(z) = 1.23 × 10⁻³ · ((1+z)/2 × 10⁶)^{−0.672}``.

    Reference: Chluba (2015), arXiv:1506.06582, Eq. 25b.

    Parameters
    ----------
    z : float or array_like
        Redshift.

    Returns
    -------
    ndarray of float64
        Dimensionless critical frequency.
    """
    z = np.asarray(z, dtype=np.float64)
    return 1.23e-3 * ((1.0 + z) / 2.0e6) ** (-0.672)


def x_c(z: ArrayLike) -> NDArray[np.float64]:
    """Combined critical frequency for photon absorption.

    ``x_c² = x_c^{DC}² + x_c^{BR}²`` (quadrature addition).

    Photons with ``x ≪ x_c`` are absorbed by DC/BR; photons with
    ``x ≫ x_c`` survive.

    Parameters
    ----------
    z : float or array_like
        Redshift.

    Returns
    -------
    ndarray of float64
        Dimensionless critical frequency.
    """
    dc = x_c_dc(z)
    br = x_c_br(z)
    return np.sqrt(dc**2 + br**2)


def photon_survival_probability(
    x: ArrayLike, z: float, cosmo: CosmoLike | None = None
) -> NDArray[np.float64]:
    """Photon survival probability ``P_s(x, z)``.

    Probability that a photon injected at frequency ``x`` and redshift
    ``z`` survives absorption by DC/BR processes.

    - ``cosmo=None`` (default): the analytic form
      ``P_s = exp(−x_c(z)/x)`` (Chluba 2015, arXiv:1506.06582, Eq. 24).
    - ``cosmo`` given: the cosmology-aware estimate that
      :func:`greens_function_photon` uses.  For ``z > 5e4`` it equals the
      analytic form; for ``z ≤ 5e4`` it is ``exp(−τ_ff)``, with the DC+BR
      optical depth ``τ_ff`` integrated from ``z = 200`` to ``z`` for that
      cosmology (Chluba 2015, Eqs. 29 and 32).  In the y-era the two
      differ by orders of magnitude at ``x ≲ x_c``.

    Parameters
    ----------
    x : float or array_like
        Dimensionless frequency.
    z : float
        Redshift.
    cosmo : Mapping or Cosmology, optional
        Cosmological parameters, for example
        :data:`~spectroxide.cosmology.DEFAULT_COSMO`.  Default *None*
        (analytic form).

    Returns
    -------
    ndarray of float64
        Survival probability ``P_s ∈ [0, 1]``; zero for non-positive ``x``.
    """
    if cosmo is not None:
        cosmo = _cosmo_mapping(cosmo)
        x_arr = np.asarray(x, dtype=np.float64)
        flat = [
            _photon_survival_probability_numerical(float(xi), float(z), cosmo)
            for xi in x_arr.ravel()
        ]
        return np.asarray(flat, dtype=np.float64).reshape(x_arr.shape)
    x = np.asarray(x, dtype=np.float64)
    xc = float(x_c(z))
    # Avoid division by zero
    with np.errstate(divide="ignore", over="ignore"):
        ratio = xc / np.where(x > 1e-30, x, 1e-30)
        result = np.exp(-ratio)
    result = np.where(x > 1e-30, result, 0.0)
    return result


# ---------------------------------------------------------------------------
# Numerical photon survival probability (Chluba 2015, Eq. 29/32)
# ---------------------------------------------------------------------------

_H_PLANCK = 6.626_070_15e-34  # J·s (exact by SI definition)
_M_E_C2 = _M_ELECTRON * _C_LIGHT**2
_ALPHA_FS = 7.297_352_5693e-3
_LAMBDA_ELECTRON = _H_PLANCK / (_M_ELECTRON * _C_LIGHT)
_I4_PLANCK = 4.0 * G3_PLANCK  # = 4 pi^4/15
_BR_PREFACTOR = _ALPHA_FS * _LAMBDA_ELECTRON**3 / (2.0 * np.pi * np.sqrt(6.0 * np.pi))
_S3_PI = np.sqrt(3.0) / np.pi


def _softplus(a):
    """softplus(a) = ln(1 + exp(a)), with asymptotic shortcuts."""
    if a > 20.0:
        return a
    elif a < -20.0:
        return np.exp(a)
    else:
        return np.log1p(np.exp(a))


def _gaunt_ff_nr(x_e, theta_e, z_charge):
    """Non-relativistic free-free Gaunt factor (softplus interpolation).

    ``x_e`` is h nu / (k T_e) = x / rho_e, not the grid variable x = h nu / (k T_z).
    This is Draine (2011), *Physics of the Interstellar and Intergalactic
    Medium*, Ch. 10 interpolation formula (equation number believed to be
    10.8, not confirmed), rewritten in (x_e, theta_e); the low-frequency limit
    is the classical Gaunt factor, Draine Eq. 10.9. Chluba, Ravenni & Bolliet
    (2020) remains the accurate reference for exact Gaunt factors but is not
    the source of this fit. Mirrors ``gaunt_ff_nr`` in ``src/bremsstrahlung.rs``.
    """
    if theta_e < 1e-30 or x_e < 1e-30:
        return 1.0
    arg = _S3_PI * (np.log(2.25 / (x_e * z_charge)) + 0.5 * np.log(theta_e)) + 1.425
    return 1.0 + _softplus(arg)


def _dc_high_freq_suppression(x):
    """DC high-frequency suppression factor."""
    if x > 100.0:
        return 0.0
    return np.exp(-2.0 * x) * (
        1.0 + 1.5 * x + 29.0 / 24.0 * x**2 + 11.0 / 16.0 * x**3 + 5.0 / 12.0 * x**4
    )


def _dc_emission_coefficient(x, theta_z):
    """DC emission coefficient K_DC (per Thomson time)."""
    rel_corr = 1.0 / (1.0 + 14.16 * theta_z)
    h_dc = _dc_high_freq_suppression(x)
    return (4.0 * _ALPHA_FS / (3.0 * np.pi)) * theta_z**2 * _I4_PLANCK * rel_corr * h_dc


def _br_emission_coefficient_with_he(
    x, theta_e, theta_z, n_h, n_he, n_e, x_e_frac, y_he_ii, y_he_i
):
    """BR emission coefficient K_BR (per Thomson time), with He ionization."""
    if theta_e < 1e-30 or n_e < 1e-30:
        return 0.0
    phi = theta_z / theta_e
    temp_factor = theta_e ** (-3.5) * np.exp(-x * phi) / phi**3

    # The Gaunt factor takes x_e = x * phi = h nu / (k T_e), not the grid x.
    g_z1 = _gaunt_ff_nr(x * phi, theta_e, 1.0)
    g_z2 = _gaunt_ff_nr(x * phi, theta_e, 2.0)

    n_hii = min(x_e_frac, 1.0) * n_h
    n_heiii = y_he_ii * n_he
    n_heii = max(y_he_i - y_he_ii, 0.0) * n_he

    species_sum = n_hii * g_z1 + 4.0 * n_heiii * g_z2 + n_heii * g_z1
    return _BR_PREFACTOR * temp_factor * species_sum


# Largest z at which the numerical P_s uses the raw tau_ff integral. Independent of
# the photon-GF validity window (1e4, 3e5): at 1e4 the analytic exp(-x_c/x)
# over-absorbs badly, and the two forms are closest near 5e4 (ADR 0005 addendum).
P_S_TAU_FF_Z_MAX = 5.0e4


def _photon_survival_probability_numerical(x, z_h, cosmo):
    """Numerical P_s using integrated DC+BR optical depth for the y-era.

    For z_h > 5e4, falls back to the analytic exp(-x_c/x).
    For z_h <= 5e4, integrates tau_ff from z=200 to z_h.

    Reference: Chluba (2015), arXiv:1506.06582, Eq. 29/32
    """
    if z_h > P_S_TAU_FF_Z_MAX:
        return float(photon_survival_probability(np.array([x]), z_h)[0])
    if x < 1e-30:
        return 0.0
    z_end = 200.0
    if z_h <= z_end:
        return 1.0

    n_steps = 500
    log_z_h = np.log(z_h)
    log_z_end = np.log(z_end)
    d_log_z = (log_z_h - log_z_end) / n_steps

    # Overflow at x > 500 → +inf propagates through `rate` and triggers the
    # tau > 500 → 0 saturation below (mirrors Rust tau_ff_survival; a large
    # finite sentinel would pass finite-checks elsewhere).
    bose_factor = np.expm1(x) if x <= 500.0 else np.inf
    inv_x3 = 1.0 / (x * x * x)
    f_he = cosmo["y_p"] / (4.0 * (1.0 - cosmo["y_p"]))

    # Full ionization fraction including H (Peebles TLA below the Saha switch)
    # and He (He²⁺ at z≳8000, He⁺ at 2000≲z≲8000). Mirrors Rust
    # tau_ff_survival, which uses recombination::ionization_fraction. The
    # previous raw-Saha branching exponentially underestimated X_e below
    # z ≈ 1500 (no freeze-out) and omitted helium electrons at z > 1500.
    z_grid = np.exp(log_z_end + np.arange(n_steps + 1) * d_log_z)
    x_e_grid = np.asarray(ionization_fraction(z_grid, cosmo), dtype=np.float64)

    tau = 0.0
    for i in range(n_steps + 1):
        z = float(z_grid[i])
        opz = 1.0 + z

        tz = _K_BOLTZMANN * cosmo["t_cmb"] * opz / _M_E_C2
        te = tz  # T_e ~ T_z in y-era

        x_e_frac = float(x_e_grid[i])

        n_h = _cosmo_n_h(z, cosmo)
        n_he = f_he * n_h
        n_e = x_e_frac * n_h

        k_dc = _dc_emission_coefficient(x, tz)
        y_he_ii = _saha_he_ii(z, cosmo)
        y_he_i = _saha_he_i(z, cosmo)
        k_br = _br_emission_coefficient_with_he(
            x, te, tz, n_h, n_he, n_e, min(x_e_frac, 1.0), y_he_ii, y_he_i
        )

        rate = (k_dc + k_br) * bose_factor * inv_x3
        dtau_dz = n_e * _SIGMA_THOMSON * _C_LIGHT / (opz * _cosmo_hubble(z, cosmo))

        weight = 0.5 if (i == 0 or i == n_steps) else 1.0
        tau += weight * rate * dtau_dz * z * d_log_z

    if tau > 500.0:
        return 0.0
    return np.exp(-tau)


# ---------------------------------------------------------------------------
# Photon injection helpers
# ---------------------------------------------------------------------------


def _y_compton(z: ArrayLike, cosmo: CosmoLike | None = None) -> FloatOrArray:
    """Integrated Compton y-parameter ``y_γ(z) = ∫₀ᶻ θ_e σ_T n_e c / H dz'``.

    Internal helper for the photon-injection broadened bump. 128-point
    midpoint rule in ``ln(1+z)``, mirroring the Rust
    ``Cosmology::compton_y_parameter`` exactly (32-point Gauss–Legendre
    under-resolved the steep X_e drop at recombination by up to ~2.5%).
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO

    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    scalar = np.ndim(z) == 0

    n_nodes = 128
    ln_max = np.log(1.0 + z_arr)  # shape (N,)
    h_step = ln_max / n_nodes  # shape (N,)

    # Midpoint nodes for all (z, node) pairs: shape (N, 128)
    i_mid = np.arange(n_nodes) + 0.5
    u = h_step[:, np.newaxis] * i_mid[np.newaxis, :]
    zp = np.exp(u) - 1.0  # shape (N, 128)
    opz = 1.0 + zp

    # Matter temperature tracks T_γ while Compton coupling is strong
    # (z ≳ z_dec ≈ 200) and drops as (1+z)² after decoupling. Mirrors the
    # Rust Cosmology::compton_y_parameter (audit M1): using T_γ below z_dec
    # overestimates the integrand by a few % at worst.
    z_dec = 200.0
    t_z = cosmo["t_cmb"] * opz
    t_matter = cosmo["t_cmb"] * opz**2 / (1.0 + z_dec)
    theta_e = (
        _K_BOLTZMANN * np.where(zp > z_dec, t_z, t_matter) / (_M_ELECTRON * _C_LIGHT**2)
    )

    # _cosmo_n_e needs ionization_fraction which handles arrays
    # Flatten for the call, then reshape
    zp_flat = zp.ravel()
    n_e_flat = _cosmo_n_e(zp_flat, cosmo)
    n_e = np.asarray(n_e_flat, dtype=np.float64).reshape(zp.shape)

    h_z = _cosmo_hubble(zp, cosmo)  # array-safe

    # Integrand: theta_e * sigma_T * c * n_e / h_z, shape (N, 128)
    integrand = theta_e * _SIGMA_THOMSON * _C_LIGHT * n_e / h_z

    # Midpoint sum: shape (N,)
    result = integrand.sum(axis=1) * h_step

    if scalar:
        return float(result[0])
    return result


# Photon injection uses the universal j_mu(z) visibility; an
# x'-dependent transition table was tried and removed (poorly motivated).


def _f_cs(x):
    """Compton scattering helper f(x) = exp(-x) * (1 + x^2/2)."""
    return np.exp(-x) * (1.0 + 0.5 * x**2)


def _alpha_cs(x, yg):
    """Compton scattering alpha parameter.

    alpha(x, y_gamma) = (3 - 2*f(x)) / sqrt(1 + x*y_gamma)
    """
    return (3.0 - 2.0 * _f_cs(x)) / np.sqrt(1.0 + x * yg)


def _beta_cs(x, yg):
    """Compton scattering beta parameter.

    beta(x, y_gamma) = 1 / (1 + x*y_gamma*(1-f(x)))
    """
    return 1.0 / (1.0 + x * yg * (1.0 - _f_cs(x)))


def _broadened_bump(x_obs, x_inj, yg):
    """Broadened surviving photon bump (Chluba 2015, Eq. 38-39).

    The Compton-scattered photon distribution is log-normal in x
    (Gaussian in ln x). Returns (F(x_obs), f_int) where F is a
    normalized log-normal (integral = 1, one surviving photon) and
    f_int = <x>/x_inj is the mean energy ratio of the surviving photon.

    For yg < 1e-6, falls back to a narrow Gaussian at x_inj.

    Parameters
    ----------
    x_obs : ndarray
        Observation frequencies.
    x_inj : float
        Injection frequency.
    yg : float
        Compton y-parameter at injection redshift.

    Returns
    -------
    bump : ndarray
        Normalized log-normal bump shape (integral = 1).
    f_int : float
        Mean energy ratio <x>/x_inj = exp((alpha+beta)*yg)/(1+x'*yg).
    """
    if yg < 1e-6:
        # Narrow Gaussian fallback
        sigma = 0.005 * x_inj
        norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        bump = np.exp(-((x_obs - x_inj) ** 2) / (2.0 * sigma**2)) * norm
        return bump, 1.0

    alpha = _alpha_cs(x_inj, yg)
    beta = _beta_cs(x_inj, yg)

    denom = 1.0 + x_inj * yg
    # Log-normal parameters
    # Median at x_med = x_inj * exp(alpha*yg) / denom (the mode sits at
    # x_med * exp(-sigma_ln^2), lower by O(y_γ); negligible for y_γ ≪ 1)
    # Mean at x_mean = x_inj * exp((alpha+beta)*yg) / denom = x_inj * f_int
    mu_ln = np.log(x_inj) + alpha * yg - np.log(denom)  # = ln(x_med)
    sigma_ln_sq = 2.0 * beta * yg
    sigma_ln = np.sqrt(max(sigma_ln_sq, 1e-30))

    exp_arg = (alpha + beta) * yg
    # Clamp to avoid f64 overflow (exp(709) ≈ 8.2e307)
    f_int = np.exp(min(exp_arg, 700.0)) / denom

    # Log-normal PDF: F(x) = 1/(x * sigma_ln * sqrt(2pi)) * exp(-(ln(x)-mu_ln)^2/(2*sigma_ln^2))
    # This integrates to 1 and has mean = exp(mu_ln + sigma_ln^2/2) = x_inj * f_int
    safe_x = np.where(x_obs > 1e-30, x_obs, 1e-30)
    ln_x = np.log(safe_x)
    bump = np.exp(-((ln_x - mu_ln) ** 2) / (2.0 * sigma_ln_sq)) / (
        safe_x * sigma_ln * np.sqrt(2.0 * np.pi)
    )
    bump = np.where(x_obs > 1e-30, bump, 0.0)

    return bump, f_int


@_val.renamed_kwargs(x_obs="x")
def greens_function_photon(
    x: ArrayLike,
    x_inj: float,
    z_h: float,
    sigma_x: float = 0.0,
    number_conserving: bool = False,
    cosmo: CosmoLike | None = None,
) -> NDArray[np.float64]:
    """Green's function for monochromatic photon injection.

    Returns ``Δn(x)`` per unit ``ΔN/N`` injected at frequency
    ``x_inj`` and redshift ``z_h``.  Uses the universal ``J_μ(z)``
    visibility (same as heat injection) to blend between pure-μ-era and
    pure-y-era contributions:

    ``G_ph = J_μ G_μ + (1 − J_μ) G_y``.

    The surviving photon line enters with weight ``P_s (1 − J_μ)`` and is
    always present unless ``P_s = 0``.  Its shape: a log-normal broadened
    by the Compton ``y_γ`` when ``y_γ ≥ 1e-6`` (``sigma_x`` adds in
    quadrature); otherwise a Gaussian of width ``sigma_x``, falling back
    to a narrow ``0.005 x_inj`` Gaussian when ``sigma_x = 0``.

    When ``P_s = 0``, this does not reduce to
    ``α_ρ x_inj · greens_function(x, z_h)``, because the two combine the
    visibilities differently::

        G_ph / (α_ρ x_inj) = G_th − (1 − J_μ)(1 − J_bb*) G_bb/4
                             + (1 − J_μ − J_y) Y_SZ/4.

    At ``P_s = 0`` the photon form closes energy exactly, while ``G_th``
    does not (see ``dev/audit/greens_audit.md``, M-3 and M-5).

    Reference: Chluba (2015), arXiv:1506.06582.

    Parameters
    ----------
    x : float or array_like
        Observation frequency.
    x_inj : float
        Injection frequency (must be positive and finite).
    z_h : float
        Injection redshift.  Must lie outside the μ–y transition band
        ``(1e4, 3e5)``; otherwise a :class:`ValueError` is raised.
    sigma_x : float, optional
        Intrinsic Gaussian width of the surviving photon line (default 0;
        the line is still drawn with a width of ``0.005 x_inj``).  When
        ``y_γ ≥ 1e-6`` it widens the log-normal but leaves ``f_int``
        unchanged, so the result over-counts energy by
        ``P_s (1 − J_μ) [exp((σ_x/x_inj)²/2) − 1]`` of the injected
        ``α_ρ x_inj ΔN/N``: measured +0.50% (``σ_x = 0.5``) and +2.00%
        (``σ_x = 1``) at ``x_inj = 5``, ``z_h = 5e3``.
    number_conserving : bool, optional
        If True, drop the temperature-shift component so the result
        satisfies ``∫ x² G dx ≈ 0`` (CosmoTherm convention for stored
        Green's function entries).  Default False.
    cosmo : Mapping, optional
        Cosmological parameters feeding both the photon survival
        probability ``P_s`` (DC + BR optical-depth integral) and the
        Compton ``y_γ`` broadening of the surviving line.
        Defaults to :data:`~spectroxide.cosmology.DEFAULT_COSMO`.

    Returns
    -------
    ndarray of float64
        Spectral distortion ``Δn(x)`` per unit ``ΔN/N``.

    Raises
    ------
    ValueError
        If ``x``, ``x_inj``, or ``z_h`` are out of range, or if
        ``z_h`` falls inside the μ–y transition.
    """
    _val.validate_z_h(z_h)
    _val.validate_x_positive(x, label="x")
    _val.validate_x_inj(x_inj)
    _val.warn_z_h_regime(z_h)
    _val.warn_x_inj_regime(x_inj)
    _val.validate_photon_gf_regime(z_h)
    x = np.asarray(x, dtype=np.float64)

    _j_bb_star = j_bb_star(z_h)
    cosmo = _cosmo_mapping(cosmo)
    p_s = _photon_survival_probability_numerical(x_inj, z_h, cosmo)

    alpha_x = ALPHA_RHO * x_inj

    # Universal mu-y transition (same as heat injection Green's function (GF))
    _j_mu = j_mu(z_h)

    # --- mu-era contribution ---
    mu_factor = 1.0 - p_s * X_BALANCED / x_inj
    mu_part = (3.0 / KAPPA_C) * _j_bb_star * mu_factor * mu_shape(x)

    # Temperature shift (energy conservation residual)
    lam = 1.0 - mu_factor * _j_bb_star
    t_part = lam / 4.0 * temperature_shift_shape(x)

    # Deep mu-era short-circuit: when J_mu ≈ 1, the y-era contribution
    # is zero and computing it risks f_int overflow (exp((a+b)*yg) for
    # yg > ~500).  Return pure mu-era result directly.
    if _j_mu > 1.0 - 1e-12:
        if number_conserving:
            return alpha_x * mu_part
        else:
            return alpha_x * (mu_part + t_part)

    # --- y-era contribution ---
    # Compton y-parameter determines broadening of surviving bump
    yg = _y_compton(z_h, cosmo)

    # Broadened surviving photon bump (log-normal in x)
    bump_shape, f_int = _broadened_bump(x, x_inj, yg)

    # Smooth y-era: energy balance coefficient.
    # f_int = <x>/x_inj = mean energy ratio of the log-normal bump.
    # Surviving photon carries energy P_s * f_int * alpha_x.
    # Remaining (1 - P_s * f_int) goes into smooth y.
    coeff_y = 1.0 - p_s * f_int
    y_smooth = coeff_y * 0.25 * y_shape(x)

    # Combine mu and y using universal visibility J_mu(z).
    if number_conserving:
        smooth = alpha_x * (_j_mu * mu_part + (1.0 - _j_mu) * y_smooth)
    else:
        smooth = alpha_x * (_j_mu * (mu_part + t_part) + (1.0 - _j_mu) * y_smooth)

    # Surviving photon bump (broadened by Compton scattering).
    safe_x = np.where(x > 1e-30, x, 1e-30)
    if yg >= 1e-6:
        alpha = _alpha_cs(x_inj, yg)
        beta = _beta_cs(x_inj, yg)
        denom = 1.0 + x_inj * yg
        mu_ln = np.log(x_inj) + alpha * yg - np.log(denom)
        sigma_ln_sq = 2.0 * beta * yg
        if sigma_x > 0.0:
            sigma_ln_sq += (sigma_x / x_inj) ** 2
        sigma_ln = np.sqrt(max(sigma_ln_sq, 1e-30))
        ln_x = np.log(safe_x)
        bump_broad = np.exp(-((ln_x - mu_ln) ** 2) / (2.0 * sigma_ln_sq)) / (
            safe_x * sigma_ln * np.sqrt(2.0 * np.pi)
        )
        surviving = p_s * (1.0 - _j_mu) * G2_PLANCK / (safe_x**2) * bump_broad
    elif sigma_x > 0.0:
        norm_g = 1.0 / (sigma_x * np.sqrt(2.0 * np.pi))
        gauss = np.exp(-((x - x_inj) ** 2) / (2.0 * sigma_x**2)) * norm_g
        surviving = p_s * (1.0 - _j_mu) * G2_PLANCK / (safe_x**2) * gauss
    else:
        surviving = p_s * (1.0 - _j_mu) * G2_PLANCK / (safe_x**2) * bump_shape
    result = smooth + surviving

    return result


def _cosmo_mapping(cosmo):
    """Return ``cosmo`` as a complete mapping.

    Converts a ``Cosmology`` dataclass, fills keys missing from a partial
    mapping from :data:`DEFAULT_COSMO`, and returns ``DEFAULT_COSMO`` for
    *None*.
    """
    if cosmo is None:
        return DEFAULT_COSMO
    if hasattr(cosmo, "to_dict"):
        cosmo = cosmo.to_dict()
    _val.validate_cosmology(cosmo)
    return {**DEFAULT_COSMO, **cosmo}


def mu_from_photon_injection(
    x_inj: float,
    z_h: float,
    delta_n_over_n: float,
    cosmo: CosmoLike | None = None,
) -> float:
    """μ from monochromatic photon injection.

    .. math::

        \\mu = \\alpha_\\rho \\, x_{inj} \\, \\frac{3}{\\kappa_c}
               J_{bb}^*(z_h) J_\\mu(z_h)
               \\left(1 - P_s \\frac{x_0}{x_{inj}}\\right)
               \\frac{\\Delta N}{N}.

    Uses the universal ``J_μ(z)`` visibility function (same as heat
    injection).

    Sign behavior
    --------------
    The sign of μ depends on the regime:

    - ``x_inj > x₀`` and ``P_s ≈ 1``: ``μ > 0`` (energy-dominated).
    - ``x_inj < x₀`` and ``P_s ≈ 1``: ``μ < 0`` (number-dominated).
    - ``P_s ≈ 0`` (soft photons absorbed): ``μ > 0`` always (pure energy
      injection).

    Reference: Chluba (2015), arXiv:1506.06582.

    Parameters
    ----------
    x_inj : float
        Injection frequency (positive, finite).
    z_h : float
        Injection redshift (must lie outside the μ–y transition band).
    delta_n_over_n : float
        Fractional photon-number perturbation ``ΔN/N``.
    cosmo : Mapping or Cosmology, optional
        Cosmology for the survival probability ``P_s``; see
        :func:`photon_survival_probability`.  Default *None* uses the
        analytic ``P_s = exp(−x_c/x)``.  Pass a cosmology (for example
        :data:`~spectroxide.cosmology.DEFAULT_COSMO`) to match the μ
        coefficient of :func:`greens_function_photon` with the same
        ``cosmo``.  The two agree for ``z_h > 5e4`` either way; for
        ``z_h ≤ 5e4`` with ``cosmo=None`` the function warns, because the
        analytic ``P_s`` is then wrong by large factors.

    Returns
    -------
    float
        Dimensionless μ-parameter.

    See Also
    --------
    y_from_photon_injection : the y-parameter partner.
    """
    _val.validate_x_inj(x_inj)
    _val.validate_z_h(z_h)
    _val.warn_z_h_regime(z_h)
    _val.warn_x_inj_regime(x_inj)
    _val.validate_photon_gf_regime(z_h)
    if cosmo is None and z_h <= P_S_TAU_FF_Z_MAX:
        import warnings

        warnings.warn(
            f"mu_from_photon_injection: z_h={z_h:g} <= 5e4 with cosmo=None uses "
            "the analytic P_s = exp(-x_c/x), which differs by large factors "
            "from the P_s of greens_function_photon and y_from_photon_injection "
            "here. Pass cosmo=DEFAULT_COSMO (or your cosmology) to match them.",
            UserWarning,
            stacklevel=2,
        )
    _j_bb_star = float(j_bb_star(z_h))
    _j_mu = j_mu(z_h)
    p_s = float(photon_survival_probability(np.array([x_inj]), z_h, cosmo=cosmo)[0])

    mu_factor = 1.0 - p_s * X_BALANCED / x_inj

    return float(
        ALPHA_RHO
        * x_inj
        * (3.0 / KAPPA_C)
        * _j_bb_star
        * _j_mu
        * mu_factor
        * delta_n_over_n
    )


def y_from_photon_injection(
    x_inj: float,
    z_h: float,
    delta_n_over_n: float,
    cosmo: CosmoLike | None = None,
) -> float:
    """Compton y from monochromatic photon injection.

    The y partner of :func:`mu_from_photon_injection`: the coefficient of
    ``Y_SZ(x)`` in :func:`greens_function_photon`,

    .. math::

        y = \\frac{\\alpha_\\rho \\, x_{inj}}{4}
            \\left[1 - J_\\mu(z_h)\\right]
            \\left(1 - P_s f_{int}\\right)
            \\frac{\\Delta N}{N}.

    Here ``α_ρ x_inj ΔN/N`` is the injected ``Δρ/ρ``.  The factor
    ``1 − P_s f_int`` is the part of it that heats electrons: absorbed
    photons give up all their energy, and a surviving photon keeps the
    fraction ``f_int = ⟨x⟩/x_inj`` after Compton scattering with
    ``y_γ(z_h)``.  So ``y → Δρ/(4ρ)`` when the photon is absorbed
    (``P_s → 0``) in the y-era, and ``y → 0`` when it survives unscattered.
    The surviving line itself is not included.  ``y = 0`` in the deep
    μ-era (``J_μ ≈ 1``).

    Reference: Chluba (2015), arXiv:1506.06582, Sect. 4.

    Parameters
    ----------
    x_inj : float
        Injection frequency (positive, finite).
    z_h : float
        Injection redshift (must lie outside the μ–y transition band).
    delta_n_over_n : float
        Fractional photon-number perturbation ``ΔN/N``.
    cosmo : Mapping or Cosmology, optional
        Cosmology for ``y_γ`` and the cosmology-aware ``P_s``, as in
        :func:`greens_function_photon`; missing keys come from
        :data:`~spectroxide.cosmology.DEFAULT_COSMO`.  Default *None* uses
        ``DEFAULT_COSMO``, so the result equals the y coefficient of
        :func:`greens_function_photon` with the same ``cosmo``.  (The
        analytic ``P_s`` of :func:`mu_from_photon_injection`'s default is
        wrong by large factors at ``z_h ≤ 5e4``, where y lives.)

    Returns
    -------
    float
        Dimensionless Compton y-parameter.
    """
    _val.validate_x_inj(x_inj)
    _val.validate_z_h(z_h)
    _val.warn_z_h_regime(z_h)
    _val.warn_x_inj_regime(x_inj)
    _val.validate_photon_gf_regime(z_h)
    _j_mu = float(j_mu(z_h))
    # Same deep-μ short-circuit as greens_function_photon.
    if _j_mu > 1.0 - 1e-12:
        return 0.0
    cosmo = _cosmo_mapping(cosmo)
    p_s = float(photon_survival_probability(np.array([x_inj]), z_h, cosmo=cosmo)[0])
    yg = _y_compton(z_h, cosmo)
    _, f_int = _broadened_bump(np.array([x_inj]), x_inj, yg)
    return float(
        ALPHA_RHO * x_inj * 0.25 * (1.0 - _j_mu) * (1.0 - p_s * f_int) * delta_n_over_n
    )


@_val.renamed_kwargs(x_grid="x")
def distortion_from_photon_injection(
    x: ArrayLike,
    x_inj: float,
    dn_dz: Callable[[ArrayLike], ArrayLike],
    z_min: float,
    z_max: float,
    n_z: int = 5000,
    sigma_x: float = 0.0,
    cosmo: CosmoLike | None = None,
) -> NDArray[np.float64]:
    """Spectral distortion from a photon injection history.

    .. math::

        \\Delta n(x) = \\int_{z_{min}}^{z_{max}}
                       G_{ph}(x, x_{inj}, z') \\,
                       \\frac{d(\\Delta N / N)}{dz'} \\, dz'.

    Active redshifts must avoid the μ–y transition band
    ``(1e4, 3e5)``; if the integration range overlaps it, points landing
    in the band raise :class:`ValueError` (preceded by a warning).

    Parameters
    ----------
    x : array_like
        Observation frequency grid.
    x_inj : float
        Injection frequency (positive, finite).
    dn_dz : callable
        Source rate ``d(ΔN/N)/dz`` (positive for injection).
    z_min : float
        Minimum integration redshift.
    z_max : float
        Maximum integration redshift.
    n_z : int, optional
        Number of redshift integration points (default 5000).
    sigma_x : float, optional
        Gaussian width for the surviving photon δ-function (default 0).
    cosmo : Mapping, optional
        Cosmological parameters for Compton y-parameter broadening.
        Defaults to :data:`~spectroxide.cosmology.DEFAULT_COSMO`.

    Returns
    -------
    ndarray of float64
        Distortion ``Δn(x)`` evaluated on ``x``.

    Raises
    ------
    ValueError
        If any active source redshift falls in the μ–y transition band.
    """
    _val.validate_x_positive(x)
    _val.validate_x_inj(x_inj)
    _val.validate_z_range(z_min, z_max, n_z)
    _val.warn_x_inj_regime(x_inj)
    _val.warn_z_max_regime(z_max)
    transition_overlap = (
        z_min < _val.PHOTON_GF_MU_ERA_Z_MIN and z_max > _val.PHOTON_GF_Y_ERA_Z_MAX
    )
    if transition_overlap:
        import warnings as _warnings

        _warnings.warn(
            f"distortion_from_photon_injection integrates over z in "
            f"[{z_min:.2e}, {z_max:.2e}], which overlaps the mu-y "
            f"transition ({_val.PHOTON_GF_Y_ERA_Z_MAX:.0e} < z "
            f"< {_val.PHOTON_GF_MU_ERA_Z_MIN:.0e}). Source samples that "
            "land in the band are skipped (the photon GF is undefined "
            "there); the integral underestimates mu/y by an amount that "
            "depends on the source weight in the band. Use the PDE solver "
            "(run_photon_sweep) for transition-era injection.",
            stacklevel=3,
        )
    x = np.asarray(x, dtype=np.float64)
    ln_min = np.log(1.0 + z_min)
    ln_max = np.log(1.0 + z_max)
    dln = (ln_max - ln_min) / max(n_z - 1, 1)

    # Build redshift array and precompute source values
    j_arr = np.arange(n_z)
    ln_1pz = ln_min + j_arr * dln
    z_arr = np.exp(ln_1pz) - 1.0
    dz_dln = 1.0 + z_arr

    source_arr = _call_vectorized(dn_dz, z_arr) * dz_dln  # shape (n_z,)

    w = np.full(n_z, dln)
    w[0] = 0.5 * dln
    w[-1] = 0.5 * dln

    sw = source_arr * w  # shape (n_z,)

    # Only iterate over non-negligible source points outside the
    # mu-y transition band (where the photon GF is undefined).
    active = np.abs(sw) >= 1e-50
    if transition_overlap:
        # Match Rust's skip behavior: drop samples in (Y_ERA_Z_MAX, MU_ERA_Z_MIN)
        # rather than panic. The slack matches the Rust TOL constant.
        tol = 1.0e-6
        lo = _val.PHOTON_GF_Y_ERA_Z_MAX * (1.0 + tol)
        hi = _val.PHOTON_GF_MU_ERA_Z_MIN * (1.0 - tol)
        active &= (z_arr <= lo) | (z_arr >= hi)
    active_idx = np.where(active)[0]

    delta_n = np.zeros_like(x)
    for j in active_idx:
        delta_n += (
            greens_function_photon(x, x_inj, float(z_arr[j]), sigma_x, cosmo=cosmo)
            * sw[j]
        )

    return delta_n


# ---------------------------------------------------------------------------
# Unit conversion: Δn(x) → ΔI(ν) in physical units
# ---------------------------------------------------------------------------


def strip_gbb(x: ArrayLike, delta_n: ArrayLike) -> Tuple[NDArray[np.float64], float]:
    """Remove the unobservable temperature-shift component of a spectrum.

    The Far Infrared Absolute Spectrophotometer measures the CMB spectrum
    with the absolute temperature as a free parameter, so a uniform shift
    ΔT/T is unobservable.  CosmoTherm therefore defines the *distortion*
    as the number-conserving part of
    ``Δn`` (Chluba & Sunyaev 2012, arXiv:1109.6552): the part satisfying
    ``∫ x² Δn dx = 0``.  Any nonzero photon-number perturbation is
    absorbed into ``α · G_bb(x)``.

    This projection is orthogonal to ``μ`` and ``y`` because both
    ``M(x)`` and ``Y_SZ(x)`` conserve photon number
    (``∫ x² M dx ≈ 0``, ``∫ x² Y dx ≈ 0``), so there is no cross-talk.

    Parameters
    ----------
    x : array_like
        Dimensionless frequency grid.
    delta_n : array_like
        Spectral distortion in occupation-number space.

    Returns
    -------
    delta_n_stripped : ndarray of float64
        Number-conserving distortion (``∫ x² Δn_stripped dx ≈ 0``).
    alpha : float
        Temperature-shift coefficient ``ΔT/T``.
    """
    x = np.asarray(x, dtype=np.float64)
    delta_n = np.asarray(delta_n, dtype=np.float64)
    gbb = g_bb(x)

    alpha = float(_trapz(x**2 * delta_n, x) / _trapz(x**2 * gbb, x))

    return delta_n - alpha * gbb, alpha


DEFAULT_DECOMP_X_MIN = 0.5
DEFAULT_DECOMP_X_MAX = 18.0

# Fit band of method="gf_fit"; the same band as the paper's Table 1
# visibility fit (dev/scripts/fit_visibility_conservation.py).
GF_FIT_X_MIN = 0.5
GF_FIT_X_MAX = 20.0


@_val.renamed_kwargs(x_grid="x")
def decompose_distortion(
    x: ArrayLike,
    delta_n: ArrayLike,
    z_h: float | None = None,
    method: str = "bf",
    x_range: Tuple[float, float] | None = None,
) -> dict:
    """Decompose ``Δn(x)`` into ``(μ, y, ΔT/T)`` components.

    Default method: Bianchini & Fabbian (2022) nonlinear Bose–Einstein fit
    (``method="bf"``), matching the Rust ``spectroxide::distortion::decompose_distortion``.

    **``method="bf"`` (default):** Nonlinear least-squares fit of
    Δn(x) = [n_pl(x/(1+δ)) − n_pl(x)] + [n_BE(x+μ) − n_pl(x)] + y·Y_SZ(x)
    on x ∈ [0.5, 18], minimizing the intensity residual
    ``∫ [x³ (Δn − model)]² dx`` (ADR 0006), bootstrapped from a linear
    Gram-Schmidt initial guess and refined by Levenberg-Marquardt. See
    :func:`_decompose_nonlinear_be`.

    **``method="gs"``:** Linear Gram-Schmidt orthogonalization of
    (Y_SZ, M, G) over the same band and with the same intensity weight
    (Chluba & Jeong 2014, Appendix A).
    Agrees with ``bf`` on μ and y to numerical precision at realistic
    injection amplitudes; see :func:`_decompose_gram_schmidt`.

    **``method="nc"``:** the temperature shift is removed by photon-number
    conservation (:func:`strip_gbb`, over the whole grid), then μ and y are
    fitted linearly to the stripped spectrum with the same band and
    intensity weight. This tracks the Chluba (2013) visibility functions
    (J_μ to 0.031, J_y to 0.006 on the 118 Table 1 bursts), where ``bf``
    and ``gs`` give J_μ up to 1.25 in the μ–y transition. Use it when
    comparing with visibility-function targets. See
    :func:`_decompose_number_conserving`.

    **``method="gf_fit"`` (requires z_h):** Green's-function spectral fit
    for visibility-function calibration, the estimator that defines J_μ
    and J_y in Chluba (2013, arXiv:1304.6120, Eqs. 5–6). Per unit Δρ/ρ
    (measured from ``delta_n``) the ansatz is
    ``(3/κ_c) P M + (J_y/4) Y_SZ + ((1 − P − J_y)/4) G_bb`` with
    ``P = J_μ J_bb*``. The number-conserving strip (:func:`strip_gbb`,
    over the whole grid) removes the G_bb term, leaving two linear
    amplitudes, P and J_y, both fitted in closed form by minimizing
    ``∫ [x³ (model − Δn_nc)]² dx`` (trapezoid rule) on ``x_range``,
    default ``[0.5, 20]``. Range, weight and quadrature match the paper's
    Table 1 fit (``dev/scripts/fit_visibility_conservation.py``). The
    ansatz depends on J_μ and J_bb* only through P, so ``j_mu_fit`` is the
    analytic J_μ(z_h) and ``j_bb_star_fit`` is P divided by it; ``j_y`` is
    the fitted J_y.

    **Caution:** use this method for visibility calibration, not for
    production μ/y extraction.

    Parameters
    ----------
    x : array_like
        Dimensionless frequency grid ``x = h ν / (k_B T_z)``.
    delta_n : array_like
        Spectral distortion ``Δn(x)`` (same length as ``x``).
    z_h : float, optional
        Injection redshift.  Required when ``method="gf_fit"``; ignored
        (with a warning) for the other methods.
    method : {"bf", "gs", "nc", "gf_fit"}, optional
        Decomposition method (default ``"bf"``).
    x_range : (float, float), optional
        Fit band for ``method="gf_fit"`` (default
        ``(GF_FIT_X_MIN, GF_FIT_X_MAX)``). The other methods raise if it
        is given.

    Returns
    -------
    dict
        Keys ``mu`` (float), ``y`` (float), ``dT`` (float, ΔT/T),
        ``drho`` (float, Δρ/ρ), ``dn_over_n`` (float, ΔN/N), and
        ``residual`` (ndarray, ``Δn − model`` on ``x``).  When
        ``gf_fit`` is used, also includes ``j_mu_fit``, ``j_bb_star_fit``,
        ``j_y``, ``fit_success``, ``fit_residual``.

    Raises
    ------
    ValueError
        If ``method`` is unknown, if ``method="gf_fit"`` is selected
        without supplying ``z_h``, or if ``x_range`` is given for another
        method.
    """
    if x_range is not None and method != "gf_fit":
        raise ValueError(
            "decompose_distortion: x_range is only used by method='gf_fit'."
        )
    if method == "bf":
        if z_h is not None:
            import warnings

            warnings.warn(
                "decompose_distortion: z_h is ignored for method='bf'. "
                "Pass method='gf_fit' to use the Green's-function spectral fit.",
                stacklevel=2,
            )
        return _decompose_nonlinear_be(x, delta_n)

    if method in ("gs", "nc"):
        if z_h is not None:
            import warnings

            warnings.warn(
                f"decompose_distortion: z_h is ignored for method={method!r}. "
                "Pass method='gf_fit' to use the Green's-function spectral fit.",
                stacklevel=2,
            )
        if method == "gs":
            return _decompose_gram_schmidt(x, delta_n)
        return _decompose_number_conserving(x, delta_n)

    if method != "gf_fit":
        raise ValueError(
            f"decompose_distortion: unknown method={method!r}; "
            "expected 'bf', 'gs', 'nc', or 'gf_fit'."
        )
    if z_h is None:
        raise ValueError("decompose_distortion: method='gf_fit' requires z_h.")

    _val.validate_x_positive(x)
    _val.validate_array_lengths(x, delta_n)
    _val.warn_x_grid_narrow(x)
    x = np.asarray(x, dtype=np.float64)
    delta_n = np.asarray(delta_n, dtype=np.float64)
    mu_to_energy = 3.0 / KAPPA_C  # ≈ 1.401

    # Model-independent integrals
    dx = np.diff(x)
    x_mid = 0.5 * (x[:-1] + x[1:])
    dn_mid = 0.5 * (delta_n[:-1] + delta_n[1:])
    drho = np.sum(x_mid**3 * dn_mid * dx)
    dn_n = np.sum(x_mid**2 * dn_mid * dx)
    drho_over_rho = drho / G3_PLANCK
    dn_over_n = dn_n / G2_PLANCK

    # NC-strip the spectrum over the whole grid (number conservation is a
    # property of the whole spectrum), then fit on the band.
    if not np.all(np.isfinite(delta_n)):
        raise ValueError(
            "decompose_distortion: delta_n must be finite for method='gf_fit'."
        )
    dn_nc, _alpha = strip_gbb(x, delta_n)

    # The ansatz per unit Δρ/ρ is
    #   (3/κ_c) P M + (J_y/4) Y_SZ + ((1 − P − J_y)/4) G_bb,  P ≡ J_μ J_bb*.
    # strip_gbb is linear and removes G_bb exactly, so the stripped model is
    # P a + J_y b with fixed vectors a, b: a two-parameter linear
    # least-squares problem with a closed-form solution.
    x_lo, x_hi = (GF_FIT_X_MIN, GF_FIT_X_MAX) if x_range is None else x_range
    band = np.flatnonzero((x >= x_lo) & (x <= x_hi))
    if band.size < 3:
        raise ValueError(
            f"decompose_distortion: fewer than 3 grid points in [{x_lo}, {x_hi}]."
        )
    m_nc, _ = strip_gbb(x, mu_shape(x))
    y_nc, _ = strip_gbb(x, y_shape(x))
    a_vec = drho_over_rho * mu_to_energy * m_nc[band]
    b_vec = drho_over_rho * 0.25 * y_nc[band]
    # ∫ [x³ r]² dx by the trapezoid rule on the band: weights x⁶ Δx_trap.
    xb = x[band]
    dxb = np.diff(xb)
    w_trap = np.zeros(band.size)
    w_trap[:-1] += 0.5 * dxb
    w_trap[1:] += 0.5 * dxb
    w = xb**6 * w_trap
    gram = np.array(
        [
            [np.sum(w * a_vec * a_vec), np.sum(w * a_vec * b_vec)],
            [np.sum(w * a_vec * b_vec), np.sum(w * b_vec * b_vec)],
        ]
    )
    det = gram[0, 0] * gram[1, 1] - gram[0, 1] ** 2
    if not (np.isfinite(det) and det > 0.0):
        raise ValueError(
            "decompose_distortion: method='gf_fit' needs a spectrum with nonzero "
            f"energy (got Δρ/ρ = {drho_over_rho:.3e})."
        )
    rhs = np.array([np.sum(w * a_vec * dn_nc[band]), np.sum(w * b_vec * dn_nc[band])])
    p_fit, j_y_val = (float(v) for v in np.linalg.solve(gram, rhs))
    fit_residual = float(
        np.sum(w * (a_vec * p_fit + b_vec * j_y_val - dn_nc[band]) ** 2)
    )
    fit_success = bool(np.isfinite(p_fit) and np.isfinite(j_y_val))

    # Only the product P is constrained. Hold J_μ at its Chluba (2013)
    # value and assign the fitted product to J_bb*.
    j_mu_fit = float(j_mu(z_h))
    j_bb_star_fit = p_fit / j_mu_fit

    # Extract μ, y from the fitted visibility functions
    mu = mu_to_energy * p_fit * drho_over_rho
    y_val = 0.25 * j_y_val * drho_over_rho

    # ΔT/T from energy conservation
    dT = drho_over_rho / 4.0 - mu / (4.0 * mu_to_energy) - y_val

    # Residual
    residual = delta_n - mu * mu_shape(x) - y_val * y_shape(x) - dT * g_bb(x)

    return {
        "mu": mu,
        "y": y_val,
        "dT": dT,
        "drho": drho_over_rho,
        "dn_over_n": dn_over_n,
        "residual": residual,
        "j_mu_fit": j_mu_fit,
        "j_bb_star_fit": j_bb_star_fit,
        "j_y": j_y_val,
        "fit_success": fit_success,
        "fit_residual": fit_residual,
    }


def _band_intensity_weights(x_grid, x_min, x_max):
    """Band mask and intensity weights ``x⁶ dx`` (ADR 0006).

    A weighted sum of squared Δn residuals with these weights is the squared
    intensity residual ``∫ [x³ (Δn − model)]² dx``.
    """
    mask, dx = _band_trap_weights(x_grid, x_min, x_max)
    return mask, dx * x_grid**6


def _band_trap_weights(x_grid, x_min, x_max):
    """Indices and trapezoidal weights for points inside [x_min, x_max]."""
    n = len(x_grid)
    mask = (x_grid >= x_min) & (x_grid <= x_max)
    dx = np.zeros(n)
    dx[0] = x_grid[1] - x_grid[0]
    dx[-1] = x_grid[-1] - x_grid[-2]
    dx[1:-1] = 0.5 * (x_grid[2:] - x_grid[:-2])
    return mask, dx


def _energy_integrals(x_grid, delta_n):
    """Trapezoidal ∫x³Δn/G3 and ∫x²Δn/G2."""
    dx = np.diff(x_grid)
    x_mid = 0.5 * (x_grid[:-1] + x_grid[1:])
    dn_mid = 0.5 * (delta_n[:-1] + delta_n[1:])
    drho = np.sum(x_mid**3 * dn_mid * dx)
    dn_n = np.sum(x_mid**2 * dn_mid * dx)
    return drho / G3_PLANCK, dn_n / G2_PLANCK


def _decompose_gram_schmidt(
    x_grid: ArrayLike,
    delta_n: ArrayLike,
    x_min: float = DEFAULT_DECOMP_X_MIN,
    x_max: float = DEFAULT_DECOMP_X_MAX,
) -> dict:
    """CJ2014 Appendix-A Gram–Schmidt decomposition over ``[x_min, x_max]``.

    Constructs an orthonormal basis ``(e_y, e_μ, e_T)`` from
    ``(Y_SZ, M, G_bb)`` using Gram–Schmidt under the intensity inner
    product
    ``⟨a, b⟩ = ∫_{x_min}^{x_max} x⁶ a(x) b(x) dx`` (trapezoid rule).
    CJ2014 build their vectors from intensities ``ΔI ∝ x³Δn`` in channels
    uniform in ν and sum them without weight; this integral is the
    continuum limit of that sum (ADR 0006),
    then projects ``Δn`` and back-substitutes for ``(μ, y, ΔT/T)`` in the
    linear basis ``Δn ≈ μ M + y · Y_SZ + δT · G``.

    Reference: Chluba & Jeong (2014), arXiv:1306.5751, Appendix A.

    Parameters
    ----------
    x_grid : array_like
        Dimensionless frequency grid.
    delta_n : array_like
        Spectral distortion ``Δn(x)``.
    x_min : float, optional
        Lower band edge (default ``0.5``, Primordial Inflation Explorer-like).
    x_max : float, optional
        Upper band edge (default ``18.0``).

    Returns
    -------
    dict
        Keys ``mu``, ``y``, ``dT``, ``drho``, ``dn_over_n`` (all floats),
        and ``residual`` (ndarray).
    """
    _val.validate_x_positive(x_grid)
    _val.validate_array_lengths(x_grid, delta_n)
    _val.warn_x_grid_narrow(x_grid)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    delta_n = np.asarray(delta_n, dtype=np.float64)

    drho_over_rho, dn_over_n = _energy_integrals(x_grid, delta_n)

    mask, dx = _band_intensity_weights(x_grid, x_min, x_max)
    xb = x_grid[mask]
    wb = dx[mask]
    dn_b = delta_n[mask]
    if len(xb) < 3:
        import warnings as _warnings

        _warnings.warn(
            f"_decompose_gram_schmidt: only {len(xb)} grid point(s) fall in the "
            f"decomposition band [{x_min}, {x_max}]; returning mu=y=dT=0. "
            "Widen the band or supply a denser x grid to extract physical mu/y.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "mu": 0.0,
            "y": 0.0,
            "dT": 0.0,
            "drho": drho_over_rho,
            "dn_over_n": dn_over_n,
            "residual": delta_n.copy(),
        }

    m_v = mu_shape(xb)
    y_v = y_shape(xb)
    g_v = g_bb(xb)

    def ip(a, b):
        return float(np.sum(a * b * wb))

    y_norm = np.sqrt(ip(y_v, y_v))
    e_y = y_v / y_norm
    m_y = ip(m_v, e_y)
    m_perp = m_v - m_y * e_y
    m_perp_norm = np.sqrt(ip(m_perp, m_perp))
    e_mu = m_perp / m_perp_norm
    g_y = ip(g_v, e_y)
    g_mu = ip(g_v, e_mu)
    g_perp = g_v - g_y * e_y - g_mu * e_mu
    g_perp_norm = np.sqrt(ip(g_perp, g_perp))
    e_t = g_perp / g_perp_norm

    a_y = ip(dn_b, e_y)
    a_mu = ip(dn_b, e_mu)
    a_t = ip(dn_b, e_t)

    dT = a_t / g_perp_norm
    mu = (a_mu - dT * g_mu) / m_perp_norm
    y = (a_y - dT * g_y - mu * m_y) / y_norm

    residual = delta_n - mu * mu_shape(x_grid) - y * y_shape(x_grid) - dT * g_bb(x_grid)
    return {
        "mu": float(mu),
        "y": float(y),
        "dT": float(dT),
        "drho": float(drho_over_rho),
        "dn_over_n": float(dn_over_n),
        "residual": residual,
    }


def _decompose_number_conserving(
    x_grid: ArrayLike,
    delta_n: ArrayLike,
    x_min: float = DEFAULT_DECOMP_X_MIN,
    x_max: float = DEFAULT_DECOMP_X_MAX,
) -> dict:
    """μ and y with the temperature shift removed by photon-number conservation.

    Mirrors the Rust ``spectroxide::distortion::decompose_number_conserving``.
    ``ΔT/T`` is fixed, not fitted: :func:`strip_gbb` removes the ``G_bb``
    component that carries the photon-number change, from ``Δn`` and from
    the shapes ``−G_bb/x`` (linearized Bose–Einstein μ) and ``Y_SZ``. μ and y
    then come from a linear least-squares fit to the stripped spectrum
    minimizing ``∫ [x³ (Δn − model)]² dx`` on ``[x_min, x_max]`` (ADR 0006).
    The photon-number integrals use the trapezoid rule of :func:`strip_gbb`;
    the Rust version uses the midpoint-product rule, and on the 4000-point
    production grid the two differ by at most 5×10⁻⁵ Δρ/ρ in μ.

    Returns
    -------
    dict
        Keys ``mu``, ``y``, ``dT`` (the number-conserving shift), ``drho``,
        ``dn_over_n`` (all floats), and ``residual`` (ndarray).
    """
    _val.validate_x_positive(x_grid)
    _val.validate_array_lengths(x_grid, delta_n)
    _val.warn_x_grid_narrow(x_grid)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    delta_n = np.asarray(delta_n, dtype=np.float64)
    drho_over_rho, dn_over_n = _energy_integrals(x_grid, delta_n)

    dn_s, dT = strip_gbb(x_grid, delta_n)
    m_s, _ = strip_gbb(x_grid, -g_bb(x_grid) / x_grid)
    y_s, _ = strip_gbb(x_grid, y_shape(x_grid))
    mask, w = _band_intensity_weights(x_grid, x_min, x_max)
    if mask.sum() < 3:
        import warnings as _warnings

        _warnings.warn(
            f"_decompose_number_conserving: only {int(mask.sum())} grid point(s) fall in "
            f"the decomposition band [{x_min}, {x_max}]; returning mu=y=0. "
            "Widen the band or supply a denser x grid to extract physical mu/y.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "mu": 0.0,
            "y": 0.0,
            "dT": dT,
            "drho": drho_over_rho,
            "dn_over_n": dn_over_n,
            "residual": delta_n.copy(),
        }
    a = np.stack([m_s[mask], y_s[mask]], axis=1) * np.sqrt(w[mask])[:, None]
    b = dn_s[mask] * np.sqrt(w[mask])
    mu, y_val = np.linalg.lstsq(a, b, rcond=None)[0]
    residual = dn_s - mu * m_s - y_val * y_s
    return {
        "mu": float(mu),
        "y": float(y_val),
        "dT": float(dT),
        "drho": float(drho_over_rho),
        "dn_over_n": float(dn_over_n),
        "residual": residual,
    }


def _decompose_nonlinear_be(
    x_grid: ArrayLike,
    delta_n: ArrayLike,
    x_min: float = DEFAULT_DECOMP_X_MIN,
    x_max: float = DEFAULT_DECOMP_X_MAX,
    max_iter: int = 100,
    tol: float = 1.0e-12,
) -> dict:
    """Bianchini & Fabbian (2022) nonlinear Bose–Einstein fit.

    Fits the model

    ``Δn(x) = [n_pl(x/(1+δ)) − n_pl(x)] + [n_BE(x+μ) − n_pl(x)] + y · Y_SZ(x)``

    by Levenberg–Marquardt over the band ``[x_min, x_max]``, minimizing
    the intensity residual ``∫ [x³ (Δn − model)]² dx`` (ADR 0006),
    bootstrapped from :func:`_decompose_gram_schmidt` (converted to the
    B&F parameterization using ``δ_BF = δ_GS + μ/β_μ``).  The LM iteration
    refines the ``O(μ²)`` nonlinear correction; for realistic injection
    amplitudes ``|μ| ≲ 10⁻³`` the answer differs from Gram–Schmidt on
    ``μ`` and ``y`` at the numerical-noise level.

    Reference: Bianchini & Fabbian (2022), arXiv:2206.02762, Eqs. (1)–(4).

    Parameters
    ----------
    x_grid : array_like
        Dimensionless frequency grid.
    delta_n : array_like
        Spectral distortion ``Δn(x)``.
    x_min : float, optional
        Lower band edge (default ``0.5``).
    x_max : float, optional
        Upper band edge (default ``18.0``).
    max_iter : int, optional
        Maximum number of LM iterations (default 100).
    tol : float, optional
        Convergence tolerance on the relative parameter step
        (default ``1e-12``).

    Returns
    -------
    dict
        Keys ``mu``, ``y``, ``dT``, ``drho``, ``dn_over_n`` (all floats),
        and ``residual`` (ndarray).
    """
    _val.validate_x_positive(x_grid)
    _val.validate_array_lengths(x_grid, delta_n)
    _val.warn_x_grid_narrow(x_grid)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    delta_n = np.asarray(delta_n, dtype=np.float64)

    drho_over_rho, dn_over_n = _energy_integrals(x_grid, delta_n)

    mask, dx = _band_intensity_weights(x_grid, x_min, x_max)
    xb = x_grid[mask]
    wb = dx[mask]
    dn_b = delta_n[mask]
    if len(xb) < 3:
        import warnings as _warnings

        _warnings.warn(
            f"_decompose_nonlinear_be: only {len(xb)} grid point(s) fall in the "
            f"decomposition band [{x_min}, {x_max}]; returning mu=y=dT=0. "
            "Widen the band or supply a denser x grid to extract physical mu/y.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "mu": 0.0,
            "y": 0.0,
            "dT": 0.0,
            "drho": drho_over_rho,
            "dn_over_n": dn_over_n,
            "residual": delta_n.copy(),
        }

    def _planck_safe(x):
        x = np.asarray(x, dtype=np.float64)
        small = np.abs(x) < 1e-6
        big = x > 500.0
        mid = ~(small | big)
        out = np.zeros_like(x)
        out[small] = 1.0 / x[small] - 0.5 + x[small] / 12.0
        out[big] = np.exp(-x[big])
        out[mid] = 1.0 / np.expm1(x[mid])
        return out

    def _g_bb_safe(x):
        x = np.asarray(x, dtype=np.float64)
        small = np.abs(x) < 1e-6
        big = x > 100.0
        mid = ~(small | big)
        out = np.zeros_like(x)
        out[small] = 1.0 / x[small] - x[small] / 12.0
        out[big] = x[big] * np.exp(-x[big])
        em = np.expm1(x[mid])
        out[mid] = x[mid] * (1.0 + em) / (em * em)
        return out

    def model_at(xi, mu, delta, y_par):
        # B&F 2022: μ inside the exponential, δ as linear Taylor coefficient.
        return (
            (_planck_safe(xi + mu) - _planck_safe(xi))
            + delta * g_bb(xi)
            + y_par * y_shape(xi)
        )

    def chi2_at(mu, delta, y_par):
        r = dn_b - model_at(xb, mu, delta, y_par)
        return float(np.sum(r * r * wb))

    # Bootstrap from GS (translated to BF parameterization).
    gs = _decompose_gram_schmidt(x_grid, delta_n, x_min, x_max)
    mu = gs["mu"]
    delta = gs["dT"] + gs["mu"] / BETA_MU
    y_par = gs["y"]

    lam = 1e-6
    prev_chi2 = chi2_at(mu, delta, y_par)

    for _ in range(max_iter):
        xpm = xb + mu
        d_mu = np.where(
            np.abs(xpm) < 1e-6,
            -1.0 / (xpm * xpm),
            -_g_bb_safe(xpm) / xpm,
        )
        d_delta = _g_bb_safe(xb)
        d_y = y_shape(xb)
        jac = np.stack([d_mu, d_delta, d_y], axis=1)
        r = dn_b - model_at(xb, mu, delta, y_par)
        w_col = wb[:, None]
        ata = jac.T @ (jac * w_col)
        atr = jac.T @ (r * wb)

        accepted = False
        step = np.zeros(3)
        for _ls in range(20):
            diag = np.diag(np.diag(ata))
            a_damped = ata + lam * diag + np.eye(3) * 1e-40
            try:
                step = np.linalg.solve(a_damped, atr)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            mu_new = mu + step[0]
            delta_new = delta + step[1]
            y_new = y_par + step[2]
            if delta_new <= -0.5:
                lam *= 10.0
                continue
            chi2_new = chi2_at(mu_new, delta_new, y_new)
            if chi2_new < prev_chi2:
                mu, delta, y_par = mu_new, delta_new, y_new
                prev_chi2 = chi2_new
                lam = max(lam * 0.5, 1e-10)
                accepted = True
                break
            lam *= 2.0
        if not accepted:
            break
        scale = max(abs(mu), abs(delta), abs(y_par), 1.0)
        if np.max(np.abs(step)) < tol * scale:
            break

    residual = delta_n - np.array([model_at(xi, mu, delta, y_par) for xi in x_grid])
    return {
        "mu": float(mu),
        "y": float(y_par),
        "dT": float(delta),
        "drho": float(drho_over_rho),
        "dn_over_n": float(dn_over_n),
        "residual": residual,
    }


@_val.renamed_kwargs(dn="delta_n")
def delta_n_to_delta_I(
    x: ArrayLike, delta_n: ArrayLike, t_cmb: float = 2.726
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert ``Δn(x)`` to ``(ν [GHz], ΔI [Jy/sr])``.

    Uses ``ΔI = (2 h ν³ / c²) Δn`` with ``ν = x k_B T₀ / h``.

    Parameters
    ----------
    x : array_like
        Dimensionless frequency ``x = h ν / (k_B T_z)``.
    delta_n : array_like
        Spectral distortion ``Δn(x)``.
    t_cmb : float, optional
        Cosmic microwave background temperature today, in **K**.  Default
        2.726.

    Returns
    -------
    nu_ghz : ndarray of float64
        Frequency in **GHz**.
    di_jy : ndarray of float64
        Intensity distortion in **Jy/sr** (= 10⁻²⁶ W m⁻² Hz⁻¹ sr⁻¹).
    """
    x = np.asarray(x, dtype=float)
    delta_n = np.asarray(delta_n, dtype=float)

    nu_hz = x * _K_BOLTZMANN * t_cmb / _H_PLANCK
    nu_ghz = nu_hz / 1e9

    # ΔI = (2hν³/c²) × Δn, converted to Jy/sr
    di_si = 2.0 * _H_PLANCK * nu_hz**3 / _C_LIGHT**2 * delta_n
    di_jy = di_si / 1e-26

    return nu_ghz, di_jy
