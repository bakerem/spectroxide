"""
Green's function for the cosmological thermalization problem.

Provides a fast, approximate method for computing spectral distortions
from arbitrary energy release histories. The distortion from a delta-function
energy injection at redshift ``z_h`` is decomposed into mu, y, and temperature
shift components using visibility/branching functions.

Ported from ``src/greens.rs`` and ``src/spectrum.rs``.

Conventions
-----------
- Frequency variable: x = h ν / (k_B T_z), dimensionless.
- Redshift z is dimensionless; ``z_h`` denotes the *injection* redshift.
- All cosmology routines accept either ``DEFAULT_COSMO`` (Chluba 2013),
  ``PLANCK2015_COSMO``, or ``PLANCK2018_COSMO`` (re-exported from this
  module) — or any user dict with the same keys.

References
----------
- Chluba (2013), MNRAS 436, 2232 [arXiv:1304.6120].
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

#: Type alias for a scalar or NumPy array of float64 values.
FloatOrArray = Union[float, NDArray[np.float64]]


def _call_vectorized(
    func: Callable[..., ArrayLike], z_arr: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Call a user-provided callable with an array, falling back to scalar loop.

    Tries ``func(z_arr)`` first. Only the specific broadcasting failure modes
    ("cannot broadcast", "only integer scalar arrays can be converted", etc.)
    and shape mismatches trigger the scalar-loop fallback. Other
    TypeError/ValueError exceptions are assumed to be genuine bugs in
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
    # clear error, which is what we want.
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

    Crosses zero at ``x ≈ 3.83`` (the SZ null in the CMB intensity spectrum).

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
    ``z_μ`` from equating the DC+BR photon production rate to the Hubble
    rate (Chluba & Sunyaev 2012), and 5/2 from the DC opacity scaling
    (Danese & de Zotti 1982; Hu & Silk 1993). These are *not* fit parameters.

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
    to the PDE Green's function in the μ–y transition era.  Approaches 1
    for z ≪ 6 × 10⁴ (pure y-era) and 0 for z ≫ 6 × 10⁴.  The transition
    scale z ~ 6 × 10⁴ is physically motivated by y_γ(z) ~ 1, but the
    precise value and exponent are fit parameters.

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

    Accuracy (vs. PDE)
    ------------------
    - Deep μ-era (z_h > 2 × 10⁵): spectral shape accurate to <5%.
    - y-era (z_h < 10⁴): spectral shape accurate to <1%.
    - Transition era (z_h ~ 3 × 10⁴–10⁵): ~8–13% shape error.

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
    - Chluba (2013), MNRAS 436, 2232 [arXiv:1304.6120].
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


def distortion_from_heating(
    x_grid: ArrayLike,
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
    x_grid : array_like
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
        Distortion ``Δn(x)`` evaluated on ``x_grid``.

    Raises
    ------
    ValueError
        If ``x_grid`` contains non-positive entries or the redshift range
        is invalid.
    """
    _val.validate_x_positive(x_grid)
    _val.validate_z_range(z_min, z_max, n_z)
    _val.warn_z_max_regime(z_max)
    _val.warn_analytic_gf_heating(z_min, z_max)
    _val.warn_convolution_resolution(n_z, z_min, z_max)
    x_grid = np.asarray(x_grid, dtype=np.float64)
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
    m_x = mu_shape(x_grid)
    y_x = y_shape(x_grid)
    g_x = g_bb(x_grid)

    # Weighted sum via outer products: delta_n = sum_z hw(z) * G(x, z)
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
# constants live in ``spectroxide.cosmology``. Re-imported here so legacy
# ``from spectroxide.greens import hubble`` / ``cosmic_time`` / ``DEFAULT_COSMO``
# / ``_C_LIGHT`` etc. keep working.
from .cosmology import (  # noqa: E402,F401
    DEFAULT_COSMO,
    COSMOTHERM_GF_COSMO,
    PLANCK2015_COSMO,
    PLANCK2018_COSMO,
    _C_LIGHT,
    _K_BOLTZMANN,
    _HBAR,
    _G_NEWTON,
    _M_PROTON,
    _M_ELECTRON,
    _SIGMA_THOMSON,
    _KM_PER_MPC,
    _MPC_M,
    _EV_IN_JOULES,
    _E_RYDBERG,
    _E_ION_N2,
    _E_HE_II_ION,
    _E_HE_I_ION,
    _LAMBDA_LYA,
    _LAMBDA_2S1S,
    _cosmo_h0,
    _cosmo_omega_gamma,
    _cosmo_omega_rel,
    _cosmo_hubble,
    _cosmo_n_h,
    _cosmo_n_e,
    _saha_hydrogen,
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


def photon_survival_probability(x: ArrayLike, z: float) -> NDArray[np.float64]:
    """Analytic photon survival probability ``P_s(x, z) = exp(−x_c(z)/x)``.

    Probability that an injected photon at frequency ``x`` survives
    absorption by DC/BR processes.

    Reference: Chluba (2015), arXiv:1506.06582, Eq. 24.

    Parameters
    ----------
    x : float or array_like
        Dimensionless frequency.
    z : float
        Redshift.

    Returns
    -------
    ndarray of float64
        Survival probability ``P_s ∈ [0, 1]``; zero for non-positive ``x``.
    """
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

_H_PLANCK = 2.0 * np.pi * _HBAR
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


def _gaunt_ff_nr(x, theta_e, z_charge):
    """Non-relativistic free-free Gaunt factor (softplus interpolation)."""
    if theta_e < 1e-30 or x < 1e-30:
        return 1.0
    arg = _S3_PI * (np.log(2.25 / (x * z_charge)) + 0.5 * np.log(theta_e)) + 1.425
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

    g_z1 = _gaunt_ff_nr(x, theta_e, 1.0)
    g_z2 = _gaunt_ff_nr(x, theta_e, 2.0)

    n_hii = min(x_e_frac, 1.0) * n_h
    n_heiii = y_he_ii * n_he
    n_heii = max(y_he_i - y_he_ii, 0.0) * n_he

    species_sum = n_hii * g_z1 + 4.0 * n_heiii * g_z2 + n_heii * g_z1
    return _BR_PREFACTOR * temp_factor * species_sum


def _photon_survival_probability_numerical(x, z_h, cosmo):
    """Numerical P_s using integrated DC+BR optical depth for the y-era.

    For z_h > 5e4 (mu-era), falls back to the analytic exp(-x_c/x).
    For z_h <= 5e4 (y-era), integrates tau_ff from z=200 to z_h.

    Reference: Chluba (2015), arXiv:1506.06582, Eq. 29/32
    """
    if z_h > 5.0e4:
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

    bose_factor = np.expm1(x) if x <= 500.0 else 1e200
    inv_x3 = 1.0 / (x * x * x)
    f_he = cosmo["y_p"] / (4.0 * (1.0 - cosmo["y_p"]))

    tau = 0.0
    for i in range(n_steps + 1):
        log_z = log_z_end + i * d_log_z
        z = np.exp(log_z)
        opz = 1.0 + z

        tz = _K_BOLTZMANN * cosmo["t_cmb"] * opz / _M_E_C2
        te = tz  # T_e ~ T_z in y-era

        # Ionization fraction
        if z > 6000.0:
            x_e_frac = 1.0 + f_he
        elif z > 1500.0:
            x_e_frac = min(_saha_hydrogen(z, cosmo), 1.0)
        else:
            x_e_frac = _saha_hydrogen(z, cosmo)

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

# Gauss-Legendre nodes/weights for 32-point quadrature (half-interval [0,1])
_GL32_X, _GL32_W = np.polynomial.legendre.leggauss(32)


def _y_compton(z: ArrayLike, cosmo: CosmoLike | None = None) -> FloatOrArray:
    """Integrated Compton y-parameter ``y_γ(z) = ∫₀ᶻ θ_e σ_T n_e c / H dz'``.

    Internal helper for the photon-injection broadened bump. 32-point
    Gauss–Legendre quadrature in ``ln(1+z)``.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO

    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    scalar = np.ndim(z) == 0

    ln_min = 0.0  # ln(1+0)
    ln_max = np.log(1.0 + z_arr)  # shape (N,)

    # Map GL nodes from [-1,1] to [ln_min, ln_max] for each z
    # mid, half: shape (N,)
    mid = 0.5 * (ln_max + ln_min)
    half = 0.5 * (ln_max - ln_min)

    # GL nodes for all (z, node) pairs: shape (N, 32)
    u = mid[:, np.newaxis] + half[:, np.newaxis] * _GL32_X[np.newaxis, :]
    zp = np.exp(u) - 1.0  # shape (N, 32)
    opz = 1.0 + zp

    t_z = cosmo["t_cmb"] * opz
    theta_e = _K_BOLTZMANN * t_z / (_M_ELECTRON * _C_LIGHT**2)

    # _cosmo_n_e needs ionization_fraction which handles arrays
    # Flatten for the call, then reshape
    zp_flat = zp.ravel()
    n_e_flat = _cosmo_n_e(zp_flat, cosmo)
    n_e = np.asarray(n_e_flat, dtype=np.float64).reshape(zp.shape)

    h_z = _cosmo_hubble(zp, cosmo)  # array-safe

    # Integrand: theta_e * sigma_T * c * n_e / h_z, shape (N, 32)
    integrand = theta_e * _SIGMA_THOMSON * _C_LIGHT * n_e / h_z

    # Weighted sum over GL nodes: shape (N,)
    result = np.dot(integrand, _GL32_W) * half

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
    # Mode (peak) at x_peak = x_inj * exp(alpha*yg) / denom
    # Mean at x_mean = x_inj * exp((alpha+beta)*yg) / denom = x_inj * f_int
    mu_ln = np.log(x_inj) + alpha * yg - np.log(denom)  # = ln(x_peak)
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


def greens_function_photon(
    x_obs: ArrayLike,
    x_inj: float,
    z_h: float,
    sigma_x: float = 0.0,
    number_conserving: bool = False,
    cosmo: CosmoLike | None = None,
) -> NDArray[np.float64]:
    """Green's function for monochromatic photon injection.

    Returns ``Δn(x_obs)`` per unit ``ΔN/N`` injected at frequency
    ``x_inj`` and redshift ``z_h``.  Uses the universal ``J_μ(z)``
    visibility (same as heat injection) to blend between pure-μ-era and
    pure-y-era contributions:

    ``G_ph = J_μ G_μ + (1 − J_μ) G_y``.

    The surviving photon δ-function is represented as a Gaussian of
    width ``sigma_x`` (set 0 to omit it, e.g. for ``P_s ≈ 0`` tests).

    When ``P_s = 0``, reduces to
    ``α_ρ x_inj · greens_function(x_obs, z_h)``.

    Reference: Chluba (2015), arXiv:1506.06582.

    Parameters
    ----------
    x_obs : float or array_like
        Observation frequency.
    x_inj : float
        Injection frequency (must be positive and finite).
    z_h : float
        Injection redshift.  Must lie outside the μ–y transition band
        ``(5e4, 2e5)``; otherwise a :class:`ValueError` is raised.
    sigma_x : float, optional
        Gaussian width for the surviving photon δ-function (default 0).
    number_conserving : bool, optional
        If True, drop the temperature-shift component so the result
        satisfies ``∫ x² G dx ≈ 0`` (CosmoTherm convention for stored
        Green's function entries).  Default False.
    cosmo : Mapping, optional
        Cosmological parameters for Compton y-parameter broadening.
        Defaults to :data:`~spectroxide.cosmology.DEFAULT_COSMO`.

    Returns
    -------
    ndarray of float64
        Spectral distortion ``Δn(x_obs)`` per unit ``ΔN/N``.

    Raises
    ------
    ValueError
        If ``x_obs``, ``x_inj``, or ``z_h`` are out of range, or if
        ``z_h`` falls inside the μ–y transition.
    """
    _val.validate_z_h(z_h)
    _val.validate_x_positive(x_obs, label="x_obs")
    _val.validate_x_inj(x_inj)
    _val.warn_z_h_regime(z_h)
    _val.warn_x_inj_regime(x_inj)
    _val.validate_photon_gf_regime(z_h)
    x_obs = np.asarray(x_obs, dtype=np.float64)

    _j_bb_star = j_bb_star(z_h)
    _cosmo = cosmo if cosmo is not None else DEFAULT_COSMO
    p_s = _photon_survival_probability_numerical(x_inj, z_h, _cosmo)

    alpha_x = ALPHA_RHO * x_inj

    # Universal mu-y transition (same as heat injection GF)
    _j_mu = j_mu(z_h)

    # --- mu-era contribution ---
    mu_factor = 1.0 - p_s * X_BALANCED / x_inj
    mu_part = (3.0 / KAPPA_C) * _j_bb_star * mu_factor * mu_shape(x_obs)

    # Temperature shift (energy conservation residual)
    lam = 1.0 - mu_factor * _j_bb_star
    t_part = lam / 4.0 * temperature_shift_shape(x_obs)

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
    bump_shape, f_int = _broadened_bump(x_obs, x_inj, yg)

    # Smooth y-era: energy balance coefficient.
    # f_int = <x>/x_inj = mean energy ratio of the log-normal bump.
    # Surviving photon carries energy P_s * f_int * alpha_x.
    # Remaining (1 - P_s * f_int) goes into smooth y.
    coeff_y = 1.0 - p_s * f_int
    y_smooth = coeff_y * 0.25 * y_shape(x_obs)

    # Combine mu and y via universal visibility J_mu(z).
    if number_conserving:
        smooth = alpha_x * (_j_mu * mu_part + (1.0 - _j_mu) * y_smooth)
    else:
        smooth = alpha_x * (_j_mu * (mu_part + t_part) + (1.0 - _j_mu) * y_smooth)

    # Surviving photon bump (broadened by Compton scattering).
    safe_x = np.where(x_obs > 1e-30, x_obs, 1e-30)
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
        gauss = np.exp(-((x_obs - x_inj) ** 2) / (2.0 * sigma_x**2)) * norm_g
        surviving = p_s * (1.0 - _j_mu) * G2_PLANCK / (safe_x**2) * gauss
    else:
        surviving = p_s * (1.0 - _j_mu) * G2_PLANCK / (safe_x**2) * bump_shape
    result = smooth + surviving

    return result


def mu_from_photon_injection(x_inj: float, z_h: float, delta_n_over_n: float) -> float:
    """μ from monochromatic photon injection.

    .. math::

        \\mu = \\alpha_\\rho \\, x_{inj} \\, \\frac{3}{\\kappa_c}
               J_{bb}^*(z_h) J_\\mu(z_h)
               \\left(1 - P_s \\frac{x_0}{x_{inj}}\\right)
               \\frac{\\Delta N}{N}.

    Uses the universal ``J_μ(z)`` visibility function (same as heat
    injection).

    Sign behaviour
    --------------
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

    Returns
    -------
    float
        Dimensionless μ-parameter.
    """
    _val.validate_x_inj(x_inj)
    _val.validate_z_h(z_h)
    _val.warn_z_h_regime(z_h)
    _val.warn_x_inj_regime(x_inj)
    _val.validate_photon_gf_regime(z_h)
    _j_bb_star = float(j_bb_star(z_h))
    _j_mu = j_mu(z_h)
    p_s = float(photon_survival_probability(np.array([x_inj]), z_h)[0])

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


def distortion_from_photon_injection(
    x_grid: ArrayLike,
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
    ``(5e4, 2e5)``; if the integration range overlaps it, points landing
    in the band raise :class:`ValueError` (preceded by a warning).

    Parameters
    ----------
    x_grid : array_like
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
        Distortion ``Δn(x)`` evaluated on ``x_grid``.

    Raises
    ------
    ValueError
        If any active source redshift falls in the μ–y transition band.
    """
    _val.validate_x_positive(x_grid)
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
    x_grid = np.asarray(x_grid, dtype=np.float64)
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
        # Match Rust's skip behaviour: drop samples in (Y_ERA_Z_MAX, MU_ERA_Z_MIN)
        # rather than panic. The slack matches the Rust TOL constant.
        tol = 1.0e-6
        lo = _val.PHOTON_GF_Y_ERA_Z_MAX * (1.0 + tol)
        hi = _val.PHOTON_GF_MU_ERA_Z_MIN * (1.0 - tol)
        active &= (z_arr <= lo) | (z_arr >= hi)
    active_idx = np.where(active)[0]

    delta_n = np.zeros_like(x_grid)
    for j in active_idx:
        delta_n += (
            greens_function_photon(x_grid, x_inj, float(z_arr[j]), sigma_x, cosmo=cosmo)
            * sw[j]
        )

    return delta_n


# ---------------------------------------------------------------------------
# Unit conversion: Δn(x) → ΔI(ν) in physical units
# ---------------------------------------------------------------------------

_H_PLANCK = 6.626_070_15e-34  # J·s


DEFAULT_DECOMP_X_MIN = 0.5
DEFAULT_DECOMP_X_MAX = 18.0


def decompose_distortion(
    x_grid: ArrayLike,
    delta_n: ArrayLike,
    z_h: float | None = None,
    method: str = "bf",
) -> dict:
    """Decompose ``Δn(x)`` into ``(μ, y, ΔT/T)`` components.

    Default method: Bianchini & Fabbian (2022) nonlinear Bose–Einstein fit
    (``method="bf"``), matching the Rust ``spectroxide::distortion::decompose_distortion``.

    **``method="bf"`` (default):** Nonlinear least-squares fit of
    Δn(x) = [n_pl(x/(1+δ)) − n_pl(x)] + [n_BE(x+μ) − n_pl(x)] + y·Y_SZ(x)
    on x ∈ [0.5, 18], bootstrapped from a linear Gram-Schmidt initial guess
    and refined by Levenberg-Marquardt. See :func:`_decompose_nonlinear_be`.

    **``method="gs"``:** Linear Gram-Schmidt orthogonalisation of
    (Y_SZ, M, G) over the same band (Chluba & Jeong 2014, Appendix A).
    Agrees with ``bf`` on μ and y to numerical precision at realistic
    injection amplitudes; see :func:`_decompose_gram_schmidt`.

    **``method="gf_fit"`` (requires z_h):** Three-component Green's-function
    spectral fit for visibility-function calibration against PDE spectra.
    NC-strips the spectrum, fixes J_y from Chluba (2013) Eq. 5, then fits
    J_μ and J_bb* by minimising the x³-weighted residual. Use this for
    visibility calibration, NOT for production μ/y extraction.

    Parameters
    ----------
    x_grid : array_like
        Dimensionless frequency grid ``x = h ν / (k_B T_z)``.
    delta_n : array_like
        Spectral distortion ``Δn(x)`` (same length as ``x_grid``).
    z_h : float, optional
        Injection redshift.  Required when ``method="gf_fit"``; ignored
        (with a warning) for the other methods.
    method : {"bf", "gs", "gf_fit"}, optional
        Decomposition method (default ``"bf"``).

    Returns
    -------
    dict
        Keys ``mu`` (float), ``y`` (float), ``dT`` (float, ΔT/T),
        ``drho`` (float, Δρ/ρ), ``dn_over_n`` (float, ΔN/N), and
        ``residual`` (ndarray, ``Δn − model`` on ``x_grid``).  When
        ``gf_fit`` is used, also includes ``j_mu_fit``, ``j_bb_star_fit``,
        ``j_y``, ``fit_success``, ``fit_residual``.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or if ``method="gf_fit"`` is selected
        without supplying ``z_h``.
    RuntimeError
        If the ``gf_fit`` L-BFGS-B optimisation fails to converge.
    """
    if method == "bf":
        if z_h is not None:
            import warnings

            warnings.warn(
                "decompose_distortion: z_h is ignored for method='bf'. "
                "Pass method='gf_fit' to use the Green's-function spectral fit.",
                stacklevel=2,
            )
        return _decompose_nonlinear_be(x_grid, delta_n)

    if method == "gs":
        if z_h is not None:
            import warnings

            warnings.warn(
                "decompose_distortion: z_h is ignored for method='gs'. "
                "Pass method='gf_fit' to use the Green's-function spectral fit.",
                stacklevel=2,
            )
        return _decompose_gram_schmidt(x_grid, delta_n)

    if method != "gf_fit":
        raise ValueError(
            f"decompose_distortion: unknown method={method!r}; "
            "expected 'bf', 'gs', or 'gf_fit'."
        )
    if z_h is None:
        raise ValueError("decompose_distortion: method='gf_fit' requires z_h.")

    from scipy.optimize import minimize as sp_minimize
    from .cosmotherm import strip_gbb

    _val.validate_x_positive(x_grid)
    _val.validate_array_lengths(x_grid, delta_n)
    _val.warn_x_grid_narrow(x_grid)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    delta_n = np.asarray(delta_n, dtype=np.float64)
    mu_to_energy = 3.0 / KAPPA_C  # ≈ 1.401

    # Model-independent integrals
    dx = np.diff(x_grid)
    x_mid = 0.5 * (x_grid[:-1] + x_grid[1:])
    dn_mid = 0.5 * (delta_n[:-1] + delta_n[1:])
    drho = np.sum(x_mid**3 * dn_mid * dx)
    dn_n = np.sum(x_mid**2 * dn_mid * dx)
    drho_over_rho = drho / G3_PLANCK
    dn_over_n = dn_n / G2_PLANCK

    # NC-strip the PDE spectrum
    dn_nc, _alpha = strip_gbb(x_grid, delta_n)

    # J_y from Chluba's independent fitting formula (fixed)
    j_y_val = j_y(z_h)

    def _gf_model(x, j_mu_val, j_bb_star_val, j_y_fixed):
        """Three-component GF ansatz per unit Δρ/ρ."""
        return (
            (3.0 / KAPPA_C) * j_mu_val * j_bb_star_val * mu_shape(x)
            + 0.25 * j_y_fixed * y_shape(x)
            + 0.25 * (1.0 - j_mu_val * j_bb_star_val - j_y_fixed) * g_bb(x)
        )

    def _residual(params):
        jm, jb = params
        model = _gf_model(x_grid, jm, jb, j_y_val) * drho_over_rho
        model_nc, _ = strip_gbb(x_grid, model)
        return float(np.sum((x_grid**3 * (model_nc - dn_nc)) ** 2))

    # Initial guess from analytic GF
    res = sp_minimize(
        _residual,
        [j_mu(z_h), j_bb_star(z_h)],
        bounds=[(0, 1), (0, 1)],
        method="L-BFGS-B",
    )
    if not res.success:
        raise RuntimeError(
            f"decompose_distortion: L-BFGS-B fit failed at z_h={z_h}: "
            f"{res.message} (nit={res.nit}, fun={res.fun:.3e})"
        )
    j_mu_fit = float(res.x[0])
    j_bb_star_fit = float(res.x[1])

    # Extract μ, y from fitted visibility functions
    mu = mu_to_energy * j_mu_fit * j_bb_star_fit * drho_over_rho
    y_val = 0.25 * j_y_val * drho_over_rho

    # ΔT/T from energy conservation
    dT = drho_over_rho / 4.0 - mu / (4.0 * mu_to_energy) - y_val

    # Residual
    residual = (
        delta_n - mu * mu_shape(x_grid) - y_val * y_shape(x_grid) - dT * g_bb(x_grid)
    )

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
        "fit_success": bool(res.success),
        "fit_residual": float(res.fun),
    }


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
    ``(Y_SZ, M, G_bb)`` via Gram–Schmidt under the trapezoidal inner
    product
    ``⟨a, b⟩ = ∫_{x_min}^{x_max} a(x) b(x) dx``,
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
        Lower band edge (default ``0.5``, PIXIE-like).
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

    mask, dx = _band_trap_weights(x_grid, x_min, x_max)
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

    by Levenberg–Marquardt over the band ``[x_min, x_max]``,
    bootstrapped from :func:`_decompose_gram_schmidt` (converted to the
    B&F parameterisation via ``δ_BF = δ_GS + μ/β_μ``).  The LM iteration
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

    mask, dx = _band_trap_weights(x_grid, x_min, x_max)
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

    # Bootstrap from GS (translated to BF parameterisation).
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


def delta_n_to_delta_I(
    x: ArrayLike, dn: ArrayLike, t_cmb: float = 2.726
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert ``Δn(x)`` to ``(ν [GHz], ΔI [Jy/sr])``.

    Uses ``ΔI = (2 h ν³ / c²) Δn`` with ``ν = x k_B T₀ / h``.

    Parameters
    ----------
    x : array_like
        Dimensionless frequency ``x = h ν / (k_B T_z)``.
    dn : array_like
        Spectral distortion ``Δn(x)``.
    t_cmb : float, optional
        CMB temperature today, in **K**.  Default 2.726.

    Returns
    -------
    nu_ghz : ndarray of float64
        Frequency in **GHz**.
    di_jy : ndarray of float64
        Intensity distortion in **Jy/sr** (= 10⁻²⁶ W m⁻² Hz⁻¹ sr⁻¹).
    """
    x = np.asarray(x, dtype=float)
    dn = np.asarray(dn, dtype=float)

    nu_hz = x * _K_BOLTZMANN * t_cmb / _H_PLANCK
    nu_ghz = nu_hz / 1e9

    # ΔI = (2hν³/c²) × Δn, converted to Jy/sr
    di_si = 2.0 * _H_PLANCK * nu_hz**3 / _C_LIGHT**2 * dn
    di_jy = di_si / 1e-26

    return nu_ghz, di_jy
