"""Input validation for the spectroxide public API.

Two-tier validation
-------------------
- **ERROR** (:class:`ValueError`) — nonsensical inputs that cannot
  produce meaningful results.
- **WARNING** (:func:`warnings.warn`) — inputs in untested or unreliable
  regimes; results may still be physically meaningful but require
  scrutiny.

Each public validator mutates nothing; it either returns ``None`` (the
input is accepted) or raises :class:`ValueError` (the input is rejected).
The ``warn_*`` helpers issue :class:`UserWarning` and never raise.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike

# ---------------------------------------------------------------------------
# Error validators
# ---------------------------------------------------------------------------


def validate_z_h(z_h: float | None, *, label: str = "z_h") -> None:
    """Validate an injection redshift.

    Parameters
    ----------
    z_h : float or None
        Injection redshift.  *None* is accepted and short-circuited.
    label : str, optional
        Variable name used in error messages (default ``"z_h"``).

    Raises
    ------
    ValueError
        If ``z_h`` is non-finite or non-positive.
    """
    if z_h is not None:
        if not np.isfinite(z_h):
            raise ValueError(f"{label} must be finite, got {z_h}")
        if z_h <= 0:
            raise ValueError(f"{label} must be positive, got {z_h}")


def validate_x_positive(x: ArrayLike, *, label: str = "x") -> None:
    """Validate that every element of a frequency array is positive and finite.

    Parameters
    ----------
    x : array_like
        Frequency values.  An empty array passes silently.
    label : str, optional
        Variable name used in error messages (default ``"x"``).

    Raises
    ------
    ValueError
        If any element is non-finite or non-positive.
    """
    x = np.asarray(x)
    if x.size > 0:
        if not np.all(np.isfinite(x)):
            raise ValueError(f"All {label} values must be finite")
        if np.any(x <= 0):
            raise ValueError(f"All {label} values must be positive, got min={x.min()}")


def validate_x_inj(x_inj: float) -> None:
    """Validate the injection frequency ``x_inj``.

    Parameters
    ----------
    x_inj : float
        Injection frequency (dimensionless).

    Raises
    ------
    ValueError
        If ``x_inj`` is non-finite or non-positive.
    """
    if not np.isfinite(x_inj):
        raise ValueError(f"x_inj must be finite, got {x_inj}")
    if x_inj <= 0:
        raise ValueError(f"x_inj must be positive, got {x_inj}")


def validate_z_range(z_min: float, z_max: float, n_z: int | None = None) -> None:
    """Validate an integration redshift range.

    Parameters
    ----------
    z_min : float
        Lower bound; must be finite and non-negative.
    z_max : float
        Upper bound; must be finite and strictly greater than ``z_min``.
    n_z : int, optional
        Number of integration points.  When provided, must be ≥ 10.

    Raises
    ------
    ValueError
        If any of the conditions above are violated.
    """
    if not np.isfinite(z_min):
        raise ValueError(f"z_min must be finite, got {z_min}")
    if not np.isfinite(z_max):
        raise ValueError(f"z_max must be finite, got {z_max}")
    if z_min < 0:
        raise ValueError(f"z_min must be non-negative, got {z_min}")
    if z_min >= z_max:
        raise ValueError(
            f"z_min must be less than z_max, got z_min={z_min}, z_max={z_max}"
        )
    if n_z is not None and n_z < 10:
        raise ValueError(f"n_z must be >= 10, got {n_z}")


def validate_array_lengths(x: ArrayLike, delta_n: ArrayLike) -> None:
    """Check that ``x`` and ``delta_n`` have matching lengths.

    Parameters
    ----------
    x : array_like
        Frequency grid.
    delta_n : array_like
        Spectral distortion at the same grid points.

    Raises
    ------
    ValueError
        If the lengths differ.
    """
    if len(x) != len(delta_n):
        raise ValueError(
            f"x and delta_n must have the same length, got {len(x)} and {len(delta_n)}"
        )


def validate_finite_scalar(val: float | None, label: str) -> None:
    """Validate that a scalar is finite (i.e. neither NaN nor ±Inf).

    Parameters
    ----------
    val : float or None
        Value to check.  *None* is accepted and short-circuited.
    label : str
        Variable name used in error messages.

    Raises
    ------
    ValueError
        If ``val`` is not finite.
    """
    if val is not None and not np.isfinite(val):
        raise ValueError(f"{label} must be finite, got {val}")


def require_z_h(z_h: float | None) -> None:
    """Stricter version of :func:`validate_z_h`: also forbids *None*.

    Parameters
    ----------
    z_h : float or None
        Injection redshift; must not be *None*.

    Raises
    ------
    ValueError
        If ``z_h`` is *None* or fails :func:`validate_z_h`.
    """
    if z_h is None:
        raise ValueError("z_h is required but was None")
    validate_z_h(z_h)


def validate_n_eff(n_eff: float | None) -> None:
    """Validate the effective neutrino species count ``N_eff``.

    Parameters
    ----------
    n_eff : float or None
        Effective number of relativistic species.  *None* is accepted
        and short-circuited.

    Raises
    ------
    ValueError
        If ``n_eff`` is non-finite or outside ``[0, 20]``.
    """
    if n_eff is not None:
        if not np.isfinite(n_eff):
            raise ValueError(f"n_eff must be finite, got {n_eff}")
        if n_eff < 0 or n_eff > 20:
            raise ValueError(f"n_eff must be in [0, 20], got {n_eff}")


def validate_cosmology(cosmo: Mapping[str, float] | Any | None) -> None:
    """Validate a cosmology specification.

    Parameters
    ----------
    cosmo : Mapping or Cosmology or None
        Either a dict-like object or a :class:`spectroxide.Cosmology`
        dataclass.  Required keys/attributes (when present): ``h``,
        ``omega_b``, ``omega_m``, ``y_p``, ``t_cmb``, ``n_eff``.  *None*
        is accepted and short-circuited.

    Raises
    ------
    ValueError
        If any present field is unphysical (non-finite, non-positive,
        ``omega_m < omega_b``, ``y_p ∉ [0, 1)``, etc.).
    """
    if cosmo is None:
        return

    # Support both dict and Cosmology dataclass
    def _get(key):
        if isinstance(cosmo, dict):
            return cosmo.get(key)
        return getattr(cosmo, key, None)

    h = _get("h")
    validate_finite_scalar(h, "h")
    if h is not None and h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    if h is not None and h > 10:
        raise ValueError(f"h={h} implies H0={h*100} km/s/Mpc, which is nonsensical")
    omega_b = _get("omega_b")
    validate_finite_scalar(omega_b, "omega_b")
    if omega_b is not None and omega_b <= 0:
        raise ValueError(f"omega_b must be positive, got {omega_b}")
    omega_m = _get("omega_m")
    validate_finite_scalar(omega_m, "omega_m")
    if omega_m is not None and omega_m <= 0:
        raise ValueError(f"omega_m must be positive, got {omega_m}")
    if omega_m is not None and omega_b is not None and omega_m < omega_b:
        raise ValueError(
            f"omega_m must be >= omega_b, got omega_m={omega_m}, omega_b={omega_b}"
        )
    y_p = _get("y_p")
    validate_finite_scalar(y_p, "y_p")
    if y_p is not None and not (0 <= y_p < 1):
        raise ValueError(f"y_p must be in [0, 1), got {y_p}")
    t_cmb = _get("t_cmb")
    validate_finite_scalar(t_cmb, "t_cmb")
    if t_cmb is not None and t_cmb <= 0:
        raise ValueError(f"t_cmb must be positive, got {t_cmb}")
    n_eff = _get("n_eff")
    validate_finite_scalar(n_eff, "n_eff")
    validate_n_eff(n_eff)


def validate_delta_rho(delta_rho: float) -> None:
    """Validate the fractional energy injection ``Δρ/ρ``.

    Parameters
    ----------
    delta_rho : float
        Fractional energy perturbation.

    Raises
    ------
    ValueError
        If ``delta_rho`` is not finite.

    Warns
    -----
    UserWarning
        If ``|Δρ/ρ| > 0.01`` — the linearised Kompaneets equation is no
        longer accurate in this regime.
    """
    if not np.isfinite(delta_rho):
        raise ValueError(f"delta_rho must be finite, got {delta_rho}")
    if abs(delta_rho) > 0.01:
        warnings.warn(
            f"delta_rho={delta_rho:.2e}: Strong distortion regime (|Δρ/ρ| > 0.01) "
            "is not implemented. The Kompaneets equation is linearized and results "
            "will be inaccurate for large energy injections.",
            stacklevel=3,
        )


def validate_dq_dz_callable(
    dq_dz: Callable[[float], float] | None,
    z_min: float,
    z_max: float,
) -> None:
    """Spot-check a heating-rate callable for non-finite output.

    Evaluates ``dq_dz`` at five log-spaced redshifts in
    ``[z_min, z_max]``.  Catches the common bug where users return
    ``np.inf`` or ``np.nan`` for redshifts where their formula has a
    division by zero.

    Parameters
    ----------
    dq_dz : callable or None
        Heating rate ``z -> dQ/dz``.  *None* is accepted and
        short-circuited.
    z_min : float
        Lower edge of the sample range.
    z_max : float
        Upper edge of the sample range.

    Raises
    ------
    ValueError
        If any sampled value is non-finite.
    """
    if dq_dz is None:
        return
    # 32 log-spaced points (~one per 0.2 dex over [1e2, 5e6]) catches
    # isolated singularities that 5-point sampling misses (e.g. 1/(z-z*)
    # with z* between two of the original five samples). Plus the literal
    # endpoints, which user formulas often hit edge cases at.
    sample_zs = np.unique(
        np.concatenate([[z_min, z_max], np.geomspace(z_min, z_max, 32)])
    )
    for z in sample_zs:
        val = dq_dz(z)
        if not np.isfinite(val):
            raise ValueError(
                f"dq_dz({z:.2e}) = {val}: heating rate callable returned "
                "non-finite value. Check for division by zero or NaN at "
                f"z in [{z_min:.2e}, {z_max:.2e}]."
            )


# ---------------------------------------------------------------------------
# Warning validators
# ---------------------------------------------------------------------------


def warn_z_h_regime(z_h):
    """Warn if z_h is outside the reliable Green's function regime."""
    if z_h is not None and z_h > 5e6:
        warnings.warn(
            f"z_h={z_h:.2e}: Green's function unreliable in deep thermalization "
            "regime (J_bb* < 1e-2). Use the PDE solver: "
            "solve(injection={'type': ..., 'z_h': ...}, delta_rho=...).",
            stacklevel=3,
        )
    elif z_h is not None and z_h > 3e6:
        warnings.warn(
            f"z_h={z_h:.2e}: Green's function approaching deep thermalization "
            "regime (J_bb* < 1e-1). Spectral residuals grow rapidly above 3e6; "
            "validate against the PDE solver before trusting quantitative results.",
            stacklevel=3,
        )
    if z_h is not None and z_h < 1100:
        warnings.warn(
            f"z_h={z_h:.0f}: Post-recombination injection (z < 1100). The analytic "
            "Green's function predicts a Compton-mediated y/μ signal, but "
            "Comptonization is inefficient post-recombination (X_e ~ 1e-4) and the "
            "physical signal is locked-in at the injection frequency with no "
            "mu/y redistribution. The returned G_th is therefore unphysical here; "
            "use the PDE solver (which handles the Compton visibility correctly).",
            stacklevel=3,
        )


def warn_x_inj_regime(x_inj):
    """Warn if injection frequency is in extreme regime."""
    if x_inj < 0.01:
        warnings.warn(
            f"x_inj={x_inj:.2e}: DC/BR absorption extremely strong at this "
            "frequency; survival probability ~ 0.",
            stacklevel=3,
        )
    if x_inj > 150:
        warnings.warn(
            f"x_inj={x_inj:.1f}: Injection frequency beyond validated range "
            "(x_inj ≤ 150). PDE solver is stable but results are not validated "
            "against CosmoTherm or literature at this frequency.",
            stacklevel=3,
        )


def warn_z_max_regime(z_max):
    """Warn if integration extends beyond reliable GF regime.

    The PDE solver hard-errors at ``z_start > 1e7`` (Kompaneets Fokker-Planck
    invalid for theta_e > 0.005), so the GF should warn well before that —
    we mirror the PDE soft-warning threshold (``5e6``) and escalate at ``1e7``.
    """
    if z_max > 1e7:
        warnings.warn(
            f"z_max={z_max:.2e}: Integration extends beyond Kompaneets validity "
            "(theta_e > 0.005). Both GF and PDE results have qualitative errors here.",
            stacklevel=3,
        )
    elif z_max > 5e6:
        warnings.warn(
            f"z_max={z_max:.2e}: Approaching theta_e^2 corrections regime; expect "
            "~1% systematic errors from Kompaneets Fokker-Planck approximation.",
            stacklevel=3,
        )


PHOTON_GF_Y_ERA_Z_MAX = 5.0e4
PHOTON_GF_MU_ERA_Z_MIN = 2.0e5


def validate_photon_gf_regime(z_h: float | None) -> None:
    """Reject photon-GF injection redshifts in the μ–y transition.

    The simple ``μ + y`` decomposition is not valid for
    ``5 × 10⁴ < z_h < 2 × 10⁵`` — residual r-type contributions become
    important and users must run the PDE solver there.

    Parameters
    ----------
    z_h : float or None
        Injection redshift.  *None* is accepted and short-circuited.

    Raises
    ------
    ValueError
        If ``z_h`` lies inside the forbidden transition band.
    """
    if z_h is None:
        return
    # Fractional slack absorbs log/exp roundoff at the boundaries.
    tol = 1.0e-6
    lo = PHOTON_GF_Y_ERA_Z_MAX * (1.0 + tol)
    hi = PHOTON_GF_MU_ERA_Z_MIN * (1.0 - tol)
    if lo < z_h < hi:
        raise ValueError(
            f"z_h={z_h:.3e}: photon Green's function is not valid in the "
            f"mu-y transition era ({PHOTON_GF_Y_ERA_Z_MAX:.0e} < z_h "
            f"< {PHOTON_GF_MU_ERA_Z_MIN:.0e}). Use the PDE solver instead."
        )


def warn_x_grid_narrow(x_grid):
    """Warn if frequency grid is too narrow for decomposition."""
    x_grid = np.asarray(x_grid)
    if x_grid.size > 0 and x_grid.max() < 10:
        warnings.warn(
            f"x_grid.max()={x_grid.max():.1f}: Frequency grid too narrow for "
            "accurate mu/y decomposition.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Green's function table & grid resolution warnings
# ---------------------------------------------------------------------------


def warn_analytic_gf_heating(z_min, z_max):
    """Warn when analytic GF covers the mu-y transition region.

    The analytic Green's function has 8-13% spectral shape errors in the
    transition era (3e4 < z < 2e5) because it decomposes into pure mu +
    pure y + temperature shift.  The PDE-based Green's function table
    (GreensTable) captures the true intermediate shapes.
    """
    # Only warn if the integration range overlaps the transition region
    if z_min < 2e5 and z_max > 3e4:
        warnings.warn(
            "Analytic Green's function has 8-13% spectral shape errors in the "
            "mu-y transition region (3e4 < z < 2e5). For percent-level accuracy, "
            "use the PDE-based Green's function table:\n"
            "  table = spectroxide.load_or_build_greens_table()\n"
            "  dn = table.distortion_from_heating(x, dq_dz, z_min, z_max)",
            stacklevel=4,
        )


def warn_table_z_density(z_injections):
    """Warn if z-injection grid is too sparse for accurate interpolation.

    Cubic spline interpolation in log(z_h) needs adequate sampling,
    especially in the transition region (3e4-2e5) where the Green's
    function shape changes rapidly.
    """
    z = np.asarray(z_injections)
    if z.size < 2:
        warnings.warn(
            f"Only {z.size} z-injection point(s). Need >= 50 for "
            "reliable Green's function table.",
            stacklevel=3,
        )
        return

    # Check points per log-decade in the transition region
    in_transition = z[(z >= 3e4) & (z <= 2e5)]
    if in_transition.size > 0:
        log_range = np.log10(2e5) - np.log10(3e4)  # ~0.82 decades
        density = in_transition.size / log_range
        if density < 10:
            warnings.warn(
                f"Only {in_transition.size} z-injection points in the mu-y "
                f"transition region (3e4-2e5), giving {density:.0f} points per "
                "log-decade. This region has the steepest shape changes. "
                "Recommend >= 10 points per log-decade (use >= 150 total "
                "z-injection points with default log-spacing).",
                stacklevel=3,
            )

    # Check overall density
    total_decades = np.log10(z.max() / z.min())
    if total_decades > 0:
        overall_density = z.size / total_decades
        if overall_density < 15:
            warnings.warn(
                f"Green's function table has {z.size} z-injection points over "
                f"{total_decades:.1f} log-decades ({overall_density:.0f}/decade). "
                "Recommend >= 15 points per log-decade for reliable cubic "
                "spline interpolation.",
                stacklevel=3,
            )


def warn_convolution_resolution(n_z, z_min, z_max):
    """Warn if the convolution integral has too few redshift points.

    Trapezoidal integration in ln(1+z) needs sufficient sampling to
    resolve features in the heating rate and the Green's function
    visibility transitions.
    """
    if n_z is None or z_min is None or z_max is None:
        return
    if z_min <= 0 or z_max <= z_min:
        return  # Will be caught by validate_z_range
    log_range = np.log10(1.0 + z_max) - np.log10(1.0 + z_min)
    if log_range > 0:
        density = n_z / log_range
        if density < 500:
            warnings.warn(
                f"Convolution uses {n_z} redshift points over "
                f"{log_range:.1f} log-decades ({density:.0f}/decade). "
                "This may under-resolve sharp heating features or "
                "visibility transitions. Recommend n_z >= 2000 for "
                "broad z-ranges (>2 decades).",
                stacklevel=3,
            )


def warn_grid_resolution_photon(n_points, injection_type):
    """Warn if PDE grid is too coarse for photon injection scenarios.

    Monochromatic photon injection creates sharp spectral features that
    require fine frequency grids to resolve. The DEBUG preset (1000 points)
    gives ~10% errors at injection peaks.
    """
    photon_types = {
        "monochromatic_photon",
        "monochromatic-photon",
        "decaying_particle_photon",
        "decaying-particle-photon",
    }
    if injection_type is not None and injection_type.replace("_", "-") in {
        t.replace("_", "-") for t in photon_types
    }:
        if n_points is not None and n_points < 2000:
            warnings.warn(
                f"Photon injection with n_points={n_points}. Spectral peaks "
                "from monochromatic photon injection are poorly resolved "
                "below 2000 grid points (~10% amplitude errors). "
                "Use n_points >= 2000, or PRODUCTION settings for "
                "publication quality.",
                stacklevel=3,
            )


def warn_table_z_coverage(z_injections, z_min_query, z_max_query):
    """Warn if convolution bounds extend beyond the table's z-injection range.

    Extrapolation beyond the table edges uses cubic spline extrapolation,
    which can produce large errors.
    """
    z = np.asarray(z_injections)
    if z.size == 0:
        return
    if z_min_query < z.min() * 0.9:
        warnings.warn(
            f"Integration lower bound z_min={z_min_query:.0f} is below the "
            f"table's minimum z_h={z.min():.0f}. Values below the table "
            "range use extrapolation, which may be inaccurate.",
            stacklevel=3,
        )
    if z_max_query > z.max() * 1.1:
        warnings.warn(
            f"Integration upper bound z_max={z_max_query:.0f} is above the "
            f"table's maximum z_h={z.max():.0f}. Values above the table "
            "range use extrapolation, which may be inaccurate.",
            stacklevel=3,
        )
