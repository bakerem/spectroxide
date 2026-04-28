"""
Precomputed Green's-function tables for CMB spectral distortions.

Two table classes
-----------------
- :class:`GreensTable` — 2-D heating Green's function ``G_th(x, z_h)``.
- :class:`PhotonGreensTable` — 3-D photon-injection Green's function
  ``G_ph(x, x_inj, z_h)``.

Tables are built by running the PDE solver at many injection redshifts,
then interpolating for fast convolution of arbitrary injection
histories.  This eliminates the 30–70% shape errors of the analytic
Green's function in the μ-to-y transition region ``3 × 10⁴ < z < 10⁵``.

Usage::

    from spectroxide import load_or_build_greens_table

    # Build (or load cached) heating table
    table = load_or_build_greens_table()
    # Force a rebuild via:
    table = load_or_build_greens_table(rebuild=True)

    # Evaluate Δn(x) per unit Δρ/ρ at injection redshift z_h
    dn = table.greens_function(x, z_h=5e4) * delta_rho

    # Convolve with an arbitrary heating history
    dn = table.distortion_from_heating(x, dq_dz, z_min=1e3, z_max=3e6)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import greens

trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
from .solver import (
    get_physics_hash,
    run_sweep,
    run_photon_sweep,
    run_photon_sweep_batch,
)

#: Type alias for a heating-rate callable ``z -> dQ/dz``.
HeatingRate = Callable[[ArrayLike], ArrayLike]


class GreensTableHashMismatch(UserWarning):
    """Warned when a cached Green's function table was built by a different
    physics-code version than the currently-installed Rust binary.

    The cached table and the current binary may produce inconsistent
    results. Regenerate via ``load_or_build_greens_table(rebuild=True)``
    to bring the cache back in sync. To suppress the warning, pass
    ``verify_hash=False`` to ``load``.
    """


def _check_table_hash(stored, path):
    expected = get_physics_hash()
    if stored != expected:
        warnings.warn(
            f"Cached Green's function table at {path} was built with a "
            f"different version of the spectroxide binary "
            f"(table hash: {stored or '<none>'}, current: {expected}). "
            f"Results may be inconsistent with the current binary. "
            f"Regenerate via load_or_build_greens_table(rebuild=True).",
            GreensTableHashMismatch,
            stacklevel=3,
        )


def _get_interpolator_class():
    """Lazy import of scipy.interpolate.RegularGridInterpolator."""
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:
        raise ImportError(
            "scipy is required for Green's function tables. "
            "Install it with: pip install scipy"
        ) from None
    return RegularGridInterpolator


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = Path.home() / ".spectroxide"
_DEFAULT_HEATING_CACHE = _DEFAULT_CACHE_DIR / "greens_table.npz"
_DEFAULT_PHOTON_CACHE = _DEFAULT_CACHE_DIR / "photon_greens_table.npz"

_DEFAULT_Z_INJECTIONS = np.logspace(np.log10(1e3), np.log10(5e6), 150)
_DEFAULT_PHOTON_X_INJ = np.logspace(np.log10(0.1), np.log10(20.0), 10)

_HEATING_CHUNK_SIZE = 50  # z_h values per Rust sweep call for checkpointing


def _checkpoint_path(cache_path):
    """Derive the checkpoint path from the final cache path."""
    p = Path(cache_path)
    return p.parent / (p.stem + ".partial.npz")


# ---------------------------------------------------------------------------
# GreensTable — heating Green's function table (2D: x * z_h)
# ---------------------------------------------------------------------------


@dataclass
class GreensTable:
    """Precomputed heating Green's function table.

    Stores G_th(x, z_h) = Delta-n per unit Delta-rho/rho at each
    (frequency, injection redshift) grid point. Built from PDE solver
    runs at each z_h.

    Attributes
    ----------
    z_h : np.ndarray
        Injection redshifts, shape (N_z,).
    x : np.ndarray
        Frequency grid, shape (N_x,).
    g_th : np.ndarray
        Green's function values, shape (N_x, N_z).
        ``g_th[:, j]`` is Delta-n(x) per unit Delta-rho/rho for injection at z_h[j].
    mu : np.ndarray
        Mu parameter per z_h, shape (N_z,).
    y_param : np.ndarray
        y parameter per z_h, shape (N_z,).
    delta_rho_over_rho : np.ndarray
        Energy conservation check per z_h, shape (N_z,).
    metadata : dict
        Build parameters and provenance info.
    """

    z_h: np.ndarray
    x: np.ndarray
    g_th: np.ndarray
    mu: np.ndarray
    y_param: np.ndarray
    delta_rho_over_rho: np.ndarray
    metadata: dict = field(default_factory=dict)
    _interp: Optional[object] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._build_interpolator()

    def _build_interpolator(self):
        """Build per-frequency cubic spline interpolators in log(z_h).

        Uses independent 1D cubic splines on the raw cached ``G_th`` for
        each frequency point.  Callers wanting a number-conserving result
        should apply :func:`~spectroxide.cosmotherm.strip_gbb` themselves.
        """
        from scipy.interpolate import CubicSpline

        log_z = np.log(self.z_h)
        self._log_z = log_z

        if len(self.z_h) >= 2:
            self._splines = [
                CubicSpline(log_z, self.g_th[i, :], extrapolate=True)
                for i in range(len(self.x))
            ]
        else:
            # Single z_h: no interpolation possible, return stored value directly
            self._splines = None
            self._g_clean_single = self.g_th[:, 0]
        self._log_x = np.log(self.x)

    def greens_function(self, x: ArrayLike, z_h: float) -> NDArray[np.float64]:
        """Evaluate the tabulated PDE Green's function ``G_th(x, z_h)``.

        Cubic-spline interpolation in ``ln(z_h)`` per stored frequency,
        followed by linear interpolation across frequencies in ``ln x``.
        Both axes are clipped to the stored range; queries outside the
        range are pinned to the nearest edge (no extrapolation).

        No analytic Green's function is consulted — the result depends
        only on the cached PDE table.

        Parameters
        ----------
        x : float or array_like
            Dimensionless frequency.
        z_h : float
            Injection redshift.

        Returns
        -------
        ndarray of float64
            ``Δn(x)`` per unit ``Δρ/ρ``.
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))

        if self._splines is None:
            # Single z_h: return the stored NC-stripped values directly
            g_at_z = self._g_clean_single
        else:
            log_z = np.log(np.clip(z_h, self.z_h[0], self.z_h[-1]))
            g_at_z = np.array([s(log_z) for s in self._splines])

        # Interpolate to requested x grid
        log_x_query = np.log(np.clip(x, self.x[0], self.x[-1]))
        return np.interp(log_x_query, self._log_x, g_at_z)

    def distortion_from_heating(
        self,
        x_grid: ArrayLike,
        dq_dz: HeatingRate,
        z_min: float,
        z_max: float,
        n_z: int = 5000,
    ) -> NDArray[np.float64]:
        """Convolve the table with an arbitrary heating history.

        .. math::

            \\Delta n(x) = \\int_{z_{min}}^{z_{max}}
                            G_{th}(x, z') \\frac{d(\\Delta\\rho/\\rho_\\gamma)}{dz'}
                            \\, dz',

        integrated in ``ln(1+z)`` with the trapezoidal rule.

        The convolution evaluates linearly in ``(log x, log z_h)`` against
        the *raw* cached ``G_th`` (no build-time NC strip), summed on the
        cache's native ``self.x`` grid, then linearly interpolated to
        ``x_grid``.  No NC strip is applied; callers that want the
        number-conserving Δn should call :func:`~spectroxide.cosmotherm.strip_gbb`.

        Parameters
        ----------
        x_grid : array_like
            Output frequency grid.
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
        ndarray of float64
            ``Δn(x)`` evaluated on ``x_grid``.
        """
        from . import _validation as _val
        from scipy.interpolate import RegularGridInterpolator

        _val.warn_convolution_resolution(n_z, z_min, z_max)
        _val.warn_table_z_coverage(self.z_h, z_min, z_max)

        x_grid = np.asarray(x_grid, dtype=np.float64)
        ln_min = np.log(1.0 + z_min)
        ln_max = np.log(1.0 + z_max)
        ln_z = np.linspace(ln_min, ln_max, n_z)
        z_arr = np.exp(ln_z) - 1.0
        dln = ln_z[1] - ln_z[0]

        # Heating weights
        try:
            rates = np.asarray(dq_dz(z_arr), dtype=np.float64)
            if rates.shape != z_arr.shape:
                raise ValueError
        except (TypeError, ValueError):
            rates = np.array([dq_dz(float(z)) for z in z_arr])

        hw = rates * (1.0 + z_arr) * dln
        hw[0] *= 0.5
        hw[-1] *= 0.5

        active = np.abs(hw) >= 1e-50
        if not np.any(active):
            return np.zeros_like(x_grid)
        hw_active = hw[active]
        z_active = z_arr[active]

        # Lazy-build linear interpolator on RAW g_th (matches the
        # hand-rolled paper-figure flow; avoids cubic-spline ringing).
        if getattr(self, "_gf_interp_raw", None) is None:
            self._gf_interp_raw = RegularGridInterpolator(
                (np.log10(self.x), np.log10(self.z_h)),
                self.g_th,
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )

        # Vectorised query: (n_active * n_x, 2) points → reshape to (n_active, n_x).
        log_x_cache = np.log10(self.x)
        log_z_clip = np.log10(np.clip(z_active, self.z_h[0], self.z_h[-1]))
        n_x = len(self.x)
        n_active = len(z_active)
        log_x_tiled = np.tile(log_x_cache, n_active)
        log_z_tiled = np.repeat(log_z_clip, n_x)
        gf_vals = self._gf_interp_raw(np.column_stack([log_x_tiled, log_z_tiled]))
        gf_mat = gf_vals.reshape(n_active, n_x)  # (n_active, n_x)

        # Sum on cache.x grid: dn[i] = Σ_j gf_mat[j, i] * hw_active[j]
        dn = gf_mat.T @ hw_active

        # Linear interp to caller's x_grid (no-op when x_grid == self.x).
        log_x_query = np.log(np.clip(x_grid, self.x[0], self.x[-1]))
        return np.interp(log_x_query, np.log(self.x), dn)

    def mu_y_from_heating(
        self,
        dq_dz: HeatingRate,
        z_min: float,
        z_max: float,
        n_z: int = 5000,
    ) -> Tuple[float, float]:
        """Compute ``(μ, y)`` from arbitrary heating.

        Parameters
        ----------
        dq_dz : callable
            Heating rate ``d(Δρ/ρ)/dz`` (positive for heating).
        z_min : float
            Minimum integration redshift.
        z_max : float
            Maximum integration redshift.
        n_z : int, optional
            Number of redshift integration points (default 5000).

        Returns
        -------
        tuple of (float, float)
            ``(μ, y)`` parameters (dimensionless).
        """
        log_z_grid = np.log(
            np.clip(
                np.logspace(np.log10(z_min), np.log10(z_max), n_z),
                self.z_h[0],
                self.z_h[-1],
            )
        )
        log_z_h = np.log(self.z_h)
        mu_interp = np.interp(log_z_grid, log_z_h, self.mu)
        mu = self._integrate_scalar(mu_interp, dq_dz, z_min, z_max, n_z)
        y = self._y_from_heating(dq_dz, z_min, z_max, n_z)
        return mu, y

    def _y_from_heating(self, dq_dz, z_min, z_max, n_z=5000):
        y_interp = np.interp(
            np.log(
                np.clip(
                    np.logspace(np.log10(z_min), np.log10(z_max), n_z),
                    self.z_h[0],
                    self.z_h[-1],
                )
            ),
            np.log(self.z_h),
            self.y_param,
        )
        return self._integrate_scalar(y_interp, dq_dz, z_min, z_max, n_z)

    @staticmethod
    def _integrate_scalar(param_arr, dq_dz, z_min, z_max, n_z):
        """Integrate param(z) * dq_dz(z) * (1+z) dln(1+z)."""
        ln_min = np.log(1.0 + z_min)
        ln_max = np.log(1.0 + z_max)
        ln_z = np.linspace(ln_min, ln_max, n_z)
        z_arr = np.exp(ln_z) - 1.0
        dln = ln_z[1] - ln_z[0]

        rates = np.array([dq_dz(z) for z in z_arr])
        integrand = param_arr * rates * (1.0 + z_arr)
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

        return float(_trapz(integrand, dx=dln))

    def save(self, path: str | Path | None = None) -> None:
        """Save the table to a compressed ``.npz`` file.

        Parameters
        ----------
        path : str or Path, optional
            Output path.  Default
            ``~/.spectroxide/greens_table.npz``.

        Returns
        -------
        None
        """
        if path is None:
            path = _DEFAULT_HEATING_CACHE
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            z_h=self.z_h,
            x=self.x,
            g_th=self.g_th,
            mu=self.mu,
            y_param=self.y_param,
            delta_rho_over_rho=self.delta_rho_over_rho,
            metadata_json=json.dumps(self.metadata),
        )

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        verify_hash: bool = True,
    ) -> "GreensTable":
        """Load a table from a ``.npz`` file.

        Parameters
        ----------
        path : str or Path, optional
            Input path.  Default ``~/.spectroxide/greens_table.npz``.
        verify_hash : bool, optional
            If *True* (default), check that the cached table's
            ``physics_hash`` metadata matches the current Rust binary's
            compile-time hash; emits a :class:`GreensTableHashMismatch`
            warning if not.  Disable for synthetic / test tables that
            were not built by the Rust solver.

        Returns
        -------
        GreensTable
            Reconstructed table object.

        Warns
        -----
        GreensTableHashMismatch
            If ``verify_hash=True`` and the cached hash differs from the
            currently installed binary.
        """
        if path is None:
            path = _DEFAULT_HEATING_CACHE
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata_json"]))
        if verify_hash:
            _check_table_hash(metadata.get("physics_hash"), path)
        return cls(
            z_h=data["z_h"],
            x=data["x"],
            g_th=data["g_th"],
            mu=data["mu"],
            y_param=data["y_param"],
            delta_rho_over_rho=data["delta_rho_over_rho"],
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# PhotonGreensTable — photon injection GF table (3D: x_obs * x_inj * z_h)
# ---------------------------------------------------------------------------


@dataclass
class PhotonGreensTable:
    """Precomputed photon injection Green's function table.

    Stores G_ph(x_obs, x_inj, z_h) = Delta-n per unit Delta-N/N at each
    (observation frequency, injection frequency, injection redshift) grid point.

    Attributes
    ----------
    z_h : np.ndarray
        Injection redshifts, shape (N_z,).
    x : np.ndarray
        Observation frequency grid, shape (N_x,).
    x_inj : np.ndarray
        Injection frequencies, shape (N_xinj,).
    g_ph : np.ndarray
        Green's function values, shape (N_x, N_xinj, N_z).
    metadata : dict
        Build parameters and provenance info.
    """

    z_h: np.ndarray
    x: np.ndarray
    x_inj: np.ndarray
    g_ph: np.ndarray
    metadata: dict = field(default_factory=dict)
    _interp: Optional[object] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._build_interpolator()

    def _build_interpolator(self):
        """Build 3D interpolator in (log x, log x_inj, log z_h)."""
        log_x = np.log(self.x)
        log_xi = np.log(self.x_inj)
        log_z = np.log(self.z_h)
        RGI = _get_interpolator_class()
        self._interp = RGI(
            (log_x, log_xi, log_z),
            self.g_ph,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def greens_function_photon(
        self, x_obs: ArrayLike, x_inj: float, z_h: float
    ) -> NDArray[np.float64]:
        """Interpolate ``G_ph(x_obs, x_inj, z_h)``.

        Drop-in replacement for :func:`spectroxide.greens.greens_function_photon`.
        Uses 3-D linear interpolation in ``(log x_obs, log x_inj,
        log z_h)`` with edge clipping (no extrapolation).

        Parameters
        ----------
        x_obs : float or array_like
            Observation frequency.
        x_inj : float
            Injection frequency.
        z_h : float
            Injection redshift.

        Returns
        -------
        ndarray of float64
            ``Δn(x_obs)`` per unit ``ΔN/N``.
        """
        x_obs = np.atleast_1d(np.asarray(x_obs, dtype=np.float64))
        log_xo = np.log(np.clip(x_obs, self.x[0], self.x[-1]))
        log_xi = np.full_like(
            log_xo, np.log(np.clip(x_inj, self.x_inj[0], self.x_inj[-1]))
        )
        log_z = np.full_like(log_xo, np.log(np.clip(z_h, self.z_h[0], self.z_h[-1])))
        pts = np.column_stack([log_xo, log_xi, log_z])
        return self._interp(pts)

    def distortion_from_photon_injection(
        self,
        x_grid: ArrayLike,
        x_inj: float,
        dn_dz: Callable[[float], float],
        z_min: float,
        z_max: float,
        n_z: int = 5000,
    ) -> NDArray[np.float64]:
        """Convolve the table with a photon-injection history.

        Parameters
        ----------
        x_grid : array_like
            Observation frequency grid.
        x_inj : float
            Injection frequency.
        dn_dz : callable
            Source rate ``d(ΔN/N)/dz`` (positive for injection).
        z_min : float
            Minimum integration redshift.
        z_max : float
            Maximum integration redshift.
        n_z : int, optional
            Number of redshift integration points (default 5000).

        Returns
        -------
        ndarray of float64
            Distortion ``Δn(x)`` on ``x_grid``.
        """
        from . import _validation as _val

        _val.warn_convolution_resolution(n_z, z_min, z_max)
        _val.warn_table_z_coverage(self.z_h, z_min, z_max)

        x_grid = np.asarray(x_grid, dtype=np.float64)
        ln_min = np.log(1.0 + z_min)
        ln_max = np.log(1.0 + z_max)
        ln_z = np.linspace(ln_min, ln_max, n_z)
        z_arr = np.exp(ln_z) - 1.0
        dln = ln_z[1] - ln_z[0]

        delta_n = np.zeros_like(x_grid)
        for i, z in enumerate(z_arr):
            rate = dn_dz(z)
            if rate == 0.0:
                continue
            g = self.greens_function_photon(x_grid, x_inj, z)
            weight = rate * (1.0 + z) * dln
            if i == 0 or i == len(z_arr) - 1:
                weight *= 0.5
            delta_n += g * weight

        return delta_n

    def save(self, path: str | Path | None = None) -> None:
        """Save the table to a compressed ``.npz`` file.

        Parameters
        ----------
        path : str or Path, optional
            Output path.  Default
            ``~/.spectroxide/photon_greens_table.npz``.

        Returns
        -------
        None
        """
        if path is None:
            path = _DEFAULT_PHOTON_CACHE
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            z_h=self.z_h,
            x=self.x,
            x_inj=self.x_inj,
            g_ph=self.g_ph,
            metadata_json=json.dumps(self.metadata),
        )

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        verify_hash: bool = True,
    ) -> "PhotonGreensTable":
        """Load a table from a ``.npz`` file.

        Parameters
        ----------
        path : str or Path, optional
            Input path.  Default
            ``~/.spectroxide/photon_greens_table.npz``.
        verify_hash : bool, optional
            If *True* (default), check the cached table's
            ``physics_hash`` against the current Rust binary; emits a
            :class:`GreensTableHashMismatch` warning on mismatch.

        Returns
        -------
        PhotonGreensTable
            Reconstructed table object.

        Warns
        -----
        GreensTableHashMismatch
            If ``verify_hash=True`` and the hashes differ.
        """
        if path is None:
            path = _DEFAULT_PHOTON_CACHE
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata_json"]))
        if verify_hash:
            _check_table_hash(metadata.get("physics_hash"), path)
        return cls(
            z_h=data["z_h"],
            x=data["x"],
            x_inj=data["x_inj"],
            g_ph=data["g_ph"],
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------


def _build_greens_table(
    z_injections: ArrayLike | None = None,
    delta_rho: float = 1.0e-5,
    n_points: int = 2000,
    x_min: float = 0.01,
    x_max: float = 30.0,
    n_x: int = 500,
    z_end: float = 0.0,
    cosmo_params: Mapping[str, float] | None = None,
    number_conserving: bool = True,
    cache_path: str | Path | None = None,
    timeout: float = 600.0,
    progress: bool = True,
    checkpoint: bool = True,
    dy_max: float | None = None,
) -> "GreensTable":
    """Build a heating Green's-function table from PDE solver runs.

    Runs the Rust PDE solver at each injection redshift (parallelised
    internally by the Rust binary) and normalises by ``Δρ/ρ`` to obtain
    ``G_th`` per unit injection.

    Supports checkpointing: splits ``z_h`` into chunks of size
    :data:`_HEATING_CHUNK_SIZE` and saves intermediate results after
    each chunk.  If interrupted, resumes from the last completed chunk
    on the next call.

    Parameters
    ----------
    z_injections : array_like, optional
        Injection redshifts.  Default *None* — uses 150 log-spaced
        points from ``1e3`` to ``5e6`` (:data:`_DEFAULT_Z_INJECTIONS`).
    delta_rho : float, optional
        Fractional energy injection per burst (default ``1e-5``).
    n_points : int, optional
        PDE grid points (default 2000).
    x_min : float, optional
        Lower edge of the output frequency grid (default 0.01).
    x_max : float, optional
        Upper edge of the output frequency grid (default 30.0).
    n_x : int, optional
        Number of output frequency points (default 500).
    z_end : float, optional
        Final redshift for PDE evolution (default 0.0).
    cosmo_params : Mapping, optional
        Cosmological parameters.  Default *None* (Rust defaults).
    number_conserving : bool, optional
        Use number-conserving mode (default *True*, matches CosmoTherm).
    cache_path : str or Path, optional
        Where to save the table.  Default
        ``~/.spectroxide/greens_table.npz``.
    timeout : float, optional
        Per-chunk timeout in seconds (default 600).
    progress : bool, optional
        Print progress messages (default *True*).
    checkpoint : bool, optional
        Enable checkpointing (default *True*).

    Returns
    -------
    GreensTable
        Newly built table, also written to ``cache_path``.
    """
    from . import _validation as _val

    if z_injections is None:
        z_injections = _DEFAULT_Z_INJECTIONS.copy()
    z_injections = np.asarray(z_injections, dtype=np.float64)
    _val.warn_table_z_density(z_injections)

    if cache_path is None:
        save_path = _DEFAULT_HEATING_CACHE
    else:
        save_path = Path(cache_path)
    save_path = Path(save_path)
    ckpt_path = _checkpoint_path(save_path)

    x_out = np.logspace(np.log10(x_min), np.log10(x_max), n_x)
    n_zh = len(z_injections)

    g_th = np.zeros((n_x, n_zh))
    mu_arr = np.zeros(n_zh)
    y_arr = np.zeros(n_zh)
    drho_arr = np.zeros(n_zh)
    completed = np.zeros(n_zh, dtype=bool)

    # Try to resume from checkpoint
    if checkpoint and ckpt_path.exists():
        try:
            ckpt = np.load(ckpt_path, allow_pickle=False)
            ckpt_zh = ckpt["z_h"]
            ckpt_completed = ckpt["completed"].astype(bool)
            # Verify grid compatibility
            if (
                len(ckpt_zh) == n_zh
                and np.allclose(ckpt_zh, z_injections, rtol=1e-12)
                and ckpt["g_th"].shape == (n_x, n_zh)
            ):
                g_th = ckpt["g_th"]
                mu_arr = ckpt["mu"]
                y_arr = ckpt["y_param"]
                drho_arr = ckpt["delta_rho_over_rho"]
                completed = ckpt_completed
                n_done = int(completed.sum())
                if progress:
                    print(f"Resuming from checkpoint: {n_done}/{n_zh} z_h complete")
            else:
                if progress:
                    print("Checkpoint grid mismatch, starting fresh")
        except Exception:
            if progress:
                print("Checkpoint corrupt, starting fresh")

    # Find indices still needed
    todo_indices = np.where(~completed)[0]
    if len(todo_indices) == 0:
        if progress:
            print("All z_h already complete in checkpoint")
    else:
        if progress:
            print(
                f"Building heating Green's function table: "
                f"{len(todo_indices)}/{n_zh} redshifts remaining..."
            )

        # Process in chunks for checkpointing
        chunk_size = _HEATING_CHUNK_SIZE
        for chunk_start in range(0, len(todo_indices), chunk_size):
            chunk_idx = todo_indices[chunk_start : chunk_start + chunk_size]
            chunk_zh = z_injections[chunk_idx]

            data = run_sweep(
                delta_rho=delta_rho,
                z_injections=chunk_zh.tolist(),
                z_end=z_end,
                cosmo_params=cosmo_params,
                n_points=n_points,
                number_conserving=number_conserving,
                timeout=timeout,
                dy_max=dy_max,
            )

            results = data["results"]
            if len(results) != len(chunk_idx):
                raise RuntimeError(
                    f"Expected {len(chunk_idx)} results, got {len(results)}"
                )

            for local_j, r in enumerate(results):
                j = chunk_idx[local_j]
                x_pde = np.asarray(r["x"])
                dn_pde = np.asarray(r["delta_n"])
                drho_actual = r.get("drho", delta_rho)

                if abs(drho_actual) > 1e-30:
                    scale = 1.0 / drho_actual
                else:
                    scale = 1.0 / delta_rho

                g_th[:, j] = np.interp(x_out, x_pde, dn_pde * scale)
                mu_arr[j] = r.get("pde_mu", 0.0) * scale
                y_arr[j] = r.get("pde_y", 0.0) * scale
                drho_arr[j] = drho_actual
                completed[j] = True

            if progress:
                n_done = int(completed.sum())
                print(f"  {n_done}/{n_zh} redshifts complete")

            # Save checkpoint
            if checkpoint:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    ckpt_path,
                    z_h=z_injections,
                    x=x_out,
                    g_th=g_th,
                    mu=mu_arr,
                    y_param=y_arr,
                    delta_rho_over_rho=drho_arr,
                    completed=completed,
                )

    if progress:
        print("Done.")

    metadata = {
        "delta_rho": delta_rho,
        "n_points": n_points,
        "z_end": z_end,
        "number_conserving": number_conserving,
        "n_zh": n_zh,
        "n_x": n_x,
        "x_min": x_min,
        "x_max": x_max,
        "physics_hash": get_physics_hash(),
    }
    if cosmo_params is not None:
        metadata["cosmo_params"] = cosmo_params

    table = GreensTable(
        z_h=z_injections,
        x=x_out,
        g_th=g_th,
        mu=mu_arr,
        y_param=y_arr,
        delta_rho_over_rho=drho_arr,
        metadata=metadata,
    )

    table.save(save_path)

    # Clean up checkpoint
    if checkpoint and ckpt_path.exists():
        ckpt_path.unlink()

    return table


def _build_photon_greens_table(
    x_inj_values: ArrayLike | None = None,
    z_injections: ArrayLike | None = None,
    delta_n_over_n: float = 1.0e-5,
    n_points: int = 2000,
    x_min: float = 0.01,
    x_max: float = 30.0,
    n_x: int = 500,
    z_end: float = 0.0,
    cosmo_params: Mapping[str, float] | None = None,
    number_conserving: bool = True,
    cache_path: str | Path | None = None,
    timeout: float = 600.0,
    progress: bool = True,
    checkpoint: bool = True,
) -> "PhotonGreensTable":
    """Build a photon-injection Green's-function table.

    Runs the PDE solver for each ``(x_inj, z_h)`` pair: ``N_xinj × N_zh``
    PDE runs total, which can be slow for large grids.

    Supports checkpointing: saves intermediate results after each
    ``x_inj`` value completes (all ``z_h`` for that ``x_inj``).  If
    interrupted, resumes from the last completed ``x_inj`` on the next
    call.

    Parameters
    ----------
    x_inj_values : array_like, optional
        Injection frequencies.  Default *None* — 10 log-spaced points
        from 0.1 to 20 (:data:`_DEFAULT_PHOTON_X_INJ`).
    z_injections : array_like, optional
        Injection redshifts.  Default *None* — 150 log-spaced points
        from ``1e3`` to ``5e6``.
    delta_n_over_n : float, optional
        Photon-number injection fraction ``ΔN/N`` (default ``1e-5``).
    n_points : int, optional
        PDE grid points (default 2000).
    x_min : float, optional
        Lower edge of the output frequency grid (default 0.01).
    x_max : float, optional
        Upper edge of the output frequency grid (default 30.0).
    n_x : int, optional
        Number of output frequency points (default 500).
    z_end : float, optional
        Final redshift for PDE evolution (default 0.0).
    cosmo_params : Mapping, optional
        Cosmological parameters.
    number_conserving : bool, optional
        Use number-conserving mode (default *True*).
    cache_path : str or Path, optional
        Where to save the table.  Default
        ``~/.spectroxide/photon_greens_table.npz``.
    timeout : float, optional
        Per-``x_inj`` timeout in seconds (default 600).
    progress : bool, optional
        Print progress messages (default *True*).
    checkpoint : bool, optional
        Enable checkpointing (default *True*).

    Returns
    -------
    PhotonGreensTable
        Newly built table, also written to ``cache_path``.
    """
    if x_inj_values is None:
        x_inj_values = _DEFAULT_PHOTON_X_INJ.copy()
    if z_injections is None:
        z_injections = _DEFAULT_Z_INJECTIONS.copy()
    x_inj_values = np.asarray(x_inj_values, dtype=np.float64)
    z_injections = np.asarray(z_injections, dtype=np.float64)

    if cache_path is None:
        save_path = _DEFAULT_PHOTON_CACHE
    else:
        save_path = Path(cache_path)
    save_path = Path(save_path)
    ckpt_path = _checkpoint_path(save_path)

    x_out = np.logspace(np.log10(x_min), np.log10(x_max), n_x)
    n_xinj = len(x_inj_values)
    n_zh = len(z_injections)

    g_ph = np.zeros((n_x, n_xinj, n_zh))
    completed = np.zeros(n_xinj, dtype=bool)
    total = n_xinj * n_zh

    # Try to resume from checkpoint
    if checkpoint and ckpt_path.exists():
        try:
            ckpt = np.load(ckpt_path, allow_pickle=False)
            ckpt_xinj = ckpt["x_inj"]
            ckpt_completed = ckpt["completed"].astype(bool)
            if (
                len(ckpt_xinj) == n_xinj
                and np.allclose(ckpt_xinj, x_inj_values, rtol=1e-12)
                and ckpt["g_ph"].shape == (n_x, n_xinj, n_zh)
            ):
                g_ph = ckpt["g_ph"]
                completed = ckpt_completed
                n_done = int(completed.sum())
                if progress:
                    print(
                        f"Resuming from checkpoint: {n_done}/{n_xinj} x_inj "
                        f"complete ({n_done * n_zh}/{total} PDE runs)"
                    )
            else:
                if progress:
                    print("Checkpoint grid mismatch, starting fresh")
        except Exception:
            if progress:
                print("Checkpoint corrupt, starting fresh")

    todo_indices = np.where(~completed)[0]
    if len(todo_indices) == 0:
        if progress:
            print("All x_inj already complete in checkpoint")
    else:
        if progress:
            n_remaining = len(todo_indices) * n_zh
            print(
                f"Building photon injection Green's function table: "
                f"{len(todo_indices)}/{n_xinj} x_inj remaining "
                f"({n_remaining}/{total} PDE runs)..."
            )

        scale = 1.0 / delta_n_over_n

        for k in todo_indices:
            xi = float(x_inj_values[k])
            data = run_photon_sweep(
                x_inj=xi,
                delta_n_over_n=float(delta_n_over_n),
                z_injections=z_injections.tolist(),
                z_end=z_end,
                cosmo_params=cosmo_params,
                n_points=n_points,
                number_conserving=number_conserving,
                timeout=timeout,
            )

            results = data["results"]
            for j, r in enumerate(results):
                x_pde = np.asarray(r["x"])
                dn_pde = np.asarray(r["delta_n"])
                g_ph[:, k, j] = np.interp(x_out, x_pde, dn_pde * scale)

            completed[k] = True
            n_done = int(completed.sum())

            if progress:
                print(f"  {n_done}/{n_xinj} x_inj complete " f"(x_inj={xi:.3e})")

            # Save checkpoint after each x_inj
            if checkpoint:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    ckpt_path,
                    z_h=z_injections,
                    x=x_out,
                    x_inj=x_inj_values,
                    g_ph=g_ph,
                    completed=completed,
                )

    if progress:
        print("Done.")

    metadata = {
        "delta_n_over_n": delta_n_over_n,
        "n_points": n_points,
        "z_end": z_end,
        "number_conserving": number_conserving,
        "n_xinj": n_xinj,
        "n_zh": n_zh,
        "n_x": n_x,
        "x_min": x_min,
        "x_max": x_max,
        "physics_hash": get_physics_hash(),
    }
    if cosmo_params is not None:
        metadata["cosmo_params"] = cosmo_params

    table = PhotonGreensTable(
        z_h=z_injections,
        x=x_out,
        x_inj=x_inj_values,
        g_ph=g_ph,
        metadata=metadata,
    )

    table.save(save_path)

    # Clean up checkpoint
    if checkpoint and ckpt_path.exists():
        ckpt_path.unlink()

    return table


# ---------------------------------------------------------------------------
# Load-or-build convenience functions
# ---------------------------------------------------------------------------


def load_or_build_greens_table(
    cache_path: str | Path | None = None,
    rebuild: bool = False,
    verify_hash: bool = True,
    **kwargs: Any,
) -> "GreensTable":
    """Load a cached heating table, or build one if not found.

    Parameters
    ----------
    cache_path : str or Path, optional
        Cache file path.  Default
        ``~/.spectroxide/greens_table.npz``.
    rebuild : bool, optional
        Force rebuild even if the cache exists (default *False*).
    verify_hash : bool, optional
        If *True* (default), verify the cached table's ``physics_hash``
        against the current Rust binary; emits a
        :class:`GreensTableHashMismatch` warning on mismatch.  Pass
        ``rebuild=True`` to regenerate after a code change.
    **kwargs
        Forwarded to the private ``_build_greens_table`` builder when a
        new table needs to be generated (e.g. ``z_h_grid``, ``n_threads``).

    Returns
    -------
    GreensTable
        Loaded or newly built table.

    Warns
    -----
    GreensTableHashMismatch
        If ``verify_hash=True`` and the cached hash differs.
    """
    if cache_path is None:
        cache_path = _DEFAULT_HEATING_CACHE
    cache_path = Path(cache_path)

    if not rebuild and cache_path.exists():
        try:
            return GreensTable.load(cache_path, verify_hash=verify_hash)
        except Exception:
            pass  # rebuild on load failure (corrupt file, schema change, etc.)

    return _build_greens_table(cache_path=cache_path, **kwargs)


def load_or_build_photon_greens_table(
    cache_path: str | Path | None = None,
    rebuild: bool = False,
    verify_hash: bool = True,
    **kwargs: Any,
) -> "PhotonGreensTable":
    """Load a cached photon table, or build one if not found.

    Parameters
    ----------
    cache_path : str or Path, optional
        Cache file path.  Default
        ``~/.spectroxide/photon_greens_table.npz``.
    rebuild : bool, optional
        Force rebuild even if the cache exists (default *False*).
    verify_hash : bool, optional
        If *True* (default), verify the cached table's ``physics_hash``
        against the current Rust binary; emits a
        :class:`GreensTableHashMismatch` warning on mismatch.
    **kwargs
        Forwarded to the private ``_build_photon_greens_table`` builder
        when a new table needs to be generated (e.g. ``x_inj_grid``,
        ``z_h_grid``, ``n_threads``).

    Returns
    -------
    PhotonGreensTable
        Loaded or newly built table.

    Warns
    -----
    GreensTableHashMismatch
        If ``verify_hash=True`` and the cached hash differs.
    """
    if cache_path is None:
        cache_path = _DEFAULT_PHOTON_CACHE
    cache_path = Path(cache_path)

    if not rebuild and cache_path.exists():
        try:
            return PhotonGreensTable.load(cache_path, verify_hash=verify_hash)
        except Exception:
            pass

    return _build_photon_greens_table(cache_path=cache_path, **kwargs)
