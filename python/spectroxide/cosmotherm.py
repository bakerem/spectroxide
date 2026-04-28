"""
CosmoTherm reference-data loader.

.. warning::

   **Development / cross-validation use only.** This module is a thin
   wrapper around output files from Jens Chluba's CosmoTherm code. It
   exists to support internal cross-checks against published reference
   data and is **not** part of the public spectroxide API. Loaders,
   conventions, and file paths can change without notice. End users
   should use :mod:`spectroxide.solver` (PDE),
   :mod:`spectroxide.greens` (analytic Green's function), or
   :mod:`spectroxide.greens_table` (precomputed PDE Green's function)
   instead.

Loads and parses data files from Jens Chluba's CosmoTherm code:

- **DI files** — predicted ΛCDM spectral distortions (ASCII, two columns).
- **Green's function database** — precomputed exact GF (large
  ASCII table).

References
----------
- Chluba & Sunyaev (2012), MNRAS 419, 1294.
- Chluba (2013), MNRAS 436, 2232.
- Chluba (2016), MNRAS 460, 227.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Physical constants for unit conversion
K_BOLTZMANN = 1.380_649e-23  # J/K
HPLANCK = 6.626_070_15e-34  # J·s
C_LIGHT = 2.997_924_58e8  # m/s
T_CMB_DEFAULT = 2.726  # K

# Location of bundled data files
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cosmotherm"


def load_di_file(
    path: str | Path | None = None,
    name: str | None = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load a CosmoTherm DI (distortion intensity) file.

    Parameters
    ----------
    path : str or Path, optional
        Full path to the DI file.  Either ``path`` or ``name`` must be
        provided.
    name : str, optional
        Filename (e.g. ``"DI_damping.dat"``) looked up in the bundled
        ``data/cosmotherm/`` directory.

    Returns
    -------
    nu_ghz : ndarray of float64
        Frequency in **GHz**.
    di_jy : ndarray of float64
        Spectral distortion intensity in **Jy/sr**.

    Raises
    ------
    ValueError
        If both ``path`` and ``name`` are *None*.
    FileNotFoundError
        If the resolved path does not exist.
    """
    if path is None:
        if name is None:
            raise ValueError("Provide either path or name")
        path = _DATA_DIR / name
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DI file not found: {path}")

    data = np.loadtxt(path, comments="#", usecols=(0, 1))
    return data[:, 0], data[:, 1]


def di_to_delta_n(
    nu_ghz: ArrayLike,
    di_jy: ArrayLike,
    t_cmb: float = T_CMB_DEFAULT,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert distortion intensity ``ΔI`` to occupation number ``Δn(x)``.

    Inverts ``ΔI = (2 h ν³ / c²) Δn``, with
    ``x = h ν / (k_B T_CMB)``.

    Parameters
    ----------
    nu_ghz : array_like
        Frequency in **GHz**.
    di_jy : array_like
        Distortion intensity in **Jy/sr** (1 Jy = 10⁻²⁶ W m⁻² Hz⁻¹).
    t_cmb : float, optional
        CMB temperature today, in **K**.  Default 2.726.

    Returns
    -------
    x : ndarray of float64
        Dimensionless frequency ``h ν / (k_B T_CMB)``.
    delta_n : ndarray of float64
        Occupation-number distortion ``Δn(x)``.
    """
    nu_ghz = np.asarray(nu_ghz, dtype=np.float64)
    di_jy = np.asarray(di_jy, dtype=np.float64)

    nu_hz = nu_ghz * 1e9
    x = HPLANCK * nu_hz / (K_BOLTZMANN * t_cmb)

    # DI [Jy/sr] → DI [W/m²/Hz/sr]
    di_si = di_jy * 1e-26

    # Dn = DI × c² / (2 h nu³)
    delta_n = di_si * C_LIGHT**2 / (2.0 * HPLANCK * nu_hz**3)

    return x, delta_n


def load_greens_database(path=None, include_metadata=False):
    """Load the CosmoTherm precomputed Green's function database.

    The file ``Greens_data.dat`` has a 7-line header followed by data rows.
    Each row has: x, G(x, z_h1), G(x, z_h2), ..., G(x, z_hN).
    The injection redshifts z_h are listed in the header.

    .. note::

        The database stores Green's function entries WITHOUT the G_bb
        temperature shift component (``add_G_term`` is disabled in
        Greens.cpp). The temperature shift is tracked separately via
        the Tgin/Tglast metadata in the header. Use ``include_metadata=True``
        and :func:`reconstruct_full_gf` to get the full GF including
        the temperature shift.

        CosmoTherm also applies ``exp(-(z/2e6)^{5/2})`` analytically
        during convolution (Greens.cpp lines 499, 507, 847).

    Parameters
    ----------
    path : str or Path, optional
        Path to ``Greens_data.dat``. If None, looks in the bundled
        ``data/cosmotherm/`` directory.
    include_metadata : bool, optional
        If True, return a 4th element with header metadata (Tgin, Tglast,
        rho arrays). Default False for backward compatibility.

    Returns
    -------
    z_h : ndarray, shape (N_z,)
        Injection redshifts.
    x : ndarray, shape (N_x,)
        Dimensionless frequencies.
    g_th : ndarray, shape (N_x, N_z)
        Green's function G_th(x, z_h) — the residual mu+y spectral
        distortion, WITHOUT the G_bb temperature shift.
    metadata : dict, optional
        Only returned if ``include_metadata=True``. Contains:

        - ``tgin``: ndarray, shape (N_z,) — initial blackbody temperature [K]
        - ``tglast``: ndarray, shape (N_z,) — last blackbody temperature [K]
          (diverges from tgin at high z where thermalization occurs)
        - ``rho``: ndarray, shape (N_z,) — Delta-rho/rho used for each entry
    """
    if path is None:
        path = _DATA_DIR / "Greens_data.dat"
        if not path.exists():
            # Search nested directory (e.g. Greens.v1.0.3/Gdatabase/)
            for candidate in _DATA_DIR.glob("Greens*/Gdatabase/Greens_data.dat"):
                path = candidate
                break
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Green's function database not found: {path}\n"
            "Download it with: data/cosmotherm/download_greens.sh"
        )

    with open(path) as f:
        lines = f.readlines()

    # Parse header: first 7 lines contain metadata including z_h values
    # Format: comment, nG, z_h values, Tgin values, Tglast values, rho values, empty
    header_lines = []
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or stripped == "":
            header_lines.append(stripped)
            data_start = i + 1
        else:
            break

    # Parse header: line with single integer is nG (number of redshifts).
    # Lines with many numbers are (in order): z_h, Tgin, Tglast, rho.
    z_h = None
    tgin = None
    tglast = None
    rho = None
    n_g = None
    numeric_arrays = []
    for hl in header_lines:
        hl_clean = hl.lstrip("#").strip()
        parts = hl_clean.split()
        if len(parts) == 1:
            try:
                n_g = int(parts[0])
            except ValueError:
                pass
        elif len(parts) > 10:
            try:
                arr = np.array([float(p) for p in parts])
                numeric_arrays.append(arr)
            except ValueError:
                continue

    # Assign arrays in order: z_h, Tgin, Tglast, rho
    if len(numeric_arrays) >= 1:
        z_h = numeric_arrays[0]
    if len(numeric_arrays) >= 2:
        tgin = numeric_arrays[1]
    if len(numeric_arrays) >= 3:
        tglast = numeric_arrays[2]
    if len(numeric_arrays) >= 4:
        rho = numeric_arrays[3]

    # Parse data rows.  CosmoTherm format: x G(x,z1) ... G(x,zN) trailing
    # The trailing value (1 column) is written but not used by CosmoTherm's
    # own reader (Greens.cpp line 742: ``ifile >> dum;``).
    n_z = n_g if n_g is not None else (len(z_h) if z_h is not None else None)
    x_vals = []
    g_rows = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        x_vals.append(float(parts[0]))
        if n_z is not None:
            g_rows.append([float(p) for p in parts[1 : 1 + n_z]])
        else:
            g_rows.append([float(p) for p in parts[1:]])

    x = np.array(x_vals)
    g_th = np.array(g_rows)

    if z_h is None:
        n_z_cols = g_th.shape[1]
        z_h = np.logspace(3, np.log10(3e6), n_z_cols)
        import warnings

        warnings.warn(
            f"Could not parse z_h from header; using {n_z_cols} default values"
        )

    if include_metadata:
        metadata = {
            "tgin": tgin,
            "tglast": tglast,
            "rho": rho,
        }
        return z_h, x, g_th, metadata

    return z_h, x, g_th


def _compute_g_bb_jy(x, t_cmb=T_CMB_DEFAULT):
    """Compute G_bb(x) = fac * x^4 * exp(-x) / (1-exp(-x))^2 in Jy/sr.

    This matches CosmoTherm's ``compute_G()`` function (Greens.cpp).
    """
    h_over_kb = HPLANCK / K_BOLTZMANN
    fac = (
        2.0
        * HPLANCK
        / C_LIGHT**2
        / (h_over_kb / t_cmb) ** 3
        * 1.0e26  # W/m^2/Hz/sr -> Jy/sr
    )
    ex = np.exp(-x)
    return fac * x**4 * ex / (1.0 - ex) ** 2


def reconstruct_full_gf(x, g_th, z_h, metadata, apply_exp=True, t_cmb=T_CMB_DEFAULT):
    """Add the G_bb temperature shift back to CosmoTherm GF entries.

    CosmoTherm's ``Greens_data.dat`` stores GF entries WITHOUT the G_bb
    temperature shift component (``add_G_term`` is disabled by default in
    Greens.cpp). This function reconstructs the full *physical* GF by:

    1. Optionally applying ``exp(-(z/2e6)^{5/2})`` to the stored mu+y part
       (CosmoTherm applies this during convolution, not during storage).
    2. Adding back the G_bb temperature shift from the Tgin/Tglast metadata.

    The result follows CosmoTherm's ``output_Greens_function``
    (Greens.cpp lines 871-884) with ``add_G_term`` enabled::

        G_physical(x, z_h) = G_stored(x, z_h) * exp(-(z_h/2e6)^{5/2})
                            + G_bb(x) * (Tglast - Tgin) / Tgin / Drho

    Note: the exp factor suppresses the mu+y residual at high z, but the
    G_bb temperature shift is NOT suppressed (it is physical).

    Parameters
    ----------
    x : ndarray, shape (N_x,)
        Dimensionless frequencies from the database.
    g_th : ndarray, shape (N_x, N_z)
        Stored GF entries (mu+y only, no temperature shift, no exp factor).
    z_h : ndarray, shape (N_z,)
        Injection redshifts.
    metadata : dict
        From ``load_greens_database(..., include_metadata=True)``.
        Must contain ``tgin``, ``tglast``, and ``rho`` arrays.
    apply_exp : bool, optional
        If True (default), apply ``exp(-(z_h/2e6)^{5/2})`` to the stored
        mu+y part before adding the temperature shift. This gives the
        physical spectral response. Set False to get the raw sum without
        exp suppression.
    t_cmb : float, optional
        CMB temperature [K] (default 2.726).

    Returns
    -------
    g_th_full : ndarray, shape (N_x, N_z)
        Full GF including the G_bb temperature shift component.
    """
    tgin = metadata["tgin"]
    tglast = metadata["tglast"]
    rho = metadata["rho"]

    if tgin is None or tglast is None or rho is None:
        raise ValueError(
            "Metadata missing Tgin/Tglast/rho arrays. "
            "Ensure the Greens_data.dat file has the full header."
        )

    g_bb = _compute_g_bb_jy(x, t_cmb)  # shape (N_x,)

    # Temperature shift amplitude per unit Drho/rho:
    # alpha(z_h) = (Tglast - Tgin) / Tgin / Drho
    dt_over_t = (tglast - tgin) / tgin  # shape (N_z,)
    alpha = dt_over_t / rho  # shape (N_z,)

    # Apply exp suppression to the stored mu+y part
    if apply_exp:
        exp_factor = np.exp(-((z_h / 2.0e6) ** 2.5))  # shape (N_z,)
        g_muy = g_th * exp_factor[np.newaxis, :]
    else:
        g_muy = g_th.copy()

    # G_full = G_muy + G_bb * alpha
    g_th_full = g_muy + g_bb[:, np.newaxis] * alpha[np.newaxis, :]

    return g_th_full


def cosmotherm_gf_to_delta_n(x, gf_jy, t_cmb=T_CMB_DEFAULT):
    """Convert CosmoTherm GF database values (Jy/sr) to occupation number Δn.

    The CosmoTherm GF database stores G_th in units of ΔI [Jy/sr] per Δρ/ρ.
    Our analytical Green's function returns Δn (dimensionless) per Δρ/ρ.
    This function converts: Δn = ΔI / (2hν³/c² × 1e26).

    Parameters
    ----------
    x : array_like
        Dimensionless frequency h*nu / (k*T_CMB).
    gf_jy : array_like
        CosmoTherm GF values in Jy/sr per Δρ/ρ.
    t_cmb : float, optional
        CMB temperature [K] (default 2.726).

    Returns
    -------
    ndarray
        Green's function in Δn per Δρ/ρ.
    """
    x = np.asarray(x, dtype=np.float64)
    gf_jy = np.asarray(gf_jy, dtype=np.float64)
    nu_hz = x * K_BOLTZMANN * t_cmb / HPLANCK
    # 2hν³/c² in Jy/sr per unit Δn
    conversion = 2.0 * HPLANCK * nu_hz**3 / C_LIGHT**2 / 1e-26
    return gf_jy / conversion


# ---------------------------------------------------------------------------
# CosmoTherm GF convolution for continuous injection
# ---------------------------------------------------------------------------

# eV -> Joule conversion
_EV_SI = 1.602_176_634e-19  # J per eV


def convolve_cosmotherm_gf(
    z_h, x, g_th, dq_dz, z_min=1001.0, z_max=4.995e6, n_z=2000, x_out=None
):
    """Convolve CosmoTherm GF database with a heating rate to get ΔI(x).

    Computes::

        ΔI(x) = ∫ G_stored(x, z) × exp(-(z/2e6)^{5/2}) × S(z) d(ln(1+z))

    where ``S(z) = d(Δρ/ρ)/dz`` is the heating rate and the exp factor
    is applied analytically (matching CosmoTherm's Greens.cpp).

    Parameters
    ----------
    z_h : ndarray, shape (N_z,)
        Injection redshifts from the database.
    x : ndarray, shape (N_x,)
        Dimensionless frequencies from the database.
    g_th : ndarray, shape (N_x, N_z)
        Stored GF entries (mu+y only, no G_bb, no exp factor).
    dq_dz : callable
        Heating rate ``d(Δρ/ρ)/dz`` as a function of z.
    z_min : float, optional
        Lower integration limit (default 1001).
    z_max : float, optional
        Upper integration limit (default 4.995e6).
    n_z : int, optional
        Number of integration points in ln(1+z) (default 2000).
    x_out : ndarray, optional
        Output frequency grid. If None, uses the database grid ``x``.

    Returns
    -------
    x_out : ndarray
        Frequency grid.
    di_jy : ndarray
        Spectral distortion ΔI [Jy/sr].
    """
    from scipy.interpolate import RegularGridInterpolator

    if x_out is None:
        x_out = x.copy()

    # Build 2D interpolator in log-log space: axes = (log10(x), log10(z_h))
    log_x = np.log10(x)
    log_zh = np.log10(z_h)
    interp = RegularGridInterpolator(
        (log_x, log_zh),
        g_th,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # Integration grid uniform in ln(1+z)
    ln_min = np.log(1.0 + z_min)
    ln_max = np.log(1.0 + z_max)
    ln_grid = np.linspace(ln_min, ln_max, n_z)
    dln = ln_grid[1] - ln_grid[0]

    log_xq = np.log10(np.clip(x_out, x.min(), x.max()))
    n_x = len(x_out)

    # Precompute z grid and heating rates
    z_grid = np.exp(ln_grid) - 1.0
    dz_dln = 1.0 + z_grid

    # Try vectorized call to dq_dz, fall back to scalar loop
    try:
        heating_arr = np.asarray(dq_dz(z_grid), dtype=np.float64)
        if heating_arr.shape != z_grid.shape:
            raise ValueError
    except (TypeError, ValueError):
        heating_arr = np.array([dq_dz(float(zj)) for zj in z_grid], dtype=np.float64)

    heating_arr = heating_arr * dz_dln

    # Trapezoidal weights
    w = np.full(n_z, dln)
    w[0] = 0.5 * dln
    w[-1] = 0.5 * dln

    # exp suppression
    exp_arr = np.exp(-((z_grid / 2.0e6) ** 2.5))

    hw = heating_arr * w * exp_arr  # shape (n_z,)

    # Filter out negligible contributions
    active = np.abs(hw) >= 1e-50
    if not np.any(active):
        return x_out, np.zeros(n_x)

    hw_active = hw[active]
    z_active = z_grid[active]
    n_active = int(np.sum(active))

    # Build all query points at once: shape (n_active * n_x, 2)
    log_z_active = np.log10(np.maximum(z_active, z_h.min()))
    # Tile x for each active z, tile z for each x
    log_xq_tiled = np.tile(log_xq, n_active)  # shape (n_active * n_x,)
    log_z_tiled = np.repeat(log_z_active, n_x)  # shape (n_active * n_x,)
    pts = np.column_stack([log_xq_tiled, log_z_tiled])  # shape (n_active * n_x, 2)

    gf_all = interp(pts).reshape(n_active, n_x)  # shape (n_active, n_x)

    # Weighted sum: di = sum_j gf_all[j, :] * hw_active[j]
    di = np.dot(hw_active, gf_all)  # shape (n_x,)

    return x_out, di


def strip_gbb(x: ArrayLike, delta_n: ArrayLike) -> Tuple[NDArray[np.float64], float]:
    """Remove the unobservable temperature-shift component of a spectrum.

    FIRAS measures the CMB spectrum with the absolute temperature as a
    free parameter, so a uniform shift ΔT/T is unobservable.  CosmoTherm
    therefore defines the *distortion* as the number-conserving part of
    ``Δn`` (Chluba & Sunyaev 2012, arXiv:1109.6552): the part satisfying
    ``∫ x² Δn dx = 0``.  Any nonzero photon-number perturbation is
    absorbed into ``α · G_bb(x)``.

    This projection is **orthogonal** to ``μ`` and ``y`` because both
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
    from .greens import g_bb

    gbb = g_bb(x)

    alpha = float(_trapz(x**2 * delta_n, x) / _trapz(x**2 * gbb, x))

    return delta_n - alpha * gbb, alpha


# ---------------------------------------------------------------------------
# CosmoTherm-convention DM heating rates
# ---------------------------------------------------------------------------


def _get_cosmotherm_cosmo(cosmo):
    """Return cosmology dict, defaulting to COSMOTHERM_GF_COSMO."""
    if cosmo is not None:
        return cosmo
    # Lazy import to avoid circular dependency
    from .greens import COSMOTHERM_GF_COSMO

    return COSMOTHERM_GF_COSMO


def ct_heating_rate_swave(
    z: ArrayLike,
    f_ann_CT: float,
    cosmo: Any = None,
) -> NDArray[np.float64]:
    """Heating rate ``d(Δρ/ρ)/dz`` for s-wave DM annihilation.

    Uses the paper/CosmoTherm convention with ``f_ann_CT`` in **eV/s**:

    .. math::

        dQ/dV/dt = f_{ann,CT} \\, n_H(z) \\, (1+z)^3.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    f_ann_CT : float
        Annihilation parameter in **eV/s**.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to
        :data:`spectroxide.greens.COSMOTHERM_GF_COSMO`.

    Returns
    -------
    float or ndarray of float64
        ``d(Δρ/ρ)/dz`` (positive for heating, dimensionless per unit
        redshift).
    """
    from .greens import _cosmo_hubble, _cosmo_n_h, rho_gamma

    cosmo = _get_cosmotherm_cosmo(cosmo)
    n_h = _cosmo_n_h(z, cosmo)
    rho_g = rho_gamma(z, cosmo)
    hz = _cosmo_hubble(z, cosmo)

    # dQ/dt = f_ann_CT [eV/s] × n_H(z) × (1+z)³  [eV/m³/s]
    dq_dt = f_ann_CT * _EV_SI * n_h * (1.0 + z) ** 3  # W/m^3
    return dq_dt / (rho_g * hz * (1.0 + z))


def ct_heating_rate_pwave(
    z: ArrayLike,
    f_ann_CT: float,
    cosmo: Any = None,
) -> NDArray[np.float64]:
    """Heating rate ``d(Δρ/ρ)/dz`` for p-wave DM annihilation.

    Same as :func:`ct_heating_rate_swave` but with an extra ``(1+z)``
    factor from ``⟨σv⟩ ∝ v² ∝ T ∝ (1+z)``:

    .. math::

        dQ/dV/dt = f_{ann,CT} \\, n_H(z) \\, (1+z)^4.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    f_ann_CT : float
        Annihilation parameter in **eV/s**.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to
        :data:`spectroxide.greens.COSMOTHERM_GF_COSMO`.

    Returns
    -------
    float or ndarray of float64
        ``d(Δρ/ρ)/dz``.
    """
    from .greens import _cosmo_hubble, _cosmo_n_h, rho_gamma

    cosmo = _get_cosmotherm_cosmo(cosmo)
    n_h = _cosmo_n_h(z, cosmo)
    rho_g = rho_gamma(z, cosmo)
    hz = _cosmo_hubble(z, cosmo)

    # p-wave: extra (1+z) factor from ⟨σv⟩ ∝ (1+z)
    # dQ/dt = f_ann_CT [eV/s] × n_H(z) × (1+z)⁴  [eV/m³/s]
    dq_dt = f_ann_CT * _EV_SI * n_h * (1.0 + z) ** 4
    return dq_dt / (rho_g * hz * (1.0 + z))


def ct_heating_rate_decay(
    z: ArrayLike,
    f_x_eV: float,
    gamma_x: float,
    cosmo: Any = None,
) -> NDArray[np.float64]:
    """Heating rate ``d(Δρ/ρ)/dz`` for decaying particle.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    f_x_eV : float
        Energy injection parameter ``f_X`` in **eV**.
    gamma_x : float
        Decay rate ``Γ_X`` in **1/s**.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to
        :data:`spectroxide.greens.COSMOTHERM_GF_COSMO`.

    Returns
    -------
    float or ndarray of float64
        ``d(Δρ/ρ)/dz`` (positive for heating).
    """
    from .greens import _cosmo_hubble, _cosmo_n_h, rho_gamma, cosmic_time

    cosmo = _get_cosmotherm_cosmo(cosmo)
    n_h = _cosmo_n_h(z, cosmo)
    rho_g = rho_gamma(z, cosmo)
    hz = _cosmo_hubble(z, cosmo)

    t = cosmic_time(z, cosmo)
    # dQ/dt = f_X × Γ_X × n_H(z) × exp(-Γ_X × t) [eV/m^3/s]
    dq_dt = f_x_eV * _EV_SI * gamma_x * n_h * np.exp(-gamma_x * t)
    return dq_dt / (rho_g * hz * (1.0 + z))


def cosmotherm_gf_distortion(
    scenario,
    params,
    z_h=None,
    x=None,
    g_th=None,
    z_min=1001.0,
    z_max=4.995e6,
    n_z=2000,
    x_out=None,
    cosmo=None,
    gf_path=None,
):
    """Convenience wrapper: load GF database + convolve for a DM scenario.

    Parameters
    ----------
    scenario : str
        One of ``"swave"``, ``"pwave"``, ``"decay"``.
    params : dict
        Scenario parameters:

        - ``"swave"``/``"pwave"``: ``{"f_ann_CT": float}``
        - ``"decay"``: ``{"f_x_eV": float, "gamma_x": float}``

    z_h, x, g_th : ndarray, optional
        Pre-loaded database arrays. If None, loads from ``gf_path``.
    z_min, z_max, n_z : float/int, optional
        Integration range and resolution.
    x_out : ndarray, optional
        Output frequency grid.
    cosmo : dict, optional
        Cosmological parameters.
    gf_path : str or Path, optional
        Path to ``Greens_data.dat``.

    Returns
    -------
    x_out : ndarray
        Dimensionless frequency grid.
    nu_ghz : ndarray
        Frequency in GHz.
    di_jy : ndarray
        Spectral distortion ΔI [Jy/sr].
    """
    # Load database if not provided
    if z_h is None or x is None or g_th is None:
        z_h, x, g_th = load_greens_database(path=gf_path)

    # Select heating rate function
    if scenario == "swave":
        dq_dz = lambda z: ct_heating_rate_swave(z, params["f_ann_CT"], cosmo)
    elif scenario == "pwave":
        dq_dz = lambda z: ct_heating_rate_pwave(z, params["f_ann_CT"], cosmo)
    elif scenario == "decay":
        dq_dz = lambda z: ct_heating_rate_decay(
            z, params["f_x_eV"], params["gamma_x"], cosmo
        )
    else:
        raise ValueError(
            f"Unknown scenario: {scenario!r}. "
            f"Expected 'swave', 'pwave', or 'decay'."
        )

    x_out, di_jy = convolve_cosmotherm_gf(
        z_h,
        x,
        g_th,
        dq_dz,
        z_min=z_min,
        z_max=z_max,
        n_z=n_z,
        x_out=x_out,
    )

    # Convert x to nu_ghz
    nu_hz = x_out * K_BOLTZMANN * T_CMB_DEFAULT / HPLANCK
    nu_ghz = nu_hz / 1e9

    return x_out, nu_ghz, di_jy
