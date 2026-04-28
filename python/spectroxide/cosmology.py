"""
Flat ΛCDM background quantities.

Cosmology presets, the Hubble rate, photon and baryon densities, the
Saha + Peebles three-level-atom recombination history, and helpers
derived from them (free-electron density, baryon-photon ratio, cosmic
time). The Green's-function and PDE-table modules pull from this
module rather than redefining cosmology themselves.

References
----------
- Peebles (1968), ApJ 153, 1.
- Pequignot, Petitjean & Boisson (1991), A&A 251, 680 (case-B α).
- Chluba & Thomas (2011), MNRAS 412, 748 (TLA fudge factor F = 1.125).
- Planck Collaboration (2016), A&A 594, A13 (Planck 2015 parameters).
- Planck Collaboration (2020), A&A 641, A6 (Planck 2018 VI parameters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _validation as _val

#: Type alias for a cosmology parameter mapping (or any mapping accepting
#: the required keys ``h``, ``omega_b``, ``omega_m``, ``y_p``, ``t_cmb``,
#: ``n_eff``).
CosmoLike = Mapping[str, float]

#: Type alias for a scalar or NumPy array of float64 values.
FloatOrArray = Union[float, NDArray[np.float64]]


# ---------------------------------------------------------------------------
# Cosmology dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cosmology:
    """Flat ΛCDM parameters. Densities are fractions Ω, not ωh².

    Parameters
    ----------
    h : float
        H₀ / (100 km/s/Mpc). Default 0.71.
    omega_b : float
        Ω_b. Default 0.044.
    omega_m : float
        Ω_m (total matter). Default 0.26.
    y_p : float
        Helium mass fraction. Default 0.24.
    t_cmb : float
        T_CMB today [K]. Default 2.726.
    n_eff : float
        N_eff. Default 3.046.

    Examples
    --------
    >>> Cosmology.planck2018().h
    0.6736
    """

    h: float = 0.71
    omega_b: float = 0.044
    omega_m: float = 0.26
    y_p: float = 0.24
    t_cmb: float = 2.726
    n_eff: float = 3.046

    def __post_init__(self):
        _val.validate_cosmology(self)

    @classmethod
    def default(cls) -> "Cosmology":
        """Chluba 2013 parameters (code default)."""
        return cls()

    @classmethod
    def planck2015(cls) -> "Cosmology":
        """Planck 2015 best-fit parameters."""
        return cls(
            h=0.6727,
            omega_b=0.049169,
            omega_m=0.313906,
            y_p=0.2467,
            t_cmb=2.7255,
            n_eff=3.046,
        )

    @classmethod
    def planck2018(cls) -> "Cosmology":
        """Planck 2018 best-fit parameters."""
        return cls(
            h=0.6736,
            omega_b=0.04930,
            omega_m=0.31530,
            y_p=0.2454,
            t_cmb=2.7255,
            n_eff=3.044,
        )

    def to_dict(self) -> dict:
        """Convert to a dict accepted by :func:`spectroxide.solver.solve`
        (``cosmo=``) and :func:`spectroxide.solver.run_sweep`
        (``cosmo_params=``)."""
        return {
            "h": self.h,
            "omega_b": self.omega_b,
            "omega_m": self.omega_m,
            "y_p": self.y_p,
            "t_cmb": self.t_cmb,
            "n_eff": self.n_eff,
        }


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
#
# These dicts are the canonical input format consumed by the helpers below
# (``_cosmo_hubble`` etc.) and by the Rust CLI.  Where the values match a
# :class:`Cosmology` classmethod exactly, we derive the dict from it to keep
# them in sync.  The two CosmoTherm-comparison presets intentionally differ
# from the Planck dataclass values (n_eff = 3.04 to match CosmoTherm v1.0.3;
# t_cmb = 2.726 to match the Fixsen 1996 value baked into CosmoTherm's DI
# files) so they stay as standalone dict literals.

#: Chluba (2013) Green's-function paper parameters.
DEFAULT_COSMO = Cosmology.default().to_dict()

#: Matches CosmoTherm v1.0.3 (Greens.cpp uses ``N_eff = 3.04``). Use this
#: when convolving against CosmoTherm's GF table.
COSMOTHERM_GF_COSMO = {**DEFAULT_COSMO, "n_eff": 3.04}

#: Planck 2015 (Planck XIII, Table 4). ``T_CMB`` follows CosmoTherm's DI
#: files (Fixsen 1996 = 2.726 K), not the Planck paper value 2.7255 K — use
#: this for CT comparisons and ``Cosmology.planck2015()`` otherwise.
PLANCK2015_COSMO = {**Cosmology.planck2015().to_dict(), "t_cmb": 2.726}

#: Planck 2018 VI, Table 2 (TT,TE,EE+lowE+lensing).
PLANCK2018_COSMO = Cosmology.planck2018().to_dict()


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_C_LIGHT = 2.997_924_58e8  # m/s
_K_BOLTZMANN = 1.380_649e-23  # J/K
_HBAR = 1.054_571_817e-34  # J·s
_G_NEWTON = 6.674_30e-11  # m³/(kg·s²)
_M_PROTON = 1.672_621_923_69e-27  # kg
_M_ELECTRON = 9.109_383_7015e-31  # kg
_SIGMA_THOMSON = 6.652_458_7321e-29  # m²
_KM_PER_MPC = 3.240_779_29e-20  # (km/s/Mpc) → 1/s
_MPC_M = 3.085_677_581e22  # meters per Mpc
_EV_IN_JOULES = 1.602_176_634e-19  # J
_E_RYDBERG = 13.605_693_122_994 * _EV_IN_JOULES  # Hydrogen ionization [J]
_E_ION_N2 = _E_RYDBERG / 4.0  # Ionization from n=2 [J]
_E_HE_II_ION = 54.4178 * _EV_IN_JOULES  # He II ionization [J]
_E_HE_I_ION = 24.5874 * _EV_IN_JOULES  # He I ionization [J]
_LAMBDA_LYA = 1.215_670e-7  # Lyman-alpha wavelength [m]
_LAMBDA_2S1S = 8.2245809  # Two-photon decay rate [s^-1]


# ---------------------------------------------------------------------------
# Background internals
# ---------------------------------------------------------------------------


def _cosmo_h0(cosmo):
    """H0 in 1/s."""
    return 100.0 * cosmo["h"] * _KM_PER_MPC


def _cosmo_omega_gamma(cosmo):
    """Photon density parameter."""
    rho_gamma = (
        np.pi**2
        / 15.0
        * _K_BOLTZMANN**4
        * cosmo["t_cmb"] ** 4
        / (_HBAR**3 * _C_LIGHT**3)
    )
    h0 = _cosmo_h0(cosmo)
    rho_crit = 3.0 * h0**2 * _C_LIGHT**2 / (8.0 * np.pi * _G_NEWTON)
    return rho_gamma / rho_crit


def _cosmo_omega_rel(cosmo):
    """Relativistic density parameter (photons + neutrinos)."""
    og = _cosmo_omega_gamma(cosmo)
    return og * (1.0 + cosmo["n_eff"] * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0))


def _cosmo_hubble(z, cosmo):
    """Hubble rate H(z) in 1/s."""
    h0 = _cosmo_h0(cosmo)
    om = cosmo["omega_m"]
    orel = _cosmo_omega_rel(cosmo)
    olam = 1.0 - om - orel
    opz = 1.0 + z
    return h0 * np.sqrt(om * opz**3 + orel * opz**4 + olam)


def _cosmo_n_h(z, cosmo):
    """Hydrogen number density [1/m^3]."""
    h0 = _cosmo_h0(cosmo)
    rho_crit = 3.0 * h0**2 / (8.0 * np.pi * _G_NEWTON)
    rho_b0 = cosmo["omega_b"] * rho_crit
    return (1.0 - cosmo["y_p"]) * rho_b0 * (1.0 + z) ** 3 / _M_PROTON


# ---------------------------------------------------------------------------
# Recombination (port of src/recombination.rs)
# ---------------------------------------------------------------------------


def _thermal_de_broglie(t):
    """(m_e k_B T / 2pi hbar^2)^{3/2} [m^-3]."""
    return (_M_ELECTRON * _K_BOLTZMANN * t / (2.0 * np.pi * _HBAR**2)) ** 1.5


def _solve_saha_quadratic(s):
    """Solve X^2/(1-X) = S for X (scalar)."""
    if s > 1e10:
        return 1.0
    elif s < 1e-10:
        return s**0.5
    else:
        return (-s + (s * s + 4.0 * s) ** 0.5) / 2.0


def _saha_he_ii(z, cosmo):
    """He II -> He I Saha fraction (54.4 eV). g-ratio = 1."""
    t = cosmo["t_cmb"] * (1.0 + z)
    n_he = _cosmo_n_h(z, cosmo) * cosmo["y_p"] / (4.0 * (1.0 - cosmo["y_p"]))
    if n_he < 1e-30:
        return 1.0
    s = _thermal_de_broglie(t) * np.exp(-_E_HE_II_ION / (_K_BOLTZMANN * t)) / n_he
    return _solve_saha_quadratic(s)


def _saha_he_i(z, cosmo):
    """He I -> He Saha fraction (24.6 eV). g-ratio = 4."""
    t = cosmo["t_cmb"] * (1.0 + z)
    n_he = _cosmo_n_h(z, cosmo) * cosmo["y_p"] / (4.0 * (1.0 - cosmo["y_p"]))
    if n_he < 1e-30:
        return 1.0
    s = 4.0 * _thermal_de_broglie(t) * np.exp(-_E_HE_I_ION / (_K_BOLTZMANN * t)) / n_he
    return _solve_saha_quadratic(s)


def _helium_electron_fraction(z, cosmo):
    """Free electrons per H atom from helium: f_He * (y_HeI + y_HeII)."""
    f_he = cosmo["y_p"] / (4.0 * (1.0 - cosmo["y_p"]))
    y_he_ii = _saha_he_ii(z, cosmo)
    y_he_i = _saha_he_i(z, cosmo)
    return f_he * (y_he_i + y_he_ii)


def _saha_hydrogen(z, cosmo):
    """Hydrogen Saha ionization fraction."""
    t = cosmo["t_cmb"] * (1.0 + z)
    n_h = _cosmo_n_h(z, cosmo)
    s = _thermal_de_broglie(t) * np.exp(-_E_RYDBERG / (_K_BOLTZMANN * t)) / n_h
    return _solve_saha_quadratic(s)


def _alpha_recomb(t):
    """Case-B recombination coefficient [m^3/s] (Pequignot+ 1991, F=1.125)."""
    tt = t / 1.0e4
    f = 1.125
    return f * 1e-19 * 4.309 * tt ** (-0.6166) / (1.0 + 0.6703 * tt**0.5300)


def _beta_ion(t_rad):
    """Photoionization rate from n=2 [s^-1]."""
    alpha = _alpha_recomb(t_rad)
    return (
        alpha * _thermal_de_broglie(t_rad) * np.exp(-_E_ION_N2 / (_K_BOLTZMANN * t_rad))
    )


def _peebles_c(z, x_e, cosmo):
    """Peebles C factor: fraction of excited atoms reaching ground state."""
    t_rad = cosmo["t_cmb"] * (1.0 + z)
    n_h = _cosmo_n_h(z, cosmo)
    h = _cosmo_hubble(z, cosmo)
    k_h = _LAMBDA_LYA**3 / (8.0 * np.pi * h)
    n_1s = n_h * max(1.0 - x_e, 0.0)
    rate_lya = 1.0 / (k_h * n_1s) if n_1s > 1e-30 else 1e30
    rate_ion = _beta_ion(t_rad)
    rate_down = rate_lya + _LAMBDA_2S1S
    denom = rate_down + rate_ion
    return rate_down / denom if denom > 0.0 else 1.0


def _peebles_step(z_new, x_h, dz, cosmo):
    """Single forward Euler step of the Peebles TLA ODE."""
    t = cosmo["t_cmb"] * (1.0 + z_new)
    n_h = _cosmo_n_h(z_new, cosmo)
    h = _cosmo_hubble(z_new, cosmo)
    c_r = _peebles_c(z_new, min(x_h, 1.0), cosmo)
    alpha = _alpha_recomb(t)
    x_saha = min(_saha_hydrogen(z_new, cosmo), 1.0)
    one_minus_xs = max(1.0 - x_saha, 1e-30)
    rhs_factor = c_r * alpha * n_h / (h * (1.0 + z_new))
    saha_term = x_saha**2 * max(1.0 - x_h, 0.0) / one_minus_xs
    f_val = rhs_factor * (x_h**2 - saha_term)
    return np.clip(x_h - dz * f_val, 1e-5, 1.0)


def _find_saha_switch(cosmo):
    """Find redshift where Saha H drops below 0.99."""
    z = 1800.0
    while z > 1000.0:
        if _saha_hydrogen(z, cosmo) < 0.99:
            return z + 1.0
        z -= 1.0
    return 1500.0


def _ionization_fraction_scalar(z, cosmo, z_switch, z_ode, x_h_ode):
    """Evaluate X_e at a single redshift using precomputed ODE table."""
    if z > 8000.0:
        return 1.0 + _helium_electron_fraction(z, cosmo)
    if z > z_switch:
        x_h = min(_saha_hydrogen(z, cosmo), 1.0)
        return x_h + _helium_electron_fraction(z, cosmo)
    if z < z_ode[0]:
        # Below table: freeze-out (z_ode[0] is the lowest z after ascending sort)
        return x_h_ode[0] + _helium_electron_fraction(max(z, 1.0), cosmo)
    # Interpolate (z_ode is ascending)
    x_h_interp = np.interp(z, z_ode, x_h_ode)
    return x_h_interp + _helium_electron_fraction(z, cosmo)


# Cache for recombination ODE tables, keyed by cosmology tuple
_recomb_table_cache = {}


def _cosmo_key(cosmo):
    """Hashable key for a cosmology dict (only params that affect recombination)."""
    return (cosmo["omega_b"], cosmo["h"], cosmo["y_p"], cosmo["t_cmb"])


def _get_recomb_table(cosmo):
    """Get or build cached Peebles TLA recombination table.

    Returns (z_ode, x_h_ode, z_switch) where z_ode is ascending.
    """
    key = _cosmo_key(cosmo)
    if key in _recomb_table_cache:
        return _recomb_table_cache[key]

    z_switch = _find_saha_switch(cosmo)
    z_min_ode = 1.0
    dz_step = 0.5
    n_steps = max(int(np.ceil((z_switch - z_min_ode) / dz_step)), 1)
    dz_actual = (z_switch - z_min_ode) / n_steps

    z_ode = [z_switch]
    x_h_ode = [min(_saha_hydrogen(z_switch, cosmo), 1.0)]
    x_h = x_h_ode[0]
    for i in range(n_steps):
        z_new = z_switch - (i + 1) * dz_actual
        x_h = _peebles_step(z_new, x_h, dz_actual, cosmo)
        z_ode.append(z_new)
        x_h_ode.append(x_h)

    z_ode = np.array(z_ode[::-1])
    x_h_ode = np.array(x_h_ode[::-1])
    result = (z_ode, x_h_ode, z_switch)
    if len(_recomb_table_cache) > 10:
        _recomb_table_cache.clear()
    _recomb_table_cache[key] = result
    return result


def ionization_fraction(z: ArrayLike, cosmo: CosmoLike | None = None) -> FloatOrArray:
    """Free-electron fraction ``X_e(z) = n_e / n_H``.

    Uses Saha equilibrium for helium (He II at 54.4 eV, He I at 24.6 eV)
    and the Peebles three-level atom ODE for hydrogen recombination,
    with fudge factor F = 1.125 (Chluba & Thomas 2011).  The ODE table is
    cached per cosmology for fast repeated lookups.

    Parameters
    ----------
    z : float or array_like
        Redshift(s).
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to :data:`DEFAULT_COSMO`.

    Returns
    -------
    float or ndarray of float64
        Total free-electron fraction (H + He) per hydrogen atom.  Returns
        a Python float when ``z`` is a scalar, otherwise an array shaped
        like ``z``.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    scalar = np.ndim(z) == 0

    z_ode, x_h_ode, z_switch = _get_recomb_table(cosmo)

    result = np.array(
        [
            _ionization_fraction_scalar(zi, cosmo, z_switch, z_ode, x_h_ode)
            for zi in z_arr
        ]
    )
    return float(result[0]) if scalar else result


def _cosmo_n_e(z, cosmo, x_e=None):
    """Free electron density [1/m^3].

    Uses the full Peebles three-level atom recombination history
    (cached ODE table) to compute X_e(z) when x_e is not provided.
    """
    if x_e is None:
        x_e = ionization_fraction(z, cosmo)
    return x_e * _cosmo_n_h(z, cosmo)


# ---------------------------------------------------------------------------
# Public background quantities
# ---------------------------------------------------------------------------


def baryon_photon_ratio(z: ArrayLike, cosmo: CosmoLike | None = None) -> FloatOrArray:
    """Baryon-to-photon energy density ratio ``R(z) = 3 ρ_b / (4 ρ_γ)``.

    Used in the photon-baryon sound speed and pre-recombination
    perturbation theory.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to :data:`DEFAULT_COSMO`.

    Returns
    -------
    float or ndarray of float64
        Dimensionless baryon-to-photon energy density ratio.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    og = _cosmo_omega_gamma(cosmo)
    return 3.0 * cosmo["omega_b"] / (4.0 * og) / (1.0 + np.asarray(z))


def hubble(z: ArrayLike, cosmo: CosmoLike | None = None) -> FloatOrArray:
    """Hubble rate ``H(z)`` for a flat ΛCDM background.

    .. math::

        H(z) = H_0 \\sqrt{\\Omega_m (1+z)^3
                          + \\Omega_r (1+z)^4 + \\Omega_\\Lambda},

    with ``Ω_r`` including the photon and ``N_eff`` neutrino contributions.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to :data:`DEFAULT_COSMO`.

    Returns
    -------
    float or ndarray of float64
        Hubble rate in **inverse seconds** (1/s).
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    return _cosmo_hubble(z, cosmo)


def n_hydrogen(z: ArrayLike, cosmo: CosmoLike | None = None) -> FloatOrArray:
    """Hydrogen number density ``n_H(z)``.

    Computed from the baryon density assuming primordial mass fraction
    ``Y_p`` of helium:
    ``n_H(z) = (1 − Y_p) ρ_b(z) / m_p``.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to :data:`DEFAULT_COSMO`.

    Returns
    -------
    float or ndarray of float64
        Number density in **1/m³**.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    return _cosmo_n_h(z, cosmo)


def n_electron(
    z: ArrayLike,
    cosmo: CosmoLike | None = None,
    x_e: float | NDArray[np.float64] | None = None,
) -> FloatOrArray:
    """Free-electron number density ``n_e(z) = X_e(z) · n_H(z)``.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to :data:`DEFAULT_COSMO`.
    x_e : float or array_like, optional
        Ionization fraction.  If *None* (default), uses the cached
        Saha+Peebles recombination history from :func:`ionization_fraction`.

    Returns
    -------
    float or ndarray of float64
        Number density in **1/m³**.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    return _cosmo_n_e(z, cosmo, x_e=x_e)


def omega_gamma(cosmo: CosmoLike | None = None) -> float:
    """Present-day photon density parameter ``Ω_γ``.

    .. math::

        \\Omega_\\gamma = \\rho_{\\gamma,0} / \\rho_{crit,0},
        \\qquad \\rho_{\\gamma,0} = \\frac{\\pi^2}{15}
                                       \\frac{(k_B T_0)^4}{(\\hbar c)^3}.

    Parameters
    ----------
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to :data:`DEFAULT_COSMO`.

    Returns
    -------
    float
        Dimensionless photon density parameter.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    return _cosmo_omega_gamma(cosmo)


def rho_gamma(z: ArrayLike, cosmo: CosmoLike | None = None) -> FloatOrArray:
    """Photon energy density ``ρ_γ(z) = (π²/15) (k_B T(z))⁴ / (ℏc)³``.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to :data:`DEFAULT_COSMO`.

    Returns
    -------
    float or ndarray of float64
        Photon energy density in **J/m³**.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    rho_g0 = (
        np.pi**2
        / 15.0
        * _K_BOLTZMANN**4
        * cosmo["t_cmb"] ** 4
        / (_HBAR**3 * _C_LIGHT**3)
    )
    return rho_g0 * (1.0 + np.asarray(z, dtype=np.float64)) ** 4


def cosmic_time(
    z: float,
    cosmo: CosmoLike | None = None,
    z_upper: float = 1.0e9,
    n_points: int = 128,
) -> float:
    """Cosmic time ``t(z)`` by quadrature of ``dt/dz = −1/((1+z) H(z))``.

    Integrates from ``z_upper`` (effectively early-time t = 0) down to
    ``z`` using the midpoint rule in ``ln(1+z)``.

    Parameters
    ----------
    z : float
        Redshift at which to evaluate ``t(z)``.
    cosmo : Mapping, optional
        Cosmological parameters.  Defaults to :data:`DEFAULT_COSMO`.
    z_upper : float, optional
        Upper integration limit (default ``1e9``).  Increase for very
        early-Universe applications (e.g., neutrino decoupling).
    n_points : int, optional
        Number of quadrature points (default 128).  Increase for higher
        accuracy at small ``z``.

    Returns
    -------
    float
        Cosmic time in **seconds** (s).
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    # Integrate dt = -dz / ((1+z)*H(z)) from z_upper to z
    # i.e. t(z) = integral from z to z_upper of dz' / ((1+z')*H(z'))
    u_low = np.log(1.0 + z)
    u_high = np.log(1.0 + z_upper)
    h = (u_high - u_low) / n_points

    integral = 0.0
    for i in range(n_points):
        u = u_low + (i + 0.5) * h
        zp = np.exp(u) - 1.0
        # du = dz/(1+z), so dz = (1+z) du
        # integrand in u: 1/H(z)
        integral += h / _cosmo_hubble(zp, cosmo)

    return integral
