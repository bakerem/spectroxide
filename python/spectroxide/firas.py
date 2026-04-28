"""FIRAS spectral-distortion constraints with full covariance matrix.

Provides χ² fitting of arbitrary spectral distortions against the
COBE/FIRAS monopole residuals using the full 43 × 43
frequency-frequency covariance matrix from the LAMBDA archive.

Data sources
------------
From https://lambda.gsfc.nasa.gov/product/cobe/firas_products.html:

- Monopole spectrum: Fixsen et al. 1996, ApJ 473, 576 (Table 4).
- Covariance matrix: FIRAS_COVARIANCE_MATRIX_LOWF.FITS.

The per-pixel covariance is converted to a monopole covariance using the
correlation structure from the pixel covariance and the published
monopole 1-σ errors:
``C_monopole_ij = σ_i σ_j corr_ij``.

Usage
-----
>>> from spectroxide.firas import FIRASData
>>> firas = FIRASData()
>>> chi2 = firas.chi2(model_kJy)             # χ² for a model in kJy/sr
>>> mu_limit = firas.upper_limit_mu()        # 95% CL upper limit on |μ|
>>> result = firas.fit_distortion(delta_n_func)  # joint (μ, y, ΔT/T) fit
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2 as chi2_dist

from .greens import (
    mu_shape as _mu_shape,
    y_shape as _y_shape,
    g_bb as _g_bb,
    planck as _planck,
)

# Physical constants (must match greens.py)
_H_PLANCK = 6.62607015e-34  # J s
_K_BOLTZMANN = 1.380649e-23  # J/K
_C_LIGHT = 2.99792458e8  # m/s
_T_CMB = 2.726  # K (Fixsen & Mather 2002)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _load_monopole_data() -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Load the FIRAS monopole spectrum from the bundled text table.

    Returns
    -------
    freq_cm : ndarray, shape (43,)
        Frequency in **cm⁻¹**.
    spectrum_MJy : ndarray, shape (43,)
        Monopole spectrum in **MJy/sr**.
    residual_kJy : ndarray, shape (43,)
        Residual monopole spectrum in **kJy/sr**.
    sigma_kJy : ndarray, shape (43,)
        1-σ uncertainty in **kJy/sr**.
    galaxy_kJy : ndarray, shape (43,)
        Modelled galaxy spectrum at the poles in **kJy/sr**.
    """
    fpath = _DATA_DIR / "firas_monopole_spec_v1.txt"
    data = np.loadtxt(fpath)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]


def _load_correlation_matrix() -> NDArray[np.float64]:
    """Load the 43×43 frequency-frequency correlation matrix.

    Derived from FIRAS_COVARIANCE_MATRIX_LOWF.FITS per-pixel covariance.

    Returns
    -------
    ndarray, shape (43, 43)
        Symmetric correlation matrix (dimensionless).
    """
    fpath = _DATA_DIR / "firas_correlation_matrix.txt"
    return np.loadtxt(fpath)


def _freq_cm_to_ghz(freq_cm: ArrayLike) -> NDArray[np.float64]:
    """Convert wavenumber to frequency.

    Parameters
    ----------
    freq_cm : array_like
        Wavenumber in **cm⁻¹**.

    Returns
    -------
    ndarray of float64
        Frequency in **GHz**.
    """
    return np.asarray(freq_cm, dtype=np.float64) * _C_LIGHT * 1e-7


def _freq_cm_to_x(freq_cm: ArrayLike, t_cmb: float = _T_CMB) -> NDArray[np.float64]:
    """Convert wavenumber to dimensionless ``x = h ν / (k_B T_z)``.

    Parameters
    ----------
    freq_cm : array_like
        Wavenumber in **cm⁻¹**.
    t_cmb : float, optional
        CMB temperature today, in **K**.  Default :data:`_T_CMB` = 2.726.

    Returns
    -------
    ndarray of float64
        Dimensionless frequency.
    """
    nu_hz = np.asarray(freq_cm, dtype=np.float64) * _C_LIGHT * 100.0
    return _H_PLANCK * nu_hz / (_K_BOLTZMANN * t_cmb)


def _dn_to_dI_kJy(
    x: ArrayLike, dn: ArrayLike, t_cmb: float = _T_CMB
) -> NDArray[np.float64]:
    """Convert ``Δn(x)`` to ``ΔI`` in kJy/sr at the given ``x`` grid.

    Uses ``ΔI = (2 h ν³ / c²) Δn`` with ``ν = x k_B T₀ / h``.

    Parameters
    ----------
    x : array_like
        Dimensionless frequency.
    dn : array_like
        Spectral distortion ``Δn(x)``.
    t_cmb : float, optional
        CMB temperature today, in **K**.  Default :data:`_T_CMB`.

    Returns
    -------
    ndarray of float64
        Intensity distortion in **kJy/sr**.
    """
    nu_hz = np.asarray(x, dtype=np.float64) * _K_BOLTZMANN * t_cmb / _H_PLANCK
    di_si = 2.0 * _H_PLANCK * nu_hz**3 / _C_LIGHT**2 * np.asarray(dn)
    return di_si / 1e-23


def _galactic_dust_template_kJy(
    freq_cm: ArrayLike, t_dust: float = 9.0
) -> NDArray[np.float64]:
    """High-latitude galactic-dust template ``ν² B_ν(T_dust)`` in kJy/sr.

    Functional form from Fixsen et al. 1996, §6.1: the residual
    high-latitude galactic emission is fit by
    ``G(ν) = G₀ ν² B_ν(T = 9 K)``, with ``G₀`` profiled out of the
    cosmological fit.  The absolute normalisation here is arbitrary
    (``G₀`` is dimensionless and floats); only the *shape* matters for
    the profile likelihood.

    Parameters
    ----------
    freq_cm : array_like
        Wavenumber in **cm⁻¹**.
    t_dust : float, optional
        Effective dust temperature, in **K**.  Default 9.0.

    Returns
    -------
    ndarray of float64
        Template values in **kJy/sr** with arbitrary normalisation.
    """
    nu_hz = freq_cm * _C_LIGHT * 100  # cm^-1 -> Hz
    prefactor = 2.0 * _H_PLANCK * nu_hz**3 / _C_LIGHT**2  # W/m²/Hz/sr per (1/(e^x-1))
    x_dust = _H_PLANCK * nu_hz / (_K_BOLTZMANN * t_dust)
    bnu_si = prefactor / np.expm1(x_dust)  # B_ν(T_dust) in W/m²/Hz/sr
    template_si = nu_hz**2 * bnu_si  # ν² · B_ν, arbitrary normalization
    return template_si / 1e-23  # -> kJy/sr


class FIRASData:
    """FIRAS monopole data with full covariance matrix.

    Loads the FIRAS monopole spectrum (43 frequency channels,
    68–640 GHz) and the full frequency-frequency covariance matrix from
    the bundled data files.

    Parameters
    ----------
    t_cmb : float, optional
        CMB temperature today, in **K**.  Default 2.726 (Fixsen & Mather
        2002).
    t_dust : float, optional
        Effective dust temperature (in **K**) used by the galactic
        nuisance template ``ν² B_ν(T_dust)``.  Default 9.0 (Fixsen 1996
        §6.1).

    Attributes
    ----------
    n_freq : int
        Number of frequency channels (43).
    freq_cm : ndarray of float64
        Frequencies in **cm⁻¹**.
    freq_ghz : ndarray of float64
        Frequencies in **GHz**.
    x : ndarray of float64
        Dimensionless frequencies ``x = h ν / (k_B T_CMB)``.
    spectrum_MJy : ndarray of float64
        Monopole spectrum in **MJy/sr**.
    residual_kJy : ndarray of float64
        Residual monopole spectrum in **kJy/sr**.
    sigma_kJy : ndarray of float64
        1-σ diagonal uncertainties in **kJy/sr**.
    galaxy_kJy : ndarray of float64
        Modelled high-latitude galactic spectrum in **kJy/sr**.
    cov : ndarray, shape (43, 43)
        Full monopole covariance matrix in ``(kJy/sr)²``.
    cov_inv : ndarray, shape (43, 43)
        Inverse covariance matrix in ``(kJy/sr)⁻²``.
    corr : ndarray, shape (43, 43)
        Frequency-frequency correlation matrix (dimensionless).
    """

    def __init__(self, t_cmb: float = _T_CMB, t_dust: float = 9.0) -> None:
        self.t_cmb = t_cmb
        self.t_dust = t_dust

        # Load data
        freq_cm, spectrum_MJy, residual_kJy, sigma_kJy, galaxy_kJy = (
            _load_monopole_data()
        )
        corr = _load_correlation_matrix()

        self.n_freq = len(freq_cm)
        self.freq_cm = freq_cm
        self.freq_ghz = _freq_cm_to_ghz(freq_cm)
        self.x = _freq_cm_to_x(freq_cm, t_cmb)
        self.spectrum_MJy = spectrum_MJy
        self.residual_kJy = residual_kJy
        self.sigma_kJy = sigma_kJy
        self.galaxy_kJy = galaxy_kJy
        self.corr = corr

        # Monopole covariance: C_ij = sigma_i * sigma_j * corr_ij
        self.cov = np.outer(sigma_kJy, sigma_kJy) * corr
        self.cov_inv = np.linalg.inv(self.cov)

        # Pre-evaluate spectral templates at FIRAS frequencies
        self._M = _mu_shape(self.x)  # M(x) [occupation number]
        self._Y = _y_shape(self.x)  # Y_SZ(x) [occupation number]
        self._G = _g_bb(self.x)  # G_bb(x) [occupation number]

        # Templates in kJy/sr (per unit μ, y, ΔT/T)
        self._M_kJy = _dn_to_dI_kJy(self.x, self._M, t_cmb)
        self._Y_kJy = _dn_to_dI_kJy(self.x, self._Y, t_cmb)
        self._G_kJy = _dn_to_dI_kJy(self.x, self._G, t_cmb)

        # Galactic dust nuisance template ν² · B_ν(T_dust), Fixsen 1996 §6.1
        self._gal_kJy = _galactic_dust_template_kJy(self.freq_cm, t_dust)

    # ------------------------------------------------------------------
    # Core chi-squared
    # ------------------------------------------------------------------

    def chi2(self, model_kJy: ArrayLike) -> float:
        """Compute ``χ²`` of a model prediction against the FIRAS residuals.

        Parameters
        ----------
        model_kJy : array_like, shape (43,)
            Model prediction for the residual spectrum in **kJy/sr**.

        Returns
        -------
        float
            ``χ² = (r − m)ᵀ C⁻¹ (r − m)``.
        """
        r = self.residual_kJy - np.asarray(model_kJy)
        return float(r @ self.cov_inv @ r)

    def chi2_null(self) -> float:
        """``χ²`` of the null hypothesis (no distortion).

        Returns
        -------
        float
            ``rᵀ C⁻¹ r`` where ``r`` is the FIRAS residual.
        """
        return self.chi2(np.zeros(self.n_freq))

    # ------------------------------------------------------------------
    # Distortion templates
    # ------------------------------------------------------------------

    def mu_template_kJy(self) -> NDArray[np.float64]:
        """μ-distortion template ``M(x)`` evaluated at the FIRAS frequencies.

        Returns
        -------
        ndarray of float64, shape (43,)
            Template in **kJy/sr** per unit μ.
        """
        return self._M_kJy.copy()

    def y_template_kJy(self) -> NDArray[np.float64]:
        """y-distortion template ``Y_SZ(x)`` at the FIRAS frequencies.

        Returns
        -------
        ndarray of float64, shape (43,)
            Template in **kJy/sr** per unit y.
        """
        return self._Y_kJy.copy()

    def gbb_template_kJy(self) -> NDArray[np.float64]:
        """Temperature-shift template ``G_bb(x)`` at the FIRAS frequencies.

        Returns
        -------
        ndarray of float64, shape (43,)
            Template in **kJy/sr** per unit ΔT/T.
        """
        return self._G_kJy.copy()

    def galactic_template_kJy(self) -> NDArray[np.float64]:
        """High-latitude galactic dust template ν²·B_ν(T_dust) in kJy/sr.

        Functional form from Fixsen et al. 1996, §6.1: G(ν) = G₀·ν²·B_ν(9 K),
        with G₀ profiled out as a nuisance parameter. Normalization is
        arbitrary (only the shape matters for the profile likelihood).
        """
        return self._gal_kJy.copy()

    # ------------------------------------------------------------------
    # Single-parameter fits
    # ------------------------------------------------------------------

    def fit_amplitude(self, template_kJy: ArrayLike) -> dict:
        """Fit a single amplitude ``A`` to the FIRAS residuals.

        Minimises ``χ² = (r − A t)ᵀ C⁻¹ (r − A t)`` analytically.

        Parameters
        ----------
        template_kJy : array_like, shape (43,)
            Spectral template in **kJy/sr** (at unit amplitude).

        Returns
        -------
        dict
            Keys ``amplitude`` (float, best-fit), ``sigma`` (float, 1-σ),
            ``chi2_min`` (float, minimum χ²), ``chi2_null`` (float, χ² of
            the zero-amplitude model), ``snr`` (float, ``|A| / σ``).
        """
        t = np.asarray(template_kJy)
        r = self.residual_kJy

        tCt = t @ self.cov_inv @ t
        tCr = t @ self.cov_inv @ r
        rCr = r @ self.cov_inv @ r

        a_hat = tCr / tCt
        sigma_a = 1.0 / np.sqrt(tCt)
        chi2_min = rCr - tCr**2 / tCt

        return {
            "amplitude": float(a_hat),
            "sigma": float(sigma_a),
            "chi2_min": float(chi2_min),
            "chi2_null": float(rCr),
            "snr": float(abs(a_hat) / sigma_a),
        }

    def upper_limit(self, template_kJy: ArrayLike, cl: float = 0.95) -> float:
        """Upper limit on the amplitude of a spectral template.

        The limit is ``|A| < |Â| + z_{cl} σ``, where ``z_{cl}`` is the
        quantile of the standard normal for the desired confidence
        level.  For a two-sided bound at 95% CL, ``z_{cl} ≈ 1.96``.

        For consistency with the literature convention (Fixsen et al.
        1996), this returns the two-sided limit: the region
        ``|A| > A_limit`` is excluded.

        Parameters
        ----------
        template_kJy : array_like, shape (43,)
            Spectral template in **kJy/sr** (at unit amplitude).
        cl : float, optional
            Confidence level (default 0.95).  Must lie in ``(0, 1)``.

        Returns
        -------
        float
            Upper limit on ``|A|`` at the given CL.
        """
        from scipy.stats import norm

        fit = self.fit_amplitude(template_kJy)
        z_cl = norm.ppf(0.5 + cl / 2)  # two-sided
        return abs(fit["amplitude"]) + z_cl * fit["sigma"]

    def fit_amplitude_marginalised(
        self,
        template_kJy: ArrayLike,
        nuisance_kJy: Sequence[ArrayLike] | None = None,
    ) -> dict:
        """Fit amplitude ``A`` marginalised over a list of nuisance templates.

        Fits the model ``data = A · template + Σ_j b_j · nuisance_j`` and
        returns the marginalised constraint on ``A`` (i.e. ``A`` after
        profiling over the ``b_j``).

        Parameters
        ----------
        template_kJy : array_like, shape (43,)
            Signal template in **kJy/sr**.
        nuisance_kJy : sequence of array_like, optional
            Nuisance templates to marginalise over.  Default
            ``[G_bb]`` — the temperature shift is always unobservable.

        Returns
        -------
        dict
            Keys ``amplitude``, ``sigma``, ``chi2_min``, ``chi2_null``,
            ``snr`` (all floats); see :meth:`fit_amplitude`.
        """
        if nuisance_kJy is None:
            nuisance_kJy = [self._G_kJy]

        t = np.asarray(template_kJy)
        r = self.residual_kJy
        nuis = [np.asarray(n) for n in nuisance_kJy]

        # Build full design matrix [template | nuisance]
        A = np.column_stack([t] + nuis)
        AtCinv = A.T @ self.cov_inv
        fisher = AtCinv @ A
        param_cov = np.linalg.inv(fisher)
        theta = param_cov @ (AtCinv @ r)

        a_hat = theta[0]
        sigma_a = np.sqrt(param_cov[0, 0])

        resid = r - A @ theta
        chi2_min = float(resid @ self.cov_inv @ resid)
        rCr = float(r @ self.cov_inv @ r)

        return {
            "amplitude": float(a_hat),
            "sigma": float(sigma_a),
            "chi2_min": chi2_min,
            "chi2_null": rCr,
            "snr": float(abs(a_hat) / sigma_a),
        }

    def upper_limit_mu(
        self,
        cl: float = 0.95,
        marginalise_y: bool = True,
        marginalise_galactic: bool = True,
    ) -> float:
        """FIRAS upper limit on ``|μ|`` using the full covariance matrix.

        Marginalises over ``G_bb`` (unobservable temperature shift).
        Optionally also marginalises over ``y`` and over the galactic
        dust nuisance ``ν² B_ν(T_dust)`` (Fixsen 1996 §6.1).

        Parameters
        ----------
        cl : float, optional
            Confidence level (default 0.95).
        marginalise_y : bool, optional
            If *True* (default), also marginalise over y-distortion.
        marginalise_galactic : bool, optional
            If *True* (default), also marginalise over the galactic dust
            template.

        Returns
        -------
        float
            Upper limit on ``|μ|`` at the given CL.
        """
        from scipy.stats import norm

        nuisance = [self._G_kJy]
        if marginalise_y:
            nuisance.append(self._Y_kJy)
        if marginalise_galactic:
            nuisance.append(self._gal_kJy)
        fit = self.fit_amplitude_marginalised(self._M_kJy, nuisance)
        z_cl = norm.ppf(0.5 + cl / 2)
        return abs(fit["amplitude"]) + z_cl * fit["sigma"]

    def upper_limit_y(
        self,
        cl: float = 0.95,
        marginalise_mu: bool = True,
        marginalise_galactic: bool = True,
    ) -> float:
        """FIRAS upper limit on ``|y|`` using the full covariance matrix.

        Marginalises over ``G_bb`` (unobservable temperature shift).
        Optionally also marginalises over ``μ`` and over the galactic
        dust nuisance ``ν² B_ν(T_dust)`` (Fixsen 1996 §6.1).

        Parameters
        ----------
        cl : float, optional
            Confidence level (default 0.95).
        marginalise_mu : bool, optional
            If *True* (default), also marginalise over μ-distortion.
        marginalise_galactic : bool, optional
            If *True* (default), also marginalise over the galactic dust
            template.

        Returns
        -------
        float
            Upper limit on ``|y|`` at the given CL.
        """
        from scipy.stats import norm

        nuisance = [self._G_kJy]
        if marginalise_mu:
            nuisance.append(self._M_kJy)
        if marginalise_galactic:
            nuisance.append(self._gal_kJy)
        fit = self.fit_amplitude_marginalised(self._Y_kJy, nuisance)
        z_cl = norm.ppf(0.5 + cl / 2)
        return abs(fit["amplitude"]) + z_cl * fit["sigma"]

    # ------------------------------------------------------------------
    # Multi-parameter fit (μ, y, ΔT/T)
    # ------------------------------------------------------------------

    def fit_distortion(
        self,
        delta_n: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,
        model_kJy: ArrayLike | None = None,
    ) -> dict:
        """Joint fit of ``(μ, y, ΔT/T)`` to FIRAS residuals or a model.

        Fits ``ΔI = μ · M + y · Y + (ΔT/T) · G_bb`` (in kJy/sr).

        - If ``delta_n`` is provided, it is converted to kJy/sr and
          subtracted from the residual before fitting.
        - If ``model_kJy`` is provided, it is also subtracted.

        Parameters
        ----------
        delta_n : callable or array_like, optional
            If callable, ``delta_n(x)`` returns ``Δn`` at the FIRAS
            frequencies.  If array_like, shape (43,), ``Δn`` at the FIRAS
            frequencies.  Default *None*.
        model_kJy : array_like, shape (43,), optional
            Additional model prediction in **kJy/sr** to subtract before
            fitting.  Default *None*.

        Returns
        -------
        dict
            Keys ``mu``, ``y``, ``delta_t`` (floats, best-fit
            parameters), ``mu_sigma``, ``y_sigma``, ``delta_t_sigma``
            (floats, 1-σ uncertainties), ``param_cov`` (ndarray,
            shape (3, 3)), ``chi2`` (float), ``ndof`` (int, 43 − 3 = 40),
            and ``pte`` (float, probability-to-exceed under the χ² null).
        """
        # Build design matrix
        A = np.column_stack([self._M_kJy, self._Y_kJy, self._G_kJy])

        # Data vector
        r = self.residual_kJy.copy()
        if model_kJy is not None:
            r = r - np.asarray(model_kJy)
        if delta_n is not None:
            if callable(delta_n):
                dn = delta_n(self.x)
            else:
                dn = np.asarray(delta_n)
            r = r - _dn_to_dI_kJy(self.x, dn, self.t_cmb)

        # Weighted least squares: (A^T C^{-1} A) θ = A^T C^{-1} r
        AtCinv = A.T @ self.cov_inv
        fisher = AtCinv @ A
        param_cov = np.linalg.inv(fisher)
        theta = param_cov @ (AtCinv @ r)

        # Residual
        resid = r - A @ theta
        chi2_val = float(resid @ self.cov_inv @ resid)
        ndof = self.n_freq - 3
        pte = float(1.0 - chi2_dist.cdf(chi2_val, ndof))

        return {
            "mu": float(theta[0]),
            "y": float(theta[1]),
            "delta_t": float(theta[2]),
            "mu_sigma": float(np.sqrt(param_cov[0, 0])),
            "y_sigma": float(np.sqrt(param_cov[1, 1])),
            "delta_t_sigma": float(np.sqrt(param_cov[2, 2])),
            "param_cov": param_cov,
            "chi2": chi2_val,
            "ndof": ndof,
            "pte": pte,
        }

    # ------------------------------------------------------------------
    # Fisher matrix for arbitrary templates
    # ------------------------------------------------------------------

    def fisher_matrix(self, templates_kJy: Sequence[ArrayLike]) -> NDArray[np.float64]:
        """Fisher information matrix for a set of spectral templates.

        ``F_{ij} = t_iᵀ C⁻¹ t_j``.

        Parameters
        ----------
        templates_kJy : sequence of array_like, each shape (43,)
            Spectral templates in **kJy/sr** (at unit amplitude).

        Returns
        -------
        ndarray, shape (n_templates, n_templates)
            Fisher matrix in ``(kJy/sr)⁻²``.
        """
        A = np.column_stack(templates_kJy)
        return A.T @ self.cov_inv @ A

    # ------------------------------------------------------------------
    # Model prediction at FIRAS frequencies
    # ------------------------------------------------------------------

    def predict_kJy(
        self,
        mu: float = 0.0,
        y: float = 0.0,
        delta_t: float = 0.0,
        extra_dn: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Predicted FIRAS signal for given distortion parameters.

        ``ΔI = μ · M_kJy + y · Y_kJy + (ΔT/T) · G_kJy + dn_kJy(extra)``.

        Parameters
        ----------
        mu : float, optional
            μ-distortion amplitude (default 0).
        y : float, optional
            y-distortion amplitude (default 0).
        delta_t : float, optional
            Temperature shift ``ΔT/T`` (default 0).
        extra_dn : callable or array_like, optional
            Additional ``Δn(x)`` on top of the (μ, y, ΔT) model.  If
            callable, evaluated at the FIRAS dimensionless frequencies.

        Returns
        -------
        ndarray, shape (43,)
            Predicted residual signal in **kJy/sr**.
        """
        signal = mu * self._M_kJy + y * self._Y_kJy + delta_t * self._G_kJy
        if extra_dn is not None:
            if callable(extra_dn):
                dn = extra_dn(self.x)
            else:
                dn = np.asarray(extra_dn)
            signal = signal + _dn_to_dI_kJy(self.x, dn, self.t_cmb)
        return signal

    def chi2_distortion(
        self,
        mu: float = 0.0,
        y: float = 0.0,
        delta_t: float = 0.0,
        extra_dn: Callable[[ArrayLike], ArrayLike] | ArrayLike | None = None,
    ) -> float:
        """``χ²`` for a distortion model against the FIRAS residuals.

        Parameters
        ----------
        mu : float, optional
            μ-distortion amplitude (default 0).
        y : float, optional
            y-distortion amplitude (default 0).
        delta_t : float, optional
            Temperature shift ``ΔT/T`` (default 0).
        extra_dn : callable or array_like, optional
            Additional ``Δn(x)`` beyond the (μ, y, ΔT) parametrisation.

        Returns
        -------
        float
            ``χ²`` value.
        """
        return self.chi2(self.predict_kJy(mu, y, delta_t, extra_dn))

    # ------------------------------------------------------------------
    # Constraint on an arbitrary model spectrum
    # ------------------------------------------------------------------

    def limit_on_model(
        self,
        spectrum_func: Callable[[ArrayLike], ArrayLike],
        cl: float = 0.95,
        marginalise_gbb: bool = True,
        marginalise_galactic: bool = True,
    ) -> dict:
        """Upper limit on the amplitude of an arbitrary spectral distortion.

        Given a unit-amplitude model spectrum ``Δn(x)``, finds the
        maximum amplitude ``A`` consistent with the FIRAS data.

        Parameters
        ----------
        spectrum_func : callable
            ``spectrum_func(x)`` returns ``Δn(x)`` at unit model amplitude.
        cl : float, optional
            Confidence level (default 0.95).
        marginalise_gbb : bool, optional
            If *True* (default), marginalise over ``G_bb`` (the
            unobservable temperature shift).
        marginalise_galactic : bool, optional
            If *True* (default), also marginalise over the galactic dust
            template ``ν² B_ν(T_dust)`` (Fixsen 1996 §6.1).

        Returns
        -------
        dict
            Keys ``amplitude`` (float, best-fit), ``sigma`` (float, 1-σ),
            ``upper_limit`` (float, two-sided upper limit on ``|A|``).
        """
        from scipy.stats import norm

        dn = spectrum_func(self.x)
        model_kJy = _dn_to_dI_kJy(self.x, dn, self.t_cmb)

        nuisance = []
        if marginalise_gbb:
            nuisance.append(self._G_kJy)
        if marginalise_galactic:
            nuisance.append(self._gal_kJy)

        if nuisance:
            fit = self.fit_amplitude_marginalised(model_kJy, nuisance)
            a_hat = fit["amplitude"]
            sigma_a = fit["sigma"]
        else:
            tCt = model_kJy @ self.cov_inv @ model_kJy
            tCr = model_kJy @ self.cov_inv @ self.residual_kJy
            a_hat = tCr / tCt
            sigma_a = 1.0 / np.sqrt(tCt)

        z_cl = norm.ppf(0.5 + cl / 2)

        return {
            "amplitude": float(a_hat),
            "sigma": float(sigma_a),
            "upper_limit": float(abs(a_hat) + z_cl * sigma_a),
        }

    # ------------------------------------------------------------------
    # Profile likelihood with floating blackbody temperature
    # ------------------------------------------------------------------

    def profile_limit_floating_T(
        self,
        template_dn_func,
        cl=0.95,
        t_range=None,
        marginalise_galactic=True,
        use_diagonal=False,
    ):
        """One-sided profile-likelihood upper limit with the CMB temperature floated.

        Fits the model

            I_obs(ν) − B(ν, T) = 4π ν³ A · 𝒯(x(T)) + G₀ · ν² B(ν, T_d)

        to the 43 FIRAS monopole residuals using the full 43 × 43
        covariance matrix. The first term on the RHS converts the
        occupation-number template ``𝒯`` (supplied as ``template_dn_func``)
        to specific intensity at amplitude ``A``. The second term is a
        galactic dust foreground following Fixsen et al. (1996); residual
        galactic emission is not perfectly subtracted from the FIRAS
        monopole and is partially degenerate with broadband distortion
        shapes, so when ``marginalise_galactic=True`` (default) we
        marginalise over a fixed-shape ``ν² B(ν, T_d)`` template with
        ``T_d = 9 K`` and free amplitude ``G₀``.

        The CMB reference temperature ``T`` is itself a free parameter
        (it encodes the unobservable temperature shift), and both the
        residuals and the template shape ``𝒯(x(T))`` with
        ``x(T) = 2π ν / T`` depend nonlinearly on ``T``.  We therefore
        profile over ``T`` by scanning a grid around ``T₀``; at each ``T``
        the best-fit ``A`` and ``G₀`` are obtained analytically (the
        model is linear in both), and we take the ``T`` that minimises
        χ². The one-sided 95% CL upper limit corresponds to ``Δχ² = 2.71``
        on the profile likelihood ratio (Wilks' theorem).

        Parameters
        ----------
        template_dn_func : callable
            ``template_dn_func(x)`` returns Delta-n(x) per unit signal
            amplitude at dimensionless frequencies x = h*nu/(k*T).
        cl : float
            One-sided confidence level (default 0.95).
        t_range : tuple of float, optional
            (T_min, T_max) search range for T [K].
            Default: (2.720, 2.732), well within FIRAS precision.
        marginalise_galactic : bool
            If True (default), profile over a galactic dust normalization G₀
            with shape ν²·B_ν(T_dust). The dust template is independent of
            the trial T_CMB.

        Returns
        -------
        dict with keys:
            amplitude : float
                Best-fit signal amplitude at profiled T.
            sigma : float
                1-sigma uncertainty on amplitude.
            upper_limit : float
                One-sided upper limit on amplitude at given CL,
                clipped at zero.
            t_best : float
                Best-fit blackbody temperature [K].
            chi2_min : float
                Minimum chi-squared (profiled over both A and T).
        """
        from scipy.stats import norm

        fit = self._joint_fit_floating_T(
            template_dn_func,
            t_range=t_range,
            marginalise_galactic=marginalise_galactic,
            use_diagonal=use_diagonal,
        )
        a_hat = fit["amplitude"]
        sigma_a = fit["sigma"]

        z_cl = norm.ppf(cl)  # 1.645 for 95%
        upper_limit = max(a_hat + z_cl * sigma_a, 0.0)

        return {
            "amplitude": float(a_hat),
            "sigma": float(sigma_a),
            "upper_limit": float(upper_limit),
            "t_best": float(fit["t_best"]),
            "chi2_min": float(fit["chi2_min"]),
        }

    def _joint_fit_floating_T(
        self,
        template_dn_func,
        t_range=None,
        marginalise_galactic=True,
        use_diagonal=False,
    ):
        """Joint fit of (signal A, T_CMB) with dust amplitude profiled.

        Internal helper used by ``profile_limit_floating_T``. Returns the
        best-fit signal amplitude, its 1-sigma uncertainty (marginalised
        over T and G_0), the best-fit T_CMB, and the joint chi²_min.

        ``use_diagonal=True`` replaces the full covariance with diagonal
        noise (matching the Chluba, Cyr & Johnson 2024 prescription).
        """
        from scipy.optimize import minimize_scalar

        if t_range is None:
            t_range = (2.720, 2.732)

        nu_hz = self.freq_cm * _C_LIGHT * 100  # cm^-1 -> Hz
        i_obs_kJy = self.spectrum_MJy * 1e3
        prefactor = 2.0 * _H_PLANCK * nu_hz**3 / _C_LIGHT**2

        cov_inv = np.diag(1.0 / self.sigma_kJy**2) if use_diagonal else self.cov_inv

        gal_kJy = self._gal_kJy if marginalise_galactic else None

        def _chi2_profiled(T):
            x = _H_PLANCK * nu_hz / (_K_BOLTZMANN * T)
            bnu_kJy = prefactor / np.expm1(x) / 1e-23
            r = i_obs_kJy - bnu_kJy
            dn = template_dn_func(x)
            t_kJy = prefactor * dn / 1e-23

            if gal_kJy is None:
                tCt = t_kJy @ cov_inv @ t_kJy
                tCr = t_kJy @ cov_inv @ r
                if tCt < 1e-50:
                    return float(r @ cov_inv @ r)
                return float(r @ cov_inv @ r - tCr**2 / tCt)

            M = np.column_stack([t_kJy, gal_kJy])
            MtCinv = M.T @ cov_inv
            fisher2 = MtCinv @ M
            try:
                theta_star = np.linalg.solve(fisher2, MtCinv @ r)
            except np.linalg.LinAlgError:
                return float(r @ cov_inv @ r)
            resid = r - M @ theta_star
            return float(resid @ cov_inv @ resid)

        result = minimize_scalar(_chi2_profiled, bounds=t_range, method="bounded")
        T_best = result.x
        chi2_min = result.fun

        x_best = _H_PLANCK * nu_hz / (_K_BOLTZMANN * T_best)
        bnu_kJy_best = prefactor / np.expm1(x_best) / 1e-23
        r_best = i_obs_kJy - bnu_kJy_best

        dn_best = template_dn_func(x_best)
        t_kJy_best = prefactor * dn_best / 1e-23

        npl = 1.0 / np.expm1(x_best)
        g_bb_dn = x_best * npl * (1.0 + npl)
        g_kJy = prefactor * g_bb_dn / (1e-23 * T_best)

        cols = [t_kJy_best, g_kJy]
        if gal_kJy is not None:
            cols.append(gal_kJy)
        A_mat = np.column_stack(cols)
        fisher = A_mat.T @ cov_inv @ A_mat
        try:
            param_cov = np.linalg.inv(fisher)
            theta = param_cov @ (A_mat.T @ cov_inv @ r_best)
            a_hat = float(theta[0])
            var_a = float(param_cov[0, 0])
            sigma_a = float(np.sqrt(var_a)) if var_a > 0 else float("nan")
        except np.linalg.LinAlgError:
            # Template degenerate with T_CMB (e.g. G_bb shape) — Fisher singular.
            a_hat = float("nan")
            sigma_a = float("nan")

        return {
            "amplitude": a_hat,
            "sigma": sigma_a,
            "t_best": float(T_best),
            "chi2_min": float(chi2_min),
        }

    # ------------------------------------------------------------------
    # Convenience: diagonal-only comparison
    # ------------------------------------------------------------------

    def chi2_diagonal(self, model_kJy):
        """Chi-squared using only diagonal errors (no correlations).

        Useful for comparing the impact of off-diagonal correlations.
        """
        r = self.residual_kJy - np.asarray(model_kJy)
        return float(np.sum((r / self.sigma_kJy) ** 2))

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"FIRASData(n_freq={self.n_freq}, "
            f"freq_range=[{self.freq_ghz[0]:.1f}, {self.freq_ghz[-1]:.1f}] GHz, "
            f"x_range=[{self.x[0]:.2f}, {self.x[-1]:.2f}])"
        )


# ------------------------------------------------------------------
# Module-level convenience
# ------------------------------------------------------------------

# Scalar bounds from the literature (for quick checks without loading data)
MU_FIRAS_95 = 9e-5  # Fixsen et al. 1996, 95% CL
Y_FIRAS_95 = 1.5e-5  # Fixsen et al. 1996, 95% CL
MU_FIRAS_68 = 4.5e-5  # 68% CL (1-sigma)
Y_FIRAS_68 = 7.5e-6  # 68% CL (1-sigma)
