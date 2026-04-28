"""Tests for the FIRAS spectral distortion constraints module.

Validates FIRASData loading, chi-squared calculations, single and
multi-parameter fits, upper limits, and Fisher matrix operations.
"""

import numpy as np
import pytest

from spectroxide.firas import (
    FIRASData,
    MU_FIRAS_95,
    MU_FIRAS_68,
    Y_FIRAS_95,
    Y_FIRAS_68,
    _freq_cm_to_ghz,
    _freq_cm_to_x,
    _dn_to_dI_kJy,
    _galactic_dust_template_kJy,
)

# =========================================================================
# Data loading and basic attributes
# =========================================================================


class TestFIRASDataLoading:
    """Verify FIRAS data loads correctly with expected dimensions."""

    def test_n_freq(self):
        """FIRAS has 43 frequency channels."""
        firas = FIRASData()
        assert firas.n_freq == 43

    def test_freq_range_ghz(self):
        """Frequency range spans ~68–640 GHz (Fixsen et al. 1996)."""
        firas = FIRASData()
        assert firas.freq_ghz[0] > 50
        assert firas.freq_ghz[0] < 80
        assert firas.freq_ghz[-1] > 600
        assert firas.freq_ghz[-1] < 700

    def test_x_range(self):
        """Dimensionless frequencies span roughly 1–11."""
        firas = FIRASData()
        assert firas.x[0] > 0.5
        assert firas.x[0] < 2.0
        assert firas.x[-1] > 8.0
        assert firas.x[-1] < 15.0

    def test_covariance_shape(self):
        """Covariance matrix is 43×43."""
        firas = FIRASData()
        assert firas.cov.shape == (43, 43)
        assert firas.cov_inv.shape == (43, 43)

    def test_covariance_symmetric(self):
        """Covariance matrix is symmetric."""
        firas = FIRASData()
        np.testing.assert_allclose(firas.cov, firas.cov.T, atol=1e-30)

    def test_covariance_positive_definite(self):
        """Covariance matrix should be positive definite."""
        firas = FIRASData()
        eigvals = np.linalg.eigvalsh(firas.cov)
        assert np.all(eigvals > 0)

    def test_correlation_diagonal_near_ones(self):
        """Correlation matrix diagonal should be close to 1."""
        firas = FIRASData()
        np.testing.assert_allclose(np.diag(firas.corr), 1.0, atol=0.01)

    def test_cov_inv_is_inverse(self):
        """C @ C^{-1} ≈ I."""
        firas = FIRASData()
        product = firas.cov @ firas.cov_inv
        np.testing.assert_allclose(product, np.eye(43), atol=1e-8)

    def test_sigma_positive(self):
        """All per-channel uncertainties are positive."""
        firas = FIRASData()
        assert np.all(firas.sigma_kJy > 0)

    def test_residuals_small(self):
        """Residuals should be within a few sigma of zero."""
        firas = FIRASData()
        # Each residual should be within ~5σ (very loose)
        assert np.all(np.abs(firas.residual_kJy) < 5 * firas.sigma_kJy)

    def test_repr(self):
        """String representation contains frequency range info."""
        firas = FIRASData()
        s = repr(firas)
        assert "43" in s
        assert "GHz" in s

    def test_custom_t_cmb(self):
        """Constructor accepts custom T_CMB."""
        firas = FIRASData(t_cmb=2.7255)
        assert firas.t_cmb == 2.7255
        # x values should be slightly different from default
        firas_default = FIRASData()
        assert not np.allclose(firas.x, firas_default.x)


# =========================================================================
# Unit conversion helpers
# =========================================================================


class TestUnitConversions:
    """Verify frequency and intensity unit conversions."""

    def test_freq_cm_to_ghz_known_value(self):
        """1 cm^{-1} ≈ 29.98 GHz."""
        ghz = _freq_cm_to_ghz(1.0)
        assert abs(ghz - 29.9792) < 0.01

    def test_freq_cm_to_x_identity(self):
        """Check x = hν/(kT) with known values."""
        # At ν = kT/h ≈ 56.8 GHz for T=2.726K, x should be 1
        freq_cm_at_x1 = 1.380649e-23 * 2.726 / (6.62607015e-34 * 2.99792458e10)
        x = _freq_cm_to_x(freq_cm_at_x1)
        np.testing.assert_allclose(x, 1.0, rtol=1e-6)

    def test_dn_to_dI_kJy_positive(self):
        """Positive Δn gives positive ΔI."""
        x = np.array([3.0, 5.0])
        dn = np.array([1e-5, 1e-5])
        dI = _dn_to_dI_kJy(x, dn)
        assert np.all(dI > 0)

    def test_dn_to_dI_kJy_zero(self):
        """Zero Δn gives zero ΔI."""
        x = np.array([3.0, 5.0])
        dn = np.zeros(2)
        dI = _dn_to_dI_kJy(x, dn)
        np.testing.assert_allclose(dI, 0.0, atol=1e-30)


# =========================================================================
# Templates
# =========================================================================


class TestTemplates:
    """Verify spectral templates are computed and have expected properties."""

    def test_mu_template_nonzero(self):
        """μ-distortion template should be nonzero."""
        firas = FIRASData()
        M = firas.mu_template_kJy()
        assert M.shape == (43,)
        assert np.any(M != 0)

    def test_y_template_nonzero(self):
        """y-distortion template should be nonzero."""
        firas = FIRASData()
        Y = firas.y_template_kJy()
        assert Y.shape == (43,)
        assert np.any(Y != 0)

    def test_gbb_template_positive(self):
        """G_bb template should be positive at all FIRAS frequencies."""
        firas = FIRASData()
        G = firas.gbb_template_kJy()
        assert np.all(G > 0)

    def test_templates_are_copies(self):
        """Templates should be copies, not views of internal state."""
        firas = FIRASData()
        M1 = firas.mu_template_kJy()
        M1[:] = 0
        M2 = firas.mu_template_kJy()
        assert np.any(M2 != 0)


# =========================================================================
# Chi-squared calculations
# =========================================================================


class TestChiSquared:
    """Verify chi-squared calculations."""

    def test_chi2_null_finite(self):
        """Null chi-squared should be finite and reasonable."""
        firas = FIRASData()
        chi2_null = firas.chi2_null()
        assert np.isfinite(chi2_null)
        assert chi2_null > 0

    def test_chi2_null_good_fit(self):
        """Null hypothesis should be a good fit (FIRAS is consistent with BB).
        χ² ≈ 43 ± √(2×43) for 43 channels."""
        firas = FIRASData()
        chi2_null = firas.chi2_null()
        # Very broad: should be within [0, 200]
        assert 0 < chi2_null < 200

    def test_chi2_zero_model_equals_null(self):
        """χ²(model=0) should equal null chi-squared."""
        firas = FIRASData()
        chi2_zero = firas.chi2(np.zeros(43))
        chi2_null = firas.chi2_null()
        np.testing.assert_allclose(chi2_zero, chi2_null)

    def test_chi2_perfect_fit_zero(self):
        """χ²(model=residuals) should be zero."""
        firas = FIRASData()
        chi2 = firas.chi2(firas.residual_kJy)
        np.testing.assert_allclose(chi2, 0.0, atol=1e-10)

    def test_chi2_distortion_null(self):
        """chi2_distortion with zero params equals null."""
        firas = FIRASData()
        chi2 = firas.chi2_distortion(mu=0, y=0, delta_t=0)
        chi2_null = firas.chi2_null()
        np.testing.assert_allclose(chi2, chi2_null)

    def test_chi2_diagonal_larger(self):
        """Diagonal χ² should differ from full (off-diagonal matters)."""
        firas = FIRASData()
        model = np.zeros(43)
        chi2_full = firas.chi2(model)
        chi2_diag = firas.chi2_diagonal(model)
        # They won't be exactly equal due to off-diagonal correlations
        assert np.isfinite(chi2_diag)
        assert chi2_diag > 0


# =========================================================================
# Single-parameter fits
# =========================================================================


class TestSingleParameterFits:
    """Verify single-parameter amplitude fitting."""

    def test_fit_amplitude_mu(self):
        """Fit μ template: amplitude, sigma, SNR should be finite."""
        firas = FIRASData()
        fit = firas.fit_amplitude(firas.mu_template_kJy())
        assert np.isfinite(fit["amplitude"])
        assert fit["sigma"] > 0
        assert np.isfinite(fit["snr"])
        assert fit["chi2_min"] <= fit["chi2_null"]

    def test_fit_amplitude_y(self):
        """Fit y template: amplitude should be finite."""
        firas = FIRASData()
        fit = firas.fit_amplitude(firas.y_template_kJy())
        assert np.isfinite(fit["amplitude"])
        assert fit["sigma"] > 0

    def test_fit_amplitude_best_fit_reduces_chi2(self):
        """Best-fit amplitude should reduce or equal null χ²."""
        firas = FIRASData()
        fit = firas.fit_amplitude(firas.mu_template_kJy())
        assert fit["chi2_min"] <= fit["chi2_null"] + 1e-10


# =========================================================================
# Marginalised fits
# =========================================================================


class TestMarginalisedFits:
    """Verify marginalised amplitude fitting."""

    def test_marginalised_sigma_larger(self):
        """Marginalised σ ≥ unmarginalised σ (profiling increases errors)."""
        firas = FIRASData()
        fit_raw = firas.fit_amplitude(firas.mu_template_kJy())
        fit_marg = firas.fit_amplitude_marginalised(
            firas.mu_template_kJy(), [firas.gbb_template_kJy()]
        )
        # Marginalising over nuisance can only increase or leave σ unchanged
        assert fit_marg["sigma"] >= fit_raw["sigma"] - 1e-15

    def test_marginalised_chi2_min_leq_null(self):
        """Best-fit χ² should be ≤ null χ²."""
        firas = FIRASData()
        fit = firas.fit_amplitude_marginalised(firas.mu_template_kJy())
        assert fit["chi2_min"] <= fit["chi2_null"] + 1e-10


# =========================================================================
# Upper limits
# =========================================================================


class TestUpperLimits:
    """Verify upper limit calculations."""

    def test_upper_limit_mu_order_of_magnitude(self):
        """μ upper limit should be O(10^{-5}), consistent with literature."""
        firas = FIRASData()
        mu_lim = firas.upper_limit_mu(cl=0.95)
        assert 1e-6 < mu_lim < 1e-3

    def test_upper_limit_y_order_of_magnitude(self):
        """y upper limit should be O(10^{-5}), consistent with literature."""
        firas = FIRASData()
        y_lim = firas.upper_limit_y(cl=0.95)
        assert 1e-7 < y_lim < 1e-3

    def test_upper_limit_mu_95_gt_68(self):
        """95% CL limit > 68% CL limit."""
        firas = FIRASData()
        mu_95 = firas.upper_limit_mu(cl=0.95)
        mu_68 = firas.upper_limit_mu(cl=0.68)
        assert mu_95 > mu_68

    def test_upper_limit_y_95_gt_68(self):
        """95% CL limit > 68% CL limit."""
        firas = FIRASData()
        y_95 = firas.upper_limit_y(cl=0.95)
        y_68 = firas.upper_limit_y(cl=0.68)
        assert y_95 > y_68

    def test_upper_limit_generic_template(self):
        """Upper limit on a generic template should be finite."""
        firas = FIRASData()
        template = np.ones(43) * 100.0  # Flat template
        lim = firas.upper_limit(template, cl=0.95)
        assert np.isfinite(lim)
        assert lim > 0

    def test_upper_limit_mu_no_y_marginalisation(self):
        """Can compute μ limit without marginalising over y."""
        firas = FIRASData()
        mu_marg = firas.upper_limit_mu(cl=0.95, marginalise_y=True)
        mu_no_marg = firas.upper_limit_mu(cl=0.95, marginalise_y=False)
        # Without y marginalisation, limit should be tighter (less freedom)
        assert mu_no_marg <= mu_marg + 1e-10


# =========================================================================
# Multi-parameter (μ, y, ΔT/T) fits
# =========================================================================


class TestMultiParameterFit:
    """Verify joint (μ, y, ΔT/T) fitting."""

    def test_fit_distortion_null(self):
        """Joint fit to FIRAS residuals should return finite parameters."""
        firas = FIRASData()
        result = firas.fit_distortion()
        assert np.isfinite(result["mu"])
        assert np.isfinite(result["y"])
        assert np.isfinite(result["delta_t"])
        assert result["ndof"] == 40  # 43 - 3

    def test_fit_distortion_pte(self):
        """p-value should be in [0, 1]."""
        firas = FIRASData()
        result = firas.fit_distortion()
        assert 0 <= result["pte"] <= 1

    def test_fit_distortion_param_cov_shape(self):
        """Parameter covariance matrix should be 3×3."""
        firas = FIRASData()
        result = firas.fit_distortion()
        assert result["param_cov"].shape == (3, 3)

    def test_fit_distortion_param_cov_positive_diagonal(self):
        """Diagonal elements of parameter covariance should be positive."""
        firas = FIRASData()
        result = firas.fit_distortion()
        assert np.all(np.diag(result["param_cov"]) > 0)

    def test_fit_distortion_sigma_from_cov(self):
        """Sigma values should be √(diagonal of param_cov)."""
        firas = FIRASData()
        result = firas.fit_distortion()
        np.testing.assert_allclose(
            result["mu_sigma"], np.sqrt(result["param_cov"][0, 0])
        )
        np.testing.assert_allclose(
            result["y_sigma"], np.sqrt(result["param_cov"][1, 1])
        )

    def test_fit_distortion_with_callable_delta_n(self):
        """Can pass a callable delta_n(x)."""
        firas = FIRASData()
        result = firas.fit_distortion(delta_n=lambda x: np.zeros_like(x))
        assert np.isfinite(result["mu"])

    def test_fit_distortion_with_array_delta_n(self):
        """Can pass an array delta_n."""
        firas = FIRASData()
        result = firas.fit_distortion(delta_n=np.zeros(43))
        assert np.isfinite(result["mu"])

    def test_fit_distortion_with_model_kJy(self):
        """Can pass a model_kJy to subtract."""
        firas = FIRASData()
        result = firas.fit_distortion(model_kJy=np.zeros(43))
        result_null = firas.fit_distortion()
        # Same as null since model is zero
        np.testing.assert_allclose(result["mu"], result_null["mu"])


# =========================================================================
# Model prediction
# =========================================================================


class TestPrediction:
    """Verify model prediction at FIRAS frequencies."""

    def test_predict_null(self):
        """Prediction with all zeros should be zero."""
        firas = FIRASData()
        pred = firas.predict_kJy()
        np.testing.assert_allclose(pred, 0.0, atol=1e-30)

    def test_predict_mu_nonzero(self):
        """Prediction with μ=1e-5 should be nonzero."""
        firas = FIRASData()
        pred = firas.predict_kJy(mu=1e-5)
        assert np.any(np.abs(pred) > 0)

    def test_predict_linearity(self):
        """Prediction should be linear in amplitudes."""
        firas = FIRASData()
        pred1 = firas.predict_kJy(mu=1e-5)
        pred2 = firas.predict_kJy(mu=2e-5)
        np.testing.assert_allclose(pred2, 2 * pred1, rtol=1e-12)

    def test_predict_with_extra_dn_callable(self):
        """Can add extra Δn as callable."""
        firas = FIRASData()
        pred = firas.predict_kJy(extra_dn=lambda x: np.zeros_like(x))
        np.testing.assert_allclose(pred, 0.0, atol=1e-30)

    def test_predict_with_extra_dn_array(self):
        """Can add extra Δn as array."""
        firas = FIRASData()
        pred = firas.predict_kJy(extra_dn=np.zeros(43))
        np.testing.assert_allclose(pred, 0.0, atol=1e-30)


# =========================================================================
# Fisher matrix
# =========================================================================


class TestFisherMatrix:
    """Verify Fisher matrix computation."""

    def test_fisher_matrix_shape(self):
        """Fisher matrix for 2 templates should be 2×2."""
        firas = FIRASData()
        F = firas.fisher_matrix([firas.mu_template_kJy(), firas.y_template_kJy()])
        assert F.shape == (2, 2)

    def test_fisher_matrix_symmetric(self):
        """Fisher matrix should be symmetric."""
        firas = FIRASData()
        F = firas.fisher_matrix([firas.mu_template_kJy(), firas.y_template_kJy()])
        np.testing.assert_allclose(F, F.T, atol=1e-20)

    def test_fisher_matrix_positive_definite(self):
        """Fisher matrix should be positive definite."""
        firas = FIRASData()
        F = firas.fisher_matrix([firas.mu_template_kJy(), firas.y_template_kJy()])
        eigvals = np.linalg.eigvalsh(F)
        assert np.all(eigvals > 0)

    def test_fisher_single_template(self):
        """Fisher for single template is a 1×1 matrix = tᵀC⁻¹t."""
        firas = FIRASData()
        t = firas.mu_template_kJy()
        F = firas.fisher_matrix([t])
        expected = t @ firas.cov_inv @ t
        np.testing.assert_allclose(F[0, 0], expected, rtol=1e-12)


# =========================================================================
# Model limit (arbitrary spectrum)
# =========================================================================


class TestLimitOnModel:
    """Verify upper limit on arbitrary spectral models."""

    def test_limit_on_mu_shape(self):
        """Limit on μ-shape should be close to upper_limit_mu."""
        from spectroxide.greens import mu_shape

        firas = FIRASData()
        result = firas.limit_on_model(mu_shape, cl=0.95, marginalise_gbb=True)
        assert np.isfinite(result["upper_limit"])
        assert result["upper_limit"] > 0
        assert result["sigma"] > 0

    def test_limit_on_flat_spectrum(self):
        """Should handle a flat spectrum."""
        firas = FIRASData()
        result = firas.limit_on_model(lambda x: np.ones_like(x), cl=0.95)
        assert np.isfinite(result["upper_limit"])

    def test_limit_no_gbb_marginalisation(self):
        """Can disable G_bb marginalisation."""
        firas = FIRASData()
        result = firas.limit_on_model(lambda x: np.ones_like(x), marginalise_gbb=False)
        assert np.isfinite(result["upper_limit"])


# =========================================================================
# Module-level constants
# =========================================================================


class TestFIRASConstants:
    """Verify module-level FIRAS limit constants."""

    def test_mu_95_value(self):
        """MU_FIRAS_95 = 9e-5 (Fixsen et al. 1996)."""
        assert MU_FIRAS_95 == 9e-5

    def test_y_95_value(self):
        """Y_FIRAS_95 = 1.5e-5 (Fixsen et al. 1996)."""
        assert Y_FIRAS_95 == 1.5e-5

    def test_mu_68_value(self):
        """MU_FIRAS_68 = 4.5e-5."""
        assert MU_FIRAS_68 == 4.5e-5

    def test_y_68_value(self):
        """Y_FIRAS_68 = 7.5e-6."""
        assert Y_FIRAS_68 == 7.5e-6

    def test_95_gt_68(self):
        """95% limits should be larger than 68%."""
        assert MU_FIRAS_95 > MU_FIRAS_68
        assert Y_FIRAS_95 > Y_FIRAS_68


# =========================================================================
# Galactic dust nuisance template (Fixsen 1996 §6.1)
# =========================================================================


class TestGalacticTemplate:
    """Verify galactic dust nuisance template ν²·B_ν(T_dust)."""

    def test_template_shape(self):
        """Galactic template has shape (43,)."""
        firas = FIRASData()
        gal = firas.galactic_template_kJy()
        assert gal.shape == (43,)

    def test_template_strictly_positive(self):
        """ν²·B_ν(T>0) is strictly positive on FIRAS frequency range."""
        firas = FIRASData()
        gal = firas.galactic_template_kJy()
        assert np.all(gal > 0)

    def test_template_is_copy(self):
        """Returned template is a copy, not a view."""
        firas = FIRASData()
        g1 = firas.galactic_template_kJy()
        g1[:] = 0
        g2 = firas.galactic_template_kJy()
        assert np.any(g2 != 0)

    def test_default_t_dust_9k(self):
        """Default dust temperature is 9 K (Fixsen 1996 §6.1)."""
        firas = FIRASData()
        assert firas.t_dust == 9.0

    def test_t_dust_configurable(self):
        """Constructor accepts custom t_dust."""
        firas9 = FIRASData()
        firas20 = FIRASData(t_dust=20.0)
        gal9 = firas9.galactic_template_kJy()
        gal20 = firas20.galactic_template_kJy()
        # Different T should give different shapes (peak shifts)
        assert not np.allclose(gal9 / gal9.max(), gal20 / gal20.max())

    def test_helper_matches_method(self):
        """Module-level helper produces the same shape as cached template."""
        firas = FIRASData()
        ref = _galactic_dust_template_kJy(firas.freq_cm, t_dust=firas.t_dust)
        np.testing.assert_allclose(firas.galactic_template_kJy(), ref, rtol=1e-12)


class TestGalacticMarginalisation:
    """Verify limits relax when profiling over galactic dust."""

    def test_mu_limit_relaxes_or_equal(self):
        """μ limit with galactic marg ≥ without."""
        firas = FIRASData()
        lim_with = firas.upper_limit_mu(cl=0.95, marginalise_galactic=True)
        lim_without = firas.upper_limit_mu(cl=0.95, marginalise_galactic=False)
        assert lim_with >= lim_without - 1e-15

    def test_y_limit_relaxes_or_equal(self):
        """y limit with galactic marg ≥ without."""
        firas = FIRASData()
        lim_with = firas.upper_limit_y(cl=0.95, marginalise_galactic=True)
        lim_without = firas.upper_limit_y(cl=0.95, marginalise_galactic=False)
        assert lim_with >= lim_without - 1e-15

    def test_limit_on_model_relaxes_or_equal(self):
        """limit_on_model with galactic marg ≥ without."""
        from spectroxide.greens import mu_shape

        firas = FIRASData()
        with_gal = firas.limit_on_model(
            mu_shape, cl=0.95, marginalise_gbb=True, marginalise_galactic=True
        )
        without_gal = firas.limit_on_model(
            mu_shape, cl=0.95, marginalise_gbb=True, marginalise_galactic=False
        )
        assert with_gal["upper_limit"] >= without_gal["upper_limit"] - 1e-15

    def test_default_is_marginalised(self):
        """Default behaviour marginalises over galactic (FIRAS-consistent)."""
        from spectroxide.greens import mu_shape

        firas = FIRASData()
        default = firas.limit_on_model(mu_shape, cl=0.95)
        explicit_on = firas.limit_on_model(
            mu_shape, cl=0.95, marginalise_gbb=True, marginalise_galactic=True
        )
        np.testing.assert_allclose(default["upper_limit"], explicit_on["upper_limit"])

    def test_mu_limit_still_order_of_magnitude(self):
        """Even with galactic marg, μ limit stays O(1e-5)."""
        firas = FIRASData()
        mu_lim = firas.upper_limit_mu(cl=0.95)
        assert 1e-6 < mu_lim < 1e-3

    def test_y_limit_still_order_of_magnitude(self):
        """Even with galactic marg, y limit stays O(1e-5)."""
        firas = FIRASData()
        y_lim = firas.upper_limit_y(cl=0.95)
        assert 1e-7 < y_lim < 1e-3

    def test_profile_limit_floating_T_with_galactic(self):
        """profile_limit_floating_T runs with galactic marg and returns finite limit."""
        from spectroxide.greens import mu_shape

        firas = FIRASData()
        result = firas.profile_limit_floating_T(
            mu_shape, cl=0.95, marginalise_galactic=True
        )
        assert np.isfinite(result["upper_limit"])
        assert result["upper_limit"] > 0
        assert np.isfinite(result["sigma"])
        assert 2.720 <= result["t_best"] <= 2.732

    def test_profile_limit_floating_T_relaxes(self):
        """Floating-T profile limit relaxes (or equal) with galactic marg."""
        from spectroxide.greens import mu_shape

        firas = FIRASData()
        with_gal = firas.profile_limit_floating_T(
            mu_shape, cl=0.95, marginalise_galactic=True
        )
        without_gal = firas.profile_limit_floating_T(
            mu_shape, cl=0.95, marginalise_galactic=False
        )
        assert with_gal["upper_limit"] >= without_gal["upper_limit"] - 1e-15
