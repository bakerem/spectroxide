"""Tests for the CosmoTherm data loader and G_bb stripping functions.

Tests cover unit conversions, number-conserving G_bb stripping, and
CosmoTherm heating rate conversions.
"""

import numpy as np
import pytest

from spectroxide.cosmotherm import (
    di_to_delta_n,
    cosmotherm_gf_to_delta_n,
    strip_gbb,
    _compute_g_bb_jy,
)
from spectroxide.greens import (
    planck,
    g_bb,
    mu_shape,
    y_shape,
    G3_PLANCK,
)

# NumPy compatibility
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# =========================================================================
# Unit conversion: DI → Δn
# =========================================================================


class TestDiToDeltaN:
    """Verify DI [Jy/sr] to occupation number conversion."""

    def test_round_trip_identity(self):
        """Converting Δn → ΔI → Δn should recover the original."""
        x = np.linspace(1.0, 10.0, 100)
        dn_orig = 1e-5 * mu_shape(x)  # A small μ-distortion

        # Δn → ΔI [Jy/sr]
        h = 6.62607015e-34
        k = 1.380649e-23
        c = 2.99792458e8
        t_cmb = 2.726
        nu_hz = x * k * t_cmb / h
        di_si = 2 * h * nu_hz**3 / c**2 * dn_orig  # W/m²/Hz/sr
        di_jy = di_si / 1e-26
        nu_ghz = nu_hz / 1e9

        x_out, dn_out = di_to_delta_n(nu_ghz, di_jy, t_cmb)

        np.testing.assert_allclose(x_out, x, rtol=1e-10)
        np.testing.assert_allclose(dn_out, dn_orig, rtol=1e-10)

    def test_zero_di_gives_zero_dn(self):
        """Zero ΔI should give zero Δn."""
        nu_ghz = np.array([100.0, 200.0, 300.0])
        di_jy = np.zeros(3)
        x, dn = di_to_delta_n(nu_ghz, di_jy)
        np.testing.assert_allclose(dn, 0.0, atol=1e-30)

    def test_positive_di_gives_positive_dn(self):
        """Positive ΔI should give positive Δn."""
        nu_ghz = np.array([100.0, 300.0, 600.0])
        di_jy = np.array([1e6, 1e6, 1e6])  # 1 MJy/sr
        x, dn = di_to_delta_n(nu_ghz, di_jy)
        assert np.all(dn > 0)


class TestCosmothermGFConversion:
    """Verify CosmoTherm GF database unit conversion."""

    def test_round_trip(self):
        """Jy/sr → Δn → Jy/sr should recover original."""
        x = np.array([1.0, 3.0, 5.0, 10.0])
        # Create fake GF values in Jy/sr
        gf_jy_orig = np.array([1e8, 5e8, 2e8, 1e7])
        dn = cosmotherm_gf_to_delta_n(x, gf_jy_orig)

        # Convert back: ΔI = (2hν³/c²) × Δn
        h = 6.62607015e-34
        k = 1.380649e-23
        c = 2.99792458e8
        t_cmb = 2.726
        nu_hz = x * k * t_cmb / h
        di_jy_back = 2 * h * nu_hz**3 / c**2 * dn / 1e-26

        np.testing.assert_allclose(di_jy_back, gf_jy_orig, rtol=1e-10)


# =========================================================================
# G_bb in Jy/sr
# =========================================================================


class TestComputeGbbJy:
    """Verify CosmoTherm-convention G_bb in Jy/sr."""

    def test_positive_at_all_x(self):
        """G_bb should be positive everywhere."""
        x = np.linspace(0.1, 20.0, 200)
        g = _compute_g_bb_jy(x)
        assert np.all(g > 0)

    def test_peaks_around_x4(self):
        """G_bb(x) ∝ x⁴ exp(-x)/(1-exp(-x))² peaks near x≈4."""
        x = np.linspace(0.5, 15.0, 1000)
        g = _compute_g_bb_jy(x)
        x_peak = x[np.argmax(g)]
        assert 3.0 < x_peak < 5.0


# =========================================================================
# strip_gbb: number-conserving stripping
# =========================================================================


class TestStripGbb:
    """Verify number-conserving G_bb stripping."""

    def test_pure_gbb_fully_stripped(self):
        """A pure G_bb signal should be completely removed."""
        x = np.linspace(0.1, 30.0, 5000)
        gbb = g_bb(x)
        alpha_in = 1e-4
        dn = alpha_in * gbb

        dn_nc, alpha_out = strip_gbb(x, dn)

        np.testing.assert_allclose(alpha_out, alpha_in, rtol=1e-3)
        # Stripped spectrum should have near-zero photon number
        n_residual = _trapz(x**2 * dn_nc, x)
        assert abs(n_residual) < 1e-6 * abs(_trapz(x**2 * dn, x))

    def test_mu_shape_mostly_preserved(self):
        """M(x) approximately conserves photon number, so stripping removes little."""
        x = np.linspace(0.1, 30.0, 5000)
        dn = 1e-5 * mu_shape(x)

        dn_nc, alpha = strip_gbb(x, dn)

        # α should be small relative to the μ amplitude
        assert abs(alpha) < 0.1 * 1e-5
        # Spectrum should be mostly unchanged (within 10% of peak)
        peak = np.max(np.abs(dn))
        np.testing.assert_allclose(dn_nc, dn, atol=0.1 * peak)

    def test_y_shape_mostly_preserved(self):
        """Y_SZ(x) approximately conserves photon number, so stripping removes little."""
        x = np.linspace(0.1, 30.0, 5000)
        dn = 1e-5 * y_shape(x)

        dn_nc, alpha = strip_gbb(x, dn)

        assert abs(alpha) < 0.1 * 1e-5
        peak = np.max(np.abs(dn))
        np.testing.assert_allclose(dn_nc, dn, atol=0.1 * peak)

    def test_mixed_signal(self):
        """Strip G_bb from a mixed μ + G_bb signal."""
        x = np.linspace(0.1, 30.0, 5000)
        mu_amp = 2e-5
        alpha_in = 5e-4
        dn = mu_amp * mu_shape(x) + alpha_in * g_bb(x)

        dn_nc, alpha_out = strip_gbb(x, dn)

        np.testing.assert_allclose(alpha_out, alpha_in, rtol=0.05)
        # After stripping, residual should be close to μ × M(x)
        peak = np.max(np.abs(mu_amp * mu_shape(x)))
        np.testing.assert_allclose(dn_nc, mu_amp * mu_shape(x), atol=0.1 * peak)

    def test_number_conservation(self):
        """Stripped spectrum should satisfy ∫x² Δn dx ≈ 0."""
        x = np.linspace(0.1, 30.0, 5000)
        dn = 1e-5 * g_bb(x) + 3e-6 * mu_shape(x) + 1e-6 * y_shape(x)

        dn_nc, _ = strip_gbb(x, dn)

        n_integral = _trapz(x**2 * dn_nc, x)
        assert abs(n_integral) < 1e-8


# =========================================================================
# CosmoTherm heating rate conversions
# =========================================================================


class TestCTHeatingRates:
    """Verify CosmoTherm parameter conversion to heating rates."""

    def test_swave_positive(self):
        """s-wave heating rate should be positive (heating)."""
        from spectroxide.cosmotherm import ct_heating_rate_swave

        rate = ct_heating_rate_swave(z=1e5, f_ann_CT=1e-24)
        assert rate > 0

    def test_swave_scales_with_fann(self):
        """Rate should scale linearly with f_ann_CT."""
        from spectroxide.cosmotherm import ct_heating_rate_swave

        r1 = ct_heating_rate_swave(z=1e5, f_ann_CT=1e-24)
        r2 = ct_heating_rate_swave(z=1e5, f_ann_CT=2e-24)
        np.testing.assert_allclose(r2, 2.0 * r1, rtol=1e-10)

    def test_pwave_vs_swave_ratio(self):
        """p-wave dQ/dz = s-wave dQ/dz × (1+z) because ⟨σv⟩ ∝ v² ∝ T ∝ (1+z)."""
        from spectroxide.cosmotherm import ct_heating_rate_swave, ct_heating_rate_pwave

        z = 1e5
        rs = ct_heating_rate_swave(z, 1e-24)
        rp = ct_heating_rate_pwave(z, 1e-24)
        assert rp > 0
        assert rs > 0
        np.testing.assert_allclose(rp / rs, 1.0 + z, rtol=1e-10)

    def test_decay_positive(self):
        """Decay heating rate should be positive."""
        from spectroxide.cosmotherm import ct_heating_rate_decay

        rate = ct_heating_rate_decay(z=1e5, f_x_eV=1e-3, gamma_x=1e-8)
        assert rate > 0

    def test_decay_exponential_suppression(self):
        """At late times (low z), decay rate should be exponentially small."""
        from spectroxide.cosmotherm import ct_heating_rate_decay

        rate_early = ct_heating_rate_decay(z=1e5, f_x_eV=1e-3, gamma_x=1e-10)
        rate_late = ct_heating_rate_decay(z=1e3, f_x_eV=1e-3, gamma_x=1e-10)
        # Cosmic time at z=1e3 is much later than z=1e5
        assert rate_late < rate_early
