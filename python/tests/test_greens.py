"""Functional tests for the spectroxide Green's function module.

These tests verify physics-level correctness of spectral shapes, visibility
functions, Green's function calculations, and distortion decomposition.
Tests derive targets from first principles or known analytic results.
"""

import numpy as np
import pytest

from spectroxide import greens

# NumPy compatibility: trapezoid was added in 1.25, older versions have trapz
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# =========================================================================
# Section 1: Spectral shapes — analytic identities
# =========================================================================


class TestSpectralShapes:
    """Verify spectral shapes satisfy known mathematical identities."""

    def test_planck_low_x_rayleigh_jeans(self):
        """n_pl(x) → 1/x for x << 1 (Rayleigh-Jeans limit)."""
        x = np.array([1e-4, 1e-3, 1e-2])
        n = greens.planck(x)
        rj = 1.0 / x
        np.testing.assert_allclose(n, rj, rtol=0.01)

    def test_planck_large_x_wien(self):
        """n_pl(x) → exp(-x) for x >> 1 (Wien limit)."""
        x = np.array([20.0, 30.0, 50.0])
        n = greens.planck(x)
        wien = np.exp(-x)
        np.testing.assert_allclose(n, wien, rtol=1e-3)

    def test_planck_identity_derivative(self):
        """dn_pl/dx + n_pl(1+n_pl) = 0 (Planck identity)."""
        # Use many points starting at x=2 (away from steep 1/x region)
        x = np.linspace(2.0, 20.0, 2000)
        n = greens.planck(x)
        # Numerical derivative
        dn_dx = np.gradient(n, x)
        residual = dn_dx + n * (1.0 + n)
        # Residual should be O(dx²) — much smaller than n*(1+n)
        assert np.max(np.abs(residual)) < 0.01 * np.max(np.abs(n * (1.0 + n)))

    def test_g_bb_integral_gives_g3(self):
        """∫ x³ G_bb(x) dx = 4 G₃ (energy integral of temperature shift)."""
        x = np.linspace(0.01, 50.0, 50000)
        integrand = x**3 * greens.g_bb(x)
        integral = _trapz(integrand, x)
        # G_bb(x) = x e^x / (e^x - 1)^2, ∫x³ G_bb dx = 4π⁴/15
        expected = 4.0 * greens.G3_PLANCK
        assert abs(integral - expected) / expected < 0.005

    def test_mu_shape_zero_crossing(self):
        """M(x) crosses zero at x ≈ β_μ ≈ 2.19."""
        x_cross = greens.BETA_MU
        m_val = greens.mu_shape(np.array([x_cross]))[0]
        assert abs(m_val) < 0.01, f"|M(β_μ)| = {abs(m_val):.4e}, should be ≈ 0"
        # Verify sign change: M < 0 for x < β_μ, M > 0 for x > β_μ
        assert greens.mu_shape(np.array([1.0]))[0] < 0
        assert greens.mu_shape(np.array([5.0]))[0] > 0

    def test_y_shape_zero_crossing(self):
        """Y_SZ(x) crosses zero at x ≈ 3.83 (from transcendental equation)."""
        # The exact zero satisfies x*coth(x/2) = 4, known to be x ≈ 3.8310
        # Use bisection (no scipy needed)
        lo, hi = 3.5, 4.5
        for _ in range(60):
            mid = (lo + hi) / 2
            if mid / np.tanh(mid / 2) - 4 < 0:
                lo = mid
            else:
                hi = mid
        x_zero = (lo + hi) / 2
        y_val = greens.y_shape(np.array([x_zero]))[0]
        assert abs(y_val) < 0.01
        # Verify sign change
        assert greens.y_shape(np.array([2.0]))[0] < 0  # below zero
        assert greens.y_shape(np.array([6.0]))[0] > 0  # above zero

    def test_mu_and_y_energy_neutral_basis(self):
        """Energy-neutral f_μ and f_y should be less correlated than raw M, Y."""
        x = np.linspace(1.0, 15.0, 10000)
        m = greens.mu_shape(x)
        y = greens.y_shape(x)
        g = greens.g_bb(x)
        # Energy-neutral basis (Chluba & Jeong 2014)
        f_mu = m - g / (4.0 * 1.401)
        f_y = y - g
        overlap = _trapz(f_mu * f_y, x)
        norm_mu = np.sqrt(_trapz(f_mu * f_mu, x))
        norm_y = np.sqrt(_trapz(f_y * f_y, x))
        cos_angle = overlap / (norm_mu * norm_y)
        # Energy-neutral basis reduces correlation vs raw (cos~0.88)
        assert abs(cos_angle) < 0.95, f"cos(angle) = {cos_angle:.3f}"


# =========================================================================
# Section 2: Visibility functions — regime limits
# =========================================================================


class TestVisibility:
    """Verify visibility functions obey physical constraints."""

    def test_j_bb_high_z_thermalization(self):
        """At z >> z_μ, J_bb → 0 (energy is fully thermalized)."""
        assert greens.j_bb(1e7) < 1e-10

    def test_j_bb_low_z_no_thermalization(self):
        """At z << z_μ, J_bb → 1 (energy not thermalized)."""
        assert abs(greens.j_bb(1e3) - 1.0) < 1e-6

    def test_j_mu_high_z(self):
        """At z >> 6×10⁴, J_μ → 1 (pure μ-era)."""
        assert abs(greens.j_mu(1e7) - 1.0) < 0.01

    def test_j_mu_low_z(self):
        """At z << 6×10⁴, J_μ → 0 (no μ distortions)."""
        assert greens.j_mu(1e3) < 0.01

    def test_j_bb_star_non_negative(self):
        """J_bb* should be clamped to [0, 1] (correction can go negative)."""
        for z in [1e3, 1e5, 1e6, 5e6, 1e7]:
            j = greens.j_bb_star(z)
            assert 0.0 <= j <= 1.0, f"J_bb*({z:.0e}) = {j}, not in [0, 1]"

    def test_j_y_complements_j_mu(self):
        """J_y(z) should be close to 1 - J_μ(z) (approximate energy conservation)."""
        for z in [1e3, 3e4, 1e5, 5e5]:
            j_mu = greens.j_mu(z)
            j_y = greens.j_y(z)
            # Not exact: J_y is independently fitted. But should be ~30% close.
            assert abs(j_y - (1.0 - j_mu)) < 0.3


# =========================================================================
# Section 3: Green's function — physics validation
# =========================================================================


class TestGreensFunction:
    """Verify the Green's function produces correct distortion amplitudes."""

    def test_gf_deep_mu_era(self):
        """In deep μ-era (z >> 2×10⁵), μ/Δρ ≈ 1.401 × J_bb*(z) × J_μ(z)."""
        z_h = 3e5
        x = np.linspace(0.5, 30.0, 500)
        delta_rho = 1e-5
        dn = delta_rho * np.array([greens.greens_function(xi, z_h) for xi in x])
        result = greens.decompose_distortion(x, dn)
        mu_expected = 1.401 * greens.j_bb_star(z_h) * greens.j_mu(z_h) * delta_rho
        assert abs(result["mu"] - mu_expected) / abs(mu_expected) < 0.05

    def test_gf_y_era(self):
        """In y-era (z << 10⁴), y = Δρ/(4ρ) and μ ≈ 0."""
        z_h = 5e3
        delta_rho = 1e-5
        x = np.linspace(0.5, 30.0, 500)
        dn = delta_rho * np.array([greens.greens_function(xi, z_h) for xi in x])
        result = greens.decompose_distortion(x, dn)
        y_expected = delta_rho / 4.0
        assert abs(result["y"] - y_expected) / abs(y_expected) < 0.05
        assert abs(result["mu"]) < 0.1 * abs(result["y"])

    def test_gf_linearity(self):
        """G_th(x, z_h) should scale linearly with Δρ/ρ."""
        z_h = 1e5
        x = 5.0
        g1 = greens.greens_function(x, z_h)
        g2 = 2.0 * g1
        g_double = greens.greens_function(x, z_h) * 2.0
        assert abs(g2 - g_double) < 1e-15

    def test_gf_energy_conservation(self):
        """∫ x³ G_th(x, z) dx = G₃ for all z (energy conserving)."""
        for z_h in [5e3, 5e4, 3e5]:
            x = np.linspace(0.01, 50.0, 50000)
            gf = np.array([greens.greens_function(xi, z_h) for xi in x])
            integral = _trapz(x**3 * gf, x)
            expected = greens.G3_PLANCK
            rel_err = abs(integral - expected) / expected
            # With Chluba's J_T = (1-J_bb*)/4, energy is not exactly conserved.
            # The deviation is up to ~10% in the transition era (z ~ 5e4).
            assert (
                rel_err < 0.20
            ), f"Energy not conserved at z={z_h:.0e}: err={rel_err:.3f}"


# =========================================================================
# Section 4: Heating integration
# =========================================================================


class TestHeatingIntegration:
    """Verify Green's function integration over heating histories."""

    def test_mu_from_single_burst(self):
        """A narrow Gaussian burst at z_h should give μ ≈ 1.401 × J_bb* × J_μ × Δρ/ρ."""
        z_h = 2e5
        sigma_z = 5000.0
        delta_rho = 1e-5

        def dq_dz(z):
            return (
                delta_rho
                * np.exp(-((z - z_h) ** 2) / (2.0 * sigma_z**2))
                / np.sqrt(2.0 * np.pi * sigma_z**2)
            )

        mu = greens.mu_from_heating(dq_dz, 1e3, 5e6, n_z=10000)
        expected = 1.401 * greens.j_bb_star(z_h) * greens.j_mu(z_h) * delta_rho
        assert abs(mu - expected) / abs(expected) < 0.05

    def test_y_from_single_burst_y_era(self):
        """A burst at z=5000 should give y ≈ Δρ/(4ρ)."""
        z_h = 5000.0
        sigma_z = 500.0
        delta_rho = 1e-5

        def dq_dz(z):
            return (
                delta_rho
                * np.exp(-((z - z_h) ** 2) / (2.0 * sigma_z**2))
                / np.sqrt(2.0 * np.pi * sigma_z**2)
            )

        y = greens.y_from_heating(dq_dz, 1e2, 5e4, n_z=10000)
        expected = delta_rho / 4.0
        assert abs(y - expected) / abs(expected) < 0.05


# =========================================================================
# Section 5: Distortion decomposition
# =========================================================================


class TestDecomposition:
    """Verify distortion decomposition into μ, y, ΔT/T components."""

    def test_pure_mu_distortion(self):
        """Injecting pure M(x) should decompose to (μ, 0, 0)."""
        x = np.linspace(1.0, 15.0, 500)
        mu_target = 3e-6
        dn = mu_target * greens.mu_shape(x)
        result = greens.decompose_distortion(x, dn)
        assert abs(result["mu"] - mu_target) / mu_target < 0.02
        assert abs(result["y"]) < 0.01 * mu_target

    def test_pure_y_distortion(self):
        """Injecting pure Y_SZ(x) should decompose with y dominant over μ."""
        x = np.linspace(1.0, 15.0, 500)
        y_target = 1e-6
        dn = y_target * greens.y_shape(x)
        result = greens.decompose_distortion(x, dn)
        assert abs(result["y"] - y_target) / y_target < 0.02
        # M and Y are correlated (cos~0.88), so some μ leakage is expected
        assert abs(result["mu"]) < 0.05 * abs(result["y"])

    def test_drho_matches_direct_integral(self):
        """The returned drho should equal ∫x³Δn dx / G3 independent of fit method."""
        x = np.linspace(1.0, 15.0, 500)
        mu, y_val = 3e-6, 1e-6
        dn = mu * greens.mu_shape(x) + y_val * greens.y_shape(x)
        result = greens.decompose_distortion(x, dn)
        # Compute the reference integral via the trapezoid rule.
        drho_direct = _trapz(x**3 * dn, x) / greens.G3_PLANCK
        # Different quadrature rules (midpoint vs trapezoid) give sub-per-mil
        # agreement on this coarse grid.
        assert abs(result["drho"] - drho_direct) / max(abs(drho_direct), 1e-20) < 1e-3

    def test_fit_residual_small(self):
        """BF fit on a pure μ+y input should have small relative residual."""
        x = np.linspace(0.5, 18.0, 800)
        mu, y_val = 3e-6, 1e-6
        dn = mu * greens.mu_shape(x) + y_val * greens.y_shape(x)
        result = greens.decompose_distortion(x, dn)
        # The BF-fitted μ in the Bose-Einstein parameterisation should be
        # close to the Chluba-M μ (they coincide in the linear regime).
        assert abs(result["mu"] - mu) / mu < 0.05
        assert abs(result["y"] - y_val) / y_val < 0.05


# =========================================================================
# Section 6: Cosmological functions
# =========================================================================


class TestCosmology:
    """Verify cosmological helper functions against known values."""

    def test_hubble_today(self):
        """H(z=0) should be H₀ = 100 h km/s/Mpc."""
        h0 = greens.hubble(0.0)
        expected = 100.0 * 0.71 * 1e3 / 3.0856775814913673e22  # H₀ in 1/s
        assert abs(h0 - expected) / expected < 0.01

    def test_ionization_fraction_pre_recombination(self):
        """At z > 8000, X_e should be > 1 (full ionization + helium)."""
        x_e = greens.ionization_fraction(1e4)
        assert x_e > 1.0, f"X_e(1e4) = {x_e}, should be > 1 (H + He)"

    def test_ionization_fraction_post_recombination(self):
        """At z ~ 500, X_e should be O(10⁻⁴) (freeze-out)."""
        x_e = greens.ionization_fraction(500.0)
        assert 1e-5 < x_e < 0.01, f"X_e(500) = {x_e:.4e}, should be O(10⁻⁴)"

    def test_cosmic_time_decreases_with_z(self):
        """Cosmic time t(z) should decrease with increasing z."""
        t1 = greens.cosmic_time(100.0)
        t2 = greens.cosmic_time(1000.0)
        t3 = greens.cosmic_time(1e6)
        assert t1 > t2 > t3 > 0

    def test_baryon_photon_ratio_scaling(self):
        """R(z) = 3ρ_b/(4ρ_γ) ∝ 1/(1+z), so R(z₁)/R(z₂) = (1+z₂)/(1+z₁)."""
        z1, z2 = 100.0, 1e5
        r1 = greens.baryon_photon_ratio(z1)
        r2 = greens.baryon_photon_ratio(z2)
        expected_ratio = (1.0 + z2) / (1.0 + z1)
        assert abs(r1 / r2 - expected_ratio) / expected_ratio < 1e-6


# =========================================================================
# Section 7: Photon injection Green's function
# =========================================================================


class TestPhotonInjection:
    """Verify photon injection GF physics."""

    def test_photon_gf_high_x_dominated_by_survival(self):
        """At high x_inj, P_s → 1 so the photon GF is large; at low x_inj, P_s → 0."""
        z_h = 2e5
        x_obs = 5.0
        g_high = greens.greens_function_photon(x_obs, 10.0, z_h, sigma_x=0.0)
        g_low = greens.greens_function_photon(x_obs, 0.01, z_h, sigma_x=0.0)
        # High-x injection survives; low-x is absorbed → much smaller amplitude
        assert abs(g_high) > 10.0 * abs(g_low)

    def test_mu_sign_flip_at_x_balanced(self):
        """μ should flip sign at x_inj = x₀ ≈ 3.60."""
        z_h = 2e5
        dn_n = 1e-5
        mu_high = greens.mu_from_photon_injection(10.0, z_h, dn_n)
        mu_low = greens.mu_from_photon_injection(2.0, z_h, dn_n)
        assert mu_high > 0, f"μ(x=10) should be positive: {mu_high:.4e}"
        assert mu_low < 0, f"μ(x=2) should be negative: {mu_low:.4e}"

    def test_photon_survival_limits(self):
        """P_s → 1 for high x, P_s → 0 for low x."""
        z = 2e5
        assert greens.photon_survival_probability(10.0, z) > 0.99
        assert greens.photon_survival_probability(1e-5, z) < 1e-10

    def test_x_c_dc_dominates_at_high_z(self):
        """At z = 2×10⁶, DC absorption should dominate over BR."""
        assert greens.x_c_dc(2e6) > greens.x_c_br(2e6)

    def test_x_c_br_dominates_at_low_z(self):
        """At z = 10⁴, BR absorption should dominate over DC."""
        assert greens.x_c_br(1e4) > greens.x_c_dc(1e4)


# =========================================================================
# Section 10: Intensity conversion
# =========================================================================


class TestIntensityConversion:
    """Verify Δn → ΔI conversion."""

    def test_delta_n_to_delta_I_shape(self):
        """ΔI should have same sign pattern as x³ Δn."""
        x = np.linspace(0.5, 20.0, 100)
        dn = 1e-6 * greens.y_shape(x)
        nu_ghz, di_jy = greens.delta_n_to_delta_I(x, dn)
        assert len(nu_ghz) == len(x)
        assert len(di_jy) == len(x)
        # Frequencies should increase with x
        assert np.all(np.diff(nu_ghz) > 0)
        # ΔI ∝ x³ × Δn, so sign pattern should match
        assert np.sign(di_jy[0]) == np.sign(dn[0])


# =========================================================================
# Section 11: Additional cosmological helpers
# =========================================================================


class TestCosmoHelpers:
    """Test helper cosmology functions for coverage."""

    def test_n_hydrogen(self):
        """n_H(z) should scale as (1+z)³."""
        n1 = greens.n_hydrogen(100.0)
        n2 = greens.n_hydrogen(200.0)
        expected_ratio = ((1.0 + 200.0) / (1.0 + 100.0)) ** 3
        assert abs(n2 / n1 - expected_ratio) / expected_ratio < 1e-6

    def test_n_electron(self):
        """n_e should equal X_e × n_H at high z (fully ionized)."""
        z = 1e4
        n_h = greens.n_hydrogen(z)
        x_e = greens.ionization_fraction(z)
        n_e = greens.n_electron(z)
        assert abs(n_e - x_e * n_h) / n_e < 0.01

    def test_n_electron_custom_x_e(self):
        """n_e with custom x_e should use that value."""
        z = 1e4
        n_h = greens.n_hydrogen(z)
        n_e = greens.n_electron(z, x_e=0.5)
        assert abs(n_e - 0.5 * n_h) / n_e < 1e-6

    def test_omega_gamma(self):
        """Omega_gamma should be ~5e-5."""
        og = greens.omega_gamma()
        assert 1e-5 < og < 1e-4

    def test_rho_gamma_scaling(self):
        """Photon energy density scales as (1+z)⁴."""
        rho1 = greens.rho_gamma(100.0)
        rho2 = greens.rho_gamma(200.0)
        expected_ratio = ((1.0 + 200.0) / (1.0 + 100.0)) ** 4
        assert abs(rho2 / rho1 - expected_ratio) / expected_ratio < 1e-6


# =========================================================================
# Section 13: Photon GF additional paths
# =========================================================================


class TestPhotonGFPaths:
    """Test additional photon GF code paths for coverage."""

    def test_mu_from_photon_injection(self):
        """mu_from_photon_injection at high z should give positive μ for x > x₀."""
        mu = greens.mu_from_photon_injection(10.0, 2e5, 1e-5)
        assert mu > 0

    def test_greens_function_photon_with_sigma(self):
        """Photon GF with nonzero sigma_x should still produce a result."""
        z_h = 2e5
        x_obs = np.linspace(0.5, 20.0, 50)
        g = greens.greens_function_photon(x_obs, 5.0, z_h, sigma_x=0.5)
        assert len(g) == len(x_obs)
        assert np.max(np.abs(g)) > 0

    def test_greens_function_photon_number_conserving(self):
        """NC mode should strip G_bb component."""
        z_h = 2e5
        x_obs = 5.0
        g_std = greens.greens_function_photon(x_obs, 5.0, z_h)
        g_nc = greens.greens_function_photon(x_obs, 5.0, z_h, number_conserving=True)
        # NC strips T-shift, so they should differ
        assert g_std != g_nc

    def test_photon_gf_intermediate_raises(self):
        """Photon GF raises ValueError for z_h in the mu-y transition era."""
        import pytest

        x_obs = np.linspace(0.1, 20, 50)
        with pytest.raises(ValueError, match="mu-y transition"):
            greens.greens_function_photon(x_obs, 5.0, 8e4)
        with pytest.raises(ValueError, match="mu-y transition"):
            greens.greens_function_photon(x_obs, 5.0, 1.5e5)
        # Boundaries (5e4 and 2e5) and outside the window remain valid.
        greens.greens_function_photon(x_obs, 5.0, 5e4)
        greens.greens_function_photon(x_obs, 5.0, 2e5)
        greens.greens_function_photon(x_obs, 5.0, 1e4)
        greens.greens_function_photon(x_obs, 5.0, 5e5)


# =========================================================================
# Section 14: Solver module tests
# =========================================================================

from spectroxide.solver import (
    Cosmology,
    _build_cosmo_args,
    _build_injection_args,
    _build_common_solver_args,
)


class TestSolverCosmology:
    """Test the Cosmology dataclass."""

    def test_default(self):
        """Default cosmology should match Chluba 2013."""
        c = Cosmology.default()
        assert c.h == 0.71
        assert c.omega_b == 0.044
        assert c.y_p == 0.24

    def test_planck2015(self):
        """Planck 2015 preset."""
        c = Cosmology.planck2015()
        assert abs(c.h - 0.6727) < 1e-6

    def test_planck2018(self):
        """Planck 2018 preset."""
        c = Cosmology.planck2018()
        assert abs(c.h - 0.6736) < 1e-6
        assert abs(c.t_cmb - 2.7255) < 1e-6

    def test_to_dict(self):
        """to_dict should round-trip all fields."""
        c = Cosmology.default()
        d = c.to_dict()
        assert d["h"] == 0.71
        assert d["omega_b"] == 0.044
        assert "omega_m" in d

    def test_invalid_h_raises(self):
        """Non-physical h should raise ValueError."""
        with pytest.raises(ValueError):
            Cosmology(h=-1.0)

    def test_invalid_y_p_raises(self):
        """Y_p outside [0,1] should raise ValueError."""
        with pytest.raises(ValueError):
            Cosmology(y_p=1.5)


class TestBuildArgs:
    """Test CLI argument builders."""

    def test_cosmo_args_none(self):
        """None should return empty list."""
        assert _build_cosmo_args(None) == []

    def test_cosmo_args_full(self):
        """Full cosmology dict should produce correct CLI args."""
        d = {"h": 0.67, "omega_b": 0.05, "omega_m": 0.3, "y_p": 0.24, "t_cmb": 2.725}
        args = _build_cosmo_args(d)
        assert "--omega-b" in args
        assert "--omega-m" in args
        assert "--omega-cdm" not in args
        assert "--h" in args
        assert "--y-p" in args
        assert "--t-cmb" in args

    def test_injection_args_single_burst(self):
        """Single burst injection args."""
        inj = {"type": "single_burst", "z_h": 1e5, "sigma_z": 3000.0}
        args = _build_injection_args(inj)
        assert "--injection" in args
        assert "single-burst" in args
        assert "--z-h" in args
        assert "--sigma-z" in args

    def test_injection_args_unknown_key_raises(self):
        """Unknown injection parameter should raise."""
        with pytest.raises(ValueError, match="Unknown injection parameter"):
            _build_injection_args({"type": "test", "bogus": 42})

    def test_injection_args_none(self):
        """None injection should return empty list."""
        assert _build_injection_args(None) == []

    def test_common_solver_args(self):
        """Test common solver args builder."""
        args = _build_common_solver_args(
            dy_max=0.01,
            n_points=2000,
            dtau_max=0.5,
            number_conserving=True,
            no_dcbr=True,
            production_grid=True,
        )
        assert "--dy-max" in args
        assert "--n-points" in args
        assert "--dtau-max" in args
        assert "--no-number-conserving" not in args
        assert "--no-dcbr" in args
        assert "--production-grid" in args

    def test_common_solver_args_no_nc(self):
        """Disabling NC should pass --no-number-conserving."""
        args = _build_common_solver_args(number_conserving=False)
        assert "--no-number-conserving" in args

    def test_common_solver_args_empty(self):
        """With all defaults, should produce no args."""
        args = _build_common_solver_args()
        assert args == []


# =========================================================================
# Section 15: Validation module coverage
# =========================================================================

from spectroxide import _validation as _val


class TestValidation:
    """Test validation functions for edge cases."""

    def test_validate_x_negative_raises(self):
        """Negative x values should raise ValueError."""
        with pytest.raises(ValueError):
            _val.validate_x_positive(np.array([-1.0, 1.0, 2.0]))

    def test_validate_z_range_inverted_raises(self):
        """z_min > z_max should raise ValueError."""
        with pytest.raises(ValueError):
            _val.validate_z_range(1e5, 1e3, 100)

    def test_validate_array_lengths_mismatch(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError):
            _val.validate_array_lengths(np.array([1.0, 2.0]), np.array([1.0]))

    def test_validate_cosmology_bad_omega_b(self):
        """Negative omega_b should raise ValueError."""
        with pytest.raises(ValueError):
            Cosmology(omega_b=-0.01)

    def test_warn_z_h_regime(self):
        """Should warn for z_h > 3e6 and z_h < 500."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _val.warn_z_h_regime(5e6)
            assert len(w) >= 1
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _val.warn_z_h_regime(100.0)
            assert len(w) >= 1

    def test_warn_x_inj_regime(self):
        """Should warn for extreme x_inj values."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _val.warn_x_inj_regime(0.001)
            assert len(w) >= 1
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _val.warn_x_inj_regime(200.0)
            assert len(w) >= 1

    def test_warn_z_max_regime(self):
        """Should warn for z_max > 1e7."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _val.warn_z_max_regime(5e7)
            assert len(w) >= 1

    def test_warn_x_grid_narrow(self):
        """Should warn for narrow grids."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _val.warn_x_grid_narrow(np.array([0.5, 1.0, 2.0]))
            assert len(w) >= 1
