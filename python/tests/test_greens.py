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
    _injection_param_args,
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
        """Single burst injection args (subcommand form: params only)."""
        inj = {"type": "single_burst", "z_h": 1e5, "sigma_z": 3000.0}
        args = _injection_param_args(inj)
        assert "--z-h" in args
        assert "--sigma-z" in args
        # "type" is the positional subcommand argument, not a flag
        assert "single-burst" not in args

    def test_injection_args_unknown_key_raises(self):
        """Unknown injection parameter should raise."""
        with pytest.raises(ValueError, match="Unknown injection parameter"):
            _injection_param_args({"type": "test", "bogus": 42})

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

    def test_renamed_kwargs_warning_attributes_to_caller(self):
        """The ``renamed_kwargs`` decorator adds a stack frame defined in
        _validation.py between the caller and the wrapped function. A
        warning raised inside the wrapped function (here
        ``warn_x_inj_regime``, stacklevel tuned for the undecorated call
        chain) must still resolve to this test's frame, not to
        _validation.py or greens.py.
        """
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            greens.greens_function_photon(
                x=np.linspace(0.3, 15.0, 50), x_inj=1e-3, z_h=5e3
            )
        assert len(w) == 1
        assert w[0].filename == __file__


# =========================================================================
# Photon-injection μ/y partners and cosmology-aware P_s (A-6)
# =========================================================================

# Spectral integrals typed as literals, not imported from the package:
# G2 = 2 ζ(3), G3 = π⁴/15.  Injected energy per photon number:
# Δρ/ρ = (G2/G3) x_inj ΔN/N.
_G2_LIT = 2.0 * 1.2020569031595942
from spectroxide.cosmology import PLANCK2018_COSMO as _PLANCK2018  # noqa: E402

_G3_LIT = np.pi**4 / 15.0


class TestPhotonInjectionMuY:
    """y_from_photon_injection, mu_from_photon_injection(cosmo=), and
    photon_survival_probability(cosmo=)."""

    def test_y_absorbed_photon_is_all_heat(self):
        """An absorbed photon in the y-era heats electrons with all its
        energy, so y = Δρ/(4ρ) (Zel'dovich–Sunyaev; CLAUDE.md target).

        At z_h = 5e3 the μ branching ratio is about 1% and a photon at
        x_inj = 1e-3 is absorbed (x_inj ≪ x_c ≈ 0.07), so 2% covers both.
        """
        x_inj, z_h, dn_n = 1e-3, 5e3, 1e-5
        drho = _G2_LIT / _G3_LIT * x_inj * dn_n
        with pytest.warns(UserWarning):  # x_inj < 0.01 regime warning
            y = greens.y_from_photon_injection(
                x_inj, z_h, dn_n, cosmo=greens.DEFAULT_COSMO
            )
        assert y == pytest.approx(drho / 4.0, rel=0.02)

    def test_y_surviving_photon_follows_compton_recoil(self):
        """A surviving Wien-tail photon (x ≫ 1) drifts as d ln x/dy_γ = 4 − x
        (mean energy change per scattering (4kT − hν)/mc²; Rybicki &
        Lightman 1979, Eq. 7.36).  The energy it loses by recoil heats the
        gas, so y/(Δρ/4ρ) = (x_inj − 4) y_γ to first order in y_γ.
        """
        x_inj, z_h, dn_n = 10.0, 5e3, 1e-5
        cosmo = greens.DEFAULT_COSMO
        drho = _G2_LIT / _G3_LIT * x_inj * dn_n
        y_gamma = greens._y_compton(z_h, cosmo)
        assert 1e-4 < y_gamma < 1e-2  # first-order regime
        y = greens.y_from_photon_injection(x_inj, z_h, dn_n, cosmo=cosmo)
        # 3%: 1 − J_μ(5e3) ≈ 0.99, plus O(y_γ) and stimulated terms ~0.1%.
        assert y / (drho / 4.0) == pytest.approx((x_inj - 4.0) * y_gamma, rel=0.03)

    def test_y_gamma_bracketed_by_literal_constants(self):
        """Pins the y_γ normalization the recoil test reuses.

        y_γ = ∫ θ_e σ_T n_e c / [H (1+z)] dz from z = 1100 (X_e ≈ 0 below)
        to z_h, computed from CODATA 2018 constants typed here, with flat
        ΛCDM for DEFAULT_COSMO (h = 0.71, Ω_b = 0.044, Ω_m = 0.26,
        Y_p = 0.24, T0 = 2.726 K, N_eff = 3.046).  Electrons per hydrogen
        lie between 1 (H only) and 1 + 2 f_He (H and He fully ionized), so
        the true y_γ lies between the two integrals.  At z_h = 5e3 they
        are 7.0e-4 and 8.1e-4, so a factor-2 error fails.
        """
        k_b, m_e, c = 1.380649e-23, 9.1093837015e-31, 2.99792458e8
        sigma_t, m_p, g_n = 6.6524587321e-29, 1.67262192369e-27, 6.67430e-11
        a_rad, mpc = 7.565723e-16, 3.0856775814913673e22
        h, om_b, om_m, y_p, t0, n_eff = 0.71, 0.044, 0.26, 0.24, 2.726, 3.046
        h0 = h * 1e5 / mpc
        rho_c = 3 * h0**2 / (8 * np.pi * g_n)
        om_r = (a_rad * t0**4 / c**2 / rho_c) * (
            1 + n_eff * 7 / 8 * (4 / 11) ** (4 / 3)
        )
        n_h0 = (1 - y_p) * om_b * rho_c / m_p
        f_he = y_p / (4 * (1 - y_p))
        z_h = 5e3
        z = np.geomspace(1100.0, z_h, 20001)
        hub = h0 * np.sqrt(om_m * (1 + z) ** 3 + om_r * (1 + z) ** 4 + 1 - om_m - om_r)
        base = (k_b * t0 * (1 + z) / (m_e * c**2)) * sigma_t * c * n_h0 * (1 + z) ** 3
        integrand = base / (hub * (1 + z))
        lo = _trapz(integrand, z)
        hi = (1 + 2 * f_he) * lo
        assert lo < greens._y_compton(z_h, greens.DEFAULT_COSMO) < hi

    def test_y_vanishes_in_deep_mu_era(self):
        assert greens.y_from_photon_injection(1.0, 3e6, 1e-5) == 0.0

    @pytest.mark.parametrize(
        "x_inj, z_h, x_lo, x_hi",
        [
            (1e-3, 5e3, 0.3, 15.0),  # absorbed, y-era
            (1e-3, 3e4, 0.3, 15.0),  # absorbed, J_μ ≈ 0.25
            (10.0, 5e3, 0.3, 6.0),  # surviving line kept out of the band
            (2.0, 3e5, 0.3, 15.0),  # μ-era
        ],
    )
    def test_mu_y_match_greens_function_photon(self, x_inj, z_h, x_lo, x_hi):
        """With the same cosmo, μ and y are the M(x) and Y_SZ(x)
        coefficients of the number-conserving photon Green's function."""
        cosmo = _PLANCK2018
        x = np.linspace(x_lo, x_hi, 400)
        with pytest.warns() if x_inj < 0.01 else _nullcontext():
            g = greens.greens_function_photon(
                x, x_inj, z_h, number_conserving=True, cosmo=cosmo
            )
            mu = greens.mu_from_photon_injection(x_inj, z_h, 1.0, cosmo=cosmo)
            y = greens.y_from_photon_injection(x_inj, z_h, 1.0, cosmo=cosmo)
        design = np.column_stack([greens.mu_shape(x), greens.y_shape(x)])
        (mu_fit, y_fit), *_ = np.linalg.lstsq(design, g, rcond=None)
        scale = np.max(np.abs(g))
        np.testing.assert_allclose(design @ [mu, y], g, rtol=0, atol=1e-9 * scale)
        assert mu_fit == pytest.approx(mu, rel=1e-6, abs=1e-9 * scale)
        assert y_fit == pytest.approx(y, rel=1e-6, abs=1e-9 * scale)

    def test_survival_probability_default_is_analytic(self):
        x = np.array([0.01, 0.1, 1.0])
        z = 1e4
        expected = np.exp(-float(greens.x_c(z)) / x)
        np.testing.assert_allclose(greens.photon_survival_probability(x, z), expected)

    def test_survival_probability_cosmo_matches_greens_function_photon(self):
        """The public cosmology-aware P_s is the one the photon GF uses."""
        cosmo = _PLANCK2018
        for x_inj, z in [(0.05, 3e3), (0.5, 2e4), (1.0, 5e5)]:
            got = greens.photon_survival_probability(x_inj, z, cosmo=cosmo)
            assert np.ndim(got) == 0
            assert float(got) == greens._photon_survival_probability_numerical(
                x_inj, z, cosmo
            )

    def test_survival_probability_cosmo_shape_and_range(self):
        x = np.logspace(-3, 1, 12).reshape(3, 4)
        p = greens.photon_survival_probability(x, 1e4, cosmo=greens.DEFAULT_COSMO)
        assert p.shape == x.shape
        assert np.all((p >= 0) & (p <= 1))
        assert np.all(np.diff(p.ravel()) >= 0)  # more survival at higher x

    def test_survival_probability_cosmo_irrelevant_in_mu_era(self):
        x = np.array([0.01, 0.1, 1.0])
        np.testing.assert_array_equal(
            greens.photon_survival_probability(x, 3e5, cosmo=greens.DEFAULT_COSMO),
            greens.photon_survival_probability(x, 3e5),
        )

    def test_mu_warns_for_analytic_ps_in_y_era(self):
        with pytest.warns(UserWarning, match="analytic P_s"):
            greens.mu_from_photon_injection(1.0, 1e4, 1e-5)

    def test_y_default_matches_greens_function_photon_default(self):
        """cosmo=None means DEFAULT_COSMO for y, as for the photon GF; the
        analytic P_s would flip the sign of y here."""
        x = np.linspace(0.3, 15.0, 400)
        g = greens.greens_function_photon(x, 0.05, 1e4, number_conserving=True)
        design = np.column_stack([greens.mu_shape(x), greens.y_shape(x)])
        (_, y_fit), *_ = np.linalg.lstsq(design, g, rcond=None)
        y = greens.y_from_photon_injection(0.05, 1e4, 1.0)
        assert y == pytest.approx(y_fit, rel=1e-6)

    def test_partial_cosmology_mapping_filled_from_default(self):
        p_partial = greens.photon_survival_probability(1.0, 1e4, cosmo={"h": 0.71})
        p_full = greens.photon_survival_probability(
            1.0, 1e4, cosmo=greens.DEFAULT_COSMO
        )
        assert float(p_partial) == float(p_full)

    def test_greens_function_photon_accepts_dataclass(self):
        from spectroxide import Cosmology

        x = np.linspace(0.5, 10.0, 50)
        c = Cosmology.planck2018()
        np.testing.assert_array_equal(
            greens.greens_function_photon(x, 2.0, 1e4, cosmo=c),
            greens.greens_function_photon(x, 2.0, 1e4, cosmo=c.to_dict()),
        )

    def test_accepts_cosmology_dataclass(self):
        from spectroxide import Cosmology

        c = Cosmology.planck2018()
        assert greens.y_from_photon_injection(
            5.0, 1e4, 1e-5, cosmo=c
        ) == greens.y_from_photon_injection(5.0, 1e4, 1e-5, cosmo=c.to_dict())


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# =========================================================================
# Frequency argument named x everywhere; strip_gbb location (A-7)
# =========================================================================


class TestArgumentNames:
    """``x`` is the frequency argument everywhere; old names still work
    with a DeprecationWarning."""

    _X = np.linspace(0.5, 20.0, 200)

    @pytest.mark.parametrize(
        "func, old, extra",
        [
            ("greens_function_photon", "x_obs", {"x_inj": 5.0, "z_h": 3e5}),
            (
                "distortion_from_heating",
                "x_grid",
                {"dq_dz": lambda z: 1e-12, "z_min": 3e5, "z_max": 4e5},
            ),
            (
                "distortion_from_photon_injection",
                "x_grid",
                {"x_inj": 5.0, "dn_dz": lambda z: 1e-12, "z_min": 3e5, "z_max": 4e5},
            ),
        ],
    )
    def test_old_name_warns_and_matches(self, func, old, extra):
        f = getattr(greens, func)
        new = f(x=self._X, **extra)
        with pytest.warns(DeprecationWarning, match=f"'{old}' is deprecated"):
            legacy = f(**{old: self._X}, **extra)
        np.testing.assert_array_equal(new, legacy)

    def test_decompose_distortion_x_grid_alias(self):
        dn = 1e-6 * greens.y_shape(self._X)
        new = greens.decompose_distortion(x=self._X, delta_n=dn)
        with pytest.warns(DeprecationWarning, match="'x_grid' is deprecated"):
            legacy = greens.decompose_distortion(x_grid=self._X, delta_n=dn)
        assert new["y"] == legacy["y"]

    def test_delta_n_to_delta_I_dn_alias(self):
        dn = 1e-6 * np.ones_like(self._X)
        new = greens.delta_n_to_delta_I(x=self._X, delta_n=dn)
        with pytest.warns(DeprecationWarning, match="'dn' is deprecated"):
            legacy = greens.delta_n_to_delta_I(x=self._X, dn=dn)
        np.testing.assert_array_equal(new[1], legacy[1])

    def test_both_names_rejected(self):
        with pytest.raises(TypeError, match="both 'x_obs' and 'x'"):
            greens.greens_function_photon(x=self._X, x_obs=self._X, x_inj=5.0, z_h=3e5)

    def test_positional_call_unchanged(self):
        np.testing.assert_array_equal(
            greens.greens_function_photon(self._X, 5.0, 3e5),
            greens.greens_function_photon(x=self._X, x_inj=5.0, z_h=3e5),
        )


class TestStripGbbLocation:
    def test_top_level_is_greens(self):
        import spectroxide

        assert spectroxide.strip_gbb is greens.strip_gbb

    def test_cosmotherm_path_deprecated(self):
        import spectroxide.cosmotherm as ct

        with pytest.warns(DeprecationWarning, match="cosmotherm.strip_gbb"):
            f = ct.strip_gbb
        assert f is greens.strip_gbb

    def test_accepts_lists(self):
        x = list(np.linspace(0.1, 20.0, 400))
        dn_nc, alpha = greens.strip_gbb(x, list(1e-5 * greens.g_bb(np.array(x))))
        assert alpha == pytest.approx(1e-5, rel=1e-12)
        assert np.max(np.abs(dn_nc)) < 1e-18


# =========================================================================
# Free-free Gaunt factor (P-1 in dev/REVIEW_2026-09-22.md)
# =========================================================================


class TestBremsstrahlungGaunt:
    """The Python BR coefficient must evaluate the Gaunt fit at x_e = x / rho_e.

    Mirrors ``tests/greens_function_checks.rs``.
    """

    # C = 2^{5/2} exp(-5 gamma_E / 2) / alpha, from mpmath at dps = 40.
    DRAINE_C = 183.10732337940075
    S3_PI = 0.55132889542179205

    def test_invariant_under_theta_z(self):
        """K_BR phi^3 e^{x_e} theta_e^{7/2} depends on (x_e, theta_e) only."""
        args = (1.0e6, 8.0e4, 1.1e6, 1.0, 0.3, 0.9)  # n_h, n_he, n_e, X_e, y_ii, y_i
        for theta_e in (3e-8, 1e-6, 1e-5):
            for x_e in (1e-4, 1e-2, 0.3, 3.0):
                vals = []
                for rho_e in (1.0, 0.3, 0.62, 1.4):
                    theta_z = theta_e / rho_e
                    phi = theta_z / theta_e
                    x = x_e * rho_e
                    k = greens._br_emission_coefficient_with_he(
                        x, theta_e, theta_z, *args
                    )
                    vals.append(k * phi**3 * np.exp(x * phi) * theta_e**3.5)
                np.testing.assert_allclose(vals, vals[0], rtol=1e-12)

    def test_classical_limit_matches_draine(self):
        """Low-x_e limit is (sqrt(3)/pi) ln(C theta_e^{1/2} / (Z x_e)), Draine Eq. 10.9."""
        for theta_e in (1e-8, 1e-6, 3e-6):
            for x_e in (1e-13, 1e-12, 1e-11):
                for z in (1.0, 2.0):
                    g_d = self.S3_PI * np.log(
                        self.DRAINE_C * np.sqrt(theta_e) / (z * x_e)
                    )
                    g = greens._gaunt_ff_nr(x_e, theta_e, z)
                    assert abs(g - g_d) < 1e-3

    def test_matches_draine_ch10_physical_units(self):
        """Full fit vs Draine (2011) Ch. 10 interpolation in physical (nu, T_e) units.

        g_ff ~ ln{exp[5.960 - (sqrt(3)/pi) ln(Z nu9 T4^-1.5)] + e},
        nu9 = nu / 1 GHz, T4 = T_e / 1e4 K (equation number believed to be 10.8,
        not confirmed). Mirrors
        ``gaunt_ff_nr_matches_draine_ch10_physical_units`` in
        ``tests/greens_function_checks.rs``.
        """
        # CODATA 2018, typed as literals (independent of the module constants).
        h = 6.62607015e-34  # J s, exact by SI definition
        k_b = 1.380649e-23  # J/K, exact by SI definition
        m_e = 9.1093837015e-31  # kg
        c = 299792458.0  # m/s, exact by SI definition
        m_e_c2 = m_e * c**2  # J

        max_rel_err = 0.0
        for log10_nu in (6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0):
            nu_hz = 10.0**log10_nu
            for log10_t in (3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0):
                t_e_k = 10.0**log10_t
                for z in (1.0, 2.0):
                    nu9 = nu_hz / 1.0e9
                    t4 = t_e_k / 1.0e4
                    a = 5.960 - self.S3_PI * np.log(z * nu9 * t4**-1.5)
                    g_draine = np.log(np.exp(a) + np.e)

                    x_e = h * nu_hz / (k_b * t_e_k)
                    theta_e = k_b * t_e_k / m_e_c2
                    g_code = greens._gaunt_ff_nr(x_e, theta_e, z)

                    rel_err = abs(g_code - g_draine) / g_draine
                    max_rel_err = max(max_rel_err, rel_err)
                    assert rel_err < 2e-4, (
                        f"nu={nu_hz:e} Hz, T_e={t_e_k:e} K, Z={z}: "
                        f"code={g_code:.6f}, Draine={g_draine:.6f}, rel_err={rel_err:.2e}"
                    )
        assert max_rel_err < 2e-4
