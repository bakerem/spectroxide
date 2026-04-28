"""Tests for the solver module's Python-side logic.

Tests cover SolverResult, quality presets, argument building helpers,
and the solve() dispatcher (Green's function mode only — PDE mode
requires the compiled Rust binary).
"""

import numpy as np
import pytest

from spectroxide.solver import (
    Cosmology,
    SolverResult,
    PRODUCTION,
    DEBUG,
    _apply_settings,
    _resolve_quality_settings,
    _build_cosmo_args,
    _build_injection_args,
    _build_common_solver_args,
    run_single,
    solve,
)

# =========================================================================
# Quality presets
# =========================================================================


class TestQualityPresets:
    """Verify PRODUCTION and DEBUG preset dictionaries."""

    def test_production_n_points(self):
        """PRODUCTION uses 4000 grid points."""
        assert PRODUCTION["n_points"] == 4000

    def test_production_grid(self):
        """PRODUCTION uses production grid."""
        assert PRODUCTION["production_grid"] is True

    def test_debug_n_points(self):
        """DEBUG uses 1000 grid points."""
        assert DEBUG["n_points"] == 1000

    def test_debug_grid(self):
        """DEBUG does not use production grid."""
        assert DEBUG["production_grid"] is False

    def test_apply_settings_default_production(self):
        """Default settings should merge with PRODUCTION."""
        merged = _apply_settings({})
        assert merged["n_points"] == 4000

    def test_apply_settings_debug(self):
        """debug=True should merge with DEBUG."""
        merged = _apply_settings({}, debug=True)
        assert merged["n_points"] == 1000

    def test_apply_settings_override(self):
        """Explicit kwargs should override preset."""
        merged = _apply_settings({"n_points": 2000})
        assert merged["n_points"] == 2000

    def test_apply_settings_none_values_ignored(self):
        """None values should not override preset."""
        merged = _apply_settings({"n_points": None})
        assert merged["n_points"] == 4000

    def test_resolve_quality_settings(self):
        """Resolve should return (n_points, production_grid, dtau_max)."""
        n, pg, dtau = _resolve_quality_settings(None, None, None, False)
        assert n == 4000
        assert pg is True

    def test_resolve_quality_settings_debug(self):
        """Debug mode should return DEBUG values."""
        n, pg, dtau = _resolve_quality_settings(None, None, None, True)
        assert n == 1000
        assert pg is False


# =========================================================================
# Cosmology argument building
# =========================================================================


class TestBuildCosmoArgs:
    """Verify CLI argument generation from cosmology dicts."""

    def test_none_returns_empty(self):
        """None cosmo_params → empty args list."""
        assert _build_cosmo_args(None) == []

    def test_h_passthrough(self):
        """h should be passed through as --h."""
        args = _build_cosmo_args({"h": 0.6736})
        assert "--h" in args
        assert args[args.index("--h") + 1] == "0.6736"

    def test_omega_b_passthrough(self):
        """omega_b is a fraction now; the wrapper should forward it verbatim."""
        args = _build_cosmo_args({"h": 0.71, "omega_b": 0.044, "omega_m": 0.26})
        assert "--omega-b" in args
        assert args[args.index("--omega-b") + 1] == "0.044"

    def test_omega_m_passthrough(self):
        """omega_m forwards directly to --omega-m (the CLI computes ω_cdm)."""
        args = _build_cosmo_args({"h": 0.71, "omega_b": 0.044, "omega_m": 0.26})
        assert "--omega-m" in args
        assert args[args.index("--omega-m") + 1] == "0.26"
        # --omega-cdm is no longer emitted.
        assert "--omega-cdm" not in args

    def test_omega_b_without_m_raises(self):
        """omega_b/omega_m must be paired; the Rust CLI rejects single inputs."""
        with pytest.raises(ValueError, match="must be passed together"):
            _build_cosmo_args({"omega_b": 0.044})
        with pytest.raises(ValueError, match="must be passed together"):
            _build_cosmo_args({"omega_m": 0.26})

    def test_y_p_passthrough(self):
        """y_p should be passed through."""
        args = _build_cosmo_args({"y_p": 0.24})
        assert "--y-p" in args

    def test_cosmology_object_to_dict(self):
        """Cosmology.to_dict() should produce valid args."""
        cosmo = Cosmology.planck2018()
        args = _build_cosmo_args(cosmo.to_dict())
        assert "--h" in args
        assert "--omega-b" in args


# =========================================================================
# Injection argument building
# =========================================================================


class TestBuildInjectionArgs:
    """Verify injection scenario CLI argument generation."""

    def test_single_burst(self):
        """SingleBurst should produce --injection single-burst args."""
        args = _build_injection_args(
            {
                "type": "single_burst",
                "z_h": 5e4,
            }
        )
        assert "--injection" in args
        assert "single-burst" in args
        assert "--z-h" in args

    def test_decaying_particle(self):
        """DecayingParticle should produce correct args."""
        args = _build_injection_args(
            {
                "type": "decaying_particle",
                "f_x": 1e-3,
                "gamma_x": 1e-8,
            }
        )
        assert "decaying-particle" in args
        assert "--f-x" in args

    def test_unknown_param_raises(self):
        """Unknown injection parameter should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown injection parameter"):
            _build_injection_args({"type": "single_burst", "bogus_param": 42})

    def test_none_returns_empty(self):
        """None injection → empty args."""
        assert _build_injection_args(None) == []


# =========================================================================
# SolverResult
# =========================================================================


class TestSolverResult:
    """Verify SolverResult dataclass."""

    def test_construction(self):
        """Can construct a SolverResult."""
        x = np.linspace(0.1, 20.0, 100)
        dn = np.zeros(100)
        result = SolverResult(
            x=x,
            delta_n=dn,
            mu=0.0,
            y=0.0,
            delta_rho_over_rho=0.0,
            method="greens_function",
        )
        assert result.method == "greens_function"
        assert result.z_h is None

    def test_delta_I_property(self):
        """delta_I should convert Δn to (nu_ghz, intensity) tuple."""
        x = np.linspace(1.0, 10.0, 50)
        dn = 1e-5 * np.ones(50)
        result = SolverResult(
            x=x,
            delta_n=dn,
            mu=0.0,
            y=0.0,
            delta_rho_over_rho=0.0,
            method="greens_function",
        )
        output = result.delta_I
        # delta_n_to_delta_I returns (nu_ghz, delta_I)
        assert isinstance(output, tuple)
        nu_ghz, dI = output
        assert len(nu_ghz) == 50
        assert len(dI) == 50
        assert np.all(np.isfinite(dI))
        # Positive Δn → positive ΔI
        assert np.all(dI > 0)

    def test_z_h_optional(self):
        """z_h should be optional and default to None."""
        x = np.linspace(0.1, 20.0, 10)
        result = SolverResult(
            x=x,
            delta_n=np.zeros(10),
            mu=0.0,
            y=0.0,
            delta_rho_over_rho=0.0,
            method="pde",
        )
        assert result.z_h is None

    def test_z_h_set(self):
        """z_h can be set at construction."""
        x = np.linspace(0.1, 20.0, 10)
        result = SolverResult(
            x=x,
            delta_n=np.zeros(10),
            mu=0.0,
            y=0.0,
            delta_rho_over_rho=0.0,
            method="greens_function",
            z_h=5e4,
        )
        assert result.z_h == 5e4


# =========================================================================
# solve() — Green's function mode
# =========================================================================


class TestSolveGF:
    """Verify solve() in Green's function mode (no Rust binary needed)."""

    def test_single_burst_mu_era(self):
        """Single burst at z=5e5 should produce μ-distortion."""
        result = solve(method="greens_function", z_h=5e5, delta_rho=1e-5)
        assert isinstance(result, SolverResult)
        assert result.method == "greens_function"
        assert result.mu > 0
        # μ/Δρ should be close to 1.401 for deep μ-era
        mu_over_drho = result.mu / 1e-5
        assert 0.5 < mu_over_drho < 1.5

    def test_single_burst_y_era(self):
        """Single burst at z=3e4 → y ≈ J_y(z_h)/4 (Chluba 2013 GF).

        At z=3e4: J_y ≈ 0.86, so y/Δρ ≈ 0.215.
        """
        result = solve(method="greens_function", z_h=3e4, delta_rho=1e-5)
        y_over_drho = result.y / 1e-5
        assert 0.20 < y_over_drho < 0.23

    def test_solve_returns_solver_result(self):
        """solve() should return SolverResult type."""
        result = solve(method="greens_function", z_h=1e5, delta_rho=1e-5)
        assert isinstance(result, SolverResult)
        assert hasattr(result, "delta_I")

    def test_solve_custom_x_grid(self):
        """Can pass a custom frequency grid."""
        x = np.logspace(-1, 1.5, 200)
        result = solve(method="greens_function", z_h=1e5, delta_rho=1e-5, x=x)
        assert len(result.x) == 200
        np.testing.assert_allclose(result.x, x)

    def test_solve_with_cosmology_object(self):
        """Can pass a Cosmology object."""
        cosmo = Cosmology.planck2018()
        result = solve(method="greens_function", z_h=1e5, delta_rho=1e-5, cosmo=cosmo)
        assert np.isfinite(result.mu)

    def test_solve_with_dq_dz(self):
        """Can pass a custom heating rate function."""

        # Simple constant heating over a narrow z-range
        def dq_dz(z):
            if 4e4 < z < 6e4:
                return 1e-10
            return 0.0

        result = solve(
            method="greens_function",
            dq_dz=dq_dz,
            z_min=1e3,
            z_max=3e6,
        )
        assert isinstance(result, SolverResult)
        assert np.isfinite(result.mu)


# =========================================================================
# run_single
# =========================================================================


class TestRunSingle:
    """Verify run_single() Green's function calculations."""

    def test_single_burst_returns_dict(self):
        """run_single should return a dict with expected keys."""
        result = run_single(z_h=1e5, delta_rho=1e-5)
        assert "x" in result
        assert "delta_n" in result
        assert "mu" in result
        assert "y" in result
        assert "delta_rho" in result

    def test_single_burst_mu_positive(self):
        """μ should be positive for heating in μ-era."""
        result = run_single(z_h=5e5, delta_rho=1e-5)
        assert result["mu"] > 0

    def test_single_burst_y_positive(self):
        """y should be positive for heating in y-era."""
        result = run_single(z_h=5e3, delta_rho=1e-5)
        assert result["y"] > 0

    def test_custom_x_grid(self):
        """run_single should accept a custom x grid."""
        x = np.logspace(-1, 1.5, 100)
        result = run_single(z_h=1e5, delta_rho=1e-5, x=x)
        assert len(result["x"]) == 100

    def test_custom_heating_rate(self):
        """run_single with dq_dz should return finite results."""
        result = run_single(dq_dz=lambda z: 1e-15 if 1e4 < z < 1e5 else 0.0)
        assert np.isfinite(result["mu"])
        assert np.isfinite(result["y"])

    def test_missing_z_h_and_dq_dz_raises(self):
        """Must provide either z_h or dq_dz."""
        with pytest.raises(ValueError):
            run_single()
