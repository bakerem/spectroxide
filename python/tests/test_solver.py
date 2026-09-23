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
    _injection_param_args,
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
        """SingleBurst params translate to flags (type is positional)."""
        args = _injection_param_args(
            {
                "type": "single_burst",
                "z_h": 5e4,
            }
        )
        assert "--z-h" in args
        assert "single-burst" not in args

    def test_decaying_particle(self):
        """DecayingParticle should produce correct args."""
        args = _injection_param_args(
            {
                "type": "decaying_particle",
                "f_x": 1e-3,
                "gamma_x": 1e-8,
            }
        )
        assert "--f-x" in args
        assert "--gamma-x" in args

    def test_unknown_param_raises(self):
        """Unknown injection parameter should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown injection parameter"):
            _injection_param_args({"type": "single_burst", "bogus_param": 42})


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

    def test_solve_rejects_cosmology(self):
        """The heat Green's function is not cosmology-aware, so cosmo= is an
        error, not a silently ignored argument (A-1)."""
        cosmo = Cosmology.planck2018()
        with pytest.raises(TypeError, match="cosmo"):
            solve(method="greens_function", z_h=1e5, delta_rho=1e-5, cosmo=cosmo)

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


# =========================================================================
# solve() — argument checks before dispatch (A-1, R-7)
# =========================================================================

# A project root with no binary: any call that got past the argument checks
# would fail with FileNotFoundError or RuntimeError, not TypeError.
_NO_BINARY = "/nonexistent/spectroxide-root"


class TestSolveArgumentChecks:
    """solve() rejects bad or unusable arguments before any work."""

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="method='gf' is not valid"):
            solve(method="gf", z_h=1e5)

    def test_unknown_keyword_before_pde_solve(self):
        with pytest.raises(TypeError, match="unexpected keyword.*n_pionts"):
            solve(
                injection={"type": "single_burst", "z_h": 1e5},
                n_pionts=100,
                project_root=_NO_BINARY,
            )

    def test_removed_dark_photon_depletion(self):
        with pytest.raises(TypeError, match="dark_photon_depletion"):
            solve(dark_photon_depletion=1e-3, project_root=_NO_BINARY)

    @pytest.mark.parametrize(
        "extra",
        [
            {"z_start": 3e5},
            {"z_end": 100.0},
            {"photon_source": lambda x, z: 0.0},
            {"n_points": 2000},
            {"debug": True},
            {"table": "some.npz"},
            {"verify_hash": False},
        ],
    )
    def test_greens_function_rejects_unused(self, extra):
        name = next(iter(extra))
        with pytest.raises(TypeError, match=name):
            solve(method="greens_function", z_h=1e5, **extra)

    def test_greens_function_rejects_z_h_and_dq_dz(self):
        with pytest.raises(TypeError, match="not both"):
            solve(method="greens_function", z_h=1e5, dq_dz=lambda z: 0.0)

    def test_greens_function_burst_rejects_z_integration_args(self):
        with pytest.raises(TypeError, match="n_z"):
            solve(method="greens_function", z_h=1e5, n_z=100)

    def test_greens_function_heating_rejects_delta_rho(self):
        with pytest.raises(TypeError, match="delta_rho"):
            solve(method="greens_function", dq_dz=lambda z: 0.0, delta_rho=1e-4)

    def test_table_rejects_cosmo_before_loading(self):
        # table=None would load or build the default table; the check must
        # fire first.
        with pytest.raises(TypeError, match="cosmo"):
            solve(method="table", z_h=1e5, cosmo={"h": 0.7})

    def test_table_rejects_injection_for_heat_table(self):
        with pytest.raises(TypeError, match="injection"):
            solve(method="table", z_h=1e5, injection={"x_inj": 1.0})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"injection": {"type": "single_burst", "z_h": 1e5}, "dq_dz": abs},
            {"dq_dz": abs, "photon_source": lambda x, z: 0.0},
            {"dn_planck": 1e-5, "dq_dz": abs},
        ],
    )
    def test_pde_rejects_two_sources(self, kwargs):
        with pytest.raises(TypeError, match="exactly one"):
            solve(project_root=_NO_BINARY, **kwargs)

    def test_pde_rejects_top_level_z_h_with_injection(self):
        with pytest.raises(TypeError, match="z_h is not used"):
            solve(
                injection={"type": "single_burst", "z_h": 1e5},
                z_h=1e5,
                project_root=_NO_BINARY,
            )

    def test_pde_z_h_without_source(self):
        with pytest.raises(ValueError, match="z_h is only used"):
            solve(z_h=1e5, project_root=_NO_BINARY)

    def test_pde_injection_rejects_z_integration_args(self):
        with pytest.raises(TypeError, match="z_max"):
            solve(
                injection={"type": "single_burst", "z_h": 1e5},
                z_max=1e6,
                project_root=_NO_BINARY,
            )

    def test_pde_heating_rejects_x_grid(self):
        with pytest.raises(TypeError, match="n_x"):
            solve(dq_dz=abs, n_x=100, project_root=_NO_BINARY)

    def test_pde_rejects_table(self):
        with pytest.raises(TypeError, match="table"):
            solve(
                injection={"type": "single_burst", "z_h": 1e5},
                table="some.npz",
                project_root=_NO_BINARY,
            )

    def test_cosmo_params_alias_warns(self):
        # The alias is accepted; the call then fails only because the
        # binary is missing, which proves the checks passed.
        with pytest.warns(DeprecationWarning, match="cosmo_params"):
            with pytest.raises((FileNotFoundError, RuntimeError)):
                solve(
                    injection={"type": "single_burst", "z_h": 1e5},
                    cosmo_params={"h": 0.7},
                    project_root=_NO_BINARY,
                )

    def test_cosmo_params_and_cosmo_conflict(self):
        with pytest.raises(TypeError, match="both 'cosmo' and 'cosmo_params'"):
            solve(
                injection={"type": "single_burst", "z_h": 1e5},
                cosmo={"h": 0.7},
                cosmo_params={"h": 0.7},
                project_root=_NO_BINARY,
            )


# =========================================================================
# Stale-binary warning (R-4)
# =========================================================================


class TestStaleBinaryWarning:
    """The wrapper warns, but does not rebuild, when Rust sources are newer
    than the prebuilt binary."""

    @staticmethod
    def _make_tree(root, *, binary_time, source_times):
        import os

        binary = root / "target" / "release" / "spectroxide"
        binary.parent.mkdir(parents=True)
        binary.write_text("")
        os.utime(binary, (binary_time, binary_time))
        for rel, t in source_times.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
            os.utime(path, (t, t))
        return binary

    @pytest.mark.parametrize(
        "newer", ["src/kompaneets.rs", "src/bin/check.rs", "Cargo.toml", "Cargo.lock"]
    )
    def test_warns_when_source_newer(self, tmp_path, newer):
        from spectroxide.solver import _warn_if_stale_binary

        times = {
            "src/kompaneets.rs": 1000.0,
            "src/bin/check.rs": 1000.0,
            "Cargo.toml": 1000.0,
            "Cargo.lock": 1000.0,
        }
        times[newer] = 3000.0
        binary = self._make_tree(tmp_path, binary_time=2000.0, source_times=times)
        with pytest.warns(RuntimeWarning, match="older than " + newer):
            _warn_if_stale_binary(tmp_path, binary)
        # No rebuild: the binary is untouched.
        assert binary.stat().st_mtime == 2000.0

    def test_silent_when_binary_newest(self, tmp_path):
        import warnings

        from spectroxide.solver import _warn_if_stale_binary

        binary = self._make_tree(
            tmp_path,
            binary_time=2000.0,
            source_times={"src/lib.rs": 1000.0, "Cargo.toml": 1000.0},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_stale_binary(tmp_path, binary)

    def test_ignores_non_rust_files(self, tmp_path):
        import warnings

        from spectroxide.solver import _warn_if_stale_binary

        binary = self._make_tree(
            tmp_path,
            binary_time=2000.0,
            source_times={"src/lib.rs": 1000.0, "src/notes.md": 3000.0},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_stale_binary(tmp_path, binary)

    def test_run_rust_binary_warns(self, tmp_path):
        """End to end through _run_rust_binary with a stand-in binary."""
        import os

        from spectroxide.solver import _run_rust_binary

        binary = self._make_tree(
            tmp_path, binary_time=2000.0, source_times={"src/lib.rs": 3000.0}
        )
        binary.write_text("#!/bin/sh\necho '{\"results\": []}'\n")
        binary.chmod(0o755)
        os.utime(binary, (2000.0, 2000.0))
        cmd = ["cargo", "run", "--release", "--bin", "spectroxide", "--", "info"]
        with pytest.warns(RuntimeWarning, match="older than src/lib.rs"):
            out = _run_rust_binary(cmd, cwd=tmp_path)
        assert out == {"results": []}


# =========================================================================
# Silent fallbacks removed (R-7)
# =========================================================================


def _fake_binary(root, stdout_json):
    """Install a stand-in binary under ``root`` that prints ``stdout_json``."""
    import json as _json

    binary = root / "target" / "release" / "spectroxide"
    binary.parent.mkdir(parents=True)
    payload = _json.dumps(stdout_json).replace("'", "'\\''")
    binary.write_text(f"#!/bin/sh\necho '{payload}'\n")
    binary.chmod(0o755)
    return binary


class TestNoSilentFallbacks:
    """Schema drift, NaN results, capped tables, and non-dict mappings."""

    _X = [1.0, 2.0, 3.0]
    _DN = [1e-6, 2e-6, 1e-6]

    def test_schema_drift_raises(self, tmp_path):
        # "mu" in place of "pde_mu": the old code returned mu = 0.0.
        _fake_binary(
            tmp_path,
            {
                "results": [
                    {"x": self._X, "delta_n": self._DN, "mu": 1e-5, "pde_y": 0.0}
                ]
            },
        )
        with pytest.raises(RuntimeError, match="lacks.*pde_mu"):
            solve(
                injection={"type": "single_burst", "z_h": 1e5},
                project_root=tmp_path,
            )

    def test_nan_mu_kept_as_nan_with_warning(self, tmp_path):
        # The Rust serializer writes NaN as null.
        _fake_binary(
            tmp_path,
            {
                "results": [
                    {
                        "x": self._X,
                        "delta_n": self._DN,
                        "pde_mu": None,
                        "pde_y": 1e-6,
                        "drho": 1e-5,
                    }
                ]
            },
        )
        with pytest.warns(RuntimeWarning, match="non-finite values for pde_mu"):
            r = solve(
                injection={"type": "single_burst", "z_h": 1e5},
                project_root=tmp_path,
            )
        assert r.mu is not None and np.isnan(r.mu)
        assert r.y == 1e-6

    def test_photon_source_n_z_cap_warns(self, tmp_path):
        _fake_binary(
            tmp_path,
            {
                "results": [
                    {
                        "x": self._X,
                        "delta_n": self._DN,
                        "pde_mu": 0.0,
                        "pde_y": 0.0,
                        "drho": 0.0,
                    }
                ]
            },
        )
        with pytest.warns(UserWarning, match="n_z=600 exceeds"):
            solve(
                photon_source=lambda x, z: 0.0,
                n_z=600,
                n_x=5,
                project_root=tmp_path,
            )

    def test_validate_cosmology_non_dict_mapping(self):
        from types import MappingProxyType

        from spectroxide._validation import validate_cosmology

        with pytest.raises(ValueError, match="h must be positive"):
            validate_cosmology(MappingProxyType({"h": -0.7}))
        validate_cosmology(MappingProxyType({"h": 0.7}))


class TestDeltaIUsesRunTcmb:
    """SolverResult.delta_I uses the run's T_CMB, not a fixed 2.726 K (A-7)."""

    def test_default_t_cmb(self):
        r = SolverResult(
            x=np.array([1.0]),
            delta_n=np.array([1e-5]),
            mu=0.0,
            y=0.0,
            delta_rho_over_rho=0.0,
            method="pde",
        )
        assert r.t_cmb == 2.726

    def test_frequency_scales_with_t_cmb(self):
        # ν = x k_B T0 / h with CODATA 2018 exact k_B and h, typed here.
        k_b, h = 1.380649e-23, 6.62607015e-34
        x = np.array([1.0, 3.0])
        r = SolverResult(
            x=x,
            delta_n=np.full(2, 1e-5),
            mu=0.0,
            y=0.0,
            delta_rho_over_rho=0.0,
            method="pde",
            t_cmb=2.7255,
        )
        nu_ghz, di = r.delta_I
        np.testing.assert_allclose(nu_ghz, x * k_b * 2.7255 / h / 1e9, rtol=1e-12)
        # ΔI ∝ ν³ at fixed Δn: ratio to the 2.726 K conversion is (T/2.726)³.
        _, di_ref = r.__class__(**{**r.__dict__, "t_cmb": 2.726}).delta_I
        np.testing.assert_allclose(di / di_ref, (2.7255 / 2.726) ** 3, rtol=1e-12)

    def test_pde_result_carries_cosmo_t_cmb(self, tmp_path):
        _fake_binary(
            tmp_path,
            {
                "results": [
                    {
                        "x": [1.0, 2.0],
                        "delta_n": [1e-6, 1e-6],
                        "pde_mu": 0.0,
                        "pde_y": 0.0,
                        "drho": 0.0,
                    }
                ]
            },
        )
        inj = {"type": "single_burst", "z_h": 1e5}
        r = solve(injection=inj, cosmo=Cosmology.planck2018(), project_root=tmp_path)
        assert r.t_cmb == 2.7255
        r = solve(injection=inj, project_root=tmp_path)
        assert r.t_cmb == 2.726
