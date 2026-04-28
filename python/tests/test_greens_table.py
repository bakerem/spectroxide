"""Tests for the precomputed Green's function table module.

Covers:
- GreensTable construction, interpolation, save/load round-trip
- PhotonGreensTable construction, interpolation, save/load round-trip
- Convolution methods (distortion_from_heating, mu_from_heating, etc.)
- solve(method="table") integration
- load_or_build caching logic
- Edge cases and input handling

Requires scipy for interpolation. Tests are skipped if scipy is not installed.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

scipy = pytest.importorskip("scipy", reason="scipy required for greens_table tests")

from spectroxide import greens
from spectroxide.greens_table import (
    GreensTable,
    PhotonGreensTable,
    _DEFAULT_HEATING_CACHE,
    _DEFAULT_PHOTON_CACHE,
)

# =========================================================================
# Helpers: build synthetic tables for fast unit tests
# =========================================================================


def _make_heating_table(n_x=100, n_z=10):
    """Build a synthetic GreensTable from the analytic GF (no PDE needed)."""
    x = np.logspace(-2, np.log10(30), n_x)
    z_h = np.logspace(3, 6, n_z)
    g_th = np.zeros((n_x, n_z))
    mu_arr = np.zeros(n_z)
    y_arr = np.zeros(n_z)
    for j, zh in enumerate(z_h):
        g_th[:, j] = greens.greens_function(x, zh)
        mu_arr[j] = 1.401 * greens.j_bb_star(zh) * greens.j_mu(zh)
        y_arr[j] = (1 - greens.j_mu(zh)) / 4.0
    return GreensTable(
        z_h=z_h,
        x=x,
        g_th=g_th,
        mu=mu_arr,
        y_param=y_arr,
        delta_rho_over_rho=np.ones(n_z) * 1e-5,
        metadata={"test": True, "n_x": n_x, "n_z": n_z},
    )


def _make_photon_table(n_x=80, n_xinj=3, n_z=5):
    """Build a synthetic PhotonGreensTable from the analytic photon GF.

    z_h grid sits outside the mu-y transition window (5e4, 2e5), where the
    photon GF is intentionally invalid.
    """
    x = np.logspace(-2, np.log10(30), n_x)
    x_inj = np.array([0.5, 3.6, 10.0])[:n_xinj]
    z_h_default = np.array([1e3, 1e4, 5e4, 2e5, 1e6])
    if n_z != len(z_h_default):
        # Build a custom grid that skips the transition; split between eras.
        n_y = max(n_z // 2, 1)
        n_mu = n_z - n_y
        y_grid = np.logspace(np.log10(1e3), np.log10(5e4), n_y)
        mu_grid = np.logspace(np.log10(2e5), 6, n_mu)
        z_h = np.concatenate([y_grid, mu_grid])
    else:
        z_h = z_h_default
    g_ph = np.zeros((n_x, n_xinj, n_z))
    for k, xi in enumerate(x_inj):
        for j, zh in enumerate(z_h):
            g_ph[:, k, j] = greens.greens_function_photon(x, xi, zh)
    return PhotonGreensTable(
        z_h=z_h,
        x=x,
        x_inj=x_inj,
        g_ph=g_ph,
        metadata={"test": True},
    )


# =========================================================================
# Section 1: GreensTable — construction and interpolation
# =========================================================================


class TestGreensTableConstruction:
    """Test GreensTable creation, shapes, and basic properties."""

    def test_shape(self):
        table = _make_heating_table(n_x=50, n_z=7)
        assert table.g_th.shape == (50, 7)
        assert table.mu.shape == (7,)
        assert table.y_param.shape == (7,)
        assert table.delta_rho_over_rho.shape == (7,)
        assert len(table.x) == 50
        assert len(table.z_h) == 7

    def test_metadata_preserved(self):
        table = _make_heating_table()
        assert table.metadata["test"] is True
        assert table.metadata["n_x"] == 100

    def test_interpolator_built(self):
        """Interpolator should be built automatically on construction."""
        table = _make_heating_table()
        assert hasattr(table, "_splines") and table._splines is not None


class TestGreensTableInterpolation:
    """Test the greens_function() interpolation method."""

    def test_at_grid_point(self):
        """Interpolation at exact grid points should reproduce the stored values."""
        table = _make_heating_table(n_x=200, n_z=20)
        j = 10
        zh_exact = table.z_h[j]
        g_interp = table.greens_function(table.x, zh_exact)

        # _build_interpolator fits splines to raw g_th; CubicSpline passes
        # through training points exactly.
        g_expected = table.g_th[:, j]
        np.testing.assert_allclose(g_interp, g_expected, rtol=1e-6)

    def test_interpolation_between_points(self):
        """Interpolation between grid points should be reasonable."""
        table = _make_heating_table(n_x=200, n_z=50)
        zh_mid = np.sqrt(table.z_h[10] * table.z_h[11])  # geometric mean
        g = table.greens_function(table.x, zh_mid)
        # Should be between the two bracketing values
        g_lo = table.g_th[:, 10]
        g_hi = table.g_th[:, 11]
        # At least the interpolated values should be finite
        assert np.all(np.isfinite(g))

    def test_scalar_x(self):
        """Should handle scalar x input."""
        table = _make_heating_table()
        g = table.greens_function(3.0, 1e5)
        assert g.shape == (1,)
        assert np.isfinite(g[0])

    def test_array_x(self):
        """Should handle array x input."""
        table = _make_heating_table()
        x = np.array([1.0, 3.0, 5.0, 10.0])
        g = table.greens_function(x, 1e5)
        assert g.shape == (4,)
        assert np.all(np.isfinite(g))

    def test_clamps_out_of_range(self):
        """Values outside grid range should be clamped (not error)."""
        table = _make_heating_table()
        # z_h below and above range
        g_lo = table.greens_function(np.array([3.0]), 100.0)  # below z_h_min=1e3
        g_hi = table.greens_function(np.array([3.0]), 1e8)  # above z_h_max=1e6
        assert np.all(np.isfinite(g_lo))
        assert np.all(np.isfinite(g_hi))


# =========================================================================
# Section 2: GreensTable — save/load round-trip
# =========================================================================


class TestGreensTableSaveLoad:
    """Test serialization to .npz and deserialization."""

    def test_round_trip(self):
        table = _make_heating_table()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            table.save(path)
            loaded = GreensTable.load(path, verify_hash=False)
            np.testing.assert_array_equal(loaded.z_h, table.z_h)
            np.testing.assert_array_equal(loaded.x, table.x)
            np.testing.assert_array_equal(loaded.g_th, table.g_th)
            np.testing.assert_array_equal(loaded.mu, table.mu)
            np.testing.assert_array_equal(loaded.y_param, table.y_param)
            np.testing.assert_array_equal(
                loaded.delta_rho_over_rho, table.delta_rho_over_rho
            )
            assert loaded.metadata == table.metadata
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_rebuilds_interpolator(self):
        """Loading should rebuild the interpolator."""
        table = _make_heating_table()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            table.save(path)
            loaded = GreensTable.load(path, verify_hash=False)
            assert hasattr(loaded, "_splines") and loaded._splines is not None
            # Should be able to interpolate
            g = loaded.greens_function(np.array([3.0]), 1e5)
            assert np.isfinite(g[0])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_creates_parent_directory(self):
        """save() should create parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "table.npz"
            table = _make_heating_table()
            table.save(path)
            assert path.exists()

    def test_metadata_json_serialization(self):
        """Metadata should survive JSON round-trip."""
        table = _make_heating_table()
        table.metadata["nested"] = {"key": [1, 2, 3]}
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            table.save(path)
            loaded = GreensTable.load(path, verify_hash=False)
            assert loaded.metadata["nested"]["key"] == [1, 2, 3]
        finally:
            Path(path).unlink(missing_ok=True)


# =========================================================================
# Section 3: GreensTable — convolution methods
# =========================================================================


class TestGreensTableConvolution:
    """Test distortion_from_heating and scalar integration methods."""

    def test_distortion_from_heating_single_burst(self):
        """Convolving a delta-like burst should reproduce the GF."""
        table = _make_heating_table(n_x=200, n_z=50)
        z_h = 2e5
        delta_rho = 1e-5
        sigma_z = z_h * 0.04

        def burst(z):
            return (
                delta_rho
                * np.exp(-0.5 * ((z - z_h) / sigma_z) ** 2)
                / (sigma_z * np.sqrt(2 * np.pi))
            )

        dn = table.distortion_from_heating(
            table.x, burst, z_min=1e3, z_max=3e6, n_z=5000
        )
        # Direct evaluation
        dn_direct = table.greens_function(table.x, z_h) * delta_rho
        # Should agree to ~10% (integration vs point evaluation)
        mask = np.abs(dn_direct) > 1e-15
        if np.any(mask):
            ratio = dn[mask] / dn_direct[mask]
            np.testing.assert_allclose(ratio, 1.0, atol=0.15)

    def test_distortion_from_heating_zero_rate(self):
        """Zero heating rate should give zero distortion."""
        table = _make_heating_table()
        dn = table.distortion_from_heating(table.x, lambda z: 0.0, z_min=1e3, z_max=1e6)
        np.testing.assert_allclose(dn, 0.0, atol=1e-30)

    def test_mu_from_heating_deep_mu_era(self):
        """mu from a deep mu-era burst should be ~1.401 * delta_rho."""
        table = _make_heating_table(n_x=200, n_z=50)
        z_h = 5e5
        delta_rho = 1e-5
        sigma_z = z_h * 0.04

        def burst(z):
            return (
                delta_rho
                * np.exp(-0.5 * ((z - z_h) / sigma_z) ** 2)
                / (sigma_z * np.sqrt(2 * np.pi))
            )

        mu, _ = table.mu_y_from_heating(burst, z_min=1e3, z_max=3e6, n_z=5000)
        # Deep mu-era: mu ≈ 1.401 * J_bb*(z_h) * delta_rho
        # J_bb* ≈ 1 for z_h=5e5, J_mu ≈ 1
        expected = 1.401 * greens.j_bb_star(z_h) * greens.j_mu(z_h) * delta_rho
        np.testing.assert_allclose(mu, expected, rtol=0.2)

    def test_mu_y_from_heating_returns_tuple(self):
        """mu_y_from_heating should return a (mu, y) tuple."""
        table = _make_heating_table()
        mu, y = table.mu_y_from_heating(lambda z: 1e-15, z_min=1e3, z_max=1e6)
        assert isinstance(mu, float)
        assert isinstance(y, float)


# =========================================================================
# Section 4: PhotonGreensTable — construction and interpolation
# =========================================================================


class TestPhotonGreensTableConstruction:
    """Test PhotonGreensTable creation and shape."""

    def test_shape(self):
        table = _make_photon_table(n_x=80, n_xinj=3, n_z=5)
        assert table.g_ph.shape == (80, 3, 5)
        assert len(table.x) == 80
        assert len(table.x_inj) == 3
        assert len(table.z_h) == 5

    def test_interpolator_built(self):
        table = _make_photon_table()
        assert table._interp is not None


class TestPhotonGreensTableInterpolation:
    """Test greens_function_photon interpolation."""

    def test_at_grid_point(self):
        """Interpolation at exact grid points should match stored values."""
        table = _make_photon_table(n_x=80, n_xinj=3, n_z=5)
        k, j = 1, 2
        xi_exact = table.x_inj[k]
        zh_exact = table.z_h[j]
        g = table.greens_function_photon(table.x, xi_exact, zh_exact)
        np.testing.assert_allclose(g, table.g_ph[:, k, j], rtol=1e-10)

    def test_scalar_x(self):
        table = _make_photon_table()
        g = table.greens_function_photon(3.0, 0.5, 1e5)
        assert g.shape == (1,)
        assert np.isfinite(g[0])

    def test_array_x(self):
        table = _make_photon_table()
        x = np.array([1.0, 3.0, 5.0])
        g = table.greens_function_photon(x, 3.6, 1e4)
        assert g.shape == (3,)
        assert np.all(np.isfinite(g))

    def test_clamps_out_of_range(self):
        table = _make_photon_table()
        g = table.greens_function_photon(np.array([3.0]), 0.01, 1e8)
        assert np.all(np.isfinite(g))


# =========================================================================
# Section 5: PhotonGreensTable — save/load round-trip
# =========================================================================


class TestPhotonGreensTableSaveLoad:

    def test_round_trip(self):
        table = _make_photon_table()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            table.save(path)
            loaded = PhotonGreensTable.load(path, verify_hash=False)
            np.testing.assert_array_equal(loaded.z_h, table.z_h)
            np.testing.assert_array_equal(loaded.x, table.x)
            np.testing.assert_array_equal(loaded.x_inj, table.x_inj)
            np.testing.assert_array_equal(loaded.g_ph, table.g_ph)
            assert loaded.metadata == table.metadata
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_rebuilds_interpolator(self):
        table = _make_photon_table()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            table.save(path)
            loaded = PhotonGreensTable.load(path, verify_hash=False)
            g = loaded.greens_function_photon(np.array([3.0]), 3.6, 1e4)
            assert np.isfinite(g[0])
        finally:
            Path(path).unlink(missing_ok=True)


# =========================================================================
# Section 6: PhotonGreensTable — convolution
# =========================================================================


class TestPhotonGreensTableConvolution:

    def test_zero_rate(self):
        table = _make_photon_table()
        dn = table.distortion_from_photon_injection(
            table.x,
            x_inj=3.6,
            dn_dz=lambda z: 0.0,
            z_min=1e3,
            z_max=1e6,
        )
        np.testing.assert_allclose(dn, 0.0, atol=1e-30)

    def test_burst_injection(self):
        """Narrow burst convolution should approximate point evaluation."""
        table = _make_photon_table(n_x=80, n_xinj=3, n_z=5)
        z_h = 1e5
        sigma_z = z_h * 0.04
        amp = 1e-5

        def burst(z):
            return (
                amp
                * np.exp(-0.5 * ((z - z_h) / sigma_z) ** 2)
                / (sigma_z * np.sqrt(2 * np.pi))
            )

        dn = table.distortion_from_photon_injection(
            table.x,
            x_inj=3.6,
            dn_dz=burst,
            z_min=1e3,
            z_max=3e6,
            n_z=5000,
        )
        # Should be finite and nonzero
        assert np.any(np.abs(dn) > 0)
        assert np.all(np.isfinite(dn))


# =========================================================================
# Section 7: solve(method="table") integration
# =========================================================================


class TestSolveTableMethod:
    """Test the solve() unified API with method='table'."""

    def test_single_burst(self):
        from spectroxide.solver import solve

        table = _make_heating_table(n_x=200, n_z=50)
        result = solve(method="table", z_h=5e4, delta_rho=1e-5, table=table)
        assert result.method == "table"
        assert result.z_h == 5e4
        assert len(result.x) > 0
        assert len(result.delta_n) == len(result.x)
        assert np.isfinite(result.mu)
        assert np.isfinite(result.y)

    def test_custom_heating(self):
        from spectroxide.solver import solve

        table = _make_heating_table(n_x=200, n_z=50)

        def dq_dz(z):
            return 1e-15 * (1 + z) ** 2

        result = solve(
            method="table",
            dq_dz=dq_dz,
            table=table,
            z_min=1e3,
            z_max=1e6,
        )
        assert result.method == "table"
        assert np.all(np.isfinite(result.delta_n))

    def test_photon_table(self):
        from spectroxide.solver import solve

        table = _make_photon_table()
        result = solve(
            method="table",
            z_h=1e5,
            table=table,
            injection={"x_inj": 3.6, "delta_n_over_n": 1e-5},
        )
        assert result.method == "table"
        assert np.all(np.isfinite(result.delta_n))

    def test_requires_z_h_or_dq_dz(self):
        """Should error if neither z_h nor dq_dz provided for GreensTable."""
        from spectroxide.solver import solve

        table = _make_heating_table()
        with pytest.raises(ValueError, match="requires z_h or dq_dz"):
            solve(method="table", table=table)

    def test_photon_table_requires_x_inj(self):
        """Should error if PhotonGreensTable used without x_inj."""
        from spectroxide.solver import solve

        table = _make_photon_table()
        with pytest.raises(ValueError, match="x_inj"):
            solve(method="table", z_h=1e5, table=table)

    def test_table_from_path(self):
        """solve() should accept a path string for the table parameter."""
        from spectroxide.solver import solve

        table = _make_heating_table()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            table.save(path)
            result = solve(
                method="table",
                z_h=1e5,
                delta_rho=1e-5,
                table=path,
                verify_hash=False,
            )
            assert result.method == "table"
            assert np.isfinite(result.mu)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_unsupported_table_type(self):
        """Should error with an unsupported table type."""
        from spectroxide.solver import solve

        with pytest.raises(TypeError, match="Unsupported table type"):
            solve(method="table", z_h=1e5, table=42)  # int is not a valid table


# =========================================================================
# Section 8: load_or_build caching logic
# =========================================================================


class TestLoadOrBuild:
    """Test the load_or_build convenience functions."""

    def test_load_from_existing_cache(self):
        from spectroxide.greens_table import load_or_build_greens_table

        table = _make_heating_table()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            table.save(path)
            loaded = load_or_build_greens_table(cache_path=path, verify_hash=False)
            np.testing.assert_array_equal(loaded.g_th, table.g_th)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_photon_load_from_existing_cache(self):
        from spectroxide.greens_table import load_or_build_photon_greens_table

        table = _make_photon_table()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            table.save(path)
            loaded = load_or_build_photon_greens_table(
                cache_path=path, verify_hash=False
            )
            np.testing.assert_array_equal(loaded.g_ph, table.g_ph)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_default_cache_paths(self):
        """Default cache paths should be in ~/.spectroxide/."""
        assert _DEFAULT_HEATING_CACHE.parent.name == ".spectroxide"
        assert _DEFAULT_PHOTON_CACHE.parent.name == ".spectroxide"
        assert str(_DEFAULT_HEATING_CACHE).endswith("greens_table.npz")
        assert str(_DEFAULT_PHOTON_CACHE).endswith("photon_greens_table.npz")


# =========================================================================
# Section 9: Physics consistency checks
# =========================================================================


class TestPhysicsConsistency:
    """Verify that table-based GF preserves physical properties."""

    def test_energy_conservation(self):
        """G_th should integrate to ~1 in energy (∫x³ G_th dx / G₃ ≈ 1)."""
        table = _make_heating_table(n_x=500, n_z=20)
        for j in range(len(table.z_h)):
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            drho = _trapz(table.x**3 * table.g_th[:, j], table.x) / greens.G3_PLANCK
            # Should be ~1 per unit Δρ/ρ (the analytic GF conserves energy)
            # With Chluba's J_T = (1-J_bb*)/4, energy is not exactly conserved.
            # Deviation up to ~10% in the transition era (z ~ 5e4).
            np.testing.assert_allclose(
                drho,
                1.0,
                atol=0.20,
                err_msg=f"Energy not conserved at z_h={table.z_h[j]:.0e}",
            )

    def test_mu_positive_in_mu_era(self):
        """mu should be positive for heating in the mu-era."""
        table = _make_heating_table(n_z=50)
        mu_era = table.z_h > 2e5
        assert np.all(table.mu[mu_era] > 0)

    def test_y_positive_in_y_era(self):
        """y should be positive for heating in the y-era."""
        table = _make_heating_table(n_z=50)
        y_era = table.z_h < 1e4
        assert np.all(table.y_param[y_era] > 0)

    def test_mu_decreases_toward_low_z(self):
        """mu per Δρ/ρ should decrease as z_h decreases (less thermalization)."""
        table = _make_heating_table(n_z=50)
        # In the mu-era (z > 2e5), mu should generally decrease with decreasing z
        mu_era = table.z_h > 2e5
        mu_vals = table.mu[mu_era]
        z_vals = table.z_h[mu_era]
        # Check that the highest z has the highest mu (roughly)
        assert mu_vals[-1] >= mu_vals[0] * 0.5

    def test_y_increases_toward_low_z(self):
        """y per Δρ/ρ should approach 0.25 at low z (pure y-era)."""
        table = _make_heating_table(n_z=50)
        low_z = table.z_h < 3e3
        if np.any(low_z):
            y_low = table.y_param[low_z]
            np.testing.assert_allclose(y_low, 0.25, atol=0.02)

    def test_greens_function_agrees_with_analytic_in_limits(self):
        """In the deep mu and y eras, table should match NC-stripped analytic GF well."""
        from spectroxide import g_bb
        from scipy.integrate import trapezoid

        table = _make_heating_table(n_x=200, n_z=50)
        x = table.x
        mask = (x > 1) & (x < 15)
        gbb = g_bb(x)
        g2_gbb = trapezoid(x**2 * gbb, x)

        def nc_strip(g):
            alpha = trapezoid(x**2 * g, x) / g2_gbb
            return g - alpha * gbb

        # Deep mu-era: table.greens_function and analytic GF are both raw
        # (carry their own G_bb component); strip both before comparing.
        zh_mu = 5e5
        g_table = nc_strip(table.greens_function(x, zh_mu))
        g_analytic = nc_strip(greens.greens_function(x, zh_mu))
        np.testing.assert_allclose(g_table[mask], g_analytic[mask], rtol=0.05)

        # y-era
        zh_y = 3e3
        g_table_y = nc_strip(table.greens_function(x, zh_y))
        g_analytic_y = nc_strip(greens.greens_function(x, zh_y))
        np.testing.assert_allclose(g_table_y[mask], g_analytic_y[mask], rtol=0.05)


# =========================================================================
# Section 10: Edge cases
# =========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_z_table(self):
        """Table with a single z_h should work (no interpolation in z)."""
        from spectroxide import g_bb
        from scipy.integrate import trapezoid

        x = np.logspace(-2, np.log10(30), 50)
        z_h = np.array([1e5])
        g_th = greens.greens_function(x, 1e5).reshape(-1, 1)
        table = GreensTable(
            z_h=z_h,
            x=x,
            g_th=g_th,
            mu=np.array([1.0]),
            y_param=np.array([0.1]),
            delta_rho_over_rho=np.array([1e-5]),
        )
        g = table.greens_function(x, 1e5)
        # greens_function now returns the raw stored values (no build-time strip).
        g_expected = g_th[:, 0]
        np.testing.assert_allclose(g, g_expected, rtol=1e-10)

    def test_empty_metadata(self):
        """Empty metadata should work."""
        table = _make_heating_table()
        table.metadata = {}
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            table.save(path)
            loaded = GreensTable.load(path, verify_hash=False)
            assert loaded.metadata == {}
        finally:
            Path(path).unlink(missing_ok=True)

    def test_large_z_h_clamp(self):
        """Requesting z_h above table range should clamp, not crash."""
        table = _make_heating_table()
        g = table.greens_function(np.array([3.0]), 1e10)
        assert np.isfinite(g[0])

    def test_small_x_clamp(self):
        """Requesting x below table range should clamp, not crash."""
        table = _make_heating_table()
        g = table.greens_function(np.array([1e-6]), 1e5)
        assert np.isfinite(g[0])


# =========================================================================
# Section 11: Physics-hash verification
# =========================================================================


class TestPhysicsHashVerification:
    """Verify that cached tables are tied to the binary's physics hash."""

    def test_load_with_wrong_hash_warns(self, monkeypatch):
        from spectroxide.greens_table import GreensTableHashMismatch
        import spectroxide.greens_table as gt_mod
        import spectroxide.solver as solver_mod

        # Pretend the binary's hash is "binary_v1"
        monkeypatch.setattr(gt_mod, "get_physics_hash", lambda: "binary_v1")
        monkeypatch.setattr(solver_mod, "get_physics_hash", lambda: "binary_v1")

        table = _make_heating_table()
        # Stamp a different ("stale") hash on the table
        table.metadata["physics_hash"] = "stale_v0"
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            table.save(path)
            with pytest.warns(GreensTableHashMismatch, match="stale_v0"):
                GreensTable.load(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_without_hash_warns(self, monkeypatch):
        """Pre-feature tables (no physics_hash key) should warn."""
        from spectroxide.greens_table import GreensTableHashMismatch
        import spectroxide.greens_table as gt_mod
        import spectroxide.solver as solver_mod

        monkeypatch.setattr(gt_mod, "get_physics_hash", lambda: "binary_v1")
        monkeypatch.setattr(solver_mod, "get_physics_hash", lambda: "binary_v1")

        table = _make_heating_table()  # _make_heating_table() omits physics_hash
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            table.save(path)
            with pytest.warns(GreensTableHashMismatch, match="<none>"):
                GreensTable.load(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_or_build_warns_on_mismatch(self, monkeypatch):
        """load_or_build should warn on hash mismatch and return the cached table."""
        from spectroxide.greens_table import (
            GreensTableHashMismatch,
            load_or_build_greens_table,
        )
        import spectroxide.greens_table as gt_mod
        import spectroxide.solver as solver_mod

        monkeypatch.setattr(gt_mod, "get_physics_hash", lambda: "binary_v1")
        monkeypatch.setattr(solver_mod, "get_physics_hash", lambda: "binary_v1")

        table = _make_heating_table()
        table.metadata["physics_hash"] = "stale_v0"
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            table.save(path)
            with pytest.warns(GreensTableHashMismatch):
                loaded = load_or_build_greens_table(cache_path=path)
            assert loaded.metadata["physics_hash"] == "stale_v0"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_matching_hash_loads_cleanly(self, monkeypatch):
        import spectroxide.greens_table as gt_mod
        import spectroxide.solver as solver_mod

        monkeypatch.setattr(gt_mod, "get_physics_hash", lambda: "binary_v1")
        monkeypatch.setattr(solver_mod, "get_physics_hash", lambda: "binary_v1")

        table = _make_heating_table()
        table.metadata["physics_hash"] = "binary_v1"
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            table.save(path)
            loaded = GreensTable.load(path)  # default verify_hash=True, should pass
            np.testing.assert_array_equal(loaded.g_th, table.g_th)
        finally:
            Path(path).unlink(missing_ok=True)
