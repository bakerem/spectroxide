"""Dark-photon conversion with the neutral-hydrogen photon mass (ADR 0008).

Where possible, targets come from literal constants typed here and from an
independent scan written in the test (CLAUDE.md pitfalls 9 and 11). Tests
marked "property test" check a structural property of the code's own output
instead. Rust-Python agreement is checked in ``test_parity.py``.
"""

import numpy as np
import pytest

from spectroxide.cosmology import (
    DEFAULT_COSMO,
    _get_recomb_table,
    _cosmo_hubble,
    _cosmo_n_h,
    _helium_electron_fraction,
    ionization_fraction,
)
from spectroxide.dark_photon import (
    _ionization_state,
    conversion_probability,
    cell_average,
    gamma_con,
    photon_mass_sq_ev2,
    plasma_frequency_ev,
    resonance_redshift,
    tau_per_epsilon_sq,
)

# Literal CODATA 2018 values, independent of the package constants.
A0_M = 5.291_772_109_03e-11  # Bohr radius [m]
HBARC_EV_CM = 1.973_269_804e-5  # ħc [eV cm]
ME_EV = 510_998.950_00  # m_e c² [eV]
ALPHA = 7.297_352_5693e-3
KB_EV = 8.617_333_262e-5  # [eV/K]
HBAR_EV_S = 6.582_119_569e-16  # [eV s]


def _caputo_mass_sq(z, x, cosmo=DEFAULT_COSMO):
    """Caputo et al. (2020) Eq. 1 rebuilt from literal constants, in eV²."""
    c_pl = 4.0 * np.pi * ALPHA * HBARC_EV_CM**3 / ME_EV
    c_n = 4.0 * np.pi * 4.5 * (A0_M * 100.0) ** 3
    z = np.atleast_1d(np.asarray(z, dtype=float))
    x_e = ionization_fraction(z, cosmo)
    x_he = np.array([_helium_electron_fraction(zi, cosmo) for zi in z])
    n_h = _cosmo_n_h(z, cosmo) * 1e-6
    x_h = np.clip(x_e - x_he, 0.0, 1.0)
    omega = x * KB_EV * cosmo["t_cmb"] * (1.0 + z)
    return c_pl * x_e * n_h - c_n * omega**2 * (1.0 - x_h) * n_h


def test_neutral_coefficient_from_bohr_radius():
    coeff = 4.0 * np.pi * 4.5 * (A0_M * 100.0) ** 3
    assert coeff == pytest.approx(8.38e-24, rel=1e-3)
    c_pl = 4.0 * np.pi * ALPHA * HBARC_EV_CM**3 / ME_EV
    assert c_pl == pytest.approx(1.4e-21, rel=0.02)  # Caputo Eq. 1 plasma term


def test_vectorized_ionization_state_matches_scalar_history():
    z = np.geomspace(10.0, 3e7, 400)
    x_e, x_he = _ionization_state(z, DEFAULT_COSMO)
    np.testing.assert_allclose(x_e, ionization_fraction(z, DEFAULT_COSMO), rtol=1e-7)
    ref_he = [_helium_electron_fraction(zi, DEFAULT_COSMO) for zi in z]
    np.testing.assert_allclose(x_he, ref_he, rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize(
    "z,x", [(300.0, 1.0), (668.0, 4.0), (1200.0, 10.0), (5e4, 3.0)]
)
def test_photon_mass_matches_caputo_eq1(z, x):
    expected = float(_caputo_mass_sq(z, x)[0])
    scale = max(plasma_frequency_ev(z) ** 2, abs(expected))
    assert abs(photon_mass_sq_ev2(z, x) - expected) < 1e-6 * scale


@pytest.mark.parametrize("m_ev", [1e-7, 1e-6, 1e-5])
def test_reduces_to_gamma_con_when_hydrogen_ionized(m_ev):
    x = np.array([0.01, 0.5, 1.0, 4.0, 10.0, 30.0])
    gc, z_res = gamma_con(1e-7, m_ev)
    conv = conversion_probability(1e-7, m_ev, x)
    assert all(c.size == 1 for c in conv.crossings)
    np.testing.assert_allclose([c[0] for c in conv.crossings], z_res, rtol=1e-7)
    np.testing.assert_allclose(conv.tau, gc / x, rtol=1e-6)
    np.testing.assert_allclose(conv.probability, -np.expm1(-gc / x), rtol=1e-6)


def test_x4_resonance_moves_at_1e_minus_11_ev():
    """m = 1e-11 eV: neutral term ≈ 1.9× plasma at z_res ≈ 668 for x = 4."""
    m_ev, x, eps = 1e-11, 4.0, 1e-7
    z_res = resonance_redshift(m_ev)
    assert z_res == pytest.approx(668.0, abs=5.0)
    ratio = 1.0 - float(_caputo_mass_sq(z_res, x)[0]) / plasma_frequency_ev(z_res) ** 2
    assert 1.7 < ratio < 2.1

    # Independent dense scan + bisection + finite-difference slope.
    target = m_ev**2
    zs = np.geomspace(10.0, 3e3, 20000)
    f = _caputo_mass_sq(zs, x) - target
    idx = np.nonzero((f[:-1] < 0) != (f[1:] < 0))[0]
    expected = []
    for i in idx:
        lo, hi = zs[i], zs[i + 1]
        f_lo = f[i]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f_mid = float(_caputo_mass_sq(mid, x)[0]) - target
            if (f_lo < 0) == (f_mid < 0):
                lo, f_lo = mid, f_mid
            else:
                hi = mid
        expected.append(0.5 * (lo + hi))
    expected = np.sort(expected)[::-1]
    assert expected.size >= 1
    assert np.all(expected > 1.1 * z_res)

    h = 1e-4
    tau_expected = 0.0
    for z in expected:
        zp, zm = (1 + z) * np.exp(h) - 1, (1 + z) * np.exp(-h) - 1
        dm2 = float(_caputo_mass_sq(zp, x)[0] - _caputo_mass_sq(zm, x)[0])
        dln = abs(dm2 / (2 * h) / target)
        omega = x * KB_EV * DEFAULT_COSMO["t_cmb"] * (1 + z)
        h_ev = HBAR_EV_S * _cosmo_hubble(z, DEFAULT_COSMO)
        tau_expected += np.pi * eps**2 * target / (omega * h_ev * dln)

    conv = conversion_probability(eps, m_ev, [x])
    np.testing.assert_allclose(conv.crossings[0], expected, rtol=1e-6)
    assert conv.tau[0] == pytest.approx(tau_expected, rel=2e-3)


@pytest.mark.parametrize("m_ev", [1e-12, 1e-11, 1e-10, 1e-9, 1e-7])
def test_every_crossing_lies_at_or_above_z_res(m_ev):
    """Property test: m_γ² ≤ ω_pl², so every crossing has z ≥ z_res."""
    z_res = resonance_redshift(m_ev)
    conv = conversion_probability(1e-7, m_ev, np.geomspace(0.05, 30.0, 60))
    for c in conv.crossings:
        assert c.size >= 1
        assert c[-1] >= z_res * (1 - 1e-7)


def test_low_x_reduces_to_gamma_con_after_recombination():
    """m = 1e-11 eV (z_res ≈ 668): τ·x → γ_con as x → 0, residual ∝ x²."""
    m_ev, eps = 1e-11, 1e-7
    z_ode, x_h_ode, _ = _get_recomb_table(DEFAULT_COSMO)
    xs = np.array([1e-3, 1e-2])
    conv = conversion_probability(eps, m_ev, xs)
    dev = []
    for i, x in enumerate(xs):
        assert conv.crossings[i].size == 1
        z = conv.crossings[i][0]
        assert z == pytest.approx(resonance_redshift(m_ev), rel=1e-5)
        dz = max(0.1, 1e-4 * z)
        xe = ionization_fraction(np.array([z - dz, z, z + dz]))
        d = abs(3.0 + (1 + z) * (xe[2] - xe[0]) / (2 * dz) / xe[1])
        t_ev = KB_EV * DEFAULT_COSMO["t_cmb"] * (1 + z)
        h_ev = HBAR_EV_S * _cosmo_hubble(z, DEFAULT_COSMO)
        gc = np.pi * eps**2 * m_ev**2 / (d * t_ev * h_ev)
        dev.append(conv.tau[i] * x / gc - 1.0)
    assert abs(dev[0]) < 1e-6
    assert dev[1] < 0.0
    assert 85.0 < dev[1] / dev[0] < 115.0


def _band_integral(x, tau_bar):
    """∫_{1.2}^{11} x³ n_pl τ̄ dx with τ̄ constant over each cell."""
    e = np.concatenate([[x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]]])
    tot = 0.0
    for i in range(x.size):
        lo, hi = max(e[i], 1.2), min(e[i + 1], 11.0)
        if hi <= lo:
            continue
        xx = lo + (np.arange(16) + 0.5) * (hi - lo) / 16
        tot += np.sum(xx**3 / np.expm1(xx)) * (hi - lo) / 16 * tau_bar[i]
    return tot


def test_cell_average_is_independent_of_grid_offset_at_tangency():
    """Property test: m = 1e-12 eV has tangent crossings near x ≈ 2.73 and 2.98.

    Shifting a linear grid (Δx = 0.03) by fractions of a cell must move the
    cell-averaged FIRAS-band integral by less than 1%.
    """
    vals = []
    for f in [0.0, 0.2, 0.4, 0.6, 0.8]:
        x = 0.5 + (np.arange(400) + f) * 0.03
        vals.append(_band_integral(x, cell_average(1e-7, 1e-12, x)[0]))
    vals = np.array(vals)
    assert (vals.max() - vals.min()) / vals.min() < 0.01


def test_cell_average_on_nonuniform_grid_matches_smooth_point_values():
    """Away from tangencies the cell average equals the point value to O(Δx²)."""
    x = np.geomspace(0.1, 20.0, 200)
    tau_bar, p_bar = cell_average(1e-7, 1e-10, x)
    tau = conversion_probability(1e-7, 1e-10, x).tau
    np.testing.assert_allclose(tau_bar[1:-1], tau[1:-1], rtol=5e-3)
    np.testing.assert_allclose(p_bar, -np.expm1(-tau_bar), rtol=1e-6)


def test_ionization_state_accepts_scalars():
    x_e, x_he = _ionization_state(700.0, DEFAULT_COSMO)
    assert isinstance(x_e, float) and isinstance(x_he, float)
    assert x_e == pytest.approx(ionization_fraction(700.0), rel=1e-7)


def test_tau_per_epsilon_sq_point_values():
    """``cell_averaged=False`` gives point values of τ at ε = 1."""
    x = np.array([0.5, 1.0, 4.0])
    t1 = tau_per_epsilon_sq(1e-11, x, cell_averaged=False)
    np.testing.assert_allclose(
        conversion_probability(3e-8, 1e-11, x).tau, 9e-16 * t1, rtol=1e-12
    )


def test_cell_average_piece_between_two_tangencies():
    """m = 1e-12: cell [2.675, 3.025] holds both tangencies (x ≈ 2.730, 2.988).

    Target: composite Gauss–Legendre in u = |x − x_t|^(1/2) toward each
    tangency on its own side, with tangencies located here by bisection on the
    crossing count. τ(x) has small kinks, so the target itself moves by about
    0.1% with the panel count (8 to 128 panels give 3.0786e-8 to 3.0810e-8).
    The 0.3% tolerance sits above that floor and below the 0.8% error of an
    average that ignores the tangencies (``n_sub_tangent=2``).
    """
    eps, m = 1e-7, 1e-12

    def count(x):
        return len(conversion_probability(eps, m, [x]).crossings[0])

    def locate(lo, hi):
        c_lo = count(lo)
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if count(mid) == c_lo else (lo, mid)
        return 0.5 * (lo + hi)

    t1, t2 = locate(2.70, 2.75), locate(2.95, 3.00)
    nodes, weights = np.polynomial.legendre.leggauss(32)

    def u_int(t, a, b, panels=8):
        top = np.sqrt(b - t) if t <= a else np.sqrt(t - a)
        edges = np.linspace(0.0, top, panels + 1)
        total = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            u = lo + 0.5 * (hi - lo) * (nodes + 1.0)
            x = t + u**2 if t <= a else t - u**2
            tau = conversion_probability(eps, m, x).tau
            total += 0.5 * (hi - lo) * np.sum(weights * tau * 2 * u)
        return total

    a, b, mid = 2.675, 3.025, 0.5 * (t1 + t2)
    ref = (
        u_int(t1, a, t1) + u_int(t1, t1, mid) + u_int(t2, mid, t2) + u_int(t2, t2, b)
    ) / (b - a)
    got = cell_average(eps, m, [2.5, 2.85, 3.2])[0][1]
    assert got == pytest.approx(ref, rel=3e-3)


def test_empty_grid_and_invalid_sub_sample_counts():
    conv = conversion_probability(1e-7, 1e-11, [])
    assert conv.tau.size == 0 and conv.crossings == []
    with pytest.raises(ValueError):
        cell_average(1e-7, 1e-11, [1.0, 2.0], n_sub_cell=0)
    with pytest.raises(ValueError):
        cell_average(1e-7, 1e-11, [1.0, 2.0], n_sub_tangent=0)
