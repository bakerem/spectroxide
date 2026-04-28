"""Adversarial input tests for the Python spectroxide API.

Each test documents an ATTACK (bad input) and GUARD (validation that blocks it).
"""

import math

import numpy as np
import pytest

from spectroxide import _validation as _val
from spectroxide.solver import Cosmology, run_single

# =========================================================================
# Section 1: Bad Cosmology dataclass construction
# =========================================================================


class TestBadCosmology:
    def test_negative_h(self):
        # ATTACK: h < 0 gives negative H0
        # GUARD: __post_init__ calls validate_cosmology which rejects h <= 0
        with pytest.raises(ValueError, match="h must be positive"):
            Cosmology(h=-0.7)

    def test_zero_h(self):
        # ATTACK: h=0 causes div-by-zero in derived quantities
        # GUARD: validate_cosmology rejects h <= 0
        with pytest.raises(ValueError, match="h must be positive"):
            Cosmology(h=0.0)

    def test_nan_h(self):
        # ATTACK: NaN h propagates silently
        # GUARD: validate_finite_scalar catches NaN
        with pytest.raises(ValueError, match="h must be finite"):
            Cosmology(h=float("nan"))

    def test_inf_h(self):
        # ATTACK: infinite H0
        # GUARD: validate_finite_scalar catches Inf
        with pytest.raises(ValueError, match="h must be finite"):
            Cosmology(h=float("inf"))

    def test_huge_h(self):
        # ATTACK: h=100 is nonsensical
        # GUARD: validate_cosmology rejects h > 10
        with pytest.raises(ValueError, match="nonsensical"):
            Cosmology(h=100.0)

    def test_negative_omega_b(self):
        # ATTACK: negative baryon density
        # GUARD: validate_cosmology rejects omega_b <= 0
        with pytest.raises(ValueError, match="omega_b must be positive"):
            Cosmology(omega_b=-0.04)

    def test_yp_one(self):
        # ATTACK: Y_p=1.0 causes div-by-zero in f_he
        # GUARD: validate_cosmology requires y_p < 1.0
        with pytest.raises(ValueError, match="y_p"):
            Cosmology(y_p=1.0)

    def test_yp_negative(self):
        # ATTACK: negative helium fraction
        # GUARD: validate_cosmology requires y_p >= 0
        with pytest.raises(ValueError, match="y_p"):
            Cosmology(y_p=-0.1)

    def test_nan_t_cmb(self):
        # ATTACK: NaN temperature
        # GUARD: validate_finite_scalar catches NaN
        with pytest.raises(ValueError, match="t_cmb must be finite"):
            Cosmology(t_cmb=float("nan"))

    def test_zero_t_cmb(self):
        # ATTACK: zero temperature
        # GUARD: validate_cosmology rejects t_cmb <= 0
        with pytest.raises(ValueError, match="t_cmb must be positive"):
            Cosmology(t_cmb=0.0)

    def test_negative_neff(self):
        # ATTACK: negative neutrino species
        # GUARD: validate_n_eff rejects n_eff < 0
        with pytest.raises(ValueError, match="n_eff"):
            Cosmology(n_eff=-1.0)

    def test_huge_neff(self):
        # ATTACK: N_eff = 100 far exceeds physical
        # GUARD: validate_n_eff rejects n_eff > 20
        with pytest.raises(ValueError, match="n_eff"):
            Cosmology(n_eff=100.0)

    def test_omega_m_less_than_omega_b(self):
        # ATTACK: omega_m < omega_b implies negative CDM
        # GUARD: validate_cosmology rejects omega_m < omega_b
        with pytest.raises(ValueError, match="omega_m must be >= omega_b"):
            Cosmology(omega_b=0.3, omega_m=0.1)

    def test_all_presets_valid(self):
        # GUARD: all presets construct without error
        Cosmology.default()
        Cosmology.planck2015()
        Cosmology.planck2018()


# =========================================================================
# Section 2: Bad greens_function inputs
# =========================================================================


class TestBadGreensFunction:
    def test_negative_zh(self):
        # ATTACK: negative injection redshift
        # GUARD: validate_z_h raises
        with pytest.raises(ValueError, match="z_h must be positive"):
            run_single(z_h=-1e5, delta_rho=1e-5)

    def test_zero_zh(self):
        # ATTACK: z_h=0 is at present day, no distortion physics
        # GUARD: validate_z_h raises
        with pytest.raises(ValueError, match="z_h must be positive"):
            run_single(z_h=0.0, delta_rho=1e-5)

    def test_nan_delta_rho(self):
        # ATTACK: NaN energy injection
        # GUARD: validate_delta_rho raises
        with pytest.raises(ValueError, match="delta_rho must be finite"):
            run_single(z_h=1e5, delta_rho=float("nan"))


# =========================================================================
# Section 3: Bad heating integration inputs
# =========================================================================


class TestBadHeatingIntegration:
    def test_backwards_z_range(self):
        # ATTACK: z_min > z_max
        # GUARD: validate_z_range raises
        with pytest.raises(ValueError, match="z_min must be less than z_max"):
            _val.validate_z_range(1e6, 1e3)

    def test_negative_z_min(self):
        # ATTACK: negative lower bound
        # GUARD: validate_z_range raises
        with pytest.raises(ValueError, match="z_min must be non-negative"):
            _val.validate_z_range(-100, 1e6)

    def test_nz_too_small(self):
        # ATTACK: n_z < 10 gives wildly inaccurate integration
        # GUARD: validate_z_range raises
        with pytest.raises(ValueError, match="n_z must be >= 10"):
            _val.validate_z_range(1e3, 1e6, n_z=3)


# =========================================================================
# Section 4: Bad decompose_distortion inputs
# =========================================================================


class TestBadDecompose:
    def test_length_mismatch(self):
        # ATTACK: x and delta_n have different lengths
        # GUARD: validate_array_lengths raises
        with pytest.raises(ValueError, match="same length"):
            _val.validate_array_lengths([1, 2, 3], [1, 2])


# =========================================================================
# Section 5: Bad run_single inputs
# =========================================================================


class TestBadRunSingle:
    def test_no_zh_no_dqdz(self):
        # ATTACK: neither z_h nor dq_dz provided
        # GUARD: run_single raises
        with pytest.raises(ValueError, match="Either z_h.*or dq_dz"):
            run_single()

    def test_inf_delta_rho(self):
        # ATTACK: infinite energy injection
        # GUARD: validate_delta_rho raises
        with pytest.raises(ValueError, match="delta_rho must be finite"):
            run_single(z_h=1e5, delta_rho=float("inf"))


# =========================================================================
# Section 6: Direct validator unit tests
# =========================================================================


class TestValidators:
    def test_validate_finite_scalar_nan(self):
        with pytest.raises(ValueError, match="must be finite"):
            _val.validate_finite_scalar(float("nan"), "test_val")

    def test_validate_finite_scalar_inf(self):
        with pytest.raises(ValueError, match="must be finite"):
            _val.validate_finite_scalar(float("inf"), "test_val")

    def test_validate_finite_scalar_neg_inf(self):
        with pytest.raises(ValueError, match="must be finite"):
            _val.validate_finite_scalar(float("-inf"), "test_val")

    def test_validate_finite_scalar_normal(self):
        # Should not raise
        _val.validate_finite_scalar(42.0, "test_val")
        _val.validate_finite_scalar(0.0, "test_val")
        _val.validate_finite_scalar(-1.5, "test_val")

    def test_validate_finite_scalar_none(self):
        # None should be silently accepted
        _val.validate_finite_scalar(None, "test_val")

    def test_require_zh_none(self):
        with pytest.raises(ValueError, match="z_h is required"):
            _val.require_z_h(None)

    def test_require_zh_negative(self):
        with pytest.raises(ValueError, match="z_h must be positive"):
            _val.require_z_h(-100)

    def test_require_zh_valid(self):
        # Should not raise
        _val.require_z_h(1e5)

    def test_validate_n_eff_negative(self):
        with pytest.raises(ValueError, match="n_eff"):
            _val.validate_n_eff(-1.0)

    def test_validate_n_eff_too_large(self):
        with pytest.raises(ValueError, match="n_eff"):
            _val.validate_n_eff(25.0)

    def test_validate_n_eff_valid(self):
        # Should not raise
        _val.validate_n_eff(3.046)
        _val.validate_n_eff(0.0)
        _val.validate_n_eff(20.0)

    def test_validate_cosmology_dict_nan(self):
        # ATTACK: NaN in dict-style cosmology
        with pytest.raises(ValueError, match="must be finite"):
            _val.validate_cosmology({"h": float("nan")})

    def test_validate_cosmology_dict_yp_one(self):
        # ATTACK: y_p=1.0 via dict
        with pytest.raises(ValueError, match="y_p"):
            _val.validate_cosmology({"y_p": 1.0})

    def test_validate_cosmology_none(self):
        # None should be accepted
        _val.validate_cosmology(None)

    def test_validate_x_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            _val.validate_x_positive(np.array([-1.0, 1.0, 2.0]))

    def test_validate_x_inj_zero(self):
        with pytest.raises(ValueError, match="x_inj must be positive"):
            _val.validate_x_inj(0.0)
