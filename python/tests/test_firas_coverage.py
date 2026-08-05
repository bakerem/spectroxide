"""FIRAS closed-loop coverage calibration (Part II §II.5 of
dev/PLAN_KOMPANEETS_MOMENT_VERIFICATION_2026-07-07.md).

The paper's limits rest on firas.py's fitting machinery. Prior work compared
the CCJ24 statistic (~3%); this calibrates our OWN pipeline's statistical
coverage by Monte Carlo: inject known signals, add noise drawn from the full
43×43 covariance, refit with the PRODUCTION path, and check that recovered
amplitudes are unbiased, error bars are calibrated, 95% intervals cover at 95%,
and the χ² goodness-of-fit follows χ²_dof under the null.

The mock is fed through the identical production fit by setting
``FIRASData.residual_kJy = mock`` before calling ``fit_amplitude`` — no
re-implementation of the linear algebra.

Fast smoke tests run at small N; the full-N run is marked ``slow``.
Run: pytest python/tests/test_firas_coverage.py
     pytest python/tests/test_firas_coverage.py -m slow   (full N)
"""

import os
import sys

import numpy as np
import pytest
from scipy.stats import chi2 as chi2_dist
from scipy.stats import kstest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectroxide.firas import FIRASData  # noqa: E402

SEED = 20260707


def _mc_fits(a_true, n_draws, seed):
    """Draw `n_draws` mock monopoles = a_true·μ-template + noise(cov), refit
    each through the production fit_amplitude. Returns (a_hat, chi2_min,
    chi2_null, sigma_reported) arrays and the fixed 1σ error."""
    fd = FIRASData()
    template = np.asarray(fd.mu_template_kJy())
    cov = fd.cov
    chol = np.linalg.cholesky(cov)
    n = fd.n_freq
    rng = np.random.default_rng(seed)

    orig = fd.residual_kJy.copy()
    a_hat = np.empty(n_draws)
    c_min = np.empty(n_draws)
    c_null = np.empty(n_draws)
    sigma = np.empty(n_draws)
    for i in range(n_draws):
        noise = chol @ rng.standard_normal(n)
        mock = a_true * template + noise
        fd.residual_kJy = mock  # drive the identical production path
        res = fd.fit_amplitude(template)
        a_hat[i] = res["amplitude"]
        c_min[i] = res["chi2_min"]
        c_null[i] = res["chi2_null"]
        sigma[i] = res["sigma"]
    fd.residual_kJy = orig
    return a_hat, c_min, c_null, sigma, n


def _sigma_scale():
    fd = FIRASData()
    return fd.fit_amplitude(np.asarray(fd.mu_template_kJy()))["sigma"]


# ---------------------------------------------------------------------------
# Fast smoke tests
# ---------------------------------------------------------------------------

def test_unbiased_recovery_smoke():
    sig = _sigma_scale()
    n = 800
    for k, a_true in enumerate([0.0, sig, 5.0 * sig]):
        a_hat, *_ , sigma_arr, _ = _mc_fits(a_true, n, SEED + k)
        bias = abs(a_hat.mean() - a_true)
        # Standard error of the mean of a_hat is σ/sqrt(N); allow 4×.
        tol = 4.0 * sig / np.sqrt(n)
        assert bias < tol, f"bias {bias:.3e} > {tol:.3e} at a_true={a_true:.3e}"
        # Reported σ is amplitude-independent (data-independent) and equals sig.
        assert np.allclose(sigma_arr, sig, rtol=1e-9)


def test_gof_null_smoke():
    # Under the null (a_true=0), chi2_null = moccᵀ C⁻¹ mock ~ χ²_n and
    # chi2_min (one param fit) ~ χ²_{n-1}. KS test, loose threshold.
    _, c_min, c_null, _, n = _mc_fits(0.0, 1000, SEED + 100)
    p_null = kstest(c_null, chi2_dist(n).cdf).pvalue
    p_min = kstest(c_min, chi2_dist(n - 1).cdf).pvalue
    assert c_null.mean() == pytest.approx(n, rel=0.1)
    assert c_min.mean() == pytest.approx(n - 1, rel=0.1)
    assert p_null > 1e-3, f"chi2_null not χ²_{n}: KS p={p_null:.2e}"
    assert p_min > 1e-3, f"chi2_min not χ²_{n - 1}: KS p={p_min:.2e}"


# ---------------------------------------------------------------------------
# Full-N calibration (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_error_calibration_full():
    sig = _sigma_scale()
    n = 10000
    a_hat, *_ = _mc_fits(2.0 * sig, n, SEED + 200)
    # Empirical scatter of â must equal the reported σ (calibrated error bars).
    assert a_hat.std(ddof=1) == pytest.approx(sig, rel=0.05)
    # Bias ≪ σ/sqrt(N)·few.
    assert abs(a_hat.mean() - 2.0 * sig) < 4.0 * sig / np.sqrt(n)


@pytest.mark.slow
def test_coverage_95_full():
    sig = _sigma_scale()
    n = 10000
    for k, a_true in enumerate([0.0, sig, 5.0 * sig]):
        a_hat, *_ = _mc_fits(a_true, n, SEED + 300 + k)
        # Two-sided 95% interval â ± 1.96σ must cover a_true 95% of the time.
        covered = np.mean(np.abs(a_hat - a_true) <= 1.96 * sig)
        # Binomial MC error on 95% at N=1e4: σ ≈ sqrt(.95·.05/N) ≈ 2.2e-3; allow 4σ.
        assert abs(covered - 0.95) < 0.01, f"coverage {covered:.4f} at a_true={a_true:.3e}"
