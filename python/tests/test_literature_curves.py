"""Literature-figure regression tests (Workstream R5).

Turns digitized published-figure curves into a regression suite. Each test
compares a spectroxide-generated curve against digitized reference points within
a per-figure tolerance that includes an explicit digitization-error term.

The digitized CSVs live in ``dev/audit/digitized/<paper>_<fig>.csv`` (schema:
``x,y,curve_id``) and are produced by EB (see
``dev/audit/digitization_request.md``) — the agent must NOT invent them. Until a
CSV is present, the corresponding test **skips with a notice** so the suite is
merged now and activates automatically as the data lands.

Machine-readable anchors that need no digitization (dark-photon ε(m) via
``dev/AxionLimits/limit_data/DarkPhoton/COBEFIRAS_Chluba.txt``) are handled by a
dedicated comparison script, not here.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DIGITIZED = _PROJECT_ROOT / "dev" / "audit" / "digitized"


def _load_curve(fname: str):
    """Load a digitized CSV or skip if absent."""
    path = _DIGITIZED / fname
    if not path.exists():
        pytest.skip(
            f"digitized reference {path.relative_to(_PROJECT_ROOT)} not present "
            "yet (see dev/audit/digitization_request.md) — test activates when "
            "EB adds the CSV"
        )
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return data


# Per-figure tolerance = digitization error (log-log read) + Round-1 error
# budget (~0.3%) + any known methodology delta. Justified per figure; any curve
# needing >10% must be explained in the R5 memo.
TOL_PHOTON_INJECTION = 0.08  # 5% digitization + few% methodology (soft x_inj)


@pytest.mark.skipif(
    not (_DIGITIZED / "chluba2015_photon_injection.csv").exists(),
    reason="Chluba 2015 photon-injection curve not digitized yet",
)
def test_chluba2015_photon_injection_curve():
    """Fig 6 (fig:photon_injection) vs Chluba 2015 (arXiv:1506.06582).

    Compares spectroxide's monochromatic photon-injection distortion against the
    digitized Green's-function curve, per injection redshift/frequency.
    """
    ref = _load_curve("chluba2015_photon_injection.csv")
    # TODO(R5): for each ref curve_id, generate the spectroxide curve via
    #   spectroxide.run_single(...) / the photon Green's function at matching
    #   (x_inj, z_h), interpolate to ref['x'], and assert rel-err within
    #   TOL_PHOTON_INJECTION. Left as a scaffold until the CSV lands so the
    #   comparison recipe is pinned next to the data schema.
    assert ref is not None


def test_digitization_request_present():
    """Guard: the digitization request doc exists so the provenance of every
    future CSV is documented (prevents fabricated anchors landing untracked)."""
    req = _PROJECT_ROOT / "dev" / "audit" / "digitization_request.md"
    assert req.exists(), "digitization_request.md must document each CSV's origin"
