"""Regenerate the Fig. 8 dark-photon FIRAS PDE-limit cache, one mass at a time.

Extracted from ``notebooks/observational/dp_firas_ccj24_overlay.ipynb``
(cells 1, 3, 5, 7), which is the stated provenance of
``dev/data/dp_firas_pde_limits.npz`` (see cell 14 of
``notebooks/paper_figures/dark_photon_constraints.ipynb``). The method is
copied verbatim -- self-consistent eps_ref iteration, profile-likelihood
limit for the eps_ref/converged template, CCJ24-statistic limit as a pure
re-fit of the same template (no extra PDE solve) -- with two changes only:

1. Restartable, one output file per mass, so a run that dies partway
   through can resume (task requirement for a 7 GB / OOM-prone box).
2. No ThreadPoolExecutor: masses are processed one at a time (or by
   launching this script once per mass from a driver that caps
   concurrency), because the notebook's ``workers=10`` would run far too
   many solver subprocesses at once on this machine.

The convergence method is unchanged. Later changes (2026-09-24, audit
A4/A5/C1): the grid's last segment ends at M_MAX (1.5e-4 eV since 2026-09-25) so it keeps 70 masses;
the CCJ24 statistic uses Delta chi2 = 3.84 from the minimum over eps >= 0;
the PDE templates use the default plasma-only photon mass, m_gamma = omega_pl,
as CCJ24 do (ADR 0008 makes neutral hydrogen an option, off here);
each per-mass file records CACHE_VERSION, and files from another version are
recomputed; ``--assemble`` writes the npz from the grid's masses only. The mass
grid (``--list-masses``) is the *overlay* notebook's 70-mass grid with its last segment ending at M_MAX (70 masses), otherwise
as written there; see the task report for how this differs from the old
cache's actual 80-mass content (a discrepancy neither notebook's current
code reproduces).

Usage
-----
    python dev/scripts/dp_firas_pde_limits.py --list-masses
    python dev/scripts/dp_firas_pde_limits.py --mass 1e-9 --outdir DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure cargo is on PATH for the Rust solver binary (matches notebook cell 1)
cargo_bin = Path.home() / ".cargo" / "bin"
if cargo_bin.is_dir() and str(cargo_bin) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(cargo_bin) + os.pathsep + os.environ.get("PATH", "")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from spectroxide import g_bb  # noqa: E402
from spectroxide.dark_photon import gc_per_epsilon_sq, resonance_redshift  # noqa: E402
from spectroxide.firas import FIRASData  # noqa: E402
from spectroxide.solver import solve  # noqa: E402

# --- 70-mass grid from dp_firas_ccj24_overlay.ipynb cell 7, cut at M_MAX ---
MASSES_PDE = np.unique(
    np.concatenate(
        [
            np.geomspace(1e-12, 2e-9, 20),
            np.geomspace(3e-9, 1e-7, 30),
            np.geomspace(2e-7, 5e-5, 12),
            np.geomspace(6e-5, 1.5e-4, 8),
        ]
    )
)
# Fig. 8 stops at m = 1.5e-4 eV, as in the submitted paper (EB, 2026-09-25;
# 1e-4 eV from 2026-09-24 until then). The last segment ends there so the grid
# keeps 70 masses (audit A5).
M_MAX = 1.5e-4
MASSES_PDE = MASSES_PDE[MASSES_PDE <= M_MAX]

# Bump when the physics or the statistics change, so stale per-mass files
# are recomputed instead of reused.
CACHE_VERSION = "2026-09-25-plasma-only-dchi2-3.84"

DTAU_MAX = 3.0  # tighter than paper Fig. 8's default; overlay notebook cell 5
EPS_REF_INIT = 1e-8
EXTRAP_TRIGGER = 3.0
RTOL = 0.05
MAX_ITER = 10

_firas = None


def firas():
    global _firas
    if _firas is None:
        _firas = FIRASData()
    return _firas


# --- overlay notebook cell 3: the two statistics ---------------------------
def _deprojected_model(template_dn_func):
    f = firas()
    S = _firas_dn_to_dI_kJy(f.x, template_dn_func(f.x))
    G = f.gbb_template_kJy()
    g = f.galactic_template_kJy()
    w = 1.0 / f.sigma_kJy**2

    def dot(a, b):
        return float(np.sum(a * w * b))

    m2 = np.array([[dot(G, G), dot(g, G)], [dot(G, g), dot(g, g)]])
    dT, dg = np.linalg.solve(m2, [dot(S, G), dot(S, g)])
    return S - dT * G - dg * g, dot


def _firas_dn_to_dI_kJy(x, dn):
    from spectroxide.firas import _dn_to_dI_kJy

    return _dn_to_dI_kJy(x, dn)


DCHI2_CCJ = 3.84  # CCJ24 reading (EB, 2026-09-24; audit A4)


def ccj24_limit(template_dn_func):
    """CCJ24-style limit: diagonal errors, T and dust removed, and
    chi2(a) = chi2_min + DCHI2_CCJ with the minimum taken over a >= 0.

    chi2(a) - chi2(a_hat) = A (a - a_hat)^2, so the crossing above the
    constrained minimum a* = max(a_hat, 0) is
    a_hat + sqrt((a* - a_hat)^2 + DCHI2_CCJ / A). This is the closed form
    of the grid scan in notebooks/observational/dp_firas_limit_conventions.ipynb
    (column dchi2_3.84), which reproduces the CCJ24 curve.
    """
    S_perp, dot = _deprojected_model(template_dn_func)
    d = firas().residual_kJy
    A = dot(S_perp, S_perp)
    B = dot(d, S_perp)
    a_hat = B / A
    a_star = max(a_hat, 0.0)
    return a_hat + np.sqrt((a_star - a_hat) ** 2 + DCHI2_CCJ / A)


def profile_limit(template_dn_func, cl=0.95):
    return firas().profile_limit_floating_T(template_dn_func, cl=cl)["upper_limit"]


# --- overlay notebook cell 5: PDE templates + eps_ref iteration ------------
def strip_gbb_nc(x, delta_n):
    gbb = g_bb(x)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed it
    alpha = trapz(x**2 * delta_n, x) / trapz(x**2 * gbb, x)
    return delta_n - alpha * gbb, alpha


def run_dp_pde(epsilon, m_ev, npts):
    result = solve(
        injection={"type": "dark_photon_resonance", "epsilon": epsilon, "m_ev": m_ev},
        z_end=100,
        n_points=npts,
        dtau_max=DTAU_MAX,
        timeout=5400,
    )
    return result.x, result.delta_n


def _pde_single_pass(m_dp, eps_ref, npts, limit_fn):
    zr = resonance_redshift(m_dp)
    if zr is None:
        return None, None, None, None
    gpe2, _ = gc_per_epsilon_sq(m_dp)
    gc_ref = gpe2 * eps_ref**2
    x_pde, dn_pde = run_dp_pde(eps_ref, m_dp, npts)
    dn_nc, _ = strip_gbb_nc(x_pde, dn_pde)
    dn_per_gc = dn_nc / gc_ref

    def template(x, _x=x_pde, _dn=dn_per_gc):
        return np.interp(x, _x, _dn)

    gc_95 = limit_fn(template)
    eps_lim = np.sqrt(gc_95 / gpe2) if gc_95 > 0 and gpe2 > 0 else np.inf
    return eps_lim, x_pde, dn_per_gc, zr


def _pde_worker(
    m_dp,
    eps_ref_init,
    limit_fn=None,
    extrap_trigger=EXTRAP_TRIGGER,
    rtol=RTOL,
    max_iter=MAX_ITER,
):
    zr = resonance_redshift(m_dp)
    if zr is None:
        return m_dp, None, None, None, None, 0, False
    npts = 4000 if zr < 1e6 else 8000

    history = []
    eps_ref = eps_ref_init
    converged = False
    for it in range(max_iter):
        eps_lim, x_pde, dn_per_gc, _ = _pde_single_pass(m_dp, eps_ref, npts, limit_fn)
        if eps_lim is None or not np.isfinite(eps_lim):
            break
        history.append((eps_ref, eps_lim, x_pde, dn_per_gc))

        if it > 0:
            prev = history[-2][1]
            if abs(eps_lim - prev) / prev < rtol:
                converged = True
                break
        if it > 1:
            two_back = history[-3][1]
            if abs(eps_lim - two_back) / two_back < rtol:
                hi = -1 if eps_lim > prev else -2
                _, eps_lim, x_pde, dn_per_gc = history[hi]
                converged = True
                break
        if it == 0 and eps_lim / eps_ref < extrap_trigger:
            converged = True
            break
        eps_ref = eps_lim

    if not history:
        return m_dp, zr, None, None, None, 0, False
    return m_dp, zr, eps_lim, x_pde, dn_per_gc, len(history), converged


def process_one_mass(m_dp: float, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"m_{m_dp:.10e}.json"
    if outfile.exists():
        cached = json.loads(outfile.read_text())
        if cached.get("cache_version") == CACHE_VERSION:
            return cached

    t0 = time.time()
    m_dp, zr, eps_pl, x_pde, dn_per_gc, n_iter, converged = _pde_worker(
        m_dp, EPS_REF_INIT, limit_fn=profile_limit
    )
    result = {
        "m": m_dp,
        "z_res": zr,
        "runtime_s": time.time() - t0,
        "cache_version": CACHE_VERSION,
    }
    if eps_pl is None or zr is None:
        result.update({"eps_pl": None, "eps_cj": None, "converged": False, "n_iter": n_iter})
    else:
        gpe2, _ = gc_per_epsilon_sq(m_dp)

        def template(x, _x=x_pde, _dn=dn_per_gc):
            return np.interp(x, _x, _dn)

        gc_cj = ccj24_limit(template)
        eps_cj = np.sqrt(gc_cj / gpe2) if gc_cj > 0 and gpe2 > 0 else np.inf
        result.update(
            {
                "eps_pl": float(eps_pl),
                "eps_cj": float(eps_cj),
                "gpe2": float(gpe2),
                "gc_pl": float(gpe2 * eps_pl**2),
                "gc_cj": float(gpe2 * eps_cj**2),
                "converged": bool(converged),
                "n_iter": int(n_iter),
            }
        )
    outfile.write_text(json.dumps(result))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, help="dark photon mass in eV")
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--list-masses", action="store_true")
    ap.add_argument(
        "--assemble",
        type=Path,
        help="write the Fig. 8 cache (npz) from the per-mass JSON files in --outdir",
    )
    args = ap.parse_args()

    if args.list_masses:
        for m in MASSES_PDE:
            print(f"{m:.10e}")
        return

    if args.assemble is not None:
        rows = []
        for m in MASSES_PDE:
            path = args.outdir / f"m_{m:.10e}.json"
            r = json.loads(path.read_text()) if path.exists() else None
            if r is None or r.get("cache_version") != CACHE_VERSION:
                ap.error(f"missing or stale result for m = {m:.3e} eV in {args.outdir}")
            if r.get("eps_pl") is not None:
                rows.append(r)
        np.savez(
            args.assemble,
            m=np.array([r["m"] for r in rows]),
            e_pl=np.array([r["eps_pl"] for r in rows]),
            e_cj=np.array([r["eps_cj"] for r in rows]),
            gc_pl=np.array([r["gc_pl"] for r in rows]),
            gc_cj=np.array([r["gc_cj"] for r in rows]),
            cv=np.array([r["converged"] for r in rows]),
        )
        print(f"wrote {len(rows)} masses to {args.assemble}")
        return
    if args.mass is None:
        ap.error("--mass, --list-masses or --assemble required")

    result = process_one_mass(args.mass, args.outdir)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
