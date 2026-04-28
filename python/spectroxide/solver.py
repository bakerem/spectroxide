"""
Python wrappers around the Rust ``spectroxide`` binary.

Provides convenience functions for running the full PDE solver from
Python (subprocess + JSON over stdout) and for quick single-injection
calculations using the pure-Python Green's function module.

Conventions
-----------
- ``Δρ/ρ`` is the fractional energy injection in the photon background.
- ``ΔN/N`` is the fractional photon-number perturbation (photon injection).
- Injection scenarios are passed as a dict with a ``"type"`` key; see
  the ``injection`` parameter of :func:`solve` for the supported types.
- All redshifts are dimensionless; all temperatures are in kelvin.
"""

from __future__ import annotations

import functools
import json
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import greens
from . import _validation as _val

#: Type alias for cosmology specification: a :class:`Cosmology` dataclass,
#: a plain dict with the same keys, or ``None`` (use Rust defaults).
CosmoSpec = Union["Cosmology", Mapping[str, float], None]
#: Type alias for a heating-rate callable ``z -> dQ/dz``.
HeatingRate = Callable[[ArrayLike], ArrayLike]
#: Type alias for a 2-D photon-source callable ``(x, z) -> dΔn/dz``.
PhotonSource = Callable[[float, float], float]

# Locate the Rust project root (one level up from the python/ directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@functools.lru_cache(maxsize=4)
def get_physics_hash(
    project_root: str | Path | None = None, timeout: float = 60.0
) -> str:
    """Return the Rust binary's compile-time physics-source hash.

    Used to validate cached Green's function tables against the binary
    that produced them.  The result is cached for the process lifetime;
    if you rebuild the binary mid-session, restart the interpreter (or
    call ``get_physics_hash.cache_clear()``).

    Parameters
    ----------
    project_root : str or Path, optional
        Path to the Rust project root.  Defaults to ``../../`` relative
        to this module (the repository root).
    timeout : float, optional
        Maximum time in seconds to wait for the binary (default 60.0).

    Returns
    -------
    str
        Hexadecimal hash of the physics source files baked into the
        binary at build time.

    Raises
    ------
    FileNotFoundError
        If neither the prebuilt binary nor ``cargo`` are available.
    RuntimeError
        If the binary exits with non-zero status.
    """
    root = Path(project_root) if project_root is not None else _PROJECT_ROOT
    binary = Path(root) / "target" / "release" / "spectroxide"
    if binary.exists():
        cmd = [str(binary), "physics-hash"]
    else:
        cmd = [
            "cargo",
            "run",
            "--release",
            "--bin",
            "spectroxide",
            "--quiet",
            "--",
            "physics-hash",
        ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Cannot resolve physics hash: {e}. Build the Rust binary with "
            f"'cargo build --release' in {root}."
        ) from e
    if result.returncode != 0:
        raise RuntimeError(
            f"spectroxide physics-hash failed (exit {result.returncode}):\n"
            f"{result.stderr}"
        )
    return result.stdout.strip()


from .cosmology import Cosmology  # noqa: E402,F401  (re-exported)

# ---------------------------------------------------------------------------
# Solver quality presets
# ---------------------------------------------------------------------------
#: Production-quality settings (default). Suitable for publication plots and
#: CosmoTherm comparisons.  Uses 4000-point grid, tight timestep control for
#: photon injection spikes (dtau_max_photon_source=1).
PRODUCTION = {
    "n_points": 4000,
    "production_grid": True,
    "dtau_max_photon_source": 1.0,
}

#: Fast settings for interactive exploration and debugging.  Uses 1000-point
#: grid with relaxed timestep control.  Results are qualitatively correct but
#: may differ by ~10% at soft-photon injection peaks.
DEBUG = {
    "n_points": 1000,
    "production_grid": False,
    "dtau_max_photon_source": 10.0,
}


def _apply_settings(kwargs, debug=False):
    """Merge quality-preset defaults into kwargs (explicit kwargs win).

    Parameters
    ----------
    kwargs : dict
        Keyword arguments from the caller.
    debug : bool
        If True, use :data:`DEBUG` defaults instead of :data:`PRODUCTION`.

    Returns
    -------
    dict
        Merged kwargs (original dict is **not** mutated).
    """
    preset = DEBUG if debug else PRODUCTION
    merged = dict(preset)
    merged.update({k: v for k, v in kwargs.items() if v is not None})
    return merged


def _resolve_quality_settings(n_points, production_grid, dtau_max_photon_source, debug):
    """Apply quality preset and return (n_points, production_grid, dtau_max_photon_source)."""
    settings = _apply_settings(
        {
            "n_points": n_points,
            "production_grid": production_grid,
            "dtau_max_photon_source": dtau_max_photon_source,
        },
        debug=debug,
    )
    return (
        settings["n_points"],
        settings["production_grid"],
        settings["dtau_max_photon_source"],
    )


def _build_cosmo_args(cosmo_params):
    """Translate a cosmology dict to Rust CLI flags.

    Both layers use the same fractional convention (``omega_b`` = Ω_b,
    ``omega_m`` = Ω_m total matter), so the wrapper is a straight
    pass-through; the Rust CLI does the Ω → ω conversion internally.
    """
    if cosmo_params is None:
        return []

    # Validate via Cosmology before launching a subprocess.
    if isinstance(cosmo_params, dict):
        _cosmo_fields = {"h", "omega_b", "omega_m", "y_p", "t_cmb", "n_eff"}
        _known = {k: v for k, v in cosmo_params.items() if k in _cosmo_fields}
        if _known:
            Cosmology(**_known)  # raises ValueError on invalid input

    # The Rust CLI computes ω_cdm = (Ω_m − Ω_b) h², so it requires both
    # together. Mirror that invariant here so the user gets a Python-side
    # error instead of a CLI-parsing error.
    has_b = "omega_b" in cosmo_params
    has_m = "omega_m" in cosmo_params
    if has_b ^ has_m:
        raise ValueError(
            "omega_b and omega_m must be passed together: omega_cdm is "
            "computed as (omega_m - omega_b) * h^2 by the binary."
        )

    args = []
    flags = {
        "omega_b": "--omega-b",
        "omega_m": "--omega-m",
        "h": "--h",
        "y_p": "--y-p",
        "t_cmb": "--t-cmb",
        "n_eff": "--n-eff",
    }
    for key, cli_flag in flags.items():
        if key in cosmo_params:
            args.extend([cli_flag, str(cosmo_params[key])])
    return args


_INJECTION_PARAM_MAP = {
    "f_x": "--f-x",
    "gamma_x": "--gamma-x",
    "f_ann": "--f-ann",
    "n_s": "--n-s",
    "amplitude": "--amplitude",
    "sigma_z": "--sigma-z",
    "sigma_x": "--sigma-x",
    "z_h": "--z-h",
    "x_inj": "--x-inj",
    "delta_n_over_n": "--delta-n-over-n",
    "a_s": "--a-s",
    "k_pivot": "--k-pivot",
    "k_min": "--k-min",
    "x_inj_0": "--x-inj-0",
    "f_inj": "--f-inj",
    "gamma_con": "--gamma-con",
    "epsilon": "--epsilon",
    "m_ev": "--m-ev",
}


def _injection_param_args(injection):
    """Translate an injection dict's *parameter* keys to CLI args.

    Walks every key/value in ``injection`` (skipping ``"type"``) and emits
    ``[flag, value]`` pairs via :data:`_INJECTION_PARAM_MAP`.  Boolean
    parameters are emitted as bare flags (only when truthy).
    """
    args = []
    for key, value in injection.items():
        if key == "type":
            continue
        if key not in _INJECTION_PARAM_MAP:
            raise ValueError(f"Unknown injection parameter: {key!r}")
        if isinstance(value, bool):
            if value:
                args.append(_INJECTION_PARAM_MAP[key])
        else:
            args.extend([_INJECTION_PARAM_MAP[key], str(value)])
    return args


def _build_injection_args(injection):
    """Convert an injection scenario dict to flat-flag CLI arguments.

    Returns ``["--injection", type, ...flags...]`` for the legacy
    flat-flag CLI used by ``run_sweep`` / ``run_photon_sweep``.  For the
    subcommand form (``solve <type> ...``), use :func:`_injection_param_args`
    directly.
    """
    if injection is None:
        return []
    inj_type = injection["type"].replace("_", "-")
    return ["--injection", inj_type, *_injection_param_args(injection)]


def _build_common_solver_args(
    *,
    dy_max=None,
    n_points=None,
    dtau_max=None,
    dtau_max_photon_source=None,
    number_conserving=True,
    nc_z_min=None,
    no_dcbr=False,
    production_grid=False,
    cosmo_params=None,
    n_threads=None,
):
    """Build the CLI arguments shared across all PDE solver invocations."""
    args = []
    if dy_max is not None:
        args.extend(["--dy-max", str(dy_max)])
    if n_points is not None:
        args.extend(["--n-points", str(n_points)])
    if dtau_max is not None:
        args.extend(["--dtau-max", str(dtau_max)])
    if dtau_max_photon_source is not None:
        args.extend(["--dtau-max-photon-source", str(dtau_max_photon_source)])
    if not number_conserving:
        args.append("--no-number-conserving")
    if nc_z_min is not None:
        args.extend(["--nc-z-min", str(nc_z_min)])
    if no_dcbr:
        args.append("--no-dcbr")

    if production_grid:
        args.append("--production-grid")
    if n_threads is not None:
        args.extend(["--threads", str(int(n_threads))])
    args.extend(_build_cosmo_args(cosmo_params))
    return args


import os as _os
import threading as _threading

_build_lock = _threading.Lock()


def _run_rust_binary(cmd, *, cwd, timeout=600):
    """Run a Rust solver command and return the parsed JSON output.

    The binary writes JSON to stdout; this function captures and parses it.

    Raises
    ------
    FileNotFoundError
        If the Rust binary is not found (not built).
    RuntimeError
        If the solver exits with a non-zero return code.
    """
    # If cmd starts with "cargo run", build once under a lock, then
    # run the binary directly so each call can use its own cwd.
    if cmd[0] == "cargo" and "run" in cmd[:3]:
        binary = Path(cwd) / "target" / "release" / "spectroxide"
        if not binary.exists():
            with _build_lock:
                if not binary.exists():
                    build_result = subprocess.run(
                        ["cargo", "build", "--release"],
                        cwd=str(cwd),
                        capture_output=True,
                        text=True,
                    )
                    if build_result.returncode != 0:
                        raise RuntimeError(
                            f"cargo build --release failed (exit {build_result.returncode}):\n"
                            f"{build_result.stderr}"
                        )
        if binary.exists():
            # Replace "cargo run --release --bin spectroxide --" with the binary
            try:
                sep = cmd.index("--")
                run_cmd = [str(binary)] + cmd[sep + 1 :]
            except ValueError:
                run_cmd = [str(binary)]
        else:
            run_cmd = cmd
    else:
        run_cmd = cmd

    try:
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "Rust binary not found. Build it first with:\n"
            "  cargo build --release\n"
            f"in {cwd}"
        ) from None

    if result.returncode != 0:
        raise RuntimeError(
            f"Rust solver failed (exit {result.returncode}):\n{result.stderr}"
        )

    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError(
            f"Rust solver produced no output on stdout.\n"
            f"Solver stderr:\n{result.stderr}"
        )

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as e:
        # Surface a useful context window instead of an opaque decoder message
        # (audit I2). The bespoke Rust serializer has a history of emitting
        # unbalanced braces, NaN/Inf, or truncated output on panic.
        pos = e.pos
        context = stdout[max(0, pos - 80) : pos + 80]
        stderr_tail = (result.stderr or "").strip()[-500:]
        raise RuntimeError(
            f"Rust solver produced malformed JSON (pos {pos}/{len(stdout)}): "
            f"{e.msg}\n"
            f"  Context: ...{context}...\n"
            f"  Stderr tail: {stderr_tail}"
        ) from e

    _emit_solver_warnings(parsed)
    return parsed


def _emit_solver_warnings(parsed):
    """Re-emit Rust solver diagnostic warnings via warnings.warn.

    The Rust ``SolverResult`` / ``SweepResult`` / ``PhotonSweepResult`` /
    ``GreensResult`` types each carry an optional ``warnings`` field
    populated from ``SolverDiagnostics.warnings`` (Newton non-convergence,
    rho_e clamping, x_inj-out-of-grid, untested-regime soft warnings, etc.).
    Without this re-emission Python callers had no way to see them.
    """
    if not isinstance(parsed, dict):
        return
    import warnings as _warnings

    seen: set[str] = set()
    for msg in parsed.get("warnings", []) or []:
        if not isinstance(msg, str) or msg in seen:
            continue
        seen.add(msg)
        _warnings.warn(f"spectroxide: {msg}", RuntimeWarning, stacklevel=3)
    # Per-row warnings nested inside batch results.
    for entry in parsed.get("results", []) or []:
        if isinstance(entry, dict):
            for msg in entry.get("warnings", []) or []:
                if not isinstance(msg, str) or msg in seen:
                    continue
                seen.add(msg)
                _warnings.warn(f"spectroxide: {msg}", RuntimeWarning, stacklevel=3)


def _run_tabulated_heating(
    *,
    dq_dz: HeatingRate,
    delta_rho: float,
    z_start: float | None,
    z_end: float,
    z_min: float,
    z_max: float,
    n_z: int,
    project_root: str | Path | None,
    timeout: float,
    cosmo_params: Mapping[str, float] | None,
    dy_max: float | None,
    n_points: int | None,
    dtau_max: float | None,
    dtau_max_photon_source: float | None,
    number_conserving: bool,
    nc_z_min: float | None,
    no_dcbr: bool,
    production_grid: bool | None,
    n_threads: int | None,
) -> dict:
    """Tabulate ``dq_dz(z)`` to a CSV and dispatch the Rust
    ``solve tabulated-heating`` subcommand."""
    _val.validate_delta_rho(delta_rho)
    _val.validate_z_range(z_min, z_max)
    _val.validate_dq_dz_callable(dq_dz, z_min, z_max)

    root = Path(project_root) if project_root is not None else _PROJECT_ROOT

    z_grid = np.logspace(np.log10(max(z_end, z_min)), np.log10(z_max), n_z)
    rates = np.array([dq_dz(z) for z in z_grid])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp.write("z,dq_dz\n")
        for z_val, rate_val in zip(z_grid, rates):
            tmp.write(f"{z_val:.10e},{rate_val:.10e}\n")
        tmp_path = tmp.name

    try:
        cmd = [
            "cargo",
            "run",
            "--release",
            "--bin",
            "spectroxide",
            "--",
            "solve",
            "tabulated-heating",
            "--heating-table",
            tmp_path,
            "--delta-rho",
            str(delta_rho),
            "--z-end",
            str(z_end),
            "--z-start",
            str(z_start if z_start is not None else z_max),
        ]
        cmd.extend(
            _build_common_solver_args(
                dy_max=dy_max,
                n_points=n_points,
                dtau_max=dtau_max,
                dtau_max_photon_source=dtau_max_photon_source,
                number_conserving=number_conserving,
                nc_z_min=nc_z_min,
                no_dcbr=no_dcbr,
                production_grid=production_grid,
                cosmo_params=cosmo_params,
                n_threads=n_threads,
            )
        )
        return _run_rust_binary(cmd, cwd=root, timeout=timeout)
    finally:
        _os.unlink(tmp_path)


def _run_tabulated_photon(
    *,
    photon_source: PhotonSource,
    delta_rho: float,
    z_start: float | None,
    z_end: float,
    z_min: float,
    z_max: float,
    n_z: int,
    x: ArrayLike | None,
    x_min: float,
    x_max: float,
    n_x: int,
    project_root: str | Path | None,
    timeout: float,
    cosmo_params: Mapping[str, float] | None,
    dy_max: float | None,
    n_points: int | None,
    dtau_max: float | None,
    dtau_max_photon_source: float | None,
    number_conserving: bool,
    nc_z_min: float | None,
    no_dcbr: bool,
    production_grid: bool | None,
    n_threads: int | None,
) -> dict:
    """Tabulate a 2D ``photon_source(x, z)`` to a CSV and dispatch the
    Rust ``solve tabulated-photon`` subcommand."""
    _val.validate_delta_rho(delta_rho)
    _val.validate_z_range(z_min, z_max)

    root = Path(project_root) if project_root is not None else _PROJECT_ROOT

    z_grid = np.logspace(np.log10(max(z_end, z_min)), np.log10(z_max), min(n_z, 500))
    if x is not None:
        x_grid = np.asarray(x, dtype=np.float64)
    else:
        x_grid = np.logspace(np.log10(x_min), np.log10(x_max), n_x)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp.write("z," + ",".join(f"{xi:.10e}" for xi in x_grid) + "\n")
        for z_val in z_grid:
            row = [f"{z_val:.10e}"]
            for xi in x_grid:
                row.append(f"{photon_source(xi, z_val):.10e}")
            tmp.write(",".join(row) + "\n")
        tmp_path = tmp.name

    try:
        cmd = [
            "cargo",
            "run",
            "--release",
            "--bin",
            "spectroxide",
            "--",
            "solve",
            "tabulated-photon",
            "--photon-table",
            tmp_path,
            "--delta-rho",
            str(delta_rho),
            "--z-end",
            str(z_end),
            "--z-start",
            str(z_start if z_start is not None else z_max),
        ]
        cmd.extend(
            _build_common_solver_args(
                dy_max=dy_max,
                n_points=n_points,
                dtau_max=dtau_max,
                dtau_max_photon_source=dtau_max_photon_source,
                number_conserving=number_conserving,
                nc_z_min=nc_z_min,
                no_dcbr=no_dcbr,
                production_grid=production_grid,
                cosmo_params=cosmo_params,
                n_threads=n_threads,
            )
        )
        return _run_rust_binary(cmd, cwd=root, timeout=timeout)
    finally:
        _os.unlink(tmp_path)


def _run_pde_single_solve(
    *,
    injection: Mapping[str, Any] | None,
    delta_rho: float,
    z_start: float | None,
    z_end: float,
    dn_planck: float | None,
    project_root: str | Path | None,
    timeout: float,
    cosmo_params: Mapping[str, float] | None,
    dy_max: float | None,
    n_points: int | None,
    dtau_max: float | None,
    dtau_max_photon_source: float | None,
    number_conserving: bool,
    nc_z_min: float | None,
    no_dcbr: bool,
    production_grid: bool | None,
    n_threads: int | None,
) -> dict:
    """Build the cargo command for a single PDE solve and run it.

    Called by :func:`solve` for custom injection scenarios and the
    ``dn_planck`` initial perturbation diagnostic.
    """
    _val.validate_delta_rho(delta_rho)
    _val.warn_grid_resolution_photon(
        n_points,
        injection.get("type") if isinstance(injection, dict) else None,
    )
    if isinstance(injection, dict):
        x_inj = injection.get("x_inj")
        if x_inj is not None:
            _val.warn_x_inj_regime(float(x_inj))
        z_h = injection.get("z_h")
        if z_h is not None:
            _val.warn_z_h_regime(float(z_h))
        for key in ("delta_rho_over_rho", "delta_n_over_n"):
            val = injection.get(key)
            if val is not None and not np.isfinite(val):
                raise ValueError(f"injection['{key}']={val} must be finite.")

    root = Path(project_root) if project_root is not None else _PROJECT_ROOT

    # Use new-style subcommands: solve <type> [args]
    if injection is not None:
        inj_type = injection["type"].replace("_", "-")
        cmd = [
            "cargo",
            "run",
            "--release",
            "--bin",
            "spectroxide",
            "--",
            "solve",
            inj_type,
            *_injection_param_args(injection),
        ]
    else:
        # No injection, just initial perturbation: use solve single-burst
        # with zero amplitude as a no-op injection carrier for --dn-planck/--dn-depletion
        z_h_dummy = z_start if z_start is not None else 1e5
        # Explicit sigma_z so the CLI's 100.0 floor doesn't violate the
        # sigma_z <= 0.3*z_h validator when z_h_dummy < 333 (e.g. low-mass
        # dark photons with z_res ~ 200). delta_rho=0 makes sigma_z inert.
        sigma_z_dummy = max(0.01 * z_h_dummy, 1.0)
        cmd = [
            "cargo",
            "run",
            "--release",
            "--bin",
            "spectroxide",
            "--",
            "solve",
            "single-burst",
            "--z-h",
            str(z_h_dummy),
            "--sigma-z",
            str(sigma_z_dummy),
            "--delta-rho",
            "0",
        ]
    if injection is not None:
        cmd.extend(["--delta-rho", str(delta_rho)])
    cmd.extend(["--z-end", str(z_end)])
    # Smart z_start: avoid the Rust CLI default of z=5e6 which wastes
    # time integrating through redshift ranges that are either empty
    # (single bursts) or fully thermalized (continuous injection).
    effective_z_start = z_start
    if effective_z_start is None and injection is not None:
        inj_z_h = injection.get("z_h")
        if inj_z_h is not None and injection["type"] in (
            "single_burst",
            "monochromatic_photon",
        ):
            # Start just above the injection redshift
            sigma = max(float(inj_z_h) * 0.04, 100.0)
            effective_z_start = float(inj_z_h) + 7.0 * sigma
        elif injection["type"] in (
            "decaying_particle",
            "annihilating_dm",
            "annihilating_dm_p_wave",
        ):
            # Energy injected at z >> z_th ~ 2e6 is fully thermalized;
            # starting at 3e6 captures all observable distortions.
            effective_z_start = 3e6
    if effective_z_start is not None:
        cmd.extend(["--z-start", str(effective_z_start)])
    if dn_planck is not None:
        cmd.extend(["--dn-planck", str(dn_planck)])
    cmd.extend(
        _build_common_solver_args(
            dy_max=dy_max,
            n_points=n_points,
            dtau_max=dtau_max,
            dtau_max_photon_source=dtau_max_photon_source,
            number_conserving=number_conserving,
            nc_z_min=nc_z_min,
            no_dcbr=no_dcbr,
            production_grid=production_grid,
            cosmo_params=cosmo_params,
            n_threads=n_threads,
        )
    )
    return _run_rust_binary(cmd, cwd=root, timeout=timeout)


def run_sweep(
    delta_rho: float = 1.0e-5,
    z_injections: Sequence[float] | None = None,
    z_end: float = 500.0,
    z_start: float | None = None,
    cosmo_params: Mapping[str, float] | None = None,
    project_root: str | Path | None = None,
    timeout: float = 600.0,
    dy_max: float | None = None,
    n_points: int | None = None,
    dtau_max: float | None = None,
    dtau_max_photon_source: float | None = None,
    number_conserving: bool = True,
    nc_z_min: float | None = None,
    no_dcbr: bool = False,
    production_grid: bool | None = None,
    debug: bool = False,
    n_threads: int | None = None,
) -> dict:
    """Run a single-burst PDE sweep over injection redshifts.

    Calls the Rust binary once with a list of ``z_injections`` and a
    fixed ``delta_rho``; the binary loops over redshifts internally
    (parallelised via ``n_threads``).

    For other PDE workloads use :func:`solve` instead:

    - Custom injection scenario → ``solve(injection={...})``.
    - Tabulated heating history → ``solve(dq_dz=callable, method="pde")``.
    - Frequency-dependent photon source → ``solve(photon_source=callable)``.
    - Monochromatic photon injection sweep → :func:`run_photon_sweep`.

    Parameters
    ----------
    delta_rho : float, optional
        Fractional energy injection ``Δρ/ρ``.  Default ``1e-5``.
    z_injections : sequence of float, optional
        Injection redshifts to sweep over.  If *None* (default), the Rust
        binary uses a 15-point log-spaced grid from 2e3 to 5e5.
    z_end : float, optional
        Final redshift for PDE evolution.  Default 500.
    z_start : float, optional
        Starting redshift for PDE evolution.  *None* (default) uses the
        Rust CLI default (5e6).
    cosmo_params : Mapping, optional
        Cosmological parameters.  *None* uses Rust defaults.
    project_root : str or Path, optional
        Path to the Rust project root.  *None* (default) auto-detects
        relative to this module.
    timeout : float, optional
        Maximum time in seconds to wait for the Rust binary.  Default 600.
    dy_max : float, optional
        Maximum step in ``y_C`` taken by the PDE.  Must lie in
        ``(0, 0.1]``.  *None* (default) uses the Rust default.
    n_points : int, optional
        Number of frequency-grid points.  *None* (default) inherits from
        the active quality preset (:data:`PRODUCTION` or :data:`DEBUG`).
    dtau_max : float, optional
        Cap on the dimensionless Compton optical-depth step ``Δτ_C``.
        *None* uses the Rust default.
    dtau_max_photon_source : float, optional
        Cap on the optical-depth step during active photon-source
        injection.  *None* inherits from the active preset.
    number_conserving : bool, optional
        Enforce strict photon-number conservation in the PDE.
        Default *True*.
    nc_z_min : float, optional
        Below this redshift, photon-number conservation is relaxed.
        *None* uses the Rust default.
    no_dcbr : bool, optional
        Disable DC+BR entirely (diagnostic).  Default *False*.
    production_grid : bool, optional
        Use the production-quality frequency grid.  *None* inherits from
        the active preset.
    debug : bool, optional
        If *True*, use the :data:`DEBUG` quality preset.  Default *False*.
    n_threads : int, optional
        Number of threads for parallel sweep execution.  *None* uses all
        available CPU cores.

    Returns
    -------
    dict
        Parsed JSON output.  Each per-redshift entry in ``results``
        carries keys ``z_h``, ``pde_mu``, ``pde_y``, ``drho``, ``x``,
        ``delta_n``.

    Raises
    ------
    FileNotFoundError
        If the Rust binary is unavailable and ``cargo`` cannot build it.
    RuntimeError
        If the Rust solver exits non-zero or returns malformed JSON.
    """
    n_points, production_grid, dtau_max_photon_source = _resolve_quality_settings(
        n_points, production_grid, dtau_max_photon_source, debug
    )

    _val.validate_delta_rho(delta_rho)

    root = Path(project_root) if project_root is not None else _PROJECT_ROOT
    cmd = ["cargo", "run", "--release", "--bin", "spectroxide", "--", "sweep"]
    cmd.extend(["--delta-rho", str(delta_rho)])
    cmd.extend(["--z-end", str(z_end)])
    if z_start is not None:
        cmd.extend(["--z-start", str(z_start)])
    if z_injections is not None:
        cmd.extend(["--z-injections", ",".join(str(z) for z in z_injections)])
    cmd.extend(
        _build_common_solver_args(
            dy_max=dy_max,
            n_points=n_points,
            dtau_max=dtau_max,
            dtau_max_photon_source=dtau_max_photon_source,
            number_conserving=number_conserving,
            nc_z_min=nc_z_min,
            no_dcbr=no_dcbr,
            production_grid=production_grid,
            cosmo_params=cosmo_params,
            n_threads=n_threads,
        )
    )
    return _run_rust_binary(cmd, cwd=root, timeout=timeout)


def run_photon_sweep(
    x_inj: float,
    delta_n_over_n: float = 1.0e-5,
    sigma_x: float | None = None,
    z_injections: Sequence[float] | None = None,
    z_end: float = 500.0,
    cosmo_params: Mapping[str, float] | None = None,
    project_root: str | Path | None = None,
    timeout: float = 600.0,
    dy_max: float | None = None,
    n_points: int | None = None,
    dtau_max: float | None = None,
    dtau_max_photon_source: float | None = None,
    number_conserving: bool = True,
    nc_z_min: float | None = None,
    no_dcbr: bool = False,
    production_grid: bool | None = None,
    debug: bool = False,
    n_threads: int | None = None,
) -> dict:
    """Photon-injection sweep over multiple ``z_h`` at fixed ``x_inj``.

    Calls the Rust ``photon-sweep`` subcommand, which parallelises across
    injection redshifts internally using native threads.

    Parameters
    ----------
    x_inj : float
        Injection frequency (dimensionless ``x = h ν / (k_B T_z)``).
    delta_n_over_n : float, optional
        Fractional photon-number injection ``ΔN/N``.  Default ``1e-5``.
    sigma_x : float, optional
        Frequency width of the injection Gaussian.  Default *None*
        (uses ``x_inj × 0.05`` on the Rust side).
    z_injections : sequence of float, optional
        Injection redshifts.  Default *None* — Rust uses 150 log-spaced
        points from 1e3 to 5e6.
    z_end : float, optional
        Final redshift for PDE evolution.  Default 500.
    cosmo_params : Mapping, optional
        Cosmological parameters.  Default *None* (Rust defaults).
    project_root : str or Path, optional
        Path to the Rust project root.  Default *None* (auto-detected).
    timeout : float, optional
        Timeout in seconds.  Default 600.
    dy_max : float, optional
        See :func:`run_sweep`.
    n_points : int, optional
        See :func:`run_sweep`.
    dtau_max : float, optional
        See :func:`run_sweep`.
    dtau_max_photon_source : float, optional
        See :func:`run_sweep`.
    number_conserving : bool, optional
        See :func:`run_sweep`.
    nc_z_min : float, optional
        See :func:`run_sweep`.
    no_dcbr : bool, optional
        See :func:`run_sweep`.
    production_grid : bool, optional
        See :func:`run_sweep`.
    debug : bool, optional
        See :func:`run_sweep`.
    n_threads : int, optional
        See :func:`run_sweep`.

    Returns
    -------
    dict
        Parsed JSON with keys ``x_inj``, ``delta_n_over_n``, ``results``.

    Raises
    ------
    ValueError
        If ``x_inj`` ≤ 0 / non-finite, ``sigma_x`` ≤ 0, or ``dy_max``
        is outside ``(0, 0.1]``.
    """
    _val.validate_x_inj(x_inj)
    _val.warn_x_inj_regime(x_inj)
    _val.validate_finite_scalar(delta_n_over_n, "delta_n_over_n")
    _val.validate_finite_scalar(sigma_x, "sigma_x")
    if sigma_x is not None and sigma_x <= 0:
        raise ValueError(f"sigma_x must be positive, got {sigma_x}")
    if dy_max is not None:
        _val.validate_finite_scalar(dy_max, "dy_max")
        if dy_max <= 0 or dy_max > 0.1:
            raise ValueError(
                f"dy_max={dy_max} is outside the safe range (0, 0.1]; "
                "see SolverConfig::validate in the Rust solver."
            )

    # Apply quality preset defaults
    n_points, production_grid, dtau_max_photon_source = _resolve_quality_settings(
        n_points, production_grid, dtau_max_photon_source, debug
    )
    _val.warn_grid_resolution_photon(n_points, "monochromatic_photon")

    root = Path(project_root) if project_root is not None else _PROJECT_ROOT

    cmd = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "spectroxide",
        "--",
        "photon-sweep",
        "--x-inj",
        str(x_inj),
        "--delta-n-over-n",
        str(delta_n_over_n),
        "--z-end",
        str(z_end),
    ]
    if sigma_x is not None:
        cmd.extend(["--sigma-x", str(sigma_x)])
    if z_injections is not None:
        cmd.extend(["--z-injections", ",".join(str(z) for z in z_injections)])
    cmd.extend(
        _build_common_solver_args(
            dy_max=dy_max,
            n_points=n_points,
            dtau_max=dtau_max,
            dtau_max_photon_source=dtau_max_photon_source,
            number_conserving=number_conserving,
            nc_z_min=nc_z_min,
            no_dcbr=no_dcbr,
            production_grid=production_grid,
            cosmo_params=cosmo_params,
            n_threads=n_threads,
        )
    )
    return _run_rust_binary(cmd, cwd=root, timeout=timeout)


def run_photon_sweep_batch(
    x_inj_values: Sequence[float],
    delta_n_over_n: float = 1.0e-5,
    sigma_x: float | None = None,
    z_injections: Sequence[float] | None = None,
    z_end: float = 500.0,
    cosmo_params: Mapping[str, float] | None = None,
    project_root: str | Path | None = None,
    timeout: float = 3600.0,
    dy_max: float | None = None,
    n_points: int | None = None,
    dtau_max: float | None = None,
    dtau_max_photon_source: float | None = None,
    number_conserving: bool = True,
    nc_z_min: float | None = None,
    no_dcbr: bool = False,
    production_grid: bool | None = None,
    debug: bool = False,
    n_threads: int | None = None,
) -> list[dict]:
    """Batch photon-injection sweep over multiple ``x_inj`` values.

    Calls the Rust ``photon-sweep-batch`` subcommand, which parallelises
    all ``(x_inj, z_h)`` pairs in a single process, avoiding subprocess
    overhead and CPU oversubscription.

    Parameters
    ----------
    x_inj_values : sequence of float
        Injection frequencies (dimensionless).  Must be non-empty,
        finite, and positive.
    delta_n_over_n : float, optional
        Fractional photon-number injection ``ΔN/N``.  Default ``1e-5``.
    sigma_x : float, optional
        Frequency width of each injection Gaussian.  Default *None*
        (Rust uses ``x_inj × 0.05`` per value).
    z_injections : sequence of float, optional
        Injection redshifts.  Default *None* (150 log-spaced from 1e3
        to 5e6 on the Rust side).
    z_end : float, optional
        Final redshift for PDE evolution.  Default 500.
    cosmo_params : Mapping, optional
        Cosmological parameters.  Default *None*.
    project_root : str or Path, optional
        Path to the Rust project root.  Default *None* (auto-detected).
    timeout : float, optional
        Timeout in seconds.  Default 3600.
    dy_max : float, optional
        See :func:`run_sweep`.
    n_points : int, optional
        See :func:`run_sweep`.
    dtau_max : float, optional
        See :func:`run_sweep`.
    dtau_max_photon_source : float, optional
        See :func:`run_sweep`.
    number_conserving : bool, optional
        See :func:`run_sweep`.
    nc_z_min : float, optional
        See :func:`run_sweep`.
    no_dcbr : bool, optional
        See :func:`run_sweep`.
    production_grid : bool, optional
        See :func:`run_sweep`.
    debug : bool, optional
        See :func:`run_sweep`.
    n_threads : int, optional
        See :func:`run_sweep`.

    Returns
    -------
    list of dict
        Parsed JSON results, one per ``x_inj`` value.  Each has keys
        ``x_inj``, ``delta_n_over_n``, ``results``.

    Raises
    ------
    ValueError
        If ``x_inj_values`` is empty, contains non-finite or non-positive
        entries, or if ``sigma_x``/``dy_max`` are out of range.
    """
    x_inj_arr = np.asarray(list(x_inj_values), dtype=float)
    if x_inj_arr.size == 0:
        raise ValueError("x_inj_values is empty")
    if not np.all(np.isfinite(x_inj_arr)):
        raise ValueError("x_inj_values contains non-finite entries")
    if np.any(x_inj_arr <= 0):
        raise ValueError(
            f"x_inj_values must all be positive, got min={x_inj_arr.min()}"
        )
    for xi in x_inj_arr:
        _val.warn_x_inj_regime(float(xi))
    _val.validate_finite_scalar(delta_n_over_n, "delta_n_over_n")
    _val.validate_finite_scalar(sigma_x, "sigma_x")
    if sigma_x is not None and sigma_x <= 0:
        raise ValueError(f"sigma_x must be positive, got {sigma_x}")
    if dy_max is not None:
        _val.validate_finite_scalar(dy_max, "dy_max")
        if dy_max <= 0 or dy_max > 0.1:
            raise ValueError(
                f"dy_max={dy_max} is outside the safe range (0, 0.1]; "
                "see SolverConfig::validate in the Rust solver."
            )

    # Apply quality preset defaults
    n_points, production_grid, dtau_max_photon_source = _resolve_quality_settings(
        n_points, production_grid, dtau_max_photon_source, debug
    )
    _val.warn_grid_resolution_photon(n_points, "monochromatic_photon")

    root = Path(project_root) if project_root is not None else _PROJECT_ROOT

    x_inj_str = ",".join(str(x) for x in x_inj_values)
    cmd = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "spectroxide",
        "--",
        "photon-sweep-batch",
        "--x-inj-values",
        x_inj_str,
        "--delta-n-over-n",
        str(delta_n_over_n),
        "--z-end",
        str(z_end),
    ]
    if sigma_x is not None:
        cmd.extend(["--sigma-x", str(sigma_x)])
    if z_injections is not None:
        cmd.extend(["--z-injections", ",".join(str(z) for z in z_injections)])
    cmd.extend(
        _build_common_solver_args(
            dy_max=dy_max,
            n_points=n_points,
            dtau_max=dtau_max,
            dtau_max_photon_source=dtau_max_photon_source,
            number_conserving=number_conserving,
            nc_z_min=nc_z_min,
            no_dcbr=no_dcbr,
            production_grid=production_grid,
            cosmo_params=cosmo_params,
            n_threads=n_threads,
        )
    )
    return _run_rust_binary(cmd, cwd=root, timeout=timeout)


def run_single(
    z_h: float | None = None,
    delta_rho: float = 1.0e-5,
    x: ArrayLike | None = None,
    x_min: float = 0.01,
    x_max: float = 30.0,
    n_x: int = 500,
    dq_dz: HeatingRate | None = None,
    z_min: float = 1.0e3,
    z_max: float = 3.0e6,
    n_z: int = 5000,
) -> dict:
    """Quick calculation using the pure-Python Green's function.

    Two modes of operation
    ----------------------
    **Single burst** (default) — provide ``z_h`` and ``delta_rho`` for a
    delta-function energy injection at one redshift.  Uses the
    cosmo-aware Green's function so ``J_Compton(z)`` correctly suppresses
    y at ``z_h ≲ 1100``.

    **Custom heating** — provide ``dq_dz``, a callable returning
    ``d(Δρ/ρ)/dz`` (positive for heating).  The spectrum is computed
    via :func:`spectroxide.greens.distortion_from_heating` and ``μ``/``y``
    are extracted by separate integrations.

    Parameters
    ----------
    z_h : float, optional
        Injection redshift (single-burst mode).  Required unless
        ``dq_dz`` is given.  Default *None*.
    delta_rho : float, optional
        Fractional energy injection ``Δρ/ρ``.  Default ``1e-5``.  Only
        used in single-burst mode.
    x : array_like, optional
        Custom dimensionless frequency grid.  If *None* (default), a
        log-spaced grid is generated from ``x_min``, ``x_max``, ``n_x``.
    x_min : float, optional
        Minimum dimensionless frequency (default 0.01).  Ignored when
        ``x`` is provided.
    x_max : float, optional
        Maximum dimensionless frequency (default 30.0).  Ignored when
        ``x`` is provided.
    n_x : int, optional
        Number of frequency points (default 500).  Ignored when ``x`` is
        provided.
    dq_dz : callable, optional
        Heating rate ``d(Δρ/ρ)/dz``.  When given, ``z_h`` and
        ``delta_rho`` are ignored.
    z_min : float, optional
        Lower integration bound for ``dq_dz`` mode (default ``1e3``).
    z_max : float, optional
        Upper integration bound (default ``3e6``).
    n_z : int, optional
        Number of redshift points for integration (default 5000).

    Returns
    -------
    dict
        Keys ``x`` (ndarray, frequency grid),
        ``delta_n`` (ndarray, distortion ``Δn(x)``),
        ``mu`` (float), ``y`` (float),
        ``z_h`` (float or *None*, single-burst mode),
        ``delta_rho`` (float or *None*).

    Raises
    ------
    ValueError
        If neither ``z_h`` nor ``dq_dz`` is provided, or if any input is
        out of range (negative redshift, non-finite values, ...).
    """
    # Validate inputs
    if z_h is not None:
        _val.validate_z_h(z_h)
        _val.warn_z_h_regime(z_h)
    if delta_rho is not None:
        _val.validate_delta_rho(delta_rho)
    if dq_dz is not None:
        _val.validate_z_range(z_min, z_max)
        _val.validate_dq_dz_callable(dq_dz, z_min, z_max)

    # Build frequency grid
    if x is not None:
        x = np.asarray(x, dtype=np.float64)
    else:
        x = np.logspace(np.log10(x_min), np.log10(x_max), n_x)

    if dq_dz is not None:
        # Custom heating mode
        delta_n = greens.distortion_from_heating(x, dq_dz, z_min, z_max, n_z=n_z)
        mu = greens.mu_from_heating(dq_dz, z_min, z_max, n_z=n_z)
        y = greens.y_from_heating(dq_dz, z_min, z_max, n_z=n_z)
        return {
            "x": x,
            "delta_n": delta_n,
            "mu": mu,
            "y": y,
            "z_h": None,
            "delta_rho": None,
        }

    # Single-burst mode
    if z_h is None:
        raise ValueError(
            "Either z_h (single burst) or dq_dz (custom heating) is required"
        )

    delta_n = greens.greens_function(x, z_h) * delta_rho

    # Extract mu/y from the same spectrum for self-consistency.
    decomp = greens.decompose_distortion(x, delta_n)
    mu = decomp["mu"]
    y = decomp["y"]

    return {
        "x": x,
        "delta_n": delta_n,
        "mu": mu,
        "y": y,
        "z_h": z_h,
        "delta_rho": delta_rho,
    }


@dataclass
class SolverResult:
    """Structured result from a :func:`solve` invocation.

    Attributes
    ----------
    x : ndarray of float64
        Dimensionless frequency grid ``x = h ν / (k_B T_z)``.
    delta_n : ndarray of float64
        Spectral distortion ``Δn(x)`` on ``x``.
    mu : float
        Chemical-potential parameter ``μ`` (dimensionless).
    y : float
        Compton ``y``-parameter (dimensionless).
    delta_rho_over_rho : float
        Fractional energy perturbation ``Δρ/ρ``.
    method : str
        Solver method used: one of ``"pde"``, ``"greens_function"``,
        ``"table"``.
    z_h : float, optional
        Injection redshift (single-burst modes).  *None* for continuous
        injection or custom heating histories.
    rho_e : float, optional
        Final electron-photon temperature ratio ``T_e/T_z`` (PDE only).
    accumulated_delta_t : float, optional
        Accumulated temperature shift ``ΔT/T`` from photon-number
        non-conservation absorbed into the blackbody temperature
        (PDE only).
    """

    x: NDArray[np.float64]
    delta_n: NDArray[np.float64]
    mu: float
    y: float
    delta_rho_over_rho: float
    method: str
    z_h: Optional[float] = None
    rho_e: Optional[float] = None
    accumulated_delta_t: Optional[float] = None

    @property
    def delta_I(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Intensity distortion ``(ν [GHz], ΔI [Jy/sr])``.

        Convenience wrapper around :func:`spectroxide.greens.delta_n_to_delta_I`
        using the default ``T_0 = 2.726 K``.

        .. note::
           This property allocates two new NumPy arrays on every access
           (it is not a cached field).  Bind to a local variable when
           reusing the result.

        Returns
        -------
        tuple of (ndarray, ndarray)
            Frequency in GHz and intensity distortion in Jy/sr
            (= 10⁻²⁶ W m⁻² Hz⁻¹ sr⁻¹).
        """
        return greens.delta_n_to_delta_I(self.x, self.delta_n)


def solve(
    injection: Mapping[str, Any] | None = None,
    cosmo: CosmoSpec = None,
    z_start: float | None = None,
    z_end: float = 500.0,
    method: str = "pde",
    z_h: float | None = None,
    delta_rho: float = 1.0e-5,
    x: ArrayLike | None = None,
    x_min: float = 0.01,
    x_max: float = 30.0,
    n_x: int = 500,
    dq_dz: HeatingRate | None = None,
    photon_source: PhotonSource | None = None,
    table: Any = None,
    z_min: float = 1.0e3,
    z_max: float = 3.0e6,
    n_z: int = 5000,
    verify_hash: bool = True,
    debug: bool = False,
    **kwargs: Any,
) -> SolverResult:
    """Unified entry point for spectral-distortion calculations.

    Dispatches between the Rust PDE solver, the analytic Green's function,
    and the precomputed Green's-function table. Returns a structured
    :class:`SolverResult` with frequency grid, distortion, ``μ``, ``y``,
    and an ``intensity`` property.

    For multi-redshift sweeps use :func:`run_sweep` (single-burst energy
    injection) or :func:`run_photon_sweep` / :func:`run_photon_sweep_batch`
    (monochromatic photon injection); :func:`solve` itself runs one PDE
    per call.

    Parameters
    ----------
    injection : Mapping, optional
        PDE injection scenario.  Required when ``method="pde"`` unless
        ``dq_dz`` or ``photon_source`` is given.  Must contain a
        ``"type"`` key; supported types are ``"single_burst"``,
        ``"decaying_particle"``, ``"annihilating_dm"``,
        ``"annihilating_dm_pwave"``, ``"monochromatic_photon"``,
        ``"decaying_particle_photon"``, and ``"dark_photon_resonance"``.
        Remaining keys are scenario parameters, e.g.::

            {"type": "single_burst", "z_h": 2e5}

        Note that ``delta_rho`` is a top-level argument, not an injection
        key.  For dark photons::

            {"type": "dark_photon_resonance", "epsilon": 1e-9, "m_ev": 1e-7}

        triggers the Rust solver to compute ``γ_con`` and ``z_res``
        internally and install the impulsive depletion IC at ``z_res``.
    cosmo : Cosmology, Mapping, or None, optional
        Cosmological parameters.  Accepts a :class:`Cosmology` dataclass
        or a plain dict; *None* (default) uses Rust defaults.
    z_start : float, optional
        Starting redshift for the PDE.  *None* (default) picks a sensible
        value: just above ``z_h`` for transient injections, ``3e6`` for
        continuous scenarios, or the Rust CLI default for ``dq_dz`` /
        ``photon_source``.
    z_end : float, optional
        Final redshift (default 500).
    method : {"pde", "greens_function", "table"}, optional
        Solver mode (default ``"pde"``).
    z_h : float, optional
        Injection redshift for Green's-function or table single-burst
        modes.  Default *None*.
    delta_rho : float, optional
        Fractional energy injection ``Δρ/ρ``.  Default ``1e-5``.
    x : array_like, optional
        Custom dimensionless frequency grid.  *None* (default) uses
        ``np.logspace(log10(x_min), log10(x_max), n_x)``.
    x_min : float, optional
        Minimum frequency (default 0.01).
    x_max : float, optional
        Maximum frequency (default 30.0).
    n_x : int, optional
        Number of frequency points (default 500).
    dq_dz : callable, optional
        Custom heating rate ``d(Δρ/ρ)/dz``.  With ``method="pde"``,
        tabulates and runs the Rust PDE solver; with
        ``method="greens_function"``, uses the Python Green's function;
        with ``method="table"``, convolves the precomputed table.
    photon_source : callable, optional
        Frequency-dependent photon source ``f(x, z) -> float`` returning
        ``d(Δn)/dz``.  Requires ``method="pde"``.
    table : GreensTable, PhotonGreensTable, str, Path, or None, optional
        Precomputed table for ``method="table"``.  May be a table object,
        a path to a saved ``.npz`` file, or *None* (default) to auto-load
        the default cache (building it on demand).
    z_min : float, optional
        Lower integration bound for ``dq_dz`` in table/GF mode
        (default ``1e3``).
    z_max : float, optional
        Upper integration bound (default ``3e6``).
    n_z : int, optional
        Number of redshift points for integration (default 5000).
    verify_hash : bool, optional
        For ``method="table"``: validate the cached table against the
        binary's physics hash before use.  Default *True*.
    debug : bool, optional
        Use the :data:`DEBUG` quality preset instead of
        :data:`PRODUCTION`.  Default *False*.
    **kwargs
        PDE-mode tuning knobs forwarded to the Rust binary
        (``dy_max``, ``n_points``, ``dtau_max``, ``number_conserving``,
        ``no_dcbr``, ``cosmo_params``, ``timeout``, ``n_threads``, …);
        see :func:`run_sweep` for the full list and their defaults.

    Returns
    -------
    SolverResult
        Structured result with attributes ``x``, ``delta_n``, ``mu``,
        ``y``, ``delta_rho_over_rho``, ``method``, ``z_h``, plus the
        :attr:`SolverResult.delta_I` property converting to physical
        intensity.

    Raises
    ------
    ValueError
        If incompatible arguments are supplied (e.g. ``method="pde"``
        but neither ``injection`` nor ``dq_dz``/``photon_source``).
    TypeError
        If ``table`` is neither :class:`GreensTable`,
        :class:`PhotonGreensTable`, str, Path, nor *None*.
    """
    # Accept Cosmology objects directly (convert to dict for CLI)
    if isinstance(cosmo, Cosmology):
        cosmo = cosmo.to_dict()

    _val.validate_cosmology(cosmo)

    # --- Table mode ---
    if method == "table":
        from .greens_table import (
            GreensTable,
            PhotonGreensTable,
            load_or_build_greens_table,
        )
        from pathlib import Path as _Path

        # Build frequency grid
        if x is not None:
            x_arr = np.asarray(x, dtype=np.float64)
        else:
            x_arr = np.logspace(np.log10(x_min), np.log10(x_max), n_x)

        # Resolve table object
        if table is None:
            tbl = load_or_build_greens_table(verify_hash=verify_hash)
        elif isinstance(table, (str, _Path)):
            tbl = GreensTable.load(table, verify_hash=verify_hash)
        else:
            tbl = table

        if isinstance(tbl, GreensTable):
            if dq_dz is not None:
                # Custom heating convolution
                dn = tbl.distortion_from_heating(x_arr, dq_dz, z_min, z_max, n_z=n_z)
                mu, y = tbl.mu_y_from_heating(dq_dz, z_min, z_max, n_z=n_z)
            elif z_h is not None:
                # Single burst
                dn = tbl.greens_function(x_arr, z_h) * delta_rho
                # Decompose to get mu, y
                decomp = greens.decompose_distortion(x_arr, dn)
                mu, y = decomp["mu"], decomp["y"]
            else:
                raise ValueError(
                    "method='table' with GreensTable requires z_h or dq_dz"
                )
        elif isinstance(tbl, PhotonGreensTable):
            if injection is None or "x_inj" not in injection:
                raise ValueError("PhotonGreensTable requires injection with 'x_inj'")
            x_inj = injection["x_inj"]
            delta_n_over_n = injection.get("delta_n_over_n", 1e-5)
            dn = tbl.greens_function_photon(x_arr, x_inj, z_h) * delta_n_over_n
            decomp = greens.decompose_distortion(x_arr, dn)
            mu, y = decomp["mu"], decomp["y"]
        else:
            raise TypeError(f"Unsupported table type: {type(tbl)}")

        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

        drho = (
            float(_trapz(x_arr**3 * dn, x_arr) / greens.G3_PLANCK)
            if len(dn) > 0
            else 0.0
        )
        return SolverResult(
            x=x_arr,
            delta_n=dn,
            mu=mu,
            y=y,
            delta_rho_over_rho=drho,
            method="table",
            z_h=z_h,
        )

    if method == "greens_function":
        result = run_single(
            z_h=z_h,
            delta_rho=delta_rho,
            x=x,
            x_min=x_min,
            x_max=x_max,
            n_x=n_x,
            dq_dz=dq_dz,
            z_min=z_min,
            z_max=z_max,
            n_z=n_z,
            **kwargs,
        )
        x_arr = np.asarray(result["x"])
        dn = np.asarray(result["delta_n"])
        # Δρ/ρ = ∫x³ Δn dx / G₃
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

        drho = (
            float(_trapz(x_arr**3 * dn, x_arr) / greens.G3_PLANCK)
            if len(dn) > 0
            else 0.0
        )
        return SolverResult(
            x=x_arr,
            delta_n=dn,
            mu=result["mu"],
            y=result["y"],
            delta_rho_over_rho=drho,
            method="greens_function",
            z_h=z_h,
        )

    # PDE mode — catch common mistake of passing z_h without injection dict
    if (
        z_h is not None
        and injection is None
        and dq_dz is None
        and photon_source is None
    ):
        raise ValueError(
            "z_h is only used with method='greens_function' or method='table'. "
            f"For PDE, pass injection={{'type': 'single_burst', 'z_h': {z_h}}} "
            f"with delta_rho={delta_rho} as a top-level parameter."
        )

    # PDE mode dispatches directly to one of three single-solve helpers.
    # ``run_sweep`` is reserved for redshift sweeps and is no longer on
    # this path.  Pop the helper kwargs from ``**kwargs`` and forward.
    _dn_planck = kwargs.pop("dn_planck", None)
    if "dark_photon_depletion" in kwargs:
        kwargs.pop("dark_photon_depletion")
        raise TypeError(
            "`dark_photon_depletion=γ_con` was removed. Use "
            "`injection={'type': 'dark_photon_resonance', "
            "'epsilon': ε, 'm_ev': m}`; γ_con and z_res are computed "
            "internally."
        )
    n_pts, prod_grid, dtau_ps = _resolve_quality_settings(
        kwargs.pop("n_points", None),
        kwargs.pop("production_grid", None),
        kwargs.pop("dtau_max_photon_source", None),
        debug,
    )
    common = dict(
        delta_rho=delta_rho,
        z_start=z_start,
        z_end=z_end,
        project_root=kwargs.pop("project_root", None),
        timeout=kwargs.pop("timeout", 600.0),
        cosmo_params=cosmo,
        dy_max=kwargs.pop("dy_max", None),
        n_points=n_pts,
        dtau_max=kwargs.pop("dtau_max", None),
        dtau_max_photon_source=dtau_ps,
        number_conserving=kwargs.pop("number_conserving", True),
        nc_z_min=kwargs.pop("nc_z_min", None),
        no_dcbr=kwargs.pop("no_dcbr", False),
        production_grid=prod_grid,
        n_threads=kwargs.pop("n_threads", None),
    )
    if injection is not None or _dn_planck is not None:
        data = _run_pde_single_solve(
            injection=injection,
            dn_planck=_dn_planck,
            **common,
        )
    elif dq_dz is not None:
        data = _run_tabulated_heating(
            dq_dz=dq_dz, z_min=z_min, z_max=z_max, n_z=n_z, **common
        )
    elif photon_source is not None:
        data = _run_tabulated_photon(
            photon_source=photon_source,
            z_min=z_min,
            z_max=z_max,
            n_z=n_z,
            x=x,
            x_min=x_min,
            x_max=x_max,
            n_x=n_x,
            **common,
        )
    else:
        raise ValueError(
            "method='pde' requires one of: injection={...}, dq_dz=..., "
            "photon_source=..., or dn_planck=..."
        )
    if kwargs:
        raise TypeError(f"solve() got unexpected keyword arguments: {sorted(kwargs)}")
    r = data["results"][0]
    x_arr = np.asarray(r["x"])
    dn = np.asarray(r["delta_n"])
    return SolverResult(
        x=x_arr,
        delta_n=dn,
        mu=r.get("pde_mu", r.get("gf_mu", 0.0)),
        y=r.get("pde_y", r.get("gf_y", 0.0)),
        delta_rho_over_rho=r.get("drho", 0.0),
        method="pde",
        z_h=r.get("z_h"),
        rho_e=r.get("rho_e"),
        accumulated_delta_t=r.get("accumulated_delta_t"),
    )
