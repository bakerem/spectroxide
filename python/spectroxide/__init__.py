"""
spectroxide: Python API for CMB spectral distortion calculations.

This package provides:

- ``greens``: Pure-Python Green's function implementation for fast approximate
  spectral distortion calculations (ported from the Rust library).
- ``solver``: Wrapper that calls the Rust PDE solver binary.

Quick start::

    from spectroxide import solve, greens_function

    # Unified entry point (handles both GF and PDE modes)
    result = solve(injection={'type': 'single_burst', 'z_h': 2e5}, delta_rho=1e-5)
    result_gf = solve(method="greens_function", z_h=2e5, delta_rho=1e-5)

    # Direct Green's function evaluation
    dn = greens_function(3.0, 2e5)

For redshift sweeps use ``run_sweep()`` (single-burst energy injection) or
``run_photon_sweep()`` / ``run_photon_sweep_batch()`` (monochromatic photon
injection); the Rust binary loops internally and parallelises across cores.
``run_single()`` is a thin convenience wrapper around the analytic Green's
function.

G_bb stripping (``strip_gbb``) and style helpers (``apply_style``, ``C``,
``SINGLE_COL``, ``DOUBLE_COL``) are re-exported at top level.

CosmoTherm comparison utilities are available via submodule import::

    from spectroxide.cosmotherm import load_di_file, convolve_cosmotherm_gf

Plot parameter constants are available via submodule import::

    from spectroxide.plot_params import FONT_SIZE, LW
"""

from .cosmology import (
    DEFAULT_COSMO,
    PLANCK2015_COSMO,
    PLANCK2018_COSMO,
    ionization_fraction,
    hubble,
    n_hydrogen,
    n_electron,
    omega_gamma,
    rho_gamma,
    cosmic_time,
    baryon_photon_ratio,
)
from .greens import (
    # Constants
    Z_MU,
    BETA_MU,
    KAPPA_C,
    G3_PLANCK,
    G2_PLANCK,
    G1_PLANCK,
    ALPHA_RHO,
    X_BALANCED,
    # Spectral shapes
    planck,
    g_bb,
    mu_shape,
    y_shape,
    temperature_shift_shape,
    # Visibility functions
    j_bb,
    j_bb_star,
    j_mu,
    j_y,
    # Green's function (energy injection)
    greens_function,
    distortion_from_heating,
    mu_from_heating,
    y_from_heating,
    # Green's function (photon injection)
    x_c_dc,
    x_c_br,
    x_c,
    photon_survival_probability,
    greens_function_photon,
    mu_from_photon_injection,
    distortion_from_photon_injection,
    # Decomposition
    decompose_distortion,
    # Unit conversion
    delta_n_to_delta_I,
)

from .cosmotherm import strip_gbb
from .solver import (
    run_sweep,
    run_photon_sweep,
    run_photon_sweep_batch,
    run_single,
    solve,
    SolverResult,
    Cosmology,
    PRODUCTION,
    DEBUG,
    get_physics_hash,
)
from .greens_table import (
    GreensTable,
    PhotonGreensTable,
    GreensTableHashMismatch,
    load_or_build_greens_table,
    load_or_build_photon_greens_table,
)
from .firas import FIRASData, MU_FIRAS_95, Y_FIRAS_95, MU_FIRAS_68, Y_FIRAS_68
from .style import apply_style, C, SINGLE_COL, DOUBLE_COL

__version__ = "0.1.0"

__all__ = [
    # Constants (8)
    "Z_MU",
    "BETA_MU",
    "KAPPA_C",
    "G3_PLANCK",
    "G2_PLANCK",
    "G1_PLANCK",
    "ALPHA_RHO",
    "X_BALANCED",
    # Cosmology presets (3)
    "DEFAULT_COSMO",
    "PLANCK2015_COSMO",
    "PLANCK2018_COSMO",
    # Spectral shapes (5)
    "planck",
    "g_bb",
    "mu_shape",
    "y_shape",
    "temperature_shift_shape",
    # Visibility functions (4)
    "j_bb",
    "j_bb_star",
    "j_mu",
    "j_y",
    # Green's function — energy injection (4)
    "greens_function",
    "distortion_from_heating",
    "mu_from_heating",
    "y_from_heating",
    # Green's function — photon injection (7)
    "x_c_dc",
    "x_c_br",
    "x_c",
    "photon_survival_probability",
    "greens_function_photon",
    "mu_from_photon_injection",
    "distortion_from_photon_injection",
    # Recombination (1)
    "ionization_fraction",
    # Cosmology functions (6)
    "hubble",
    "n_hydrogen",
    "n_electron",
    "omega_gamma",
    "rho_gamma",
    "cosmic_time",
    # Cosmology helpers (1)
    "baryon_photon_ratio",
    # Analysis (3)
    "decompose_distortion",
    "delta_n_to_delta_I",
    "strip_gbb",
    # Solver (10)
    "run_sweep",
    "run_photon_sweep",
    "run_photon_sweep_batch",
    "run_single",
    "solve",
    "SolverResult",
    "Cosmology",
    "PRODUCTION",
    "DEBUG",
    "get_physics_hash",
    # Green's function tables (5)
    "GreensTable",
    "PhotonGreensTable",
    "GreensTableHashMismatch",
    "load_or_build_greens_table",
    "load_or_build_photon_greens_table",
    # FIRAS constraints (5)
    "FIRASData",
    "MU_FIRAS_95",
    "Y_FIRAS_95",
    "MU_FIRAS_68",
    "Y_FIRAS_68",
    # Style (4)
    "apply_style",
    "C",
    "SINGLE_COL",
    "DOUBLE_COL",
]
