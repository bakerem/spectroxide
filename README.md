# spectroxide

[![CI](https://github.com/bakerem/spectroxide/actions/workflows/ci.yml/badge.svg)](https://github.com/bakerem/spectroxide/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bakerem/spectroxide/graph/badge.svg?token=KUQLBC7733)](https://codecov.io/gh/bakerem/spectroxide)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Numerical solver for **CMB spectral distortions** from energy and photon injection in the early Universe.

Energy released into the photon-baryon plasma at redshifts $z \sim 10^3 - 2 \times 10^6$ creates deviations from a perfect blackbody spectrum. These spectral distortions --- $\mu$-type (chemical potential) and $y$-type (Compton) --- encode information about early-universe physics, from the dissipation of primordial acoustic waves to exotic particle decays and dark sector interactions.

spectroxide solves the coupled photon-electron Boltzmann equation including Compton scattering (Kompaneets equation), double Compton emission, bremsstrahlung, and Hubble expansion. It provides both a **full PDE solver** (Rust) and a **fast Green's function approximation** (Rust + Python).

## Features

- **Full PDE solver** in Rust: implicit Kompaneets + coupled DC/BR with adaptive stepping
- **Green's function** mode for fast approximate calculations (pure Python, no compilation needed)
- **9 built-in injection scenarios**: single burst, decaying particles (heat or photon channel), DM annihilation (s-wave/p-wave), dark photon oscillation, monochromatic photon injection, and tabulated sources (plus custom heating via Rust API)
- **Comprehensive test suite**: 430+ unit, integration, and doc-tests
- **Zero production dependencies** in Rust (pure `std` library)

## Installation

### Quick install (recommended)

The install script handles everything --- Rust toolchain, compilation, and Python package:

```bash
git clone https://github.com/bakerem/spectroxide.git
cd spectroxide

# Into a new conda environment (recommended)
./install.sh --conda spectroxide --extras notebook

# Or into your current Python environment
./install.sh
```

The `--extras` flag controls which Python dependencies are installed:

| Extra      | Includes                          | Use case                          |
|------------|-----------------------------------|-----------------------------------|
| `plot`     | numpy, matplotlib                 | Scripts and plotting (default)    |
| `notebook` | numpy, matplotlib, jupyter        | Interactive notebooks             |
| `dev`      | numpy, matplotlib, jupyter, scipy | Development and testing           |
| `doc`      | sphinx, nbsphinx, pydata-sphinx-theme | Building documentation        |

Run `./install.sh --help` to see all options (skip steps, verbose output, etc.).

### Manual installation

<details>
<summary>Click to expand step-by-step instructions</summary>

**Rust** (required for the PDE solver and CLI):

If you don't have Rust installed, the easiest way is via [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

**Python 3.9+** (required for the Python package and notebooks):

```bash
conda create -n spectroxide python=3.11
conda activate spectroxide
```

**Build and install:**

```bash
cargo build --release                 # build Rust PDE solver
cargo test --release                  # run all tests (some solver tests are slow in debug)
pip install -e "python/.[plot]"       # Python package with matplotlib
pip install -e "python/.[notebook]"   # ... or with Jupyter too
```

</details>

## Quick start

### Python: PDE solver

```python
from spectroxide import run_sweep
import numpy as np

# Single-burst injection at z = 2e5 (mu-era)
result = run_sweep(
    injection={"type": "single_burst", "z_h": 2e5, "sigma_z": 5000},
    delta_rho=1e-5,
    z_start=3e5, z_end=1e3,
)
r = result["results"][0]
print(f"mu = {r['pde_mu']:.3e}, y = {r['pde_y']:.3e}")

# Full spectrum is in r["x"], r["delta_n"]
x, delta_n = np.array(r["x"]), np.array(r["delta_n"])

# Decaying particle
result = run_sweep(
    injection={"type": "decaying_particle", "f_x": 5e5, "gamma_x": 5e4},
    z_start=5e6, z_end=1e3,
)

# DM annihilation (s-wave)
result = run_sweep(
    injection={"type": "annihilating_dm", "f_ann": 1e-22},
    z_start=5e6, z_end=1e3,
)

# Dark photon oscillation (NWA resonant conversion)
result = run_sweep(
    injection={"type": "dark_photon_resonance", "epsilon": 1e-7, "m_ev": 1e-5},
    z_end=1e3,  # z_start auto-set to z_res
)
```

### Python: Green's function (fast approximate, no Rust needed)

```python
from spectroxide import run_single, mu_from_heating, y_from_heating
import numpy as np

# Quick estimate via Green's function
result = run_single(z_h=2e5, delta_rho=1e-5)
print(f"mu = {result['mu']:.3e}, y = {result['y']:.3e}")

# mu and y from an arbitrary heating history dQ/dz(z)
dq_dz = lambda z: 1e-5 * np.exp(-((z - 2e5)**2) / (2 * 5000**2)) / (5000 * np.sqrt(2*np.pi))
mu = mu_from_heating(dq_dz, 1e3, 5e6)
y = y_from_heating(dq_dz, 1e3, 5e6)
```

### Rust API

```rust
use spectroxide::prelude::*;

// Green's function (fast, approximate)
let g = greens::greens_function(3.0, 2e5);

// Full PDE solver
let cosmo = Cosmology::default();
let mut solver = ThermalizationSolver::new(cosmo, GridConfig::default());
solver.set_injection(InjectionScenario::SingleBurst {
    z_h: 2e5, delta_rho_over_rho: 1e-5, sigma_z: 5000.0,
});
solver.set_config(SolverConfig {
    z_start: 5e5, z_end: 1e3, ..SolverConfig::default()
});
let snapshots = solver.run_with_snapshots(&[1e3]);
```

### CLI

```bash
# Show help and available subcommands
cargo run --release --bin spectroxide

# Specific injection scenarios (solve subcommand)
cargo run --release --bin spectroxide -- solve decaying-particle --f-x 5e5 --gamma-x 5e4
cargo run --release --bin spectroxide -- solve annihilating-dm --f-ann 1e-22
cargo run --release --bin spectroxide -- solve dark-photon-resonance --epsilon 1e-7 --m-ev 1e-5
# Sweep over injection redshifts
cargo run --release --bin spectroxide -- sweep --delta-rho 1e-5

# Green's function mode (no PDE)
cargo run --release --bin spectroxide -- greens --z-h 2e5 --delta-rho 1e-5
```

Output is written to stdout as JSON (pipe to a file with `> output.json`).

## Example notebooks

| Notebook | Description |
|----------|-------------|
| [`01_getting_started.ipynb`](notebooks/tutorials/01_getting_started.ipynb) | Green's function basics, first PDE runs, PDE vs GF comparison |
| [`02_energy_injection.ipynb`](notebooks/tutorials/02_energy_injection.ipynb) | PDE: decaying particles, DM annihilation (s-wave, p-wave), amplitude scaling |
| [`03_new_physics.ipynb`](notebooks/tutorials/03_new_physics.ipynb) | PDE: dark photon depletion, monochromatic photon injection, $\mu$ sign flip |
| [`04_custom_scenarios.ipynb`](notebooks/tutorials/04_custom_scenarios.ipynb) | Custom injection scenarios and tabulated heating histories |
| [`05_observational_constraints.ipynb`](notebooks/tutorials/05_observational_constraints.ipynb) | FIRAS/PIXIE limits, $\mu$-$y$ plane, mock PIXIE observation |
| [`06_greens_table.ipynb`](notebooks/tutorials/06_greens_table.ipynb) | Precomputed Green's function tables for fast convolution |

Additional notebooks in [`notebooks/physics/`](notebooks/physics/) (photon injection, dark photons) and [`notebooks/observational/`](notebooks/observational/) (FIRAS photon injection limits). Development and validation notebooks are in [`dev/notebooks/`](dev/notebooks/).

## Injection scenarios

| Scenario | Key parameters |
|----------|---------------|
| `SingleBurst` | $z_h$, $\Delta\rho/\rho$ |
| `DecayingParticle` | $f_X$, $\Gamma_X$ |
| `DecayingParticlePhoton` | $x_{\rm inj,0}$, $f_{\rm inj}$, $\Gamma_X$ |
| `AnnihilatingDM` | $f_{\rm ann}$ |
| `AnnihilatingDMPWave` | $f_{\rm ann}$ |
| `MonochromaticPhotonInjection` | $x_{\rm inj}$, $\Delta N/N$, $z_h$ |
| `DarkPhotonResonance` | $\epsilon$, $m_{A'}$ (eV) |
| `TabulatedHeating` | CSV file |
| `TabulatedPhotonSource` | CSV file |
| `Custom` | user-defined function |



## Architecture

```
src/
├── lib.rs                 # Library root + prelude
├── main.rs                # CLI binary entry
├── cli.rs                 # CLI argument parsing and dispatch
├── output.rs              # JSON serialization of solver results
├── kompaneets.rs          # Compton scattering (IMEX Newton solver)
├── double_compton.rs      # DC emission (photon-number changing)
├── bremsstrahlung.rs      # BR emission (non-relativistic Gaunt factor)
├── solver.rs              # PDE integrator (coupled Kompaneets + DC/BR)
├── greens.rs              # Green's function approximation
├── energy_injection.rs    # Injection scenarios
├── dark_photon.rs         # NWA helpers for γ↔A' (plasma freq, γ_con)
├── distortion.rs          # mu/y/DeltaT decomposition + intensity conversion
├── cosmology.rs           # Flat LCDM background
├── spectrum.rs            # Planck, Bose-Einstein, spectral shapes
├── grid.rs                # Non-uniform frequency grid
├── electron_temp.rs       # Electron temperature (quasi-stationary)
├── recombination.rs       # Peebles 3-level atom + Saha
├── constants.rs           # CODATA 2018 constants
└── bin/check_adiabatic.rs # Adiabatic cooling validation utility

python/spectroxide/
├── __init__.py            # Public API
├── greens.py              # Pure Python Green's function (NumPy)
├── greens_table.py        # Precomputed Green's function tables
├── solver.py              # Rust binary wrapper + run_single()
├── firas.py               # FIRAS data and constraint utilities
├── dark_photon.py         # NWA helpers (γ_con, z_res) — Python port
├── cosmotherm.py          # CosmoTherm data loaders (submodule import)
├── plot_params.py         # Plot constants (submodule import)
├── style.py               # Matplotlib style helpers
└── _validation.py         # Input validation (errors + warnings)

tests/
├── heat_injection.rs      # Core physics integration tests
├── adversarial_inputs.rs  # Edge cases, bad inputs
├── cosmotherm_comparison.rs # PDE vs CosmoTherm reference data
├── greens_function_checks.rs # GF spectral shapes, limits, conservation
├── coverage_gaps.rs        # Additional coverage
├── convergence_order.rs   # Grid/timestep convergence
├── cli_integration.rs     # CLI end-to-end
└── science_suite.rs       # End-to-end physics validation

dev/
├── scripts/               # Validation and diagnostic scripts
└── notebooks/             # Validation and dev notebooks (4 notebooks)
```

## Citation

If you use spectroxide in your research, please cite the accompanying paper
and Chluba & Sunyaev (2012). If you use the Green's function mode, please
also cite Chluba (2013) and Chluba (2015):

- Chluba & Sunyaev (2012), "The evolution of CMB spectral distortions in the early Universe", MNRAS 419, 1294 ([arXiv:1109.6552](https://arxiv.org/abs/1109.6552), [doi:10.1111/j.1365-2966.2011.19786.x](https://doi.org/10.1111/j.1365-2966.2011.19786.x)) --- CosmoTherm thermalization solver
- Chluba (2013), "Green's function of the cosmological thermalization problem", MNRAS 434, 352 ([arXiv:1304.6120](https://arxiv.org/abs/1304.6120), [doi:10.1093/mnras/stt1025](https://doi.org/10.1093/mnras/stt1025)) --- Green's function I
- Chluba (2015), "Green's function of the cosmological thermalization problem -- II. Effect of photon injection and constraints", MNRAS 454, 4182 ([arXiv:1506.06582](https://arxiv.org/abs/1506.06582), [doi:10.1093/mnras/stv2243](https://doi.org/10.1093/mnras/stv2243)) --- Green's function II (photon injection)

A machine-readable `CITATION.cff` is also included in the repository root.

## Contributing

Contributions --- new injection scenarios, improved physics, validation, bug fixes ---
are welcome. Most contributions to spectroxide (including the bulk of the original
codebase) are written with LLM assistance, and the workflow is built around that:
the human supplies the physics (analytic limits, paper references, dimensional
checks), the LLM supplies the implementation, and the human is responsible for
verifying that tests have *independent* targets rather than ones calibrated to
the code's own output.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. If you are using an
LLM, also drop [CONTRIBUTING_CLAUDE.md](CONTRIBUTING_CLAUDE.md) into its system
prompt --- it encodes the numerical pitfalls and review rules that have caught
real bugs during development.

## License

[MIT](LICENSE)
