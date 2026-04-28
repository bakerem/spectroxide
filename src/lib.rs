//! # spectroxide
//!
//! An open-source solver for the cosmological thermalization problem:
//! computing spectral distortions of the CMB from energy release in the
//! early Universe.
//!
//! ## Overview
//!
//! The code evolves the photon occupation number `n(x, z)` (where
//! `x = hν/(kT_z)` is the dimensionless frequency) through the
//! coupled photon-electron Boltzmann equation in an expanding universe.
//!
//! Physical processes included:
//! - **Compton scattering** (Kompaneets equation): frequency redistribution
//! - **Double Compton emission**: photon-number changing, γe → γγe
//! - **Bremsstrahlung**: photon-number changing, e+ion → e+ion+γ
//! - **Hubble expansion**: cosmological redshifting
//!
//! ## Usage
//!
//! ```rust,no_run
//! use spectroxide::prelude::*;
//!
//! // Create solver with default cosmology
//! let cosmo = Cosmology::default();
//! let mut solver = ThermalizationSolver::new(cosmo, GridConfig::fast());
//!
//! // Set energy injection (e.g., delta-function burst at z=2×10⁵)
//! solver.set_injection(InjectionScenario::SingleBurst {
//!     z_h: 2e5,
//!     delta_rho_over_rho: 1e-5,
//!     sigma_z: 100.0,
//! }).unwrap();
//!
//! // Run the solver, collecting 50 evenly log-spaced snapshots
//! let snapshots = solver.run(50);
//! let mu = snapshots.last().unwrap().mu;
//! ```
//!
//! ## Green's Function Mode
//!
//! For fast approximate calculations of small distortions:
//!
//! ```rust
//! use spectroxide::greens::*;
//!
//! // Distortion from delta-function injection at z_h = 2×10⁵
//! let x = 3.0; // dimensionless frequency
//! let g = greens_function(x, 2e5);
//! ```

/// Content hash of the physics-relevant Rust source files at compile time.
///
/// Used to invalidate cached Green's function tables when the underlying
/// physics code changes. See `build.rs` for the curated file list and the
/// `physics-hash` CLI subcommand for runtime access.
pub const PHYSICS_HASH: &str = env!("PHYSICS_HASH");

pub mod bremsstrahlung;
pub mod cli;
pub mod constants;
pub mod cosmology;
pub mod dark_photon;
pub mod distortion;
pub mod double_compton;
pub mod electron_temp;
pub mod energy_injection;
pub mod greens;
pub mod grid;
pub mod kompaneets;
pub mod output;
pub mod recombination;
pub mod solver;
pub mod spectrum;
/// Convenient imports for common usage.
pub mod prelude {
    pub use crate::constants;
    pub use crate::cosmology::Cosmology;
    pub use crate::dark_photon;
    pub use crate::distortion;
    pub use crate::energy_injection::InjectionScenario;
    pub use crate::greens;
    pub use crate::grid::{FrequencyGrid, GridConfig, RefinementZone};
    pub use crate::output::{
        GreensResult, OutputFormat, PhotonSweepBatchResult, PhotonSweepResult, PhotonSweepRow,
        SolverResult, SweepResult, SweepRow,
    };
    pub use crate::solver::{SolverBuilder, SolverConfig, SolverDiagnostics, ThermalizationSolver};
    pub use crate::spectrum;
}
