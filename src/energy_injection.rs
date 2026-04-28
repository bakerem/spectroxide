//! Energy injection scenarios for the early Universe.
//!
//! Provides heating rate functions Q̇(z) for various physical mechanisms:
//! - Single (quasi-instantaneous) energy injection
//! - Decaying particles (heat or photon channel)
//! - Annihilating dark matter (s-wave and p-wave)
//! - Monochromatic photon injection
//! - Tabulated heating / photon source (from external data)
//! - Custom (user-supplied function)
//!
//! All heating rates are expressed as d(Δρ_γ/ρ_γ)/dt in units of 1/s.
//!
//! References:
//! - Chluba & Sunyaev (2012), MNRAS 419, 1294 [Eq. 22-30]

use crate::constants::*;
use crate::cosmology::Cosmology;
use crate::grid::RefinementZone;
use crate::spectrum::planck;

/// Compute vacuum survival fraction for decaying particle photon injection.
///
/// Returns exp(-Γ_X × t(z)), where t(z) is the cosmic time at redshift z.
fn vacuum_survival(z: f64, gamma_x: f64, cosmo: &Cosmology) -> f64 {
    (-gamma_x * cosmo.cosmic_time(z)).exp()
}

/// Energy injection scenario specification.
#[non_exhaustive]
pub enum InjectionScenario {
    /// Single burst at redshift z_h with fractional energy Δρ/ρ.
    SingleBurst {
        /// Central injection redshift.
        z_h: f64,
        /// Fractional energy injected into the photon bath `Δρ/ρ`.
        delta_rho_over_rho: f64,
        /// Gaussian width in redshift of the injection profile.
        sigma_z: f64,
    },

    /// Decaying particle with lifetime 1/Γ_X
    DecayingParticle {
        /// f*_X: energy per baryon released, in eV.
        f_x: f64,
        /// Γ_X: decay rate, in 1/s.
        gamma_x: f64,
    },

    /// Annihilating dark matter (s-wave, <σv> = const)
    AnnihilatingDM {
        /// f_ann: energy injection rate parameter [eV/s].
        /// Defined as f_eff × ⟨σv⟩ × m_χ × n_χ,0² / n_H,0 (paper convention).
        /// Rate: dE/(dt dV) = f_ann × n_H(z) × (1+z)³.
        /// This matches the CosmoTherm convention (e.g. f_ann = 1e-22 eV/s for s-wave).
        f_ann: f64,
    },

    /// Annihilating dark matter (p-wave, <σv> ∝ v² ∝ T ∝ (1+z))
    ///
    /// Rate: dE/(dt dV) = f_ann × n_H(z) × (1+z)⁴, an extra (1+z) relative to
    /// s-wave capturing the velocity-dependent cross section <σv> ∝ v² ∝ T ∝ (1+z).
    AnnihilatingDMPWave {
        /// f_ann: energy injection rate parameter [eV/s].
        /// Same definition as AnnihilatingDM but includes the present-day value
        /// of the velocity-dependent cross section.
        f_ann: f64,
    },

    /// Monochromatic photon injection/removal at a specific frequency.
    ///
    /// Injects ΔN/N photons as a Gaussian in both frequency and redshift,
    /// approximating a delta-function injection at (x_inj, z_h). This creates
    /// both energy AND photon number perturbations, producing qualitatively
    /// different spectral distortions from pure energy injection.
    ///
    /// Key physics: injection at x < x₀ ≈ 3.60 produces **negative** μ.
    ///
    /// The heating rate method returns the energy injection rate from the
    /// photon injection: d(Δρ/ρ)/dt = (α_ρ × x_inj) × d(ΔN/N)/dt.
    /// The frequency-dependent source is applied separately via
    /// `photon_source_rate`.
    ///
    /// References:
    ///   Chluba (2015), arXiv:1506.06582
    ///   Arsenadze et al. (2025), arXiv:2409.12940, Appendix C+D
    MonochromaticPhotonInjection {
        /// Injection frequency (dimensionless x = hν/(kT_z))
        x_inj: f64,
        /// Total fractional photon number to inject: ΔN/N
        delta_n_over_n: f64,
        /// Central injection redshift
        z_h: f64,
        /// Gaussian width in redshift
        sigma_z: f64,
        /// Gaussian width in frequency (should match grid resolution)
        sigma_x: f64,
    },

    /// Decaying particle X → γγ (spontaneous vacuum decay).
    ///
    /// Each decay produces two photons at x_inj(z) = x_inj_0 / (1+z),
    /// where x_inj_0 = E_γ/(kT_0) = m_X c²/(2kT_0).
    ///
    /// The photon source term follows Bolliet & Chluba (2021), Eq. 3:
    ///   dn/dt|_inj = G₂ × f_inj × Γ_X × S(z) × G(x, x_inj, σ_x) / x²
    ///
    /// where S(z) = exp(-Γ_X × t(z)) is the vacuum survival fraction.
    ///
    /// **NOTE**: Stimulated emission (Bose enhancement) is NOT included.
    /// Because X → γγ deposits **both** final-state photons into the same
    /// mode at x_inj, the physical rate carries a factor (1 + n(x_inj))²
    /// (one (1+n) per emitted photon, squared because both land in the
    /// same occupied mode). This is significant at x_inj ≪ 1 where
    /// n_pl ≈ 1/x_inj ≫ 1, and is sub-percent for x_inj ≳ few. See
    /// commit 0b6cfa4 for a prototype implementation.
    ///
    /// All injected photons are routed through `photon_source_rate()` and
    /// are evolved self-consistently by the PDE solver (Compton + DC/BR).
    ///
    /// Reference: Bolliet & Chluba (2021), MNRAS 507, 3148 [arXiv:2012.07292]
    DecayingParticlePhoton {
        /// x_inj,0 = E_γ/(kT_0) = m_X c²/(2kT_0)
        x_inj_0: f64,
        /// Dimensionless injection amplitude (Eq. 5 of B&C 2021).
        /// f_inj ≈ 2(Ω_cdm/Ω_γ) × f_dm × G₃/(G₂ × x_inj,0)
        f_inj: f64,
        /// Γ_X: vacuum decay rate [1/s]
        gamma_x: f64,
    },

    /// Dark photon (γ ↔ A') resonant conversion in the narrow-width
    /// approximation.
    ///
    /// Applied as an **initial condition** at the resonance redshift:
    /// Δn(x) = -[1 - exp(-γ_con/x)] × n_pl(x) at z_start = z_res, where
    /// γ_con = π ε² m² / (|d ln ω_pl²/d ln a|_{z_res} × T_γ(z_res) × H(z_res)).
    /// The solver then evolves this IC with Kompaneets + DC/BR.
    ///
    /// The 1/x factor in the conversion probability captures the frequency
    /// dependence P(x) ∝ 1/ω for ultrarelativistic photons.
    ///
    /// References:
    ///   Mirizzi, Redondo & Sigl (2009), JCAP 0903, 026
    ///   Chluba, Cyr & Johnson (2024), MNRAS 535, 1874
    ///   Arsenadze et al. (2025), JHEP 03, 018
    DarkPhotonResonance {
        /// Kinetic mixing parameter ε
        epsilon: f64,
        /// Dark photon mass m_{A'}, in eV.
        m_ev: f64,
    },

    /// Tabulated heating rate loaded from a file.
    ///
    /// The z_table and rate_table are sorted ascending in z.
    /// `heating_rate_per_redshift(z)` interpolates linearly in log(z),
    /// returning 0 outside the table bounds.
    ///
    /// This enables Python (or any external tool) to define arbitrary
    /// heating rates by writing a CSV table and passing it to the Rust
    /// PDE solver.
    TabulatedHeating {
        /// Redshift grid (ascending)
        z_table: Vec<f64>,
        /// d(Δρ/ρ)/dz at each z (positive = heating)
        rate_table: Vec<f64>,
    },

    /// Tabulated frequency-dependent photon source loaded from a file.
    ///
    /// The source function is defined on a 2D grid (z, x), with bilinear
    /// interpolation.  Returns 0 outside the table bounds.
    TabulatedPhotonSource {
        /// Redshift grid (ascending)
        z_table: Vec<f64>,
        /// Frequency grid (ascending)
        x_grid: Vec<f64>,
        /// Source rate: `source_2d[iz][ix] = d(Δn)/dz` at `(z_table[iz], x_grid[ix])`.
        source_2d: Vec<Vec<f64>>,
    },

    /// Custom heating function
    Custom(Box<dyn Fn(f64, &Cosmology) -> f64 + Send + Sync>),
}

/// Interpolate a value from a table sorted ascending in z, using linear
/// interpolation in log(z). Returns 0 outside the table bounds.
fn interp_log_z(z: f64, z_table: &[f64], val_table: &[f64]) -> f64 {
    if z_table.is_empty() || z < z_table[0] || z > z_table[z_table.len() - 1] {
        return 0.0;
    }
    let log_z = z.ln();
    // Binary search on log(z_table)
    let idx = match z_table.binary_search_by(|probe| probe.total_cmp(&z)) {
        Ok(i) => return val_table[i],
        Err(i) => {
            if i == 0 {
                return val_table[0];
            }
            if i >= z_table.len() {
                return val_table[z_table.len() - 1];
            }
            i - 1
        }
    };
    let log_z_lo = z_table[idx].ln();
    let log_z_hi = z_table[idx + 1].ln();
    let t = (log_z - log_z_lo) / (log_z_hi - log_z_lo);
    val_table[idx] + t * (val_table[idx + 1] - val_table[idx])
}

/// Bilinear interpolation on a 2D table (z ascending, x ascending).
/// Returns 0 outside bounds.
fn interp_2d(z: f64, x: f64, z_table: &[f64], x_grid: &[f64], source_2d: &[Vec<f64>]) -> f64 {
    if z_table.is_empty() || x_grid.is_empty() {
        return 0.0;
    }
    if z < z_table[0] || z > z_table[z_table.len() - 1] {
        return 0.0;
    }
    if x < x_grid[0] || x > x_grid[x_grid.len() - 1] {
        return 0.0;
    }

    // Find z index
    let iz = match z_table.binary_search_by(|p| p.total_cmp(&z)) {
        Ok(i) => i.min(z_table.len() - 2),
        Err(i) => {
            if i == 0 {
                0
            } else {
                (i - 1).min(z_table.len() - 2)
            }
        }
    };
    let tz = if iz + 1 < z_table.len() && z_table[iz + 1] > z_table[iz] {
        (z - z_table[iz]) / (z_table[iz + 1] - z_table[iz])
    } else {
        0.0
    };

    // Find x index
    let ix = match x_grid.binary_search_by(|p| p.total_cmp(&x)) {
        Ok(i) => i.min(x_grid.len() - 2),
        Err(i) => {
            if i == 0 {
                0
            } else {
                (i - 1).min(x_grid.len() - 2)
            }
        }
    };
    let tx = if ix + 1 < x_grid.len() && x_grid[ix + 1] > x_grid[ix] {
        (x - x_grid[ix]) / (x_grid[ix + 1] - x_grid[ix])
    } else {
        0.0
    };

    let f00 = source_2d[iz][ix];
    let f01 = source_2d[iz].get(ix + 1).copied().unwrap_or(f00);
    let f10 = source_2d.get(iz + 1).map(|r| r[ix]).unwrap_or(f00);
    let f11 = source_2d
        .get(iz + 1)
        .and_then(|r| r.get(ix + 1))
        .copied()
        .unwrap_or(f00);

    (1.0 - tz) * ((1.0 - tx) * f00 + tx * f01) + tz * ((1.0 - tx) * f10 + tx * f11)
}

/// Load a tabulated heating rate from a CSV file.
///
/// File format: one header line (`z,dq_dz`), then data rows.
/// The z column must be positive. Data is sorted ascending by z.
///
/// Returns `TabulatedHeating` variant.
pub fn load_heating_table(path: &str) -> Result<InjectionScenario, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read heating table '{path}': {e}"))?;

    let mut z_table = Vec::new();
    let mut rate_table = Vec::new();
    let mut seen_data = false;

    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Skip header line (first non-comment, non-empty line that looks like a header)
        if !seen_data
            && line.contains("z")
            && !line.chars().next().map_or(true, |c| c.is_ascii_digit())
        {
            continue;
        }
        seen_data = true;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            return Err(format!(
                "Heating table line {}: expected 'z,dq_dz', got '{line}'",
                i + 1
            ));
        }
        let z: f64 = parts[0]
            .trim()
            .parse()
            .map_err(|_| format!("Heating table line {}: invalid z '{}'", i + 1, parts[0]))?;
        let rate: f64 = parts[1]
            .trim()
            .parse()
            .map_err(|_| format!("Heating table line {}: invalid rate '{}'", i + 1, parts[1]))?;
        z_table.push(z);
        rate_table.push(rate);
    }

    // Sort by z ascending
    let mut indices: Vec<usize> = (0..z_table.len()).collect();
    indices.sort_by(|&a, &b| z_table[a].total_cmp(&z_table[b]));
    let z_sorted: Vec<f64> = indices.iter().map(|&i| z_table[i]).collect();
    let rate_sorted: Vec<f64> = indices.iter().map(|&i| rate_table[i]).collect();

    Ok(InjectionScenario::TabulatedHeating {
        z_table: z_sorted,
        rate_table: rate_sorted,
    })
}

/// Load a tabulated photon source from a CSV file.
///
/// File format:
///   Header: `z,x1,x2,...,xN`
///   Each row: `z_i,source(x1,z_i),source(x2,z_i),...,source(xN,z_i)`
///
/// Returns `TabulatedPhotonSource` variant.
pub fn load_photon_source_table(path: &str) -> Result<InjectionScenario, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read photon source table '{path}': {e}"))?;

    let mut lines = content
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'));

    // Parse header to get x grid
    let header = lines
        .next()
        .ok_or_else(|| "Photon source table is empty".to_string())?;
    let header_parts: Vec<&str> = header.split(',').collect();
    if header_parts.len() < 2 {
        return Err("Photon source table header must have at least z and one x value".to_string());
    }
    let mut x_grid = Vec::with_capacity(header_parts.len() - 1);
    for s in &header_parts[1..] {
        let val: f64 = s
            .trim()
            .parse()
            .map_err(|_| format!("Invalid x value in header: '{s}'"))?;
        x_grid.push(val);
    }
    let n_x = x_grid.len();

    let mut z_table = Vec::new();
    let mut source_2d = Vec::new();

    for (i, line) in lines.enumerate() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 1 + n_x {
            return Err(format!(
                "Photon source table row {}: expected {} columns, got {}",
                i + 2,
                1 + n_x,
                parts.len()
            ));
        }
        let z: f64 = parts[0].trim().parse().map_err(|_| {
            format!(
                "Photon source table row {}: invalid z '{}'",
                i + 2,
                parts[0]
            )
        })?;
        let mut row = Vec::with_capacity(n_x);
        for s in &parts[1..1 + n_x] {
            let val: f64 = s
                .trim()
                .parse()
                .map_err(|_| format!("Photon source table row {}: invalid value '{s}'", i + 2))?;
            row.push(val);
        }
        z_table.push(z);
        source_2d.push(row);
    }

    // Sort by z ascending
    let mut indices: Vec<usize> = (0..z_table.len()).collect();
    indices.sort_by(|&a, &b| z_table[a].total_cmp(&z_table[b]));
    let z_sorted: Vec<f64> = indices.iter().map(|&i| z_table[i]).collect();
    let source_sorted: Vec<Vec<f64>> = indices.iter().map(|&i| source_2d[i].clone()).collect();

    Ok(InjectionScenario::TabulatedPhotonSource {
        z_table: z_sorted,
        x_grid,
        source_2d: source_sorted,
    })
}

impl InjectionScenario {
    /// CLI-friendly name for this injection scenario.
    pub fn name(&self) -> &str {
        match self {
            InjectionScenario::SingleBurst { .. } => "single-burst",
            InjectionScenario::DecayingParticle { .. } => "decaying-particle",
            InjectionScenario::AnnihilatingDM { .. } => "annihilating-dm",
            InjectionScenario::AnnihilatingDMPWave { .. } => "annihilating-dm-pwave",
            InjectionScenario::MonochromaticPhotonInjection { .. } => "monochromatic-photon",
            InjectionScenario::DecayingParticlePhoton { .. } => "decaying-particle-photon",
            InjectionScenario::DarkPhotonResonance { .. } => "dark-photon-resonance",
            InjectionScenario::TabulatedHeating { .. } => "tabulated-heating",
            InjectionScenario::TabulatedPhotonSource { .. } => "tabulated-photon",

            InjectionScenario::Custom(_) => "custom",
        }
    }

    /// Validate parameters, returning an error message if invalid.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            InjectionScenario::SingleBurst {
                z_h,
                sigma_z,
                delta_rho_over_rho,
            } => {
                if !z_h.is_finite() || *z_h <= 0.0 {
                    return Err(format!("z_h must be positive and finite, got {z_h}"));
                }
                if !sigma_z.is_finite() || *sigma_z <= 0.0 {
                    return Err(format!(
                        "sigma_z must be positive and finite, got {sigma_z}"
                    ));
                }
                if *sigma_z > 0.3 * *z_h {
                    return Err(format!(
                        "sigma_z must be ≲ 0.3 × z_h for the narrow-Gaussian approximation \
                         (got sigma_z={sigma_z}, z_h={z_h}, ratio={:.3}). Wide injections \
                         should use a continuous scenario instead.",
                        sigma_z / z_h
                    ));
                }
                if !delta_rho_over_rho.is_finite() {
                    return Err(format!(
                        "delta_rho_over_rho must be finite, got {delta_rho_over_rho}"
                    ));
                }
                Ok(())
            }
            InjectionScenario::DecayingParticle { f_x, gamma_x } => {
                if !f_x.is_finite() || *f_x <= 0.0 {
                    return Err(format!("f_x must be positive and finite, got {f_x}"));
                }
                if !gamma_x.is_finite() || *gamma_x <= 0.0 {
                    return Err(format!(
                        "gamma_x must be positive and finite, got {gamma_x}"
                    ));
                }
                Ok(())
            }
            InjectionScenario::AnnihilatingDM { f_ann }
            | InjectionScenario::AnnihilatingDMPWave { f_ann } => {
                if !f_ann.is_finite() || *f_ann <= 0.0 {
                    return Err(format!("f_ann must be positive and finite, got {f_ann}"));
                }
                Ok(())
            }
            InjectionScenario::MonochromaticPhotonInjection {
                x_inj,
                delta_n_over_n,
                z_h,
                sigma_z,
                sigma_x,
            } => {
                if !x_inj.is_finite() || *x_inj <= 0.0 {
                    return Err(format!("x_inj must be positive and finite, got {x_inj}"));
                }
                if !delta_n_over_n.is_finite() {
                    return Err(format!(
                        "delta_n_over_n must be finite, got {delta_n_over_n}"
                    ));
                }
                if !z_h.is_finite() || *z_h <= 0.0 {
                    return Err(format!("z_h must be positive and finite, got {z_h}"));
                }
                if !sigma_z.is_finite() || *sigma_z <= 0.0 {
                    return Err(format!(
                        "sigma_z must be positive and finite, got {sigma_z}"
                    ));
                }
                if *sigma_z > 0.3 * *z_h {
                    return Err(format!(
                        "sigma_z must be ≲ 0.3 × z_h for the narrow-Gaussian approximation \
                         (got sigma_z={sigma_z}, z_h={z_h}, ratio={:.3})",
                        sigma_z / z_h
                    ));
                }
                if !sigma_x.is_finite() || *sigma_x <= 0.0 {
                    return Err(format!(
                        "sigma_x must be positive and finite, got {sigma_x}"
                    ));
                }
                // The Gaussian-in-x source normalisation ∫ x² · profile(x) dx
                // = G₂ uses the factor 1/x² evaluated at the local x, not x_inj.
                // This is accurate when σ_x ≪ x_inj so that x ≈ x_inj over the
                // Gaussian's support. Above σ_x/x_inj ~ 0.3 the normalisation
                // drifts and the effective ΔN/N deviates from the requested
                // value — reject rather than silently inject the wrong number
                // of photons.
                if *sigma_x > 0.3 * *x_inj {
                    return Err(format!(
                        "sigma_x must be ≲ 0.3 × x_inj for the Gaussian ΔN/N \
                         normalisation to be accurate (got sigma_x={sigma_x}, \
                         x_inj={x_inj}, ratio={:.3})",
                        sigma_x / x_inj
                    ));
                }
                Ok(())
            }
            InjectionScenario::DecayingParticlePhoton {
                x_inj_0,
                f_inj,
                gamma_x,
                ..
            } => {
                if !x_inj_0.is_finite() || *x_inj_0 <= 0.0 {
                    return Err(format!(
                        "x_inj_0 must be positive and finite, got {x_inj_0}"
                    ));
                }
                if !f_inj.is_finite() || *f_inj <= 0.0 {
                    return Err(format!("f_inj must be positive and finite, got {f_inj}"));
                }
                if !gamma_x.is_finite() || *gamma_x <= 0.0 {
                    return Err(format!(
                        "gamma_x must be positive and finite, got {gamma_x}"
                    ));
                }
                Ok(())
            }
            InjectionScenario::DarkPhotonResonance { epsilon, m_ev } => {
                if !epsilon.is_finite() || *epsilon <= 0.0 {
                    return Err(format!(
                        "epsilon must be positive and finite, got {epsilon}"
                    ));
                }
                if !m_ev.is_finite() || *m_ev <= 0.0 {
                    return Err(format!("m_ev must be positive and finite, got {m_ev}"));
                }
                Ok(())
            }
            InjectionScenario::TabulatedHeating {
                z_table,
                rate_table,
            } => {
                if z_table.is_empty() {
                    return Err("z_table is empty".into());
                }
                if z_table.len() != rate_table.len() {
                    return Err(format!(
                        "z_table and rate_table must have same length, got {} and {}",
                        z_table.len(),
                        rate_table.len()
                    ));
                }
                // Finiteness first so the monotonicity check below can rely on
                // a total order (NaN slips past `<=` silently).
                for (i, &z) in z_table.iter().enumerate() {
                    if !z.is_finite() {
                        return Err(format!("z_table[{i}]={z} is not finite"));
                    }
                }
                for (i, &r) in rate_table.iter().enumerate() {
                    if !r.is_finite() {
                        return Err(format!("rate_table[{i}]={r} is not finite"));
                    }
                }
                if z_table[0] <= 0.0 {
                    return Err(format!(
                        "z_table[0]={} must be positive (interpolation uses log z)",
                        z_table[0]
                    ));
                }
                // interp_log_z uses binary search in log(z) assuming strict
                // ascending order. A non-monotone table silently produces
                // garbage rates with no error — CLAUDE.md calls this out as
                // a silent-failure mode. Reject at validate() time.
                for i in 1..z_table.len() {
                    if z_table[i] <= z_table[i - 1] {
                        return Err(format!(
                            "z_table must be strictly ascending: z[{}]={} not > z[{}]={}",
                            i,
                            z_table[i],
                            i - 1,
                            z_table[i - 1]
                        ));
                    }
                }
                Ok(())
            }
            InjectionScenario::TabulatedPhotonSource {
                z_table,
                x_grid,
                source_2d,
            } => {
                if z_table.is_empty() {
                    return Err("z_table is empty".into());
                }
                if x_grid.is_empty() {
                    return Err("x_grid is empty".into());
                }
                if z_table.len() != source_2d.len() {
                    return Err(format!(
                        "z_table and source_2d must have same length, got {} and {}",
                        z_table.len(),
                        source_2d.len()
                    ));
                }
                for (i, &z) in z_table.iter().enumerate() {
                    if !z.is_finite() {
                        return Err(format!("z_table[{i}]={z} is not finite"));
                    }
                }
                for (i, &x) in x_grid.iter().enumerate() {
                    if !x.is_finite() {
                        return Err(format!("x_grid[{i}]={x} is not finite"));
                    }
                }
                if z_table[0] <= 0.0 {
                    return Err(format!("z_table[0]={} must be positive", z_table[0]));
                }
                if x_grid[0] <= 0.0 {
                    return Err(format!("x_grid[0]={} must be positive", x_grid[0]));
                }
                for i in 1..z_table.len() {
                    if z_table[i] <= z_table[i - 1] {
                        return Err(format!(
                            "z_table must be strictly ascending: z[{}]={} not > z[{}]={}",
                            i,
                            z_table[i],
                            i - 1,
                            z_table[i - 1]
                        ));
                    }
                }
                for i in 1..x_grid.len() {
                    if x_grid[i] <= x_grid[i - 1] {
                        return Err(format!(
                            "x_grid must be strictly ascending: x[{}]={} not > x[{}]={}",
                            i,
                            x_grid[i],
                            i - 1,
                            x_grid[i - 1]
                        ));
                    }
                }
                for (iz, row) in source_2d.iter().enumerate() {
                    if row.len() != x_grid.len() {
                        return Err(format!(
                            "source_2d[{iz}] has length {}, expected {} (len(x_grid))",
                            row.len(),
                            x_grid.len()
                        ));
                    }
                    for (ix, &v) in row.iter().enumerate() {
                        if !v.is_finite() {
                            return Err(format!("source_2d[{iz}][{ix}]={v} is not finite"));
                        }
                    }
                }
                Ok(())
            }
            InjectionScenario::Custom(f) => {
                // Spot-check the closure for finiteness across the full
                // redshift band the solver typically integrates over. NaN
                // or Inf values from a buggy user closure would otherwise
                // propagate into the solver and trip a deep-stack panic
                // far from the actual culprit.
                //
                // 16 log-spaced samples cover ~6 decades at < ½ dec/sample —
                // enough to catch isolated singularities (e.g. 1/(z - z*))
                // that 5-point spot-checks miss.
                let cosmo = Cosmology::default();
                let z_lo = 1.0e2_f64.ln();
                let z_hi = 5.0e6_f64.ln();
                for i in 0..16 {
                    let t = i as f64 / 15.0;
                    let z = (z_lo + t * (z_hi - z_lo)).exp();
                    let val = f(z, &cosmo);
                    if !val.is_finite() {
                        return Err(format!(
                            "Custom heating closure returned non-finite value {val} at z={z:.3e}; \
                             reject closures that emit NaN/Inf in the integration band."
                        ));
                    }
                }
                Ok(())
            }
        }
    }

    /// Compute the heating rate d(Δρ_γ/ρ_γ)/dt at redshift z.
    ///
    /// Returns the rate in units of [1/s].
    pub fn heating_rate(&self, z: f64, cosmo: &Cosmology) -> f64 {
        match self {
            InjectionScenario::SingleBurst {
                z_h,
                delta_rho_over_rho,
                sigma_z,
            } => {
                // Narrow Gaussian in redshift
                // d(Δρ/ρ)/dz = (Δρ/ρ) × Gaussian(z - z_h, σ_z)
                // d(Δρ/ρ)/dt = d(Δρ/ρ)/dz × dz/dt = d(Δρ/ρ)/dz × (-H(1+z))
                let gauss = (-(z - z_h).powi(2) / (2.0 * sigma_z * sigma_z)).exp()
                    / (2.0 * std::f64::consts::PI * sigma_z * sigma_z).sqrt();
                delta_rho_over_rho * gauss * cosmo.hubble(z) * (1.0 + z)
            }

            InjectionScenario::DecayingParticle { f_x, gamma_x } => {
                // dE/dt = f*_X × Γ_X × N_H × exp(-Γ_X t)
                // d(Δρ/ρ)/dt = dE/dt / ρ_γ
                let t = cosmo.cosmic_time(z);
                let n_h = cosmo.n_h(z);
                let rho_gamma = cosmo.rho_gamma(z);

                // f_x in eV → convert to Joules
                let f_x_joules = f_x * EV_IN_JOULES;
                f_x_joules * gamma_x * n_h * (-gamma_x * t).exp() / rho_gamma
            }

            InjectionScenario::AnnihilatingDM { f_ann } => {
                // s-wave DM annihilation. Paper convention: f_ann [eV/s].
                // dE/(dt dV) = f_ann × n_H(z) × (1+z)³
                // d(Δρ/ρ)/dt = f_ann × n_H(z) × (1+z)³ / ρ_γ(z)
                //            ∝ (1+z)⁶ / (1+z)⁴ = (1+z)²
                let n_h = cosmo.n_h(z);
                let rho_gamma = cosmo.rho_gamma(z);
                let f_ann_si = f_ann * EV_IN_JOULES;
                f_ann_si * n_h * (1.0 + z).powi(3) / rho_gamma
            }

            InjectionScenario::AnnihilatingDMPWave { f_ann } => {
                // p-wave DM annihilation: ⟨σv⟩ ∝ v² ∝ T ∝ (1+z), adding one (1+z).
                // dE/(dt dV) = f_ann × n_H(z) × (1+z)⁴
                // d(Δρ/ρ)/dt = f_ann × n_H(z) × (1+z)⁴ / ρ_γ(z)
                //            ∝ (1+z)⁷ / (1+z)⁴ = (1+z)³
                let n_h = cosmo.n_h(z);
                let rho_gamma = cosmo.rho_gamma(z);
                let f_ann_si = f_ann * EV_IN_JOULES;
                f_ann_si * n_h * (1.0 + z).powi(4) / rho_gamma
            }

            InjectionScenario::MonochromaticPhotonInjection { .. } => {
                // Photon injection has no direct electron heating.
                // All photons are injected via photon_source_rate() into Δn,
                // which adds energy directly.
                0.0
            }

            InjectionScenario::DecayingParticlePhoton { .. } => {
                // All injected energy goes through photon_source_rate() and is
                // evolved self-consistently by the PDE solver (Compton + DC/BR).
                0.0
            }

            InjectionScenario::DarkPhotonResonance { .. } => {
                // Dark photon resonance is applied as an initial condition at z_res
                // via `initial_delta_n`, not as a bulk heating rate.
                0.0
            }

            InjectionScenario::TabulatedHeating {
                z_table,
                rate_table,
            } => {
                // rate_table stores d(Δρ/ρ)/dz (positive = heating).
                // Convert to d(Δρ/ρ)/dt = d(Δρ/ρ)/dz × |dz/dt| = d(Δρ/ρ)/dz × H(z)(1+z)
                let dq_dz = interp_log_z(z, z_table, rate_table);
                dq_dz * cosmo.hubble(z) * (1.0 + z)
            }

            InjectionScenario::TabulatedPhotonSource { .. } => {
                // All energy goes through the photon channel
                0.0
            }

            InjectionScenario::Custom(f) => f(z, cosmo),
        }
    }

    /// Frequency-dependent photon injection/removal rate.
    ///
    /// Returns d(Δn)/dt at frequency x, in units of [1/s].
    /// This is the direct modification to the photon occupation number at
    /// each frequency, separate from the bulk heating captured by `heating_rate`.
    ///
    /// For `MonochromaticPhotonInjection`, returns a Gaussian profile in
    /// both x and z centered at (x_inj, z_h).
    ///
    /// Returns 0.0 for scenarios without frequency-dependent photon injection.
    pub fn photon_source_rate(&self, x: f64, z: f64, cosmo: &Cosmology) -> f64 {
        match self {
            InjectionScenario::MonochromaticPhotonInjection {
                x_inj,
                delta_n_over_n,
                z_h,
                sigma_z,
                sigma_x,
            } => {
                // Inject ALL photons — no P_s filtering.
                // DC/BR absorption is handled dynamically by the PDE solver.
                // Energy from absorbed photons is recovered via the solver's
                // source-only Kompaneets correction.

                // Temporal profile: Gaussian in z
                let gauss_z = (-(z - z_h).powi(2) / (2.0 * sigma_z * sigma_z)).exp()
                    / (2.0 * std::f64::consts::PI * sigma_z * sigma_z).sqrt();

                // Frequency profile: Gaussian in x, normalized so that
                // ∫ x² × profile(x) dx = G₂ (preserving ΔN/N normalization)
                let gauss_x = (-(x - x_inj).powi(2) / (2.0 * sigma_x * sigma_x)).exp()
                    / (sigma_x * (2.0 * std::f64::consts::PI).sqrt());

                // d(Δn)/dt = delta_n_over_n × G₂ / x² × gauss_x × gauss_z × H(1+z)
                let rate_per_z =
                    delta_n_over_n * G2_PLANCK / (x * x).max(1e-30) * gauss_x * gauss_z;
                rate_per_z * cosmo.hubble(z) * (1.0 + z)
            }
            InjectionScenario::DecayingParticlePhoton {
                x_inj_0,
                f_inj,
                gamma_x,
            } => {
                let x_inj = x_inj_0 / (1.0 + z);
                let sigma_x = 0.05 * x_inj;

                // Quick exit for frequencies far from injection
                if sigma_x < 1e-30 || (x - x_inj).abs() > 5.0 * sigma_x {
                    return 0.0;
                }

                // Gaussian frequency profile
                let gauss_x = (-(x - x_inj).powi(2) / (2.0 * sigma_x * sigma_x)).exp()
                    / (sigma_x * (2.0 * std::f64::consts::PI).sqrt());

                let survival = vacuum_survival(z, *gamma_x, cosmo);

                // dn/dt = G₂ × f_inj × Γ_X × S(z) × G(x, x_inj, σ_x) / x²
                //
                // This is the **spontaneous** emission rate. Because X → γγ
                // puts **both** final-state photons into the same mode at
                // x_inj, the full photon source carries a Bose-enhancement
                // factor (1 + n_pl(x_inj))² (one (1+n) per emitted photon,
                // squared for two photons in the same state). It is omitted
                // here, matching Chluba (2015), and is validated for
                // x_inj ≳ few where n_pl(x_inj) ≪ 1 (sub-percent correction).
                //
                // WARNING: for x_inj ≪ 1 the Bose factor grows as 1/x_inj²
                // and dominates — e.g. at x_inj = 0.05, (1 + n_pl)² ≈ 440.
                // Users injecting at very low frequencies must regard the
                // resulting μ/y amplitudes as **lower bounds**. The
                // redistribution dynamics (DC/BR absorption) still govern
                // the observable shape, but the amplitude is systematically
                // underestimated.
                G2_PLANCK * f_inj * gamma_x * survival * gauss_x / (x * x).max(1e-30)
            }

            InjectionScenario::TabulatedPhotonSource {
                z_table,
                x_grid,
                source_2d,
            } => {
                // Table stores d(Δn)/dz; convert to d(Δn)/dt [1/s] like TabulatedHeating.
                let dn_dz = interp_2d(z, x, z_table, x_grid, source_2d);
                dn_dz * cosmo.hubble(z) * (1.0 + z)
            }

            _ => 0.0,
        }
    }

    /// Whether this scenario has frequency-dependent photon injection/depletion.
    pub fn has_photon_source(&self) -> bool {
        matches!(
            self,
            InjectionScenario::MonochromaticPhotonInjection { .. }
                | InjectionScenario::DecayingParticlePhoton { .. }
                | InjectionScenario::TabulatedPhotonSource { .. }
        )
    }

    /// Return refinement zones for adaptive grid resolution near injection features.
    ///
    /// Photon injection scenarios need extra grid points near the injection
    /// frequency to resolve the narrow Gaussian source profile and the
    /// DC/BR absorption/re-emission dynamics at low x.
    pub fn refinement_zones(&self) -> Vec<RefinementZone> {
        match self {
            InjectionScenario::MonochromaticPhotonInjection { x_inj, sigma_x, .. } => {
                // Refine around x_inj ± 5σ_x with 300 log-spaced points
                let half_width = 5.0 * sigma_x;
                vec![RefinementZone {
                    x_center: *x_inj,
                    x_width: half_width,
                    n_points: 300,
                }]
            }
            InjectionScenario::DecayingParticlePhoton { x_inj_0, .. } => {
                // The injection frequency redshifts: x_inj(z) = x_inj_0 / (1+z).
                // At high z (e.g., z=1e6), x_inj can be ~1e-7 × x_inj_0.
                // RefinementZone uses a linear (center, half-width) API, so
                // we explicitly cover [x_lo, x_hi] with x_center the
                // arithmetic mean and x_width the half-range. Downstream
                // grid construction clamps to [x_min, x_max].
                let x_lo = (x_inj_0 * 1e-7).max(1e-8);
                let x_hi = *x_inj_0;
                let x_center = 0.5 * (x_lo + x_hi);
                let x_width = 0.5 * (x_hi - x_lo);
                vec![RefinementZone {
                    x_center,
                    x_width,
                    n_points: 400,
                }]
            }
            _ => Vec::new(),
        }
    }

    /// Return the characteristic injection redshift(s) for this scenario.
    ///
    /// For burst-like scenarios, returns `Some((z_center, z_upper))` where
    /// `z_upper` is the highest redshift at which injection is active
    /// (typically z_h + 5σ_z for Gaussians). `z_center` is the peak.
    ///
    /// For continuous scenarios (decaying particles, annihilation), returns
    /// `None` — injection happens at all z.
    pub fn characteristic_redshift(&self) -> Option<(f64, f64)> {
        match self {
            InjectionScenario::SingleBurst { z_h, sigma_z, .. } => {
                Some((*z_h, z_h + 7.0 * sigma_z))
            }
            InjectionScenario::MonochromaticPhotonInjection { z_h, sigma_z, .. } => {
                Some((*z_h, z_h + 7.0 * sigma_z))
            }
            // Continuous injection: active at all redshifts
            _ => None,
        }
    }

    /// Dark-photon NWA parameters (γ_con, z_res), if applicable.
    ///
    /// Returns `Some((γ_con, z_res))` for `DarkPhotonResonance`,
    /// `None` for all other scenarios. Returns `None` if the resonance
    /// falls outside the supported redshift range.
    pub fn dark_photon_params(&self, cosmo: &Cosmology) -> Option<(f64, f64)> {
        match self {
            InjectionScenario::DarkPhotonResonance { epsilon, m_ev } => {
                crate::dark_photon::gamma_con(*epsilon, *m_ev, cosmo)
            }
            _ => None,
        }
    }

    /// Initial-condition perturbation Δn(x) to be installed at `z_start`.
    ///
    /// Scenarios that deposit their distortion as an impulsive event (notably
    /// `DarkPhotonResonance`) return `Some(Δn_init)` here; the solver applies
    /// it at `z_start` before beginning redshift evolution. All other
    /// scenarios return `None` and are evolved from Δn = 0.
    pub fn initial_delta_n(&self, x_grid: &[f64], cosmo: &Cosmology) -> Option<Vec<f64>> {
        match self {
            InjectionScenario::DarkPhotonResonance { .. } => {
                let (gamma_con, _z_res) = self.dark_photon_params(cosmo)?;
                let dn: Vec<f64> = x_grid
                    .iter()
                    .map(|&x| {
                        let p = 1.0 - (-gamma_con / x).exp();
                        -p * planck(x)
                    })
                    .collect();
                Some(dn)
            }
            _ => None,
        }
    }

    /// Suggest a lower `x_min` for the frequency grid when needed.
    ///
    /// Low-frequency photon injection can be artificially absorbed by the
    /// Dirichlet boundary at `x_min` if the source support extends below the
    /// existing grid floor.
    pub fn suggested_x_min(&self) -> Option<f64> {
        match self {
            InjectionScenario::MonochromaticPhotonInjection { x_inj, .. } => {
                // Need x_min well below x_inj to prevent boundary absorption.
                // Use x_inj / 100, but never below 1e-7 (numerical floor).
                let suggested = (x_inj / 100.0).max(1e-7);
                if suggested < 1e-4 {
                    Some(suggested)
                } else {
                    None // default x_min = 1e-4 is fine
                }
            }
            InjectionScenario::DecayingParticlePhoton { x_inj_0, .. } => {
                let suggested = (x_inj_0 / 100.0).max(1e-7);
                if suggested < 1e-4 {
                    Some(suggested)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check for strong distortion regime and return warnings.
    ///
    /// The Kompaneets equation solver uses a linearized perturbation approach
    /// (Δn = n - n_pl) that breaks down for |Δρ/ρ| > ~0.01. Returns a
    /// list of warning messages for scenarios that may exceed this limit.
    pub fn warn_strong_distortion(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        match self {
            InjectionScenario::SingleBurst {
                delta_rho_over_rho, ..
            } if delta_rho_over_rho.abs() > 0.01 => {
                warnings.push(format!(
                    "Strong distortion regime: |Δρ/ρ| = {:.2e} > 0.01. \
                     The solver uses linearized Kompaneets and results will be \
                     inaccurate for large energy injections. \
                     Physical distortions are small (Δρ/ρ ~ 10⁻⁵ for μ-era).",
                    delta_rho_over_rho.abs()
                ));
            }
            InjectionScenario::MonochromaticPhotonInjection {
                delta_n_over_n,
                x_inj,
                ..
            } => {
                if delta_n_over_n.abs() > 0.01 {
                    warnings.push(format!(
                        "Strong distortion regime: |ΔN/N| = {:.2e} > 0.01. \
                         The solver uses linearized Kompaneets and results will be \
                         inaccurate for large photon injections.",
                        delta_n_over_n.abs()
                    ));
                }
                if *x_inj > 150.0 {
                    warnings.push(format!(
                        "High-frequency photon injection: x_inj = {:.1} exceeds the \
                         validated range (x_inj ≤ 150). PDE solver is stable but results \
                         are not validated against CosmoTherm or literature at this frequency.",
                        x_inj
                    ));
                }
            }
            _ => {}
        }
        warnings
    }

    /// Warn when the dark-photon NWA resonance falls outside the validated
    /// redshift range (roughly z ∈ [50, 3×10⁶] per CLAUDE.md).
    ///
    /// Returns `Some(Err(msg))` when no resonance exists at all in the
    /// supported band (hard error), `Some(Ok(warnings))` with a regime warning
    /// when z_res lands outside the validated window, and `None` for
    /// non-dark-photon scenarios or when everything is fine.
    pub fn warn_dark_photon_range(&self, cosmo: &Cosmology) -> Vec<String> {
        let mut warnings = Vec::new();
        if let InjectionScenario::DarkPhotonResonance { m_ev, .. } = self {
            match self.dark_photon_params(cosmo) {
                None => {
                    warnings.push(format!(
                        "DarkPhotonResonance: no plasma-frequency resonance for m_ev={m_ev:.3e} \
                         in the searched band z ∈ [10, 3×10⁷]. The depletion IC will be zero \
                         and the run produces no distortion from this channel."
                    ));
                }
                Some((_g, z_res)) => {
                    if z_res < 50.0 {
                        warnings.push(format!(
                            "DarkPhotonResonance: z_res={z_res:.3e} is below the validated \
                             range (z ≳ 50). Recombination history and Compton coupling \
                             are not trusted at such low z; treat results as indicative."
                        ));
                    } else if z_res > 3.0e6 {
                        warnings.push(format!(
                            "DarkPhotonResonance: z_res={z_res:.3e} is above the validated \
                             range (z ≲ 3×10⁶). Kompaneets Fokker-Planck has O(θ_e²) \
                             corrections that grow above this; results may have percent-level \
                             systematic errors."
                        ));
                    }
                }
            }
        }
        warnings
    }

    /// Warn when a tabulated-source table doesn't cover the solver's
    /// integration range `[z_end, z_start]`.
    ///
    /// `interp_log_z` / `interp_2d` return 0.0 outside the table — a silent
    /// extrapolation that produces zero injection at redshifts the user may
    /// have intended to cover. Surface the mismatch as a warning at build
    /// time so it can't go unnoticed.
    pub fn warn_tabulated_coverage(&self, z_start: f64, z_end: f64) -> Vec<String> {
        let mut warnings = Vec::new();
        let (z_table, name) = match self {
            InjectionScenario::TabulatedHeating { z_table, .. } => (z_table, "TabulatedHeating"),
            InjectionScenario::TabulatedPhotonSource { z_table, .. } => {
                (z_table, "TabulatedPhotonSource")
            }
            _ => return warnings,
        };
        if z_table.is_empty() {
            return warnings;
        }
        let z_min = z_table[0];
        let z_max = z_table[z_table.len() - 1];
        // 10% headroom on each side to avoid warning for benign edge cases
        if z_end < z_min * 0.9 {
            warnings.push(format!(
                "{name}: table covers z ∈ [{z_min:.3e}, {z_max:.3e}] but solver runs down \
                 to z_end={z_end:.3e}. Rates are silently zero below z_table.min(); \
                 the run will miss any physical injection below that bound."
            ));
        }
        if z_start > z_max * 1.1 {
            warnings.push(format!(
                "{name}: table covers z ∈ [{z_min:.3e}, {z_max:.3e}] but solver starts at \
                 z_start={z_start:.3e}. Rates are silently zero above z_table.max(); \
                 early evolution sees no injection."
            ));
        }
        warnings
    }

    /// Warn if stimulated emission (Bose enhancement) is missing for photon decay.
    ///
    /// `DecayingParticlePhoton` uses the vacuum decay rate only. For the
    /// canonical X → γγ channel, the physical rate carries a factor
    /// (1 + n(x_inj))² (one (1+n) per emitted photon, both into the same
    /// mode at x_inj); for single-photon channels (e.g. X → γ X') the
    /// factor is (1 + n(x_inj)). The squared form is significant at
    /// x_inj ≪ 1 where n_pl ≈ 1/x_inj ≫ 1.
    pub fn warn_stimulated_emission(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if let InjectionScenario::DecayingParticlePhoton { x_inj_0, .. } = self {
            let n_pl = 1.0 / ((*x_inj_0).exp() - 1.0).max(1e-30);
            let factor_sq = (1.0 + n_pl).powi(2);
            warnings.push(format!(
                "DecayingParticlePhoton: stimulated emission (Bose enhancement) is NOT \
                 included. For X → γγ (both photons in the same mode at x_inj) the \
                 physical rate should include a factor (1 + n(x_inj))² which is \
                 significant at low x_inj. At x_inj_0 = {x_inj_0:.2e}, (1 + n)² at z=0 \
                 is ~{factor_sq:.0}. See Chluba (2015) for details."
            ));
        }
        warnings
    }

    /// Compute the physical d(Δρ/ρ)/dz.
    ///
    /// **Sign convention**: For positive energy injection (heating_rate > 0),
    /// this returns a NEGATIVE value because energy enters as z decreases
    /// (dt > 0, dz < 0). This is the correct physical sign.
    ///
    /// **WARNING**: The Green's function routines (`mu_from_heating`, etc.)
    /// expect a POSITIVE dq/dz for heating. Use `heating_rate_per_redshift().abs()`
    /// or pass a positive Gaussian directly when calling Green's function methods.
    pub fn heating_rate_per_redshift(&self, z: f64, cosmo: &Cosmology) -> f64 {
        // d(Δρ/ρ)/dz = d(Δρ/ρ)/dt × dt/dz = -d(Δρ/ρ)/dt / (H(1+z))
        let rate = self.heating_rate(z, cosmo);
        -rate / (cosmo.hubble(z) * (1.0 + z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_burst_normalization() {
        let cosmo = Cosmology::default();
        let z_h = 2.0e5;
        let drho = 1e-5;
        let sigma = 100.0;

        let scenario = InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: sigma,
        };

        // Integrate d(Δρ/ρ)/dz over z to check normalization
        let n_z = 10000;
        let z_min = z_h - 5.0 * sigma;
        let z_max = z_h + 5.0 * sigma;
        let dz = (z_max - z_min) / n_z as f64;

        let mut total = 0.0;
        for i in 0..n_z {
            let z = z_min + (i as f64 + 0.5) * dz;
            total += scenario.heating_rate_per_redshift(z, &cosmo).abs() * dz;
        }

        assert!(
            (total - drho).abs() / drho < 0.05,
            "Integrated energy = {total:.3e}, expected {drho:.3e}"
        );
    }

    #[test]
    fn test_interp_log_z_basic() {
        let z_table = vec![1e3, 1e4, 1e5, 1e6];
        let val_table = vec![1.0, 2.0, 3.0, 4.0];

        // At table points
        assert!((interp_log_z(1e3, &z_table, &val_table) - 1.0).abs() < 1e-10);
        assert!((interp_log_z(1e6, &z_table, &val_table) - 4.0).abs() < 1e-10);

        // Midpoint in log space between 1e3 and 1e4 is ~3162
        let mid = interp_log_z(3162.3, &z_table, &val_table);
        assert!((mid - 1.5).abs() < 0.01, "mid = {mid}, expected ~1.5");

        // Outside bounds should be 0
        assert_eq!(interp_log_z(500.0, &z_table, &val_table), 0.0);
        assert_eq!(interp_log_z(2e6, &z_table, &val_table), 0.0);
    }

    #[test]
    fn test_tabulated_heating_matches_burst() {
        // Create a tabulated heating rate that mimics a SingleBurst
        let cosmo = Cosmology::default();
        let z_h = 1e5_f64;
        let drho = 1e-5_f64;
        let sigma = (z_h * 0.04).max(100.0);

        let burst = InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z: sigma,
        };

        // Tabulate dq/dz on a dense grid
        let n = 2000;
        let z_min = (z_h - 7.0 * sigma).max(100.0);
        let z_max = z_h + 7.0 * sigma;
        let mut z_table = Vec::with_capacity(n);
        let mut rate_table = Vec::with_capacity(n);
        for i in 0..n {
            let z = z_min + (z_max - z_min) * i as f64 / (n - 1) as f64;
            let dq_dz = burst.heating_rate_per_redshift(z, &cosmo).abs();
            z_table.push(z);
            rate_table.push(dq_dz);
        }

        let tabulated = InjectionScenario::TabulatedHeating {
            z_table,
            rate_table,
        };

        // Compare heating_rate at several z values
        for &z in &[z_h - 2.0 * sigma, z_h, z_h + 2.0 * sigma] {
            let ref_rate = burst.heating_rate(z, &cosmo);
            let tab_rate = tabulated.heating_rate(z, &cosmo);
            if ref_rate.abs() > 1e-100 {
                let err = (tab_rate - ref_rate).abs() / ref_rate.abs();
                assert!(
                    err < 0.01,
                    "z={z:.2e}: err={err:.2e} (ref={ref_rate:.4e}, tab={tab_rate:.4e})"
                );
            }
        }
    }

    #[test]
    fn test_load_heating_table() {
        // Write a temp file and load it
        let dir = std::env::temp_dir();
        let path = dir.join("test_heating_table.csv");
        std::fs::write(&path, "z,dq_dz\n1e3,1.0e-10\n1e4,2.0e-10\n1e5,3.0e-10\n").unwrap();

        let scenario = load_heating_table(path.to_str().unwrap()).unwrap();
        match &scenario {
            InjectionScenario::TabulatedHeating {
                z_table,
                rate_table,
            } => {
                assert_eq!(z_table.len(), 3);
                assert_eq!(rate_table.len(), 3);
                assert!((z_table[0] - 1e3).abs() < 1.0);
            }
            _ => panic!("Expected TabulatedHeating"),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_heating_rate_all_scenarios() {
        let cosmo = Cosmology::default();
        let z = 5e4;

        // SingleBurst at z=z_h should have positive rate at the peak
        let z_h = 5e4;
        let drho = 1e-5;
        let sigma_z = 200.0;
        let burst = InjectionScenario::SingleBurst {
            z_h,
            delta_rho_over_rho: drho,
            sigma_z,
        };
        let rate = burst.heating_rate(z, &cosmo);
        assert!(
            rate > 0.0,
            "SingleBurst at peak should have positive rate: {rate:.4e}"
        );
        // At the peak z=z_h, analytical rate = drho * Gauss(0) * H(z_h) * (1+z_h)
        // Gauss(0) = 1/(sqrt(2π) σ_z) ≈ 1/(2.507 × 200) = 1.995e-3
        let gauss_peak = 1.0 / (2.0 * std::f64::consts::PI * sigma_z * sigma_z).sqrt();
        let expected_rate = drho * gauss_peak * cosmo.hubble(z_h) * (1.0 + z_h);
        assert!(
            (rate / expected_rate - 1.0).abs() < 1e-10,
            "SingleBurst peak rate = {rate:.6e}, expected = {expected_rate:.6e}"
        );

        // DecayingParticle should have positive rate at any z
        let decay = InjectionScenario::DecayingParticle {
            f_x: 1e-5,
            gamma_x: 1e-12,
        };
        let rate = decay.heating_rate(z, &cosmo);
        assert!(
            rate > 0.0,
            "DecayingParticle should have positive rate: {rate:.4e}"
        );

        // AnnihilatingDM should have positive rate
        let ann = InjectionScenario::AnnihilatingDM { f_ann: 2e-21 };
        let rate = ann.heating_rate(z, &cosmo);
        assert!(
            rate > 0.0,
            "AnnihilatingDM should have positive rate: {rate:.4e}"
        );

        // P-wave should also be positive
        let pwave = InjectionScenario::AnnihilatingDMPWave { f_ann: 2e-31 };
        let rate = pwave.heating_rate(z, &cosmo);
        assert!(
            rate > 0.0,
            "AnnihilatingDMPWave should have positive rate: {rate:.4e}"
        );

        // MonochromaticPhotonInjection: heating_rate should be 0 (energy is in photons)
        let mono = InjectionScenario::MonochromaticPhotonInjection {
            x_inj: 5.0,
            delta_n_over_n: 1e-5,
            sigma_x: 0.5,
            z_h: 3e5,
            sigma_z: 100.0,
        };
        let rate = mono.heating_rate(z, &cosmo);
        assert_eq!(
            rate, 0.0,
            "MonochromaticPhotonInjection heating_rate should be 0"
        );

        // All should be finite at all redshifts
        let scenarios: Vec<&InjectionScenario> = vec![&burst, &decay, &ann, &pwave, &mono];
        for scenario in &scenarios {
            for &zz in &[1e3, 1e4, 1e5, 1e6] {
                let r = scenario.heating_rate(zz, &cosmo);
                assert!(r.is_finite(), "{} at z={zz:.0e}: NaN/Inf", scenario.name());
                let r_z = scenario.heating_rate_per_redshift(zz, &cosmo);
                assert!(
                    r_z.is_finite(),
                    "{} per_z at z={zz:.0e}: NaN/Inf",
                    scenario.name()
                );
            }
        }
    }

    /// `photon_source_rate` for MonochromaticPhotonInjection must match its
    /// closed-form definition at the peak, not just be "> 1e-20".
    ///
    /// Oracle:             From Chluba 2015 / the scenario definition, at
    ///                     (x=x_inj, z=z_h) both Gaussians peak and the rate is
    ///                       r_peak = (ΔN/N) · G₂ / x_inj² ·
    ///                                [1/(σ_x √2π)] · [1/(σ_z √2π)] ·
    ///                                H(z_h) · (1+z_h)
    /// Expected:           r_peak computed from CODATA + Planck 2018 cosmology
    ///                     at z=3e5, x=5, σ_x=0.5, σ_z=100 → r_peak ≈ 3.7e6 s⁻¹
    /// Oracle uncertainty: 1e-14 relative (exact closed form, f64 roundoff only)
    /// Tolerance:          1e-10 relative
    ///
    /// Previous version asserted `rate > 1e-20` — 26 orders of magnitude
    /// below the physical value. A 12-OOM formula bug would have passed.
    #[test]
    fn test_photon_source_rate_nonzero() {
        let cosmo = Cosmology::default();
        let z_h = 3.0e5;
        let x_inj = 5.0;
        let delta_n_over_n = 1e-5;
        let sigma_x = 0.5;
        let sigma_z = 100.0;

        let mono = InjectionScenario::MonochromaticPhotonInjection {
            x_inj,
            delta_n_over_n,
            sigma_x,
            z_h,
            sigma_z,
        };
        assert!(mono.has_photon_source());

        // Peak rate: closed-form analytic value at (x=x_inj, z=z_h).
        let two_pi = std::f64::consts::TAU;
        let peak_gauss_x = 1.0 / (sigma_x * two_pi.sqrt());
        let peak_gauss_z = 1.0 / (sigma_z * two_pi.sqrt());
        let expected_peak = delta_n_over_n * G2_PLANCK / (x_inj * x_inj)
            * peak_gauss_x
            * peak_gauss_z
            * cosmo.hubble(z_h)
            * (1.0 + z_h);

        let rate = mono.photon_source_rate(x_inj, z_h, &cosmo);
        let rel_err = (rate - expected_peak).abs() / expected_peak;
        assert!(
            rel_err < 1e-10,
            "photon_source_rate at peak: got {rate:.6e} vs analytic {expected_peak:.6e}, \
             rel_err={rel_err:.2e} (tol 1e-10)",
        );

        // Off-peak suppression: at x = x_inj + 10σ_x the Gaussian factor is
        // exp(-50) ≈ 2e-22. Assert ratio below 1e-20 (room for σ_z factor too).
        let rate_far = mono.photon_source_rate(x_inj + 10.0 * sigma_x, z_h, &cosmo);
        assert!(
            rate_far / rate < 1e-20,
            "rate at x_inj + 10σ should be exp(-50)-suppressed: {rate_far:.2e} vs peak {rate:.2e}"
        );

        // Variant coverage: DecayingParticlePhoton has a source; SingleBurst doesn't.
        let dp = InjectionScenario::DecayingParticlePhoton {
            x_inj_0: 5e3,
            f_inj: 1e-5,
            gamma_x: 1e-12,
        };
        assert!(dp.has_photon_source());
        let burst = InjectionScenario::SingleBurst {
            z_h: 5e4,
            delta_rho_over_rho: 1e-5,
            sigma_z: 200.0,
        };
        assert!(!burst.has_photon_source());
    }

    #[test]
    fn test_characteristic_redshift_all_variants() {
        let burst = InjectionScenario::SingleBurst {
            z_h: 5e4,
            delta_rho_over_rho: 1e-5,
            sigma_z: 200.0,
        };
        let (z_h, z_start) = burst.characteristic_redshift().unwrap();
        assert!((z_h - 5e4).abs() < 1.0);
        assert!(z_start > z_h); // z_start = z_h + 7*sigma

        let mono = InjectionScenario::MonochromaticPhotonInjection {
            x_inj: 5.0,
            delta_n_over_n: 1e-5,
            sigma_x: 0.5,
            z_h: 3e5,
            sigma_z: 100.0,
        };
        let (z_h, _) = mono.characteristic_redshift().unwrap();
        assert!((z_h - 3e5).abs() < 1.0);

        // Continuous scenarios return None
        let dm = InjectionScenario::AnnihilatingDM { f_ann: 2e-21 };
        assert!(dm.characteristic_redshift().is_none());
    }

    #[test]
    fn test_warn_strong_distortion() {
        // Small distortion: no warning
        let small = InjectionScenario::SingleBurst {
            z_h: 5e4,
            delta_rho_over_rho: 1e-5,
            sigma_z: 2000.0,
        };
        assert!(small.warn_strong_distortion().is_empty());

        // Large distortion: should warn
        let large = InjectionScenario::SingleBurst {
            z_h: 5e4,
            delta_rho_over_rho: 0.1,
            sigma_z: 2000.0,
        };
        assert!(!large.warn_strong_distortion().is_empty());

        // Photon injection with small amplitude: no warning
        let mono_small = InjectionScenario::MonochromaticPhotonInjection {
            x_inj: 5.0,
            delta_n_over_n: 1e-5,
            z_h: 5e4,
            sigma_z: 2000.0,
            sigma_x: 0.5,
        };
        assert!(mono_small.warn_strong_distortion().is_empty());

        // Photon injection with large amplitude: should warn
        let mono_large = InjectionScenario::MonochromaticPhotonInjection {
            x_inj: 5.0,
            delta_n_over_n: 0.1,
            z_h: 5e4,
            sigma_z: 2000.0,
            sigma_x: 0.5,
        };
        assert!(!mono_large.warn_strong_distortion().is_empty());

        // Other scenarios: no warning
        let dm = InjectionScenario::AnnihilatingDM { f_ann: 2e-24 };
        assert!(dm.warn_strong_distortion().is_empty());

        // Photon injection at x_inj <= 150: no high-frequency warning
        let mono_150 = InjectionScenario::MonochromaticPhotonInjection {
            x_inj: 150.0,
            delta_n_over_n: 1e-5,
            z_h: 5e4,
            sigma_z: 2000.0,
            sigma_x: 7.5,
        };
        assert!(mono_150.warn_strong_distortion().is_empty());

        // Photon injection at x_inj > 150: should warn
        let mono_200 = InjectionScenario::MonochromaticPhotonInjection {
            x_inj: 200.0,
            delta_n_over_n: 1e-5,
            z_h: 5e4,
            sigma_z: 2000.0,
            sigma_x: 10.0,
        };
        let warnings = mono_200.warn_strong_distortion();
        assert!(warnings.len() == 1);
        assert!(warnings[0].contains("x_inj"));
        assert!(warnings[0].contains("150"));
    }

    #[test]
    fn test_tabulated_photon_source() {
        let cosmo = Cosmology::default();

        // Create a simple 2D table mimicking a Gaussian photon source
        let z_table = vec![1e5, 2e5, 3e5];
        let x_grid = vec![1.0, 5.0, 10.0];
        let source_2d = vec![
            vec![0.0, 1e-10, 0.0], // z=1e5
            vec![0.0, 5e-10, 0.0], // z=2e5: peak
            vec![0.0, 1e-10, 0.0], // z=3e5
        ];
        let tab = InjectionScenario::TabulatedPhotonSource {
            z_table,
            x_grid,
            source_2d,
        };

        // Should have photon source, no heating
        assert!(tab.has_photon_source());
        assert_eq!(tab.heating_rate(2e5, &cosmo), 0.0);

        // Photon source rate at peak (z=2e5, x=5) should be positive
        let rate = tab.photon_source_rate(5.0, 2e5, &cosmo);
        assert!(
            rate > 0.0 && rate.is_finite(),
            "TabulatedPhotonSource rate at peak: {rate:.4e}"
        );

        // Rate at x=1 should be zero (no source at x=1 in the table)
        let rate_off = tab.photon_source_rate(1.0, 2e5, &cosmo);
        assert!(
            rate_off < rate * 0.01,
            "Rate at x=1 should be much less than peak: {rate_off:.4e} vs {rate:.4e}"
        );

        // Validation should pass
        assert!(tab.validate().is_ok());

        // Name check
        assert_eq!(tab.name(), "tabulated-photon");
    }

    #[test]
    fn test_tabulated_photon_source_bilinear_interpolation() {
        // Test that interp_2d performs correct bilinear interpolation between grid points.
        // Grid: z_table=[1e5, 2e5], x_grid=[5.0, 10.0]
        // source_2d = [[f00, f01], [f10, f11]] = [[1.0, 0.0], [3.0, 0.0]]
        //
        // At the midpoint (z=1.5e5, x=7.5):
        //   tz = 0.5,  tx = 0.5
        //   bilinear = (1-tz)*((1-tx)*f00 + tx*f01) + tz*((1-tx)*f10 + tx*f11)
        //            = 0.5*(0.5*1 + 0.5*0) + 0.5*(0.5*3 + 0.5*0)
        //            = 0.5*0.5 + 0.5*1.5 = 0.25 + 0.75 = 1.0
        let cosmo = Cosmology::default();
        let z_table = vec![1e5, 2e5];
        let x_grid = vec![5.0, 10.0];
        let source_2d = vec![
            vec![1.0, 0.0], // z=1e5
            vec![3.0, 0.0], // z=2e5
        ];
        let tab = InjectionScenario::TabulatedPhotonSource {
            z_table,
            x_grid,
            source_2d,
        };

        // At midpoint, expected dn_dz = 1.0 before Hubble conversion
        // photon_source_rate multiplies by H(z)*(1+z), so we divide it back out.
        let z_mid = 1.5e5;
        let h_factor = cosmo.hubble(z_mid) * (1.0 + z_mid);
        let rate_mid = tab.photon_source_rate(7.5, z_mid, &cosmo);
        let dn_dz = rate_mid / h_factor;
        assert!(
            (dn_dz - 1.0).abs() < 1e-10,
            "Bilinear interpolation at midpoint: dn_dz={dn_dz:.6}, expected 1.0"
        );

        // At z=1e5, x=5.0 (exact grid point): expected 1.0
        let rate_corner = tab.photon_source_rate(5.0, 1e5, &cosmo);
        let h_corner = cosmo.hubble(1e5) * (1.0 + 1e5);
        let dn_dz_corner = rate_corner / h_corner;
        assert!(
            (dn_dz_corner - 1.0).abs() < 1e-10,
            "Exact grid point at z=1e5, x=5: dn_dz={dn_dz_corner:.6}, expected 1.0"
        );

        // Outside the table range: should return 0
        assert_eq!(
            tab.photon_source_rate(5.0, 3e5, &cosmo),
            0.0,
            "Outside z range should return 0"
        );
        assert_eq!(
            tab.photon_source_rate(20.0, 1.5e5, &cosmo),
            0.0,
            "Outside x range should return 0"
        );
    }

    #[test]
    fn test_custom_injection_scenario() {
        let cosmo = Cosmology::default();

        // Custom scenario: constant heating rate of 1e-15 /s
        let custom =
            InjectionScenario::Custom(Box::new(|_z: f64, _cosmo: &Cosmology| -> f64 { 1e-15 }));

        let rate = custom.heating_rate(1e5, &cosmo);
        assert!(
            (rate - 1e-15).abs() < 1e-25,
            "Custom heating rate should be 1e-15: got {rate}"
        );

        // Should not have photon source
        assert!(!custom.has_photon_source());

        // Name check
        assert_eq!(custom.name(), "custom");

        // Rate at different z should be the same (constant)
        let rate2 = custom.heating_rate(5e5, &cosmo);
        assert!(
            (rate2 - rate).abs() < 1e-25,
            "Constant custom rate: {rate} vs {rate2}"
        );
    }
}
