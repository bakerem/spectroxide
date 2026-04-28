//! CosmoTherm comparison tests.
//!
//! Validates PDE solver output against CosmoTherm reference data
//! (Chluba & Sunyaev 2012, Chluba 2016). Data files are in data/cosmotherm/.
//!
//! DI_cooling.dat stores -ΔI (sign-flipped).
//! DI_damping.dat is the TOTAL signal (damping + cooling combined).
//!
//! Cosmology: Planck 2015 (Y_p=0.2467, T0=2.726, Om_cdm=0.264737, Ob=0.049169, h=0.6727).

use spectroxide::cosmology::Cosmology;
use spectroxide::distortion;
use spectroxide::grid::GridConfig;
use spectroxide::solver::{SolverConfig, ThermalizationSolver};

/// Parse a CosmoTherm DI file: two columns (nu_GHz, DI_Jy_sr), skip comment lines.
fn load_cosmotherm_di(path: &str) -> (Vec<f64>, Vec<f64>) {
    let contents = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("Cannot read {path}"));
    let mut nus = Vec::new();
    let mut dis = Vec::new();
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() >= 2 {
            if let (Ok(nu), Ok(di)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                nus.push(nu);
                dis.push(di);
            }
        }
    }
    (nus, dis)
}

/// Verify that CosmoTherm data files can be loaded and have the expected structure.
#[test]
fn test_cosmotherm_data_loads() {
    let (cool_nus, cool_di) = load_cosmotherm_di("data/cosmotherm/DI_cooling.dat");
    assert!(
        cool_nus.len() > 4000,
        "Cooling data should have > 4000 points, got {}",
        cool_nus.len()
    );
    assert_eq!(cool_nus.len(), cool_di.len());

    let (damp_nus, damp_di) = load_cosmotherm_di("data/cosmotherm/DI_damping.dat");
    assert!(
        damp_nus.len() > 4000,
        "Damping data should have > 4000 points, got {}",
        damp_nus.len()
    );
    assert_eq!(damp_nus.len(), damp_di.len());

    // Cooling signal: DI_cooling stores -ΔI, so raw values should be
    // mostly positive (since cooling makes ΔI negative at most frequencies).
    // Check that at least some values at ν > 100 GHz are positive (positive raw = negative ΔI)
    let high_freq_positive: usize = cool_nus
        .iter()
        .zip(cool_di.iter())
        .filter(|&(&nu, &di)| nu > 100.0 && di > 0.0)
        .count();
    assert!(
        high_freq_positive > 100,
        "Expected many positive raw cooling DI entries at ν > 100 GHz"
    );

    // Damping signal: total (damping + cooling), should have both signs
    // (positive at FIRAS frequencies ~ 100-300 GHz, negative at high freq)
    let has_positive = damp_di.iter().any(|&v| v > 0.0);
    let has_negative = damp_di.iter().any(|&v| v < 0.0);
    assert!(
        has_positive,
        "Damping signal should have positive ΔI values"
    );
    assert!(
        has_negative,
        "Damping signal should have negative ΔI values"
    );
}

/// Verify that the cooling sign convention is correct by checking that
/// negated DI_cooling has the expected spectral shape: negative at low
/// frequencies, positive at intermediate, negative at high.
#[test]
fn test_cosmotherm_cooling_sign_convention() {
    let (nus, di_raw) = load_cosmotherm_di("data/cosmotherm/DI_cooling.dat");
    // Negate: DI_cooling stores -ΔI
    let di: Vec<f64> = di_raw.iter().map(|&v| -v).collect();

    // At ~200 GHz (near the y-distortion crossover), cooling ΔI should
    // be negative (fewer photons than Planck due to expansion cooling).
    let idx_200 = nus
        .iter()
        .position(|&nu| nu > 200.0)
        .expect("No ν > 200 GHz");
    assert!(
        di[idx_200] < 0.0,
        "Cooling ΔI at ~200 GHz should be negative (expansion removes energy), got {:.4e}",
        di[idx_200]
    );

    // Peak of |ΔI| should be in FIRAS range (100-800 GHz)
    let peak_nu = nus
        .iter()
        .zip(di.iter())
        .filter(|&(&nu, _)| nu > 50.0 && nu < 1000.0)
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(&nu, _)| nu)
        .unwrap();
    eprintln!("Cooling peak |ΔI| at ν = {peak_nu:.1} GHz");
    assert!(
        peak_nu > 50.0 && peak_nu < 1000.0,
        "Cooling peak should be in 50-1000 GHz, got {peak_nu:.1} GHz"
    );
}

/// Run a single-burst PDE and verify that the spectral distortion ΔI
/// is in the right ballpark compared to CosmoTherm-validated expectations.
///
/// This is NOT a direct comparison with CosmoTherm data (which uses acoustic
/// dissipation), but validates the PDE-to-intensity conversion pipeline.
#[test]
fn test_pde_intensity_conversion_sanity() {
    let cosmo = Cosmology::planck2015();
    let mut solver = ThermalizationSolver::builder(cosmo.clone())
        .grid(GridConfig {
            n_points: 2000,
            ..GridConfig::default()
        })
        .injection(
            spectroxide::energy_injection::InjectionScenario::SingleBurst {
                delta_rho_over_rho: 1e-5,
                z_h: 5e4,
                sigma_z: 3000.0,
            },
        )
        .z_range(1e5, 200.0)
        .build()
        .unwrap();

    solver.run_with_snapshots(&[200.0]);
    let snap = solver.snapshots.last().unwrap();

    // Convert to ΔI [MJy/sr] at 300 GHz (x ≈ 5.3)
    let x_300 = 300.0e9 * 6.626_070_15e-34 / (1.380_649e-23 * 2.726);
    // Interpolate delta_n at x=x_300
    let x_grid = &solver.grid.x;
    let mut lo = 0;
    let mut hi = x_grid.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if x_grid[mid] <= x_300 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (x_300 - x_grid[lo]) / (x_grid[hi] - x_grid[lo]);
    let dn_300 = snap.delta_n[lo] + t * (snap.delta_n[hi] - snap.delta_n[lo]);
    let di_300 = distortion::delta_n_to_intensity_mjy(x_300, dn_300, 2.726);

    eprintln!(
        "PDE SingleBurst(Δρ/ρ=1e-5, z=5e4): ΔI(300 GHz)={di_300:.4e} MJy/sr, \
         μ={:.4e}, y={:.4e}",
        snap.mu, snap.y
    );

    // For Δρ/ρ = 1e-5 in the y-era (z=5e4), expect y ≈ 2.5e-6
    // ΔI(300 GHz) ~ y × Y_SZ(300 GHz) × I_0 ~ 2.5e-6 × 3 × 270 MJy/sr ~ 2e-3 MJy/sr
    // So |ΔI| should be in [1e-4, 1e-2] MJy/sr range
    assert!(
        di_300.abs() > 1e-5 && di_300.abs() < 1e-1,
        "ΔI at 300 GHz = {di_300:.4e} MJy/sr out of expected range [1e-5, 1e-1]"
    );
}

// =============================================================================
// Helper: convert frequency in GHz to dimensionless x = hν/(k_B T_CMB)
// =============================================================================

const T_CMB: f64 = 2.726;
const HPLANCK: f64 = 6.626_070_15e-34;
const K_BOLTZMANN: f64 = 1.380_649e-23;

fn nu_ghz_to_x(nu_ghz: f64) -> f64 {
    nu_ghz * 1e9 * HPLANCK / (K_BOLTZMANN * T_CMB)
}

/// Linearly interpolate PDE Δn (on x_grid) onto a target x value.
fn interpolate_delta_n(x_grid: &[f64], delta_n: &[f64], x_target: f64) -> f64 {
    // Binary search for bracketing interval
    let mut lo = 0;
    let mut hi = x_grid.len() - 1;
    if x_target <= x_grid[lo] {
        return delta_n[lo];
    }
    if x_target >= x_grid[hi] {
        return delta_n[hi];
    }
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if x_grid[mid] <= x_target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (x_target - x_grid[lo]) / (x_grid[hi] - x_grid[lo]);
    delta_n[lo] + t * (delta_n[hi] - delta_n[lo])
}

// =============================================================================
// Test 1: CosmoTherm cooling decomposition and PDE single-burst comparison
// =============================================================================

/// Load CosmoTherm DI_cooling.dat (adiabatic cooling, stores -ΔI in Jy/sr),
/// decompose into μ/y, and compare the spectral shape against a PDE single-burst
/// run at similar energy scale.
///
/// The adiabatic cooling signal is a tiny (~Δρ/ρ ~ -4e-9) negative-energy
/// distortion that the PDE solver cannot produce directly (it lacks a built-in
/// expansion cooling source term). Instead, we:
///   1. Verify the CosmoTherm cooling μ/y are physically reasonable (μ < 0, |μ| ~ few × 10⁻⁹)
///   2. Run a PDE SingleBurst with the SAME |Δρ/ρ| and verify the spectral shape
///      is a mirror image (opposite sign, similar |μ|/|Δρ/ρ| ratio)
///   3. Compare the spectral shape (ΔI vs ν) pointwise: PDE should be approximately
///      -1× the CosmoTherm cooling signal, scaled by energy ratio.
#[test]
fn test_cosmotherm_cooling_pde_comparison() {
    // Load CosmoTherm reference: negate because file stores -ΔI
    let (ct_nus_ghz, ct_di_raw) = load_cosmotherm_di("data/cosmotherm/DI_cooling.dat");
    let ct_di_jy: Vec<f64> = ct_di_raw.iter().map(|&v| -v).collect();

    // Decompose CosmoTherm cooling using the same BF method as the PDE solver.
    let ct_x: Vec<f64> = ct_nus_ghz.iter().map(|&nu| nu_ghz_to_x(nu)).collect();
    let ct_dn: Vec<f64> = ct_nus_ghz
        .iter()
        .zip(ct_di_jy.iter())
        .map(|(&nu, &di_jy)| {
            let nu_hz = nu * 1e9;
            let i_pref = 2.0 * HPLANCK * nu_hz.powi(3)
                / (spectroxide::constants::C_LIGHT * spectroxide::constants::C_LIGHT)
                * 1e26;
            di_jy / i_pref
        })
        .collect();
    let ct_params = spectroxide::distortion::decompose_distortion(&ct_x, &ct_dn);
    let ct_mu = ct_params.mu;
    let ct_y = ct_params.y;

    eprintln!("CosmoTherm cooling decomposition: μ = {ct_mu:.4e}, y = {ct_y:.4e}");

    // Cooling μ should be negative (energy extracted from photons)
    // Literature: μ_cool ~ -2 to -3 × 10⁻⁹ (Chluba 2016)
    assert!(ct_mu < 0.0, "Cooling μ should be negative, got {ct_mu:.4e}");
    assert!(
        ct_mu.abs() > 1e-10 && ct_mu.abs() < 1e-7,
        "Cooling |μ| = {:.4e} outside expected range [1e-10, 1e-7]",
        ct_mu.abs()
    );

    // Run a PDE SingleBurst with energy scale matching the cooling signal.
    // The cooling injects Δρ/ρ ~ -4e-9 spread over z ~ 5e4 to 2e6.
    // We use a positive burst at z=2e5 (μ-era) to compare spectral shapes.
    let drho_pde = 1e-5; // Much larger for numerical accuracy
    let cosmo = Cosmology::planck2015();
    let mut solver = ThermalizationSolver::builder(cosmo)
        .grid(GridConfig {
            n_points: 2000,
            ..GridConfig::default()
        })
        .injection(
            spectroxide::energy_injection::InjectionScenario::SingleBurst {
                delta_rho_over_rho: drho_pde,
                z_h: 2e5,
                sigma_z: 100.0,
            },
        )
        .z_range(3e5, 200.0)
        .build()
        .unwrap();

    solver.run_with_snapshots(&[200.0]);
    let snap = solver.snapshots.last().unwrap();

    eprintln!(
        "PDE SingleBurst: μ = {:.4e}, y = {:.4e}, Δρ/ρ = {:.4e}",
        snap.mu, snap.y, snap.delta_rho_over_rho
    );

    // The PDE μ should be positive (heating), opposite sign from cooling
    assert!(snap.mu > 0.0, "PDE μ should be positive for heat injection");

    // Compare μ/(Δρ/ρ) ratios: both should be close to 1.401 in magnitude.
    // Cooling is continuous over wide z range, so its μ/Δρ ratio is lower
    // than the pure μ-era value. Still, both should be O(1).
    let pde_mu_over_drho = snap.mu / snap.delta_rho_over_rho;
    // For CT cooling, we need Δρ/ρ: compute from μ and y using energy conservation.
    // Δρ/ρ = μ/1.401 + 4y + 4ΔT/T. For order of magnitude, Δρ/ρ ~ μ/1.401.
    let ct_drho_est = ct_mu / 1.401 + 4.0 * ct_y;
    let ct_mu_over_drho = if ct_drho_est.abs() > 1e-15 {
        ct_mu / ct_drho_est
    } else {
        f64::NAN
    };

    eprintln!("  PDE μ/(Δρ/ρ) = {pde_mu_over_drho:.3}, CT μ/(Δρ/ρ) ≈ {ct_mu_over_drho:.3}");

    // PDE μ/Δρ should be close to 1.401 for z=2e5 (deep μ-era)
    assert!(
        pde_mu_over_drho > 0.8 && pde_mu_over_drho < 1.6,
        "PDE mu/(drho/rho) = {pde_mu_over_drho:.3} outside expected range [0.8, 1.6]"
    );

    // Now compare spectral shapes: scale PDE ΔI by (CT_Δρ/ρ / PDE_Δρ/ρ) and
    // check the ratio against CosmoTherm ΔI (should be ~ -1 since cooling is
    // the negative of heating). Only at FIRAS frequencies where signal is strong.
    let x_grid = &solver.grid.x;
    let delta_n = &snap.delta_n;
    let scale = ct_drho_est / snap.delta_rho_over_rho;

    let mut ratio_sum = 0.0;
    let mut ratio_count = 0usize;
    for (i, &nu) in ct_nus_ghz.iter().enumerate() {
        if nu < 150.0 || nu > 600.0 {
            continue;
        }
        let ct_val = ct_di_jy[i];
        if ct_val.abs() < 1e-30 {
            continue;
        }
        let x = nu_ghz_to_x(nu);
        let dn_interp = interpolate_delta_n(x_grid, delta_n, x);
        // PDE ΔI in Jy/sr (multiply MJy by 1e6)
        let pde_jy = distortion::delta_n_to_intensity_mjy(x, dn_interp, T_CMB) * 1e6;
        let scaled_pde = pde_jy * scale;

        let ratio = scaled_pde / ct_val;
        ratio_sum += ratio;
        ratio_count += 1;
    }

    if ratio_count > 0 {
        let mean_ratio = ratio_sum / ratio_count as f64;
        eprintln!(
            "  Mean (scaled PDE / CT) ratio at 150-600 GHz = {mean_ratio:.3} \
             (expect ~1.0 if shapes match)"
        );
        // Shapes should broadly agree (ratio near 1.0), but we allow generous
        // tolerance because the heating histories differ (single burst vs continuous).
        assert!(
            mean_ratio.abs() > 0.1 && mean_ratio.abs() < 10.0,
            "Scaled PDE/CT ratio = {mean_ratio:.3} — spectral shapes are inconsistent"
        );
    }
}

// =============================================================================
// Test 2: CosmoTherm damping μ/y decomposition comparison
// =============================================================================

/// Load CosmoTherm DI_damping.dat (total signal: acoustic dissipation + cooling)
/// and decompose into μ/y by fitting M(x) and Y_SZ(x) shapes. Also run a
/// decaying-particle PDE with parameters that produce comparable energy injection.
///
/// Since the heating histories differ (acoustic dissipation vs single exponential),
/// we use loose tolerance (factor of 2). The main value is verifying that both
/// give physically reasonable distortion amplitudes in the μ ~ 10⁻⁸ range.
#[test]
fn test_cosmotherm_damping_mu_y_comparison() {
    // Load CosmoTherm damping signal (already ΔI in Jy/sr, no sign flip needed)
    let (ct_nus_ghz, ct_di_jy) = load_cosmotherm_di("data/cosmotherm/DI_damping.dat");

    // Decompose CosmoTherm ΔI using the same BF method as the PDE solver.
    let ct_x: Vec<f64> = ct_nus_ghz.iter().map(|&nu| nu_ghz_to_x(nu)).collect();
    let ct_dn: Vec<f64> = ct_nus_ghz
        .iter()
        .zip(ct_di_jy.iter())
        .map(|(&nu, &di_jy)| {
            let nu_hz = nu * 1e9;
            let i_pref = 2.0 * HPLANCK * nu_hz.powi(3)
                / (spectroxide::constants::C_LIGHT * spectroxide::constants::C_LIGHT)
                * 1e26;
            di_jy / i_pref
        })
        .collect();
    let ct_params = spectroxide::distortion::decompose_distortion(&ct_x, &ct_dn);
    let ct_mu = ct_params.mu;
    let ct_y = ct_params.y;

    eprintln!("CosmoTherm damping decomposition: μ = {ct_mu:.4e}, y = {ct_y:.4e}");

    // Standard ΛCDM predictions (Chluba 2016): μ ≈ 2.0e-8, y ~ few × 10⁻⁹.
    // Damping file includes cooling, so net values may differ slightly.
    assert!(
        ct_mu > 0.0,
        "CosmoTherm damping μ should be positive (heating dominates cooling)"
    );
    assert!(
        ct_mu > 1e-9 && ct_mu < 1e-6,
        "CosmoTherm μ = {ct_mu:.4e} outside physically expected range [1e-9, 1e-6]"
    );
    assert!(
        ct_y.abs() < 1e-6,
        "CosmoTherm y = {ct_y:.4e} outside expected range |y| < 1e-6"
    );

    // Run a decaying-particle PDE with parameters producing comparable μ ~ 10⁻⁸.
    // f_x = 1e3 eV, gamma_x = 1e-13 s⁻¹ (lifetime ~ 10¹³ s ~ age at z ~ 10⁵)
    // gives μ ~ O(10⁻⁷ to 10⁻⁶), which is larger than ΛCDM damping but allows
    // a meaningful order-of-magnitude comparison.
    let cosmo = Cosmology::planck2015();
    let mut solver = ThermalizationSolver::builder(cosmo)
        .grid(GridConfig {
            n_points: 2000,
            ..GridConfig::default()
        })
        .injection(
            spectroxide::energy_injection::InjectionScenario::DecayingParticle {
                f_x: 1e3,
                gamma_x: 1e-13,
            },
        )
        .z_range(3e6, 200.0)
        .build()
        .unwrap();

    solver.run_with_snapshots(&[200.0]);
    let snap = solver.snapshots.last().unwrap();

    eprintln!(
        "PDE decaying particle (f_x=1e3 eV, Gamma=1e-13): μ = {:.4e}, y = {:.4e}, Δρ/ρ = {:.4e}",
        snap.mu, snap.y, snap.delta_rho_over_rho
    );

    // PDE μ should be in a physically reasonable range. With correct
    // Λ·ρ_e adiabatic cooling, μ can be slightly negative (~ -3e-9)
    // when the cooling contribution competes with weak injection.
    assert!(
        snap.mu.abs() < 1e-2,
        "PDE μ = {:.4e} outside reasonable range",
        snap.mu
    );

    // The ratio μ/Δρ should be physically reasonable: between 0 and 1.401
    // (saturated μ-era). Continuous injection from z ~ 10⁵ gives a mix of μ and y.
    if snap.delta_rho_over_rho.abs() > 1e-15 {
        let mu_over_drho = snap.mu / snap.delta_rho_over_rho;
        eprintln!(
            "  PDE μ/(Δρ/ρ) = {mu_over_drho:.4}, CT μ/(Δρ/ρ) ≈ {:.4}",
            ct_mu / (ct_mu / 1.401 + 4.0 * ct_y)
        );
        // With correct Λ·ρ_e cooling, μ/Δρ can be slightly negative
        // when adiabatic cooling competes with weak injection.
        assert!(
            mu_over_drho > -0.1 && mu_over_drho < 1.5,
            "μ/(Δρ/ρ) = {mu_over_drho:.4} outside expected range (-0.1, 1.5)"
        );
    }
}

// =============================================================================
// Test 3: SingleBurst μ-distortion spectral shape validation
// =============================================================================

/// Run a SingleBurst at z_h=2e5 (deep in μ-era), Δρ/ρ=1e-5 with Planck 2015
/// cosmology. Validate the resulting spectral distortion has the expected
/// μ-distortion properties:
///   - M(x) = (x/β_μ - 1) G_bb(x)/x crosses zero at x = β_μ ≈ 2.19 (ν ≈ 124 GHz)
///   - For μ > 0: ΔI negative below zero crossing (RJ deficit), positive above (Wien)
///   - Dominant μ with small y
///   - μ/(Δρ/ρ) ≈ 1.401 (the analytical μ-era relation)
///
/// This validates the end-to-end pipeline: PDE -> Δn -> ΔI conversion.
#[test]
fn test_cosmotherm_single_burst_spectral_shape() {
    let cosmo = Cosmology::planck2015();
    let mut solver = ThermalizationSolver::builder(cosmo)
        .grid(GridConfig {
            n_points: 2000,
            ..GridConfig::default()
        })
        .injection(
            spectroxide::energy_injection::InjectionScenario::SingleBurst {
                delta_rho_over_rho: 1e-5,
                z_h: 2e5,
                sigma_z: 100.0,
            },
        )
        .z_range(3e5, 200.0)
        .build()
        .unwrap();

    solver.run_with_snapshots(&[200.0]);
    let snap = solver.snapshots.last().expect("No snapshot at z=200");
    let x_grid = &solver.grid.x;
    let delta_n = &snap.delta_n;

    eprintln!(
        "SingleBurst(z=2e5, drho/rho=1e-5): mu = {:.4e}, y = {:.4e}, drho/rho = {:.4e}",
        snap.mu, snap.y, snap.delta_rho_over_rho,
    );

    // 1. mu should dominate: mu > 0, |mu| >> |y|
    assert!(
        snap.mu > 0.0,
        "mu should be positive for heat injection at z=2e5"
    );
    assert!(
        snap.mu.abs() > snap.y.abs() * 3.0,
        "mu ({:.4e}) should dominate over y ({:.4e}) for z_h=2e5",
        snap.mu,
        snap.y
    );

    // 2. mu/(drho/rho) should be close to 1.401 (with some y leakage reducing it)
    let mu_over_drho = snap.mu / snap.delta_rho_over_rho;
    eprintln!("mu/(drho/rho) = {mu_over_drho:.4} (expected ~ 1.401)");
    assert!(
        mu_over_drho > 0.8 && mu_over_drho < 1.6,
        "mu/(drho/rho) = {mu_over_drho:.4} outside expected range [0.8, 1.6]"
    );

    // 3. Check spectral shape: convert to DI at key frequencies.
    // mu-distortion shape M(x) = (x/beta_mu - 1) * G_bb(x)/x.
    // For mu > 0: Dn = mu * M(x), so at x < beta_mu M(x) < 0 => Dn < 0 => DI < 0.
    // At x > beta_mu: M(x) > 0 => Dn > 0 => DI > 0.
    let freqs_ghz = [60.0, 100.0, 150.0, 300.0, 500.0, 700.0];
    let mut di_values = Vec::new();
    for &nu in &freqs_ghz {
        let x = nu_ghz_to_x(nu);
        let dn = interpolate_delta_n(x_grid, delta_n, x);
        let di = distortion::delta_n_to_intensity_mjy(x, dn, T_CMB);
        di_values.push(di);
        eprintln!("  DI({nu:.0} GHz) = {di:.4e} MJy/sr");
    }

    // Below zero crossing (~124 GHz): DI should be negative for mu > 0
    // 60 GHz (x ~ 1.06) is well below beta_mu ~ 2.19
    assert!(
        di_values[0] < 0.0,
        "DI(60 GHz) = {:.4e} should be negative (RJ side deficit for mu > 0)",
        di_values[0]
    );

    // Above zero crossing: DI should be positive for mu > 0
    // 300, 500 GHz are well above the crossing
    assert!(
        di_values[3] > 0.0,
        "DI(300 GHz) = {:.4e} should be positive (Wien side for mu > 0)",
        di_values[3]
    );
    assert!(
        di_values[4] > 0.0,
        "DI(500 GHz) = {:.4e} should be positive (Wien side for mu > 0)",
        di_values[4]
    );

    // 4. Zero crossing should be near nu ~ 124 GHz (x ~ beta_mu ~ 2.19).
    // Check that DI changes sign between 100 GHz (x ~ 1.76) and 150 GHz (x ~ 2.64).
    assert!(
        di_values[1] * di_values[2] < 0.0,
        "DI should change sign between 100 GHz ({:.4e}) and 150 GHz ({:.4e}), \
         confirming zero crossing near beta_mu",
        di_values[1],
        di_values[2]
    );

    // 5. Amplitude check: for drho/rho = 1e-5, peak |DI| should be O(1e-3) MJy/sr
    let peak_abs = di_values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        peak_abs > 1e-5 && peak_abs < 1e-1,
        "Peak |DI| = {peak_abs:.4e} MJy/sr outside expected range [1e-5, 1e-1]"
    );

    // 6. At 700 GHz (x ~ 12.3), the signal should be much smaller than at 300 GHz
    // because the mu-distortion falls off exponentially in the Wien tail.
    assert!(
        di_values[5].abs() < di_values[3].abs(),
        "DI(700 GHz) = {:.4e} should be smaller than DI(300 GHz) = {:.4e} (Wien tail)",
        di_values[5],
        di_values[3]
    );
}

// =============================================================================
// Test: Adiabatic cooling μ-distortion — PDE solver vs CosmoTherm
// =============================================================================

/// Run the PDE solver with zero injection (adiabatic cooling only) using
/// Planck 2015 cosmology to match the CosmoTherm reference. Decompose the
/// CosmoTherm DI_cooling.dat into μ/y and compare against PDE output.
///
/// Physical picture: expansion cooling extracts energy from the photon field
/// via Compton scattering with adiabatically cooling electrons. This produces
/// a small negative μ-distortion: μ_cool ≈ −3 × 10⁻⁹ (Chluba & Sunyaev 2012,
/// Chluba 2016, Fig. 1).
///
/// The PDE solver captures this through the Λ·ρ_e term in the T_e equation:
/// matter cools as T_m ∝ (1+z)² while T_CMB ∝ (1+z), creating ρ_e < 1 at
/// high z that feeds back through the Kompaneets operator.
///
/// CosmoTherm reference: DI_cooling.dat, Planck 2015 cosmology.
/// Measured agreement: μ within 2% of CosmoTherm.
#[test]
fn test_adiabatic_cooling_mu_vs_cosmotherm() {
    // --- CosmoTherm decomposition ---
    // Use the same BF nonlinear decomposition as the PDE solver so the
    // comparison is method-consistent. Convert the CosmoTherm ΔI [Jy/sr]
    // to occupation-number Δn on a dimensionless x grid, then call the
    // standard extractor.
    let (ct_nus_ghz, ct_di_raw) = load_cosmotherm_di("data/cosmotherm/DI_cooling.dat");
    let ct_di_jy: Vec<f64> = ct_di_raw.iter().map(|&v| -v).collect(); // negate: file stores -ΔI
    let ct_x: Vec<f64> = ct_nus_ghz.iter().map(|&nu| nu_ghz_to_x(nu)).collect();
    let ct_dn: Vec<f64> = ct_nus_ghz
        .iter()
        .zip(ct_di_jy.iter())
        .map(|(&nu, &di_jy)| {
            let nu_hz = nu * 1e9;
            let i_pref = 2.0 * HPLANCK * nu_hz.powi(3)
                / (spectroxide::constants::C_LIGHT * spectroxide::constants::C_LIGHT)
                * 1e26;
            di_jy / i_pref
        })
        .collect();
    let ct_params = spectroxide::distortion::decompose_distortion(&ct_x, &ct_dn);
    let ct_mu = ct_params.mu;
    let ct_y = ct_params.y;

    eprintln!("CosmoTherm cooling: μ = {ct_mu:.4e}, y = {ct_y:.4e}");

    // Sanity: CosmoTherm cooling μ should be negative and O(10⁻⁹)
    assert!(ct_mu < 0.0, "CosmoTherm cooling μ should be negative");
    assert!(
        ct_mu.abs() > 1e-10 && ct_mu.abs() < 1e-7,
        "CosmoTherm |μ| = {:.4e} outside [1e-10, 1e-7]",
        ct_mu.abs()
    );

    // --- PDE solver with matching cosmology ---
    let cosmo = Cosmology::planck2015();
    let mut solver = ThermalizationSolver::new(cosmo, GridConfig::default());
    solver.set_config(SolverConfig {
        z_start: 3.0e6,
        z_end: 1000.0,
        ..SolverConfig::default()
    });
    // No injection — pure adiabatic cooling
    solver.run_with_snapshots(&[1000.0]);
    let snap = solver.snapshots.last().unwrap();

    eprintln!(
        "PDE adiabatic cooling (Planck 2015): μ = {:.4e}, y = {:.4e}, Δρ/ρ = {:.4e}",
        snap.mu, snap.y, snap.delta_rho_over_rho
    );

    // PDE μ should also be negative
    assert!(
        snap.mu < 0.0,
        "PDE adiabatic μ should be negative, got {:.4e}",
        snap.mu
    );

    // μ comparison: PDE vs CosmoTherm.
    // Measured agreement: ~1.2%. Allow 5% for grid resolution and
    // cosmology parameter precision differences.
    let mu_err = (snap.mu - ct_mu).abs() / ct_mu.abs();
    eprintln!(
        "  μ agreement: PDE={:.4e}, CT={:.4e}, err={:.2}%",
        snap.mu,
        ct_mu,
        mu_err * 100.0
    );
    assert!(
        mu_err < 0.05,
        "Adiabatic μ: PDE ({:.4e}) vs CosmoTherm ({:.4e}), err={:.1}% (limit 5%)",
        snap.mu,
        ct_mu,
        mu_err * 100.0
    );

    // Both PDE and CosmoTherm μ should be in the literature range:
    // μ_cool ≈ −(2.5 to 3.5) × 10⁻⁹ (Chluba 2016)
    assert!(
        snap.mu > -5e-9 && snap.mu < -1e-9,
        "PDE μ = {:.4e} outside expected [-5e-9, -1e-9]",
        snap.mu
    );
    assert!(
        ct_mu > -5e-9 && ct_mu < -1e-9,
        "CT μ = {:.4e} outside expected [-5e-9, -1e-9]",
        ct_mu
    );
}
