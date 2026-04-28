//! Frequency grid for the photon Boltzmann equation.
//!
//! Non-uniform grid in dimensionless frequency x = hν/(kT_z), with
//! logarithmic spacing at low x (where DC/BR are important) and
//! linear or log spacing at higher x.

/// A localized region of extra grid points for resolving narrow features.
#[derive(Debug, Clone)]
pub struct RefinementZone {
    /// Center of the refinement region
    pub x_center: f64,
    /// Half-width of the refinement region
    pub x_width: f64,
    /// Number of extra points to add in this zone
    pub n_points: usize,
}

/// Configuration for the frequency grid.
///
/// Presets [`Self::fast`] (500 points) and [`Self::production`] (4000 points)
/// cover the common cases. Hand-rolled configurations must satisfy
/// [`Self::validate`]: in particular, `x_max ≥ 30` for accurate G₃ integrals.
#[derive(Debug, Clone)]
pub struct GridConfig {
    /// Lower grid bound in dimensionless frequency `x = hν/(kT_z)`.
    /// Must be positive (log spacing). Default 1e-4.
    pub x_min: f64,
    /// Upper grid bound. Must be ≥ 30 for accurate spectral integrals.
    pub x_max: f64,
    /// Total number of grid points.
    pub n_points: usize,
    /// Transition frequency from log to linear spacing.
    pub x_transition: f64,
    /// Fraction of points allocated to the log region (must be in `[0, 1)`).
    pub log_fraction: f64,
    /// Optional refinement zones for extra resolution near injection features.
    pub refinement_zones: Vec<RefinementZone>,
}

impl Default for GridConfig {
    fn default() -> Self {
        GridConfig {
            x_min: 1e-4,
            x_max: 50.0,
            n_points: 2000,
            x_transition: 0.1,
            log_fraction: 0.3,
            refinement_zones: Vec::new(),
        }
    }
}

impl GridConfig {
    /// Production-quality grid: 4000 points, `x ∈ [1e-5, 60]`. Used for all
    /// paper runs.
    pub fn production() -> Self {
        GridConfig {
            x_min: 1e-5,
            x_max: 60.0,
            n_points: 4000,
            x_transition: 0.5,
            log_fraction: 0.35,
            refinement_zones: Vec::new(),
        }
    }

    /// Fast/testing grid: 500 points, `x ∈ [1e-4, 40]`. Suitable for quick
    /// exploratory runs; distortion amplitudes are accurate to a few percent.
    pub fn fast() -> Self {
        GridConfig {
            x_min: 1e-4,
            x_max: 40.0,
            n_points: 500,
            x_transition: 0.1,
            log_fraction: 0.3,
            refinement_zones: Vec::new(),
        }
    }

    /// Validate grid configuration parameters.
    ///
    /// Returns `Err` with a descriptive message if any parameter would cause
    /// numerical failure or produce meaningless results.
    pub fn validate(&self) -> Result<(), String> {
        if !self.x_min.is_finite() || self.x_min <= 0.0 {
            return Err(format!(
                "x_min must be positive (log grid needs ln(x_min)), got {}",
                self.x_min
            ));
        }
        if !self.x_max.is_finite() || self.x_max <= self.x_min {
            return Err(format!(
                "x_max must be > x_min, got x_min={}, x_max={}",
                self.x_min, self.x_max
            ));
        }
        if self.x_max < 30.0 {
            return Err(format!(
                "x_max must be >= 30 for accurate G3 integrals (per CLAUDE.md pitfall #7: \
                 the high-x tail ∫_{{x_max}}^∞ x³/(e^x−1) dx drops below ~1e-11 only for \
                 x_max ≳ 30), got {}",
                self.x_max
            ));
        }
        if self.n_points < 10 {
            return Err(format!("n_points must be >= 10, got {}", self.n_points));
        }
        if !self.log_fraction.is_finite() || self.log_fraction < 0.0 || self.log_fraction >= 1.0 {
            return Err(format!(
                "log_fraction must be in [0, 1), got {} (1.0 would place all points below x_transition)",
                self.log_fraction
            ));
        }
        if !self.x_transition.is_finite()
            || self.x_transition <= self.x_min
            || self.x_transition >= self.x_max
        {
            return Err(format!(
                "x_transition must be in ({}, {}), got {}",
                self.x_min, self.x_max, self.x_transition
            ));
        }
        for (i, zone) in self.refinement_zones.iter().enumerate() {
            if zone.x_width <= 0.0 {
                return Err(format!(
                    "refinement_zones[{}].x_width must be positive, got {}",
                    i, zone.x_width
                ));
            }
            if zone.n_points == 0 {
                return Err(format!("refinement_zones[{}].n_points must be > 0", i));
            }
            if zone.x_center < self.x_min || zone.x_center > self.x_max {
                return Err(format!(
                    "refinement_zones[{}].x_center={} is outside grid range [{}, {}]",
                    i, zone.x_center, self.x_min, self.x_max
                ));
            }
        }
        Ok(())
    }

    /// Add a refinement zone to this grid configuration.
    pub fn with_refinement(mut self, zone: RefinementZone) -> Self {
        self.refinement_zones.push(zone);
        self
    }
}

/// The frequency grid with precomputed helper arrays.
#[derive(Debug, Clone)]
pub struct FrequencyGrid {
    /// Grid points x_i
    pub x: Vec<f64>,
    /// Grid spacing dx_i = x_{i+1} - x_i (length n-1)
    pub dx: Vec<f64>,
    /// Cell-center values: x_{i+1/2} = (x_i + x_{i+1}) / 2  (length n-1)
    pub x_half: Vec<f64>,
    /// Precomputed `x_half[j]³` (length n-1).
    pub x_half_cubed: Vec<f64>,
    /// Number of points
    pub n: usize,
}

impl FrequencyGrid {
    /// Build a FrequencyGrid from a sorted vector of grid points.
    ///
    /// # Panics
    /// Panics if `x` has fewer than 2 elements.
    pub fn from_points(x: Vec<f64>) -> Self {
        assert!(
            x.len() >= 2,
            "FrequencyGrid::from_points requires at least 2 grid points, got {}",
            x.len()
        );
        let n = x.len();
        let dx: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
        let x_half: Vec<f64> = (0..n - 1).map(|i| 0.5 * (x[i] + x[i + 1])).collect();
        let x_half_cubed: Vec<f64> = x_half.iter().map(|&xh| xh * xh * xh).collect();
        FrequencyGrid {
            x,
            dx,
            x_half,
            x_half_cubed,
            n,
        }
    }

    /// Create a frequency grid from configuration.
    ///
    /// Uses logarithmic spacing for x < x_transition and linear spacing above,
    /// with a smooth blending zone around x_transition. The blending uses a
    /// cubic Hermite (smoothstep) interpolation of the local spacing, avoiding
    /// the discontinuous 30× jump in dx that a hard log-to-linear transition
    /// would produce. This improves local truncation error near x_transition.
    pub fn new(config: &GridConfig) -> Self {
        let n = config.n_points;
        let n_log = (config.log_fraction * n as f64) as usize;
        let n_lin = n - n_log;

        let mut x = Vec::with_capacity(n + 2);

        let log_min = config.x_min.ln();
        let log_max = config.x_transition.ln();
        let dx_lin = (config.x_max - config.x_transition) / (n_lin - 1).max(1) as f64;
        let dln = if n_log > 0 {
            (log_max - log_min) / n_log as f64
        } else {
            0.0
        };
        // Log spacing at the transition point (in x-space)
        let dx_log_at_tr = config.x_transition * dln;

        // Number of blending points on each side: ~10% of each region
        let n_blend_log = if n_log > 4 { (n_log / 10).max(2) } else { 0 };
        let n_blend_lin = if n_lin > 4 { (n_lin / 10).max(2) } else { 0 };

        // Phase 1: Pure log region [x_min, x_blend_start)
        let n_pure_log = n_log.saturating_sub(n_blend_log);
        for i in 0..n_pure_log {
            let t = i as f64 / n_log as f64;
            x.push((log_min + t * (log_max - log_min)).exp());
        }

        // Phase 2: Smooth blending zone (n_blend_log + n_blend_lin points)
        let n_blend = n_blend_log + n_blend_lin;
        if n_blend > 0 && dx_log_at_tr > 0.0 {
            // If we already placed pure log points, advance from the last one
            let mut x_curr = if !x.is_empty() {
                // Step one log-spacing forward from last pure-log point
                let t = n_pure_log as f64 / n_log as f64;
                let dt = 1.0 / n_log as f64;
                (log_min + (t + dt) * (log_max - log_min)).exp()
            } else {
                config.x_min
            };
            // Override: start from where we left off
            if n_pure_log > 0 {
                // Replace: first blend point is the next log point
                x.push(x_curr);
                for j in 1..n_blend {
                    let s = j as f64 / n_blend as f64;
                    // smoothstep: 3s² - 2s³
                    let h = s * s * (3.0 - 2.0 * s);
                    let dx_local = dx_log_at_tr * (1.0 - h) + dx_lin * h;
                    x_curr += dx_local;
                    x.push(x_curr);
                }
            } else {
                for j in 0..n_blend {
                    if j > 0 {
                        let s = j as f64 / n_blend as f64;
                        let h = s * s * (3.0 - 2.0 * s);
                        let dx_local = dx_log_at_tr * (1.0 - h) + dx_lin * h;
                        x_curr += dx_local;
                    }
                    x.push(x_curr);
                }
            }
        }

        // Phase 3: Pure linear region — fill remaining n_lin - n_blend_lin points
        let n_pure_lin = n_lin.saturating_sub(n_blend_lin);
        // The blend ended somewhere past x_transition. Fill remaining linear
        // points from the blend endpoint to x_max.
        let x_after_blend = x.last().copied().unwrap_or(config.x_transition);
        if n_pure_lin > 0 {
            let dx_remaining = (config.x_max - x_after_blend) / n_pure_lin as f64;
            for i in 1..=n_pure_lin {
                x.push(x_after_blend + i as f64 * dx_remaining);
            }
        }

        // Sort and deduplicate
        x.sort_by(|a, b| a.total_cmp(b));
        x.dedup_by(|a, b| {
            let avg = 0.5 * (*a + *b);
            if avg < 1e-30 {
                return (*a - *b).abs() < 1e-30;
            }
            (*a - *b).abs() / avg < 1e-10
        });

        // Merge refinement zone points
        for zone in &config.refinement_zones {
            let lo = (zone.x_center - zone.x_width).max(config.x_min);
            let hi = (zone.x_center + zone.x_width).min(config.x_max);
            if lo >= hi || zone.n_points == 0 {
                continue;
            }
            let log_lo = lo.ln();
            let log_hi = hi.ln();
            for i in 0..zone.n_points {
                let t = i as f64 / (zone.n_points - 1).max(1) as f64;
                x.push((log_lo + t * (log_hi - log_lo)).exp());
            }
        }

        // Sort and deduplicate with relative tolerance
        if !config.refinement_zones.is_empty() {
            x.sort_by(|a, b| a.total_cmp(b));
            x.dedup_by(|a, b| {
                let avg = 0.5 * (*a + *b);
                if avg < 1e-30 {
                    return (*a - *b).abs() < 1e-30;
                }
                (*a - *b).abs() / avg < 1e-10
            });
        }

        Self::from_points(x)
    }

    /// Create a purely logarithmic grid (useful for testing).
    ///
    /// # Panics
    /// Panics if `n < 2`.
    pub fn log_uniform(x_min: f64, x_max: f64, n: usize) -> Self {
        assert!(n >= 2, "log_uniform requires n >= 2, got {n}");
        let log_min = x_min.ln();
        let log_max = x_max.ln();
        let x: Vec<f64> = (0..n)
            .map(|i| (log_min + (log_max - log_min) * i as f64 / (n - 1) as f64).exp())
            .collect();
        Self::from_points(x)
    }

    /// Create a uniform grid (useful for testing).
    ///
    /// # Panics
    /// Panics if `n < 2`.
    pub fn uniform(x_min: f64, x_max: f64, n: usize) -> Self {
        assert!(n >= 2, "uniform requires n >= 2, got {n}");
        let x: Vec<f64> = (0..n)
            .map(|i| x_min + (x_max - x_min) * i as f64 / (n - 1) as f64)
            .collect();
        Self::from_points(x)
    }

    /// Find the index of the grid point closest to a given x value.
    pub fn find_index(&self, x_target: f64) -> usize {
        match self.x.binary_search_by(|a| a.total_cmp(&x_target)) {
            Ok(i) => i,
            Err(i) => {
                if i == 0 {
                    0
                } else if i >= self.n {
                    self.n - 1
                } else if (self.x[i] - x_target).abs() < (self.x[i - 1] - x_target).abs() {
                    i
                } else {
                    i - 1
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_dx_consistency() {
        let grid = FrequencyGrid::new(&GridConfig::default());
        for i in 0..grid.dx.len() {
            let dx_expected = grid.x[i + 1] - grid.x[i];
            assert!((grid.dx[i] - dx_expected).abs() < 1e-14);
        }
    }

    #[test]
    fn test_grid_log_region() {
        // In the log region, dx/x should be approximately constant
        let grid = FrequencyGrid::new(&GridConfig::default());
        let n_log =
            (GridConfig::default().log_fraction * GridConfig::default().n_points as f64) as usize;
        if n_log > 2 {
            let ratio_first = grid.dx[0] / grid.x[0];
            let ratio_mid = grid.dx[n_log / 2] / grid.x[n_log / 2];
            assert!(
                (ratio_first - ratio_mid).abs() / ratio_first < 0.1,
                "Log region should have ~constant dx/x"
            );
        }
    }

    #[test]
    fn test_refinement_zone_adds_points() {
        let base = GridConfig::default();
        let base_grid = FrequencyGrid::new(&base);
        let refined = GridConfig {
            refinement_zones: vec![RefinementZone {
                x_center: 0.1,
                x_width: 0.05,
                n_points: 200,
            }],
            ..GridConfig::default()
        };
        let refined_grid = FrequencyGrid::new(&refined);
        assert!(
            refined_grid.n > base_grid.n,
            "Refined grid ({}) should have more points than base ({})",
            refined_grid.n,
            base_grid.n
        );
        // Check monotonicity
        for i in 1..refined_grid.n {
            assert!(
                refined_grid.x[i] > refined_grid.x[i - 1],
                "Refined grid not monotonic at i={i}: x[{i}]={} <= x[{}]={}",
                refined_grid.x[i],
                i - 1,
                refined_grid.x[i - 1]
            );
        }
    }

    #[test]
    fn test_refinement_zone_no_duplicates() {
        let config = GridConfig {
            refinement_zones: vec![RefinementZone {
                x_center: 0.1,
                x_width: 0.05,
                n_points: 300,
            }],
            ..GridConfig::default()
        };
        let grid = FrequencyGrid::new(&config);
        // No near-duplicate points (relative spacing > 1e-10)
        for i in 1..grid.n {
            let rel = (grid.x[i] - grid.x[i - 1]) / (0.5 * (grid.x[i] + grid.x[i - 1]));
            assert!(
                rel > 1e-10,
                "Near-duplicate at i={i}: x[{i}]={:.6e}, x[{}]={:.6e}, rel={:.2e}",
                grid.x[i],
                i - 1,
                grid.x[i - 1],
                rel
            );
        }
    }

    #[test]
    fn test_refinement_zone_density() {
        // Points in the refinement zone should be denser than outside
        let config = GridConfig {
            refinement_zones: vec![RefinementZone {
                x_center: 5.0,
                x_width: 1.0,
                n_points: 200,
            }],
            ..GridConfig::default()
        };
        let grid = FrequencyGrid::new(&config);
        // Count points in [4,6] vs [6,8] (same width, no refinement)
        let in_zone = grid.x.iter().filter(|&&x| x >= 4.0 && x <= 6.0).count();
        let out_zone = grid.x.iter().filter(|&&x| x >= 6.0 && x <= 8.0).count();
        assert!(
            in_zone > out_zone * 2,
            "Refinement zone should be at least 2x denser: {} in zone vs {} outside",
            in_zone,
            out_zone
        );
    }

    #[test]
    fn test_empty_refinement_unchanged() {
        let base = GridConfig::default();
        let base_grid = FrequencyGrid::new(&base);
        let with_empty = GridConfig {
            refinement_zones: Vec::new(),
            ..GridConfig::default()
        };
        let empty_grid = FrequencyGrid::new(&with_empty);
        assert_eq!(base_grid.n, empty_grid.n);
    }
}
