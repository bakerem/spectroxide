//! Helpers for dark photon (γ ↔ A') conversion in the narrow-width approximation.
//!
//! The canonical way to model γ ↔ A' resonant conversion in the partial
//! differential equation solver is
//! [`crate::energy_injection::InjectionScenario::DarkPhotonResonance`], which
//! takes (ε, m_{A'}) and installs the impulsive depletion
//! Δn(x) = -[1 - exp(-γ_con/x)] × n_pl(x) at `z_start = z_res`, with the photon
//! mass equal to the plasma frequency, as in Chluba, Cyr & Johnson (2024). Its
//! `neutral_hydrogen` option (ADR 0008) instead installs
//! Δn(x) = -⟨1 - exp(-τ(x))⟩ × n_pl(x) at the same z_res, with τ from the full
//! photon mass below and ⟨·⟩ a grid-cell average (see [`conversion_probability`]
//! and [`cell_averaged_probability`]).
//!
//! # Photon effective mass
//!
//! A photon of energy ω in the primordial gas has the effective mass
//! (Caputo, Liu, Mishra-Sharma & Ruderman 2020, PRD 102, 103533, Eq. 1)
//!
//! m_γ²(z, ω) = ω_pl²(z) − 4π α_H n_HI(z) ω²,
//!
//! where ω_pl² = 4π α n_e / m_e is the free-electron plasma term and the
//! second term is the refraction of neutral hydrogen, (n − 1)_H = 2π α_H n_HI,
//! with static polarizability α_H = 4.5 a₀³ (so 4π α_H = 8.38×10⁻²⁴ cm³).
//! Helium is neglected, as in Caputo et al.: with α_He = 1.38 a₀³ and
//! n_He/n_H = 0.079 it would add about 2% to the neutral term. Mirizzi, Redondo
//! & Sigl (2009), Eq. 20, has the same structure with a coefficient about 20%
//! larger (a value for molecular hydrogen); Caputo et al. (2020, PRD footnote 1)
//! correct it. [`photon_mass_sq_ev2`] evaluates m_γ².
//!
//! The neutral term matters only after recombination, where 1 − X_H is not
//! small. There it can exceed the plasma term: at m_{A'} = 10⁻¹¹ eV (plasma-only
//! resonance z_res ≈ 668) it is 1.9× the plasma term for x = 4, so those photons
//! do not resonate at z_res at all. Chluba, Cyr & Johnson (2024), Sec. 2, set
//! m_γ = ω_pl and so miss this; the solver follows them by default so that its
//! limits compare like for like.
//!
//! # Conversion probability
//!
//! For dimensionless frequency x = ω/(k T_γ) (constant along a photon path),
//! every crossing z_i of m_γ²(z, x) = m_{A'}² converts with the Landau–Zener
//! probability in the narrow-width approximation (Mirizzi et al. 2009, Eq. 18;
//! Caputo et al. 2020, PRL 125, 221303, Eq. 3):
//!
//! P_i(x) = π ε² m_{A'}² / (ω_i H_i |d ln m_γ²/d ln a|_i),   ω_i = x k T_γ(z_i),
//!
//! with the derivative at fixed x. Crossings add incoherently, so the photon
//! survival is exp(−τ(x)) with τ(x) = Σ_i P_i(x). Without the neutral term each
//! x has the single crossing z_res and τ(x) = γ_con/x, the Chluba, Cyr & Johnson
//! (2024) Eq. 6 result that [`gamma_con`] computes.
//!
//! [`gamma_con`] and [`resonance_redshift`] stay plasma-only. They give the
//! nominal resonance used for reporting and range warnings, and the axion
//! module reuses them unchanged; with `neutral_hydrogen` the depletion itself
//! uses the full m_γ².
//! z_res is also where the solver installs the depletion: the neutral term only
//! lowers m_γ², so a crossing needs ω_pl(z) ≥ m, and ω_pl ∝ [X_e (1+z)³]^(1/2)
//! grows with z. Every crossing therefore lies at z ≥ z_res, and z_res is the
//! latest any frequency converts. Low-x photons cross at z_res itself; higher-x
//! photons cross earlier, and installing them at z_res misses only the Compton
//! redistribution in between, y ≈ 1.5×10⁻⁶ from z = 1189 to 225 (ADR 0008).
//!
//! Where m_γ² touches m² tangentially, two crossings merge and the
//! narrow-width P_i diverges as (x_t − x)^(−1/2). The true conversion there is
//! finite (a uniform stationary-phase treatment), which is not attempted here.
//! [`cell_averaged_probability`] integrates the divergence over grid cells, so
//! the depletion no longer depends on where grid points fall; it is still the
//! narrow-width value.
//!
//! References:
//! - Mirizzi, Redondo & Sigl (2009), JCAP 0903, 026
//! - Caputo, Liu, Mishra-Sharma & Ruderman (2020), PRD 102, 103533; PRL 125, 221303
//! - Chluba, Cyr & Johnson (2024), MNRAS 535, 1874
//! - Arsenadze et al. (2025), JHEP 03, 018

use crate::constants::*;
use crate::cosmology::Cosmology;
use crate::recombination::{RecombinationHistory, helium_electron_fraction, ionization_fraction};

/// Bohr radius a₀, in m (CODATA 2018).
pub const BOHR_RADIUS: f64 = 5.291_772_109_03e-11;

/// Static electric polarizability of ground-state hydrogen, in units of a₀³
/// (exact nonrelativistic value 9/2).
pub const HYDROGEN_POLARIZABILITY_A0_CUBED: f64 = 4.5;

/// Redshift band searched for resonances by [`resonance_redshift`] and
/// [`conversion_probability`].
const Z_SCAN_MIN: f64 = 10.0;
const Z_SCAN_MAX: f64 = 3.0e7;

/// Number of log-spaced redshifts in the crossing scan of
/// [`conversion_probability`]. The spacing is Δln z = ln(3×10⁶)/2999 ≈ 0.005.
const N_Z_SCAN: usize = 3000;

/// Computes the photon plasma frequency ω_pl (in eV) at redshift `z`.
///
/// ω_pl² = 4π α n_e ℏ c / m_e, with n_e = X_e(z) × n_H(z).
pub fn plasma_frequency_ev(z: f64, cosmo: &Cosmology) -> f64 {
    let hbar_ev_s = HBAR / EV_IN_JOULES;
    let x_e = ionization_fraction(z, cosmo);
    let n_e = cosmo.n_e(z, x_e);
    let factor = 4.0 * std::f64::consts::PI * ALPHA_FS * HBAR * C_LIGHT / M_ELECTRON;
    hbar_ev_s * (n_e * factor).sqrt()
}

/// Finds the resonance redshift `z_res` where ω_pl(z_res) = m.
///
/// Returns `None` when `m` is outside the range spanned by ω_pl on
/// `[z_min, z_max] = [10, 3e7]`.
pub fn resonance_redshift(m_ev: f64, cosmo: &Cosmology) -> Option<f64> {
    let z_min = Z_SCAN_MIN;
    let z_max = Z_SCAN_MAX;
    let f = |z: f64| plasma_frequency_ev(z, cosmo) - m_ev;
    let (f_lo, f_hi) = (f(z_min), f(z_max));
    if !f_lo.is_finite() || !f_hi.is_finite() || f_lo * f_hi > 0.0 {
        return None;
    }
    let mut lo = z_min;
    let mut hi = z_max;
    let mut f_lo = f_lo;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let f_mid = f(mid);
        if (hi - lo) / mid < 1e-8 {
            return Some(mid);
        }
        if f_lo * f_mid <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// Computes |d ln ω_pl² / d ln a| at redshift `z` using a centered finite difference in z.
pub fn dln_omega_pl_sq_dlna(z: f64, cosmo: &Cosmology) -> f64 {
    let dz = (z * 1.0e-4).max(0.1);
    let x_e = ionization_fraction(z, cosmo);
    if x_e <= 1e-30 {
        return 3.0;
    }
    let x_e_plus = ionization_fraction(z + dz, cosmo);
    let x_e_minus = ionization_fraction(z - dz, cosmo);
    let dlnxe_dz = (x_e_plus - x_e_minus) / (2.0 * dz * x_e);
    ((1.0 + z) * dlnxe_dz + 3.0).abs()
}

/// Computes the narrow-width approximation dark-photon conversion parameter γ_con
/// (dimensionless).
///
/// γ_con = π ε² m² / (|d ln ω_pl²/d ln a|_{z_res} × T_γ(z_res) × H(z_res)),
/// following Chluba, Cyr & Johnson (2024), MNRAS 535, 1874, Eq. 6. Returns
/// `None` if no resonance exists in the supported redshift range.
///
/// This is the plasma-only result (m_γ = ω_pl), which the solver's dark-photon
/// depletion uses by default. With the `neutral_hydrogen` option the depletion
/// uses the full photon mass through [`conversion_probability`], which reduces to τ(x) = γ_con/x wherever neutral hydrogen is negligible at
/// the crossing (z_res ≳ 2000, or x ≪ 1).
///
/// Returned tuple: `(gamma_con, z_res)`.
pub fn gamma_con(epsilon: f64, m_ev: f64, cosmo: &Cosmology) -> Option<(f64, f64)> {
    let z_res = resonance_redshift(m_ev, cosmo)?;
    let t_cmb_ev = K_BOLTZMANN * cosmo.t_cmb * (1.0 + z_res) / EV_IN_JOULES;
    let hbar_ev_s = HBAR / EV_IN_JOULES;
    let h_ev = hbar_ev_s * cosmo.hubble(z_res);
    let d = dln_omega_pl_sq_dlna(z_res, cosmo);
    let gc = std::f64::consts::PI * epsilon * epsilon * m_ev * m_ev / (d * t_cmb_ev * h_ev);
    Some((gc, z_res))
}

/// Coefficient 4π α_H of the neutral-hydrogen term, in m³.
fn neutral_coefficient_m3() -> f64 {
    4.0 * std::f64::consts::PI * HYDROGEN_POLARIZABILITY_A0_CUBED * BOHR_RADIUS.powi(3)
}

/// Neutral hydrogen fraction 1 − X_H, with X_H = X_e − x_He clamped to [0, 1].
///
/// X_e counts helium electrons too, so it exceeds 1 before helium recombines;
/// subtracting the Saha helium contribution gives the hydrogen fraction.
fn neutral_fraction(z: f64, x_e: f64, cosmo: &Cosmology) -> f64 {
    let x_h = (x_e - helium_electron_fraction(z, cosmo)).clamp(0.0, 1.0);
    1.0 - x_h
}

/// Computes the photon effective mass squared m_γ²(z, x), in eV².
///
/// m_γ² = ω_pl²(z) − 4π α_H n_HI(z) ω², with ω = x k T_γ(z) and
/// n_HI = (1 − X_H) n_H (Caputo et al. 2020, PRD 102, 103533, Eq. 1). The
/// product 4π α_H n_HI is dimensionless, so both terms are in eV². The result
/// is negative where neutral hydrogen dominates. `x` is the dimensionless
/// frequency hν/(k T_γ), which is constant along a photon path.
///
/// X_e comes from the tabulated [`RecombinationHistory`], the same source as
/// [`conversion_probability`] and the Python mirror. Building the table costs
/// about as much as one [`ionization_fraction`] call below z ≈ 1500; the two
/// differ by ~10⁻⁵ after recombination.
pub fn photon_mass_sq_ev2(z: f64, x: f64, cosmo: &Cosmology) -> f64 {
    let x_e = RecombinationHistory::new(cosmo).x_e(z);
    let bg = Background::at(z, x_e, cosmo);
    bg.omega_pl_sq - bg.neutral_per_x_sq * x * x
}

/// x-independent pieces of m_γ² and of the conversion probability at one redshift.
#[derive(Clone, Copy)]
struct Background {
    /// ω_pl², in eV².
    omega_pl_sq: f64,
    /// 4π α_H n_HI (k T_γ)², in eV²; the neutral term is this times x².
    neutral_per_x_sq: f64,
}

impl Background {
    fn at(z: f64, x_e: f64, cosmo: &Cosmology) -> Self {
        let hbar_ev_s = HBAR / EV_IN_JOULES;
        let n_h = cosmo.n_h(z);
        let omega_pl_sq = hbar_ev_s
            * hbar_ev_s
            * (x_e * n_h)
            * (4.0 * std::f64::consts::PI * ALPHA_FS * HBAR * C_LIGHT / M_ELECTRON);
        let t_ev = K_BOLTZMANN * cosmo.t_cmb * (1.0 + z) / EV_IN_JOULES;
        let neutral_per_x_sq =
            neutral_coefficient_m3() * neutral_fraction(z, x_e, cosmo) * n_h * t_ev * t_ev;
        Background {
            omega_pl_sq,
            neutral_per_x_sq,
        }
    }
}

/// Per-frequency result of [`conversion_probability`].
#[derive(Debug, Clone)]
pub struct Conversion {
    /// Conversion depth τ(x) = Σ_i P_i(x), summed over crossings (dimensionless).
    pub tau: Vec<f64>,
    /// Total conversion probability 1 − exp(−τ(x)); the depletion is
    /// Δn(x) = −probability × n_pl(x).
    pub probability: Vec<f64>,
    /// Crossing redshifts for each x, in decreasing order. Empty when the
    /// photon never resonates in z ∈ [10, 3×10⁷].
    pub crossings: Vec<Vec<f64>>,
}

/// Evaluates X_e from the cached recombination table, the same history as
/// [`ionization_fraction`] without re-integrating the Peebles ODE per call.
struct CrossingEvaluator<'a> {
    cosmo: &'a Cosmology,
    recomb: RecombinationHistory,
}

impl<'a> CrossingEvaluator<'a> {
    fn new(cosmo: &'a Cosmology) -> Self {
        CrossingEvaluator {
            cosmo,
            recomb: RecombinationHistory::new(cosmo),
        }
    }

    fn background(&self, z: f64) -> Background {
        Background::at(z, self.recomb.x_e(z), self.cosmo)
    }

    fn mass_sq(&self, z: f64, x: f64) -> f64 {
        let bg = self.background(z);
        bg.omega_pl_sq - bg.neutral_per_x_sq * x * x
    }

    /// Bisects m_γ²(z, x) = m² on [lo, hi], a bracket with a sign change.
    fn bisect(&self, x: f64, m_sq: f64, mut lo: f64, mut hi: f64) -> f64 {
        let mut f_lo = self.mass_sq(lo, x) - m_sq;
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if (hi - lo) <= 1e-12 * mid {
                break;
            }
            let f_mid = self.mass_sq(mid, x) - m_sq;
            if (f_lo < 0.0) == (f_mid < 0.0) {
                lo = mid;
                f_lo = f_mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Returns |d ln m_γ² / d ln a| at a crossing z (where m_γ² = m²), at fixed x.
    ///
    /// Differentiates each term in ln(1+z): the plasma term scales as
    /// X_e (1+z)³ and the neutral term as (1 − X_H)(1+z)⁵ at fixed x, so
    /// d m_γ²/d ln(1+z) = ω_pl² [3 + (1+z) X_e'/X_e] − N x² [5 + (1+z) f_n'/f_n]
    /// with N = 4π α_H n_HI (k T_γ)² and f_n = 1 − X_H. The X_e and f_n slopes
    /// are centered differences with the step of [`dln_omega_pl_sq_dlna`], so the
    /// plasma-only limit reproduces [`gamma_con`].
    fn dln_mass_sq_dlna(&self, z: f64, x: f64, m_sq: f64) -> f64 {
        let dz = (z * 1.0e-4).max(0.1);
        let x_e = self.recomb.x_e(z);
        let x_e_p = self.recomb.x_e(z + dz);
        let x_e_m = self.recomb.x_e(z - dz);
        let bg = self.background(z);
        let opz = 1.0 + z;
        let dxe_dz = (x_e_p - x_e_m) / (2.0 * dz);
        let plasma = bg.omega_pl_sq * 3.0 + bg.omega_pl_sq * opz * dxe_dz / x_e;
        // Neutral term: N = c f_n (1+z)⁵ x² → dN/d ln(1+z) = 5N + c' (1+z) f_n',
        // written without dividing by f_n so a clamped f_n = 0 is safe.
        let f_n_p = neutral_fraction(z + dz, x_e_p, self.cosmo);
        let f_n_m = neutral_fraction(z - dz, x_e_m, self.cosmo);
        let dfn_dz = (f_n_p - f_n_m) / (2.0 * dz);
        let neutral = bg.neutral_per_x_sq * x * x;
        let t_ev = K_BOLTZMANN * self.cosmo.t_cmb * opz / EV_IN_JOULES;
        let neutral_per_fn = neutral_coefficient_m3() * self.cosmo.n_h(z) * t_ev * t_ev * x * x;
        let neutral_slope = 5.0 * neutral + neutral_per_fn * opz * dfn_dz;
        ((plasma - neutral_slope) / m_sq).abs()
    }
}

/// Crossing finder for one (ε, m): the x-independent redshift scan is built once.
struct Scanner<'a> {
    eval: CrossingEvaluator<'a>,
    z_scan: Vec<f64>,
    bg_scan: Vec<Background>,
    m_sq: f64,
    eps_sq: f64,
}

impl<'a> Scanner<'a> {
    fn new(epsilon: f64, m_ev: f64, cosmo: &'a Cosmology) -> Self {
        let eval = CrossingEvaluator::new(cosmo);
        let (l0, l1) = (Z_SCAN_MIN.ln(), Z_SCAN_MAX.ln());
        let z_scan: Vec<f64> = (0..N_Z_SCAN)
            .map(|i| (l0 + (l1 - l0) * i as f64 / (N_Z_SCAN - 1) as f64).exp())
            .collect();
        let bg_scan: Vec<Background> = z_scan.iter().map(|&z| eval.background(z)).collect();
        Scanner {
            eval,
            z_scan,
            bg_scan,
            m_sq: m_ev * m_ev,
            eps_sq: epsilon * epsilon,
        }
    }

    /// Returns every crossing of m_γ²(z, x) = m² in z ∈ [10, 3×10⁷], in
    /// decreasing order.
    ///
    /// Sign changes on the scan give most crossings. A pair closer than one
    /// scan step (m_γ² barely dipping through m²) shows up instead as a local
    /// extremum of m_γ² − m² whose scan neighbors share its sign; golden-section
    /// search refines each such extremum, and if it crosses zero both roots are
    /// bisected. Only two extrema within one scan step can still hide a pair.
    fn crossings(&self, x: f64) -> Vec<f64> {
        let x_sq = x * x;
        let m_sq = self.m_sq;
        let f: Vec<f64> = self
            .bg_scan
            .iter()
            .map(|bg| bg.omega_pl_sq - bg.neutral_per_x_sq * x_sq - m_sq)
            .collect();
        let z = &self.z_scan;
        let mut zs: Vec<f64> = Vec::new();
        for i in 1..N_Z_SCAN {
            if (f[i - 1] < 0.0) != (f[i] < 0.0) {
                zs.push(self.eval.bisect(x, m_sq, z[i - 1], z[i]));
            }
        }
        for i in 1..N_Z_SCAN - 1 {
            let (a, b, c) = (f[i - 1], f[i], f[i + 1]);
            let same_sign = (a < 0.0) == (b < 0.0) && (b < 0.0) == (c < 0.0);
            // A minimum above zero or a maximum below zero: the extremum points
            // towards zero and may cross it between scan points.
            let toward_zero = if b > 0.0 {
                b <= a && b <= c
            } else {
                b >= a && b >= c
            };
            if !same_sign || !toward_zero {
                continue;
            }
            let s = if b > 0.0 { 1.0 } else { -1.0 };
            let g = |zz: f64| s * (self.eval.mass_sq(zz, x) - m_sq);
            let (mut lo, mut hi) = (z[i - 1], z[i + 1]);
            let r = 0.5 * (5.0_f64.sqrt() - 1.0);
            let mut c1 = hi - r * (hi - lo);
            let mut c2 = lo + r * (hi - lo);
            let (mut g1, mut g2) = (g(c1), g(c2));
            for _ in 0..100 {
                if (hi - lo) <= 1e-12 * hi {
                    break;
                }
                if g1 < g2 {
                    hi = c2;
                    c2 = c1;
                    g2 = g1;
                    c1 = hi - r * (hi - lo);
                    g1 = g(c1);
                } else {
                    lo = c1;
                    c1 = c2;
                    g1 = g2;
                    c2 = lo + r * (hi - lo);
                    g2 = g(c2);
                }
            }
            let (z_e, g_e) = if g1 < g2 { (c1, g1) } else { (c2, g2) };
            if g_e < 0.0 {
                zs.push(self.eval.bisect(x, m_sq, z[i - 1], z_e));
                zs.push(self.eval.bisect(x, m_sq, z_e, z[i + 1]));
            }
        }
        zs.sort_by(|p, q| q.total_cmp(p));
        zs
    }

    /// Returns τ(x) = Σ P_i over the given crossings.
    fn tau(&self, x: f64, zs: &[f64]) -> f64 {
        let hbar_ev_s = HBAR / EV_IN_JOULES;
        let cosmo = self.eval.cosmo;
        zs.iter()
            .map(|&z| {
                let omega = x * K_BOLTZMANN * cosmo.t_cmb * (1.0 + z) / EV_IN_JOULES;
                let h_ev = hbar_ev_s * cosmo.hubble(z);
                let d = self.eval.dln_mass_sq_dlna(z, x, self.m_sq);
                std::f64::consts::PI * self.eps_sq * self.m_sq / (omega * h_ev * d)
            })
            .sum()
    }

    /// Returns (τ(x), number of crossings).
    fn at(&self, x: f64) -> (f64, usize) {
        let zs = self.crossings(x);
        (self.tau(x, &zs), zs.len())
    }

    /// Bisects x between `lo` (with `c_lo` crossings) and `hi` (with a different
    /// count) to the point where the count changes.
    fn bisect_count(&self, mut lo: f64, mut hi: f64, c_lo: usize) -> f64 {
        for _ in 0..100 {
            if hi - lo <= 1e-13 * hi {
                break;
            }
            let mid = 0.5 * (lo + hi);
            if self.at(mid).1 == c_lo {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Estimates the frequencies x_t of tangent crossings, independent of any x grid.
    ///
    /// With f = P(z) − N(z) x² − m² (P = ω_pl², N x² the neutral term), a
    /// tangency has f = 0 and ∂f/∂z = 0, so x² = P'/N' and
    /// h(z) = P − N P'/N' − m² = 0. The roots of h on the scan give each
    /// tangency's redshift and x_t = (P'/N')^(1/2). Derivatives are centered
    /// differences on the scan, so the estimates are approximate; callers
    /// confirm them by a change in the crossing count.
    fn tangency_estimates(&self) -> Vec<f64> {
        let bg = &self.bg_scan;
        let mut rh: Vec<Option<(f64, f64)>> = vec![None; N_Z_SCAN];
        for k in 1..N_Z_SCAN - 1 {
            let dp = bg[k + 1].omega_pl_sq - bg[k - 1].omega_pl_sq;
            let dn = bg[k + 1].neutral_per_x_sq - bg[k - 1].neutral_per_x_sq;
            if dn != 0.0 && dp / dn > 0.0 {
                let r = dp / dn;
                let h = bg[k].omega_pl_sq - bg[k].neutral_per_x_sq * r - self.m_sq;
                rh[k] = Some((r, h));
            }
        }
        let mut out = Vec::new();
        for k in 1..N_Z_SCAN - 2 {
            if let (Some((r0, h0)), Some((r1, h1))) = (rh[k], rh[k + 1])
                && (h0 < 0.0) != (h1 < 0.0)
            {
                let t = h0 / (h0 - h1);
                out.push((r0 + t * (r1 - r0)).sqrt());
            }
        }
        out.sort_by(f64::total_cmp);
        out
    }
}

/// Computes the per-frequency dark-photon conversion depth and probability,
/// including the neutral-hydrogen term in the photon mass, at the points `x_grid`.
///
/// For each x, finds every crossing z_i of m_γ²(z, x) = m_{A'}² in
/// z ∈ [10, 3×10⁷] and sums the Landau–Zener probabilities
/// P_i = π ε² m² / (ω_i H_i |d ln m_γ²/d ln a|_i), ω_i = x k T_γ(z_i)
/// (module docs). All quantities are in eV (H as ħH), so P_i is dimensionless.
///
/// The crossings come from a log-spaced scan of 3000 redshifts
/// (Δln z ≈ 0.005), refined by bisection to 10⁻¹² in z, plus a golden-section
/// check of every scan extremum for pairs closer than one step.
///
/// These are point values. Near a tangency x_t, where two crossings merge,
/// |d ln m_γ²/d ln a| → 0 and the narrow-width P_i diverges as
/// (x_t − x)^(−1/2). The divergence is integrable but makes point values on a
/// grid depend on where the grid points fall; the solver uses
/// [`cell_averaged_probability`] instead. The true conversion at a tangency
/// is finite (a uniform stationary-phase, Airy-type treatment), which neither
/// function attempts.
///
/// Without neutral hydrogen, every x has the single crossing z_res and
/// τ(x) = γ_con/x ([`gamma_con`]).
pub fn conversion_probability(
    epsilon: f64,
    m_ev: f64,
    x_grid: &[f64],
    cosmo: &Cosmology,
) -> Conversion {
    let scanner = Scanner::new(epsilon, m_ev, cosmo);
    let n = x_grid.len();
    let mut tau = Vec::with_capacity(n);
    let mut probability = Vec::with_capacity(n);
    let mut crossings = Vec::with_capacity(n);
    for &x in x_grid {
        let zs = scanner.crossings(x);
        let t = scanner.tau(x, &zs);
        tau.push(t);
        probability.push(-(-t).exp_m1());
        crossings.push(zs);
    }
    Conversion {
        tau,
        probability,
        crossings,
    }
}

/// Midpoint sub-points per cell for the cell average away from tangencies.
const N_SUB_CELL: usize = 4;
/// Midpoints in u = |x − x_t|^(1/2) on each piece of a cell near a tangency.
const N_SUB_TANGENT: usize = 32;
/// Cells within this many of their own widths of a tangency are integrated in u.
const TANGENT_REACH: f64 = 2.0;

/// Cell averages of the conversion depth and probability on a grid.
#[derive(Debug, Clone)]
pub struct CellAverage {
    /// Cell average of τ(x) (dimensionless); the linear-regime template.
    pub tau: Vec<f64>,
    /// Cell average of 1 − exp(−τ(x)); the depletion is −probability × n_pl(x_i).
    pub probability: Vec<f64>,
}

/// Averages the conversion depth and probability over each grid cell.
///
/// Cell i spans the midpoints to its neighbors, [x_{i−1/2}, x_{i+1/2}], with
/// half cells at the two ends; `x_grid` must be ascending. Away from
/// tangencies the average is the mean of 4 midpoint sub-samples. A tangency
/// x_t (where the number of crossings changes) is located by bisection on the
/// crossing count between sub-samples. Every cell within 2 cell widths of one is
/// split at x_t and integrated with 32 midpoints in u = |x − x_t|^(1/2), which
/// turns the (x_t − x)^(−1/2) divergence of the narrow-width approximation into
/// a smooth integrand. The result no longer depends on where grid points fall
/// relative to x_t (ADR 0008); it is still the narrow-width value, which
/// overstates the true, finite conversion at a tangency.
///
/// The two end half cells are not centered on their nodes, so a sloped τ is
/// biased there by about Δx/(4x); the solver grid ends (x_min, x_max) lie far
/// outside the observable band. Tangencies come from count changes between
/// sub-samples and from the grid-independent tangency curve (x² = P'/N' where
/// f = ∂f/∂z = 0), so a band of extra crossings narrower than a sub-sample is
/// still found, unless its two ends lie closer than the curve's
/// finite-difference accuracy. That can happen only for masses just below the
/// largest mass with a tangency, about 1.51×10⁻¹² eV.
pub fn cell_averaged_probability(
    epsilon: f64,
    m_ev: f64,
    x_grid: &[f64],
    cosmo: &Cosmology,
) -> CellAverage {
    cell_average_with(
        epsilon,
        m_ev,
        x_grid,
        cosmo,
        N_SUB_CELL,
        N_SUB_TANGENT,
        TANGENT_REACH,
    )
}

/// [`cell_averaged_probability`] with explicit sub-sample counts and reach
/// (convergence studies and tests).
fn cell_average_with(
    epsilon: f64,
    m_ev: f64,
    x_grid: &[f64],
    cosmo: &Cosmology,
    n_sub_cell: usize,
    n_sub_tangent: usize,
    tangent_reach: f64,
) -> CellAverage {
    assert!(
        n_sub_cell >= 1 && n_sub_tangent >= 1,
        "cell averaging needs at least one sub-sample"
    );
    let n = x_grid.len();
    let scanner = Scanner::new(epsilon, m_ev, cosmo);
    let p_of = |t: f64| -(-t).exp_m1();
    if n < 2 {
        let tau: Vec<f64> = x_grid.iter().map(|&x| scanner.at(x).0).collect();
        let probability = tau.iter().map(|&t| p_of(t)).collect();
        return CellAverage { tau, probability };
    }
    assert!(
        x_grid.windows(2).all(|w| w[1] >= w[0]),
        "cell_averaged_probability: x_grid must be ascending"
    );

    let mut edges = Vec::with_capacity(n + 1);
    edges.push(x_grid[0]);
    for w in x_grid.windows(2) {
        edges.push(0.5 * (w[0] + w[1]));
    }
    edges.push(x_grid[n - 1]);

    // Base sub-samples: (x, τ, crossing count), in ascending x.
    let mut samples: Vec<(f64, f64, usize)> = Vec::with_capacity(n * n_sub_cell);
    for i in 0..n {
        let (a, b) = (edges[i], edges[i + 1]);
        for k in 0..n_sub_cell {
            let x = a + (k as f64 + 0.5) * (b - a) / n_sub_cell as f64;
            let (t, c) = scanner.at(x);
            samples.push((x, t, c));
        }
    }

    // Tangencies, from two sources: count changes between sub-samples, and the
    // tangency curve (Scanner::tangency_estimates), which also catches a band
    // of extra crossings narrower than the sub-sample spacing.
    let mut tangencies: Vec<f64> = Vec::new();
    for w in samples.windows(2) {
        let ((lo, _, c_lo), (hi, _, c_hi)) = (w[0], w[1]);
        if c_lo != c_hi && hi > lo {
            tangencies.push(scanner.bisect_count(lo, hi, c_lo));
        }
    }
    let estimates: Vec<f64> = scanner
        .tangency_estimates()
        .into_iter()
        .filter(|&x| x > x_grid[0] && x < x_grid[n - 1])
        .collect();
    for (j, &x_e) in estimates.iter().enumerate() {
        // Bracket one estimate only: stay within 0.4 of the gap to its neighbors.
        let mut half = 1e-3 * x_e;
        if j > 0 {
            half = half.min(0.4 * (x_e - estimates[j - 1]));
        }
        if j + 1 < estimates.len() {
            half = half.min(0.4 * (estimates[j + 1] - x_e));
        }
        let (lo, hi) = (x_e - half, x_e + half);
        let (c_lo, c_hi) = (scanner.at(lo).1, scanner.at(hi).1);
        if c_lo != c_hi {
            let x_t = scanner.bisect_count(lo, hi, c_lo);
            if tangencies.iter().all(|&t| (t - x_t).abs() > 1e-9 * x_t) {
                tangencies.push(x_t);
            }
        }
    }
    tangencies.sort_by(f64::total_cmp);

    let mut tau = Vec::with_capacity(n);
    let mut probability = Vec::with_capacity(n);
    for i in 0..n {
        let (a, b) = (edges[i], edges[i + 1]);
        let width = b - a;
        let base = &samples[i * n_sub_cell..(i + 1) * n_sub_cell];
        let reach = tangent_reach * width;
        let dist = |xt: f64| (a - xt).max(xt - b).max(0.0);
        let near = width > 0.0 && tangencies.iter().any(|&xt| dist(xt) <= reach);
        if !near {
            let nb = n_sub_cell as f64;
            tau.push(base.iter().map(|s| s.1).sum::<f64>() / nb);
            probability.push(base.iter().map(|s| p_of(s.1)).sum::<f64>() / nb);
            continue;
        }
        // Split at interior tangencies. Each piece may diverge at either end,
        // so integrate each half in u = |x − x_t|^(1/2) about the tangency on
        // its own side; a piece with a tangency on one side only uses that one.
        let mut cuts = vec![a];
        cuts.extend(tangencies.iter().copied().filter(|&xt| xt > a && xt < b));
        cuts.push(b);
        let (mut int_tau, mut int_p) = (0.0, 0.0);
        let mut add = |x: f64, weight: f64| {
            let t = scanner.at(x).0;
            int_tau += t * weight;
            int_p += p_of(t) * weight;
        };
        for piece in cuts.windows(2) {
            let (p, q) = (piece[0], piece[1]);
            let left = tangencies
                .iter()
                .copied()
                .filter(|&t| t <= p && p - t <= reach)
                .fold(None, |acc: Option<f64>, t| {
                    Some(acc.map_or(t, |m| m.max(t)))
                });
            let right = tangencies
                .iter()
                .copied()
                .filter(|&t| t >= q && t - q <= reach)
                .fold(None, |acc: Option<f64>, t| {
                    Some(acc.map_or(t, |m| m.min(t)))
                });
            let segments: Vec<(f64, f64, Option<f64>)> = match (left, right) {
                (Some(l), Some(r)) => {
                    let mid = 0.5 * (p + q);
                    vec![(p, mid, Some(l)), (mid, q, Some(r))]
                }
                (Some(l), None) => vec![(p, q, Some(l))],
                (None, Some(r)) => vec![(p, q, Some(r))],
                (None, None) => vec![(p, q, None)],
            };
            for (s0, s1, origin) in segments {
                match origin {
                    Some(x_o) => {
                        let above = x_o <= s0;
                        let (u0, u1) = if above {
                            ((s0 - x_o).sqrt(), (s1 - x_o).sqrt())
                        } else {
                            ((x_o - s1).sqrt(), (x_o - s0).sqrt())
                        };
                        let du = (u1 - u0) / n_sub_tangent as f64;
                        for k in 0..n_sub_tangent {
                            let u = u0 + (k as f64 + 0.5) * du;
                            let x = if above { x_o + u * u } else { x_o - u * u };
                            add(x, 2.0 * u * du);
                        }
                    }
                    None => {
                        let dx = (s1 - s0) / n_sub_tangent as f64;
                        for k in 0..n_sub_tangent {
                            add(s0 + (k as f64 + 0.5) * dx, dx);
                        }
                    }
                }
            }
        }
        tau.push(int_tau / width);
        probability.push(int_p / width);
    }
    CellAverage { tau, probability }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn plasma_frequency_matches_first_principles() {
        // Validates ω_pl = ℏ √(4πα n_e / m_e) against (a) an independent
        // first-principles computation using only CODATA constants + X_e, and
        // (b) the redshift scaling ω_pl ∝ (1+z)^{3/2} in the fully-ionized era.
        //
        // Reference: Mirizzi, Redondo & Sigl (2009), JCAP 0903, 026 Eq. 2.
        let cosmo = Cosmology::default();

        // Independent recomputation at z=1e5 (fully ionized, X_e≈1).
        let z = 1.0e5_f64;
        let x_e = ionization_fraction(z, &cosmo);
        assert!(x_e > 0.99, "z=1e5 should be fully ionized, got X_e={x_e}");
        let n_e = cosmo.n_e(z, x_e);
        let hbar_ev_s = HBAR / EV_IN_JOULES;
        let expected = hbar_ev_s
            * (n_e * 4.0 * std::f64::consts::PI * ALPHA_FS * HBAR * C_LIGHT / M_ELECTRON).sqrt();
        let omega = plasma_frequency_ev(z, &cosmo);
        let rel_err = (omega - expected).abs() / expected;
        assert!(
            rel_err < 1e-12,
            "ω_pl(1e5) = {omega:.4e} vs first-principles {expected:.4e}, rel_err={rel_err:.2e}"
        );

        // Verify (1+z)^{3/2} scaling in fully-ionized era where n_e ∝ (1+z)³:
        // ω_pl(z2) / ω_pl(z1) = [(1+z2)/(1+z1)]^{3/2}
        let z1 = 5.0e4_f64;
        let z2 = 2.0e5_f64;
        let ratio = plasma_frequency_ev(z2, &cosmo) / plasma_frequency_ev(z1, &cosmo);
        let expected_ratio = ((1.0 + z2) / (1.0 + z1)).powf(1.5);
        let scale_err = (ratio - expected_ratio).abs() / expected_ratio;
        assert!(
            scale_err < 1e-6,
            "ω_pl(1+z)^1.5 scaling violated: ratio={ratio:.6}, expected={expected_ratio:.6}"
        );
    }

    #[test]
    fn resonance_round_trip() {
        let cosmo = Cosmology::default();
        for m_ev in [3e-8, 1e-7, 1e-6, 1e-5, 1e-4] {
            let z_res = resonance_redshift(m_ev, &cosmo).expect("no resonance");
            let omega = plasma_frequency_ev(z_res, &cosmo);
            assert_relative_eq!(omega, m_ev, max_relative = 1e-5);
        }
    }

    #[test]
    fn gamma_con_scales_as_epsilon_squared() {
        let cosmo = Cosmology::default();
        let (g1, _) = gamma_con(1e-9, 1e-6, &cosmo).unwrap();
        let (g2, _) = gamma_con(2e-9, 1e-6, &cosmo).unwrap();
        assert_relative_eq!(g2 / g1, 4.0, max_relative = 1e-10);
    }

    #[test]
    fn gamma_con_matches_chluba_cyr() {
        // At m = 1e-7 eV the resonance sits in the fully-ionized era where
        // ω_pl ∝ (1+z)^{3/2}. With ω_pl(z=1e5) ≈ 5.50e-7 eV (see
        // plasma_frequency_matches_first_principles), setting ω_pl(z_res) = m:
        //   1+z_res = (1+1e5) × (1e-7 / 5.50e-7)^{2/3} ≈ 3.21e4
        //
        // For γ_con/ε², Chluba & Cyr (2024) Eq. 6 evaluated with this z_res,
        // default Planck cosmology, and X_e=1: expect ~2e10. Tolerance caps
        // parameter drift without being so tight that a switch from Planck
        // 2015→2018 ω_b would break the test.
        let cosmo = Cosmology::default();
        let (gc, z_res) = gamma_con(1.0, 1e-7, &cosmo).unwrap();
        assert!(
            (z_res - 3.21e4).abs() / 3.21e4 < 0.05,
            "z_res = {z_res:.3e}, expected ~3.21e4 (±5%) from ω_pl ∝ (1+z)^1.5"
        );
        // γ_con/ε² with default Planck cosmology and the resonance formula
        // gives 9.3e10 (measured); the tight 20% window catches a cosmology
        // parameter drift or prefactor bug, replacing the previous 100× window.
        assert!(
            (gc - 9.3e10).abs() / 9.3e10 < 0.2,
            "γ_con/ε² = {gc:.3e}, expected ~9.3e10 from Chluba & Cyr 2024 Eq. 6 (±20%)"
        );
    }
    // ---- Neutral-hydrogen photon mass (ADR 0008) --------------------------
    //
    // Targets below come from literal constants typed here (Bohr radius, ħc,
    // m_e c², α) and from an independent scan written in each test, never from
    // `conversion_probability` itself (CLAUDE.md pitfalls 9 and 11).

    /// Literal CODATA 2018 values, kept separate from `crate::constants`.
    const A0_M_LIT: f64 = 5.291_772_109_03e-11; // Bohr radius [m]
    const HBARC_EV_CM_LIT: f64 = 1.973_269_804e-5; // ħc [eV cm]
    const ME_EV_LIT: f64 = 510_998.950_00; // m_e c² [eV]
    const ALPHA_LIT: f64 = 7.297_352_5693e-3;

    /// Caputo et al. (2020) Eq. 1 with coefficients rebuilt from literal constants:
    /// m_γ² = [4πα (ħc)³/(m_e c²)] n_e − [4π · 4.5 a₀³] ω² n_HI, densities in cm⁻³.
    fn caputo_mass_sq(z: f64, x: f64, x_e: f64, cosmo: &Cosmology) -> f64 {
        let c_pl = 4.0 * std::f64::consts::PI * ALPHA_LIT * HBARC_EV_CM_LIT.powi(3) / ME_EV_LIT;
        let a0_cm = A0_M_LIT * 100.0;
        let c_n = 4.0 * std::f64::consts::PI * 4.5 * a0_cm.powi(3);
        let n_h_cm3 = cosmo.n_h(z) * 1e-6;
        let x_h = (x_e - helium_electron_fraction(z, cosmo)).clamp(0.0, 1.0);
        let omega = x * 8.617_333_262e-5 * cosmo.t_cmb * (1.0 + z); // k_B [eV/K]
        c_pl * x_e * n_h_cm3 - c_n * omega * omega * (1.0 - x_h) * n_h_cm3
    }

    #[test]
    fn neutral_coefficient_from_bohr_radius() {
        // 4π · 4.5 a₀³ = 8.38×10⁻²⁴ cm³ (Caputo et al. 2020, Eq. 1 quotes 8.4e-24).
        let a0_cm = A0_M_LIT * 100.0;
        let expected_cm3 = 4.0 * std::f64::consts::PI * 4.5 * a0_cm.powi(3);
        assert!(
            (expected_cm3 - 8.38e-24).abs() / 8.38e-24 < 1e-3,
            "{expected_cm3:e}"
        );
        let code_cm3 = neutral_coefficient_m3() * 1e6;
        assert_relative_eq!(code_cm3, expected_cm3, max_relative = 1e-12);
    }

    #[test]
    fn photon_mass_matches_caputo_eq1() {
        // Caputo et al. (2020), Eq. 1: 1.4e-21 eV² per n_e/cm³ and 8.4e-24 eV²
        // (ω/eV)² per n_HI/cm³; the literal-constant rebuild must match the
        // quoted rounded coefficients to their 2 significant figures.
        let c_pl = 4.0 * std::f64::consts::PI * ALPHA_LIT * HBARC_EV_CM_LIT.powi(3) / ME_EV_LIT;
        assert!(
            (c_pl - 1.4e-21).abs() / 1.4e-21 < 0.02,
            "plasma coefficient {c_pl:e}"
        );
        let cosmo = Cosmology::default();
        let recomb = RecombinationHistory::new(&cosmo);
        for &(z, x) in &[(300.0, 1.0), (668.0, 4.0), (1200.0, 10.0), (5.0e4, 3.0)] {
            let x_e = recomb.x_e(z);
            let expected = caputo_mass_sq(z, x, x_e, &cosmo);
            let got = photon_mass_sq_ev2(z, x, &cosmo);
            let scale = plasma_frequency_ev(z, &cosmo).powi(2);
            assert!(
                (got - expected).abs() < 1e-8 * scale.max(expected.abs()),
                "z={z} x={x}: code {got:e} vs Caputo Eq. 1 {expected:e}"
            );
        }
    }

    #[test]
    fn conversion_reduces_to_gamma_con_when_hydrogen_ionized() {
        // Plasma-only resonances far above recombination (1 − X_H < 10⁻¹⁰):
        // one crossing at z_res and τ(x) = γ_con/x exactly (CCJ24 Eq. 6).
        let cosmo = Cosmology::default();
        let x_grid = [0.01, 0.5, 1.0, 4.0, 10.0, 30.0];
        for m_ev in [1e-7, 1e-6, 1e-5] {
            let (gc, z_res) = gamma_con(1e-7, m_ev, &cosmo).unwrap();
            let conv = conversion_probability(1e-7, m_ev, &x_grid, &cosmo);
            for (i, &x) in x_grid.iter().enumerate() {
                assert_eq!(conv.crossings[i].len(), 1, "m={m_ev:e} x={x}");
                assert_relative_eq!(conv.crossings[i][0], z_res, max_relative = 1e-7);
                assert_relative_eq!(conv.tau[i], gc / x, max_relative = 1e-6);
                let p_ref = -(-gc / x).exp_m1();
                assert_relative_eq!(conv.probability[i], p_ref, max_relative = 1e-6);
            }
        }
    }

    #[test]
    fn neutral_hydrogen_moves_the_x4_resonance_at_1e_minus_11_ev() {
        // m = 1e-11 eV: plasma-only z_res ≈ 668. At x = 4 the neutral term is
        // ~1.9× the plasma term there, so m_γ² < 0 < m² at z_res and the photon
        // does not resonate there. The crossings and τ are rebuilt here from
        // Caputo Eq. 1 with an independent scan, bisection and finite-difference
        // slope.
        let cosmo = Cosmology::default();
        let m_ev = 1e-11;
        let x = 4.0;
        let eps = 1e-7;
        let z_res = resonance_redshift(m_ev, &cosmo).unwrap();
        assert!((z_res - 668.0).abs() < 5.0, "z_res = {z_res}");

        let recomb = RecombinationHistory::new(&cosmo);
        let m2 = |z: f64| caputo_mass_sq(z, x, recomb.x_e(z), &cosmo);
        let omega_pl_sq = plasma_frequency_ev(z_res, &cosmo).powi(2);
        let ratio = (omega_pl_sq - m2(z_res)) / omega_pl_sq;
        assert!(
            (1.7..2.1).contains(&ratio),
            "neutral/plasma at z_res = {ratio}"
        );
        assert!(m2(z_res) < 0.0);

        // Independent scan: 20000 log points over [10, 3e7].
        let n = 20_000;
        let zs: Vec<f64> = (0..n)
            .map(|i| (10f64.ln() + (3e7f64.ln() - 10f64.ln()) * i as f64 / (n - 1) as f64).exp())
            .collect();
        let target = m_ev * m_ev;
        let mut expected: Vec<f64> = Vec::new();
        for w in zs.windows(2) {
            let (fa, fb) = (m2(w[0]) - target, m2(w[1]) - target);
            if (fa < 0.0) != (fb < 0.0) {
                let (mut lo, mut hi, mut flo) = (w[0], w[1], fa);
                for _ in 0..100 {
                    let mid = 0.5 * (lo + hi);
                    let fm = m2(mid) - target;
                    if (flo < 0.0) == (fm < 0.0) {
                        lo = mid;
                        flo = fm;
                    } else {
                        hi = mid;
                    }
                }
                expected.push(0.5 * (lo + hi));
            }
        }
        expected.reverse();
        assert!(!expected.is_empty());
        for &z in &expected {
            assert!(
                z > 1.1 * z_res,
                "x=4 crossing at {z} should lie well above z_res"
            );
        }

        // τ = Σ π ε² m² / (ω H |d ln m_γ²/d ln a|), slope by a symmetric
        // difference in ln(1+z) of the literal-constant m_γ².
        let hbar_ev_s = 6.582_119_569e-16;
        let tau_expected: f64 = expected
            .iter()
            .map(|&z| {
                let h = 1e-4_f64;
                let (zp, zm) = ((1.0 + z) * h.exp() - 1.0, (1.0 + z) * (-h).exp() - 1.0);
                let dln = ((m2(zp) - m2(zm)) / (2.0 * h) / target).abs();
                let omega = x * 8.617_333_262e-5 * cosmo.t_cmb * (1.0 + z);
                let h_ev = hbar_ev_s * cosmo.hubble(z);
                std::f64::consts::PI * eps * eps * target / (omega * h_ev * dln)
            })
            .sum();

        let conv = conversion_probability(eps, m_ev, &[x], &cosmo);
        assert_eq!(
            conv.crossings[0].len(),
            expected.len(),
            "{:?} vs {expected:?}",
            conv.crossings[0]
        );
        for (a, b) in conv.crossings[0].iter().zip(&expected) {
            assert_relative_eq!(*a, *b, max_relative = 1e-6);
        }
        assert_relative_eq!(conv.tau[0], tau_expected, max_relative = 2e-3);
    }

    #[test]
    fn every_crossing_lies_at_or_above_z_res() {
        // Property test: the neutral term only lowers m_γ², so a crossing needs
        // ω_pl(z) ≥ m, i.e. z ≥ z_res; and every x crosses at least once when
        // a plasma-only resonance exists.
        let cosmo = Cosmology::default();
        let x_grid: Vec<f64> = (0..60)
            .map(|i| 0.05 * (600f64).powf(i as f64 / 59.0))
            .collect();
        for m_ev in [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7] {
            let z_res = resonance_redshift(m_ev, &cosmo).unwrap();
            let conv = conversion_probability(1e-7, m_ev, &x_grid, &cosmo);
            for c in &conv.crossings {
                assert!(!c.is_empty(), "m={m_ev:e}: every x must resonate");
                let z_low = *c.last().unwrap();
                assert!(
                    z_low >= z_res * (1.0 - 1e-7),
                    "m={m_ev:e}: {z_low} < {z_res}"
                );
            }
        }
    }

    #[test]
    fn low_x_reduces_to_gamma_con_after_recombination() {
        // m = 1e-11 eV resonates at z_res ≈ 668, after recombination, where the
        // neutral fraction is ≈ 1. As x → 0 the neutral term (∝ x²) vanishes and
        // τ·x must approach the plasma-only CCJ24 Eq. 6 value, rebuilt here from
        // literal constants on the same tabulated X_e history (the direct Peebles
        // integration behind `gamma_con` differs from the table by ~6e-5 in this
        // slope). The residual must then scale as x², the neutral term's scaling.
        let cosmo = Cosmology::default();
        let (eps, m_ev) = (1e-7, 1e-11);
        let z_res = resonance_redshift(m_ev, &cosmo).unwrap();
        let recomb = RecombinationHistory::new(&cosmo);
        let xs = [1e-3, 1e-2];
        let conv = conversion_probability(eps, m_ev, &xs, &cosmo);
        let dev: Vec<f64> = xs
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                assert_eq!(conv.crossings[i].len(), 1);
                let z = conv.crossings[i][0];
                assert_relative_eq!(z, z_res, max_relative = 1e-5);
                let dz = (z * 1e-4).max(0.1);
                let xe = recomb.x_e(z);
                let slope = (recomb.x_e(z + dz) - recomb.x_e(z - dz)) / (2.0 * dz);
                let d = (3.0 + (1.0 + z) * slope / xe).abs();
                let t_ev = 8.617_333_262e-5 * cosmo.t_cmb * (1.0 + z);
                let h_ev = 6.582_119_569e-16 * cosmo.hubble(z);
                let gc = std::f64::consts::PI * eps * eps * m_ev * m_ev / (d * t_ev * h_ev);
                conv.tau[i] * x / gc - 1.0
            })
            .collect();
        assert!(dev[0].abs() < 1e-6, "x=1e-3: τx/γ_con − 1 = {:e}", dev[0]);
        assert!(dev[1] < 0.0, "neutral term must lower τ: {:e}", dev[1]);
        let ratio = dev[1] / dev[0];
        assert!(
            (85.0..115.0).contains(&ratio),
            "residual not ∝ x²: ratio {ratio}"
        );
    }

    /// FIRAS-band integral ∫_{1.2}^{11} x³ n_pl τ̄ dx with τ̄ constant over each cell.
    fn band_integral(x_grid: &[f64], tau_bar: &[f64]) -> f64 {
        let n = x_grid.len();
        let mut total = 0.0;
        for i in 0..n {
            let a = if i == 0 {
                x_grid[0]
            } else {
                0.5 * (x_grid[i - 1] + x_grid[i])
            };
            let b = if i + 1 == n {
                x_grid[n - 1]
            } else {
                0.5 * (x_grid[i] + x_grid[i + 1])
            };
            let (lo, hi) = (a.max(1.2), b.min(11.0));
            if hi <= lo {
                continue;
            }
            let k = 16;
            let h = (hi - lo) / k as f64;
            let w: f64 = (0..k)
                .map(|j| {
                    let x = lo + (j as f64 + 0.5) * h;
                    x.powi(3) / x.exp_m1() * h
                })
                .sum();
            total += w * tau_bar[i];
        }
        total
    }

    #[test]
    fn cell_average_is_independent_of_grid_offset_at_tangency() {
        // m = 1e-12 eV has three crossings for x in about [2.73, 2.98], with
        // P_i → ∞ as (x_t − x)^(−1/2) at both ends. Property test: shifting a
        // linear grid (Δx = 0.03) by fractions of a cell must not move the
        // cell-averaged FIRAS-band integral by more than 1%.
        let cosmo = Cosmology::default();
        let dx = 0.03;
        let vals: Vec<f64> = [0.0, 0.2, 0.4, 0.6, 0.8]
            .iter()
            .map(|&f| {
                let x: Vec<f64> = (0..400).map(|k| 0.5 + (k as f64 + f) * dx).collect();
                let avg = cell_averaged_probability(1e-7, 1e-12, &x, &cosmo);
                band_integral(&x, &avg.tau)
            })
            .collect();
        let (lo, hi) = vals
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });
        assert!(
            (hi - lo) / lo < 0.01,
            "offset spread {:.3e}: {vals:?}",
            (hi - lo) / lo
        );
    }

    /// Gauss–Legendre nodes and weights on [−1, 1] (Newton on P_n).
    fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| {
                let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
                let mut dp = 0.0;
                for _ in 0..100 {
                    let (mut p0, mut p1) = (1.0, x);
                    for k in 2..=n {
                        let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                        p0 = p1;
                        p1 = p2;
                    }
                    dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
                    let step = p1 / dp;
                    x -= step;
                    if step.abs() < 1e-15 {
                        break;
                    }
                }
                (x, 2.0 / ((1.0 - x * x) * dp * dp))
            })
            .collect()
    }

    #[test]
    fn cell_average_resolves_a_narrow_tangency_band() {
        // m = 1.49e-12 eV: the band with three crossings is only ~1.7e-3 wide
        // (x ≈ 2.663–2.665), narrower than a sub-sample on any practical grid,
        // and one cell holds both of its divergent ends. The target is an
        // independent reference: tangencies located here by a fine x scan of
        // the crossing count, and ∫_{1.2}^{11} x³ n_pl τ dx by Gauss–Legendre,
        // in u = |x − x_t|^(1/2) toward each tangency on its own side. Every
        // grid (two linear, one log; five offsets each) must match it to 1e-3.
        let cosmo = Cosmology::default();
        let (eps, m) = (1e-7, 1.49e-12);
        let count = |x: f64| conversion_probability(eps, m, &[x], &cosmo).crossings[0].len();
        let tau = |x: f64| conversion_probability(eps, m, &[x], &cosmo).tau[0];
        let w = |x: f64| x.powi(3) / x.exp_m1();

        let xs: Vec<f64> = (0..6000).map(|k| 2.60 + 2e-5 * k as f64).collect();
        let counts: Vec<usize> = conversion_probability(eps, m, &xs, &cosmo)
            .crossings
            .iter()
            .map(|c| c.len())
            .collect();
        let mut ts = Vec::new();
        for k in 0..xs.len() - 1 {
            if counts[k] != counts[k + 1] {
                let (mut lo, mut hi) = (xs[k], xs[k + 1]);
                for _ in 0..45 {
                    let mid = 0.5 * (lo + hi);
                    if count(mid) == counts[k] {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                ts.push(0.5 * (lo + hi));
            }
        }
        assert_eq!(
            ts.len(),
            2,
            "expected one narrow band, got tangencies {ts:?}"
        );

        let gl = gauss_legendre(48);
        let quad = |a: f64, b: f64, f: &dyn Fn(f64) -> f64| -> f64 {
            let (c, h) = (0.5 * (a + b), 0.5 * (b - a));
            gl.iter().map(|&(x, wt)| wt * h * f(c + h * x)).sum()
        };
        // ∫ over [a, b] with a divergence at t, which is a or b.
        let u_int = |t: f64, a: f64, b: f64| -> f64 {
            if t <= a {
                quad(0.0, (b - t).sqrt(), &|u| {
                    w(t + u * u) * tau(t + u * u) * 2.0 * u
                })
            } else {
                quad(0.0, (t - a).sqrt(), &|u| {
                    w(t - u * u) * tau(t - u * u) * 2.0 * u
                })
            }
        };
        let smooth = |a: f64, b: f64, panels: usize| -> f64 {
            (0..panels)
                .map(|j| {
                    let (p, q) = (
                        a + (b - a) * j as f64 / panels as f64,
                        a + (b - a) * (j + 1) as f64 / panels as f64,
                    );
                    quad(p, q, &|x| w(x) * tau(x))
                })
                .sum()
        };
        let (t1, t2) = (ts[0], ts[1]);
        let mid = 0.5 * (t1 + t2);
        let reference = smooth(1.2, t1 - 0.1, 29)
            + u_int(t1, t1 - 0.1, t1)
            + u_int(t1, t1, mid)
            + u_int(t2, mid, t2)
            + u_int(t2, t2, t2 + 0.1)
            + smooth(t2 + 0.1, 11.0, 59);

        let grids: Vec<(&str, Box<dyn Fn(f64) -> Vec<f64>>)> = vec![
            (
                "linear 0.023",
                Box::new(|f| (0..850).map(|k| 0.5 + (k as f64 + f) * 0.023).collect()),
            ),
            (
                "linear 0.036",
                Box::new(|f| (0..550).map(|k| 0.5 + (k as f64 + f) * 0.036).collect()),
            ),
            (
                "log 200",
                Box::new(|f| {
                    (0..200)
                        .map(|k| (0.5f64.ln() + (k as f64 + f) * 40f64.ln() / 199.0).exp())
                        .collect()
                }),
            ),
        ];
        for (label, grid) in &grids {
            for f in [0.0, 0.2, 0.4, 0.6, 0.8] {
                let x = grid(f);
                let avg = cell_averaged_probability(eps, m, &x, &cosmo);
                let rel = band_integral(&x, &avg.tau) / reference - 1.0;
                eprintln!("{label}, offset {f}: {rel:+.2e}");
                assert!(rel.abs() < 1e-3, "{label}, offset {f}: {rel:+.2e}");
            }
        }
    }

    /// Convergence study behind ADR 0008's tangency table: the FIRAS-band
    /// integral at m = 1e-12 eV for point values, plain midpoint averages and
    /// the u-integration, over ten grid offsets, plus timings. Prints only.
    #[test]
    #[ignore]
    fn cell_average_convergence_study() {
        let cosmo = Cosmology::default();
        // Fine reference.
        let xf: Vec<f64> = (0..20000)
            .map(|k| 0.5 + (k as f64 + 0.5) * 0.0006)
            .collect();
        let reff = cell_average_with(1e-7, 1e-12, &xf, &cosmo, 8, 64, 2.0);
        let i_ref = band_integral(&xf, &reff.tau);
        println!("reference I = {i_ref:.6e}");
        for dx in [0.023, 0.03, 0.036] {
            for (label, nsc, nst, reach) in [
                ("point", 1usize, 0usize, -1.0),
                ("mid4", 4, 0, -1.0),
                ("mid16", 16, 0, -1.0),
                ("mid64", 64, 0, -1.0),
                ("u4x8", 4, 8, 2.0),
                ("u4x16", 4, 16, 2.0),
                ("u4x32", 4, 32, 2.0),
                ("u2x32", 2, 32, 2.0),
                ("u4x64", 4, 64, 2.0),
            ] {
                let mut rel = Vec::new();
                for j in 0..10 {
                    let f = j as f64 / 10.0;
                    let x: Vec<f64> = (0..(12.0 / dx) as usize)
                        .map(|k| 0.5 + (k as f64 + f) * dx)
                        .collect();
                    let tau: Vec<f64> = if label == "point" {
                        conversion_probability(1e-7, 1e-12, &x, &cosmo).tau
                    } else {
                        cell_average_with(1e-7, 1e-12, &x, &cosmo, nsc, nst.max(1), reach).tau
                    };
                    rel.push(band_integral(&x, &tau) / i_ref - 1.0);
                }
                let mn = rel.iter().cloned().fold(f64::INFINITY, f64::min);
                let mx = rel.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                println!("dx={dx} {label:6}: rel to ref min {mn:+.4} max {mx:+.4}");
            }
        }
        let grid = crate::grid::FrequencyGrid::new(&crate::grid::GridConfig {
            n_points: 4000,
            ..Default::default()
        });
        for m in [1e-12, 1e-11, 1e-9] {
            let t0 = std::time::Instant::now();
            let _ = cell_averaged_probability(1e-7, m, &grid.x, &cosmo);
            let t1 = t0.elapsed();
            let t0 = std::time::Instant::now();
            let _ = conversion_probability(1e-7, m, &grid.x, &cosmo);
            println!(
                "m={m:e}: 4000-pt cell average {t1:?}, point values {:?}",
                t0.elapsed()
            );
        }
        for m in [1e-12, 1e-11] {
            let (gc, _) = gamma_con(1e-7, m, &cosmo).unwrap();
            let xs = [0.5, 1.0, 2.8, 4.0, 10.0];
            let c = conversion_probability(1e-7, m, &xs, &cosmo);
            for (i, &x) in xs.iter().enumerate() {
                println!(
                    "m={m:e} x={x}: P={:.4e} plasma={:.4e} ratio={:.4e} n_cross={}",
                    c.probability[i],
                    -(-gc / x).exp_m1(),
                    c.probability[i] / -(-gc / x).exp_m1(),
                    c.crossings[i].len()
                );
            }
        }
    }
}
