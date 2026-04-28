//! Structured output types and serialization for solver results.
//!
//! Provides [`SolverResult`], [`SweepResult`], and [`GreensResult`] as owned,
//! self-contained representations of completed runs, with zero-dependency
//! JSON/CSV/table serialization.
//!
//! All result types implement [`Serializable`] for uniform JSON/CSV/table output.

use crate::solver::SolverSnapshot;

/// Common serialization interface for all result types.
///
/// Enables generic output handling (see `main.rs::write_output`).
/// Uses `&mut dyn Write` for dyn-compatibility.
pub trait Serializable {
    fn to_json(&self) -> String;
    fn write_csv_dyn(&self, w: &mut dyn std::io::Write) -> std::io::Result<()>;
    fn write_table_dyn(&self, w: &mut dyn std::io::Write) -> std::io::Result<()>;
}

/// Owned result from a completed solver run.
///
/// Contains the observed-redshift snapshot, the frequency grid, and
/// diagnostic info. Does not borrow the solver, so it can outlive it.
/// For runs that need multiple snapshots, use
/// [`crate::solver::ThermalizationSolver::run_with_snapshots`] directly.
#[derive(Debug, Clone)]
#[must_use]
pub struct SolverResult {
    /// Snapshot at the requested observation redshift.
    pub snapshot: SolverSnapshot,
    /// The frequency grid `x[i]` used by the solver.
    pub x_grid: Vec<f64>,
    /// Total number of timesteps taken.
    pub step_count: usize,
    /// Number of steps where Newton iteration hit the max iteration limit.
    pub diag_newton_exhausted: usize,
    /// Diagnostic warnings collected during the run (Newton non-convergence,
    /// ρ_e clamping, NaN emission rates, untested regimes, validation soft
    /// warnings). Empty for a clean run.
    pub warnings: Vec<String>,
}

impl SolverResult {
    /// Serialize to a JSON string (zero dependencies).
    ///
    /// Output format matches the CLI convention used by the Python client:
    /// `{"results":[{"pde_mu":..., "pde_y":..., "drho":..., ...}], "diag_newton_exhausted":N}`
    pub fn to_json(&self) -> String {
        let s = &self.snapshot;
        let mut out = String::with_capacity(self.x_grid.len() * 30 + 256);
        out.push_str("{\"results\":[{");
        write_json_kv(&mut out, "pde_mu", s.mu);
        out.push(',');
        write_json_kv(&mut out, "pde_y", s.y);
        out.push(',');
        write_json_kv(&mut out, "drho", s.delta_rho_over_rho);
        out.push(',');
        write_json_kv(&mut out, "accumulated_delta_t", s.accumulated_delta_t);
        out.push(',');
        write_json_kv(&mut out, "z", s.z);
        out.push(',');
        write_json_kv(&mut out, "rho_e", s.rho_e);
        out.push(',');
        write_json_kv(&mut out, "step_count", self.step_count as f64);
        out.push(',');
        write_json_array(&mut out, "x", &self.x_grid);
        out.push(',');
        write_json_array(&mut out, "delta_n", &s.delta_n);
        out.push_str("}],");
        write_json_kv(
            &mut out,
            "diag_newton_exhausted",
            self.diag_newton_exhausted as f64,
        );
        if !self.warnings.is_empty() {
            out.push(',');
            write_json_string_array(&mut out, "warnings", &self.warnings);
        }
        out.push('}');
        out
    }

    /// Write CSV (frequency, delta_n) to a writer.
    pub fn write_csv<W: std::io::Write + ?Sized>(&self, w: &mut W) -> std::io::Result<()> {
        let s = &self.snapshot;
        writeln!(
            w,
            "# mu={:.6e} y={:.6e} delta_rho_over_rho={:.6e} z={:.1} steps={} newton_exhausted={}",
            s.mu, s.y, s.delta_rho_over_rho, s.z, self.step_count, self.diag_newton_exhausted
        )?;
        for warning in &self.warnings {
            writeln!(w, "# WARNING: {warning}")?;
        }
        writeln!(w, "x,delta_n")?;
        for (i, &x) in self.x_grid.iter().enumerate() {
            writeln!(w, "{:.8e},{:.8e}", x, s.delta_n[i])?;
        }
        Ok(())
    }

    /// Write a human-readable summary table to a writer.
    pub fn write_table<W: std::io::Write + ?Sized>(&self, w: &mut W) -> std::io::Result<()> {
        let s = &self.snapshot;
        writeln!(w, "Solver Result")?;
        writeln!(w, "=============")?;
        writeln!(w, "  z_final       = {:.1}", s.z)?;
        writeln!(w, "  mu            = {:.6e}", s.mu)?;
        writeln!(w, "  y             = {:.6e}", s.y)?;
        writeln!(w, "  delta_rho/rho = {:.6e}", s.delta_rho_over_rho)?;
        writeln!(w, "  rho_e         = {:.8}", s.rho_e)?;
        writeln!(w, "  accum_delta_T = {:.6e}", s.accumulated_delta_t)?;
        writeln!(w, "  steps         = {}", self.step_count)?;
        writeln!(w, "  grid points   = {}", self.x_grid.len())?;
        writeln!(w, "  newton_exhausted = {}", self.diag_newton_exhausted)?;
        if !self.warnings.is_empty() {
            writeln!(w, "  warnings      = {} (see below)", self.warnings.len())?;
            writeln!(w, "Warnings:")?;
            for warning in &self.warnings {
                writeln!(w, "  - {warning}")?;
            }
        }
        Ok(())
    }
}

/// One row of a sweep: PDE result + Green's function comparison at one z_h.
#[derive(Debug, Clone)]
pub struct SweepRow {
    pub z_h: f64,
    pub snapshot: SolverSnapshot,
    pub gf_mu: f64,
    pub gf_y: f64,
    pub gf_delta_n: Vec<f64>,
    pub x_grid: Vec<f64>,
    pub step_count: usize,
}

/// Result of a sweep over multiple injection redshifts.
#[derive(Debug, Clone)]
#[must_use]
pub struct SweepResult {
    pub delta_rho: f64,
    pub rows: Vec<SweepRow>,
    /// Aggregated diagnostic warnings across all sweep workers.
    pub warnings: Vec<String>,
}

impl SweepResult {
    /// Serialize to a JSON string.
    pub fn to_json(&self) -> String {
        let per_row = self.rows.first().map_or(4096, |r| {
            (r.x_grid.len() + r.snapshot.delta_n.len()) * 24 + 256
        });
        let mut out = String::with_capacity(self.rows.len() * per_row + 128);
        out.push('{');
        write_json_kv(&mut out, "delta_rho_inj", self.delta_rho);
        out.push_str(",\"results\":[");
        for (i, row) in self.rows.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('{');
            write_json_kv(&mut out, "z_h", row.z_h);
            out.push(',');
            write_json_kv(&mut out, "pde_mu", row.snapshot.mu);
            out.push(',');
            write_json_kv(&mut out, "gf_mu", row.gf_mu);
            out.push(',');
            write_json_kv(&mut out, "pde_y", row.snapshot.y);
            out.push(',');
            write_json_kv(&mut out, "gf_y", row.gf_y);
            out.push(',');
            write_json_kv(&mut out, "drho", row.snapshot.delta_rho_over_rho);
            out.push(',');
            write_json_kv(
                &mut out,
                "accumulated_delta_t",
                row.snapshot.accumulated_delta_t,
            );
            out.push(',');
            write_json_array(&mut out, "x", &row.x_grid);
            out.push(',');
            write_json_array(&mut out, "delta_n", &row.snapshot.delta_n);
            out.push(',');
            write_json_array(&mut out, "delta_n_gf", &row.gf_delta_n);
            out.push('}');
        }
        out.push(']');
        if !self.warnings.is_empty() {
            out.push(',');
            write_json_string_array(&mut out, "warnings", &self.warnings);
        }
        out.push('}');
        out
    }

    /// Write CSV summary to a writer.
    pub fn write_csv<W: std::io::Write + ?Sized>(&self, w: &mut W) -> std::io::Result<()> {
        for warning in &self.warnings {
            writeln!(w, "# WARNING: {warning}")?;
        }
        writeln!(w, "z_h,pde_mu,gf_mu,pde_y,gf_y,drho,steps")?;
        for row in &self.rows {
            writeln!(
                w,
                "{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{}",
                row.z_h,
                row.snapshot.mu,
                row.gf_mu,
                row.snapshot.y,
                row.gf_y,
                row.snapshot.delta_rho_over_rho,
                row.step_count,
            )?;
        }
        Ok(())
    }

    /// Write a human-readable summary table to stderr-style output.
    pub fn write_table<W: std::io::Write + ?Sized>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(
            w,
            "{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "z_inj", "PDE_mu", "GF_mu", "PDE_y", "GF_y", "PDE_drho", "steps"
        )?;
        writeln!(w, "{}", "-".repeat(75))?;
        for row in &self.rows {
            writeln!(
                w,
                "{:<10.0e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10}",
                row.z_h,
                row.snapshot.mu,
                row.gf_mu,
                row.snapshot.y,
                row.gf_y,
                row.snapshot.delta_rho_over_rho,
                row.step_count,
            )?;
        }
        if !self.warnings.is_empty() {
            writeln!(w, "Warnings ({}):", self.warnings.len())?;
            for warning in &self.warnings {
                writeln!(w, "  - {warning}")?;
            }
        }
        Ok(())
    }
}

/// One row of a photon sweep: PDE result at one z_h for a fixed x_inj.
#[derive(Debug, Clone)]
pub struct PhotonSweepRow {
    pub z_h: f64,
    pub snapshot: SolverSnapshot,
    pub x_grid: Vec<f64>,
    pub step_count: usize,
}

/// Result of a photon injection sweep over multiple injection redshifts at fixed x_inj.
#[derive(Debug, Clone)]
#[must_use]
pub struct PhotonSweepResult {
    pub x_inj: f64,
    pub delta_n_over_n: f64,
    pub rows: Vec<PhotonSweepRow>,
    /// Aggregated diagnostic warnings across all sweep workers.
    pub warnings: Vec<String>,
}

impl PhotonSweepResult {
    /// Serialize to a JSON string.
    pub fn to_json(&self) -> String {
        let per_row = self.rows.first().map_or(4096, |r| {
            (r.x_grid.len() + r.snapshot.delta_n.len()) * 24 + 256
        });
        let mut out = String::with_capacity(self.rows.len() * per_row + 128);
        out.push('{');
        write_json_kv(&mut out, "x_inj", self.x_inj);
        out.push(',');
        write_json_kv(&mut out, "delta_n_over_n", self.delta_n_over_n);
        out.push_str(",\"results\":[");
        for (i, row) in self.rows.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('{');
            write_json_kv(&mut out, "z_h", row.z_h);
            out.push(',');
            write_json_kv(&mut out, "pde_mu", row.snapshot.mu);
            out.push(',');
            write_json_kv(&mut out, "pde_y", row.snapshot.y);
            out.push(',');
            write_json_kv(&mut out, "drho", row.snapshot.delta_rho_over_rho);
            out.push(',');
            write_json_kv(
                &mut out,
                "accumulated_delta_t",
                row.snapshot.accumulated_delta_t,
            );
            out.push(',');
            write_json_array(&mut out, "x", &row.x_grid);
            out.push(',');
            write_json_array(&mut out, "delta_n", &row.snapshot.delta_n);
            out.push('}');
        }
        out.push(']');
        if !self.warnings.is_empty() {
            out.push(',');
            write_json_string_array(&mut out, "warnings", &self.warnings);
        }
        out.push('}');
        out
    }

    /// Write CSV summary to a writer.
    pub fn write_csv<W: std::io::Write + ?Sized>(&self, w: &mut W) -> std::io::Result<()> {
        for warning in &self.warnings {
            writeln!(w, "# WARNING: {warning}")?;
        }
        writeln!(w, "z_h,pde_mu,pde_y,drho,steps")?;
        for row in &self.rows {
            writeln!(
                w,
                "{:.6e},{:.6e},{:.6e},{:.6e},{}",
                row.z_h,
                row.snapshot.mu,
                row.snapshot.y,
                row.snapshot.delta_rho_over_rho,
                row.step_count,
            )?;
        }
        Ok(())
    }

    /// Write a human-readable summary table.
    pub fn write_table<W: std::io::Write + ?Sized>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(
            w,
            "Photon sweep: x_inj={:.4e}, delta_n_over_n={:.4e}",
            self.x_inj, self.delta_n_over_n
        )?;
        writeln!(
            w,
            "{:<10} {:>10} {:>10} {:>10} {:>10}",
            "z_inj", "PDE_mu", "PDE_y", "PDE_drho", "steps"
        )?;
        writeln!(w, "{}", "-".repeat(55))?;
        for row in &self.rows {
            writeln!(
                w,
                "{:<10.0e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10}",
                row.z_h,
                row.snapshot.mu,
                row.snapshot.y,
                row.snapshot.delta_rho_over_rho,
                row.step_count,
            )?;
        }
        if !self.warnings.is_empty() {
            writeln!(w, "Warnings ({}):", self.warnings.len())?;
            for warning in &self.warnings {
                writeln!(w, "  - {warning}")?;
            }
        }
        Ok(())
    }
}

/// Result of a batch photon injection sweep over multiple x_inj values.
///
/// Contains one `PhotonSweepResult` per x_inj value.
#[derive(Debug, Clone)]
#[must_use]
pub struct PhotonSweepBatchResult {
    pub results: Vec<PhotonSweepResult>,
    /// Aggregated diagnostic warnings across all batch workers.
    pub warnings: Vec<String>,
}

impl PhotonSweepBatchResult {
    /// Serialize to a JSON object containing per-x_inj results and aggregated warnings.
    ///
    /// Pre-warnings format was a bare JSON array. With warnings we wrap into
    /// `{"results":[...], "warnings":[...]}`. Python wrappers tolerate both.
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(self.results.len() * 4096);
        out.push_str("{\"results\":[");
        for (i, r) in self.results.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&r.to_json());
        }
        out.push(']');
        if !self.warnings.is_empty() {
            out.push(',');
            write_json_string_array(&mut out, "warnings", &self.warnings);
        }
        out.push('}');
        out
    }

    /// Write combined CSV summary to a writer.
    pub fn write_csv<W: std::io::Write + ?Sized>(&self, w: &mut W) -> std::io::Result<()> {
        for warning in &self.warnings {
            writeln!(w, "# WARNING: {warning}")?;
        }
        writeln!(w, "x_inj,z_h,pde_mu,pde_y,drho,steps")?;
        for r in &self.results {
            for row in &r.rows {
                writeln!(
                    w,
                    "{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{}",
                    r.x_inj,
                    row.z_h,
                    row.snapshot.mu,
                    row.snapshot.y,
                    row.snapshot.delta_rho_over_rho,
                    row.step_count,
                )?;
            }
        }
        Ok(())
    }

    /// Write a human-readable summary table.
    pub fn write_table<W: std::io::Write + ?Sized>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(w, "Photon sweep batch: {} x_inj values", self.results.len())?;
        for r in &self.results {
            r.write_table(w)?;
            writeln!(w)?;
        }
        if !self.warnings.is_empty() {
            writeln!(w, "Aggregated warnings ({}):", self.warnings.len())?;
            for warning in &self.warnings {
                writeln!(w, "  - {warning}")?;
            }
        }
        Ok(())
    }
}

/// Result of a Green's function calculation at a single redshift.
#[derive(Debug, Clone)]
#[must_use]
pub struct GreensResult {
    pub z_h: f64,
    pub mu: f64,
    pub y: f64,
    pub x_grid: Vec<f64>,
    pub delta_n: Vec<f64>,
    /// Diagnostic warnings (validity-range, post-recombination caveats).
    pub warnings: Vec<String>,
}

impl GreensResult {
    /// Serialize to a JSON string.
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(self.x_grid.len() * 30 + 256);
        use std::fmt::Write;
        write!(out, "{{\"results\":[{{").unwrap();
        write_json_kv(&mut out, "z_h", self.z_h);
        out.push(',');
        write_json_kv(&mut out, "gf_mu", self.mu);
        out.push(',');
        write_json_kv(&mut out, "gf_y", self.y);
        out.push(',');
        write_json_array(&mut out, "x", &self.x_grid);
        out.push(',');
        write_json_array(&mut out, "delta_n_gf", &self.delta_n);
        // Close: inner object `}`, array `]`, outer object `}`. The prior
        // `}}]}` emitted three braces and failed `json.loads` (audit H7).
        out.push_str("}]");
        if !self.warnings.is_empty() {
            out.push(',');
            write_json_string_array(&mut out, "warnings", &self.warnings);
        }
        out.push('}');
        out
    }

    /// Write CSV to a writer.
    pub fn write_csv<W: std::io::Write + ?Sized>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(
            w,
            "# mu={:.6e} y={:.6e} z_h={:.2e}",
            self.mu, self.y, self.z_h
        )?;
        for warning in &self.warnings {
            writeln!(w, "# WARNING: {warning}")?;
        }
        writeln!(w, "x,delta_n")?;
        for (i, &x) in self.x_grid.iter().enumerate() {
            writeln!(w, "{:.8e},{:.8e}", x, self.delta_n[i])?;
        }
        Ok(())
    }

    /// Write a human-readable summary.
    pub fn write_table<W: std::io::Write + ?Sized>(&self, w: &mut W) -> std::io::Result<()> {
        writeln!(w, "Green's function result at z_h = {:.2e}:", self.z_h)?;
        writeln!(w, "  mu = {:.6e}", self.mu)?;
        writeln!(w, "  y  = {:.6e}", self.y)?;
        if !self.warnings.is_empty() {
            writeln!(w, "Warnings ({}):", self.warnings.len())?;
            for warning in &self.warnings {
                writeln!(w, "  - {warning}")?;
            }
        }
        Ok(())
    }
}

// --- Serializable trait implementations ---

impl Serializable for SolverResult {
    fn to_json(&self) -> String {
        self.to_json()
    }
    fn write_csv_dyn(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.write_csv(w)
    }
    fn write_table_dyn(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.write_table(w)
    }
}

impl Serializable for SweepResult {
    fn to_json(&self) -> String {
        self.to_json()
    }
    fn write_csv_dyn(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.write_csv(w)
    }
    fn write_table_dyn(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.write_table(w)
    }
}

impl Serializable for PhotonSweepResult {
    fn to_json(&self) -> String {
        self.to_json()
    }
    fn write_csv_dyn(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.write_csv(w)
    }
    fn write_table_dyn(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.write_table(w)
    }
}

impl Serializable for PhotonSweepBatchResult {
    fn to_json(&self) -> String {
        self.to_json()
    }
    fn write_csv_dyn(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.write_csv(w)
    }
    fn write_table_dyn(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.write_table(w)
    }
}

impl Serializable for GreensResult {
    fn to_json(&self) -> String {
        self.to_json()
    }
    fn write_csv_dyn(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.write_csv(w)
    }
    fn write_table_dyn(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        self.write_table(w)
    }
}

/// Write a JSON-safe float: NaN and Inf become null (valid JSON).
fn write_json_float(out: &mut String, val: f64, precision: usize) {
    use std::fmt::Write;
    if val.is_finite() {
        write!(out, "{val:.prec$e}", prec = precision).unwrap();
    } else {
        out.push_str("null");
    }
}

fn write_json_kv(out: &mut String, key: &str, val: f64) {
    use std::fmt::Write;
    write!(out, "\"{key}\":").unwrap();
    write_json_float(out, val, 15);
}

fn write_json_array(out: &mut String, key: &str, arr: &[f64]) {
    use std::fmt::Write;
    write!(out, "\"{key}\":[").unwrap();
    for (i, &v) in arr.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_json_float(out, v, 15);
    }
    out.push(']');
}

/// Write a JSON array of strings, escaping `"`, `\`, and control characters.
fn write_json_string_array(out: &mut String, key: &str, arr: &[String]) {
    use std::fmt::Write;
    write!(out, "\"{key}\":[").unwrap();
    for (i, s) in arr.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    write!(out, "\\u{:04x}", c as u32).unwrap();
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
    out.push(']');
}

/// Parse an output format from a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Json,
    Csv,
    Table,
}

impl OutputFormat {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "json" => Ok(OutputFormat::Json),
            "csv" => Ok(OutputFormat::Csv),
            "table" => Ok(OutputFormat::Table),
            _ => Err(format!(
                "Unknown output format: '{s}'. Valid: json, csv, table"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result() -> SolverResult {
        SolverResult {
            snapshot: SolverSnapshot {
                z: 100.0,
                delta_n: vec![1e-6, 2e-6, -3e-6],
                rho_e: 1.000_01,
                mu: 5.0e-7,
                y: 1.0e-7,
                delta_rho_over_rho: 1.0e-5,
                accumulated_delta_t: 2.0e-10,
            },
            x_grid: vec![0.1, 1.0, 10.0],
            step_count: 42,
            diag_newton_exhausted: 0,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn test_output_format_from_str() {
        assert_eq!(OutputFormat::from_str("json").unwrap(), OutputFormat::Json);
        assert_eq!(OutputFormat::from_str("csv").unwrap(), OutputFormat::Csv);
        assert_eq!(
            OutputFormat::from_str("table").unwrap(),
            OutputFormat::Table
        );
        assert!(OutputFormat::from_str("xml").is_err());
        assert!(OutputFormat::from_str("").is_err());
    }

    fn sample_sweep_result() -> SweepResult {
        SweepResult {
            delta_rho: 1e-5,
            warnings: Vec::new(),
            rows: vec![
                SweepRow {
                    z_h: 1e4,
                    snapshot: SolverSnapshot {
                        z: 500.0,
                        delta_n: vec![1e-8, 2e-8],
                        rho_e: 1.0,
                        mu: 1e-10,
                        y: 2.5e-6,
                        delta_rho_over_rho: 1e-5,
                        accumulated_delta_t: 0.0,
                    },
                    gf_mu: 1.1e-10,
                    gf_y: 2.4e-6,
                    gf_delta_n: vec![1.1e-8, 1.9e-8],
                    x_grid: vec![1.0, 10.0],
                    step_count: 100,
                },
                SweepRow {
                    z_h: 2e5,
                    snapshot: SolverSnapshot {
                        z: 500.0,
                        delta_n: vec![3e-6, -1e-6],
                        rho_e: 1.000_01,
                        mu: 1.4e-5,
                        y: 1e-8,
                        delta_rho_over_rho: 1e-5,
                        accumulated_delta_t: 1e-11,
                    },
                    gf_mu: 1.38e-5,
                    gf_y: 1.1e-8,
                    gf_delta_n: vec![2.9e-6, -0.9e-6],
                    x_grid: vec![1.0, 10.0],
                    step_count: 500,
                },
            ],
        }
    }

    /// Count opening vs. closing braces/brackets in a JSON string, ignoring
    /// contents inside double-quoted strings. Used to catch the audit H7
    /// class of bug (`}}]}` emitting one too many closing braces) without
    /// taking a serde_json dev-dependency.
    fn json_is_balanced(s: &str) -> bool {
        let mut in_str = false;
        let mut escape = false;
        let mut curly: i32 = 0;
        let mut square: i32 = 0;
        for c in s.chars() {
            if escape {
                escape = false;
                continue;
            }
            if in_str {
                match c {
                    '\\' => escape = true,
                    '"' => in_str = false,
                    _ => {}
                }
                continue;
            }
            match c {
                '"' => in_str = true,
                '{' => curly += 1,
                '}' => curly -= 1,
                '[' => square += 1,
                ']' => square -= 1,
                _ => {}
            }
            if curly < 0 || square < 0 {
                return false;
            }
        }
        curly == 0 && square == 0 && !in_str
    }

    #[test]
    fn test_json_bracket_balance_all_variants() {
        // SolverResult
        assert!(
            json_is_balanced(&sample_result().to_json()),
            "SolverResult JSON has unbalanced brackets"
        );
        // SweepResult
        assert!(
            json_is_balanced(&sample_sweep_result().to_json()),
            "SweepResult JSON has unbalanced brackets"
        );
        // GreensResult
        let g = GreensResult {
            z_h: 2e5,
            mu: 1.4e-5,
            y: 1e-8,
            x_grid: vec![1.0, 10.0],
            delta_n: vec![3e-6, -1e-6],
            warnings: Vec::new(),
        };
        assert!(
            json_is_balanced(&g.to_json()),
            "GreensResult JSON has unbalanced brackets"
        );
    }
}
