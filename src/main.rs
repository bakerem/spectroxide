//! Thin CLI dispatcher for spectroxide.
//!
//! All logic lives in the library (cli.rs, output.rs). This binary just
//! parses args, calls execute_*, and writes output.

use spectroxide::cli;
use spectroxide::output::{OutputFormat, Serializable};
use std::io::Write;

fn main() {
    // Reset SIGPIPE to default so piping to `head` etc. exits cleanly
    // instead of panicking on write error. No libc dependency needed.
    #[cfg(unix)]
    {
        // SAFETY: signal() is async-signal-safe, SIG_DFL=0 is always valid.
        // SIGPIPE=13 on all Linux/macOS targets.
        unsafe {
            // fn signal(sig: i32, handler: usize) -> usize
            unsafe extern "C" {
                fn signal(sig: i32, handler: usize) -> usize;
            }
            signal(13, 0); // SIGPIPE, SIG_DFL
        }
    }

    eprintln!("spectroxide: CMB spectral distortion solver");
    eprintln!("==========================================\n");

    let cli_args: Vec<String> = std::env::args().skip(1).collect();

    let command = if cli_args.is_empty() {
        Ok(cli::Command::Help)
    } else {
        cli::parse_command(&cli_args)
    };

    let command = match command {
        Ok(cmd) => cmd,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    let result = run_command(command);
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run_command(command: cli::Command) -> Result<(), String> {
    match command {
        cli::Command::Help => {
            cli::print_help();
            Ok(())
        }
        cli::Command::Info(opts) => {
            cli::print_info(&opts)?;
            Ok(())
        }
        cli::Command::PhysicsHash => {
            println!("{}", spectroxide::PHYSICS_HASH);
            Ok(())
        }
        cli::Command::Greens(ref opts) => write_output(&cli::execute_greens(opts)?, &opts.output),
        cli::Command::Solve(ref opts) => write_output(&cli::execute_solve(opts)?, &opts.output),
        cli::Command::Sweep(ref opts) => write_output(&cli::execute_sweep(opts)?, &opts.output),
        cli::Command::PhotonSweep(ref opts) => {
            write_output(&cli::execute_photon_sweep(opts)?, &opts.output)
        }
        cli::Command::PhotonSweepBatch(ref opts) => {
            write_output(&cli::execute_photon_sweep_batch(opts)?, &opts.output)
        }
    }
}

/// Get the output writer: file if --output specified, stdout otherwise.
fn output_writer(opts: &cli::OutputOpts) -> Result<Box<dyn Write>, String> {
    if let Some(ref path) = opts.output_path {
        let f = std::fs::File::create(path)
            .map_err(|e| format!("Failed to create output file '{path}': {e}"))?;
        Ok(Box::new(std::io::BufWriter::new(f)))
    } else {
        Ok(Box::new(std::io::BufWriter::new(std::io::stdout().lock())))
    }
}

/// Write any result type to the configured output destination and format.
fn write_output(result: &dyn Serializable, opts: &cli::OutputOpts) -> Result<(), String> {
    let map_io = |e: std::io::Error| format!("Write error: {e}");
    match opts.format {
        OutputFormat::Table => result
            .write_table_dyn(&mut *output_writer(opts)?)
            .map_err(map_io),
        OutputFormat::Csv => result
            .write_csv_dyn(&mut *output_writer(opts)?)
            .map_err(map_io),
        OutputFormat::Json => {
            let mut w = output_writer(opts)?;
            write!(w, "{}", result.to_json()).map_err(map_io)?;
            if let Some(ref path) = opts.output_path {
                eprintln!("Wrote {path}");
            }
            Ok(())
        }
    }
}
