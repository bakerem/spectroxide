//! Compile-time content hash of the physics-relevant Rust source.
//!
//! Hashes a curated list of files whose contents directly affect the
//! numerical output of `sweep` / `photon-sweep` (the calls used to build
//! Green's function tables). The digest is exposed via `env!("PHYSICS_HASH")`
//! so that cached tables can be invalidated when any of these files change.
//!
//! Files outside this list (CLI parsing, output formatting, analytic GF in
//! `greens.rs`) do not affect table contents and are intentionally excluded
//! to avoid invalidating tables for unrelated edits.
//!
//! Hash algorithm: 64-bit FNV-1a over `<path>\0<bytes>\xff` for each file in
//! order. FNV-1a is dep-free and deterministic across rustc versions (unlike
//! `std::hash::DefaultHasher`).

use std::fs;

const PHYSICS_FILES: &[&str] = &[
    "src/constants.rs",
    "src/cosmology.rs",
    "src/recombination.rs",
    "src/spectrum.rs",
    "src/grid.rs",
    "src/electron_temp.rs",
    "src/bremsstrahlung.rs",
    "src/double_compton.rs",
    "src/kompaneets.rs",
    "src/energy_injection.rs",
    "src/dark_photon.rs",
    "src/solver.rs",
];

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn main() {
    let mut h: u64 = FNV_OFFSET;
    for file in PHYSICS_FILES {
        println!("cargo:rerun-if-changed={file}");
        let bytes = fs::read(file).unwrap_or_else(|e| panic!("build.rs cannot read {file}: {e}"));
        for &b in file.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h ^= 0;
        h = h.wrapping_mul(FNV_PRIME);
        for &b in &bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h ^= 0xff;
        h = h.wrapping_mul(FNV_PRIME);
    }
    println!("cargo:rustc-env=PHYSICS_HASH={h:016x}");
    println!("cargo:rerun-if-changed=build.rs");
}
