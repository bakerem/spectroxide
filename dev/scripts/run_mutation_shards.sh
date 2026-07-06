#!/usr/bin/env bash
# Workstream R2 — mutation-testing shard runner (dev/PLAN_VALIDATION_ROUND2).
#
# Runs cargo-mutants one physics module at a time (a "shard"), detached, so the
# wall-clock-bound run can proceed overnight while other work continues
# (directive 3). Parses mutants.out/{outcomes.json,missed.txt} — never scrapes
# stdout. Each shard writes to dev/audit/mutation/<module>/.
#
# IMPORTANT: before trusting the flags below, run `cargo mutants --help` against
# the INSTALLED version and reconcile — flag names have changed across releases
# (this is a known Opus failure point). Current assumptions (cargo-mutants
# v27.x): `-f <file>` scopes to one file; `--timeout <secs>` sets the per-mutant
# timeout; test args come from .cargo/mutants.toml. Verify each.
#
# Usage:  bash dev/scripts/run_mutation_shards.sh <tier1|tier2|FILE>
set -uo pipefail
cd "$(dirname "$0")/../.."

TIER1=(
  src/kompaneets.rs src/solver.rs src/double_compton.rs
  src/bremsstrahlung.rs src/electron_temp.rs src/recombination.rs
)
TIER2=(
  src/greens.rs src/distortion.rs src/cosmology.rs src/dark_photon.rs
  src/spectrum.rs src/grid.rs src/energy_injection.rs
)

case "${1:-}" in
  tier1) FILES=("${TIER1[@]}") ;;
  tier2) FILES=("${TIER2[@]}") ;;
  "")    echo "usage: $0 <tier1|tier2|src/FILE.rs>"; exit 2 ;;
  *)     FILES=("$1") ;;
esac

# Per-mutant timeout: cargo-mutants measures a baseline first; 3-5x baseline
# converts non-converging Newton loops (infinite hangs) into "killed (timeout)",
# which counts as caught (R2.2). 300s is a conservative cap.
TIMEOUT=${MUTANTS_TIMEOUT:-300}

for f in "${FILES[@]}"; do
  mod=$(basename "$f" .rs)
  outdir="dev/audit/mutation/$mod"
  mkdir -p "$outdir"
  echo "=== shard: $f -> $outdir ($(date -u +%H:%M:%S)) ==="
  # --in-place would mutate the checkout; cargo-mutants copies to a scratch dir
  # by default. Keep the outcomes.json under the module dir.
  cargo mutants -f "$f" --timeout "$TIMEOUT" \
      --output "$outdir" 2>&1 | tail -5
  echo "  survivors: $(wc -l < "$outdir/mutants.out/missed.txt" 2>/dev/null || echo '?')"
done
echo "=== ALL_SHARDS_DONE $(date -u +%H:%M:%S) ==="
