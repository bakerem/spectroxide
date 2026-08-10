#!/usr/bin/env bash
# Populate the Fig. 4 PDE cache by calling the Rust binary directly.
#
# Each solve writes its own --output file, so a dead parent (or a killed shell)
# never destroys finished work: re-running skips whatever is already on disk.
# The Python sibling run_dm_pde_cache.py held every result in one process and
# piped JSON through stdout, which lost ~4.5 h of solves when it died.
#
# Usage:  dev/scripts/dm_residual_diagnostics/run_dm_pde_cache.sh 0.002 0.001
#         (arguments are dy_max values; dtau_max/n_points fixed below)
#
# Each solve is single-threaded and needs only a few MB, so PARALLEL can go up to
# the core count; six concurrent 8000-point solves sit at load ~7 on a 12-core,
# 6.8 GB box.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN="$ROOT/target/release/spectroxide"
OUT="$HOME/.spectroxide/dm_pde_raw"
mkdir -p "$OUT"

N_POINTS="${N_POINTS:-8000}"
# dy_max only binds where theta_e > dy_max/dtau_max, i.e. 1+z > dy_max/(dtau_max*4.6e-10);
# at dtau_max=1 that is above z_start for dy>=0.002, so dtau_max sets nearly every step.
DTAU_MAX="${DTAU_MAX:-1.0}"
Z_START=5000000.0
Z_END=1001.0
# 1 / cosmic_time(5.0104e4, DEFAULT_COSMO); matches the notebook's gamma_x.
GAMMA_X=1.0705247598489247e-10

PARALLEL="${PARALLEL:-3}"
DY_LIST=("${@:-0.002}")
JOBS="$OUT/joblist.txt"
: > "$JOBS"

for dy in "${DY_LIST[@]}"; do
  for scen in Decay s-wave p-wave; do
    case "$scen" in
      Decay)  inj=(decaying-particle --f-x 7.757e5 --gamma-x "$GAMMA_X") ;;
      s-wave) inj=(annihilating-dm --f-ann 3.758e-20) ;;
      p-wave) inj=(annihilating-dm-pwave --f-ann 5.789e-26) ;;
    esac
    tag="${scen}_n${N_POINTS}_dy${dy}_dtau${DTAU_MAX}"
    dest="$OUT/${tag}.json"
    if [[ -s "$dest" ]]; then
      echo "skip $tag (exists)"
      continue
    fi
    printf '%s\t%s\t%s\n' "$tag" "$dest" "${inj[*]} --delta-rho 1e-5 --z-start $Z_START --z-end $Z_END --dy-max $dy --n-points $N_POINTS --dtau-max $DTAU_MAX" >> "$JOBS"
  done
done

echo "queued $(wc -l < "$JOBS") solves; $PARALLEL at a time"

# shellcheck disable=SC2016
xargs -P "$PARALLEL" -L 1 -a "$JOBS" bash -c '
  tag="$1"; dest="$2"; shift 2
  echo "[$(date +%T)] start $tag"
  # shellcheck disable=SC2086
  if '"$BIN"' solve $* --output "$dest.part" > "'"$OUT"'/$tag.log" 2>&1; then
    mv "$dest.part" "$dest"
    echo "[$(date +%T)] done  $tag -> $dest"
  else
    echo "[$(date +%T)] FAILED $tag (see '"$OUT"'/$tag.log)"
  fi
' _

echo "[$(date +%T)] all queued solves finished"
