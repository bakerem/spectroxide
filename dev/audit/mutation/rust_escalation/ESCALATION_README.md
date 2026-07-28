# R2 full-suite survivor-escalation pass — resume notes

**Launched:** 2026-07-11. **Driver:** `run_escalation.sh` (detached; poll logs,
do not rely on `ps` — detached runs are in a separate PID namespace).

## What this pass does
Upgrades each module's lean lower-bound score to a true full-suite score by
escalating ONLY the lean survivors against ONLY the heavy suites lean excluded.
- Lean survivors are fed per-module via `--re` built from each module's lean
  `missed.txt` (`gen_regex.py`, dead-code fns `dc_/br_heating_integral` excluded).
- `.cargo/mutants.toml` = heavy suites only (adversarial, mms, conservation_fuzz,
  science, convergence, cosmotherm, cli, coverage_gaps, heat_injection). We do
  NOT re-run `--lib`/`greens` (already survived) nor lean-caught mutants.
- `tests/*.rs` renamed `a_..j_` so cargo's alphabetical run order == ascending
  runtime; cargo fail-fast (default) stops at the first heavy suite that kills a
  mutant. Config `--test` list omits `c_greens` and `--lib`.
- Per-mutant timeout `-t 1200` (a true survivor runs the full ~750s heavy chain;
  keep well above it so survivors are not falsely timed out → false "caught").

## Final score (recombination, after this pass)
`full_suite_score(mod) = (lean_caught + lean_timeout) + (survivors caught/timeout here)`
over `(lean_total − unviable)`. Lean caught counts: `mutation_audit.md` table /
`out/<mod>/mutants.out/`. Escalation results: `out_esc/<mod>/mutants.out/`.
Running tally: `escalation_status.tsv`.

## SHIFTED modules — source changed after their lean run
Their recorded `missed.txt` line:col are stale (don't match current mutants), so
the driver RE-RUNS LEAN (`lean.toml`) on current source first → `out_relean/<mod>`
→ regenerates `regex/<mod>.re` → then escalates. This also refreshes their
lean-caught baseline (needed since the old count is for stale source).
- `double_compton` — `dc_heating_integral` deleted 07-06 (F-R2-1) shifted lines.
- `solver` — additive axion renames 07-10 shifted 68/562 survivor line-refs.
The other 11 modules matched their lean `missed.txt` exactly (verified via
`cargo mutants --list --re`), so they escalate directly.

## Parallel layout (3 copies, launched 2026-07-11)
RAM (not the sandbox) is the only bottleneck: 12 cores, ~6.9 GB (WSL2 default =
50% of a 13.8 GB host; raising it needs a disruptive `wsl --shutdown` and buys
little). CPU is idle under `--in-place` (1 mutant at a time), so we run 3 copies
concurrently, each capped `BJ=2 JT=2 TT=2` (test-threads=2). Measured under
3-way load: ~1 GB used over idle, ~5 GB free — headroom for a 4th if wanted.
- `spx-mut`   (WORKER=a): shifted modules — double_compton, solver.
- `spx-mut-b` (WORKER=b): greens, distortion, grid, spectrum (~731 survivors).
- `spx-mut-c` (WORKER=c): energy_injection, kompaneets, cosmology, recombination,
  dark_photon, bremsstrahlung, axion (~722 survivors).
Each copy has its own `queue.txt` + its own `target/` (concurrent builds need
separate target dirs — a copied cross-path `target/` breaks: unanchored rsync
`--exclude 'out/'` also drops build-script `out/` dirs like serde_core's
`private.rs`; rebuild fresh instead of copying target).

## LAUNCH — use the harness background mechanism, NOT setsid
`setsid nohup ./run_escalation.sh &` FROZE every worker: the detached session has
no controlling TTY, so the `cargo test` children hit SIGTTOU and STOP (T state —
0 CPU, logs frozen at "Found N mutants", `outcomes.json` written with 0 results
on the eventual SIGTERM). Redirecting `</dev/null` helped but they were still
torn down. **What works: launch each worker as a Claude Code `run_in_background`
Bash command** (`cd spx-mut && WORKER=a BJ=2 JT=2 TT=2 ./run_escalation.sh`);
these persist across turns and are not killed when other tool calls finish.
Diagnose liveness by `/proc/loadavg` (>0 = building/testing) + growing
`caught.txt`, never by `ps`/`pgrep` (detached namespace hides the procs).
Kill workers with OBFUSCATED pkill patterns (`p=$(printf 'run_%ss' escalation.sh)`)
so the literal isn't in the killer shell's own cmdline (else pkill -f self-kills → rc137).

## VALIDATED CONFIG (2026-07-11) — 3 copies × BJ=2 JT=2 TT=4
Launched via `run_in_background` per copy: `cd spx-mut{,-b,-c} && WORKER=x BJ=2
JT=2 TT=4 ./run_escalation.sh`. Validated: peak load ~10.8 (11/12 cores), min
avail ~4.5 GB (safe), baselines rebuild+pass ("3s build + 51s test"), mutants
caught. Wall-clock estimate ~7.5 days (worker B ~731 survivors is the long pole;
heavy-suite chain is serial/fail-fast so TT4 is only ~1.2× over TT2 — total is
CPU-bound at ~12 threads). Task IDs (this session): A=bbwho14vl, B=brvrd7ayk,
C=b7zmdrwqp. Faster option: 6 copies × TT2 (more mutants in flight) ~1.3× better.

## Restart
Relaunch each worker via `run_in_background` from its copy dir. The driver skips
any step whose `outcomes.json` exists and restores `src.pristine/`→`src/` before
each step AND `touch`es src so cargo rebuilds (rsync -a preserves old mtimes →
cargo would run a STALE mutated binary as baseline → "0s build" + baseline fail;
touch forces a real rebuild — verify baseline shows "Ns build", N>0). Big modules sharded
(kompaneets 2, distortion 3, energy_injection 3, greens 4, solver 5) so a kill
loses at most one shard. Drop partial out-dirs (no outcomes.json, or 0 results =
interrupted) before restart. NEVER run concurrent COLD builds (7 GB OOMs) — prime
each copy's target serially first.
