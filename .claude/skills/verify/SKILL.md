# Verify All

Run the full test suite and report results. Do NOT make any code changes.

## Rules — READ CAREFULLY

1. Run **one single command**: `cargo test --release 2>&1`
2. Do NOT use `| tail`, `| head`, or any pipe that truncates output
3. Do NOT use `run_in_background` — run it in the foreground and wait
4. Do NOT spawn multiple test commands in parallel
5. Set `timeout: 1800000` (30 minutes) to avoid timeouts on slow tests
6. Read the **complete** output that comes back
7. Report: total tests passed, any failures, any warnings
8. If there are failures, show the failure details

## What NOT to do

- Do NOT run `cargo test` (debug mode) — always use `--release`
- Do NOT run subsets of tests unless specifically asked
- Do NOT pipe output through tail/head/grep — you need the full output
- Do NOT run tests in background then poll — just wait for the foreground command
