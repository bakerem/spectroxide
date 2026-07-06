# R3 reference solver — STATUS: INCOMPLETE (WIP)

**Date:** 2026-07-06. Workstream R3 of PLAN_VALIDATION_ROUND2.

The clean-room Chang–Cooper reference solver was implemented by an **isolated
fresh-context subagent** (per the R3 isolation rule: it did not read `src/*.rs`,
`greens.py`, or `solver.py`). The subagent produced `refsolver.py` (366 lines)
but **hit its session limit before validating**, so this is WORK IN PROGRESS:

- ✅ Frozen ingredient table `inputs/history.csv` exported from spectroxide
  (z, X_e, H, n_e, n_H, T_γ, t_C on the Chluba-2013 cosmology) — solid,
  regenerable via `inputs/export_history.py`.
- ✅ `contract.md` — full physics + I/O spec.
- 🚧 `refsolver.py` — Chang–Cooper implementation, **not yet validated**. No
  `results.json`, no per-case spectra, no README, no convergence check yet.
- ⚠️ **Open issue flagged by the subagent:** an elementary Compton up-scatter
  (which must be a *pure y*-distortion) decomposes to a spurious μ in its
  implementation. The subagent attributed this to the decomposition-template
  definitions. **To resolve on resume:** verify the contract's `M(x)` /
  `Y_SZ(x)` templates are the intended shapes and that a pure-y input yields
  μ≈0 under the joint least-squares recipe (spectroxide's own decomposition is
  independently audited correct — distortion_audit.md P1-7 — so the issue is in
  the refsolver's transcription of the contract, or its distortion is not
  exactly Y_SZ because T-shift was not subtracted before the fit).

**Do NOT treat `refsolver.py` as a validated reference until STATUS says so.**
Resume by relaunching a fresh isolated subagent with `contract.md` +
`history.csv` (isolation intact), starting from the template check above.
