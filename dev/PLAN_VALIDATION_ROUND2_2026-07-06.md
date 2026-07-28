# Plan: Validation Round 2 — External Anchors & Test-Adequacy Audit

**Date:** 2026-07-06
**Context:** SciPost referees remain unconvinced the code is sufficiently
checked. Round 1 (dev/PLAN_VALIDATION_AUDIT_2026-07-02.md, Phases 0–2b
complete; see dev/audit/AUDIT_SUMMARY.md) delivered: test-provenance census,
eight adversarial module audits (zero production physics bugs), Rust↔Python
parity CI, method-of-manufactured-solutions convergence proofs, conservation
fuzzing, a quantified error budget, and a HyRec-2 X_e anchor.

**Residual weakness Round 2 attacks:** everything in Round 1 was still
produced *within this project*, by the same agent lineage that wrote the
code, validated mostly against analytic identities and literature fit
coefficients. A skeptical referee can still say: (i) no comparison against a
*running independent code*, (ii) no quantitative measure of whether the test
suite would actually *catch* a planted bug, (iii) the delicate numerics
(cancellation branches, unsafe indexing) are argued correct, not machine-
checked. Round 2 closes exactly those three holes.

**Design principle (inherited, non-negotiable):** attribution over assertion.
A claim counts as validated only when anchored outside the code, and the
anchor is documented next to the claim. Every workstream below produces an
artifact checkable without trusting the auditor.

---

## Read this first — standing directives for the executing agent

These override any instinct to "make the comparison agree." Violating them
invalidates the workstream.

1. **Disagreement is a finding, not a failure.** When spectroxide disagrees
   with an external code or a high-precision recomputation, you do NOT tune
   spectroxide (or the comparison harness) until curves overlap. You document
   the discrepancy, decompose it into attributable causes (different X_e,
   different branching-ratio treatment, different cosmology, genuine bug),
   and quantify each cause by swapping one ingredient at a time. Only a
   confirmed spectroxide bug triggers a code change, and then the full B5
   findings protocol applies: fix → rerun affected published figures →
   before/after numbers in dev/audit/AUDIT_SUMMARY.md → CHANGELOG.
2. **Never fabricate reference values.** If you cannot obtain a reference
   number (paywalled paper, figure you cannot digitize, tool that will not
   build), record the blockage in the memo and move on or ask EB. A
   plausible-looking invented anchor is strictly worse than a documented gap.
   This has been the single most damaging LLM failure mode in prior audits.
3. **Rust tests: `--release` only, detached for anything slow.** Never run
   `cargo test` in debug. Anything expected to exceed ~10 min must be
   launched detached (`setsid nohup ... > log 2>&1 &`) and polled via the
   log; do not sit in a foreground Bash call that times out, and do not
   `pkill` the cargo parent process (it kills the whole PID namespace).
4. **Do not add production dependencies.** The Rust crate has zero production
   deps by design. Everything new lives in dev tooling, `dev/`, `tests/`
   (dev-deps only if unavoidable — prefer none), Python scripts, or CI
   config. If a workstream seems to require a production dep, stop and
   redesign.
5. **Sandbox/network:** `github.com` and `raw.githubusercontent.com` are
   allowed hosts; `crates.io` is NOT. Install Rust tooling from git
   (`cargo install --git ...`) or ask EB before disabling the sandbox for a
   specific command. `arxiv.org` is allowed for paper sources.
6. **When a paper equation is needed, fetch the raw arXiv LaTeX source**
   (`arxiv.org/e-print/<id>`), not ar5iv/HTML renderings — a Round-1 audit
   documented ar5iv garbling equation coefficients (AUDIT_SUMMARY.md, P1-9).
7. **Numeric claims in memos must be regenerable.** Every table in a memo
   states the script (checked into `dev/scripts/` or `dev/audit/scripts/`)
   and exact command that produced it.
8. **After any multi-file Rust edit:** `cargo clippy -- -D warnings` and the
   test suite (release) before committing. `[skip ci]` on doc-only commits.
9. **Blocked-on-EB list:** maintain a short "Decisions/inputs needed from EB"
   section at the bottom of dev/audit/AUDIT_SUMMARY.md rather than stalling.

---

## Workstream R0 — Claim→anchor coverage matrix (do first; it steers everything else)

The referee's question is not "did you run checks" but "is every published
result checked." Answer it with a single table, built before any new
validation work so gaps steer prioritization, and finished as the paper's
appendix validation matrix.

- Enumerate every published figure and quoted number in the paper
  (paper_figures notebooks: `mu_y_vs_injection_redshift`,
  `photon_injection_spectra`, `firas_photon_limits`,
  `dark_photon_constraints`, `dm_scenario_comparison`,
  `cosmotherm_comparison`, `energy_conservation`, `convergence_study`,
  `pathological_heating`, `visibility_functions`, plus any in-text scalar
  claims — grep the .tex for numerals).
- For each: list every validation artifact that constrains it, classified by
  independence level: (i) analytic identity, (ii) literature
  coefficient/curve, (iii) independent code (CLASS, HyRec, refsolver,
  CosmoTherm DI files), (iv) internal-only (MMS, convergence, parity,
  conservation). A row whose strongest anchor is class (iv) is a gap.
- Artifact: `dev/audit/coverage_matrix.md`, updated as R1–R5 land; each
  Round-2 workstream must state in its memo which matrix rows it upgrades.
- Expected outcome of the first pass (verify, don't assume): heat-injection
  rows get class-(iii) from R1+R3; photon-injection rows have NO independent-
  code anchor until R3's photon case and R5's Chluba-2015/Bolliet+2020
  curves land — that is precisely why those two items exist; dark-photon
  rows anchor via the CCJ24 statistic reproduction (Round 1) + AxionLimits
  data (R5).

**Gate:** matrix complete with zero rows lacking a planned class-(i)–(iii)
anchor; any row that cannot get one is flagged to EB with a proposed
paper-text caveat instead.

---

## Workstream R1 — Cross-code comparison against CLASS's spectral-distortion module

**The single most referee-convincing item in this plan.** CLASS ≥ v3.0 ships
an independent, publicly maintained spectral-distortion module (`sd`,
Lucca, Schöneberg, Hooper, Lesgourgues & Chluba 2020, JCAP 02 (2020) 026,
arXiv:1910.04619). It computes μ, y (and PCA residuals) from heating
histories via Green's-function/branching-ratio methods — the same physics
regime as our heat-injection Green's function and PDE. It is written by a
different group, in a different language, with a different numerical
approach. Agreement here is the like-for-like external-code check the
referee asked for, without needing CosmoTherm access.

**Scope limitation (state it in the memo and the paper):** CLASS `sd`
handles *heating* histories only — it has no photon-injection or
dark-photon-resonance channel. R1 therefore anchors the heat-injection half
of the paper. The photon-injection results get their independent-code
anchor from R3's photon case and R5's literature curves; the coverage
matrix (R0) tracks this explicitly. Do not oversell R1 as validating the
whole code.

### R1.1 Setup

- Clone `https://github.com/lesgourg/class_public`, pin to the latest
  release tag; record tag + commit hash in the memo.
- **Build the C code and drive it via `.ini` files + output files only. Do
  NOT fight the `classy` Python wrapper.** Building classy against modern
  NumPy/Cython is a known time sink (Cython 3 / NumPy 2 API churn); the CLI
  path (`./class explanatory.ini`) needs only `make class` and gives
  identical numbers. If `make` fails, the usual fixes are `CC=gcc` and
  deleting the `-arch` flags in the Makefile; do not spend more than ~30 min
  before falling back to `make class` (skip the wrapper target entirely).
- Locate the SD options in `explanatory.ini` / the `sd` section of
  `input.c`: `output = Sd`, `sd_branching_approx` (options include sharp/
  soft variants and `exact` which reads Chluba's external Green's data
  files — the repo ships them under `external/distortions/`). Read
  `source/distortions.c` enough to know exactly what each option computes;
  cite line numbers in the memo.

### R1.2 Case matrix (one ingredient varies at a time)

Match cosmology first — the exact `Cosmology::default()` (Chluba 2013 /
CosmoTherm) values, not Planck: T_cmb = 2.726 K, Ω_b = 0.044, Ω_m = 0.26,
h = 0.71 (so ω_b = 0.0222, ω_cdm = 0.1089), Y_p = 0.24 (CLASS `YHe`),
N_eff = 3.046, Σm_ν = 0. Document any parameter CLASS forces that we do not
model. Then compare, in increasing coupling:

| Case | spectroxide side | CLASS side | Expected agreement |
|---|---|---|---|
| A. Adiabatic ΛCDM cooling μ and y | PDE `check_adiabatic` path + GF | on by default in sd module | few % |
| B. Decaying particle, 3 lifetimes spanning z_X ≈ 10⁶ / 10⁵ / 10⁴ | `DecayingParticle` | `DM_decay_Gamma` via injection module, f_eff = 1 | few–10 % (see R1.4) |
| C. s-wave annihilation | `AnnihilatingDM` | `DM_annihilation_efficiency`, f_eff = 1 | same |
| D. μ(z_h) / y(z_h) transfer: our GF visibility vs CLASS branching ratios | `greens.rs` J_bb/J_μ/J_y | `sd_branching_approx` sweep incl. `exact` | this comparison IS the result |

Set CLASS's energy-deposition treatment to on-the-spot (f_eff = 1, no
DarkAges tables) so the comparison isolates *thermalization*, not deposition
modeling — deposition is out of our scope.

### R1.3 Harness & artifacts

- `dev/scripts/class_sd_compare.py`: writes the `.ini` files, runs the CLASS
  binary, parses `<root>_sd.dat` / distortion output files, runs the
  matching spectroxide case (Rust binary via `run_sweep`-style subprocess
  and Python GF), emits a ratio table + figure.
- Memo `dev/audit/class_sd_comparison.md`: per-case table
  (μ_spx_PDE, μ_spx_GF, μ_CLASS per branching option, ratios), the
  ingredient-swap decomposition of every discrepancy > 2 %, and a paragraph
  the referee reply can quote.
- Keep the CLASS input/output files under `dev/output/class_sd/` (small
  ones) so the comparison is regenerable; a `README` there records the
  exact CLASS commit and build flags.

### R1.4 Anticipated sticking points (directives)

- **Unit conventions for injection.** CLASS decay/annihilation parameters
  are in different units (Γ in 1/s vs our Γ_X; annihilation in
  cm³/s/GeV-style `annihilation_efficiency` — check `input.c` for the
  exact definition in the pinned version; it changed across CLASS versions).
  Derive the mapping analytically, write it in the memo BEFORE running, and
  verify it by checking that the injected Δρ/ρ(z) histories agree between
  the two codes (CLASS can output the heating rate — `heating` output /
  `sd_...` verbose files) to <0.1 % before comparing distortions. If the
  heating histories don't match, the distortion comparison is meaningless —
  fix the mapping first.
- **Branching-ratio vs visibility methodology.** CLASS 'sharp' options and
  our Chluba-2013 J-fits are different approximations of the same
  thermalization physics; 5–10 % μ differences in the transition era
  (10⁴ ≲ z_h ≲ 3×10⁵) are expected and documented in Lucca+2020 itself
  (their Fig. comparing approximations). Do not chase these to zero. The
  strong statement is: our *PDE* should agree with CLASS's `exact`
  (CosmoTherm-derived Green's data) to a few %, because that path embeds
  actual CosmoTherm solutions — this is an indirect CosmoTherm comparison,
  say so explicitly in the memo.
- **X_e differences.** CLASS uses RECFAST/HyRec; we use Peebles+Saha. Round
  1 quantified this (xe_hyrec_comparison.md): P_s ≤0.9 %, y_γ ≤1.6 %. For
  heat injection μ/y the X_e sensitivity is weaker still; cite the Round-1
  numbers rather than re-deriving.
- **CLASS build failures in sandbox.** `make -j` occasionally trips on
  OpenMP flags in minimal containers; `make class OMPFLAG=` is the fallback.
  Record whatever was needed.
- **Do not report agreement to more digits than the CLASS output files
  carry** (they are text files with ~6 significant digits).

**Gate:** memo complete; every >2 % discrepancy decomposed; PDE-vs-`exact`
agreement number stated with its dominant attributed cause.

---

## Workstream R2 — Mutation-testing audit of the test suite

**Directly answers "the tests were written by the same agent that wrote the
code, so how do you know they can catch anything?"** Mutation testing
injects deliberate defects (sign flips, off-by-one, operator swaps, constant
perturbations) and measures whether the suite kills them. A quantified
mutation score on the physics modules, with every survivor triaged, is an
objective test-adequacy metric no amount of hand-written testing rhetoric
matches. It is also the mechanized generalization of CLAUDE.md pitfall #9.

### R2.1 Setup

- Install: `cargo install --git https://github.com/sourcefrog/cargo-mutants`
  (crates.io is sandbox-blocked; the git install works via allowed
  github.com). Pin the installed version in the memo.
- Read `cargo mutants --help` before assuming flags; the notes below are
  from memory of the tool and MUST be re-verified against the installed
  version (this is a known Opus failure point: confidently using
  remembered-but-renamed CLI flags).

### R2.2 Scoping — this is where naive runs die

A whole-crate run with release integration tests is O(days). Control it:

- **Target files only:** tier 1 = the six physics-critical modules, one run
  each: `src/kompaneets.rs`, `src/solver.rs` (the main integrator — the
  highest-consequence file; adaptive stepping, operator coupling, and the
  T_e feedback wiring all live here), `src/double_compton.rs`,
  `src/bremsstrahlung.rs`, `src/electron_temp.rs`, `src/recombination.rs`.
  Tier 2: `src/greens.rs`, `src/distortion.rs`, `src/cosmology.rs`,
  `src/dark_photon.rs`, `src/spectrum.rs`, `src/grid.rs`,
  `src/energy_injection.rs`. Use `-f <file>` per shard. Skip
  `main.rs`/`cli.rs`/`output.rs` (I/O plumbing; note the exclusion in the
  memo so the headline score is honest about its denominator).
- **Test selection:** run with release profile and a fast-but-sensitive test
  subset first: the lib unit tests plus `greens_function_checks`,
  `mms_convergence`, `conservation_fuzz`, `coverage_gaps`,
  `convergence_order`. Check how cargo-mutants passes through cargo/test
  args (`--profile`/`--release`-equivalent and `--` passthrough) in the
  installed version's help. If mutants survive the fast set, escalate those
  files to the full `heat_injection` suite.
- **Timeouts:** set an explicit per-mutant timeout (~3–5× the baseline test
  time; cargo-mutants measures a baseline first). Some mutations create
  non-converging Newton loops → infinite hangs; the timeout converts these
  to "killed (timeout)", which counts as caught.
- **Run detached** (directive 3). Budget: expect several hours per module
  shard; run shards sequentially overnight, poll logs. cargo-mutants writes
  `mutants.out/` with `outcomes.json`, `caught.txt`, `missed.txt` — parse
  those, don't scrape stdout.

### R2.3 Triage protocol (the deliverable is the triage, not the score)

Every entry in `missed.txt` (surviving mutant) gets one of:

1. **Test gap (physics-visible):** the mutation changes a physical
   prediction and no test noticed → write a new test anchored to an
   *external* target (analytic/literature — NOT to current code output; do
   not "fix" a survivor by pinning today's number, that recreates pitfall
   #9), then re-run that mutant to confirm the kill.
2. **Equivalent/benign mutant:** provably no observable effect (e.g.
   mutation in a dead diagnostic branch, or `<` → `<=` on a continuous
   quantity). Requires a one-line proof in the memo, not a shrug.
3. **Unreachable-in-domain:** the mutated branch only activates outside the
   validated parameter domain. Document, and check whether `_validation.py`
   / solver guards actually exclude that domain.

Also record mutants killed only by timeout separately — they indicate
robustness boundaries worth knowing.

### R2.4 Anticipated sticking points (directives)

- **`get_unchecked` blocks in kompaneets.rs:** mutations to the `assert!`
  guards can turn UB-protection off without failing any test. Any surviving
  mutant touching an assert guard is automatically class-1 (test gap) —
  guard asserts must be tested (see R4, Miri, which covers the same risk
  from the other side).
- **Doc-comment / logging mutants** inflate the denominator; exclude or
  bucket them so the headline score reflects physics code.
- **Do not weaken timeouts or skip slow files to make the score look
  better.** If `kompaneets.rs` is too slow to finish, say so and shard it
  by function (`-F`/regex options if the installed version supports them —
  check help).
- **cargo-mutants and the zero-dep policy:** it is a dev *tool*, it touches
  nothing in `Cargo.toml`. If it proposes adding config, put it in
  `.cargo/mutants.toml` and commit it so the run is reproducible.

### R2.5 Python side — the published limits pipeline

The FIRAS limits in the paper are produced by *Python* (`firas.py` GLS/
profiling, `greens.py`, `dark_photon.py`, `greens_table.py`), so Rust-only
mutation testing leaves the most limit-proximate code unmeasured. Run
`mutmut` (install via pip from PyPI — if PyPI is sandbox-blocked, `pip
install git+https://github.com/boxed/mutmut` via allowed github.com; pin
version) scoped to those four modules, killed by the Python suite
(`python/tests/`, 327 tests incl. parity + FIRAS anchors — fast, so no
sharding pain). Same triage protocol as R2.3. Watch specifically for
survivors in `firas.py`'s covariance/marginalisation code paths — Round 1's
P1-5 showed the anchor tests are the only thing pinning those conventions;
mutation results tell you whether they pin them tightly enough. Note:
mutmut's cache/config lives in `setup.cfg`/`pyproject.toml` keys — check
the installed version's docs rather than guessing key names.

**Artifacts:** `dev/audit/mutation_audit.md` — per-module table
(mutants generated / killed / timeout / survived), full survivor triage, new
tests listed, and the headline: "N physics-module mutants, M survivors, all
triaged; K new externally-anchored tests added." CI is NOT extended with
mutation runs (too slow); instead commit `mutants.out/outcomes.json`
snapshots under `dev/audit/mutation/` for auditability.

**Gate:** zero un-triaged survivors in the six tier-1 Rust modules and the
four Python limit-pipeline modules.

---

## Workstream R3 — Clean-room reference solver (N-version check)

A small, slow, deliberately different solver, written from the *papers*, that
never touches spectroxide source. Two implementations agreeing when they
share no code, no discretization, and no author-context is strong evidence
neither has a scheme-level bug. Round 1's MMS proved we solve our discrete
equations correctly at the design order; this workstream defends against
"you discretized the wrong equations" — from a direction the module audits
(equation-reading) cannot.

### R3.1 Isolation rules (absolute)

- New directory `dev/refsolver/`, pure Python + NumPy/SciPy (available:
  scipy 1.11.4).
- **The implementing agent must NOT read `src/*.rs` or
  `python/spectroxide/greens.py`/`solver.py`.** Allowed inputs: the
  Kompaneets equation and DC/BR emission terms from the primary papers
  (Kompaneets 1956 form as given in Chluba & Sunyaev 2012, arXiv:1109.6552;
  BR Gaunt from Chluba, Ravenni & Bolliet 2020, arXiv:1911.08861), the
  cosmology parameter values from the *paper* (Chluba 2013 defaults), and
  the I/O contract below. Run this as a separate fresh-context subagent
  whose prompt contains only the papers + contract; state in the memo that
  this isolation was enforced and how.
- To remove X_e/T_e-history ambiguity, the contract includes a *frozen
  ingredient table*: spectroxide exports `dev/refsolver/inputs/history.csv`
  with columns z, X_e, H(z), n_e, t_C — the reference solver consumes this
  table rather than re-deriving recombination. This isolates the PDE
  numerics (the thing being cross-checked); recombination already has its
  own external anchor (HyRec-2, Round 1).

### R3.2 Different by construction

- **Discretization: Chang–Cooper (1970)** finite-volume scheme for the
  Fokker–Planck/Kompaneets operator, solving for the FULL occupation n (not
  Δn). Chang–Cooper is positivity-preserving and reproduces the exact
  Bose–Einstein equilibrium of the discrete operator — a genuinely
  different scheme from our Crank–Nicolson-on-Δn with analytic Planck
  subtraction. Because it evolves full n in double precision, it will be
  *less* accurate for tiny distortions (the 10⁻⁵ signal sits on an O(1)
  background) — that is fine and expected; choose test amplitudes large
  enough to be resolvable (Δρ/ρ ~ 10⁻³–10⁻²; both codes are in the linear
  regime there, and spectroxide's nonlinear Δn² term is negligible — verify
  by amplitude-scaling both).
- Time integration: implicit Euler or TR-BDF2 via `scipy.linalg.solve_banded`
  on the tridiagonal system; fixed log-z steps, no adaptivity.
- Electron temperature: implement the quasi-stationary ρ_eq from the
  Compton-equilibrium integral ∫x⁴(n+n²)dx / 4∫x³n dx directly (full n
  makes this well-conditioned enough at the amplitudes above — the
  cancellation pitfall #4 applies to Δn formulations at 10⁻⁵, not full-n at
  10⁻³; verify conditioning numerically and report it).

### R3.3 Comparison cases & acceptance

Five cases: three heat-injection bursts (z_h = 2×10⁶, 2×10⁵, 5×10³),
adiabatic cooling, and **one monochromatic photon injection**
(x_inj ≈ 0.1, z_h ≈ 3×10⁵ — μ-era, so the outcome is a thermalized μ with
both energy and number contributions; this is the only independent-code
check the photon-injection channel gets, see R0). For the photon case the
contract specifies the injection as a narrow Gaussian in x with stated
width and total ΔN/N (a numerical delta is grid-dependent — pinning the
shape avoids a fake discrepancy from delta-representation differences),
and the comparison includes the photon-number ledger of both codes.
Chang–Cooper's positivity-preservation makes it well-suited to the sharp
bump. Compare μ, y, ΔT/T (decomposed by a shared, contract-specified
least-squares recipe — put the decomposition formulas in the contract so
both sides implement them from text) and the spectrum Δn(x) at z=0.

- Acceptance: μ within 2 % in the deep μ-era, y within 3 % in the y-era,
  transition-era within 5 %, spectra within ~5 % pointwise where
  |Δn| > 1 % of its peak. These bands come from Round 1's error budget
  (~0.3 % spectroxide discretization error) plus the reference solver's own
  estimated error (measure it by grid-doubling the reference solver).
- Disagreement outside bands → directive 1 (decompose, don't tune). The
  first suspects, in order: heating-normalization convention (Δρ/ρ
  definition instant vs integrated), grid extent (x_max ≥ 30 needed for
  energy integrals — this is CLAUDE.md pitfall #7 and WILL bite the
  reference solver too; the contract states the required domain), and
  Chang–Cooper's first-order-in-time error (halve the step and re-check
  before claiming a real discrepancy).

### R3.4 Anticipated sticking points

- **Contamination temptation:** the moment something disagrees, the natural
  move is "let me look at how spectroxide does it." Forbidden — that
  destroys the independence claim. Debug the reference solver against the
  papers and against its own convergence only. If truly stuck, the
  *orchestrating* agent (which may read spectroxide) may inspect BOTH and
  adjudicate, but the memo must then record that independence was broken
  for that item and what was found.
- **Stiffness at high z:** at z ~ 10⁶ the Compton y per step is large;
  implicit stepping handles it but Newton-free linearized implicit steps
  may need small dz — accept slowness (hours is fine, it runs once).
- **Numerical μ extraction:** fitting μ from full n requires subtracting
  the evolved-temperature Planck spectrum; the contract's decomposition
  recipe must pin the reference-temperature convention (fit ΔT jointly —
  same as our three-parameter least squares). Ambiguity here produces fake
  10 % discrepancies; that is why the recipe lives in the contract.

**Artifacts:** `dev/refsolver/` (solver + README + contract.md),
`dev/audit/refsolver_comparison.md` with the table, ratio plots in
`dev/output/refsolver/`.

**Gate:** all five cases within bands, or every excursion decomposed and
attributed.

---

## Workstream R4 — Machine-checked numerics: high-precision oracles + Miri on the unsafe kernel

Cheap, fast, and referee-legible: replace "we argue this cancellation is
handled" with "a 50-digit computation confirms it across the whole switch
domain," and "the unsafe indexing is guarded by asserts" with "the kernel
passes Miri."

### R4.1 mpmath oracles for the cancellation-critical scalar paths

Script `dev/scripts/highprec_oracle.py` (mpmath available, v1.2.1), memo
section in `dev/audit/highprec_numerics.md`:

1. **Constants:** recompute every constant defined in `src/constants.rs`
   (G₁, G₂, G₃, β_μ, κ_c, …) from its *defining integral or series* as
   stated in the docstring — at 50 digits with mpmath quadrature — and diff
   against the hard-coded values. Report digits of agreement for each. If a
   docstring does not state the definition precisely enough to recompute
   from, that is itself a (doc) finding.
2. **Pitfall #5 branch (DC/BR source near-cancellation):** compute
   n_pl(x/ρ_e) − n_pl(x) at 50 digits over a (x, ρ_e) grid straddling the
   |ρ_e−1| = 0.01 switch (ρ_e−1 ∈ ±[10⁻⁸, 10⁻¹] log-spaced, x ∈ [10⁻⁴, 30]).
   Evaluate BOTH code branches (naive difference and the analytic
   expansion, transcribed into Python from the Rust source with the
   transcription shown in the memo) in float64 against the oracle. Deliver
   the max-relative-error map and verify (a) the expansion branch is
   accurate where used, (b) the naive branch is accurate where used,
   (c) the crossover at 0.01 is on the correct side for both. If the map
   shows a region where NEITHER branch achieves the accuracy the error
   budget assumes, that is a finding.
3. **Pitfall #4 (perturbative Δρ_eq):** for a family of test distortions
   (μ-type, y-type, frozen bump), compute ρ_eq = I₄/4G₃ at 50 digits and
   confirm the code's perturbative Δρ_eq matches to O(Δn²), while the
   float64 full-integral route shows the claimed ~10⁻³ noise floor. This
   turns a CLAUDE.md war story into a plotted, checkable statement.
4. **Pitfall #1 (Kompaneets flux splitting):** verify at high precision that
   the split flux with the analytic n_pl(1+n_pl) term equals the unsplit
   flux for exact inputs, and quantify the float64 error of the naive
   finite-difference form it replaced (the claimed ~1000× signal ratio).

Sticking points: mpmath quadrature over [0, ∞) needs the standard
`mp.quad(f, [0, mp.inf])` with the integrand written to avoid overflow at
large x (factor out e^{−x}); set `mp.dps = 60` and verify stability by
re-running at dps = 80. Do not transcribe Rust expressions by paraphrase —
copy them term-by-term and show the mapping table in the memo.

### R4.2 Miri on the unsafe kernel

The `get_unchecked` hot loops in `kompaneets.rs` (Thomas solver, K_old
precompute, Newton inner loop) are guarded by entry asserts; Miri
machine-checks that no UB occurs on the exercised paths.

- `rustup +nightly component add miri` (nightly toolchain may need
  `rustup toolchain install nightly` first; both hit static.rust-lang.org —
  if sandbox-blocked, ask EB / flag for one unsandboxed command).
- Miri is 100–1000× slower than native and cannot run the release physics
  suite. Add a tiny dedicated test set: `#[cfg(miri)]`-friendly kernel
  tests with N ≈ 32–64 grid points and ~5 steps exercising every unsafe
  block (pure Compton step, coupled DC/BR Newton step, edge grids: min N
  the asserts allow, refinement-zone grid). If existing unit tests already
  qualify, filter to them: `cargo +nightly miri test --lib <filter>`.
- File I/O and `Instant::now` fail under Miri isolation; either avoid in
  the selected tests or set `MIRIFLAGS="-Zmiri-disable-isolation"`.
- Also run the same selected tests under
  `RUSTFLAGS="-Zsanitizer=..."`? No — skip sanitizers (target/toolchain
  friction, Miri already covers UB for these paths). Do add a plain
  `cargo test` (debug) run of just those kernel tests to CI so the
  `debug_assert!` input validation executes somewhere in CI; keep it to the
  small-N tests so it stays fast (this does not violate the release-only
  rule, which exists because the *full* suite is too slow in debug — say so
  in the CI config comment).
- Deliverable: CI job `miri-kernel` (nightly, the selected tests) so the
  soundness check is continuous, + a paragraph in highprec_numerics.md.

Sticking point: if Miri flags something, it is almost certainly real
(aliasing or an assert that does not actually imply the access bounds) —
treat as a confirmed finding, full B5 protocol; do NOT silence with
`-Zmiri-disable-stacked-borrows`-style flags.

**Gate:** oracle memo complete with error maps; Miri green in CI on all
unsafe-path kernel tests.

---

## Workstream R5 — Literature-figure regression suite (human-in-the-loop)

Round 1 anchored *coefficients* to papers; this anchors *curves*. Digitized
points from published figures become a regression suite: μ(z_h) and y(z_h)
Green's-function curves vs Chluba 2013 Fig.; photon-injection
μ(x_inj, z_h) surfaces vs Chluba 2015 / Bolliet, Chluba & Battye 2020;
dark-photon γ_con vs CCJ24 where relevant (partially done — see memory of
the Fig. 8 / CCJ24 statistic reproduction and the decoded Bryce figure).

Division of labor — **the agent must not invent digitized data** (directive
2). Agent prepares, EB digitizes:

1. Agent compiles `dev/audit/digitization_request.md`: exact figure list
   (paper, figure number, panel, which curves, expected axis ranges and
   scales), a CSV schema (`x, y, curve_id`), and for each figure the
   spectroxide command that will generate the comparison curve.
2. Agent first checks whether machine-readable data already exists —
   Chluba's webpage hosts Green's-function data for several papers, and
   arXiv source tarballs sometimes contain the plotted data or `.txt`
   tables next to the figures (fetch `arxiv.org/e-print/<id>` and look).
   Anything found this way skips manual digitization and is a better
   anchor; document provenance.
   **Dark photon: the AxionLimits repository is already cloned at
   `dev/AxionLimits/` with machine-readable limit curves under
   `limit_data/DarkPhoton/`** (incl. FIRAS-based ε(m) limits from the
   literature). Compare our dark-photon ε limits against the relevant
   published curves from there — but like-for-like only: match the source
   paper's statistic and confidence convention (Round 1's P1-5/P1-6 showed
   convention mismatch alone produces ~2× differences in limits), and
   record which AxionLimits file + upstream paper each comparison uses.
   Where our limit *should* differ from a published curve (different
   statistic, different X_e treatment — cf. the +25 %/−10.5 % HyRec
   sensitivity at m ≈ 1.2–2.5×10⁻⁹ eV from Round 1, and the unresolved
   ~22 % γ_con offset vs the reference figure), the memo states the
   expected offset and checks the *observed* offset against it, rather
   than claiming raw agreement.
3. EB digitizes the remainder (WebPlotDigitizer) into
   `dev/audit/digitized/<paper>_<fig>.csv`.
4. Agent writes `python/tests/test_literature_curves.py`: compare within a
   tolerance that includes an explicit digitization-error term (state it:
   typically 2–5 % for log-log figure reads; estimate per-figure from axis
   span and marker size). Runs in CI, skips-with-notice if CSVs absent.

Sticking point: tolerance inflation. The point is not "we pass with 30 %
bars"; per-figure tolerances must be justified (digitization error +
Round-1 error budget + any known methodology delta), and any curve needing
>10 % must be explained in the memo, not just tolerated.

**Gate:** request file delivered to EB; tests merged (skipping) so the
suite activates as CSVs land.

---

## Workstream R6 — Reproducibility capsule (rides on Part A / Phase 3)

The pending benchmark-pack work (Part A of the Round-1 plan) plus one
referee-visible addition: a one-command, containerized regeneration of every
paper figure.

- `Dockerfile` (or `docker/` + compose): pinned Rust toolchain, pinned
  Python env (export the working env to a lock/requirements file with
  hashes), builds the crate, installs the package, runs
  `make figures` (new top-level Makefile target) which regenerates every
  `notebooks/figures/*.pdf` via the paper_figures notebooks / remake
  scripts, then writes `figures.manifest.json` with SHA256 of every
  produced PDF's *underlying data* (hash the plotted arrays dumped to CSV,
  NOT the PDF bytes — PDFs embed timestamps and are never
  byte-reproducible; this is a classic sticking point, pre-empt it).
- CI job (weekly cron, not per-push) runs the container and diffs the
  manifest against the committed one; drift → failure with the offending
  figure named.
- Include the `benchmarks/` pack (Part A spec, unchanged) in the same
  image; Zenodo archive = container recipe + pack + manifests.

Sticking points: Docker may be unavailable in this sandbox — build/test the
Makefile path natively; the Dockerfile can be verified by EB or in CI
(GitHub Actions has Docker). Notebook execution PATH/conda issues: per
CLAUDE.md, verify the miniforge python explicitly and fall back to
executing notebook code as scripts (`jupyter nbconvert --to notebook
--execute` with the *full path* to the correct jupyter).

**Gate:** `make figures` regenerates all paper figures from a clean checkout
natively; manifest committed; Dockerfile builds in CI.

---

## Phasing & prioritization

| Phase | Content | Effort (agent-days) | Referee value |
|---|---|---|---|
| R-0 | R0 (coverage matrix, first pass) | 0.5 | high — steers the rest |
| R-1 | R4 (mpmath oracles + Miri) — cheap, do first while CLASS builds | 1–2 | medium |
| R-2 | R1 (CLASS SD comparison) | 2–3 | **highest** |
| R-3 | R2 (mutation audit), shards run overnight during R-2/R-4 work | 2–3 + wall-clock | **high** |
| R-4 | R3 (clean-room solver) | 3–4 | high |
| R-5 | R5 (literature curves — agent part) + R6 (capsule) + fold results into paper appendix & referee reply | 2 | medium |

Run R2 shards in the background from day 1 (they are wall-clock-bound, not
attention-bound). R3 uses a fresh-context subagent (isolation rule) so it
can overlap with anything.

**Dependencies:** R6 depends on the still-pending Part A Phase 3 benchmark
pack (do that first or in parallel — its spec is already written in the
Round-1 plan and is not duplicated here). R5 blocks on EB for digitization.
Nothing else blocks on EB except listed decisions (P0-6 T_CMB convention is
still open and should be resolved before the capsule freezes conventions).

## Referee-facing deliverables summary

- `dev/audit/coverage_matrix.md` — every published figure/number mapped to
  its strongest external anchor; becomes the paper-appendix validation
  matrix
- `dev/audit/class_sd_comparison.md` — independent-code agreement for heat
  injection, incl. indirect CosmoTherm check via CLASS `exact` branching
  data
- `dev/audit/mutation_audit.md` + committed `outcomes.json` — quantified
  test-suite adequacy for both the Rust solver (incl. `solver.rs`) and the
  Python limits pipeline (`firas.py` etc.), all survivors triaged
- `dev/refsolver/` + `dev/audit/refsolver_comparison.md` — N-version
  agreement from a clean-room Chang–Cooper solver, incl. the only
  independent-code photon-injection check
- `dev/audit/highprec_numerics.md` + `miri-kernel` CI job — machine-checked
  cancellation handling and UB-freedom of the unsafe kernel
- `python/tests/test_literature_curves.py` + digitized anchors — published
  curves as regression tests
- Reproducibility capsule (Docker + `make figures` + manifests + Zenodo)
- One new paper-appendix table row per workstream in the validation matrix,
  and 1–2 paragraphs for the referee reply per workstream (draft them in
  each memo's final section as you go — do not leave reply-drafting to a
  context that has not seen the numbers)
