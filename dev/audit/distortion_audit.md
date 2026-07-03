# B1 Module Audit: `src/distortion.rs`

Auditor: physics-inquisitor protocol. Independent context; formulas derived
before reading code output. Repo: spectroxide, branch `main`. Date: 2026-07-03.

## 0. Independent derivation (done before reading code numerics)

Planck occupation `n_pl(x) = 1/(e^x-1)`. Define
`G_bb(x) ≡ -x dn_pl/dx = x e^x/(e^x-1)^2 = x n_pl(1+n_pl)` (temperature-shift
shape; `Δn_T = (ΔT/T) G_bb(x)`, standard result, e.g. Chluba & Sunyaev 2012 MNRAS
419,1294 Eq. 8).

μ-distortion: Bose-Einstein `n_BE(x,μ)=1/(e^{x+μ}-1)`; `∂n_BE/∂μ|_0 = -e^x/(e^x-1)^2
= -G_bb(x)/x`. The *number-conserving* μ shape (photon number fixed by
Compton scattering, only spectral shape changes) is the standard
Zel'dovich-Sunyaev / Illarionov-Sunyaev result
`M(x) = G_bb(x)[1/β_μ - 1/x]`, `β_μ = 3ζ(3)/ζ(2) = 3ζ(3)/G₁` with
`G₁=π²/6`. Numerically `β_μ = 2.192289…`. **The task brief for this audit
quoted "β_M = 3G₂/(2G₁) ≈ 0.4561"** — I computed `3G₂/(2G₁) = 3·2ζ(3)/(2·π²/6)
= 3ζ(3)/G₁ = 2.19229`, not 0.4561. `0.4561 = 1/2.19229 = 1/β_μ`. The brief's
printed *value* is the reciprocal of the correct β_μ (its formula is right,
its decimal is wrong). Treated as a plant/typo in the prompt itself, not a
codebase claim — see §2.1 for the code's actual (correct) constant.

y-distortion (non-relativistic SZ, Zel'dovich & Sunyaev 1969):
`Y_SZ(x) = G_bb(x)[x coth(x/2) - 4]`, null at `x≈3.830` (the well-known 217 GHz
SZ null).

Orthogonality/degeneracy: both `M` and `Y_SZ` are constructed to be
photon-number-conserving, i.e. `∫x² M dx = ∫x² Y_SZ dx = 0` over `[0,∞)`
(verified numerically below, §3.3). `G_bb` is *not* number-conserving
(`∫x² G_bb dx = G₂·(∂N/∂T)/N ≠ 0`) — a pure `ΔT/T` shift changes photon
number, which is physical (blackbody at higher T has more photons). So `{M,
Y_SZ, G_bb}` span a 3-D subspace of function space that is *not* mutually
orthogonal (`G_bb` has nonzero overlap with both `M` and `Y_SZ`), which is
exactly why a Gram-Schmidt or normal-equation joint fit — not independent
projections — is required to avoid μ↔y↔T leakage. When a residual (e.g. a
frozen, un-diffused post-recombination injection bump) is *not* in
`span{M,Y_SZ,G_bb}`, any linear projection (LS or Gram-Schmidt) returns the
*optimal in-band L²-projection* of that residual, not the "true" physical
amplitude — the leftover is captured in `residual` but the reported
(μ,y,ΔT/T) triple is then a *fit statistic*, not a first-principles physical
quantity, for such spectra. This must be documented as a domain-of-validity
caveat, which the module docstrings for `decompose_gram_schmidt` mostly do
but the caller-facing `decompose_distortion` docstring does not warn about
frozen/locked-in spectra explicitly (see §4, minor doc gap).

Intensity conversion: `I_ν = (2hν³/c²) n(x)`, `x=hν/kT`, standard specific
intensity (SI, W/m²/Hz/sr). Recomputed from CODATA-2018 exact SI values
(`h=6.62607015e-34 J·s`, `k_B=1.380649e-23 J/K`, `c=2.99792458e8 m/s`) at
`x=1, T=2.725 K`: `ν=56.78 GHz`, prefactor `2hν³/c² = 2.6991e-18 W/m²/Hz`,
i.e. `269.91 MJy/sr` per unit `Δn` (1 MJy ≡ 1e-20 W/m²/Hz, so multiply the SI
prefactor by `1e20`). This is what the code prefactor + `*1e20` conversion
must reproduce exactly (§3.4).

## 1. Files read

- `src/distortion.rs` (full, 817 lines incl. tests)
- `src/spectrum.rs` (`planck`, `bose_einstein`, `g_bb`, `mu_shape`, `y_shape`,
  `delta_rho_over_rho`, `delta_n_over_n`, `compton_equilibrium_ratio`)
- `src/constants.rs` (`BETA_MU`, `KAPPA_C`, `G1_PLANCK`, `G2_PLANCK`,
  `G3_PLANCK`, `HPLANCK`, `K_BOLTZMANN`, `C_LIGHT`)
- `python/spectroxide/cosmotherm.py::strip_gbb` (Python-side G_bb-stripping)
- call sites: `src/solver.rs` (`extract_mu_y_joint`, `save_snapshot_at`),
  `tests/heat_injection.rs`, `tests/cosmotherm_comparison.rs`,
  `examples/photon_diag.rs`

## 2. Equation ↔ code mapping

| # | Physics | Code | Verdict |
|---|---|---|---|
| 2.1 | `β_μ = 3ζ(3)/G₁ = 2.192289` | `constants.rs:105`: `BETA_MU = 3.0*ZETA_3/G1_PLANCK` | **Correct.** Recomputed independently to 2.19228890820…; matches to 15 digits. Confirms the audit-brief's 0.4561 is `1/β_μ`, a reciprocal slip in the brief, not a code bug. |
| 2.2 | `G_bb(x)=x n_pl(1+n_pl)` | `spectrum.rs g_bb`, small-x branch `1/x - x/12` | **Correct.** Independently Taylor-expanded `n_pl(1+n_pl)` to O(x²) and confirmed `x·n_pl(1+n_pl) = 1/x - x/12 + O(x³)`. |
| 2.3 | `M(x)=G_bb(x)[1/β_μ - 1/x]` | `spectrum.rs mu_shape`: `(x/BETA_MU - 1.0) * g_bb(x) / x` | **Correct**, algebraically identical: `(x/β_μ-1)/x = 1/β_μ - 1/x`. Zero-crossing at `x=β_μ≈2.19` (code docstring says so; test `test_mu_shape_sign_change_at_beta_mu` checks the sign flip, not a tautological `|M(β_μ)|<ε`, which is appropriately non-circular). |
| 2.4 | `Y_SZ(x)=G_bb(x)[x coth(x/2)-4]` | `spectrum.rs y_shape` | **Correct**, incl. correctly-derived small-x expansion `-2/x + x/3` (I independently re-derived this from `G_bb≈1/x-x/12` and `x coth(x/2)-4≈-2+x²/6`, matches code comment exactly). |
| 2.5 | Gram-Schmidt basis order `e_y, e_μ, e_T` and back-substitution formulas (`decompose_gram_schmidt`, lines 82–192) | Rust | **Correct — verified by independent linear-algebra derivation** (§3.1). Exact for any Δn ∈ span{M,Y_SZ,G_bb}; optimal L² projection otherwise. |
| 2.6 | B&F (2022) nonlinear-in-μ model `Δn = [n_BE(x+μ)-n_pl(x)] + δG_bb(x) + y·Y_SZ(x)` | `decompose_nonlinear_be` lines 217–370 | **Correct.** Model, Jacobian (`∂/∂μ = -e^{x+μ}/(e^{x+μ}-1)²`, incl. correct small-`(x+μ)` limit `-1/(x+μ)²`), and LM step/backtracking are all standard and were independently re-derived to match. |
| 2.7 | `δ_BF = δ_GS + μ/β_μ` offset relation between the two parameterisations | Doc comment lines 213–216, tested in `test_bf_vs_gs_pure_mu` etc. | **Correct** — I re-derived this: linearising B&F's BE term gives `n_BE(x+μ)-n_pl(x) ≈ -μ G_bb(x)/x = μ[M(x) - G_bb(x)/β_μ]`, so a pure-μ_BF input equals `μ·M(x) + (-μ/β_μ)·G_bb(x)` in the CJ2014 basis ⇒ `δ_GS = -μ/β_μ` for pure BE, i.e. `δ_BF(=0) = δ_GS + μ/β_μ`. Matches code and passes to 1e-3 in tests. |
| 2.8 | Intensity `ΔI_ν = (2hν³/c²) Δn`, `ν=xk_BT/h`, `×1e20` for W→MJy | `delta_n_to_intensity_mjy` lines 430–436 | **Correct.** Recomputed prefactor from CODATA-2018 exact SI constants: `269.91 MJy/sr` per unit Δn at x=1, T=2.725 K — matches the formula bit-for-bit (same constants, same combination). Sign/linearity spot-checked by existing unit test and confirmed physically sensible in magnitude (μ~1e-5 injection → ~kJy/sr scale, consistent with FIRAS-era distortion amplitudes). |
| 2.9 | Default decomposition band `x∈[0.5,18]` ↔ `ν∈[28,1020] GHz` | `DEFAULT_DECOMP_X_MIN/MAX` doc comment | **Correct.** Recomputed: `x=0.5→28.39 GHz`, `x=18→1022.0 GHz`. Matches docstring to quoted precision. |

## 3. Deep checks

### 3.1 Gram-Schmidt back-substitution (independent re-derivation)

Writing `M = m_y e_y + |M_⊥| e_μ`, `G = g_y e_y + g_μ e_μ + |G_⊥| e_T` (by
construction of Gram-Schmidt) and assuming `Δn = μM + yY_SZ + δG` exactly:

```
a_y ≡ ⟨Δn,e_y⟩ = μ m_y + y|Y_SZ| + δ g_y
a_μ ≡ ⟨Δn,e_μ⟩ = μ|M_⊥| + δ g_μ
a_T ≡ ⟨Δn,e_T⟩ = δ|G_⊥|
```

Solving the (lower-triangular) system top-down:
`δ = a_T/|G_⊥|`, `μ = (a_μ - δ g_μ)/|M_⊥|`, `y = (a_y - δ g_y - μ m_y)/|Y_SZ|`.

This is **exactly** what lines 174–176 compute (`delta_t = a_t/g_perp_norm`;
`mu = (a_mu - delta_t*g_mu)/m_perp_norm`; `y = (a_y - delta_t*g_y -
mu*m_y)/y_norm`). Confirmed algebraically identical, term for term. Because
this is an exact triangular back-substitution (not an approximate fit) when
`Δn` is in-span, recovery is exact to numerical-quadrature precision — this
matches the observed `<1e-4` test tolerances, which are floor-limited by
trapezoidal quadrature error on the log-grid, not by an algorithmic
approximation. **Verified correct.**

**Minor doc inconsistency (not a code bug):** the module-level docstring
(lines 74–81) writes `M_y = M·e_y·|Y_SZ|`, `G_y = G·e_y·|Y_SZ|`, `G_μ =
G·e_μ·|M⊥|` — i.e. defines these with an extra factor of `|Y_SZ|` or
`|M_⊥|` relative to the plain inner products `m_y, g_y, g_μ` actually used in
the code (and in my derivation above). Tracing through, the doc's back-sub
formulas as literally written do not reduce to the code's formulas unless
those extra factors are implicitly cancelled elsewhere in the doc's algebra,
which it does not show. This is a **stale/inconsistent comment**, not a
functional defect — the executable code matches my from-scratch derivation
and the passing tests, not the doc's variable definitions. This is the same
issue flagged in a prior audit pass (recorded as "distortion.rs:427
duplicated formula text" in agent memory) — location has drifted with edits;
now at lines 74–81. **Recommend**: fix the docstring to define `m_y ≡
⟨M,e_y⟩` etc. without the extra norm factors, matching the code.

### 3.2 Quadrature weights (`band_weights`, lines 36–58)

The inner products in both decomposition routines use **trapezoidal
quadrature weights on the actual non-uniform grid** (`w[i] =
0.5(x[i+1]-x[i-1])` for interior points), not naive unweighted point sums.
This is the physically correct choice for a non-uniform log/linear hybrid
grid and matches CLAUDE.md's documented convention. **Verified correct
in the generic case.**

**Boundary edge case (WARNING, not a confirmed bug in current usage):** the
half-weight rule is only applied at `i==0` / `i==n-1` of the *full* `x_grid`
array, not at the edges of the selected band `[x_min,x_max]`. If a caller's
`x_min`/`x_max` happens to coincide with the actual first/last grid point
(e.g. a custom decomposition on a grid pre-trimmed to exactly the band), the
point at the true integration boundary gets a full-width extrapolated weight
rather than a half-width one, mildly double-counting the boundary
contribution. In the default configuration this never triggers: `x_grid`
always extends well below `x_min=0.5` and above `x_max=18` (grids run to
`x_max≥30`, `x_min` as low as `1e-3`–`1e-6` per CLAUDE.md pitfall #7), so
every band point is an "interior" point of the full array and gets the
correct symmetric weight. Confirmed this by inspection of all current call
sites (`tests/heat_injection.rs`, `cosmotherm_comparison.rs`,
`examples/photon_diag.rs` all use grids with `x_min` grid values `<0.5` and
`x_max` grid values `>18`). **Recommend**: document this precondition
(`x_min`/`x_max` must be strictly interior to the supplied grid) or fix
`band_weights` to use true half-weights at the band edges regardless of
their position in the parent array, for robustness against future custom
narrow-grid callers.

### 3.3 μ↔y leakage for pure inputs (independent numerical check)

Analytically, if `∫x²M dx = ∫x²Y_SZ dx = 0` exactly over `[0,∞)`, injecting
pure `M(x)` or pure `Y_SZ(x)` and fitting with the exact triangular
back-substitution (§3.1) should recover the true coefficient with zero
leakage into the other channel, up to quadrature/truncation error. Verified
numerically (independent Python, trapezoidal, `x∈[1e-4,60]`,
2×10⁶ points): `∫x²M dx / G₂ ≈ 4.2e-5`, `∫x²Y_SZ dx/G₂ ≈ 4.2e-9` — both
consistent with zero up to finite-`x_max` truncation (the `M(x)` tail decays
only as `1/x` at large x after the `1/β_μ` term dominates, so its number
integral converges more slowly — expected, not a bug). This matches and
explains the existing Rust unit tests `test_decompose_pure_mu`,
`test_decompose_pure_y`, `test_gram_schmidt_pure_mu/y` (all passing, `<0.01`
and `<1e-4` leakage tolerances respectively — ran `cargo test --release --lib
distortion::` locally: **13/13 pass**). **Verified correct**; not a
circular/tautological test since the target (near-zero leakage) follows from
an independent analytic property of `M` and `Y_SZ`, not from a value read off
code output.

### 3.4 Intensity conversion — digit-by-digit constant check

`HPLANCK=6.626070150e-34`, `K_BOLTZMANN=1.380649e-23`,
`C_LIGHT=2.99792458e8` in `constants.rs` are the CODATA-2018 *exact* SI
values by definition (2019 SI redefinition) — confirmed exact, no rounding
error possible. The conversion formula multiplies by `1e20` for W/m²/Hz →
MJy, which is dimensionally exact (`1 Jy ≡ 1e-26 W m⁻² Hz⁻¹` ⇒ `1 MJy ≡
1e-20 W m⁻² Hz⁻¹`). **Verified correct**, no approximation or
literature-constant drift possible here since all inputs are SI-exact
by definition.

### 3.5 `strip_gbb` (Python) vs Rust — convention consistency

`python/spectroxide/cosmotherm.py::strip_gbb` implements a *different*
mechanism from anything in `distortion.rs`: it projects out the
number-conserving component via `α = ⟨x²Δn⟩/⟨x²G_bb⟩`, `Δn_stripped = Δn -
αG_bb`, used only for CosmoTherm-convention plotting/fitting comparisons in
`greens.py`. Rust's `number_conserving` flag (`solver.rs`) is an unrelated
*internal solver technique*: during time-stepping it subtracts an
accumulated `G_bb` component from `Δn` before feeding it back into DC/BR to
prevent spurious number-changing feedback, then adds the accumulated
temperature shift back into the reported `Δn` at snapshot time
(`save_snapshot_at`, lines 1675–1707). These are two independent mechanisms
solving two different problems (an *output-side* FIRAS-observability
convention in Python vs. an *internal solver-stability* convention in Rust);
there is no missing "Rust counterpart" to `strip_gbb` because
`distortion.rs` itself never needs to do photon-number stripping — its job
is μ/y/T decomposition of an already-computed `Δn`, and it takes the
`number_conserving`-corrected `full_delta_n` (T-shift added back in) as
input, per `solver.rs:1716`. **False alarm** relative to the audit-brief's
suggestion that these two might be convention-mismatched; they serve
different purposes and are each internally consistent.

`decompose_gram_schmidt` is used only inside `distortion.rs`'s own test
module (cross-validation against `decompose_nonlinear_be`); production code
(`solver.rs::extract_mu_y_joint`) and all integration tests exclusively call
`decompose_distortion` → `decompose_nonlinear_be` (B&F 2022). This matches
the module docstring ("Default method: ... B&F ... For the linear
alternative ... call `decompose_gram_schmidt` directly") — **no
inconsistency in usage**.

## 4. Triage summary

**CONFIRMED BUGS:** none.

**CONVENTION MISMATCH (documented, no fix needed):** none found beyond
what's already correctly documented (B&F vs CJ2014 ΔT/T offset, §2.7 — this
is a real parameterisation difference, correctly explained in code
comments and verified by tests).

**MINOR / DOC-ONLY ISSUES:**
1. Stale docstring at `distortion.rs:74–81` defines `M_y, G_y, G_μ` with
   spurious extra norm factors inconsistent with the code's actual
   `m_y, g_y, g_μ` (plain inner products). Code is correct (verified
   independently, §3.1); only the comment is wrong. Low priority, cosmetic.
2. `band_weights` boundary half-weight logic keys off the *parent array's*
   edges, not the *band's* edges — benign under all current call-site grid
   configurations (verified by inspection) but a latent footgun for a future
   caller who passes a grid pre-trimmed to exactly `[x_min,x_max]`. Recommend
   a doc-comment precondition or a robustness fix.

**FALSE ALARMS (refuted):**
1. Audit brief's stated `β_M ≈ 0.4561` — this is `1/β_μ`, not `β_μ`; the
   code's `BETA_MU=2.19229` is correct and independently re-derived from
   `3ζ(3)/G₁`.
2. Suspected missing Rust counterpart to Python's `strip_gbb` — the two
   "number-conserving" mechanisms in Python vs. Rust solve different
   problems (FIRAS-observable stripping vs. internal solver-stability
   projection) and are each self-consistent; `distortion.rs` has no need
   for a `strip_gbb` equivalent.

**VERIFIED CORRECT (full list):** `BETA_MU` value and formula; `G_bb`,
`mu_shape`, `y_shape` formulas and all small-x Taylor branches; Gram-Schmidt
basis construction and back-substitution algebra (exact, not approximate,
for in-span inputs); B&F nonlinear model and its Jacobian; the `δ_BF = δ_GS +
μ/β_μ` offset relation; trapezoidal quadrature-weighted (not unweighted)
inner products; intensity-conversion prefactor against CODATA-2018 exact SI
constants; default decomposition band GHz mapping; photon-number
orthogonality of `M` and `Y_SZ` (numerically, `∫x²{M,Y_SZ}dx ≈ 0`); usage
consistency (production always B&F, Gram-Schmidt only in cross-validation
tests). `cargo test --release --lib distortion::` → 13/13 pass, targets
independently derivable (not circular).

## 5. Recommendations

1. Fix the stale `M_y/G_y/G_μ` docstring definitions at lines 74–81 to match
   the code's plain inner-product variables (cosmetic, low priority).
2. Either document that `band_weights`/`decompose_*` require `x_min,x_max`
   strictly interior to the supplied grid, or patch `band_weights` to use
   true half-weights at the *band* edges rather than the parent-array edges.
3. Add an explicit docstring warning to `decompose_distortion` (not just
   `decompose_gram_schmidt`) that for spectra with support outside
   `span{M,Y_SZ,G_bb}` (e.g. frozen/locked-in post-recombination
   photon-injection bumps at `z<1100`), the returned `(μ,y,ΔT/T)` is an
   in-band L² best-fit, not a first-principles decomposition — the
   `residual` field must be inspected in that regime. This is physics
   already correctly implemented; the gap is purely in caller-facing
   documentation.
