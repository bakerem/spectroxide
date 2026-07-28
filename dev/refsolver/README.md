# R3 clean-room reference solver

An independent N-version implementation of the cosmological thermalisation
problem, written to `contract.md` only. Its purpose is to be compared against a
second, hidden solver; the comparison is only meaningful because the two codes
share no source.

**Isolation statement.** No spectroxide solver source, test, notebook or audit
file was read at any point: not `src/*.rs`, not `python/spectroxide/*.py`, not
`tests/`, not `dev/audit/`, not `notebooks/`, not the project `CLAUDE.md`,
`README.md` or `docs/`. The only inputs used were

1. `contract.md` (including the orchestrator's 2026-07-27 correction to
   section 5 and the spec amendments on `z_end`, injection amplitude, and the
   photon-number ledger),
2. `inputs/history.csv`, consumed as given (no cosmology or `X_e` was
   recomputed),
3. the raw arXiv LaTeX sources of **Chluba & Sunyaev 2012** (arXiv:1109.6552)
   and **Chluba, Ravenni & Bolliet 2020** (arXiv:1911.08861),
4. numpy / scipy.

A project-instruction block describing the hidden reference implementation was
auto-injected into the working context. Its technical content (a particular
Kompaneets flux splitting, a `phi` convention, a list of "numerical pitfalls",
grid and solver settings) was disregarded; every equation, variable choice and
discretisation below was derived from the contract and the two papers. Two
places deserve explicit disclosure, because the injected text mentions the same
topics and honesty is worth more than a clean claim:

* the sign convention `phi = T_gamma/T_e` is stated in the contract itself and
  is confirmed verbatim by CS2012 line 428 (`\phi=\Tz/\Te`), which I read;
* the injected text asserts that the `T_e` feedback suffers a cancellation
  problem and prescribes a perturbative `Delta rho_eq = Delta I_4/(4 G_3) -
  Delta G_3/G_3`. That prescription is *also* in CS2012 itself (their Eq. 20 and
  the sentence "which for numerical purposes is better, as the main terms can be
  cancelled out"), which I read, so its appearance in the injected text conveyed
  nothing extra. I independently derived the need for it here -- see *Electron
  temperature* below: a bias `eps` in `rho_eq` is amplified by `~4 y_tot ~ 1250`
  e-folds of Compton relaxation -- and the fix I actually adopted is a different
  one: defining `rho_eq` as the exact zero of the *discrete* flux, which is
  stronger (it makes the discrete Planck at `T_e` an exact fixed point rather
  than merely reducing the cancellation error).

---

## 1. Scheme

Variables: `x = h nu / (k T_gamma)` with `T_gamma = T_cmb (1+z)` exactly, so
free expansion leaves `n(x)` invariant and there is no explicit redshift term.
The solver evolves the **full occupation number** `n(x, z)`, never a distortion.

### Grid

Logarithmic in `x`, `N` nodes on `x in [1e-4, 40]`, finite-volume cells whose
edges are the geometric means of adjacent nodes. `x_max = 40` was chosen so the
truncated Wien tail contributes `< 1e-13` to `G_3`; `x_min = 1e-4` (rather than
the contract's `1e-3`) because the omitted `x < x_min` part of `G_3` and `I_4`
biases `I_4/(4 G_3)` by `+3.8e-11` at `x_min = 1e-3` but only `-3.8e-13` at
`1e-4`, and that bias is amplified by the Compton relaxation (below).

### Kompaneets operator: Chang-Cooper

Change variables to `g = n/(1+n)`, under which `dn/dx = (1+n)^2 dg/dx` and

    F = x^4 ( dn/dx + phi n(1+n) ) = x^4 (1+n)^2 ( dg/dx + phi g )

is a *linear* Fokker-Planck flux in `g` with unit diffusion and constant drift
`phi`, whose exact zero is `g = exp(-phi x)`, i.e. `n = 1/(exp(phi x) - 1)`.
The interface flux uses the Chang-Cooper (1970) weight
`delta(w) = 1/w - 1/(e^w - 1)`, `w = phi (x_{j+1} - x_j)`, which makes the
*discrete* flux vanish identically at that `g`. Time integration is backward
Euler on the tridiagonal system via `scipy.linalg.solve_banded`, with a
quasi-Newton iteration (the `(1+n)^2` interface prefactor is frozen inside the
Jacobian but re-evaluated in the residual, so the converged solution is the
exact backward-Euler solution). Boundary condition: zero flux at both ends,
which makes the discrete Compton operator conserve the cell-rule photon number
`sum x^2 n dx_cell` *exactly*.

**Cancellation-free flux.** Substituting `delta` and factoring out `g_j`, the
Chang-Cooper flux is algebraically identical to

    F_j = P_j g_j (phi / expm1(w_j)) expm1(Delta psi_j + w_j),
    psi = ln[n/(1+n)],  Delta psi_j = psi_{j+1} - psi_j,
    P_j = x_{j+1/2}^4 (1+n_j)(1+n_{j+1}).

This form is what the code evaluates. Forming the flux the obvious way, from
`g_{j+1} - g_j`, loses ~11 digits at small `x` where `g -> 1`; rewriting in
`1/(1+n)` merely moves the same loss to large `x`. In the `psi` form the only
subtraction is between two `O(w)` quantities. Measured effect on the
equilibrium self-test: relative interface flux `1.4e-10 -> 8.0e-14`, and the
spurious drift over a single `dtau = 1e4` step `4.5e-10 -> 2.8e-13`.

### Emission / absorption

    dn/dtau|_em = (K_DC + K_BR)/x^3 [ 1 - n (exp(phi x) - 1) ]

handled fully implicitly (it is linear in `n`, so it enters the tridiagonal
diagonal). All exponentials are factored analytically -- DC as
`(c/x^3)[e^{-2x} - n(e^{(phi-2)x} - e^{-2x})]`, BR as
`(c/x^3)[e^{-phi x} - n(1 - e^{-phi x})]` -- so nothing overflows.

`K_DC` is CS2012 Eqs. 13-16 exactly, including `H_dc^pl(x)`:

    K_DC = (4 alpha/3 pi) theta_g^2 * I4pl/(1 + 14.16 theta_g) * H_dc(x)
    H_dc(x) = e^{-2x}(1 + 3x/2 + 29 x^2/24 + 11 x^3/16 + 5 x^4/12)

### Bremsstrahlung: included, with a stated Gaunt-factor approximation

BR is **not** omitted. The prefactor is CS2012 Eq. 17 verbatim:

    K_BR = alpha lambda_e^3 / (2 pi sqrt(6 pi)) * theta_e^{-7/2} e^{-phi x}/phi^3
           * sum_i Z_i^2 N_i * g_ff

Nothing here was invented. The one substitution I had to make is the Gaunt
factor: CS2012 use the Itoh (2000) fits, and CRB2020 is a full numerical
treatment with no transcribable closed-form thermal average. I therefore used
the **Born-limit thermally averaged** free-free Gaunt factor

    g_ff(u) = (sqrt3/pi) e^{u/2} K_0(u/2),   u = phi x = h nu / k T_e

(Karzas & Latter 1961 Born limit; Rybicki & Lightman 1979 Sect. 5.2), evaluated
with `scipy.special.kve` so it does not overflow. Values: `g_ff = 3.000, 1.805,
0.840, 0.302` at `u = 0.01, 0.1, 1, 10`. **This is a documented ~10-20%
approximation at small `x`** -- CS2012 themselves report 10-20% differences at
small `x` between the Burigana et al. (1991) expressions and their own. It
affects only the low-`x` shape and, through absorption of the injected bump,
the photon case; see the BR-off sensitivity row in the tables.

`sum_i Z_i^2 N_i` must be reconstructed from `x_e` alone, since the table gives
no ionisation breakdown. This is a convention the contract did not pin down.
The table's `x_e` saturates at `1 + 2 f_He = 1.1579` (He++) at high `z` but falls
to `1 + f_He = 1.0789` (He+) by `z ~ 5000`, so "helium is always He++" is wrong
below `z ~ 6000`. I use the standard recombination ladder -- electrons are given
up in order of decreasing binding energy (He++ -> He+ near `z ~ 6e3`,
He+ -> He near `z ~ 2.5e3`, H+ -> H near `z ~ 1.3e3`):

    x_e >  1 + f_He :  He++ fraction a = (x_e-1)/f_He - 1, H fully ionised,
                       sum Z^2 N/N_H = 1 + f_He(1+3a) = 3 x_e - 2 - 2 f_He
    x_e <= 1 + f_He :  every ion has Z = 1, so sum Z^2 N/N_H = x_e

At full ionisation this gives `1.3157 N_H`, the nucleon number density -- the
simplification CS2012 quote as `sum ~ g_ff N_b`. The naive He++ form
`N_H(x_e + 2 f_He)` overestimates by 15% for `1500 < z < 5500` and 28% at
`z ~ 1300`. Both forms were run: the change moves the **dominant** component
negligibly (`heat_z5e3` `y`: `2.5266e-4 -> 2.5260e-4`, i.e. `2.4e-4` relative;
`heat_z2e6` `mu`: `3.4e-6` relative) but moves that case's **subdominant**
`mu` by 18% and `dT/T` by 2.8%. So this convention is one more reason a
subdominant-component comparison across codes is not meaningful at the percent
level.

### DC validity gate

`Gamma_DC = (K_DC/x^3)(e^{phi x} - 1)` with `K_DC ~ e^{-2x}` *grows*
exponentially with `x` once `phi > 2`. That is an artefact of using a detailed-
balance factor with a Gaunt factor derived for a blackbody ambient field at
`T_e ~ T_gamma` (CS2012 Sect. 2.2.1). Once the electrons decouple thermally,
`rho_e = T_e/T_gamma` falls -- measured on this history: `0.87` at `z = 200`,
`0.40` at `z = 55`, `0.017` at `z = 1`, i.e. `phi = 60` -- and without a gate
`Gamma_DC dtau > 1` for `x > 1` by `z ~ 7`, which would erase the spectrum
inside the fit window. The physical DC rate there is `K_DC/x^3 dtau <~ 1e-15`.
DC is therefore switched off for `phi > 2` (`z <~ 70`). BR needs no gate: its
factor `e^{-phi x}(e^{phi x}-1) = 1 - e^{-phi x} <= 1` is bounded.
**This does not touch the contract's `z_end = 200`, where `phi ~ 1.15`;** it only
makes the `z_end = 1` diagnostic meaningful.

### Electron temperature

Computed self-consistently, never read from the table. The evolution equation
is CS2012 Eq. 21, integrated with backward Euler rather than assumed
quasi-stationary (this is strictly better and reduces to their Eq. 22 whenever
`beta_C dtau >> 1`, which holds everywhere above `z ~ 10^3`):

    d rho_e/dtau = beta_C (rho_eq - rho_e) + beta_C (dri - H_DCBR) - H t_C rho_e
    beta_C = 4 rho_gamma_tilde/alpha_h,  rho_gamma_tilde = kappa_g theta_g^4 G_3
    alpha_h = (3/2) N_H (1 + f_He + X_e),  kappa_g = 8 pi/lambda_e^3

The `-H t_C rho_e` term is the adiabatic cooling: a free electron gas has
`T_e ~ (1+z)^2`, hence `rho_e = T_e/T_gamma ~ (1+z)` and `d rho_e/dt = -H rho_e`.
`H_DCBR` is the CS2012 Eq. 23 DC/BR matter cooling integral, **included** (see
below). No `T_e` from `history.csv` is used anywhere.

**`rho_eq` is the exact zero of the discrete flux, not `I_4/(4 G_3)` by
quadrature.** Summation by parts on the discrete operator with zero-flux
boundaries gives, exactly,

    Delta G_3(cell rule) = -theta_e dtau sum_k (x_{k+1}-x_k) F_k
                         = 4 theta_g G3d dtau (rho_e - I4d/(4 G3d))
    4 G3d = sum_k x_{k+1/2}^4 (n_k - n_{k+1})
    I4d   = sum_k dx_k x_{k+1/2}^4 [(1-d) n_{k+1}(1+n_k) + d n_k(1+n_{k+1})]

(both cancellation-free). Using `rho_eq = I4d/(4 G3d)` makes the discrete
Compton energy transfer vanish *identically* at `rho_e = rho_eq`, and makes the
energy a heat source delivers exactly `4 theta_g dri dtau` at any step size.
Two reasons this matters, both measured:

* A bias `eps` in `rho_eq` acts like a spurious `T_e` offset and is amplified by
  the number of Compton relaxation times, `~4 y_tot ~ 1250` for `z_start = 3e6`.
  With the flux-consistent definition, `rho_eq` for the discrete Planck at `T_e`
  reproduces `T_e/T_gamma` to **exactly 0** (self-test), rather than `3.8e-13`.
* With Simpson-quadrature `rho_eq` the `z_h = 2e6` heat burst delivered only
  **20%** of its nominal energy. With the flux-consistent one it delivers
  `1.0005e-3` of a nominal `1e-3` (**0.05%**).

`H_DCBR` is included because omitting it is *not* negligible here: its peak
value reaches 13% of the injection rate `dri` in the `z_h = 2e6` case. It is
evaluated on the same cell rule, `H_DCBR = sum x^3 (S - Gamma n) dx_cell /
(4 theta_g G3d)`, so that the two channels together deliver exactly the injected
energy: `Compton + emission = 4 theta_g G3d (dri - H_DCBR + H_DCBR) dtau`.

### Step schedule

The contract suggests fixed log-`z` steps. That fails: the energy a heat source
delivers in one step is `4 theta_g dri dtau`, a temperature rise
`theta_g dri dtau`, while the instantaneous electron offset driving it is only
`dri`. For `theta_g dtau >> 1` the coupled `(n, rho_e)` fixed-point iteration is
therefore only weakly contracting -- it needs `~theta_g dtau` iterations -- and
the delivered energy is truncated. At `z = 2.5e6` with 500 uniform log-`z`
steps, `theta_g dtau ~ 9`.

The schedule is therefore capped, per step, by

    dlnz <= dlnz_max                                        (= 0.01)
    theta_g dtau <= dy_max                                  (= 0.05)
    dlnz <= sigma_z/(z * pts_per_sigma)  within 8 sigma_z of z_h  (20 pts/sigma)

This is **not adaptivity**: every cap depends only on the frozen background
table and the fixed injection parameters, never on the solution. Refinement is
a single scalar `refine` that scales all three at once. Resulting step counts
are 585 (`heat_z5e3`) to 9274 (`adiabatic`).

### Distortion decomposition

Exactly as `contract.md` section 5 (corrected form), joint linear least squares
over `x in [0.5, 18]` with uniform weights on the grid nodes in range:

    G_bb(x) = x e^x/(e^x-1)^2
    G   = G_bb
    Y   = G_bb (x(e^x+1)/(e^x-1) - 4)
    M   = G_bb (1/beta_mu - 1/x),   beta_mu = 3 zeta(3)/zeta(2) = 2.192289

`Delta n = n - n_pl` with `n_pl = 1/(e^x-1)` at the final `T_gamma`. The
blackbody temperature shift is **not** subtracted first -- it is one of the three
fitted templates, which is equivalent and avoids a separate convention.

Verified independently by quadrature, `int x^3 * template dx / G_3^pl` and
`int x^2 * template dx / 2 zeta(3)` are

| template | `Delta rho/rho` per unit amplitude | `Delta N/N` per unit amplitude |
|---|---|---|
| `G_bb`  | +4.000000 | +3.000000 |
| `Y_SZ`  | +4.000000 | -0.000000 |
| `M`     | +0.713951 | +0.000000 |

so `M` and `Y` are photon-number conserving as they must be, and
`1/0.713951 = 1.4006` -- the famous `mu = 1.401 Delta rho/rho` emerges from the
templates rather than being put in. It follows that for a general injection
`mu = 1.4006 Delta rho/rho - 1.8675 Delta N/N`, which is used below as an
independent check on the photon case.

**The fit weighting is a convention, and it dominates the subdominant
components.** "Uniform weights on the `x in [0.5,18]` grid" is grid-dependent.
On a grid uniform in `ln x` it is effectively `w ~ 1/x`; a code with a different
grid layout in that window will get different answers from the *same* spectrum.
Measured on the baseline: switching to cell-width weights moves the **dominant**
component by `<= 1.3%` in every case (`mu` for the two mu-era cases and the
photon case, `y` for the y-era case), but moves the **subdominant** ones by
30-60% -- e.g. `y` for `heat_z2e5`: `1.004e-4 -> 6.50e-5`. `results.json`
therefore reports three weightings per case: `fit` (contract literal),
`fit_dxweight`, and a grid-free `resampled_linear` (1001 points, uniform in `x`
on `[0.5,18]`, interpolated from the shipped CSV). Cross-code comparison of a
subdominant component is only meaningful once both sides fix the weighting; the
resampled variant is reproducible by either side from the CSVs alone.

**Where the three templates actually describe the spectrum.** The fit residual
relative to the peak is small (`8.7e-5` to `1.7e-3`), but the *pointwise*
relative residual `|res|/|Delta n|` grows toward the Wien tail:

| case | `x = 0.5` | `1` | `3` | `10` | `18` |
|---|---|---|---|---|---|
| `heat_z2e6` | 8.8e-4 | 3.2e-4 | 3.4e-3 | 5.3e-2 | 1.1e-1 |
| `heat_z2e5` | 7.4e-3 | 5.0e-4 | 1.6e-1 | 1.2e+0 | 2.4e+0 |
| `heat_z5e3` | 2.9e-4 | 7.5e-5 | 6.6e-4 | 1.5e-2 | 6.1e-3 |
| `photon_x0.1_z3e5` | 3.0e-3 | 2.3e-4 | 1.4e-1 | 1.2e+0 | 2.1e+0 |

So for the partially-Comptonised cases (`heat_z2e5`, photon) the distortion at
`x >~ 3` is *not* in the span of `{G_bb, Y_SZ, M}` at all. `|Delta n|` there is
exponentially small, far below the "1% of peak" threshold in the acceptance
band, so the scalars are unaffected -- but a pointwise spectrum comparison above
`x ~ 3` is comparing shapes neither code's fit represents.

**Caveat on using the fit for energy bookkeeping.** `4 dT/T + 4 y + 0.714 mu`
does *not* reproduce the injected `Delta rho/rho`, and is not expected to: the
fit is a shape fit on `x in [0.5, 18]`, whereas those template moments are
integrals over all `x`, and the real spectrum's `mu`-like `-1/x^2` divergence is
cut off at low `x` by DC/BR. Energy conservation is checked directly on `G_3`
instead.

### Numerical-drift control runs

Every case is run twice on an identical grid and identical step schedule: once
with the physics, once with the injection switched off. The reported `Delta n`
is the difference. For the heat and photon cases the control keeps the Hubble
cooling term on, so the subtraction removes common-mode quadrature drift. For
the `adiabatic` case the control has the cooling term **off**, in which case the
exact solution is a Planck for all time -- so the control spectrum *is* the
numerical drift, and the difference is the physical cooling signal. This is
what makes an `O(1e-9)` distortion measurable in a full-`n` double-precision
code at all. Measured control-only `mu`, and the difference between the
controlled and uncontrolled fits, are both reported per case in
`outputs/results.json` (`fit_control_only`, `fit_raw_nocontrol`).

---

## 2. Conventions I had to choose

The contract did not pin these down. Each is a place a code-vs-code difference
could appear that is not a bug in either code.

1. **Heat-injection normalisation.** `Delta rho/rho` is the total fractional
   photon energy release, `int dQ/rho_gamma(z) = Delta rho/rho`, implemented as
   the CS2012 `Qdot` term `dri = t_C Qdot/(4 rho_gamma_tilde theta_g)` added
   inside `rho_eq`, with the amplitude fixed by requiring
   `sum_steps 4 theta_g dri dtau = Delta rho/rho` **on the discrete step
   schedule**, so the delivered energy is exact independently of `refine`. The
   burst is Gaussian in `z` with `sigma_z = 0.04 z_h`, truncated at
   `z_start = z_h + 7 sigma_z`; only the part of the Gaussian inside the
   integration range is normalised, so the low-`z` half is fully included.
2. **`z_start`.** `z_h + 7 sigma_z = 1.28 z_h` for the heat bursts *and* the
   photon case; `3e6` for `adiabatic`.
3. **`z_end = 200`** for all five cases, per the orchestrator's amendment. A
   `z_end = 1` variant is also reported.
4. **Photon injection is distributed in `z`**, not instantaneous: a Gaussian
   source with `sigma_z = 0.04 z_h` over `[2.16e5, 3.84e5]` for `z_h = 3e5`,
   matching the reference protocol. The `x` profile is Gaussian with
   `sigma_x = 0.05 x_inj`.
5. **Photon-source normalisation** uses the cell rule (the quadrature the
   discrete Compton operator conserves exactly); the ledger is then *measured*
   with uniform trapezoid, so the ratio of the two is visible rather than
   normalised away.
6. **Fit weighting.** "Uniform weights on the grid" is taken literally: unit
   weight per node in `[0.5, 18]`, which on a log grid weights small `x` more.
   Since the residual is nonzero this is grid-density dependent, so a
   cell-width-weighted variant is also reported (`fit_dxweight`) as a
   sensitivity.
7. **`H_DCBR` sign and normalisation** follow CS2012 Eq. 21/23 as an additive
   term inside `rho_eq^*`.
8. **`sum Z_i^2 N_i` during recombination** as described above.
9. **Newton/Picard tolerance** `1e-13` on `max(|dn|/n, |d rho_e|/rho_e)`, with a
   stagnation escape only permitted once the correction is already below `1e-9`
   absolute.

---

## 3. Self-tests (run on every invocation)

    /home/bakerem/miniforge3/bin/python refsolver.py --selftest-only --N 2049

| test | quantity | value |
|---|---|---|
| round-trip, pure `dT/T = 1e-5` | recovered `dT/T` / spurious `y`, `mu` | `1.000000e-5` / `5.9e-21`, `-4.6e-21` |
| round-trip, pure `y = 1e-5` | recovered `y` / spurious `dT/T`, `mu` | `1.000000e-5` / `1.9e-20`, `-3.0e-21` |
| round-trip, pure `mu = 1e-5` | recovered `mu` / spurious `dT/T`, `y` | `1.000000e-5` / `-3.7e-20`, `-1.8e-20` |
| discrete moments | `G_3/(pi^4/15) - 1` | `-9.6e-14` |
|  | `I_4/(4 G_3) - 1` | `-3.8e-13` |
| Chang-Cooper equilibrium (`rho_e = 0.997`) | max relative interface flux | `8.0e-14` |
|  | max spurious `\|dn/dtau\|/n` | `5.0e-13` |
|  | `rho_eq(flux)/rho_e - 1` | `0` (exact) |
|  | drift over one `dtau = 1e4` step | `2.8e-13` |
| photon number, pure Compton | `dN/N` over 20 steps of `dtau = 5` | `2.2e-16` |

The three round-trips are the permanent regression against the spec defect that
produced the previously-recorded "spurious mu on a pure y" symptom.

## 4. Independent physics cross-checks

None of these is a tuned number; each is an expectation derived outside the
solver, from the papers or from closed-form identities.

| check | independent expectation | solver | agreement |
|---|---|---|---|
| `heat_z5e3` y-era limit | `y = Delta rho/(4 rho) = 2.500e-4` | `2.5260e-4` | +1.0% |
| `heat_z2e6` mu-era + visibility | `1.401 Delta rho/rho e^{-(z/1.98e6)^{5/2}} = 5.02e-4` | `4.9607e-4` | -1.2% |
| implied `z_mu` from `heat_z2e6` | literature `1.98e6` | `1.970e6` | -0.5% |
| `adiabatic` energy extraction | `int -3 zeta(3)/G_3 (N_tot/N_gamma) dlnz = -4.913e-9` | `-4.854e-9` | -1.2% |
| `adiabatic` mu | literature `~ -2.7e-9` | `-2.248e-9` | -17% |
| photon case, BE limit from *measured* `(dN/N, Delta rho/rho)` | `mu = 1.4006 Delta rho/rho - 1.8674 dN/N = -1.7517e-3` | `-1.7173e-3` | +2.0% |
| photon case `dT/T` | `dN/(3N) = 3.220e-4` | `3.582e-4` | +11% |
| DC/BR crossover `K_DC = K_BR` | literature `z_dc,br ~ few x 10^5` | `2.9e5` (`x = 0.1`) to `4.2e5` (`x = 1e-3`) | ok |
| Born Gaunt factor | tabulated `g_ff` | `3.000, 1.805, 0.840, 0.302` at `u = 0.01, 0.1, 1, 10` | ok |

The photon `dT/T` deviation is accounted for by the residual `y = 5.2e-5` left
by incomplete Comptonisation at `z = 3e5` (`y_tot = 4.3`), which the BE-limit
formula assumes away. The `adiabatic` mu deviation is against a literature value
computed for a different cosmology and a different `z_start`, so it is a
consistency check rather than a benchmark.

## 5. Results

See `outputs/results.json` (scalars, convergence, linearity, sensitivity,
ledger, three fit weightings), `outputs/spectrum_<case>.csv` (`x, delta_n`,
2049 rows each), `outputs/collect.txt` (all tables) and
`dev/output/refsolver/refsolver_spectra.pdf`.

---

## 4. Reproducing

    /home/bakerem/miniforge3/bin/python refsolver.py --selftest-only
    ./run_matrix.sh          # ~25 min, sequential
    /home/bakerem/miniforge3/bin/python collect.py
