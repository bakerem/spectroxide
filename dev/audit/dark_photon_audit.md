# Module audit B1 — dark_photon.rs / dark_photon.py

Auditor: physics-inquisitor. Date: 2026-07-03. Branch: main.
Files: `src/dark_photon.rs`, `python/spectroxide/dark_photon.py`,
`src/energy_injection.rs` (DarkPhotonResonance IC).
Primary refs: CCJ24 = Chluba, Cyr & Johnson 2024 (arXiv:2409.12115) Eq. 6;
Mirizzi, Redondo & Sigl 2009 (arXiv:0901.0014); Caputo et al. 2020 (arXiv:2002.05165).

## Independent derivation (done BEFORE reading numeric output)

### Landau–Zener / narrow-width conversion probability
Two-level (γ, A') system, effective energies E = m²_eff/(2ω). Diagonal splitting
Δ(t) = (ω_pl² − m²)/(2ω); off-diagonal coupling g = ε m²/(2ω). LZ flip probability
for a single resonance crossing:

    P = 1 − exp(−2π g²/|dΔ/dt|)
    2π g²/|dΔ/dt| = 2π (ε m²/2ω)² / (|dω_pl²/dt|/2ω)
                  = π ε² m⁴ / (ω |dω_pl²/dt|).

At resonance ω_pl² = m², so |dω_pl²/dt| = m² |d ln ω_pl²/dt|, one factor m² cancels:

    P = 1 − exp( −π ε² m² / (ω |d ln ω_pl²/dt|) ).           (LZ)

Numerator is **ε² m²** (not m⁴) — the extra m² is eaten by ω_pl²=m² in the rate. This
is the crux and the code has it right.

### γ_con definition
Write P(x) ≡ 1 − exp(−γ_con/x), x = ω/kT_γ. Match to (LZ) with |d ln ω_pl²/dt| = H·d,
d ≡ |d ln ω_pl²/d ln a|:

    γ_con/x = π ε² m² / (ω H d)  ⇒  γ_con = π ε² m² / (kT_γ · H · d).   ✓ = CCJ24 Eq. 6

The kT_γ in the denominator is NOT a thermal average — it is simply the frequency
normalisation x = ω/kT_γ. The full blackbody weighting is carried through the PDE by
applying P(x) mode-by-mode in the IC, which is the correct (spectrally exact) treatment.

### d-factor
ω_pl² ∝ X_e (1+z)³ = X_e a⁻³ ⇒ d ln ω_pl²/d ln a = d ln X_e/d ln a − 3.
In eln a = −ln(1+z): d = |(1+z)(1/X_e)dX_e/dz + 3|. Fully-ionized ⇒ dX_e/dz→0 ⇒ d=3. ✓

### Plasma-frequency prefactor
ω_pl²(SI) = n_e e²/(ε₀ m_e); e² = 4πε₀ ℏc α ⇒ ω_pl² = 4πα n_e ℏc/m_e (rad/s)².
Dimensional check: n_e[m⁻³]·ℏ[J·s]·c[m/s]/m_e[kg] = 1/s². ✓ Matches code `factor`.

### Small-γ_con energy loss (sanity)
Δρ/ρ = ∫x³P n dx/∫x³n dx → γ_con·G2/G3 = 0.3702·γ_con for γ_con≪1.
Note G3/G2 = 2.701 = ⟨E⟩/kT, so evaluating P at the mean photon energy is *identical*
to the spectral integral at leading order (no 22% ambiguity from that choice).

## Equation ↔ code mapping (verdicts)

| Item | Code | Derivation | Verdict |
|---|---|---|---|
| ω_pl prefactor 4πα ℏc/m_e | dark_photon.rs:28 | 4πα n_e ℏc/m_e | CORRECT |
| ω_pl → eV via ħ_eV | :29 | ħ[eV·s]·√(…1/s²) | CORRECT |
| n_e = X_e·n_H | :26–27 | ω_pl²∝n_e | CORRECT |
| z_res: ω_pl(z_res)=m, bisection | :36–61 | monotone in physical history | CORRECT |
| d = \|(1+z)dlnXe/dz + 3\| | :64–73 | \|dlnXe/dlna − 3\| | CORRECT |
| γ_con = πε²m²/(d·kT_γ·H) | :89 | CCJ24 Eq. 6 | CORRECT |
| m² (not m⁴) numerator | :89 | ω_pl²=m² cancels one m² | CORRECT |
| P(x)=1−exp(−γ_con/x) | energy_injection.rs:984 | LZ single crossing | CORRECT |
| Δn = −P·n_pl (depletion, −) | :985 | photons → A', loss | CORRECT |
| IC installed at z_start=z_res | scenario docstring | impulsive crossing | CORRECT |

## Numerical spot-checks (after derivation)
- m=1e-7: z_res=3.210e4, d=3.0000, γ_con/ε²=9.318e10 (test asserts 9.3e10 ±20%). ✓
- Ionized-era d=3.0000 exactly for m ≥ 3e-9 (z_res ≳ 3000).
- Near/below He+H recombination d deviates: m=2e-9→z_res=2459, d=3.73; m=1e-9→z_res=1569,
  d=3.37. Here the centered finite difference on X_e drives d, and analytic-vs-FD or a
  different recombination history could shift γ_con by tens of %.

## Triage

CONFIRMED BUGS: none.

CONVENTION/robustness notes (not bugs):
- N1 (d-factor near recombination): For z_res ≲ 3000 the d-factor is set by dX_e/dz and
  becomes both physically large (up to ~3.7) and numerically FD-sensitive. This is the
  ONLY place my derivation leaves room for a sizeable (tens-of-%) γ_con difference against
  another code — it depends on the recombination history and on analytic vs finite-diff
  evaluation of d. In the μ/y-relevant era (z_res ≳ 1e4) d=3 is exact and unambiguous.
- N2 (NWA validity not enforced): Code warns only on z_res range (energy_injection.rs
  ~1076–1099), not on the adiabaticity/narrow-width condition itself. Acceptable, but the
  LZ single-crossing formula assumes non-adiabatic conversion (γ_con exponent small enough
  that resonance width ≪ Hubble). Fine for the constrained ε≪1 regime.
- N3 (single resonance): bisection returns one z_res. Physical X_e(z)·(1+z)³ is monotone,
  so no missed second crossing for standard histories.

## The known ~22% open thread (memory: axion-dp-distortion-fig-tshift-error)
My derivation PINS our γ_con as the LZ-exact CCJ24 Eq. 6 value with exact prefactor and
exact d=3 in the ionized era. It therefore does NOT support a 22% error *inside our code*.
Candidates for the reference figure's ~22%-smaller γ_con:
  (a) different cosmology (Ω_b, h) → different n_e → shifts z_res and thus m²/(T·H·d);
  (b) the reference evaluating conversion at a fixed characteristic frequency with a
      convention other than ⟨E⟩=2.701kT (though 2.701 reproduces the spectral integral);
  (c) d-factor differences IF the reference resonance sits near recombination (N1).
The thermal-averaging convention is NOT a viable 22% source at leading order (G3/G2 identity).
Net: the discrepancy is a cross-code/reference convention issue, not a defect in
dark_photon.{rs,py}.

## Cross-language parity
Rust and Python are formula-identical (same prefactor, same d expression, same γ_con,
same P(x)); parity harness (python/tests/test_parity.py) covers drift. Confirmed by
independent read, not just by tests.
