# Step 5 paper drafts (apply after the citation agent finishes with paper.tex)

## A. Crank–Nicolson old half (paper.tex line 1404, after Eq. cn)

Replace
  Crank--Nicolson is second-order accurate in time and unconditionally stable.
with
  Crank--Nicolson is second-order accurate in time\radd{ for $\dn$} and unconditionally stable.
  \radd{Both halves of Eq.~\eqref{eq:cn} use the electron temperature of the new time level: the old half takes the backward-Euler estimate of $\re^{n+1}$ that starts the Newton iteration of Appendix~\ref{app:te_numerics}.
  The heating term $(\phi-1)\,\npl(1+\npl)$ in Eq.~\eqref{eq:flux_discrete} is then backward Euler in $\re$, like the electron update of Eq.~\eqref{eq:te_be}, so the heat the electrons give up in a step matches the heat the photons gain.
  Evaluating the old half at $\re^n$ instead breaks this balance after recombination, where $\re$ changes by order $\Delta\ln X_e$ per step, and loses ${\sim}\,6\%$ of the heat from a burst at $z_h = 10^3$.}

Source for 6%: dev/audit/fix_a_cn_old_half_ab.md (baseline 0.938 delivered vs 0.99986 expected).

## B. Energy-conservation paragraph (lines 1533–1539)

(i) line 1533 property (i): "\rdel{the implicit time-stepping conserves energy to high accuracy by construction}\radd{the electron and photon updates use the same electron temperature in the heating term, so the heat the electrons lose in a step equals the heat the photons gain (Sec.~\ref{...})}"

(ii) lines 1538–1539: replace "The dominant source of leakage is the first-order backward-Euler DC/BR coupling ..." with the first-order-in-Δτ residual statement. NUMBERS PENDING the regenerated energy_conservation figure:
  - heat panel: deviation range for 3e3 ≤ z_h ≤ 3e6; low-z end; 1e6–2e6 band (expected 0.4–0.6%).
  - photon panel: x_inj ≥ 3 and x_inj = 0.5 statements.
