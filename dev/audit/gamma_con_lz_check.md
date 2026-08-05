# Landau-Zener check of γ_con (m = 1.0e-07 eV)
z_res = 3.21025e+04,  ω = T_cmb(z_res) = 7.5414e+00 eV
γ_con(ε=1, m) [code] = P_NWA(ω=T_cmb, ε=1) = 9.31781e+10
|d ln ω_pl²/d ln a| at z_res = 3.0000

| regime | ε | P_NWA=ε²γ_con(1) | P_LZ=1−e^(−P_NWA) | P_numeric | rel err |
|--------|---|------------------|-------------------|-----------|---------|
| non-adiabatic | 1.6380e-07 | 2.5000e-03 | 2.4969e-03 | 2.4786e-03 | 7.330e-03 |
| boundary | 3.2760e-06 | 1.0000e+00 | 6.3212e-01 | 6.3953e-01 | 1.172e-02 |
| adiabatic | 9.8280e-06 | 9.0000e+00 | 9.9988e-01 | 9.9945e-01 | 4.262e-04 |

Worst rel err (P_numeric vs P_LZ) = 1.172e-02 → NWA CONFIRMED (threshold 5%).

The numerically-integrated conversion probability through the actual ω_pl(z) profile matches the Landau-Zener / NWA formula across the non-adiabatic, boundary, and adiabatic regimes. The code's γ_con (= P_NWA at ω=T_cmb) is validated against the underlying mixing dynamics. The ~22% discrepancy vs Bryce's frozen-absorption curve (memory: axion-dp-distortion) is therefore NOT in γ_con — it lives elsewhere (frozen-vs-thermalized treatment).
