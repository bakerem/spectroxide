"""Per-step energy ledger: burst run minus no-injection run.

Reads the "LEDGER" lines that the instrumented build (ledger_and_fix_a.patch,
env SPX_LEDGER=1) prints to stderr, for a burst run and a zero-injection run
with identical steps. All quantities are in units of Delta rho / rho_gamma:

  inj    4 theta_z dtau delta_rho_inj        heat the gas row receives
  store  (4 theta_z / R) (rho_new - rho_old)  change of gas heat content
  adia   (4 theta_z / R) lambda dtau rho_new  adiabatic loss of gas heat
  T_g    inj - store - adia                   gas-side transfer to photons
  dEph   change of photon energy in the coupled Newton step

"predicted CN-lag" is 2 theta_z dtau (rho_old - rho_new) summed over steps:
the energy the photon rows miss because the Crank-Nicolson old half of the
Kompaneets (phi - 1) term uses the previous step's rho_e while the gas row is
backward Euler.

Usage: python ledger_summary.py BURST.err BASELINE.err [x]   (x prints per step)
"""

import numpy as np, sys

cols = "z dz dtau xe rho_old rho_new rho_eq R lam q_dt inj store adia dEph Eph".split()


def load(f):
    rows = [l.split()[1:] for l in open(f) if l.startswith("LEDGER ")]
    a = np.array(rows, float)
    return {c: a[:, i] for i, c in enumerate(cols)}


b = load(sys.argv[1])
z = load(sys.argv[2])
assert np.allclose(b["z"], z["z"])
d = {k: b[k] - z[k] for k in cols}
tg = d["inj"] - d["store"] - d["adia"]
print("sum q_dt (true injected)   %.5e" % b["q_dt"].sum())
print("sum inj (solver 4θ δρinj dτ) %.5e" % d["inj"].sum())
print("sum store  %.5e   sum adia %.5e" % (d["store"].sum(), d["adia"].sum()))
print("gas-side transfer T_g %.5e" % tg.sum())
print("photon dE (excess)    %.5e" % d["dEph"].sum())
print("mismatch dEph - T_g   %.5e" % (d["dEph"] - tg).sum())
if len(sys.argv) > 3:
    for i in range(len(b["z"])):
        print(
            "%8.1f %6.2f dtau=%.3e xe=%.3e drho_e=%.3e R/lam=%.2e q=%.3e inj=%.3e adia=%.3e dE=%.3e mis=%.3e"
            % (
                b["z"][i],
                b["dz"][i],
                b["dtau"][i],
                b["xe"][i],
                b["rho_new"][i] - z["rho_new"][i],
                b["R"][i] / b["lam"][i],
                b["q_dt"][i],
                d["inj"][i],
                d["adia"][i],
                d["dEph"][i],
                d["dEph"][i] - tg[i],
            )
        )
# predicted CN-vs-BE mismatch: photons see 1/2(rho_old+rho_new) - rho_eq, gas uses rho_new
T0 = float(__import__("os").environ.get("T0", "2.7255"))
th = 1.686370e-10 * T0 * (1 + b["z"])
pred = 0.5 * 4 * th * b["dtau"] * (d["rho_old"] - d["rho_new"])
print(
    "predicted CN-lag mismatch %.5e   (actual %.5e)"
    % (pred.sum(), (d["dEph"] - tg).sum())
)
# residual per step for first few / worst
res = d["dEph"] - tg - pred
print("residual after CN-lag term: %.5e" % res.sum())
