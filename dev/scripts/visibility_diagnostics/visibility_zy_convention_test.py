#!/usr/bin/env python3
"""Is the +5% z_y offset in Fig. 2 a convention artefact?

z_y is the one visibility parameter carrying a systematic offset from Chluba
(2013): it accounts for 88% of the literature-vs-fit cost difference and is
stable to 1.35% across our own fit variants. This script tests the leading
candidate explanation — that the offset comes from how the unobservable
temperature shift / number-conservation residual is removed before the mu/y
split, which is a convention choice, not physics.

Three residual definitions, refitting (z_y, alpha_y) only with the other
parameters held at the paper fit:

  NC      number-conservation stripped (what the paper fit uses):
          enforce int x^2 Delta n dx = 0 by subtracting a multiple of G_bb.
  freeT   raw Delta n, with a free G_bb amplitude per redshift profiled out.
          This is what an observer sees: T_0 is unknown, so the blackbody
          direction carries no information.
  raw     raw Delta n, no removal at all, G_bb amplitude fixed by the Ansatz.

If z_y moves toward 6.0e4 under any definition, the offset is a convention
mismatch with Chluba's decomposition rather than a difference in the physics.
"""
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "python"))

from spectroxide.greens import mu_shape, y_shape, g_bb, KAPPA_C

MU_TO_ENERGY = 3.0 / KAPPA_C
REPO = pathlib.Path(__file__).resolve().parents[3]
DATADIR = REPO / "dev" / "data"

FIT = dict(z_y=63128.02326056765, a_y=2.653389404163302, z_mu=58428.69672709,
           a_mu=1.9465842847446648, A=0.9918167997907094, z_th=1.98e6,
           a_th=2.5, B=0.048279535221419465, beta=2.0719380110207646)
LIT_ZY, LIT_AY = 6.0e4, 2.58

data = np.load(DATADIR / "visibility_table.npz")
z_h, x_pde, dn_nc, dn_raw, drho = (data["z_h"], data["x"], data["dn_nc"],
                                   data["dn_raw"], data["drho"])

M_x, Y_x, G_x = mu_shape(x_pde), y_shape(x_pde), g_bb(x_pde)
G_int = np.trapz(x_pde**2 * G_x, x_pde)
M_nc = M_x - np.trapz(x_pde**2 * M_x, x_pde) / G_int * G_x
Y_nc = Y_x - np.trapz(x_pde**2 * Y_x, x_pde) / G_int * G_x

mask = (x_pde >= 0.5) & (x_pde <= 20.0)
w = x_pde[mask] ** 3

jm = 1.0 - np.exp(-(((1 + z_h) / FIT["z_mu"]) ** FIT["a_mu"]))
jb = np.maximum(FIT["A"] * np.exp(-((z_h / FIT["z_th"]) ** FIT["a_th"]))
                * (1 - FIT["B"] * (z_h / FIT["z_th"]) ** FIT["beta"]), 0.0)
a_mu_amp = MU_TO_ENERGY * jm * jb


def make_cost(mode):
    if mode == "NC":
        Mw, Yw = w * M_nc[mask], w * Y_nc[mask]
        Dw = w * dn_nc[:, mask]
        extra = None
    else:
        Mw, Yw = w * M_x[mask], w * Y_x[mask]
        Dw = w * dn_raw[:, mask]
        extra = w * G_x[mask]

    def cost(p):
        z_y, a_y = p
        jy = 1.0 / (1.0 + ((1 + z_h) / z_y) ** a_y)
        model = ((drho * a_mu_amp)[:, None] * Mw[None, :]
                 + (drho * 0.25 * jy)[:, None] * Yw[None, :])
        if mode == "raw":
            model = model + (drho * 0.25 * (1 - jb))[:, None] * extra[None, :]
        r = model - Dw
        if mode == "freeT":
            # profile out a free G_bb amplitude per redshift
            gg = float(extra @ extra)
            c = (r @ extra) / gg
            r = r - c[:, None] * extra[None, :]
        return float(np.sum(r * r))

    return cost


print(f"{'mode':7s} {'z_y fit':>10s} {'alpha_y fit':>12s} {'z_y vs lit':>11s} "
      f"{'C(fit)':>12s} {'C(lit z_y,a_y)':>15s} {'dC':>10s}")
print("-" * 82)
for mode in ("NC", "freeT", "raw"):
    cost = make_cost(mode)
    res = minimize(lambda t: cost([np.exp(t[0]), np.exp(t[1])]),
                   x0=[np.log(FIT["z_y"]), np.log(FIT["a_y"])],
                   method="Nelder-Mead",
                   options=dict(xatol=1e-9, fatol=1e-12, maxiter=20000))
    zy, ay = np.exp(res.x)
    c_fit = cost([zy, ay])
    c_lit = cost([LIT_ZY, LIT_AY])
    print(f"{mode:7s} {zy:10.1f} {ay:12.4f} {100 * (zy - LIT_ZY) / LIT_ZY:+10.2f}% "
          f"{c_fit:12.4f} {c_lit:15.4f} {c_lit - c_fit:10.4f}")

print("\nInterpretation: if z_y stays near 6.3e4 in all three modes, the offset is")
print("not a temperature-shift convention artefact and must be reported as a real")
print("(if observationally irrelevant) difference from Chluba's published J_y.")
