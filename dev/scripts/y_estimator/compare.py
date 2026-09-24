"""Compare two mu/y estimators on the stored PDE spectra and the CosmoTherm GF database.

A. Paper appendix (Bianchini-Fabbian): Delta n = [BE(mu) - pl] + delta*G + y*Y on x in [0.5,18],
   unweighted trapezoid L2 on Delta n (linearised: span{G/x, G, Y} = span{M, G, Y}).
B. Table 1 visibility recipe per spectrum: NC-strip, cooling baseline subtracted where available,
   model (3/kappa_c) P M_nc + (J_y/4) Y_nc times drho, cost trapz[(x^3 (model-data))^2] on [0.5,20],
   P and J_y free (closed-form 2x2).
"""
import json, sys
import numpy as np
R = "/home/bakerem/spectroxide"
sys.path.insert(0, R + "/python")
from spectroxide.greens import mu_shape, y_shape, g_bb, j_y, j_mu, j_bb_star
from spectroxide.cosmotherm import load_greens_database, cosmotherm_gf_to_delta_n

G3 = np.pi**4 / 15 * 1.0  # int x^3 n_pl ... G3 = pi^4/15
G3 = 6.493939402266829  # pi^4/15
KAPPA = 2.1419
MU_E = 3.0 / KAPPA

def strip(x, dn):
    g = g_bb(x)
    return dn - np.trapz(x**2 * dn, x) / np.trapz(x**2 * g, x) * g

def est_bf(x, dn, lo=0.5, hi=18.0, power=0):
    """Linear 3-shape fit on Delta n (paper appendix, linearised; the Rust nonlinear fit agrees to
    1e-5). power=0: unweighted, as in the code. power=3: the same model fitted to x^power Delta n
    (intensity-weighted)."""
    m = (x >= lo) & (x <= hi); xm = x[m]
    A = np.array([g_bb(xm) / xm, g_bb(xm), y_shape(xm)]).T   # BE mu shape is -G/x
    w = np.empty_like(xm); w[1:-1] = 0.5 * (xm[2:] - xm[:-2]); w[0] = 0.5*(xm[1]-xm[0]); w[-1] = 0.5*(xm[-1]-xm[-2])
    w = w * xm ** (2 * power)
    s = np.sqrt(w)
    c, *_ = np.linalg.lstsq(A * s[:, None], dn[m] * s, rcond=None)
    return -c[0], c[1], c[2]          # mu, delta, y

def est_bf_nc(x, dn, lo=0.5, hi=18.0, power=0):
    """Appendix model with the temperature shift removed first: strip G_bb from the data and the
    shapes by photon-number conservation, then fit mu (BE shape -G/x) and y. power as in est_bf."""
    m = (x >= lo) & (x <= hi); xm = x[m]
    A = np.array([strip(x, g_bb(x) / x)[m], strip(x, y_shape(x))[m]]).T
    w = np.empty_like(xm); w[1:-1] = 0.5 * (xm[2:] - xm[:-2]); w[0] = 0.5*(xm[1]-xm[0]); w[-1] = 0.5*(xm[-1]-xm[-2])
    w = w * xm ** (2 * power); s = np.sqrt(w)
    c, *_ = np.linalg.lstsq(A * s[:, None], strip(x, dn)[m] * s, rcond=None)
    return -c[0], c[1]


def est_vis(x, dn_nc, drho, lo=0.5, hi=20.0):
    """Per-spectrum Table 1 recipe: returns P = J_mu J_bb*, J_y."""
    m = (x >= lo) & (x <= hi); xm = x[m]
    Mn = strip(x, mu_shape(x))[m]; Yn = strip(x, y_shape(x))[m]
    a1 = xm**3 * MU_E * Mn * drho; a2 = xm**3 * 0.25 * Yn * drho; d = xm**3 * dn_nc[m]
    ip = lambda u, v: np.trapz(u * v, xm)
    N = np.array([[ip(a1, a1), ip(a1, a2)], [ip(a2, a1), ip(a2, a2)]])
    b = np.array([ip(a1, d), ip(a2, d)])
    return np.linalg.solve(N, b)

out = {}
# --- PDE: stored 4000-point spectra (Table 1 inputs) ---
t = np.load(R + "/dev/data/visibility_table.npz"); base = np.load(R + "/dev/data/baseline_table.npz")
x = t["x"]; zp = t["z_h"]
rows = []
for i, z in enumerate(zp):
    dn = t["dn_raw"][i] * 1.0
    j = np.where(np.isclose(base["z_h"], z, rtol=1e-10))[0]
    dn_b = dn - (base["dn_baseline"][j[0]] if len(j) else 0.0)
    E = np.trapz(x**3 * dn_b, x) / G3
    mu, dl, y = est_bf(x, dn)
    mu3, dl3, y3 = est_bf(x, dn, power=3)
    mun, yn = est_bf_nc(x, dn)
    P, Jy = est_vis(x, strip(x, dn_b), E)
    rows.append(dict(z=float(z), E=float(E), bf_mu=mu/E, bf_4y=4*y/E, bf3_mu=mu3/E, bf3_4y=4*y3/E,
                     bfnc_mu=mun/E, bfnc_4y=4*yn/E, rust_4y=4*float(t["pde_y"][i])/E,
                     rust_mu=float(t["pde_mu"][i])/E, vis_P=float(P), vis_Jy=float(Jy)))
out["pde"] = rows
# --- CosmoTherm GF database ---
zc, xc, gth = load_greens_database(R + "/data/cosmotherm/Greens.v1.0.3/Gdatabase/Greens_data.dat")
rows = []
for k, z in enumerate(zc):
    dn = cosmotherm_gf_to_delta_n(xc, gth[:, k])
    E = np.trapz(xc**3 * dn, xc) / G3
    if not np.isfinite(E) or E <= 0: continue
    mu, dl, y = est_bf(xc, dn)
    mu3, dl3, y3 = est_bf(xc, dn, power=3)
    mun, yn = est_bf_nc(xc, dn)
    P, Jy = est_vis(xc, strip(xc, dn), E)
    rows.append(dict(z=float(z), E=float(E), bf_mu=mu/E, bf_4y=4*y/E, bf3_mu=mu3/E, bf3_4y=4*y3/E,
                     bfnc_mu=mun/E, bfnc_4y=4*yn/E, vis_P=float(P), vis_Jy=float(Jy)))
out["ct"] = rows
json.dump(out, open(sys.argv[1], "w"))
for tag in ("pde", "ct"):
    print(tag, "z  E  bf_mu  bf_4y  vis_P(=JmuJbb)  vis_Jy  | formula P  Jy")
    for r in out[tag]:
        if r["z"] in () : pass
    for r in out[tag][:: max(1, len(out[tag]) // 14)]:
        z = r["z"]
        print(f"  {z:9.3g} {r['E']:.4g} {r['bf_mu']:+.3f} {r['bf_4y']:.3f} {r['vis_P']:.3f} {r['vis_Jy']:.3f} | {j_mu(z)*j_bb_star(z):.3f} {j_y(z):.3f}"
              + (f"  rust4y={r['rust_4y']:.3f}" if 'rust_4y' in r else ""))
