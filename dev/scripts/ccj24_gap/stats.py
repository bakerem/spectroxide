"""Generic GLS limit machinery to walk CCJ24 statistic -> paper profile likelihood."""
import numpy as np
from scipy.optimize import minimize_scalar
from spectroxide import g_bb, planck, mu_shape, G1_PLANCK as G1, G2_PLANCK as G2
from spectroxide.firas import FIRASData, _dn_to_dI_kJy, _H_PLANCK, _K_BOLTZMANN, _C_LIGHT

F = FIRASData()
NU = F.freq_cm * _C_LIGHT * 100
PREF = 2 * _H_PLANCK * NU**3 / _C_LIGHT**2 / 1e-23  # kJy/sr per unit dn
IOBS = F.spectrum_MJy * 1e3
CINV = {'full': F.cov_inv, 'diag': np.diag(1 / F.sigma_kJy**2)}
GAL = F.galactic_template_kJy()
Z1645 = 1.6448536269514722


def xT(T):
    return _H_PLANCK * NU / (_K_BOLTZMANN * T)


def data(T, kind):
    """Residual w.r.t. B(T). 'resid': published column (exact at 2.725, linear
    shift otherwise); 'spec': I_obs(col2) - B(T)."""
    if kind == 'spec':
        return IOBS - PREF / np.expm1(xT(T))
    return F.residual_kJy + PREF * (1 / np.expm1(xT(2.725)) - 1 / np.expm1(xT(T)))


def gls(T, tmpl, cov, dkind, xT_tmpl=None, gal=True, gcol=True):
    x = xT(T)
    xt = x if xT_tmpl is None else xT(xT_tmpl)
    cols = [PREF * tmpl(xt)]
    if gcol:
        cols.append(PREF * g_bb(x))
    if gal:
        cols.append(GAL)
    A = np.column_stack(cols)
    Ci = CINV[cov]
    Fm = A.T @ Ci @ A
    P = np.linalg.inv(Fm)
    r = data(T, dkind)
    th = P @ (A.T @ Ci @ r)
    res = r - A @ th
    return th[0], np.sqrt(P[0, 0]), float(res @ Ci @ res)


def limit(tmpl, cov='diag', dkind='resid', T='fixed', T0=2.725, clip='ahat',
          xT_tmpl=None, gal=True):
    if T == 'fixed':
        Tb = T0
    else:
        f = lambda t: gls(t, tmpl, cov, dkind, xT_tmpl, gal, gcol=False)[2]
        Tb = minimize_scalar(f, bounds=(2.720, 2.732), method='bounded').x
    a, s, c2 = gls(Tb, tmpl, cov, dkind, xT_tmpl, gal)
    if clip == 'ahat':
        ul = max(a, 0) + Z1645 * s
    else:
        ul = max(a + Z1645 * s, 0)
    return dict(ul=ul, a=a, s=s, T=Tb, chi2=c2)


Z_MU = 1.98e6


def D(x):
    return (G1 / (3 * G2)) * g_bb(x) - planck(x) / x


def analytic(zc):
    J = np.exp(-(zc / Z_MU) ** 2.5)
    return lambda x: J * D(x)
