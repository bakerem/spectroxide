#!/usr/bin/env python3
"""
Clean-room reference solver for CMB spectral distortions.

Independent N-version cross-check.  Implements the Chang-Cooper (1970)
finite-volume scheme for the FULL photon occupation number n(x,z) (not a
distortion), following ONLY the physics equations in dev/refsolver/contract.md
and the raw arXiv LaTeX of Chluba & Sunyaev 2012 (arXiv:1109.6552).

Isolation: no spectroxide solver source was read (see README.md).

Physics:
  Kompaneets (Compton):  dn/dtau = (theta_e/x^2) d/dx[ x^4 ( dn/dx + phi n(1+n) ) ]
     with phi = T_gamma/T_e, theta_e = kT_e/(m_e c^2), tau Thomson optical depth.
     Discretised via the exact change of variables g = n/(1+n):
        F = x^4 (1+n)^2 [ dg/dx + phi g ]
     which is a linear Fokker-Planck flux in g with unit diffusion and drift phi.
     Chang-Cooper weighting on g gives the exact discrete Bose-Einstein
     equilibrium g_eq = exp(-phi x)  ->  n_eq = 1/(exp(phi x)-1).

  DC emission (photon number changing):
     dn/dtau|_em = (K_DC/x^3) [ 1 - n (exp(phi x) - 1) ]
     K_DC = (4 alpha / 3 pi) theta_gamma^2 I4pl/(1+14.16 theta_gamma) H_dc(x)
     (Chluba & Sunyaev 2012 Eq. 13).  Bremsstrahlung OMITTED (see README).

  Electron temperature: quasi-stationary (CS2012 Eq. for rho_e), including the
     adiabatic -H t_C rho_e Hubble-cooling term:
        rho_e = [ beta_C rho_eq + S_inj ] / ( beta_C + H t_C )
     rho_eq = I4/(4 G3),  beta_C = 4 rho_gamma_tilde / alpha_h.
"""
import numpy as np
from scipy.linalg import solve_banded

# ---------------------------------------------------------------- constants (SI, CODATA-ish)
k_B   = 1.380649e-23        # J/K
h_P   = 6.62607015e-34      # J s
hbar  = h_P/(2*np.pi)
c     = 2.99792458e8        # m/s
m_e   = 9.1093837015e-31    # kg
mec2  = m_e*c**2            # J
alpha_fs = 7.2973525693e-3
sigma_T  = 6.6524587321e-29 # m^2
lambda_e = h_P/(m_e*c)      # Compton wavelength [m]
kappa_g  = 8*np.pi/lambda_e**3   # [1/m^3]  = 1.760e36
zeta3    = 1.2020569032
I4pl     = 4*np.pi**4/15.0
G3pl     = np.pi**4/15.0
beta_mu  = 3*zeta3/(np.pi**2/6.0)   # 3 zeta(3)/zeta(2) = 2.1923

T_cmb = 2.726
Y_p   = 0.24
f_He  = (Y_p/4.0)/(1.0-Y_p)   # n_He/n_H = 0.0789

# ---------------------------------------------------------------- history table
_HIST = None
def load_history(path):
    global _HIST
    d = np.genfromtxt(path, delimiter=',', names=True)
    order = np.argsort(d['z'])
    lz = np.log(d['z'][order])
    _HIST = dict(lz=lz,
                 x_e=d['x_e'][order],
                 H  =d['H_z_per_s'][order],
                 n_e=d['n_e_per_m3'][order],
                 n_H=d['n_H_per_m3'][order],
                 Tg =d['T_gamma_K'][order],
                 tC =d['t_C_s'][order])
    return _HIST

def hist(z):
    """Interpolate history in log z. Returns dict of scalars/arrays."""
    lz = np.log(z)
    def I(key, log=True):
        y = _HIST[key]
        if log:
            return np.exp(np.interp(lz, _HIST['lz'], np.log(y)))
        return np.interp(lz, _HIST['lz'], y)
    return dict(x_e=I('x_e'), H=I('H'), n_e=I('n_e'), n_H=I('n_H'),
                Tg=I('Tg'), tC=I('tC'))

# ---------------------------------------------------------------- grid
class Grid:
    def __init__(self, N=512, xmin=1e-3, xmax=40.0):
        self.x = np.logspace(np.log10(xmin), np.log10(xmax), N)
        self.N = N
        # interfaces at geometric means; ghost interfaces at the ends (zero flux)
        self.xh = np.sqrt(self.x[:-1]*self.x[1:])   # length N-1, interface j between node j and j+1
        # cell widths for finite-volume divergence
        edges = np.empty(N+1)
        edges[1:-1] = self.xh
        edges[0]  = self.x[0]**2/self.xh[0]     # reflect
        edges[-1] = self.x[-1]**2/self.xh[-1]
        self.dxc = edges[1:] - edges[:-1]        # length N, cell width for node j
        self.dxn = self.x[1:] - self.x[:-1]      # length N-1, node spacing for interface j

def planck(x):
    return 1.0/np.expm1(x)

# ---------------------------------------------------------------- moment integrals
def moments(n, x):
    G3 = np.trapz(x**3*n, x)
    I4 = np.trapz(x**4*n*(1.0+n), x)
    return G3, I4

def photon_number(n, x):
    return np.trapz(x**2*n, x)

# ---------------------------------------------------------------- DC coefficient
def H_dc(x):
    return np.exp(-2*x)*(1 + 1.5*x + (29.0/24.0)*x**2 + (11.0/16.0)*x**3 + (5.0/12.0)*x**4)

def K_DC(x, theta_g):
    return (4*alpha_fs/(3*np.pi))*theta_g**2 * (I4pl/(1.0+14.16*theta_g)) * H_dc(x)

# ---------------------------------------------------------------- Chang-Cooper delta
def cc_delta(w):
    """delta = 1/w - 1/(exp(w)-1), stable near w->0 (limit 1/2)."""
    out = np.empty_like(w)
    small = np.abs(w) < 1e-6
    big   = w > 500.0                 # 1/(e^w-1) -> 0
    mid   = ~small & ~big
    out[small] = 0.5 - w[small]/12.0
    out[big]   = 1.0/w[big]
    ws = w[mid]
    out[mid] = 1.0/ws - 1.0/np.expm1(ws)
    return out

# ---------------------------------------------------------------- one implicit Compton+DC step
def implicit_step(n_old, g, dtau, phi, theta_e, theta_g, delta_rho_inj=0.0,
                  do_dc=True, newton_iters=6, tol=1e-13):
    """Advance n by one implicit-Euler step of Compton (Chang-Cooper on g=n/(1+n))
    plus DC emission.  phi=T_gamma/T_e is held fixed for this step (updated by the
    outer loop).  Returns new n."""
    x   = g.x
    dxn = g.dxn
    dxc = g.dxc
    N   = g.N
    xh4 = g.xh**4
    # CC weighting: drift/diffusion ratio for g is phi (unit diffusion); Peclet w = phi*dxn
    w  = phi*dxn
    dl = cc_delta(w)          # length N-1

    # DC source/sink (linear in n): dn/dtau|_em = S_dc - Gamma_dc * n
    # K_DC = prefac0 * H_dc(x)/x^3 with H_dc = e^{-2x} poly(x).  To avoid overflow
    # (e^{phi x} at x~40) compute Gamma_dc = S_dc(e^{phi x}-1) with the e^{-2x} folded
    # in analytically: Gamma_dc = prefac0 poly/x^3 [e^{(phi-2)x} - e^{-2x}].
    if do_dc:
        prefac0 = (4*alpha_fs/(3*np.pi))*theta_g**2*(I4pl/(1.0+14.16*theta_g))
        poly = 1 + 1.5*x + (29.0/24.0)*x**2 + (11.0/16.0)*x**3 + (5.0/12.0)*x**4
        base = prefac0*poly/x**3
        S_dc     = base*np.exp(-2*x)
        Gamma_dc = base*(np.exp(np.clip((phi-2.0)*x, -700, 300)) - np.exp(-2*x))
    else:
        S_dc = np.zeros(N); Gamma_dc = np.zeros(N)

    pref = theta_e/x**2       # divergence prefactor per node

    n = n_old.copy()
    for it in range(newton_iters):
        nn = n
        gg = nn/(1.0+nn)
        # interface prefactor P = x^4 (1+n_bar)^2 (frozen from current iterate)
        nbar = 0.5*(nn[:-1]+nn[1:])
        P = xh4*(1.0+nbar)**2                      # length N-1
        dgdn = 1.0/(1.0+nn)**2                      # dg/dn per node
        # interface flux F_j = P*[ (g_{j+1}-g_j)/dxn + phi*((1-dl)g_{j+1}+dl g_j) ]
        gbar = (1.0-dl)*gg[1:] + dl*gg[:-1]
        F = P*((gg[1:]-gg[:-1])/dxn + phi*gbar)    # length N-1

        # divergence (zero flux at both ends): (F_{j+1/2}-F_{j-1/2})/dxc_j
        # F[k] is the interface between node k and k+1 (i.e. F_{k+1/2})
        divF = np.zeros(N)
        divF[:-1] += F        # +F_{j+1/2} into node j
        divF[1:]  -= F        # -F_{j-1/2} into node j+1
        divF /= dxc

        comp = pref*divF
        emis = S_dc - Gamma_dc*nn
        R = (nn - n_old)/dtau - comp - emis        # residual, want R=0

        # ---- tridiagonal Jacobian J = dR/dn ----
        # dR_j/dn_j, dR_j/dn_{j-1}, dR_j/dn_{j+1}
        # F_j depends on n_j,n_{j+1} (through g and P). We freeze P (semi-implicit on
        # the (1+n)^2 prefactor) so dF_j/dn_k only through g:
        #   dF_j/dn_j     = P_j*( -1/dxn_j + phi*dl_j ) * dgdn_j
        #   dF_j/dn_{j+1} = P_j*(  1/dxn_j + phi*(1-dl_j) ) * dgdn_{j+1}
        aL = P*(-1.0/dxn + phi*dl)          # dF_j/dn_j
        aR = P*( 1.0/dxn + phi*(1.0-dl))    # dF_j/dn_{j+1}
        dFdnj   = aL*dgdn[:-1]
        dFdnjp1 = aR*dgdn[1:]

        # d(divF)_j/dn = [dF_{j+1/2} - dF_{j-1/2}]/dxc_j
        diag = np.zeros(N); lo = np.zeros(N); up = np.zeros(N)
        # contribution of interface k=F[k] (between node k and k+1):
        #   to divF[k]:   +F[k]/dxc_k   -> depends on n_k (dFdnj[k]) and n_{k+1} (dFdnjp1[k])
        #   to divF[k+1]: -F[k]/dxc_{k+1}
        # assemble dcomp/dn = pref * d(divF)/dn
        # diagonal (node j from its own dependence)
        # divF[j] uses +F[j] (needs dFdnj[j]) and -F[j-1] (needs dFdnjp1[j-1])
        ddiv_diag = np.zeros(N)
        ddiv_lo   = np.zeros(N)   # dependence of divF[j] on n_{j-1}
        ddiv_up   = np.zeros(N)   # dependence of divF[j] on n_{j+1}
        # +F[j]/dxc[j] for j in 0..N-2
        ddiv_diag[:-1] += dFdnj/dxc[:-1]
        ddiv_up[:-1]   += dFdnjp1/dxc[:-1]
        # -F[j-1]/dxc[j] for j in 1..N-1  (interface index j-1)
        ddiv_diag[1:]  += -dFdnjp1/dxc[1:]
        ddiv_lo[1:]    += -dFdnj/dxc[1:]

        dcomp_diag = pref*ddiv_diag
        dcomp_lo   = pref*ddiv_lo
        dcomp_up   = pref*ddiv_up

        diag = 1.0/dtau - dcomp_diag + Gamma_dc
        lo   = -dcomp_lo
        up   = -dcomp_up

        # banded solve  J delta = -R
        ab = np.zeros((3, N))
        ab[0,1:]  = up[:-1]     # super-diagonal
        ab[1,:]   = diag
        ab[2,:-1] = lo[1:]      # sub-diagonal
        delta = solve_banded((1,1), ab, -R)
        n = n + delta
        n = np.maximum(n, 0.0)   # positivity
        if np.max(np.abs(delta)) < tol*np.max(np.abs(n)+1e-300):
            break
    return n

# ---------------------------------------------------------------- electron temperature
def te_coeffs(n, x, hz):
    """Compton-equilibrium ratio and rate coefficients (CS2012 Eq. for rho_e)."""
    G3, I4 = moments(n, x)
    rho_eq = I4/(4.0*G3)
    theta_g = k_B*hz['Tg']/mec2
    rho_gamma_tilde = kappa_g*theta_g**4*G3
    alpha_h = 1.5*hz['n_H']*(1.0 + f_He + hz['x_e'])
    beta_C = 4.0*rho_gamma_tilde/alpha_h        # Compton relaxation rate (per tau)
    HtC = hz['H']*hz['tC']                       # adiabatic Hubble cooling rate (per tau)
    return dict(rho_eq=rho_eq, beta_C=beta_C, HtC=HtC, theta_g=theta_g)

def rho_e_step(rho_e_old, dtau, c):
    """Backward-Euler step of  drho_e/dtau = beta_C(rho_eq - rho_e) - HtC rho_e.
    (No injection here; heat injection is added as a calibrated offset by the driver.)
    In the fast-Compton limit this reduces to the quasi-stationary rho_eq/(1+HtC/beta_C);
    when Compton freezes (beta_C -> 0) it correctly decays as rho_e ∝ (1+z)  [T_e∝(1+z)^2]."""
    return (rho_e_old/dtau + c['beta_C']*c['rho_eq']) / (1.0/dtau + c['beta_C'] + c['HtC'])

# ---------------------------------------------------------------- decomposition
def decompose(dn, x, xlo=0.5, xhi=18.0):
    m = (x>=xlo)&(x<=xhi)
    xf = x[m]; d = dn[m]
    ex = np.exp(xf)
    Gbb = xf*ex/(ex-1.0)**2
    Gt  = Gbb/xf
    Ysz = (Gbb/xf)*(xf*(ex+1.0)/(ex-1.0) - 4.0)
    Mmu = Gbb*(1.0/beta_mu - 1.0/xf)
    A = np.vstack([Gt, Ysz, Mmu]).T   # columns: dT/T, y, mu
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    return dict(dT=coef[0], y=coef[1], mu=coef[2])

# ---------------------------------------------------------------- driver
def run_case(case, grid, nz=2000, verbose=False):
    x = grid.x
    npl_of = planck  # Planck in x (photon temperature) is always 1/(e^x-1)

    # ---- set up redshift schedule and injection ----
    ctype = case['type']
    if ctype in ('heat',):
        z_h = case['z_h']; sig_z = 0.04*z_h
        z_start = z_h + 7*sig_z
        z_end = 1.0
    elif ctype == 'adiabatic':
        z_start = 3e6; z_end = 1.0; z_h=None; sig_z=None
    elif ctype == 'photon':
        z_h = case['z_h']; z_start = z_h; z_end = 1.0; sig_z=None
    zgrid = np.exp(np.linspace(np.log(z_start), np.log(z_end), nz+1))

    # ---- heat injection normalisation: choose A so int 4 theta_g drho_inj dtau = drho/rho ----
    delta_rho_inj_of_z = lambda z: 0.0
    if ctype == 'heat':
        def gauss(z): return np.exp(-(z-z_h)**2/(2*sig_z**2))
        # integrate 4 theta_g gauss dtau over the schedule
        zc = 0.5*(zgrid[:-1]+zgrid[1:])
        acc = 0.0
        for i in range(nz):
            z0,z1 = zgrid[i], zgrid[i+1]
            zc_i = 0.5*(z0+z1)
            hzc = hist(zc_i)
            dtau = abs(z0-z1)/((1+zc_i)*hzc['H']*hzc['tC'])
            theta_g = k_B*hzc['Tg']/mec2
            acc += 4*theta_g*gauss(zc_i)*dtau
        A = case['drho']/acc
        delta_rho_inj_of_z = lambda z: A*gauss(z)

    # ---- initial spectrum ----
    n = npl_of(x).copy()
    # photon injection: add Gaussian bump in x at z_start
    N_in_frac = None
    if ctype == 'photon':
        x_inj = case['x_inj']; sig_x = 0.05*x_inj
        bump_shape = np.exp(-(x-x_inj)**2/(2*sig_x**2))
        N_bump_unit = np.trapz(x**2*bump_shape, x)
        N_pl = np.trapz(x**2*n, x)
        B = case['dNoverN']*N_pl/N_bump_unit
        n = n + B*bump_shape
        N_in_frac = case['dNoverN']

    # DC gate: emission is dynamically frozen and its near-equilibrium coefficient is
    # out of its validity regime once electrons decouple (z<~1e4).  Enable only above.
    Z_DC_MIN = 1.0e4
    # ---- initialise rho_e (quasi-stationary at z_start; deep in fast-Compton regime) ----
    c0 = te_coeffs(n, x, hist(z_start))
    rho_e = c0['rho_eq']/(1.0 + c0['HtC']/c0['beta_C'])
    # ---- integrate ----
    for i in range(nz):
        z0, z1 = zgrid[i], zgrid[i+1]
        zc = np.sqrt(z0*z1)
        hzc = hist(zc)
        dtau = abs(z0-z1)/((1+zc)*hzc['H']*hzc['tC'])
        theta_g = k_B*hzc['Tg']/mec2
        dri = delta_rho_inj_of_z(zc) if ctype=='heat' else 0.0
        do_dc = zc >= Z_DC_MIN
        rho_e_old = rho_e
        n_iter = n
        # Picard on rho_e (T_e) coupled to the spectrum
        for _ in range(3):
            c = te_coeffs(n_iter, x, hzc)
            rho_e = rho_e_step(rho_e_old, dtau, c)
            rho_e_used = rho_e + dri
            phi = 1.0/rho_e_used
            theta_e = theta_g*rho_e_used
            n_new = implicit_step(n, grid, dtau, phi, theta_e, theta_g,
                                  do_dc=do_dc, newton_iters=6)
            if np.max(np.abs(n_new-n_iter)) < 1e-14*np.max(n_iter):
                n_iter = n_new; break
            n_iter = n_new
        n = n_iter

    # ---- final decomposition (Planck at final T_gamma; x already normalised by T_gamma) ----
    dn = n - npl_of(x)
    dec = decompose(dn, x)
    out = dict(mu=dec['mu'], y=dec['y'], dT=dec['dT'],
               z_start=z_start, z_end=z_end)
    if ctype=='photon':
        N_final = photon_number(n, x)
        N_pl    = photon_number(npl_of(x), x)
        out['dN_in_frac']  = N_in_frac
        out['dN_final_frac']= (N_final-N_pl)/N_pl
    # also report integrated energy change (for heat/photon sanity)
    G3f,_ = moments(n, x)
    out['drho_over_rho_measured'] = G3f/G3pl - 1.0
    return out, x, dn

# ---------------------------------------------------------------- equilibrium unit test
def test_equilibrium(grid, phi=1.0, verbose=True):
    """Planck at T_e must be a fixed point: residual flux ~ machine precision."""
    x = grid.x
    n_eq = 1.0/np.expm1(phi*x)
    theta_g = 1e-4; theta_e = theta_g/phi
    n_after = implicit_step(n_eq.copy(), grid, dtau=1e6, phi=phi, theta_e=theta_e,
                            theta_g=theta_g, do_dc=True, newton_iters=8)
    dev = np.max(np.abs(n_after-n_eq)/(n_eq+1e-300))
    if verbose:
        print(f"[equilibrium test] phi={phi}: max rel drift over huge step = {dev:.3e}")
    return dev
