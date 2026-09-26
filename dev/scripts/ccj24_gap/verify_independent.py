import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import os
S = os.path.dirname(os.path.abspath(__file__)) + '/'
R = os.path.abspath(os.path.join(S, '..', '..', '..')) + '/'
h=6.62607015e-34;k=1.380649e-23;c=2.99792458e8
D=np.loadtxt(R+'data/firas_monopole_spec_v1.txt')
fcm,spec,res,sig,gal=D.T
corr=np.loadtxt(R+'data/firas_correlation_matrix.txt')
nu=fcm*c*100; pref=2*h*nu**3/c**2/1e-23  # kJy/sr per unit dn
Cf=np.outer(sig,sig)*corr; Cfi=np.linalg.inv(Cf); Cdi=np.diag(1/sig**2)
xT=lambda T:h*nu/(k*T)
B=lambda T:pref/np.expm1(xT(T))
def Gk(T):
    x=xT(T);n=1/np.expm1(x);return pref*x*n*(1+n)
dust=pref*nu**2/np.expm1(h*nu/(k*9.0)); dust/=dust.max()
print('col2-B(2.725) vs col3 max abs diff kJy:',np.max(np.abs(spec*1e3-B(2.725)-res)),' rms/sig:',np.sqrt(np.mean(((spec*1e3-B(2.725)-res)/sig)**2)))
def linfit(d,cols,Ci):
    M=np.column_stack(cols);F=M.T@Ci@M;P=np.linalg.inv(F);th=P@M.T@Ci@d
    r=d-M@th; return th,P,float(r@Ci@r)
def floatT(tf,Ci,data='col2'):
    def chi(T):
        d=spec*1e3-B(T); return linfit(d,[pref*tf(xT(T)),dust],Ci)[2]
    Ts=np.linspace(2.720,2.732,1201); cs=[chi(T) for T in Ts]; T0=Ts[int(np.argmin(cs))]
    r=minimize_scalar(chi,bounds=(T0-2e-5,T0+2e-5),method='bounded',options={'xatol':1e-10});Tb=r.x
    d=spec*1e3-B(Tb); th,P,_=linfit(d,[pref*tf(xT(Tb)),Gk(Tb),dust],Ci)
    ahat,s=th[0],np.sqrt(P[0,0])
    # true profile over (T,G0) at fixed A: find A where dchi2=2.706
    def prof(A):
        def cA(T):
            d=spec*1e3-B(T)-A*pref*tf(xT(T)); th,P,cc=linfit(d,[dust],Ci); return cc
        rr=minimize_scalar(cA,bounds=(2.7245,2.7260),method='bounded',options={'xatol':1e-11}); return rr.fun
    from scipy.optimize import brentq
    c0=prof(ahat)
    Aup=brentq(lambda A:prof(A)-c0-norm.ppf(.95)**2,ahat,ahat+5*s)
    return ahat,s,Tb,Aup
def fixedT(tf,Ci,d,T=2.725,mode='both'):
    S_=pref*tf(xT(T)); G=Gk(T)
    Nm=np.column_stack([G,dust])
    def perp(v): 
        co=np.linalg.solve(Nm.T@Ci@Nm,Nm.T@Ci@v); return v-Nm@co
    Sp=perp(S_); dp=perp(d)
    A=Sp@Ci@Sp; ahat=(d@Ci@Sp)/A; ahat2=(dp@Ci@Sp)/A
    # chi2 differences: model-only vs both
    a=np.linspace(-2*abs(ahat),5*abs(ahat)+1/np.sqrt(A),7)
    c1=np.array([(d-ai*Sp)@Ci@(d-ai*Sp) for ai in a]); c2=np.array([(dp-ai*Sp)@Ci@(dp-ai*Sp) for ai in a])
    ddiff=np.max(np.abs((c1-c1[0])-(c2-c2[0])))
    return ahat,1/np.sqrt(A),ahat2,ddiff
chl=np.loadtxt(R+'dev/AxionLimits/limit_data/DarkPhoton/COBEFIRAS_Chluba.txt')
chl=chl[chl[:,0]<=1.5e-4]  # drop the contour-closure row at m ~ 1e-3 eV
ip=lambda tab,m:10**np.interp(np.log10(m),np.log10(tab[:,0]),np.log10(tab[:,1]))
z=norm.ppf(.95)
for f in ['1.5e-12','3.4e-09','1.6e-07','3.6e-06']:
    t=np.load(S+f'tmpl_{f}.npz'); m=float(t['m']); g=float(t['gpe2'])
    tf=lambda x,_x=t['x'],_d=t['dn_per_gc']:np.interp(x,_x,_d)
    eps=lambda A:np.sqrt(A/g)
    a,s,Tb,Aup=floatT(tf,Cfi)
    ad,sd,Tbd,_=floatT(tf,Cdi)
    fa,fs,fa2,dd=fixedT(tf,Cfi,res)
    fa_c2,fs_c2,_,_=fixedT(tf,Cfi,spec*1e3-B(2.725))
    ca,cs,ca2,dd2=fixedT(tf,Cdi,res)
    ca_c2,cs_c2,_,_=fixedT(tf,Cdi,spec*1e3-B(2.725))
    pub=ip(chl,m)
    e_paper=eps(max(a+z*s,0)); e_proflik=eps(Aup)
    e_fixfull_c2=eps(fa_c2+z*fs_c2)
    e_floor=eps(max(ca,0)+z*cs)            # CCJ24 diag, col3, floored
    e_bayes=eps(ca+cs*norm.ppf(1-0.05*norm.cdf(ca/cs)))
    print(f'\nm={m:.2e} pub(AxionLimits)={pub:.4e}')
    print(f' floatT full col2: ahat/s={a/s:.3f} Tb={Tb:.6f} eps_paper={e_paper:.4e} paper/pub={e_paper/pub:.4f}; true profile dchi2 eps={e_proflik:.4e} ratio to lin={e_proflik/e_paper:.5f}')
    print(f' fixedT full col2 (G lin): ahat/s={fa_c2/fs_c2:.3f} eps={e_fixfull_c2:.4e} floatT/fixedT={e_paper/e_fixfull_c2:.5f}')
    print(f' fixedT full col3: ahat/s={fa/fs:.3f}; diag col2: ahat/s={ca_c2/cs_c2:.3f}; diag col3: ahat/s={ca/cs:.3f}; model-only vs both dchi2 maxdiff={dd:.2e},{dd2:.2e}; ahat same? {fa2/fa:.6f}')
    k_=a/s; print(f' predicted floor ratio sqrt(z/(k+z))... paper/floor(same stat)={e_paper/eps(max(a,0)+z*s):.4f} vs sqrt((k+z)/z)={np.sqrt((k_+z)/z):.4f}')
    # factor chain
    e_diag_c2=eps(ca_c2+z*cs_c2); e_diag_c3=eps(ca+z*cs); e_full_c3=eps(fa+z*fs)
    print(f' unfloored: fixfull_c2={e_fixfull_c2:.4e} diag_c2={e_diag_c2:.4e} full_c3={e_full_c3:.4e} diag_c3={e_diag_c3:.4e}')
    print(f'  full/diag (col2)={e_fixfull_c2/e_diag_c2:.4f} (col3)={e_full_c3/e_diag_c3:.4f}; col2/col3 (diag)={e_diag_c2/e_diag_c3:.4f} (full)={e_fixfull_c2/e_full_c3:.4f}')
    print(f' floored CCJ diag col3: eps={e_floor:.4e} /pubAL={e_floor/pub:.4f}; floored full col2: /pubAL={eps(max(a,0)+z*s)/pub:.4f}')
    print(f' Bayes diag col3: eps={e_bayes:.4e} /pubAL={e_bayes/pub:.4f}')
    print(f' dchi2=2.71 floored (notebook ccj24_limit, 1.646): /pubAL={eps(max(ca,0)+np.sqrt(2.71)*cs)/pub:.4f}')
