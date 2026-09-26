"""Plateau offset candidates (record: dev/audit/ccj24_postrec_gap.md). Band ratios eps_ours/eps_CCJ24 (Fig. 8 vector), y-era normalized, for X_e histories from CLASS."""
import sys, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from postrec_gap import HERE, ROOT, X0, GMap, gamma_limit
from spectroxide.cosmology import PLANCK2018_COSMO as P18, PLANCK2015_COSMO as P15, DEFAULT_COSMO as D
import classy
TM=np.load(HERE/'grid_templates.npz',allow_pickle=True); mt=TM['m'].astype(float)
gl=np.array([gamma_limit(np.interp(X0,TM['x'][i],TM['dn_per_gc'][i]))[0] for i in range(len(mt))])
glim=lambda m: np.exp(np.interp(np.log(m),np.log(mt),np.log(gl)))
VEC=np.loadtxt(HERE/'ccj24_fig8_firas_vector.txt'); ev=lambda m:10**np.interp(np.log10(m),np.log10(VEC[:,0]),np.log10(VEC[:,1]))
def xe(c, rec='HyRec', extra={}):
    M=classy.Class()
    M.set({'h':c['h'],'omega_b':c['omega_b']*c['h']**2,'omega_cdm':(c['omega_m']-c['omega_b'])*c['h']**2,'T_cmb':c['t_cmb'],
           'YHe':c['y_p'],'N_ur':c['n_eff'],'N_ncdm':0,'recombination':rec,'reio_parametrization':'reio_none','output':'',**extra})
    M.compute(); th=M.get_thermodynamics(); z,x=th['z'],th['x_e']; M.struct_cleanup()
    o=np.argsort(z); z,x=z[o],x[o]; k=np.concatenate([[True],np.diff(z)>0])&(z>5); return z[k],x[k]
ms=np.geomspace(1.5e-12,8e-10,40)
def band(g,c):
    e=np.array([np.sqrt(glim(m)/g.gpe2(m)[0]) for m in ms]); zr=np.array([g.gpe2(m)[1] for m in ms]); r=e/ev(ms)
    # normalize to y era
    my=np.geomspace(4e-9,3e-8,12); ny=np.median([np.sqrt(glim(m)/g.gpe2(m)[0])/ev(m) for m in my])
    r=r/ny
    return np.median(r[(zr>250)&(zr<800)]), np.median(r[(zr>850)&(zr<1450)])
cases=[('HyRec P18',P18,'HyRec',{}),
       ('RECFAST P18 (1.5, Gaussians on)',P18,'RECFAST',{}),
       ('RECFAST P18, Hswitch off (F=1.14, RECFAST 1.4)',P18,'RECFAST',{'recfast_Hswitch':0}),
       ('RECFAST P18, F=1.00, Hswitch off',P18,'RECFAST',{'recfast_Hswitch':0,'recfast_fudge_H':1.0}),
       ('HyRec P18, Y_p = 0.24',{**P18,'y_p':0.24},'HyRec',{}),
       ('HyRec P15',P15,'HyRec',{}),
       ('HyRec default cosmo',D,'HyRec',{})]
print(f'{"case":50s} 250-800  850-1450   (normalized to y era; target 1.000)')
for name,c,rec,ex in cases:
    try:
        g=GMap(*xe(c,rec,ex),c); a,b=band(g,c); print(f'{name:50s} {a:.4f}  {b:.4f}')
    except Exception as err: print(name,'FAILED',err)
# uniform n_e scale on HyRec P18
z,x=xe(P18)
for f in [1.01,1.02,1.03]:
    a,b=band(GMap(z,x,P18,ne_scale=f),P18); print(f'{"HyRec P18, n_e x %.2f"%f:50s} {a:.4f}  {b:.4f}')
print('--- combinations')
for name,c,rec,ex in [('P18, Y_p=0.24, RECFAST 1.4',{**P18,'y_p':0.24},'RECFAST',{'recfast_Hswitch':0}),
                      ('P18, Y_p=0.24, RECFAST 1.5',{**P18,'y_p':0.24},'RECFAST',{}),
                      ('P18, Y_p=0.24, T0=2.726, N_eff=3.046, RECFAST 1.4',{**P18,'y_p':0.24,'t_cmb':2.726,'n_eff':3.046},'RECFAST',{'recfast_Hswitch':0})]:
    g=GMap(*xe(c,rec,ex),c); a,b=band(g,c); print(f'{name:50s} {a:.4f}  {b:.4f}')
