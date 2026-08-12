#!/usr/bin/env python3
import json,math,os,pathlib,numpy as np
system=os.environ['SYSTEM']; root=pathlib.Path('/scratch/dexuan1/runs')/('wssr_C_4096_kfac1000' if system=='C' else 'wssr_N2_4096_kfac1000')
p=root/'checkpoints/1000.npz'; z=np.load(p,allow_pickle=True); epoch=int(z['e']); data=z['d']; data=data.tolist() if data.dtype==object else data
params=z['p'].tolist(); opt=z['o'].tolist(); key=z['k']; pos=np.asarray(data['walker_data']['position']); amp=np.asarray(data['walker_data']['amplitude'])
def finite(x):
 if isinstance(x,dict): return all(finite(v) for v in x.values())
 if isinstance(x,(list,tuple)): return all(finite(v) for v in x)
 a=np.asarray(x); return True if a.dtype==object else bool(np.isfinite(a).all())
cfg=json.load(open(root/'config.json')); rows=np.loadtxt(root/'energy.txt'); var=np.loadtxt(root/'variance.txt')
unique=np.unique(np.ascontiguousarray(pos.reshape(4096,-1)),axis=0).shape[0]
assert epoch==999 and pos.shape[0]==amp.shape[0]==unique==4096 and finite(params) and finite(opt) and finite(key) and np.isfinite(rows).all() and np.isfinite(var).all()
assert cfg['vmc']['nchains']==4096 and cfg['vmc']['nepochs']==1000
if system=='C': assert cfg['problem']['nelec']==[4,2] and cfg['problem']['ion_charges']==[6.0]
else:
 ions=np.asarray(cfg['problem']['ion_pos']); assert cfg['problem']['nelec']==[7,7] and abs(np.linalg.norm(ions[1]-ions[0])-2.068)<1e-10
out={'system':system,'checkpoint':str(p),'internal_epoch':epoch,'nchains':4096,'unique_walkers':unique,'position_shape':list(pos.shape),'key_shape':list(key.shape),'finite_params':True,'finite_optimizer':True,'final_energy':float(rows[-1]),'final_variance':float(var[-1])}
(root/'validation.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out))
