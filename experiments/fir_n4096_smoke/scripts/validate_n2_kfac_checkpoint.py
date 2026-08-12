#!/usr/bin/env python3
import csv
import json
import math
import pathlib
import sys
import os
import numpy as np
from checkpoint_epoch import validate_checkpoint_epoch

ROOT=pathlib.Path(os.environ.get('CORRECTED_ROOT','/scratch/dexuan1/runs/fir_n4096_corrected/N2'))
NEW=ROOT/'kfac_preliminary/checkpoints/5000.npz'
OLD=pathlib.Path('/scratch/dexuan1/runs/kfac_pre/N2eq_R2068_kfac_pre5000/checkpoints/5000.npz')
TAG=os.environ.get('CORRECTED_TAG','')
REPORT=pathlib.Path('/scratch/dexuan1/vmcnet/experiments/fir_n4096_smoke/results')/f"n2_kfac_n4096_validation{TAG}.md"
PRE=ROOT/'validation_precheck.json'

def load(path):
    with np.load(path,allow_pickle=True) as z:
        d=z['d']; d=d.tolist() if d.dtype==object else d
        return z['e'].tolist(),d,z['p'].tolist(),z['k']

def leaf_shapes(x,prefix=''):
    if isinstance(x,dict):
        out=[]
        for k,v in sorted(x.items()): out+=leaf_shapes(v,prefix+'/'+str(k))
        return out
    if isinstance(x,(list,tuple)):
        out=[]
        for i,v in enumerate(x): out+=leaf_shapes(v,prefix+'/'+str(i))
        return out
    a=np.asarray(x); return [(prefix,a.shape,str(a.dtype))]

mode=sys.argv[1]
if mode=='pre':
    epoch,data,params,key=load(NEW); _,_,old_params,_=load(OLD)
    pos=np.asarray(data['walker_data']['position']); amp=np.asarray(data['walker_data']['amplitude'])
    flat=np.ascontiguousarray(pos.reshape(pos.shape[0],-1))
    unique=np.unique(flat,axis=0).shape[0]
    cfg=json.load(open(ROOT/'kfac_preliminary/config.json'))
    ions=np.asarray(cfg['problem']['ion_pos'],dtype=float)
    bond_length=float(np.linalg.norm(ions[1]-ions[0]))
    epoch_expected=validate_checkpoint_epoch(NEW,epoch)
    sampler_shapes=leaf_shapes(data)
    result={'epoch':int(epoch),'expected_internal_epoch':epoch_expected,'position_shape':list(pos.shape),'amplitude_shape':list(amp.shape),'nchains':int(pos.shape[0]),'unique_walkers':int(unique),'exact_duplicate_walkers':int(pos.shape[0]-unique),'key_shape':list(np.asarray(key).shape),'model_leaf_shapes_match_formal':leaf_shapes(params)==leaf_shapes(old_params),'sampler_state_shapes':sampler_shapes,'bond_length_bohr':bond_length,'ion_pos':cfg['problem']['ion_pos'],'ion_charges':cfg['problem']['ion_charges'],'nelec':cfg['problem']['nelec']}
    PRE.parent.mkdir(parents=True,exist_ok=True); PRE.write_text(json.dumps(result,indent=2)+'\n')
    assert result['epoch']==result['expected_internal_epoch'] and result['nchains']==4096 and amp.shape[0]==4096
    assert unique==4096 and result['model_leaf_shapes_match_formal']
    assert pos.shape==(4096,14,3) and amp.shape==(4096,)
    assert abs(bond_length-2.068)<1e-12 and cfg['problem']['ion_charges']==[7.0,7.0] and cfg['problem']['nelec']==[7,7]
    print(json.dumps(result))
elif mode=='post':
    rows=list(csv.DictReader(open(ROOT/'validation_eval/eval/training_metrics.csv')))
    assert len(rows)==20
    E=np.array([float(r['energy']) for r in rows]); V=np.array([float(r['variance']) for r in rows]); A=np.array([float(r['accept_ratio']) for r in rows])
    finite=bool(np.isfinite(E).all() and np.isfinite(V).all() and np.isfinite(A).all())
    em=float(E.mean()); vm=float(V.mean()); am=float(A.mean())
    passed=finite and abs(em-(-109.48))<2.0 and vm<100.0 and 0.2<am<0.8
    pre=json.loads(PRE.read_text())
    lines=['# N2 n=4096 KFAC checkpoint validation','',f'- checkpoint: `{NEW}`',f"- stored epoch: {pre['epoch']} (expected zero-based: {pre['expected_internal_epoch']})",f"- geometry: R={pre['bond_length_bohr']:.12f} Bohr, positions={pre['ion_pos']}",f"- walker position shape: `{pre['position_shape']}`",f"- chains: {pre['nchains']}",f"- exact unique walkers: {pre['unique_walkers']}",f"- exact duplicates: {pre['exact_duplicate_walkers']}",f"- model leaf shapes match formal n=1000 KFAC: {pre['model_leaf_shapes_match_formal']}",f'- frozen-eval epochs: {len(rows)}',f'- energy mean: {em:.8f} Ha',f'- variance mean: {vm:.8f} Ha^2',f'- acceptance mean: {am:.6f}',f'- finite: {finite}',f"- result: **{'PASS' if passed else 'FAIL'}**",'']
    REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text('\n'.join(lines))
    print('\n'.join(lines))
    if not passed: raise SystemExit(5)
else: raise SystemExit('usage: pre|post')
