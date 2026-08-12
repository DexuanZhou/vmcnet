#!/usr/bin/env python3
import csv
import json
import math
import pathlib
import subprocess

HERE=pathlib.Path(__file__).resolve().parent
rows=list(csv.DictReader(open(HERE/'manifest.csv')))
assert len(rows)==6
expected={('C',400,1),('C',800,2),('C',1600,2),('N2eq_R2068',400,1),('N2eq_R2068',800,2),('N2eq_R2068',1600,2)}
assert {(r['system'],int(r['rank']),int(r['warm'])) for r in rows}==expected
for r in rows:
    assert int(r['nchains'])==4096 and int(r['nburn'])==5000 and int(r['nepochs'])==50
    assert int(r['nsteps_per_param_update'])==10 and int(r['seed'])==0
    assert r['optimizer_type']=='wssr_warm_svd_right'
    assert r['sr_block_operator']=='False' and r['svd_method']=='explicit'
    assert r['checkpointing']=='disabled'
    checkpoint=pathlib.Path(r['source_dir'])/'checkpoints'/f"{r['source_epoch']}.npz"
    assert checkpoint.is_file(), checkpoint
    cfg=json.load(open(pathlib.Path(r['source_dir'])/'config.json'))
    assert cfg['initial_seed']==0
    if r['system']=='C':
        assert cfg['problem']['nelec']==[4,2]
        assert cfg['problem']['ion_pos']==[[0.0,0.0,0.0]]
        assert r['eta']=='0.8' and r['learning_rate']=='0.02'
    else:
        assert cfg['problem']['nelec']==[7,7]
        a,b=cfg['problem']['ion_pos']; distance=math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
        assert abs(distance-2.068)<1e-12
        assert r['eta']=='0.2' and r['learning_rate']=='0.002'
    # Offline dry-resolve of the fields that the array passes to the locked
    # source configuration. This intentionally avoids importing JAX/CUDA on
    # the login node.
    resolved={'nchains':int(r['nchains']),'nburn':int(r['nburn']),'nepochs':int(r['nepochs']),'seed':int(r['seed']),'rank_fields':[int(r['rank'])]*4,'warm':int(r['warm']),'geometry':r['ion_pos'],'nelec':r['nelec'],'optimizer_type':r['optimizer_type'],'eta':float(r['eta']),'learning_rate':float(r['learning_rate']),'checkpoint_disabled':r['checkpointing']=='disabled'}
    assert resolved['nchains']==4096 and resolved['nburn']==5000 and resolved['nepochs']==50
    assert resolved['rank_fields']==[int(r['rank'])]*4
    assert resolved['optimizer_type']=='wssr_warm_svd_right' and resolved['checkpoint_disabled']
script=(HERE/'fir_n4096_array.sh').read_text()
launcher=(HERE/'timed_launcher.py').read_text()
for token in ('--array=0-5%2','--time=01:00:00','--gpus-per-node=h100:1','VMCNET_DISABLE_CHECKPOINTS=1','--config.vmc.nchains=4096','--config.vmc.nburn=5000','--config.vmc.nepochs=50','--config.vmc.disable_checkpointing=True','--config.vmc.check_for_nans=True'):
    assert token in script, token
for field in ('sr_rank','sr_rank_max','sr_storage_rank','svd_working_rank'):
    assert f'wssr_warm_svd_right.{field}="${{RANK}}"' in script
assert 'checkpoint_params_loaded_fresh_walkers_retained' in launcher
assert 'return epoch, _fresh["data"], params, optimizer_state, _fresh["key"]' in launcher
subprocess.run(['bash','-n',str(HERE/'fir_n4096_array.sh'),str(HERE/'submit_all.sh')],check=True)
print('VALID: six dry-parsed configs, checkpoints, geometry, ranks, explicit path, checkpoint disable, and shell syntax')
