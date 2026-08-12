#!/usr/bin/env python3
import json
import math
import pathlib
import re
import subprocess

here = pathlib.Path(__file__).resolve().parent
formal = pathlib.Path('/scratch/dexuan1/runs/kfac_pre/N2eq_R2068_kfac_pre5000/config.json')
checkpoint = formal.parent / 'checkpoints/5000.npz'
cfg = json.loads(formal.read_text())
assert checkpoint.is_file()
assert cfg['problem']['ion_pos'] == [[0.0, 0.0, -1.034], [0.0, 0.0, 1.034]]
assert cfg['problem']['ion_charges'] == [7.0, 7.0] and cfg['problem']['nelec'] == [7, 7]
assert cfg['vmc']['nchains'] == 1000 and cfg['vmc']['nburn'] == 5000 and cfg['vmc']['nepochs'] == 5000
assert cfg['vmc']['nsteps_per_param_update'] == 10 and cfg['initial_seed'] == 0
assert cfg['vmc']['optimizer']['kfac']['learning_rate'] == 0.05
assert cfg['vmc']['optimizer']['kfac']['damping'] == 0.001
assert cfg['vmc']['optimizer']['kfac']['norm_constraint'] == 0.001
assert cfg['vmc']['optimizer']['kfac']['schedule_type'] == 'inverse_time'
assert math.isclose(cfg['vmc']['optimizer']['kfac']['learning_decay_rate'], 1e-4)

kfac = (here/'scripts/N2_kfac_preliminary_n4096.sh').read_text()
validation = (here/'scripts/N2_validate_kfac_n4096.sh').read_text()
array = (here/'scripts/N2_corrected_wssr_array.sh').read_text()
submit = (here/'submit_corrected_n2_pipeline.sh').read_text()
for token in ('--gpus-per-node=h100:1', '--time=03:00:00', '--config.vmc.nchains=4096', '--reload.use_checkpoint_file=False'):
    assert token in kfac, token
for token in ('--reload.use_checkpoint_file=True', '--reload.reburn=False', '--config.eval.use_data_from_training=True'):
    assert token in validation, token
for token in ('#SBATCH --array=0-2%2', 'COMBOS=(400:1 800:2 1600:2)', 'VMCNET_DISABLE_CHECKPOINTS=1',
              '--reload.use_checkpoint_file=True', '--reload.new_optimizer_state=True', '--reload.reburn=False',
              '--config.vmc.nchains=4096', '--config.vmc.nepochs=50', '--config.vmc.disable_checkpointing=True',
              '--config.vmc.optimizer_type=wssr_warm_svd_right', '--config.vmc.optimizer.wssr_warm_svd_right.eta=0.2',
              '--config.vmc.optimizer.wssr_warm_svd_right.learning_rate=0.002',
              '--config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization=tikhonov'):
    assert token in array, token
for field in ('sr_rank', 'sr_rank_max', 'sr_storage_rank', 'svd_working_rank'):
    assert f'wssr_warm_svd_right.{field}="${{RANK}}"' in array
assert re.search(r'afterok:\$\{KFAC\}.*N2_validate', submit)
assert re.search(r'afterok:\$\{VALIDATION\}.*N2_corrected', submit)
assert re.search(r'afterany:\$\{WSSR\}.*N2_collect', submit)
subprocess.run(['bash', '-n', str(here/'submit_corrected_n2_pipeline.sh'), *map(str, (here/'scripts').glob('N2_*.sh'))], check=True)
print('VALID: formal R=2.068 KFAC source, n4096 preliminary, checkpoint gate, 3 corrected ranks, no WSSR checkpoints, and dependency graph')
