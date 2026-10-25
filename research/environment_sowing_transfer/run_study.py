"""Fixed nested fitting sweep; no outer or foreign labels select a recipe."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from maize_clock.experiment import load_data
from maize_clock.transfer_artifacts import verify_inputs,fit_request,validate_artifact
from maize_clock.transfer_design import nested_masks,calibration_metadata
from maize_clock.transfer_fit import fit_run,select_recipe,make_model,clip_gradients
from maize_clock.extension_boosting import LazySequenceFeatures
from maize_clock.extensions import sequence_crps_loss

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/environment_sowing_transfer'
EXPERIMENTS=['germany','zone_0','zone_1','zone_2','warm_years','early_sowing','late_sowing']
SEEDS=[17,29,43]


def recipes():
    configs=[{'mode':m,'bound':1.,'position':'thermal','learning_rate':lr} for m,lr in
             [('standalone_photo',.01),('process',.01),('sequential',.001),('hazard',.001)]]
    for mode in ['recurrent','shared_attention','stage_attention']:
        for bound in [.5,1.]:
            for lr in [.001,.0005]:configs.append(dict(mode=mode,bound=bound,position='thermal',learning_rate=lr,neural_device='mps'))
    # Secondary variants inherit the family-selected recipe after inner selection.
    return configs


def tag(config):
    return config['mode']+'_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:10]


def load():
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    verify_inputs(manifest['input_hashes'],ROOT)
    a,_,source=load_data(ROOT)
    env=pd.read_csv(ROOT/'results/agronomic_transfer/germany/data/environments.csv.gz',low_memory=False)
    design=json.loads((ROOT/'results/agronomic_transfer/design.json').read_text())
    if design['source_arrays_sha256']!=manifest['input_hashes']['data/weather_features.npz']:raise ValueError('Design weather version mismatch')
    return a,env,source,design


def benchmark(neural_device="cpu"):
    torch.set_num_threads(1);torch.manual_seed(17)
    a,env,source,design=load();masks=nested_masks(env,'germany',design['warm_years'])
    m=calibration_metadata(a,env,source,masks['inner_calibration'])
    x=LazySequenceFeatures(a,m,env,history=True)
    ids=np.flatnonzero(masks['inner_calibration'])[:128]
    records=[]
    for mode in ['standalone_photo','process','sequential','recurrent','shared_attention','stage_attention','hazard']:
        config=dict(mode=mode,bound=1.,position='thermal',learning_rate=.001)
        model=make_model(m,config)
        if neural_device=='mps' and mode in ['recurrent','shared_attention','stage_attention']:model.set_neural_device('mps')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        durations=[]
        for step in range(3):
            tick=time.time();optimizer.zero_grad(set_to_none=True)
            _,cdf=model(torch.tensor(x[ids]),torch.tensor(a['G'][ids]),m['maturation_gdd'])
            loss=sequence_crps_loss(cdf,torch.tensor(a['Y'][ids,:7]),torch.ones(7),torch.tensor(a['lengths'][ids]))
            loss.backward();clip_gradients(model.parameters(),1.);optimizer.step()
            if neural_device=='mps':torch.mps.synchronize()
            durations.append(time.time()-tick)
        records.append({'mode':mode,'batch_size':128,'horizon':366,'seconds_per_update':float(np.median(durations[1:])),
                        'neural_device':neural_device,'parameter_count':sum(p.numel() for p in model.parameters()),'outer_test_evaluated':False})
        print(json.dumps(records[-1]),flush=True)
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/('accelerated_benchmark.json' if neural_device=='mps' else 'throughput_benchmark.json')).write_text(json.dumps(records,indent=2)+'\n')


def fit_task(experiment,phase,config,seed,epochs=40):
    a,env,source,design=load();masks=nested_masks(env,experiment,design['warm_years'])
    cal=masks[phase+'_calibration'];tuning=masks[phase+'_tuning']&(a['lengths']==366)
    m=calibration_metadata(a,env,source,cal);x=LazySequenceFeatures(a,m,env,history=True)
    path=OUT/'fits'/experiment/phase/(tag(config)+f'_s{seed}')
    return fit_run(a,env,m,x,np.flatnonzero(cal),np.flatnonzero(tuning),config,seed,path,phase,
                   design['source_arrays_sha256'],epochs=epochs)


def expected_fit(experiment,phase,config,seed,loaded=None,epochs=40):
    a,env,source,design=loaded or load()
    masks=nested_masks(env,experiment,design['warm_years'])
    cal=masks[phase+'_calibration'];tune=masks[phase+'_tuning']&(a['lengths']==366)
    metadata=calibration_metadata(a,env,source,cal)
    return fit_request(metadata,np.flatnonzero(cal),np.flatnonzero(tune),config,seed,phase,design['source_arrays_sha256'],epochs)


def checked_fit(experiment,phase,config,seed,loaded=None):
    path=OUT/'fits'/experiment/phase/(tag(config)+f'_s{seed}')
    return validate_artifact(path,expected_fit(experiment,phase,config,seed,loaded))


def secondary_recipes(stage):
    # Day coordinates are cast to double in the existing attention path.
    # Keep this secondary ablation on CPU, which supports that dtype.
    return {'stage_attention_day':dict(stage,position='day',neural_device='cpu'),
            'stage_attention_balanced':dict(stage,sampling='balanced')}


def select(experiment):
    configs=recipes();selection={};expected=[];loaded=load()
    for mode in sorted(set(c['mode'] for c in configs)):
        reports=[]
        for config in configs:
            if config['mode']!=mode:continue
            for seed in SEEDS:
                path=OUT/'fits'/experiment/'inner'/(tag(config)+f'_s{seed}.json')
                if not path.exists():raise ValueError('Unfinished candidate '+str(path))
                reports.append(checked_fit(experiment,'inner',config,seed,loaded));expected.append(str(path.relative_to(ROOT)))
        selection[mode]=select_recipe(reports)
    stage=selection['stage_attention']
    selection.update(secondary_recipes(stage))
    payload={'experiment':experiment,'selected':selection,'attempted_reports':expected,
             'selection_rule':'Mean three-seed inner-tuning seven-stage CRPS; no outer scores',
             'secondary_variants':'Inherit selected stage-attention bound/learning rate'}
    target=OUT/'selection'/f'{experiment}.json';target.parent.mkdir(exist_ok=True,parents=True)
    target.write_text(json.dumps(payload,indent=2)+'\n');return payload


def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',action='store_true');p.add_argument('--experiment',choices=EXPERIMENTS)
    p.add_argument('--phase',choices=['inner','final']);p.add_argument('--config-index',type=int);p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--select',action='store_true');p.add_argument('--epochs',type=int,default=40);p.add_argument('--selection-key');p.add_argument('--neural-device',choices=['cpu','mps'],default='cpu')
    args=p.parse_args()
    if args.benchmark:return benchmark(args.neural_device)
    if args.select:return select(args.experiment)
    config=recipes()[args.config_index] if args.phase=='inner' else json.loads((OUT/'selection'/f'{args.experiment}.json').read_text())['selected'][args.selection_key]
    fit_task(args.experiment,args.phase,config,args.seed,args.epochs)

if __name__=='__main__':main()
