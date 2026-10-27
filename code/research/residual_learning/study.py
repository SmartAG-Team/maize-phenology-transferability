"""MAE residual study, isolated from frozen fits and deployment artifacts."""
import argparse
import contextlib
import hashlib
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from maize_clock.transfer_artifacts import atomic_json_save, file_sha256

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'data/results/residual_learning'
EXPERIMENTS=['germany','zone_0','zone_1','zone_2','warm_years','early_sowing','late_sowing']
SEEDS=[17,29,43]
STAGES=['emergence','stem_elongation','tasseling','flowering_onset','milking','dough','maturity_bbch87']
SHRINKAGES=[0.,.25,.5,.75,1.]
CONFIGS={'bias':[{}],
         'trees':[{'max_iter':120,'max_leaf_nodes':7,'l2_regularization':1.},
                  {'max_iter':240,'max_leaf_nodes':15,'l2_regularization':1.}],
         'mlp':[{'width':32,'hidden_layers':2,'learning_rate':.001,'max_epochs':80,'patience':10,'batch_size':1024},
                {'width':64,'hidden_layers':2,'learning_rate':.001,'max_epochs':80,'patience':10,'batch_size':1024}]}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def crossfit_partitions(env,calibration,n_folds=3):
    calibration=np.asarray(calibration,bool)
    years=env.year.to_numpy(int)
    unique=np.unique(years[calibration])
    if len(unique)<n_folds+3:raise ValueError('Insufficient cross-fitting years')
    folds=[];coverage=np.zeros(len(env),int)
    for k in range(n_folds):
        held_years=unique[k::n_folds]
        hold=calibration&np.isin(years,held_years)
        remaining=np.setdiff1d(unique,held_years)
        cut=max(1,int(np.floor(.8*len(remaining))))
        fit=calibration&np.isin(years,remaining[:cut])
        tune=calibration&np.isin(years,remaining[cut:])
        assert not np.any(fit&tune|fit&hold|tune&hold)
        assert np.array_equal(fit|tune|hold,calibration)
        assert max(years[fit])<min(years[tune])
        coverage+=hold
        folds.append({'fit':fit,'tune':tune,'hold':hold})
    assert np.array_equal(coverage,calibration.astype(int))
    return folds


def choose_candidate(scores,kind):
    candidates=scores[scores.kind.eq(kind)]
    rows=[]
    for (config_id,shrinkage),group in candidates.groupby(['config_id','shrinkage'],sort=True):
        if sorted(group.seed.tolist())!=SEEDS:raise ValueError('Incomplete candidate seed set')
        if not np.isfinite(group.tuning_mae).all():raise ValueError('Nonfinite tuning MAE')
        rows.append({'config_id':config_id,'shrinkage':float(shrinkage),
                     'tuning_mae':float(group.tuning_mae.mean())})
    if not rows:raise ValueError('No residual candidates')
    return min(rows,key=lambda row:(row['tuning_mae'],row['shrinkage'],row['config_id']))


def point_scores(env,truth,base,raw_dates,dates,model,seed,cohort):
    truth=np.asarray(truth);base=np.asarray(base);dates=np.asarray(dates)
    rows=[]
    for j,endpoint in enumerate(STAGES):
        for i in np.flatnonzero((truth[:,j]>0)&(truth[:,j]<=366)):
            row=env.iloc[i]
            observed=float(truth[i,j]);predicted=float(dates[i,j]);starting=float(base[i,j])
            reached=np.isfinite(predicted) and 1<=predicted<=366
            paired=np.isfinite(starting) and 1<=starting<=366
            error=predicted-observed if reached else np.nan
            rows.append(dict(environment_id_model=row.environment_id_model,site_id=row.site_id,
                             year=int(row.year),sowing_date=row.sowing_date,endpoint=endpoint,
                             spatial_group=row.get('spatial_group',row.site_id),model=model,seed=seed,
                             cohort=cohort,observed_day=observed,baseline_day=starting,
                             raw_predicted_day=float(raw_dates[i,j]),predicted_day=predicted,
                             rounded_predicted_day=float(np.rint(predicted)) if reached else np.nan,
                             absolute_error=abs(error),bias=error,
                             rounded_absolute_error=abs(np.rint(predicted)-observed) if reached else np.nan,
                             paired_baseline_support=paired,unresolved_prediction=not reached,
                             penalized_absolute_error=abs(error) if reached else 366.,
                             within3=bool(reached and abs(error)<=3),within5=bool(reached and abs(error)<=5),
                             within7=bool(reached and abs(error)<=7)))
    return pd.DataFrame(rows)


def atomic_npz(path,**values):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,**values)
    tmp.replace(path)


def load_sources():
    from research.environment_sowing_transfer.run_study import load
    return load()


def predict_tp(model,metadata,arrays,ids,batch=512):
    """Exact median crossings of the frozen marginal T–P date distributions."""
    torch.set_num_threads(1);ids=np.asarray(ids,int);parts=[];model.eval()
    mean=np.asarray(metadata['feature_mean'],np.float32)
    std=np.asarray(metadata['feature_std'],np.float32)
    with torch.no_grad():
        thresholds=model.thresholds().numpy()
        for start in range(0,len(ids),batch):
            ix=ids[start:start+batch]
            standardized=(arrays['X'][ix]-mean)/std
            progress=model.process_progress(torch.as_tensor(standardized),torch.as_tensor(arrays['G'][ix])).numpy()
            hits=(progress[:,:,None]>=thresholds[None,None,:])
            hits&=np.arange(1,progress.shape[1]+1)[None,:,None]<=arrays['lengths'][ix,None,None]
            parts.append(np.where(hits.any(1),hits.argmax(1)+1,np.nan))
    return np.concatenate(parts) if parts else np.empty((0,7))


def masks_for(env,arrays,design,experiment):
    from maize_clock.transfer_design import nested_masks
    masks=nested_masks(env,experiment,design['warm_years'])
    return {k:v&(arrays['lengths']==366) for k,v in masks.items()}


def physical_preparation(experiment):
    from maize_clock.transfer_fit import fit_run,restore
    from maize_clock.transfer_design import calibration_metadata
    from maize_clock.extension_boosting import LazySequenceFeatures
    from research.environment_sowing_transfer.run_study import checked_fit,tag
    arrays,env,source,design=load_sources()
    masks=masks_for(env,arrays,design,experiment)
    path=OUT/experiment;path.mkdir(parents=True,exist_ok=True)
    selection=json.loads((ROOT/'data/results/environment_sowing_transfer/selection'/f'{experiment}.json').read_text())
    config=selection['selected']['standalone_photo']
    signature={'experiment':experiment,'source_sha256':design['source_arrays_sha256'],
               'calibration_ids_sha256':hashlib.sha256(np.flatnonzero(masks['final_calibration']).astype('<i8').tobytes()).hexdigest(),
               'crossfit':3,'crossfit_seed':17,'config':config}
    cached=path/'oof_baseline.npz'
    sidecar=path/'physical_preparation.json'
    if cached.exists() and sidecar.exists():
        receipt=json.loads(sidecar.read_text())
        if receipt['signature']!=signature or receipt['oof_sha256']!=file_sha256(cached):raise ValueError('Changed cross-fit cache')
    else:
        oof=np.full((len(env),7),np.nan);fold_index=np.full(len(env),-1,int);audit=[]
        for k,fold in enumerate(crossfit_partitions(env,masks['final_calibration'])):
            fit,tune,hold=(np.flatnonzero(fold[key]) for key in ['fit','tune','hold'])
            metadata=calibration_metadata(arrays,env,source,fold['fit'])
            features=LazySequenceFeatures(arrays,metadata,env,history=True)
            stem=path/f'physical_fold_{k}'
            print(json.dumps({'event':'physical_fold_start','experiment':experiment,'fold':k,'fit':len(fit),'tune':len(tune),'hold':len(hold)}),flush=True)
            with (path/f'physical_fold_{k}.log').open('a') as stream,contextlib.redirect_stdout(stream):
                fit_run(arrays,env,metadata,features,fit,tune,config,17,stem,'residual_crossfit',design['source_arrays_sha256'])
            model,saved=restore(stem.with_suffix('.pt'))
            oof[hold]=predict_tp(model,saved['metadata'],arrays,hold)
            fold_index[hold]=k
            np.savez_compressed(path/f'physical_fold_{k}_membership.npz',fit=fit,tune=tune,hold=hold)
            audit.append({'fold':k,'fit':len(fit),'tune':len(tune),'hold':len(hold),
                          'hold_years':sorted(env.iloc[hold].year.unique().astype(int).tolist()),
                          'checkpoint_sha256':file_sha256(stem.with_suffix('.pt'))})
            print(json.dumps({'event':'physical_fold_complete','experiment':experiment,'fold':k}),flush=True)
        assert np.array_equal(fold_index>=0,masks['final_calibration'])
        atomic_npz(cached,baseline_days=oof,fold_index=fold_index)
        atomic_json_save({'signature':signature,'oof_sha256':file_sha256(cached),'folds':audit},sidecar)
    for seed in SEEDS:
        checked_fit(experiment,'final',config,seed,(arrays,env,source,design))
        checkpoint=ROOT/'data/results/environment_sowing_transfer/fits'/experiment/'final'/(tag(config)+f'_s{seed}.pt')
        output=path/f'frozen_baseline_s{seed}.npz';meta=output.with_suffix('.json')
        request={'checkpoint_sha256':file_sha256(checkpoint),'source_sha256':design['source_arrays_sha256']}
        if output.exists() and meta.exists():
            old=json.loads(meta.read_text())
            if old['request']!=request or old['sha256']!=file_sha256(output):raise ValueError('Changed frozen baseline cache')
        else:
            model,saved=restore(checkpoint)
            base=predict_tp(model,saved['metadata'],arrays,np.arange(len(env)))
            atomic_npz(output,baseline_days=base)
            atomic_json_save({'request':request,'sha256':file_sha256(output)},meta)
    print(json.dumps({'event':'physical_preparation_complete','experiment':experiment}),flush=True)
    return experiment


def candidate_id(kind,config):
    return kind+'_'+digest(config)[:10]


def frozen_foreign_baselines(seed,arrays,env,source,design):
    from maize_clock.transfer_fit import restore
    from research.environment_sowing_transfer.run_study import checked_fit,tag
    config=json.loads((ROOT/'data/results/environment_sowing_transfer/selection/germany.json').read_text())['selected']['standalone_photo']
    checked_fit('germany','final',config,seed,(arrays,env,source,design))
    checkpoint=ROOT/'data/results/environment_sowing_transfer/fits/germany/final'/(tag(config)+f'_s{seed}.pt')
    return restore(checkpoint)


def run_experiment(experiment):
    from .features import build_features,project_dates
    from .learners import fit_model,predict_residual,correct_dates,macro_mae
    arrays,env,source,design=load_sources();path=OUT/experiment
    masks=masks_for(env,arrays,design,experiment)
    fit=np.flatnonzero(masks['final_calibration']);tune=np.flatnonzero(masks['final_tuning']);test=np.flatnonzero(masks['test'])
    with np.load(path/'oof_baseline.npz') as z:oof=z['baseline_days'][fit]
    subset=lambda ids:{k:v[ids] for k,v in arrays.items() if k in ['X','G','lengths']}
    xfit,names=build_features(subset(fit),env.iloc[fit],oof)
    truthfit=arrays['Y'][fit,:7].astype(float)
    residual=np.where((truthfit>0)&np.isfinite(oof),truthfit-oof,np.nan)
    candidates=[];fitted={};basecache={};featurecache={}
    code_signature={f.name:file_sha256(f) for f in Path(__file__).parent.glob('*.py') if f.name!='report.py'}
    for seed in SEEDS:
        with np.load(path/f'frozen_baseline_s{seed}.npz') as z:base=z['baseline_days']
        basecache[seed]=base
        xtune,_=build_features(subset(tune),env.iloc[tune],base[tune])
        xtest,_=build_features(subset(test),env.iloc[test],base[test])
        featurecache[seed]=(xtune,xtest)
        for kind,configs in CONFIGS.items():
            for config in configs:
                tag=candidate_id(kind,config);dest=path/f'{tag}_s{seed}.pkl';side=dest.with_suffix('.json')
                request={'source_sha256':design['source_arrays_sha256'],'oof_sha256':file_sha256(path/'oof_baseline.npz'),
                         'base_sha256':file_sha256(path/f'frozen_baseline_s{seed}.npz'),
                         'code_signature':code_signature,'config':config,'kind':kind,'seed':seed,
                         'fit_index_sha256':hashlib.sha256(fit.astype('<i8').tobytes()).hexdigest(),
                         'tune_index_sha256':hashlib.sha256(tune.astype('<i8').tobytes()).hexdigest()}
                if dest.exists():
                    old=json.loads(side.read_text())
                    if old['request']!=request or old['sha256']!=file_sha256(dest):raise ValueError('Changed residual artifact')
                    with dest.open('rb') as stream:model=pickle.load(stream)
                else:
                    print(json.dumps({'event':'residual_fit_start','experiment':experiment,'model':tag,'seed':seed}),flush=True)
                    model=fit_model(kind,xfit,residual,xtune,arrays['Y'][tune,:7],base[tune],config,seed)
                    with dest.with_suffix('.tmp').open('wb') as stream:pickle.dump(model,stream,protocol=5)
                    dest.with_suffix('.tmp').replace(dest)
                    atomic_json_save({'request':request,'sha256':file_sha256(dest),'fit_complete':True,
                                      'best_epoch':model.get('best_epoch'),'history':model.get('history',[])},side)
                fitted[(seed,tag)]=model
                pred=predict_residual(model,xtune)
                for shrinkage in SHRINKAGES:
                    dates=correct_dates(base[tune],pred,shrinkage)
                    candidates.append({'kind':kind,'config_id':tag,'seed':seed,'shrinkage':shrinkage,
                                       'tuning_mae':macro_mae(arrays['Y'][tune,:7],dates,base[tune])})
                print(json.dumps({'event':'residual_fit_complete','experiment':experiment,'model':tag,'seed':seed}),flush=True)
    candidate_table=pd.DataFrame(candidates);candidate_table.to_csv(path/'candidate_tuning.csv',index=False)
    selected={kind:choose_candidate(candidate_table,kind) for kind in CONFIGS}
    atomic_json_save({'selected':selected,'metric':'stage_equal_MAE','seeds':SEEDS,'feature_names':names,
                      'foreign_or_test_outcomes_used_for_selection':False},path/'selection.json')
    results=[]
    for seed in SEEDS:
        base=basecache[seed];xtune,xtest=featurecache[seed]
        baseline=point_scores(env.iloc[test],arrays['Y'][test,:7],base[test],base[test],base[test],'tp',seed,experiment)
        results.append(baseline)
        for kind,choice in selected.items():
            model=fitted[(seed,choice['config_id'])]
            predicted=predict_residual(model,xtest)
            raw=base[test].copy() if choice['shrinkage']==0 else base[test]+choice['shrinkage']*predicted
            dates=correct_dates(base[test],predicted,choice['shrinkage'])
            atomic_npz(path/f'{kind}_s{seed}_test_predictions.npz',environment_ids=env.iloc[test].environment_id_model.to_numpy(str),
                       truth=arrays['Y'][test,:7],baseline_days=base[test],residual=predicted,raw_dates=raw,dates=dates)
            results.append(point_scores(env.iloc[test],arrays['Y'][test,:7],base[test],raw,dates,kind,seed,experiment))
    pd.concat(results,ignore_index=True).to_csv(path/'test_scores.csv.gz',index=False)
    if experiment=='germany':
        evaluate_foreign(arrays,env,source,design,path,selected,fitted)
    atomic_json_save({'complete':True,'experiment':experiment,'primary_metric':'MAE','calibration_environments':len(fit),
                      'tuning_environments':len(tune),'test_environments':len(test),'new_residual_fits':len(fitted),
                      'crossfit_physical_fits':3,'oof_physical_seed':17,'historical_test_inspection':True,
                      'deployment_modified':False},path/'status.json')
    print(json.dumps({'event':'experiment_complete','experiment':experiment,'selected':selected}),flush=True)


def evaluate_foreign(arrays,env,source,design,path,selected,fitted):
    from .features import build_features
    from .learners import predict_residual,correct_dates
    from maize_clock.transfer_adaptation import observed_targets,local_masks
    from research.environment_sowing_transfer.evaluate_foreign import compatible_external
    china=ROOT/'data/results/agronomic_transfer/china'
    with np.load(china/'external_weather_features.npz') as z:ca={k:z[k] for k in z.files}
    ce=pd.read_csv(china/'external_environments.csv');cy=observed_targets(ce,pd.read_csv(china/'station_targets.csv'))
    ci=np.flatnonzero(local_masks(ce)['test']&(ca['lengths']==366))
    annotations=pd.read_csv(ROOT/'data/results/country_transfer/cropping_system_audit/environment_season_annotations.csv')
    fy,fi,gucheng=compatible_external(arrays,env,annotations)
    pieces=[]
    for seed in SEEDS:
        model,saved=frozen_foreign_baselines(seed,arrays,env,source,design)
        cohorts=[('North China later years',ca,ce,cy,ci)]
        for label,ids in [('Gucheng summer trials',[i for i in fi if env.iloc[i].environment_id_model in gucheng]),
                          ('Other compatible foreign emergence',[i for i in fi if env.iloc[i].environment_id_model not in gucheng])]:
            cohorts.append((label,arrays,env,fy,np.asarray(ids,int)))
        for label,a,e,y,ids in cohorts:
            if not len(ids):continue
            base=predict_tp(model,saved['metadata'],a,ids)
            sub={k:a[k][ids] for k in ['X','G','lengths']};x,_=build_features(sub,e.iloc[ids],base)
            pieces.append(point_scores(e.iloc[ids],y[ids],base,base,base,'tp',seed,label))
            for kind,choice in selected.items():
                residual_model=fitted[(seed,choice['config_id'])]
                pred=predict_residual(residual_model,x)
                raw=base.copy() if choice['shrinkage']==0 else base+choice['shrinkage']*pred
                dates=correct_dates(base,pred,choice['shrinkage'])
                pieces.append(point_scores(e.iloc[ids],y[ids],base,raw,dates,kind,seed,label))
                safe=label.lower().replace(' ','_')
                atomic_npz(path/f'{kind}_s{seed}_{safe}_predictions.npz',environment_ids=e.iloc[ids].environment_id_model.to_numpy(str),
                           truth=y[ids],baseline_days=base,residual=pred,raw_dates=raw,dates=dates)
    pd.concat(pieces,ignore_index=True).to_csv(path/'foreign_scores.csv.gz',index=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiments',nargs='+',default=EXPERIMENTS)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if any(e not in EXPERIMENTS for e in args.experiments):raise ValueError('Unknown experiment')
    started=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for completed in pool.map(physical_preparation,args.experiments):
            print(json.dumps({'prepared':completed,'elapsed_seconds':time.time()-started}),flush=True)
    if not args.prepare_only:
        for experiment in args.experiments:run_experiment(experiment)


if __name__=='__main__':main()
