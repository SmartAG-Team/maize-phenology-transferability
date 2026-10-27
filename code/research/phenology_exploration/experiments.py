"""Frozen memberships, source-only selection and original-key prediction exports."""
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from .data import ROOT,OUT,load,freeze_design,source_units,sample_units
from .core import metric,sha
from .models import (fit_calendar,fit_thermal,fit_context,predict,
                     teacher_predictions,fit_residual,residual_prediction)
from .provenance import code_signature,validate_receipt

ALPHAS=[.5,.625,.75,.875,1.,1.125,1.25,1.5,2.]
REGULARIZATION=np.logspace(-3,5,9).tolist()


def row_predictions(frame,ids,dates):
    if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate prediction native units')
    lookup={int(u):i for i,u in enumerate(ids)}
    rows=frame.u.map(lookup)
    if rows.isna().any():raise ValueError('Missing native-unit prediction')
    return dates[rows.to_numpy(int),frame['head'].to_numpy(int)]


def guard_factor(distance,strength):return np.exp(-float(strength)*np.asarray(distance))


def climate_distance(data,source_ids,target_ids):
    x=data.context[source_ids,:2];mean=x.mean(0);scale=x.std(0).clip(.1)
    model=NearestNeighbors(n_neighbors=1).fit((x-mean)/scale)
    return model.kneighbors((data.context[target_ids,:2]-mean)/scale)[0].ravel()


def _score(data,model,frame):
    ids=frame.u.unique();p=predict(model,data,ids,train=True)
    return metric(frame,row_predictions(frame,ids,p),data.train_lengths[frame.u])['penalized_mae']


def choose_fit(data,train,tune,source,name,base_train,base_final):
    trials=[]
    if name=='context_T':
        for parameter in REGULARIZATION:
            m=fit_context(data,train,parameter)
            trials.append(dict(parameter=parameter,mae=_score(data,m,tune)))
        best=min(trials,key=lambda x:(x['mae'],-x['parameter']))
        return fit_context(data,source,best['parameter']),trials
    for alpha in ALPHAS:
        m=fit_thermal(data,train,alpha,phase=name=='phase_T',base=base_train if name=='phase_T' else None)
        trials.append(dict(parameter=alpha,mae=_score(data,m,tune)))
    best=min(trials,key=lambda x:(x['mae'],abs(x['parameter']-1)))
    return fit_thermal(data,source,best['parameter'],phase=name=='phase_T',
                       base=base_final if name=='phase_T' else None),trials


def make_cases(data,design):
    folder=OUT/'memberships';folder.mkdir(exist_ok=True);cases=[];skipped=[]
    for fold in range(5):
        units=source_units(data,fold)
        requested=[]
        for sites in design['site_budgets']:
            for draw in design['draw_seeds']:
                for recipe in design['primary_recipes']:
                    requested.append((recipe,sites,draw,None))
        for ez in range(5):requested.append(('random',40,17,ez))
        for recipe,sites,draw,ez in requested:
            name=f'f{fold}_{recipe}_n{sites}_d{draw}'+('' if ez is None else f'_sourceEZ{ez+1}')
            try:sample=sample_units(units,sites,design['years_per_site'],recipe,draw,ez)
            except ValueError as error:
                skipped.append(dict(case=name,reason=str(error)));continue
            labels=data.labels[data.labels.unit.isin(sample.unit)&data.labels.pre2011_prefix_eligible]
            inner=sample.set_index('unit').inner
            fit=labels[labels.unit.map(inner).ne(2)]
            support=fit.groupby('head').u.nunique().reindex(range(7),fill_value=0)
            if support.min()<30:
                skipped.append(dict(case=name,reason='Fewer than 30 inner-fit native environments at a frozen head',support=support.tolist()));continue
            destination=folder/f'{name}.csv.gz'
            if destination.exists():
                saved=pd.read_csv(destination)
                if not saved.astype(str).equals(sample.astype(str)):
                    raise ValueError('Existing source membership changed')
            else:sample.to_csv(destination,index=False,compression={'method':'gzip','mtime':0})
            cases.append(dict(case=name,fold=fold,recipe=recipe,n_sites=sites,years_per_site=design['years_per_site'],
                draw_seed=draw,source_ez=ez,native_units=len(sample),inner_head_support=support.tolist(),
                membership_sha256=sha(folder/f'{name}.csv.gz')))
    path=OUT/'cases.json';payload=dict(cases=cases,skipped=skipped)
    if path.exists() and json.loads(path.read_text())!=payload:raise ValueError('Frozen memberships changed')
    path.write_text(json.dumps(payload,indent=2)+'\n')
    return cases


def fit_case(data,case,design,smoke=False):
    start=time.monotonic();name=case['case'];folder=OUT/('smoke' if smoke else 'cases')/name
    folder.mkdir(parents=True,exist_ok=True)
    code=code_signature()
    signature=dict(case=case,code=code,inputs=data.protected)
    done=folder/'complete.json'
    if done.exists():
        prior=json.loads(done.read_text())
        validate_receipt(prior,signature,folder)
        print(json.dumps(dict(event='reuse',case=name)),flush=True);return
    sample=pd.read_csv(OUT/'memberships'/f'{name}.csv.gz')
    source=data.labels[data.labels.unit.isin(sample.unit)&data.labels.pre2011_prefix_eligible].copy()
    inner=sample.set_index('unit').inner;train=source[source.unit.map(inner).ne(2)];tune=source[source.unit.map(inner).eq(2)]
    role=f'fold{case["fold"]}_case';target=data.labels[data.labels[role].isin(['historical_held','later_held'])].copy()
    target['case']=target[role]
    if set(source.site)&set(target.site):raise ValueError('Outer site leakage')
    if set(train.block)&set(tune.block):raise ValueError('Inner block leakage')
    fits={'calendar':fit_calendar(data,source),'T':fit_thermal(data,source)}
    base_train=fit_thermal(data,train);selection={}
    for model in ['context_T','nonlinear_T','phase_T']:
        fits[model],selection[model]=choose_fit(data,train,tune,source,model,base_train,fits['T'])
    print(json.dumps(dict(event='physical_fits',case=name,seconds=round(time.monotonic()-start,1))),flush=True)
    teacher_inner,inner_receipt=teacher_predictions(data,train)
    inner_qualified=np.array([r['support'] for r in inner_receipt]).min(0)>=30
    tune_ids=tune.u.unique();bt=predict(base_train,data,tune_ids,train=True)
    distance=climate_distance(data,train.u.unique(),tune_ids)
    rf=fit_residual(data,train,teacher_inner,'RF',17,teacher_qualified=inner_qualified)
    correction=residual_prediction(rf,data,tune_ids,bt,train=True)
    rf_trials=[];guard_trials=[]
    for strength in np.linspace(0,1,9):
        score=metric(tune,row_predictions(tune,tune_ids,bt+strength*correction),data.train_lengths[tune.u])['penalized_mae']
        rf_trials.append(dict(strength=float(strength),mae=score))
    for strength in [0.,.5,1.]:
        for k in [0.,1.,4.]:
            dates=bt+strength*guard_factor(distance,k)[:,None]*correction
            guard_trials.append(dict(strength=strength,k=k,mae=metric(tune,row_predictions(tune,tune_ids,dates),data.train_lengths[tune.u])['penalized_mae']))
    rf_choice=min(rf_trials,key=lambda x:(x['mae'],x['strength']))
    gate_choice=min(guard_trials,key=lambda x:(x['mae'],x['strength'],x['k']))
    neural_trials=[]
    for steps in [80,160,320]:
        m=fit_residual(data,train,teacher_inner,'MLP',17,steps,teacher_qualified=inner_qualified)
        correction=residual_prediction(m,data,tune_ids,bt,train=True)
        for strength in [0.,.5,1.]:
            score=metric(tune,row_predictions(tune,tune_ids,bt+strength*correction),data.train_lengths[tune.u])['penalized_mae']
            neural_trials.append(dict(steps=steps,strength=strength,mae=score))
    mlp_choice=min(neural_trials,key=lambda x:(x['mae'],x['strength'],x['steps']))
    teacher_final,final_receipt=teacher_predictions(data,source)
    final_qualified=np.array([r['support'] for r in final_receipt]).min(0)>=30
    fits['residual_RF']=fit_residual(data,source,teacher_final,'RF',17,teacher_qualified=final_qualified)
    for seed in design['optimizer_seeds']:
        key='residual_MLP' if seed==17 else f'residual_MLP_init{seed}'
        fits[key]=fit_residual(data,source,teacher_final,'MLP',seed,mlp_choice['steps'],teacher_qualified=final_qualified)
    selection.update(residual_RF=rf_trials,guarded_RF=guard_trials,residual_MLP=neural_trials)
    ids=target.u.unique();base=predict(fits['T'],data,ids);dates={}
    for model in ['calendar','T','context_T','nonlinear_T','phase_T']:
        dates[model]=predict(fits[model],data,ids)
    rf_delta=residual_prediction(fits['residual_RF'],data,ids,base)
    dates['residual_RF']=base+rf_choice['strength']*rf_delta
    distance=climate_distance(data,source.u.unique(),ids)
    dates['guarded_RF']=base+gate_choice['strength']*guard_factor(distance,gate_choice['k'])[:,None]*rf_delta
    for key in [x for x in fits if x.startswith('residual_MLP')]:
        dates[key]=base+mlp_choice['strength']*residual_prediction(fits[key],data,ids,base)
    columns=['label_id','unit','u','site','site_year','block','year','ez','head','observed','case','horizon']
    exported=target[columns].copy();metrics=[]
    for model,pred in dates.items():
        exported[model]=row_predictions(target,ids,pred)
        for period,q in exported.groupby('case'):
            metrics.append(dict(period=period,model=model,**metric(q,q[model].to_numpy())))
    assert exported.label_id.is_unique
    exported.to_csv(folder/'predictions.csv.gz',index=False)
    pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
    with (folder/'fits.pkl').open('wb') as f:
        pickle.dump(dict(models=fits,rf_choice=rf_choice,gate_choice=gate_choice,mlp_choice=mlp_choice),f)
    receipt=dict(signature=signature,source_labels=len(source),source_units=source.u.nunique(),source_sites=source.site.nunique(),
        inner_fit_sites=train.site.nunique(),inner_validation_sites=tune.site.nunique(),source_years=[int(source.year.min()),int(source.year.max())],
        target_labels=len(exported),target_label_sha256=__import__('hashlib').sha256('\n'.join(sorted(exported.label_id)).encode()).hexdigest(),
        phase_anchors='source-fitted predicted flowering; observed target flowering never supplied',
        selections=dict(rf=rf_choice,guard=gate_choice,mlp=mlp_choice),selection_trials=selection,
        inner_teachers=inner_receipt,complete_source_teachers=final_receipt,
        residual_head_audits={k:{v:m[v] for v in ['supported','teacher_head_qualified','residual_training_environments',
            'available_labels_per_head','invalid_teacher_or_features_per_head']} for k,m in fits.items() if k.startswith('residual_')},
        source_context='1981–2010 reference climate and sowing/daylength covariates; no station identity',
        residual_weather='through teacher event; retrospective timing, not operational event-time forecasts',
        elapsed_seconds=time.monotonic()-start,outputs={p.name:sha(p) for p in folder.iterdir() if p.name!='complete.json'})
    done.write_text(json.dumps(receipt,indent=2,default=lambda v:int(v) if isinstance(v,np.integer) else float(v))+'\n')
    print(json.dumps(dict(event='case_complete',case=name,seconds=round(receipt['elapsed_seconds'],1),
                         selections=receipt['selections'])),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--limit',type=int)
    parser.add_argument('--fold',type=int);parser.add_argument('--case');args=parser.parse_args()
    data=load();design=freeze_design(data);cases=make_cases(data,design)
    if args.fold is not None:cases=[c for c in cases if c['fold']==args.fold]
    if args.case:cases=[c for c in cases if c['case']==args.case]
    if args.smoke:cases=cases[:1]
    if args.limit:cases=cases[:args.limit]
    print(json.dumps(dict(event='start',cases=len(cases),smoke=args.smoke)),flush=True)
    for case in cases:fit_case(data,case,design,args.smoke)
    for path,digest in data.protected.items():
        if sha(path)!=digest:raise ValueError(f'Protected input changed: {path}')
    print(json.dumps(dict(event='selected_cases_finished',cases=len(cases),protected_inputs_unchanged=True)),flush=True)


if __name__=='__main__':main()
