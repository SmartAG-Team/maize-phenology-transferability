"""Same-budget source refits under annual-temperature zone restriction."""
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from research.phenology_exploration.models import fit_thermal,predict,teacher_predictions,fit_residual,residual_prediction
from research.phenology_exploration.core import metric,sha
from research.phenology_exploration.experiments import row_predictions
from research.phenology_exploration.neural import fit_network,predict_network
from research.phenology_exploration.support_blend import support_weight,blend_dates
from research.phenology_exploration.provenance import code_signature,validate_receipt
from .climate import OUT
from .design import dataset,freeze
from .core import temperature_distance

ALPHAS=[1/64,1/32,1/16,1/8,1/4,.5,.75,1.,1.5]


def signature(data,case):
    extra=code_signature('analysis.py','neural.py','support_blend.py')
    for name in ['core.py','climate.py','design.py','fit.py']:
        p=Path(__file__).parent/name;extra[str(p)]=sha(p)
    return dict(case=case,code=extra,inputs=data.protected,design_sha256=sha(OUT/'design.json'))


def run_case(data,case):
    name=case['case'];folder=OUT/'cases'/name;folder.mkdir(parents=True,exist_ok=True)
    sig=signature(data,case)
    if (folder/'complete.json').exists():validate_receipt(json.loads((folder/'complete.json').read_text()),sig,folder);return
    started=time.monotonic();members=pd.read_csv(OUT/'memberships'/f'{name}.csv.gz')
    assert sha(OUT/'memberships'/f'{name}.csv.gz')==case['membership_sha256']
    source=data.labels[data.labels.unit.isin(members.unit)&data.labels.pre2011_prefix_eligible]
    inner=members.set_index('unit').inner;train=source[source.unit.map(inner).ne(2)];tune=source[source.unit.map(inner).eq(2)]
    assert not set(train.block)&set(tune.block)
    role=f'fold{case["fold"]}_case';target=data.labels[data.labels[role].isin(['historical_held','later_held'])].copy()
    assert not set(source.site)&set(target.site)
    target['period']=target[role];ids=target.u.unique();ti=tune.u.unique()
    train_T=fit_thermal(data,train);final_T=fit_thermal(data,source);trials=[]
    for alpha in ALPHAS:
        m=fit_thermal(data,train,alpha);p=predict(m,data,ti,train=True)
        trials.append(dict(alpha=alpha,mae=metric(tune,row_predictions(tune,ti,p),data.train_lengths[tune.u])['penalized_mae']))
    best=min(trials,key=lambda q:(q['mae'],abs(q['alpha']-1)))
    fits={'T':final_T,'nonlinear_T':fit_thermal(data,source,best['alpha'])}
    teacher,receipts=teacher_predictions(data,train);qualified=np.array([x['support'] for x in receipts]).min(0)>=30
    base_tune=predict(train_T,data,ti,train=True)
    rf=fit_residual(data,train,teacher,'RF',17,teacher_qualified=qualified)
    correction=residual_prediction(rf,data,ti,base_tune,train=True);rf_trials=[];mlp_trials=[]
    for strength in np.linspace(0,1,9):
        rf_trials.append(dict(strength=float(strength),mae=metric(tune,row_predictions(tune,ti,base_tune+strength*correction),data.train_lengths[tune.u])['penalized_mae']))
    rf_choice=min(rf_trials,key=lambda q:(q['mae'],q['strength']))
    for steps in [80,160,320]:
        m=fit_residual(data,train,teacher,'MLP',17,steps,teacher_qualified=qualified)
        correction=residual_prediction(m,data,ti,base_tune,train=True)
        for strength in [0.,.5,1.]:
            mlp_trials.append(dict(steps=steps,strength=strength,mae=metric(tune,row_predictions(tune,ti,base_tune+strength*correction),data.train_lengths[tune.u])['penalized_mae']))
    mlp_choice=min(mlp_trials,key=lambda q:(q['mae'],q['strength'],q['steps']))
    full_teacher,full_receipts=teacher_predictions(data,source);full_qualified=np.array([x['support'] for x in full_receipts]).min(0)>=30
    fits['RF']=fit_residual(data,source,full_teacher,'RF',17,teacher_qualified=full_qualified)
    fits['MLP']=fit_residual(data,source,full_teacher,'MLP',17,mlp_choice['steps'],teacher_qualified=full_qualified)
    base=predict(final_T,data,ids);physical=predict(fits['nonlinear_T'],data,ids)
    neural=base+mlp_choice['strength']*residual_prediction(fits['MLP'],data,ids,base)
    forest=base+rf_choice['strength']*residual_prediction(fits['RF'],data,ids,base)
    # This is the established two-climate-feature rule; annual MAT changes the sampling frame only.
    unique_source=members.drop_duplicates('site');weight,geometry=support_weight(data.context[unique_source.u,:2],data.context[ids,:2])
    blend=blend_dates(neural,physical,weight[:,None],data.lengths[ids,None])
    network,norm,steps,nn_trials=fit_network(data,train,train_T['thresholds'],False,tune=tune)
    network,norm,_,_=fit_network(data,source,final_T['thresholds'],False,fixed_steps=steps)
    nn=predict_network(network,data,ids,norm,False)
    torch.save(dict(state={k:v.cpu() for k,v in network.state_dict().items()},norm=norm,initial=final_T['thresholds'],context=False,selected_steps=steps),folder/'thermal_transformer.pt')
    out=target[['label_id','unit','u','site','site_year','block','year','ez','head','observed','horizon','mat_c','period']].rename(columns={'ez':'target_zone'}).copy()
    for model,p in [('T',base),('nonlinear_T',physical),('residual_RF',forest),('residual_MLP',neural),('thermal_transformer',nn),('support_blend',blend)]:out[model]=row_predictions(target,ids,p)
    distance=temperature_distance(unique_source.mat_c.to_numpy(),out.mat_c.to_numpy())
    out['signed_mat_distance_c']=distance['signed'];out['outside_mat_range_c']=distance['outside']
    out.to_csv(folder/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    with (folder/'fits.pkl').open('wb') as f:pickle.dump(dict(models=fits,rf_choice=rf_choice,mlp_choice=mlp_choice),f)
    metadata=dict(signature=sig,source_labels=len(source),inner_fit_labels=len(train),inner_validation_labels=len(tune),
        source_mat_range_c=[float(unique_source.mat_c.min()),float(unique_source.mat_c.max())],source_mat_median_c=float(unique_source.mat_c.median()),
        selections=dict(alpha=best,rf=rf_choice,mlp=mlp_choice,transformer_updates=steps),
        selection_trials=dict(nonlinear_T=trials,RF=rf_trials,MLP=mlp_trials,thermal_transformer=nn_trials),
        teachers=dict(inner=receipts,complete_source=full_receipts),correction_support=dict(RF=fits['RF']['supported'],MLP=fits['MLP']['supported']),
        support_geometry=geometry,elapsed_seconds=time.monotonic()-started,
        outputs={p.name:sha(p) for p in folder.iterdir() if p.name!='complete.json'})
    if signature(data,case)!=sig:raise ValueError('Fitting dependencies changed during case')
    (folder/'complete.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(case=name,seconds=round(metadata['elapsed_seconds'],1),selected_alpha=best['alpha'],neural_steps=steps)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--fold',type=int);args=p.parse_args()
    data,reference,edges=dataset();cases=freeze(data,reference,edges)
    if args.fold is not None:cases=[c for c in cases if c['fold']==args.fold]
    if args.limit:cases=cases[:args.limit]
    for case in cases:run_case(data,case)
    assert all(sha(path)==value for path,value in data.protected.items())
    print(json.dumps(dict(completed_requested_cases=len(cases),protected_inputs_unchanged=True)),flush=True)


if __name__=='__main__':main()
