"""Conditional thermal-demand updating and matched local neural corrections."""
import json
import pickle
import numpy as np
import pandas as pd
import torch
from .data import OUT,load
from .core import crossing,thermal_rate,valid_prediction,sha
from .models import predict
from .adaptation import selected_history,estimate_kappa,local_offset,BUDGETS
from .experiments import row_predictions
from .neural import ThermalTransformer,predict_network,DEVICE,NEURAL_FOLDER
from .analysis import site_errors,block_interval
from .provenance import code_signature,validate_receipt


def prior_log_demand(data,model,ids):
    z=(data.context[np.asarray(ids,int)]-model['mean'])/model['scale']
    return np.clip(z@model['coeff'].T+model['intercept'],0,np.log(10000))


def contextual_local_dates(data,model,history,target,budget,kappa):
    h=selected_history(history,budget).copy()
    if len(h):
        if (h.observed.to_numpy()>=data.train_lengths[h.u]).any():raise ValueError('Local event outside historical weather prefix')
        progress=thermal_rate(data.raw[h.u]).cumsum(1)
        demand=progress[np.arange(len(h)),h.observed.to_numpy(int)]
        prior=prior_log_demand(data,model,h.u)[np.arange(len(h)),h['head'].to_numpy(int)]
        h['residual']=np.log(np.maximum(demand,1e-8))-prior
        yearly=h.groupby(['site','head','year']).residual.mean().reset_index()
        g=yearly.groupby(['site','head']).residual.agg(['median','count'])
        keys=pd.MultiIndex.from_frame(target[['site','head']]);a=g.reindex(keys)
        count=a['count'].fillna(0).to_numpy();k=np.asarray(kappa)[target['head']]
        strength=np.divide(count,count+k,out=np.zeros(len(count)),where=count+k>0)
        offset=np.nan_to_num(strength*a['median'].to_numpy(),nan=0.)
    else:offset=np.zeros(len(target))
    prior=prior_log_demand(data,model,target.u)[np.arange(len(target)),target['head'].to_numpy(int)]
    demand=np.exp(np.clip(prior+offset,0,np.log(10000)))
    progress=thermal_rate(data.raw[target.u]).cumsum(1)
    return crossing(progress,demand,data.lengths[target.u])


def run():
    data=load();folder=OUT/'paired_local_extensions';folder.mkdir(exist_ok=True)
    design=dict(method='conditional thermal requirements plus source-variance-informed local log-demand correction',
        formula='D_new=exp(log(D_context)+n/(n+kappa)*median(yearly mean(log(D_observed)-log(D_context))))',
        kappa='source-only ratio of within-site to between-site annual log-demand residual variance; clipped [0.1,30]',
        budgets=BUDGETS,historical_end=2010,source_cases='coverage/80 sites/draw17; five fixed folds',
        neural_comparison='same native history/year budgets and source-only date-shrinkage as existing local controls',
        scope='post-exploration method extension; retrospective evaluation; no prospective novelty/physiology claim')
    dp=folder/'design.json'
    if dp.exists() and json.loads(dp.read_text())!=design:raise ValueError('Conditional local design changed')
    dp.write_text(json.dumps(design,indent=2)+'\n')
    parents={}
    for k in range(5):
        for kind in ['cases',NEURAL_FOLDER]:
            p=OUT/kind/f'f{k}_coverage_n80_d17/complete.json';parents[str(p)]=sha(p)
    parents[str(OUT/'local_adaptation/verification.json')]=sha(OUT/'local_adaptation/verification.json')
    signature=dict(code=code_signature('local_extensions.py','adaptation.py','neural.py','analysis.py'),
                   inputs=data.protected,parents=parents,design=design)
    if (folder/'verification.json').exists():
        validate_receipt(json.loads((folder/'verification.json').read_text()),signature,folder);return
    original=pd.read_csv(OUT/'local_adaptation/predictions.csv.gz')
    source_kappa=pd.read_csv(OUT/'local_adaptation/source_shrinkage.csv')
    eligibility=data.labels[['label_id','pre2011_prefix_eligible']]
    outputs=[];settings=[]
    for fold in range(5):
        name=f'f{fold}_coverage_n80_d17';base=OUT/'cases'/name
        with (base/'fits.pkl').open('rb') as f:fits=pickle.load(f)['models']
        primary=pd.read_csv(base/'predictions.csv.gz').merge(eligibility,on='label_id',validate='one_to_one')
        future=original[original.fold.eq(fold)&original.budget.eq(0)].copy()
        sites=set(future.site)
        h=primary[primary['case'].eq('historical_held')&primary.pre2011_prefix_eligible&primary.site.isin(sites)].copy()
        h['horizon']=data.train_lengths[h.u]
        members=pd.read_csv(OUT/'memberships'/f'{name}.csv.gz')
        source=data.labels[data.labels.unit.isin(members.unit)&data.labels.pre2011_prefix_eligible]
        c=thermal_rate(data.raw[source.u]).cumsum(1)
        demand=c[np.arange(len(source)),source.observed.to_numpy(int)]
        log_prior=prior_log_demand(data,fits['context_T'],source.u)[np.arange(len(source)),source['head'].to_numpy(int)]
        kappa=estimate_kappa(source,np.log(demand)-log_prior)
        date_kappa=source_kappa[source_kappa.fold.eq(fold)].sort_values('head').date_kappa.to_numpy()
        for head,value in enumerate(kappa):settings.append(dict(fold=fold,head=head,log_demand_kappa=float(value)))
        models={}
        neural=pd.read_csv(OUT/NEURAL_FOLDER/name/'predictions.csv.gz').set_index('label_id')
        hi=h.u.unique()
        for kind,context in [('thermal_transformer',False),('context_thermal_transformer',True)]:
            checkpoint=torch.load(OUT/NEURAL_FOLDER/name/f'{kind}.pt',map_location='cpu',weights_only=False)
            model=ThermalTransformer(checkpoint['initial'],len(checkpoint['norm']['mean'])).to(DEVICE)
            model.load_state_dict(checkpoint['state'])
            hp=predict_network(model,data,hi,checkpoint['norm'],context,train=True)
            h[kind]=row_predictions(h,hi,hp)
            future[kind]=neural.loc[future.label_id,kind].to_numpy()
        for budget in BUDGETS:
            out=future[['label_id','unit','u','site','site_year','block','year','ez','head','observed','horizon','fold']].copy()
            out['budget']=budget
            out['conditional_local_T']=contextual_local_dates(data,fits['context_T'],h,future,budget,kappa)
            for kind in ['thermal_transformer','context_thermal_transformer']:
                out[kind]=local_offset(h,future,kind,budget,date_kappa)
            outputs.append(out)
        zero=outputs[-4]['conditional_local_T'].to_numpy()
        reference=future.context_T.to_numpy()
        if not np.allclose(zero,reference,atol=1e-6,equal_nan=True):raise ValueError('Zero-history conditional model differs from its source prior')
    f=pd.concat(outputs,ignore_index=True);f.to_csv(folder/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(settings).to_csv(folder/'source_shrinkage.csv',index=False)
    rows=[];stages=[];paired=[]
    combined=original.merge(f[['label_id','budget','conditional_local_T','thermal_transformer','context_thermal_transformer']],
                            on=['label_id','budget'],validate='one_to_one')
    models=['T','context_T','phase_T','residual_RF','residual_MLP','local_thermal_T','conditional_local_T',
            'thermal_transformer','context_thermal_transformer']
    for budget,q in combined.groupby('budget'):
        for model in models:
            err=np.where(valid_prediction(q,q[model]),abs(q[model]-q.observed),366.)
            s=site_errors(q,err);lo,hi=block_interval(s)
            rows.append(dict(budget=int(budget),model=model,mae=float(s.groupby('head').error.mean().mean()),ci_low=lo,ci_high=hi,
                             sites=q.site.nunique(),labels=len(q),completion=float(valid_prediction(q,q[model]).mean())))
            for head,hg in s.groupby('head'):stages.append(dict(budget=int(budget),model=model,head=int(head),mae=float(hg.error.mean())))
            if model=='conditional_local_T':
                for control in ['local_thermal_T','residual_MLP','context_thermal_transformer']:
                    other=np.where(valid_prediction(q,q[control]),abs(q[control]-q.observed),366.)
                    diff=site_errors(q,err-other);dlo,dhi=block_interval(diff)
                    paired.append(dict(budget=int(budget),control=control,difference=float(diff.groupby('head').error.mean().mean()),ci_low=dlo,ci_high=dhi))
    pd.DataFrame(rows).to_csv(folder/'summary.csv',index=False)
    pd.DataFrame(stages).to_csv(folder/'stage_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(folder/'paired_conditional_contrasts.csv',index=False)
    verification=dict(signature=signature,complete=True,zero_history_reproduces_context_prior=True,
        historical_predictions_weather_limited=True,local_information_aggregated_by_head_year=True,
        outputs={p.name:sha(p) for p in folder.iterdir() if p.name!='verification.json'})
    (folder/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(pd.DataFrame(rows)[['budget','model','mae']].round(3).to_string(index=False))


if __name__=='__main__':run()
