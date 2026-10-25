"""Historical local calibration with subsequent, independently held years."""
import json
import pickle
import numpy as np
import pandas as pd
from .data import OUT,load
from .core import crossing,thermal_rate,sha,valid_prediction
from .models import predict,residual_prediction
from .experiments import row_predictions
from .analysis import site_errors,block_interval
from .provenance import code_signature,validate_receipt

BUDGETS=[0,1,3,5]
BASE_MODELS=['T','context_T','phase_T','residual_RF','residual_MLP']


def selected_history(history,budget):
    f=history[history.year.le(2010)]
    if budget==0:return f.iloc[:0]
    years=f[['site','year']].drop_duplicates().sort_values(['site','year'])
    selected=years.groupby('site').tail(budget)
    return f.merge(selected,on=['site','year'],validate='many_to_one')


def local_offset(history,target,model,budget,kappa):
    h=selected_history(history,budget).copy()
    h['residual']=h.observed-h[model]
    good=valid_prediction(h,h[model].to_numpy())
    yearly=h[good].groupby(['site','head','year']).residual.mean().reset_index()
    g=yearly.groupby(['site','head']).residual.agg(['median','count'])
    keys=pd.MultiIndex.from_frame(target[['site','head']]);q=g.reindex(keys)
    strength=q['count'].to_numpy()/(q['count'].to_numpy()+np.asarray(kappa)[target['head']])
    offset=np.nan_to_num(q['median'].to_numpy()*strength,nan=0.)
    return target[model].to_numpy()+offset


def estimate_kappa(frame,values):
    f=frame[['site','head','year']].copy();f['value']=np.asarray(values)
    f=f.groupby(['site','head','year']).value.mean().reset_index()
    result=[]
    for head in range(7):
        g=f[f['head'].eq(head)].groupby('site').value.agg(['mean','var','count'])
        g=g[g['count'].ge(2)&g['var'].notna()]
        if len(g)<3:result.append(30.);continue
        within=float(g['var'].mean());between=max(0.,float(g['mean'].var())-float((g['var']/g['count']).mean()))
        result.append(float(np.clip(within/max(between,1e-8),.1,30.)))
    return np.array(result)


def history_model_predictions(models,data,ids):
    output={k:predict(models[k],data,ids,train=True) for k in ['T','context_T','phase_T'] if k in models}
    return output


def run():
    data=load();design=dict(source_cases='coverage, 80 sites, source draw 17, five outer folds',
        local_year_budgets=BUDGETS,local_calibration_end=2010,evaluation_years=[2011,2025],
        common_site_requirement='at least five historical years and at least one later native outcome',
        strength='n/(n+kappa), kappa from source-only within/between-site variance, clipped to [0.1,30]',
        comparisons='native-stage date offsets for all five models; T thermal-threshold updating as a separate control',
        stage_missingness='no local update when the designated local years have no corresponding native head',
        native_replication='aggregate within site/head/year before updating; n counts distinct head-years',
        historical_prediction='recompute every calibration prediction using weather prefixes ending in2010',
        status='exploratory retrospective adaptation, not zero-shot validation')
    folder=OUT/'local_adaptation';folder.mkdir(exist_ok=True)
    path=folder/'design.json'
    if path.exists() and json.loads(path.read_text())!=design:raise ValueError('Local adaptation design changed')
    path.write_text(json.dumps(design,indent=2)+'\n')
    parents={str(OUT/'cases'/f'f{k}_coverage_n80_d17/complete.json'):sha(OUT/'cases'/f'f{k}_coverage_n80_d17/complete.json') for k in range(5)}
    signature=dict(code=code_signature('adaptation.py','analysis.py'),inputs=data.protected,parents=parents,design=design)
    previous=folder/'verification.json'
    if previous.exists():
        validate_receipt(json.loads(previous.read_text()),signature,folder)
        print('Validated completed local adaptation reused');return
    outputs=[];parameter_rows=[];source_receipts=[]
    eligibility=data.labels[['label_id','pre2011_prefix_eligible']]
    for fold in range(5):
        name=f'f{fold}_coverage_n80_d17';case=OUT/'cases'/name
        receipt=json.loads((case/'complete.json').read_text())
        pred=pd.read_csv(case/'predictions.csv.gz').merge(eligibility,on='label_id',validate='one_to_one')
        history=pred[pred['case'].eq('historical_held')&pred.pre2011_prefix_eligible].copy()
        later=pred[pred['case'].eq('later_held')].copy()
        counts=history[['site','year']].drop_duplicates().groupby('site').size()
        sites=set(counts[counts.ge(5)].index)&set(later.site)
        history=history[history.site.isin(sites)];later=later[later.site.isin(sites)].copy()
        with (case/'fits.pkl').open('rb') as f:fit=pickle.load(f)
        history_ids=history.u.unique();history_dates=history_model_predictions(fit['models'],data,history_ids)
        base=history_dates['T']
        history_dates['residual_RF']=base+fit['rf_choice']['strength']*residual_prediction(
            fit['models']['residual_RF'],data,history_ids,base,train=True)
        history_dates['residual_MLP']=base+fit['mlp_choice']['strength']*residual_prediction(
            fit['models']['residual_MLP'],data,history_ids,base,train=True)
        for model,values in history_dates.items():history[model]=row_predictions(history,history_ids,values)
        history['horizon']=data.train_lengths[history.u]
        membership=pd.read_csv(OUT/'memberships'/f'{name}.csv.gz')
        source=data.labels[data.labels.unit.isin(membership.unit)&data.labels.pre2011_prefix_eligible]
        assert not set(source.site)&sites
        ids=source.u.unique();source_pred=predict(fit['models']['T'],data,ids,train=True)
        error=source.observed.to_numpy()-row_predictions(source,ids,source_pred)
        date_kappa=estimate_kappa(source,error)
        raw=thermal_rate(data.raw[source.u]).cumsum(1)
        demand=raw[np.arange(len(source)),source.observed.to_numpy(int)]
        thermal_kappa=estimate_kappa(source,demand)
        for h in range(7):parameter_rows.append(dict(fold=fold,head=h,date_kappa=date_kappa[h],thermal_kappa=thermal_kappa[h]))
        hraw=thermal_rate(data.raw[history.u]).cumsum(1)
        history['thermal_demand']=hraw[np.arange(len(history)),history.observed.to_numpy(int)]
        ids=later.u.unique();lookup={u:i for i,u in enumerate(ids)}
        progress=thermal_rate(data.raw[ids]).cumsum(1)
        for budget in BUDGETS:
            out=later[['label_id','unit','u','site','site_year','block','year','ez','head','observed','horizon']].copy()
            out['fold']=fold;out['budget']=budget
            for model in BASE_MODELS:out[model]=local_offset(history,later,model,budget,date_kappa)
            h=selected_history(history,budget)
            yearly=h.groupby(['site','head','year']).thermal_demand.mean().reset_index()
            grouped=yearly.groupby(['site','head']).thermal_demand.agg(['median','count'])
            key=pd.MultiIndex.from_frame(later[['site','head']]);g=grouped.reindex(key)
            count=g['count'].fillna(0).to_numpy();heads=later['head'].to_numpy(int)
            initial=np.asarray(fit['models']['T']['thresholds'])[heads]
            strength=count/(count+thermal_kappa[heads])
            updated=initial+np.nan_to_num(strength*(g['median'].to_numpy()-initial),nan=0.)
            rows=later.u.map(lookup).to_numpy(int)
            out['local_thermal_T']=crossing(progress[rows],updated,data.lengths[later.u])
            out['local_head_observations']=count
            outputs.append(out)
        source_receipts.append(dict(fold=fold,eligible_sites=len(sites),later_labels=len(later),historical_labels=len(history),
                                    source_case_sha256=sha(case/'complete.json')))
    f=pd.concat(outputs,ignore_index=True)
    assert not f.duplicated(['label_id','budget']).any()
    f.to_csv(folder/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(parameter_rows).to_csv(folder/'source_shrinkage.csv',index=False)
    results=[];stage=[]
    for budget,q in f.groupby('budget'):
        for model in BASE_MODELS+['local_thermal_T']:
            error=np.where(valid_prediction(q,q[model]),abs(q[model]-q.observed),366.)
            s=site_errors(q,error);lo,hi=block_interval(s)
            results.append(dict(budget=int(budget),model=model,mae=float(s.groupby('head').error.mean().mean()),
                ci_low=lo,ci_high=hi,sites=q.site.nunique(),labels=len(q),completion=float(valid_prediction(q,q[model]).mean())))
            for head,h in s.groupby('head'):stage.append(dict(budget=int(budget),model=model,head=int(head),mae=float(h.error.mean())))
    pd.DataFrame(results).to_csv(folder/'summary.csv',index=False)
    pd.DataFrame(stage).to_csv(folder/'stage_summary.csv',index=False)
    verification=dict(complete=True,signature=signature,source_receipts=source_receipts,historical_only_calibration=True,
        common_later_targets_across_budgets=True,source_only_shrinkage=True,
        outputs={p.name:sha(p) for p in folder.iterdir() if p.name!='verification.json'})
    (folder/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(pd.DataFrame(results)[['budget','model','mae','sites']].round(3).to_string(index=False))


if __name__=='__main__':run()
