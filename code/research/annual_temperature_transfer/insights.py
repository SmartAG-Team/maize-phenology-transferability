"""Source-backed benchmark, error anatomy and historical local calibration."""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from research.phenology_exploration.analysis import site_errors, block_interval
from research.phenology_exploration.adaptation import local_offset, BUDGETS
from research.phenology_exploration.core import valid_prediction, sha
from research.phenology_exploration.data import load
from research.phenology_exploration.models import predict
from research.phenology_exploration.experiments import row_predictions
from .climate import ROOT, OUT
from .analysis import MODELS

FOLDER=OUT/'insights'
PRIOR=ROOT/'data/results/phenology_exploration_20261023/year_matched'
ALIASES={'relaxed_T':'nonlinear_T','relaxed_phase_T':'phase_nonlinear_T','climate_blend':'support_blend'}


def correction_components(frame,reference,prediction):
    if frame.label_id.duplicated().any():
        raise ValueError('Duplicate native labels in correction comparison')
    a=np.where(valid_prediction(frame,frame[reference]),abs(frame[reference]-frame.observed),366.)
    b=np.where(valid_prediction(frame,frame[prediction]),abs(frame[prediction]-frame.observed),366.)
    gain=a-b
    q=frame[['head','site','site_year','block']].copy()
    for name,values in [('benefit',np.maximum(gain,0)),('harm',np.maximum(-gain,0)),('net_gain',gain),
                        ('improved_fraction',(gain>0).astype(float)),('harmed_fraction',(gain<0).astype(float))]:
        q[name]=values
    columns=['benefit','harm','net_gain','improved_fraction','harmed_fraction']
    return q.groupby(['head','site','block','site_year'])[columns].mean().groupby(['head','site','block']).mean().reset_index()


def historical_local_update(history,target,budget,kappa):
    if budget not in BUDGETS:
        raise ValueError('Unsupported local budget')
    if not target.year.gt(2010).all():
        raise ValueError('Local evaluation requires later targets')
    h=history[history.year.le(2010)].copy()
    if (h.observed>=h.horizon).any():
        raise ValueError('Local observation outside historical weather prefix')
    if set(h.label_id)&set(target.label_id):
        raise ValueError('Local history and target overlap')
    return local_offset(h,target,'nonlinear_T',budget,kappa)


def summarize(frame,column='error'):
    q=frame[['head','site','block',column]].rename(columns={column:'error'})
    lo,hi=block_interval(q)
    return dict(value=float(q.groupby('head').error.mean().mean()),ci_low=float(lo),ci_high=float(hi))


def model_sites(frame,model):
    ok=valid_prediction(frame,frame[model]);error=np.where(ok,abs(frame[model]-frame.observed),366.)
    q=site_errors(frame,error)
    q['completion']=site_errors(frame,ok.astype(float)).error
    q['failure14']=site_errors(frame,(error>14).astype(float)).error
    return q


def run():
    FOLDER.mkdir(parents=True,exist_ok=True)
    inputs={}
    def record(path):
        path=Path(path);inputs[str(path)]=sha(path)
    def fitted(folder):
        record(folder/'complete.json');r=json.loads((folder/'complete.json').read_text())
        for name,digest in r['outputs'].items():
            assert sha(folder/name)==digest;record(folder/name)
        return pd.read_csv(folder/'predictions.csv.gz')
    cases_path=PRIOR/'cases.json';record(cases_path)
    cases=[c for c in json.loads(cases_path.read_text())['cases'] if c['n_sites']==80 and c['source_ez'] is None]
    assert len(cases)==30
    sites=[];components=[];identities={}
    extra=['calendar','context_T','phase_nonlinear_T','context_thermal_transformer']
    for case in cases:
        name=case['case'];f=fitted(PRIOR/'cases'/name)
        for folder,columns in [('response_followup',['relaxed_T','relaxed_phase_T']),
                               ('paired_causal_neural',['thermal_transformer','context_thermal_transformer']),
                               ('support_blend',['climate_blend'])]:
            q=fitted(PRIOR/folder/name)[['label_id']+columns]
            f=f.merge(q,on='label_id',how='outer',validate='one_to_one',indicator=True)
            assert f['_merge'].eq('both').all();f=f.drop(columns='_merge')
        # The earlier restricted-grid nonlinear_T is a different specification.
        f=f.drop(columns='nonlinear_T').rename(columns=ALIASES)
        for period,q in f.groupby('case'):
            identity=tuple(sorted(q.label_id));key=(period,case['fold'])
            if key in identities:assert identities[key]==identity
            else:identities[key]=identity
            info=dict(study='prior_pooled_80',period=period,recipe=case['recipe'],draw=case['draw_seed'],fold=case['fold'])
            for model in MODELS+extra:
                sites.append(model_sites(q,model).assign(model=model,**info))
            if period=='later_held':
                for model in ['residual_RF','residual_MLP','thermal_transformer','support_blend']:
                    components.append(correction_components(q,'nonlinear_T',model).assign(model=model,**info))
    current=OUT/'site_transfer_errors.csv.gz';record(current)
    receipt=json.loads((OUT/'analysis_verification.json').read_text());assert sha(current)==receipt['files'][current.name]
    q=pd.read_csv(current);q=q[q.source_zone.eq(-1)].copy()
    for key,g in q.groupby(['period','fold']):
        # Site/head support is checked below; raw target IDs were verified in verify.py.
        assert g.groupby(['model','draw']).size().nunique()==1
    columns=['head','site','block','error','completion','failure14','period','model','draw','fold']
    sites.append(q[columns].assign(study='annual_pooled_40',recipe='random'))
    native=pd.concat(sites,ignore_index=True)
    keys=['study','period','recipe','model','head','site','block']
    assert native.groupby(keys).size().eq(3).all()
    averaged=native.groupby(keys)[['error','completion','failure14']].mean().reset_index()
    averaged.to_csv(FOLDER/'benchmark_sites.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    benchmark=[];stages=[]
    for key,g in averaged.groupby(['study','period','recipe','model']):
        info=dict(zip(['study','period','recipe','model'],key));s=summarize(g)
        benchmark.append(dict(**info,mae=s['value'],ci_low=s['ci_low'],ci_high=s['ci_high'],
                              completion=float(g.groupby('head').completion.mean().mean()),
                              failure14=float(g.groupby('head').failure14.mean().mean()),sites=g.site.nunique()))
        for head,h in g.groupby('head'):
            v=summarize(h);stages.append(dict(**info,head=int(head),mae=v['value'],ci_low=v['ci_low'],ci_high=v['ci_high']))
    pd.DataFrame(benchmark).to_csv(FOLDER/'benchmark.csv',index=False)
    pd.DataFrame(stages).to_csv(FOLDER/'stage_performance.csv',index=False)
    contrasts=[]
    for (study,period,recipe),g in averaged.groupby(['study','period','recipe']):
        pairs=[('nonlinear_T','T'),('residual_MLP','nonlinear_T'),('thermal_transformer','nonlinear_T')]
        if study=='prior_pooled_80':pairs += [('nonlinear_T','calendar'),('context_thermal_transformer','thermal_transformer'),('nonlinear_T','phase_nonlinear_T')]
        for left,right in pairs:
            a=g[g.model.eq(left)].set_index(['head','site','block']).error
            b=g[g.model.eq(right)].set_index(['head','site','block']).error
            assert a.index.equals(b.index)
            difference=(a-b).reset_index()
            for head,h in [('all',difference)]+[(str(k),v) for k,v in difference.groupby('head')]:
                result=summarize(h);contrasts.append(dict(study=study,period=period,recipe=recipe,left=left,right=right,head=head,
                    difference=result['value'],ci_low=result['ci_low'],ci_high=result['ci_high']))
    # A source-recipe comparison uses the same target site/head population.
    old=averaged[averaged.study.eq('prior_pooled_80')]
    for (period,model),g in old.groupby(['period','model']):
        a=g[g.recipe.eq('coverage')].set_index(['head','site','block']).error
        b=g[g.recipe.eq('random')].set_index(['head','site','block']).error
        assert a.index.equals(b.index)
        v=summarize((a-b).reset_index())
        contrasts.append(dict(study='prior_pooled_80',period=period,recipe='coverage_minus_random',left=model,right=model,
                              head='all',difference=v['value'],ci_low=v['ci_low'],ci_high=v['ci_high']))
    pd.DataFrame(contrasts).to_csv(FOLDER/'contrasts.csv',index=False)
    component=pd.concat(components,ignore_index=True)
    component=component.groupby(['recipe','model','head','site','block'])[['benefit','harm','net_gain','improved_fraction','harmed_fraction']].mean().reset_index()
    assert np.allclose(component.benefit-component.harm,component.net_gain)
    component.to_csv(FOLDER/'correction_sites.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    rows=[]
    for (recipe,model),g in component.groupby(['recipe','model']):
        row=dict(recipe=recipe,model=model)
        for metric in ['benefit','harm','net_gain','improved_fraction','harmed_fraction']:
            v=summarize(g,metric);row.update({metric:v['value'],metric+'_ci_low':v['ci_low'],metric+'_ci_high':v['ci_high']})
        rows.append(row)
    pd.DataFrame(rows).to_csv(FOLDER/'correction_summary.csv',index=False)
    print('Benchmark and native-error anatomy qualified',flush=True)
    # Extend the exact existing local-history cohort with the strong nonlinear baseline.
    data=load();parents={};outputs=[];history_rows=[]
    for name in ['local_adaptation/predictions.csv.gz','local_adaptation/source_shrinkage.csv','paired_local_extensions/predictions.csv.gz']:
        record(PRIOR/name)
    original=pd.read_csv(PRIOR/'local_adaptation/predictions.csv.gz')
    extension=pd.read_csv(PRIOR/'paired_local_extensions/predictions.csv.gz')
    local=original.merge(extension[['label_id','budget','thermal_transformer','context_thermal_transformer']],on=['label_id','budget'],validate='one_to_one')
    kappas=pd.read_csv(PRIOR/'local_adaptation/source_shrinkage.csv')
    eligible=data.labels[['label_id','pre2011_prefix_eligible']]
    for fold in range(5):
        name=f'f{fold}_coverage_n80_d17'
        source=PRIOR/'response_followup'/name;record(source/'fits.pkl')
        with (source/'fits.pkl').open('rb') as stream:fit=pickle.load(stream)['relaxed_T']
        primary=fitted(PRIOR/'cases'/name).merge(eligible,on='label_id',validate='one_to_one')
        target=local[local.fold.eq(fold)&local.budget.eq(0)].copy()
        reference=fitted(source).set_index('label_id')
        target['nonlinear_T']=reference.loc[target.label_id,'relaxed_T'].to_numpy()
        history=primary[primary['case'].eq('historical_held')&primary.pre2011_prefix_eligible&primary.site.isin(target.site)].copy()
        history['horizon']=data.train_lengths[history.u]
        ids=history.u.unique();values=predict(fit,data,ids,train=True)
        history['nonlinear_T']=row_predictions(history,ids,values)
        history_rows.append(history[['label_id','unit','u','site','site_year','block','year','head','observed','horizon','nonlinear_T']].assign(fold=fold))
        k=kappas[kappas.fold.eq(fold)].sort_values('head').date_kappa.to_numpy();assert len(k)==7
        for budget in BUDGETS:
            q=local[local.fold.eq(fold)&local.budget.eq(budget)].copy()
            q['nonlinear_T']=historical_local_update(history,target,budget,k)
            assert np.array_equal(q.label_id,target.label_id)
            outputs.append(q)
        zero=outputs[-4].nonlinear_T.to_numpy()
        assert np.array_equal(zero,target.nonlinear_T.to_numpy(),equal_nan=True)
    combined=pd.concat(outputs,ignore_index=True);assert not combined.duplicated(['label_id','budget']).any()
    combined.to_csv(FOLDER/'local_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.concat(history_rows,ignore_index=True).to_csv(FOLDER/'local_history.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    local_sites=[];rows=[];gains=[]
    for budget,g in combined.groupby('budget'):
        assert len(g)==49462 and g.site.nunique()==822 and g.year.gt(2010).all()
        for model in ['T','nonlinear_T','context_T','residual_RF','residual_MLP','thermal_transformer','context_thermal_transformer']:
            s=model_sites(g,model).assign(budget=int(budget),model=model);local_sites.append(s)
            v=summarize(s);rows.append(dict(budget=int(budget),model=model,mae=v['value'],ci_low=v['ci_low'],ci_high=v['ci_high'],
                                           completion=float(s.groupby('head').completion.mean().mean()),sites=822,labels=49462))
    local_sites=pd.concat(local_sites,ignore_index=True)
    local_sites.to_csv(FOLDER/'local_sites.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    for model,g in local_sites.groupby('model'):
        zero=g[g.budget.eq(0)].set_index(['head','site','block']).error
        for budget in BUDGETS[1:]:
            adapted=g[g.budget.eq(budget)].set_index(['head','site','block']).error;assert zero.index.equals(adapted.index)
            v=summarize((adapted-zero).reset_index());gains.append(dict(model=model,budget=budget,comparison='adapted_minus_zero',
                difference=v['value'],ci_low=v['ci_low'],ci_high=v['ci_high']))
    for budget,g in local_sites.groupby('budget'):
        base=g[g.model.eq('nonlinear_T')].set_index(['head','site','block']).error
        for model in ['T','residual_MLP','thermal_transformer']:
            other=g[g.model.eq(model)].set_index(['head','site','block']).error;assert base.index.equals(other.index)
            v=summarize((other-base).reset_index());gains.append(dict(model=model,budget=int(budget),comparison='model_minus_nonlinear_T',
                difference=v['value'],ci_low=v['ci_low'],ci_high=v['ci_high']))
    pd.DataFrame(rows).to_csv(FOLDER/'local_summary.csv',index=False)
    pd.DataFrame(gains).to_csv(FOLDER/'local_contrasts.csv',index=False)
    assert all(sha(p)==digest for p,digest in data.protected.items())
    receipt=dict(complete=True,prior_cases=30,current_pooled_cases=15,local_source_cases=5,local_targets=49462,local_sites=822,
        input_sha256=inputs,protected_inputs_unchanged=True,code_sha256={str(p):sha(p) for p in [Path(__file__),
            ROOT/'code/research/phenology_exploration/adaptation.py',ROOT/'code/research/phenology_exploration/models.py']},
        zero_history_matches_saved_nonlinear_prior=True,historical_predictions_use_train_prefix=True,
        local_date_kappa='exact source-T-derived date shrinkage shared with original local date-offset models',
        benefit_minus_harm_equals_net_gain=True,metric='native observations/site-year, years/site, sites/head, seven heads',
        intervals='2000 target spatial-block resamples after source-draw averaging; exploratory conditional intervals',
        output_sha256={p.name:sha(p) for p in FOLDER.iterdir() if p.is_file() and p.name not in {'verification.json','independent_verification.json'}})
    (FOLDER/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(pd.DataFrame(rows).query('model in ["T","nonlinear_T","residual_MLP","thermal_transformer"]')[['budget','model','mae']].round(3).to_string(index=False),flush=True)


if __name__=='__main__':run()
