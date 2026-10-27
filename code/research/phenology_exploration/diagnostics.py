"""Integrate matched neural fits and distinguish date skill from temporal shifts."""
import json
import numpy as np
import pandas as pd
from .data import OUT
from .core import valid_prediction,macro,sha
from .analysis import site_errors,block_interval,common_resolved,recipe_contrast
from .neural import NEURAL_FOLDER

NEURAL=['thermal_transformer','context_thermal_transformer']


def paired_shifts(frame,models,min_history=5,min_later=3):
    f=frame.copy();keys=['head','site','block','site_year','year','case']
    for m in models:
        f[m+'_valid']=valid_prediction(f,f[m])
        f.loc[~f[m+'_valid'],m]=np.nan
    values=['observed']+models
    y=f.groupby(keys)[values].mean()
    valid=f.groupby(keys)[[m+'_valid' for m in models]].all()
    for m in models:y.loc[~valid[m+'_valid'],m]=np.nan
    y=y.reset_index();rows=[]
    for (head,site,block),q in y.groupby(['head','site','block']):
        historical=q[q['case'].eq('historical_held')];later=q[q['case'].eq('later_held')]
        if historical.year.nunique()<min_history or later.year.nunique()<min_later:continue
        row=dict(head=int(head),site=site,block=block,historical_years=historical.year.nunique(),later_years=later.year.nunique(),
                 observed_change=float(later.observed.mean()-historical.observed.mean()))
        for m in models:
            row[m]=float(later[m].mean()-historical[m].mean()) if historical[m].notna().all() and later[m].notna().all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run():
    folder=OUT/NEURAL_FOLDER;cases=[c for c in json.loads((OUT/'cases.json').read_text())['cases']
        if c['n_sites']==80 and c['source_ez'] is None]
    sites=[];changes=[];joint=[];metrics=[];order=[]
    models=['T','calendar','context_T','nonlinear_T','phase_T','residual_RF','residual_MLP','relaxed_T','relaxed_phase_T']+NEURAL
    for case in cases:
        name=case['case'];primary=OUT/'cases'/name/'predictions.csv.gz'
        neural=folder/name/'predictions.csv.gz';response=OUT/'response_followup'/name/'predictions.csv.gz'
        if not neural.exists() or not (folder/name/'complete.json').exists():raise ValueError(f'Incomplete neural case {name}')
        receipt=json.loads((folder/name/'complete.json').read_text())
        for file,digest in receipt['outputs'].items():
            if sha(folder/name/file)!=digest:raise ValueError('Neural output changed')
        f=pd.read_csv(primary)
        for path,columns in [(neural,NEURAL),(response,['relaxed_T','relaxed_phase_T'])]:
            a=pd.read_csv(path,usecols=['label_id']+columns)
            if set(a.label_id)!=set(f.label_id):raise ValueError('Neural/physical comparison targets differ')
            f=f.merge(a,on='label_id',validate='one_to_one')
        info=dict(recipe=case['recipe'],n_sites=80,draw_seed=case['draw_seed'],fold=case['fold'])
        for period,q in f.groupby('case'):
            common=common_resolved(q,models)
            for m in models:
                error=np.where(valid_prediction(q,q[m]),abs(q[m]-q.observed),366.)
                table=site_errors(q,error).assign(model=m,period=period,**info)
                table['difference_vs_T']=site_errors(q,error-np.where(valid_prediction(q,q['T']),abs(q['T']-q.observed),366.)).error
                sites.append(table)
                metrics.append(dict(model=m,period=period,common_mae=macro(q.loc[common],abs(q.loc[common,m]-q.loc[common,'observed'])),
                    common_labels=int(common.sum()),all_labels=len(q),completion=float(valid_prediction(q,q[m]).mean()),**info))
                p=q.assign(value=np.where(valid_prediction(q,q[m]),q[m],np.nan)).pivot(index='unit',columns='head',values='value').reindex(columns=range(7))
                v=p.to_numpy();pairs=np.isfinite(v[:,:-1])&np.isfinite(v[:,1:])
                order.append(dict(model=m,period=period,reversed_pairs=int(((v[:,:-1]>v[:,1:])&pairs).sum()),
                                  valid_pairs=int(pairs.sum()),**info))
        c=paired_shifts(f,models)
        for key,value in info.items():c[key]=value
        changes.append(c)
    site=pd.concat(sites,ignore_index=True);site.to_csv(folder/'integrated_site_errors.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    summary=[];stage=[]
    for (period,recipe,model),q in site.groupby(['period','recipe','model']):
        a=q.groupby(['head','site','block'])[['error','difference_vs_T']].mean().reset_index()
        lo,hi=block_interval(a);contrast=a.rename(columns={'error':'absolute_error','difference_vs_T':'error'})
        dlo,dhi=block_interval(contrast)
        draws=q.groupby(['draw_seed','head']).error.mean().groupby(level='draw_seed').mean()
        summary.append(dict(period=period,recipe=recipe,model=model,mae=float(a.groupby('head').error.mean().mean()),
            ci_low=lo,ci_high=hi,difference_vs_T=float(a.groupby('head').difference_vs_T.mean().mean()),
            difference_ci_low=dlo,difference_ci_high=dhi,source_draw_sd=float(draws.std())))
        for head,h in a.groupby('head'):
            stage.append(dict(period=period,recipe=recipe,model=model,head=int(head),mae=float(h.error.mean())))
    pd.DataFrame(summary).to_csv(folder/'integrated_summary.csv',index=False)
    pd.DataFrame(stage).to_csv(folder/'integrated_stage_summary.csv',index=False)
    pd.DataFrame(metrics).to_csv(folder/'common_resolved_metrics.csv',index=False)
    pd.DataFrame(order).to_csv(folder/'stage_order_audit.csv',index=False)
    c=pd.concat(changes,ignore_index=True);c.to_csv(folder/'paired_temporal_cohorts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    shift=[]
    for (recipe,draw),q in c.groupby(['recipe','draw_seed']):
        common=np.isfinite(q[models]).all(1)
        for m in models:
            x=q.loc[common,['head','site','block']].copy();x['error']=abs(q.loc[common,m]-q.loc[common,'observed_change'])
            shift.append(x.assign(model=m,recipe=recipe,draw_seed=draw))
    shift=pd.concat(shift,ignore_index=True);shift.to_csv(folder/'temporal_shift_site_errors.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    shifts=[]
    for (recipe,model),q in shift.groupby(['recipe','model']):
        a=q.groupby(['head','site','block']).error.mean().reset_index();lo,hi=block_interval(a)
        shifts.append(dict(recipe=recipe,model=model,mae_change_days=float(a.groupby('head').error.mean().mean()),
                           ci_low=lo,ci_high=hi,sites=a.site.nunique(),native_site_head_pairs=len(a)))
    pd.DataFrame(shifts).to_csv(folder/'temporal_shift_summary.csv',index=False)
    contrasts=[]
    for (period,model),q in site.groupby(['period','model']):
        contrasts.append(dict(period=period,model=model,**recipe_contrast(q[q.recipe.eq('random')],q[q.recipe.eq('coverage')])))
    pd.DataFrame(contrasts).to_csv(folder/'paired_recipe_contrasts.csv',index=False)
    (folder/'verification.json').write_text(json.dumps(dict(complete=True,cases=len(cases),
        target_identity_matched=True,native_horizon_validity=True,shared_common_resolved_population=True,
        temporal_cohorts='same native head/site, at least five historical and three later years; original units averaged within year',
        temporal_interpretation='retrospective duration shifts; no causal warming attribution',
        files={p.name:sha(p) for p in folder.glob('*.csv*')}),indent=2)+'\n')
    print(pd.DataFrame(summary).query('period == "later_held"').round(3).to_string(index=False))
    print('Paired temporal changes:');print(pd.DataFrame(shifts).round(3).to_string(index=False))


if __name__=='__main__':run()
