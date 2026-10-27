"""Registered source-boundary sensitivity; original experiments remain intact."""
import json
import pickle
import time
import numpy as np
import pandas as pd
from .data import OUT,load
from .core import metric,sha,valid_prediction
from .models import fit_thermal,predict
from .experiments import row_predictions
from .analysis import site_errors,block_interval
from .provenance import code_signature,validate_receipt

ALPHAS=[1/64,1/32,1/16,1/8,1/4,.5,.75,1.,1.5]


def run():
    data=load();folder=OUT/'response_followup';folder.mkdir(exist_ok=True)
    design=dict(reason='Initial source-validation diagnostics selected the lower alpha boundary; expanded grid prespecified before corrected year-matched scoring',
        alpha_grid=ALPHAS,selection='source spatial-block validation only; refit complete selected source',
        population='same 80 frozen source memberships and original target labels',
        near_zero_interpretation='weak temperature dependence above the fixed base, not identified physiological adaptation',
        status='source-diagnosed follow-up after preceding exploratory results; no untouched prospective test claim')
    file=folder/'design.json'
    if file.exists() and json.loads(file.read_text())!=design:raise ValueError('Follow-up design changed')
    file.write_text(json.dumps(design,indent=2)+'\n')
    cases=json.loads((OUT/'cases.json').read_text())['cases'];site_tables=[];settings=[]
    code_snapshot=code_signature('response_followup.py','analysis.py')
    for case in cases:
        name=case['case'];dest=folder/name;dest.mkdir(exist_ok=True)
        signature=dict(code=code_snapshot,inputs=data.protected,case=case,design=design,
            original_case_sha256=sha(OUT/'cases'/name/'complete.json'),
            source_membership_sha256=sha(OUT/'memberships'/f'{name}.csv.gz'))
        if (dest/'complete.json').exists():
            validate_receipt(json.loads((dest/'complete.json').read_text()),signature,dest);continue
        start=time.monotonic();sample=pd.read_csv(OUT/'memberships'/f'{name}.csv.gz')
        source=data.labels[data.labels.unit.isin(sample.unit)&data.labels.pre2011_prefix_eligible]
        inner=sample.set_index('unit').inner
        train=source[source.unit.map(inner).ne(2)];tune=source[source.unit.map(inner).eq(2)]
        train_base=fit_thermal(data,train);final_base=fit_thermal(data,source)
        original=pd.read_csv(OUT/'cases'/name/'predictions.csv.gz')
        target=data.labels.set_index('label_id').loc[original.label_id].reset_index()
        ids=target.u.unique();ti=tune.u.unique();fits={};trials={};export=original[['label_id','unit','u','site','site_year','block','year','ez','head','observed','case','horizon','T','calendar']].copy()
        for kind,phase in [('relaxed_T',False),('relaxed_phase_T',True)]:
            values=[]
            for alpha in ALPHAS:
                model=fit_thermal(data,train,alpha,phase,train_base if phase else None)
                p=predict(model,data,ti,train=True)
                values.append(dict(alpha=alpha,mae=metric(tune,row_predictions(tune,ti,p),data.train_lengths[tune.u])['penalized_mae']))
            best=min(values,key=lambda x:(x['mae'],abs(x['alpha']-1)))
            fits[kind]=fit_thermal(data,source,best['alpha'],phase,final_base if phase else None)
            trials[kind]=values
            export[kind]=row_predictions(target,ids,predict(fits[kind],data,ids))
        export.to_csv(dest/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        with (dest/'fits.pkl').open('wb') as f:pickle.dump(fits,f)
        receipt=dict(case=case,signature=signature,selection_trials=trials,selected={k:fits[k]['alpha'] for k in fits},
            original_case_sha256=sha(OUT/'cases'/name/'complete.json'),source_membership_sha256=sha(OUT/'memberships'/f'{name}.csv.gz'),
            outputs={p.name:sha(p) for p in dest.iterdir() if p.name!='complete.json'},elapsed_seconds=time.monotonic()-start)
        (dest/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(dict(case=name,selected=receipt['selected'],seconds=round(receipt['elapsed_seconds'],1))),flush=True)
    for case in cases:
        dest=folder/case['case'];receipt=json.loads((dest/'complete.json').read_text())
        f=pd.read_csv(dest/'predictions.csv.gz')
        for model,alpha in receipt['selected'].items():settings.append(dict(case=case['case'],model=model,alpha=alpha,**{k:case[k] for k in ['fold','recipe','n_sites','draw_seed','source_ez']}))
        for period,q in f.groupby('case'):
            for model in ['T','calendar','relaxed_T','relaxed_phase_T']:
                error=np.where(valid_prediction(q,q[model]),abs(q[model]-q.observed),366.)
                s=site_errors(q,error).assign(period=period,model=model,fold=case['fold'],recipe=case['recipe'],
                    n_sites=case['n_sites'],draw_seed=case['draw_seed'],source_ez=-1 if case['source_ez'] is None else case['source_ez'])
                s['difference_vs_T']=site_errors(q,error-np.where(valid_prediction(q,q['T']),abs(q['T']-q.observed),366.)).error
                site_tables.append(s)
    allsites=pd.concat(site_tables,ignore_index=True);allsites.to_csv(folder/'site_errors.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    settings=pd.DataFrame(settings);settings.to_csv(folder/'selected_alpha.csv',index=False)
    summary=[];stages=[]
    for keys,q in allsites.groupby(['period','recipe','n_sites','source_ez','model']):
        meta=dict(zip(['period','recipe','n_sites','source_ez','model'],keys))
        a=q.groupby(['head','site','block'])[['error','difference_vs_T']].mean().reset_index()
        lo,hi=block_interval(a)
        summary.append(dict(**meta,mae=float(a.groupby('head').error.mean().mean()),ci_low=lo,ci_high=hi,
            difference_vs_T=float(a.groupby('head').difference_vs_T.mean().mean())))
        for head,s in a.groupby('head'):stages.append(dict(**meta,head=int(head),mae=float(s.error.mean())))
    pd.DataFrame(summary).to_csv(folder/'summary.csv',index=False);pd.DataFrame(stages).to_csv(folder/'stage_summary.csv',index=False)
    verification=dict(complete=True,cases=len(cases),source_only_selection=True,
        protected_inputs_unchanged=all(sha(p)==digest for p,digest in data.protected.items()),
        alpha_counts=settings.groupby(['model','alpha']).size().to_dict().__str__())
    (folder/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(pd.DataFrame(summary).query('source_ez == -1 and period == "later_held"').round(3).to_string(index=False))


if __name__=='__main__':run()
