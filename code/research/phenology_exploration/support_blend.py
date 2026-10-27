"""Source-geometry blending with a flexible thermal fallback, without target tuning."""
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from .data import OUT,load
from .core import valid_prediction,sha,macro
from .analysis import site_errors,block_interval
from .provenance import code_signature,validate_receipt


def support_weight(source,target):
    source=np.asarray(source,float);target=np.asarray(target,float)
    if len(source)<2 or not np.isfinite(source).all() or not np.isfinite(target).all():raise ValueError('Finite source climate support required')
    mean=source.mean(0);scale=source.std(0).clip(.1);z=(source-mean)/scale
    model=NearestNeighbors(n_neighbors=2).fit(z)
    loo=model.kneighbors(z)[0][:,1]
    radius=max(float(np.quantile(loo,.95)),1e-6)
    distance=model.kneighbors((target-mean)/scale,n_neighbors=1)[0].ravel()
    weight=np.minimum(1.,radius/np.maximum(distance,1e-12))**2
    return weight,dict(mean=mean.tolist(),scale=scale.tolist(),radius=radius,source_sites=len(source),
                       source_leave_one_site_distances=loo.tolist())


def blend_dates(neural,thermal,weight,horizon):
    a=np.asarray(neural,float);b=np.asarray(thermal,float);w=np.asarray(weight,float)
    va=np.isfinite(a)&(a>=0)&(a<horizon);vb=np.isfinite(b)&(b>=0)&(b<horizon)
    effective=np.where(~va&vb,0.,np.where(va&~vb,1.,w))
    result=effective*np.nan_to_num(a)+(1-effective)*np.nan_to_num(b)
    return np.where(va|vb,result,np.nan)


def run():
    data=load();folder=OUT/'support_blend';folder.mkdir(exist_ok=True)
    design=dict(method='blend source-fitted MLP and nonlinear T according to source climate support',
        climate_features=['1981–2010 reference April–September mean temperature','reference diurnal temperature range'],
        geometry='source-site standardized Euclidean distance; SD floor 0.1 degrees C',
        radius='95th percentile of leave-one-source-site nearest-neighbour distances',
        weight='min(1,radius/nearest_source_distance)^2',
        invalid_components='use the valid component; if both fail retain unresolved loss',
        ablation='validity-only MLP fallback distinguishes range repair from climate-support weighting',
        tuning='no guard parameter selected on source or target outcomes; component models retain source-only selection',
        status='exploratory response to source-zone failure diagnostics; not untouched confirmatory validation')
    dp=folder/'design.json'
    if dp.exists() and json.loads(dp.read_text())!=design:raise ValueError('Support blend design changed')
    dp.write_text(json.dumps(design,indent=2)+'\n')
    cases=json.loads((OUT/'cases.json').read_text())['cases'];tables=[];case_rows=[];radii=[]
    snapshot=code_signature('support_blend.py','analysis.py')
    for case in cases:
        name=case['case'];dest=folder/name;dest.mkdir(exist_ok=True)
        signature=dict(code=snapshot,inputs=data.protected,case=case,design=design,
            primary_sha256=sha(OUT/'cases'/name/'complete.json'),response_sha256=sha(OUT/'response_followup'/name/'complete.json'))
        if (dest/'complete.json').exists():validate_receipt(json.loads((dest/'complete.json').read_text()),signature,dest)
        else:
            f=pd.read_csv(OUT/'cases'/name/'predictions.csv.gz')
            response=pd.read_csv(OUT/'response_followup'/name/'predictions.csv.gz',usecols=['label_id','relaxed_T'])
            f=f.merge(response,on='label_id',validate='one_to_one')
            source=pd.read_csv(OUT/'memberships'/f'{name}.csv.gz').drop_duplicates('site')
            w,geometry=support_weight(data.context[source.u,:2],data.context[f.u,:2])
            out=f[['label_id','unit','u','site','site_year','block','year','ez','head','observed','horizon','case','T','relaxed_T','residual_MLP']].copy()
            out['climate_weight']=w
            out['validity_fallback']=blend_dates(out.residual_MLP.to_numpy(),out.relaxed_T.to_numpy(),np.ones(len(out)),out.horizon.to_numpy())
            out['climate_blend']=blend_dates(out.residual_MLP.to_numpy(),out.relaxed_T.to_numpy(),w,out.horizon.to_numpy())
            out.to_csv(dest/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
            receipt=dict(signature=signature,geometry=geometry,outputs={'predictions.csv.gz':sha(dest/'predictions.csv.gz')})
            (dest/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
        receipt=json.loads((dest/'complete.json').read_text());f=pd.read_csv(dest/'predictions.csv.gz')
        radii.append(dict(case=name,radius=receipt['geometry']['radius'],**{k:case[k] for k in ['fold','recipe','n_sites','draw_seed','source_ez']}))
        for period,q in f.groupby('case'):
            for model in ['T','relaxed_T','residual_MLP','validity_fallback','climate_blend']:
                error=np.where(valid_prediction(q,q[model]),abs(q[model]-q.observed),366.)
                s=site_errors(q,error)
                for control in ['T','relaxed_T','residual_MLP','validity_fallback']:
                    other=np.where(valid_prediction(q,q[control]),abs(q[control]-q.observed),366.)
                    s['vs_'+control]=site_errors(q,error-other).error
                s=s.merge(q[['site','ez']].drop_duplicates(),on='site',validate='many_to_one')
                s=s.assign(period=period,model=model,recipe=case['recipe'],n_sites=case['n_sites'],draw_seed=case['draw_seed'],
                           fold=case['fold'],source_ez=-1 if case['source_ez'] is None else case['source_ez'])
                tables.append(s)
                case_rows.append(dict(case=name,period=period,model=model,completion=float(valid_prediction(q,q[model]).mean()),
                                      mean_neural_weight=macro(q,q.climate_weight),labels=len(q)))
    table=pd.concat(tables,ignore_index=True);table.to_csv(folder/'site_errors.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(radii).to_csv(folder/'source_radii.csv',index=False);pd.DataFrame(case_rows).to_csv(folder/'case_completion.csv',index=False)
    summary=[];contrasts=[];matrix=[]
    for keys,q in table.groupby(['period','recipe','n_sites','source_ez','model']):
        meta=dict(zip(['period','recipe','n_sites','source_ez','model'],keys))
        a=q.groupby(['head','site','block'])[['error','vs_T','vs_relaxed_T','vs_residual_MLP','vs_validity_fallback']].mean().reset_index()
        lo,hi=block_interval(a)
        summary.append(dict(**meta,mae=float(a.groupby('head').error.mean().mean()),ci_low=lo,ci_high=hi))
        if meta['model']=='climate_blend':
            for control in ['T','relaxed_T','residual_MLP','validity_fallback']:
                c=a[['head','site','block','vs_'+control]].rename(columns={'vs_'+control:'error'});lo,hi=block_interval(c)
                contrasts.append(dict(**meta,control=control,difference=float(c.groupby('head').error.mean().mean()),ci_low=lo,ci_high=hi))
    for keys,q in table[table.source_ez.ge(0)].groupby(['period','source_ez','ez','model']):
        period,s,t,model=keys
        matrix.append(dict(period=period,source_ez=int(s)+1,target_ez=int(t)+1,model=model,
            mae=float(q.groupby('head').error.mean().mean()),difference_vs_T=float(q.groupby('head').vs_T.mean().mean()),sites=q.site.nunique()))
    pd.DataFrame(summary).to_csv(folder/'summary.csv',index=False)
    pd.DataFrame(contrasts).to_csv(folder/'paired_contrasts.csv',index=False)
    pd.DataFrame(matrix).to_csv(folder/'zone_matrix.csv',index=False)
    (folder/'verification.json').write_text(json.dumps(dict(complete=True,cases=len(cases),source_geometry_only=True,
        fixed_target_independent_policy=True,case_signatures=[sha(folder/c['case']/'complete.json') for c in cases],
        protected_inputs_unchanged=all(sha(p)==d for p,d in data.protected.items())),indent=2)+'\n')
    print(pd.DataFrame(summary).query('period == "later_held" and source_ez == -1 and n_sites == 80').round(3).to_string(index=False))


if __name__=='__main__':run()
