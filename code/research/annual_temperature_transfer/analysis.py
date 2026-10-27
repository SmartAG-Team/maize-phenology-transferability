"""Paired target-difficulty-adjusted transfer and temperature-distance diagnostics."""
import json
import numpy as np
import pandas as pd
from research.phenology_exploration.analysis import block_interval
from research.phenology_exploration.core import valid_prediction,sha
from .climate import OUT
from .core import transfer_penalties

MODELS=['T','nonlinear_T','residual_RF','residual_MLP','thermal_transformer','support_blend']
EDGES=[-np.inf,-3,-2,-1.5,-1,-.5,0,.5,1,1.5,2,3,np.inf]


def paired_period_penalties(frame):
    return pd.concat([transfer_penalties(q) for _,q in frame.groupby('period',sort=False)],ignore_index=True)


def run():
    cases=json.loads((OUT/'cases.json').read_text())['cases'];tables=[];identity={};receipts=[]
    for case in cases:
        folder=OUT/'cases'/case['case'];receipt=json.loads((folder/'complete.json').read_text())
        assert receipt['signature']['case']==case
        for file,digest in receipt['outputs'].items():
            if sha(folder/file)!=digest:raise ValueError('Changed fitted output')
        receipts.append(sha(folder/'complete.json'));f=pd.read_csv(folder/'predictions.csv.gz')
        assert f.label_id.is_unique
        ids=tuple(sorted(f.label_id));fold=case['fold']
        if fold in identity:assert ids==identity[fold]
        else:identity[fold]=ids
        common=np.logical_and.reduce([valid_prediction(f,f[m]) for m in MODELS])
        f['distance_bin']=pd.cut(f.signed_mat_distance_c,EDGES,labels=False,right=False)
        for model in MODELS:
            ok=valid_prediction(f,f[model]);error=np.where(ok,abs(f[model]-f.observed),366.)
            q=f[['head','site','site_year','block','period','target_zone','mat_c','signed_mat_distance_c','outside_mat_range_c','distance_bin']].copy()
            q['error']=error;q['failure14']=(error>14).astype(float);q['completion']=ok.astype(float)
            q['bias']=np.where(common,f[model]-f.observed,np.nan);q['common_valid_fraction']=common.astype(float)
            keys=['head','site','block','period','target_zone']
            values=['error','failure14','completion','bias','common_valid_fraction','mat_c','signed_mat_distance_c','outside_mat_range_c','distance_bin']
            years=q.groupby(keys+['site_year'])[values].mean().reset_index()
            s=years.groupby(keys)[values].mean().reset_index()
            s['model']=model;s['draw']=case['draw'];s['fold']=fold;s['source_zone']=case['source_zone'];tables.append(s)
    allsites=pd.concat(tables,ignore_index=True)
    # A target site's static MAT fixes its bin within each source case.
    assert np.allclose(allsites.distance_bin,np.rint(allsites.distance_bin))
    paired=paired_period_penalties(allsites)
    paired.to_csv(OUT/'site_transfer_errors.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    matrix=[];direction=[];curves=[];stages=[];pooled=[]
    def estimate(q,column):
        a=q.groupby(['head','site','block'])[column].mean().reset_index().rename(columns={column:'error'})
        support=a['head'].nunique()
        if support!=7 or not np.isfinite(a.error).all():return dict(value=np.nan,ci_low=np.nan,ci_high=np.nan,heads=support,sites=a.site.nunique())
        lo,hi=block_interval(a)
        return dict(value=float(a.groupby('head').error.mean().mean()),ci_low=lo,ci_high=hi,heads=support,sites=a.site.nunique())
    for keys,q in paired.groupby(['period','model','source_zone','target_zone']):
        period,model,source,target=keys
        row=dict(period=period,model=model,source_zone=int(source),target_zone=int(target))
        for value in ['error','penalty','failure14','completion','bias','common_valid_fraction']:
            result=estimate(q,value)
            row.update({value if k=='value' else value+'_'+k:v for k,v in result.items()})
        row['mean_signed_mat_distance_c']=float(q.groupby('site').signed_mat_distance_c.mean().mean())
        matrix.append(row)
        for head,h in q.groupby('head'):
            stages.append(dict(**row,head=int(head),stage_error=float(h.groupby('site').error.mean().mean()),stage_penalty=float(h.groupby('site').penalty.mean().mean())))
    off=paired[paired.source_zone.ge(0)&paired.source_zone.ne(paired.target_zone)].copy()
    off['direction']=np.where(off.target_zone.gt(off.source_zone),'cold_to_warm','warm_to_cold')
    for keys,q in off.groupby(['period','model','direction']):
        period,model,name=keys;row=dict(period=period,model=model,direction=name)
        for value in ['error','penalty','failure14','bias']:
            row.update({value if k=='value' else value+'_'+k:v for k,v in estimate(q,value).items()})
        direction.append(row)
    for keys,q in off.groupby(['period','model','distance_bin']):
        period,model,bin=keys;row=dict(period=period,model=model,distance_bin=int(bin),
            signed_mat_distance_c=float(q.groupby(['head','site']).signed_mat_distance_c.mean().groupby('head').mean().mean()))
        for value in ['error','penalty','failure14','bias']:
            row.update({value if k=='value' else value+'_'+k:v for k,v in estimate(q,value).items()})
        curves.append(row)
    for keys,q in paired.groupby(['period','model']):
        period,model=keys
        for name,mask in [('all_single_zone',q.source_zone.ge(0)),('off_diagonal',q.source_zone.ge(0)&q.source_zone.ne(q.target_zone)),('diagonal',q.source_zone.eq(q.target_zone)),('pooled_source',q.source_zone.eq(-1))]:
            row=dict(period=period,model=model,comparison=name)
            for value in ['error','penalty','failure14','completion']:
                row.update({value if k=='value' else value+'_'+k:v for k,v in estimate(q.loc[mask],value).items()})
            pooled.append(row)
    for name,rows in [('matrix',matrix),('directional_summary',direction),('temperature_distance_curves',curves),('stage_matrix',stages),('summary',pooled)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    diagonal=paired[paired.source_zone.eq(paired.target_zone)]
    assert np.allclose(diagonal.penalty,0)
    verification=dict(complete=True,cases=len(cases),source_case_sha256=receipts,identical_targets_across_source_zones_and_models=True,
        diagonal_is_zero=True,missing_diagonal_targets=int((~paired.diagonal_available).sum()),
        metric='equal native observations within site-year, years/site, sites/head, seven heads; invalid dates loss366',
        intervals='2000 paired whole target-block draws conditional on fitted source models; three source draws averaged before resampling',
        interpretation='descriptive environmental transfer; no causal temperature effect',files={p.name:sha(p) for p in OUT.glob('*.csv*')})
    (OUT/'analysis_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(pd.DataFrame(direction).query('period == "later_held"')[['model','direction','error','penalty','failure14']].round(3).to_string(index=False))


if __name__=='__main__':run()
