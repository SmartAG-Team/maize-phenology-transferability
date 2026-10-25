"""Independent native-prediction reconstruction of geographic evidence."""
import json
import numpy as np
import pandas as pd
from .climate import ROOT,OUT

FOLDER=OUT/'spatial'
PRIOR=ROOT/'results/phenology_exploration_20261023/year_matched'


def site_metrics(frame,model):
    p=frame[model].to_numpy();ok=np.isfinite(p)&(p>=0)&(p<frame.horizon.to_numpy())
    q=frame[['head','site','site_year','target_zone']].copy()
    q['error']=np.where(ok,abs(p-frame.observed.to_numpy()),366.);q['completion']=ok.astype(float)
    return q.groupby(['head','site','target_zone','site_year'])[['error','completion']].mean().groupby(['head','site','target_zone']).mean().reset_index()


def check_cells(expected,actual,dimensions):
    keys=list(dimensions)+['head','model','weather_cell'];metrics=['error','completion']
    if 'penalty' in expected:metrics+=['penalty','reference_completion']
    a=expected.set_index(keys).sort_index();b=actual.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    maximum=float(np.max(abs(a[metrics].to_numpy()-b[metrics].to_numpy())))
    assert maximum<1e-9
    assert np.array_equal(a.n_sites,b.n_sites)
    return len(a)*len(metrics),maximum


def run(data,check):
    for name in ['verification.json','study_network_verification.json']:
        r=json.loads((FOLDER/name).read_text());assert r['complete']
        for file,digest in r['input_sha256'].items():check(file,digest)
        check(ROOT/'research/annual_temperature_transfer'/('spatial.py' if name=='verification.json' else 'study_network.py'),r['code_sha256'])
        for file,digest in r['output_sha256'].items():check(FOLDER/file,digest)
    geo=pd.read_csv(OUT/'site_zones.csv')[['canonical_site_id','weather_cell','weather_latitude','weather_longitude']]
    def aggregate(frame,dimensions):
        f=frame.merge(geo,left_on='site',right_on='canonical_site_id',how='left',validate='many_to_one')
        assert f.weather_cell.notna().all()
        group=list(dimensions)+['head','model','weather_cell'];metrics=['error','completion']
        if 'penalty' in f:metrics+=['penalty','reference_completion']
        q=f.groupby(group)[metrics].mean().reset_index()
        q=q.merge(f.groupby(group).site.nunique().rename('n_sites').reset_index(),on=group,validate='one_to_one')
        return q
    # Old pooled models: all15random/80-site cases, exact native dates and horizons.
    old_cases=[c for c in json.loads((PRIOR/'cases.json').read_text())['cases'] if c['n_sites']==80 and c['source_ez'] is None and c['recipe']=='random']
    assert len(old_cases)==15;rows=[]
    for c in old_cases:
        name=c['case'];f=pd.read_csv(PRIOR/'cases'/name/'predictions.csv.gz',usecols=['label_id','head','site','site_year','observed','horizon','case','T','residual_MLP'])
        n=pd.read_csv(PRIOR/'response_followup'/name/'predictions.csv.gz',usecols=['label_id','relaxed_T'])
        assert set(f.label_id)==set(n.label_id);f=f.merge(n,on='label_id',validate='one_to_one').rename(columns={'relaxed_T':'nonlinear_T'})
        f=f[f['case'].eq('later_held')&f['head'].isin([3,6])].copy();f['target_zone']=0
        for model in ['T','nonlinear_T','residual_MLP']:rows.append(site_metrics(f,model).assign(model=model))
    pooled=pd.concat(rows,ignore_index=True).groupby(['head','site','model'])[['error','completion']].mean().reset_index()
    expected=aggregate(pooled,[]);actual=pd.read_csv(FOLDER/'pooled_cells.csv')
    pooled_checks,largest=check_cells(expected,actual,[])
    # Current transfer: recompute all75single-zone cases, preserving each target's diagonal.
    cases=[c for c in json.loads((OUT/'cases.json').read_text())['cases'] if c['source_zone']>=0]
    assert len(cases)==75;rows=[];identities={}
    for c in cases:
        f=pd.read_csv(OUT/'cases'/c['case']/'predictions.csv.gz',usecols=['label_id','head','site','site_year','observed','horizon','period','target_zone','nonlinear_T','residual_MLP','support_blend'])
        f=f[f.period.eq('later_held')&f['head'].eq(6)]
        ids=tuple(sorted(f.label_id))
        if c['fold'] in identities:assert identities[c['fold']]==ids
        else:identities[c['fold']]=ids
        for model in ['nonlinear_T','residual_MLP','support_blend']:
            rows.append(site_metrics(f,model).assign(model=model,draw=c['draw'],fold=c['fold'],source_zone=c['source_zone']))
    raw=pd.concat(rows,ignore_index=True);keys=['head','site','model','draw','fold']
    diagonal=raw[raw.source_zone.eq(raw.target_zone)][keys+['error','completion']].rename(columns={'error':'reference_error','completion':'reference_completion'})
    q=raw[raw.source_zone.isin([0,4])].merge(diagonal,on=keys,how='left',validate='many_to_one');assert q.reference_error.notna().all()
    q['penalty']=q.error-q.reference_error
    q=q.groupby(['source_zone','head','site','model'])[['error','completion','penalty','reference_completion']].mean().reset_index()
    expected=aggregate(q,['source_zone']);actual=pd.read_csv(FOLDER/'transfer_cells.csv')
    transfer_checks,delta=check_cells(expected,actual,['source_zone']);largest=max(largest,delta)
    # Support and cohort geography from native inventory rather than model summary tables.
    labels=data.labels;assert len(labels)==207325 and labels.label_id.is_unique
    network=pd.read_csv(FOLDER/'study_network.csv').set_index('site')
    years=labels.groupby('site').year.nunique().reindex(network.index)
    assert np.array_equal(years,network.observed_years)
    assert np.array_equal(labels.groupby('site').size().reindex(network.index),network.native_labels)
    assert set(labels.loc[labels.year.gt(2010),'site'])==set(network.index[network.later_evaluation])
    current=labels[labels.year.gt(2010)&labels['head'].isin([3,6])].merge(geo,left_on='site',right_on='canonical_site_id',validate='many_to_one')
    supports=current.groupby(['head','weather_cell']).agg(n_native_labels=('label_id','nunique'),n_observed_years=('year','nunique'),n_native_units=('unit','nunique')).sort_index()
    reference=pd.read_csv(FOLDER/'pooled_cells.csv').drop_duplicates(['head','weather_cell']).set_index(['head','weather_cell']).sort_index()
    assert supports.index.equals(reference.index)
    for column in supports:assert np.array_equal(supports[column],reference[column])
    local=pd.read_csv(PRIOR/'local_adaptation/predictions.csv.gz',usecols=['site','budget']);local=set(local.loc[local.budget.eq(0),'site'])
    weather=set()
    for fold in range(5):
        f=pd.read_csv(OUT/'weather/refits/dwd/cases'/f'f{fold}_weather_d17'/'predictions.csv.gz',usecols=['site','period'])
        weather.update(f.loc[f.period.eq('later_held'),'site'])
    assert set(network.index[network.local_cohort])==local and set(network.index[network.weather_cohort])==weather
    category={'local_only':local-weather,'weather_only':weather-local,'both':local&weather,
              'neither':set(network.index[network.later_evaluation])-(local|weather)}
    for value,sites in category.items():assert set(network.index[network.availability.eq(value)])==sites
    assert np.array_equal(network.annual_zone,(pd.read_csv(OUT/'site_zones.csv').set_index('canonical_site_id').loc[network.index,'zone']+1))
    for name,base in [('pooled','pooled_cells.csv'),('transfer','transfer_cells.csv')]:
        plotted=pd.read_csv(FOLDER/f'{name}_map.csv');native=pd.read_csv(FOLDER/base)
        for panel,g in plotted.groupby('panel'):
            source=native[native['head'].eq(g['head'].iloc[0])]
            if name=='transfer':source=source[source.source_zone.eq(g.source_zone.iloc[0])]
            if g.metric.iloc[0]=='difference':
                pivot=source.pivot(index='weather_cell',columns='model',values='error');expected=pivot.residual_MLP-pivot.nonlinear_T
            else:
                source=source[source.model.eq(g.model.iloc[0])];expected=source.set_index('weather_cell')['penalty' if name=='transfer' else 'error']
            assert np.allclose(g.value,expected.reindex(g.weather_cell),atol=1e-10)
    result=dict(complete=True,native_pooled_cases=15,native_transfer_cases=75,pooled_cell_value_checks=pooled_checks,transfer_cell_value_checks=transfer_checks,
                max_reconstruction_difference_days=largest,per_endpoint_support_reconstructed=True,study_years_and_cohorts_reconstructed=True,
                coordinate_basis='recorded native weather-cell centres',interpolation=False)
    (FOLDER/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    from .design import dataset
    from research.phenology_exploration.core import sha
    data,_,_=dataset()
    def check(path,digest):assert sha(path)==digest
    print(json.dumps(run(data,check),indent=2))
