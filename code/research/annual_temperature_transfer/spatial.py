"""Paired held-result maps on recorded native weather-cell support."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.phenology_exploration.core import sha
from .climate import ROOT,OUT

FOLDER=OUT/'spatial'
BASEMAP=ROOT/'code/research/unified_transfer/assets/naturalearth/ne_50m_admin_0_countries.shp'
HEADS={3:61,6:87}


def cell_results(frame,coordinates,dimensions=()):
    dimensions=list(dimensions);keys=dimensions+['head','model','site']
    if frame.duplicated(keys).any():raise ValueError('Duplicate native site/head/model result')
    geo=coordinates[['canonical_site_id','weather_cell','weather_latitude','weather_longitude']].copy()
    if geo.canonical_site_id.duplicated().any():raise ValueError('Ambiguous native coordinate mapping')
    if geo.groupby('weather_cell')[['weather_latitude','weather_longitude']].nunique().gt(1).any().any():
        raise ValueError('Inconsistent coordinates for a native weather cell')
    q=frame.merge(geo,left_on='site',right_on='canonical_site_id',how='left',validate='many_to_one',indicator=True)
    valid=q.weather_latitude.between(-90,90)&q.weather_longitude.between(-180,180)
    if not q['_merge'].eq('both').all() or not valid.all():raise ValueError('Missing or invalid native coordinates')
    for _,g in q.groupby(dimensions+['head']):
        supports=[set(h.site) for _,h in g.groupby('model')]
        if any(s!=supports[0] for s in supports[1:]):raise ValueError('Unmatched native model support')
    if q.groupby('weather_cell').block.nunique().gt(1).any():raise ValueError('Native cell crosses frozen spatial blocks')
    group=dimensions+['head','model','weather_cell','weather_latitude','weather_longitude','block']
    metrics=[c for c in ['error','penalty','completion','reference_completion'] if c in q]
    if not np.isfinite(q[metrics].to_numpy(float)).all():raise ValueError('Nonfinite scored result')
    result=q.groupby(group)[metrics].mean().reset_index()
    counts=q.groupby(group).site.nunique().rename('n_sites').reset_index()
    minima=q.groupby(group)[[c for c in ['completion','reference_completion'] if c in q]].min()
    unresolved=minima.lt(1-1e-12).any(axis=1).rename('any_unresolved').reset_index()
    return result.merge(counts,on=group,validate='one_to_one').merge(unresolved,on=group,validate='one_to_one')


def cell_difference(frame,left,right,value='error'):
    keys=['head','weather_cell']
    if 'source_zone' in frame:keys=['source_zone']+keys
    a=frame[frame.model.eq(left)].copy();b=frame[frame.model.eq(right)].copy()
    if a.duplicated(keys).any() or b.duplicated(keys).any():raise ValueError('Duplicate cell contrast')
    q=a.merge(b[keys+[value,'n_sites','any_unresolved']],on=keys,how='outer',validate='one_to_one',suffixes=('_left','_right'),indicator=True)
    if not q['_merge'].eq('both').all() or not q.n_sites_left.eq(q.n_sites_right).all():raise ValueError('Unmatched cell contrast support')
    q['difference']=q[value+'_left']-q[value+'_right'];q['n_sites']=q.n_sites_left
    q['any_unresolved']=q.any_unresolved_left|q.any_unresolved_right
    return q.drop(columns='_merge')


def run():
    FOLDER.mkdir(parents=True,exist_ok=True);inputs={}
    def record(path):inputs[str(path)]=sha(path)
    coordinate_path=OUT/'site_zones.csv';record(coordinate_path);geo=pd.read_csv(coordinate_path)
    source=OUT/'insights/benchmark_sites.csv.gz';record(source)
    evidence=json.loads((OUT/'insights/verification.json').read_text());assert sha(source)==evidence['output_sha256'][source.name]
    sites=pd.read_csv(source)
    sites=sites[sites.study.eq('prior_pooled_80')&sites.period.eq('later_held')&sites.recipe.eq('random')&sites['head'].isin(HEADS)&sites.model.isin(['T','nonlinear_T','residual_MLP'])].copy()
    pooled=cell_results(sites,geo)
    # Native support from exactly one original held prediction set, not repeated source draws.
    support=[]
    prior=ROOT/'data/results/phenology_exploration_20261023/year_matched'
    for fold in range(5):
        folder=prior/'cases'/f'f{fold}_random_n80_d17';receipt=json.loads((folder/'complete.json').read_text())
        path=folder/'predictions.csv.gz';assert sha(path)==receipt['outputs'][path.name];record(path)
        f=pd.read_csv(path,usecols=['label_id','unit','site','site_year','year','head','case'])
        support.append(f[f['case'].eq('later_held')])
    support=pd.concat(support,ignore_index=True);assert support.label_id.is_unique and len(support)==56447
    support=support[support['head'].isin(HEADS)].merge(geo[['canonical_site_id','weather_cell']],left_on='site',right_on='canonical_site_id',validate='many_to_one')
    assert set(map(tuple,support[['head','site']].drop_duplicates().to_numpy()))==set(map(tuple,sites[['head','site']].drop_duplicates().to_numpy()))
    support=support.groupby(['head','weather_cell']).agg(n_native_labels=('label_id','nunique'),n_observed_years=('year','nunique'),n_native_units=('unit','nunique'),support_sites=('site','nunique')).reset_index()
    pooled=pooled.merge(support,on=['head','weather_cell'],validate='many_to_one');assert pooled.n_sites.eq(pooled.support_sites).all()
    pooled.to_csv(FOLDER/'pooled_cells.csv',index=False)
    panels=[]
    for row,head in enumerate(HEADS):
        h=pooled[pooled['head'].eq(head)]
        for column,model in enumerate(['T','nonlinear_T']):
            q=h[h.model.eq(model)].copy();q['value']=q.error
            q=q.assign(panel='('+chr(97+row*3+column)+')',metric='mae',reference_model='',bbch=HEADS[head],display_min=0.,display_max=30.)
            panels.append(q)
        q=cell_difference(h,'residual_MLP','nonlinear_T');q['value']=q.difference
        panels.append(q.assign(panel='('+chr(99+row*3)+')',metric='difference',reference_model='nonlinear_T',bbch=HEADS[head],display_min=-10.,display_max=10.))
    pooled_map=pd.concat(panels,ignore_index=True);pooled_map.to_csv(FOLDER/'pooled_map.csv',index=False)
    # Transfer penalties retain matching source draws, folds and target-diagonal predictions.
    source=OUT/'site_transfer_errors.csv.gz';record(source)
    receipt=json.loads((OUT/'analysis_verification.json').read_text());assert sha(source)==receipt['files'][source.name]
    columns=['head','site','block','period','model','draw','fold','source_zone','target_zone','error','penalty','diagonal_error','completion']
    pieces=[]
    for chunk in pd.read_csv(source,usecols=columns,chunksize=250000):
        pieces.append(chunk[chunk.period.eq('later_held')&chunk['head'].eq(6)&chunk.source_zone.ge(0)&chunk.model.isin(['nonlinear_T','residual_MLP','support_blend'])])
    q=pd.concat(pieces,ignore_index=True)
    keys=['head','site','model','draw','fold'];diagonal=q[q.source_zone.eq(q.target_zone)][keys+['completion','error']].rename(columns={'completion':'reference_completion','error':'reference_error'})
    q=q[q.source_zone.isin([0,4])].merge(diagonal,on=keys,validate='many_to_one')
    assert np.allclose(q.diagonal_error,q.reference_error) and np.allclose(q.penalty,q.error-q.reference_error)
    keys=['source_zone','head','model','site','block'];assert q.groupby(keys).size().eq(3).all()
    q=q.groupby(keys)[['error','penalty','completion','reference_completion']].mean().reset_index()
    transfer=cell_results(q,geo,['source_zone']).merge(support,on=['head','weather_cell'],validate='many_to_one')
    assert transfer.n_sites.eq(transfer.support_sites).all()
    transfer.to_csv(FOLDER/'transfer_cells.csv',index=False)
    rows=[]
    for row,zone in enumerate([0,4]):
        for column,model in enumerate(['nonlinear_T','residual_MLP','support_blend']):
            h=transfer[transfer.source_zone.eq(zone)&transfer.model.eq(model)].copy();h['value']=h.penalty
            rows.append(h.assign(panel='('+chr(97+row*3+column)+')',metric='penalty',bbch=87,display_min=-30.,display_max=30.))
    transfer_map=pd.concat(rows,ignore_index=True);transfer_map.to_csv(FOLDER/'transfer_map.csv',index=False)
    summary=[]
    for name,frame in [('pooled',pooled_map),('transfer',transfer_map)]:
        for panel,g in frame.groupby('panel'):
            summary.append(dict(map=name,panel=panel,model=g.model.iloc[0],head=int(g['head'].iloc[0]),cells=len(g),sites=int(g.n_sites.sum()),
                minimum=float(g.value.min()),p10=float(g.value.quantile(.1)),median=float(g.value.median()),p90=float(g.value.quantile(.9)),maximum=float(g.value.max()),
                negative_cells=int(g.value.lt(0).sum()),positive_cells=int(g.value.gt(0).sum()),unresolved_cells=int(g.any_unresolved.sum()),
                clipped_low_cells=int(g.value.lt(g.display_min).sum()),clipped_high_cells=int(g.value.gt(g.display_max).sum())))
    pd.DataFrame(summary).to_csv(FOLDER/'summary.csv',index=False)
    for p in BASEMAP.parent.glob(BASEMAP.stem+'.*'):record(p)
    notice=BASEMAP.parent/'NOTICE.md';record(notice)
    receipt=dict(complete=True,input_sha256=inputs,code_sha256=sha(__file__),projection='EPSG:3035',coordinate_basis='recorded native weather-cell centres',
        cell_aggregation='equal monitoring sites after native-year averaging; head61/87 separate; three source draws already averaged',
        interpolation=False,paired_model_support=True,transfer_diagonal_reconstructed=True,
        display_limits_days={'mae':[0,30],'pooled_model_difference':[-10,10],'transfer_penalty':[-30,30]},
        interpretation='descriptive geographic distributions at sampled cells; no significant spatial cluster or causal climate claim',
        output_sha256={name:sha(FOLDER/name) for name in ['pooled_cells.csv','pooled_map.csv','transfer_cells.csv','transfer_map.csv','summary.csv']})
    (FOLDER/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__=='__main__':run()
