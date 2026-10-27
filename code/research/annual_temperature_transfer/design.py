"""Frozen annual-temperature zones and matched native-year source samples."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.phenology_exploration.data import load,source_units,sample_units
from research.phenology_exploration.core import sha
from .climate import ROOT,OUT
from .core import temperature_zones


def dataset():
    data=load();path=OUT/'climate/site_annual_climate.csv'
    climate=pd.read_csv(path).set_index('canonical_site_id')
    sites=sorted(set(data.labels.site)&set(climate.index[climate.annual_climate_complete]))
    reference=climate.loc[sites].copy();reference['zone'],edges=temperature_zones(reference.annual_mean_temperature_c)
    mapping=reference.zone
    labels=data.labels[data.labels.site.isin(sites)].copy();labels['ez']=labels.site.map(mapping).astype(int)
    labels['mat_c']=labels.site.map(reference.annual_mean_temperature_c)
    units=data.units[data.units.site.isin(sites)].copy();units['ez']=units.site.map(mapping).astype(int)
    units['mat_c']=units.site.map(reference.annual_mean_temperature_c)
    protected=dict(data.protected);protected[str(path)]=sha(path)
    return replace(data,labels=labels,units=units,protected=protected),reference.reset_index(),edges


def freeze(data,reference,edges):
    OUT.mkdir(exist_ok=True);(OUT/'memberships').mkdir(exist_ok=True)
    design=dict(date='2026-10-24',geometry='1981–2010 complete annual mean temperature; exact-date-site quintiles, unsplit ties',
        zone_edges_c=edges.tolist(),zones=5,sites_per_fit=40,distinct_years_per_site=8,source_draws=[17,29,43],folds=list(range(5)),
        source_zones=list(range(5))+[-1],pooled_control='random sites across all qualified annual zones; same 40-site/eight-year budget',
        models=['T','nonlinear_T','residual_RF','residual_MLP','thermal_transformer','support_blend'],
        model_selection='best-performing families from previous retrospective exploration; no prospective selection claim',
        model_inputs='unchanged original native weather and Apr–Sep/sowing context; annual MAT defines source/target zones',
        neural_optimizer_seed=17,minimum_head_support=30,source_end=2010,primary_evaluation=[2011,2025],
        protected_inputs=data.protected)
    dp=OUT/'design.json'
    if dp.exists() and json.loads(dp.read_text())!=design:raise ValueError('Frozen annual-temperature design changed')
    dp.write_text(json.dumps(design,indent=2)+'\n');reference.to_csv(OUT/'site_zones.csv',index=False)
    cases=[];skipped=[];support=[]
    for fold in range(5):
        units=source_units(data,fold)
        for zone,q in units.groupby('ez'):
            for role,g in q.groupby(q.inner.eq(2)):
                support.append(dict(fold=fold,zone=int(zone),validation=bool(role),sites_with_8_years=int(g.groupby('site').year.nunique().ge(8).sum())))
        for zone in list(range(5))+[-1]:
            for draw in [17,29,43]:
                name=f'f{fold}_z{zone+1 if zone>=0 else "pooled"}_d{draw}'
                try:m=sample_units(units,40,8,'random',draw,None if zone<0 else zone)
                except ValueError as e:skipped.append(dict(case=name,reason=str(e)));continue
                source=data.labels[data.labels.unit.isin(m.unit)&data.labels.pre2011_prefix_eligible]
                inner=m.set_index('unit').inner
                train=source[source.unit.map(inner).ne(2)]
                counts=train.groupby('head').u.nunique().reindex(range(7),fill_value=0)
                if counts.min()<30:skipped.append(dict(case=name,reason='Insufficient native-head inner-fit support',counts=counts.tolist()));continue
                path=OUT/'memberships'/f'{name}.csv.gz'
                if path.exists():
                    if not pd.read_csv(path).astype(str).equals(m.astype(str)):raise ValueError('Frozen membership changed')
                else:m.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
                cases.append(dict(case=name,fold=fold,source_zone=zone,draw=draw,n_sites=40,n_years=8,source_units=len(m),membership_sha256=sha(path)))
    payload=dict(cases=cases,skipped=skipped)
    cp=OUT/'cases.json'
    if cp.exists() and json.loads(cp.read_text())!=payload:raise ValueError('Frozen case inventory changed')
    cp.write_text(json.dumps(payload,indent=2)+'\n');pd.DataFrame(support).to_csv(OUT/'source_support.csv',index=False)
    print(reference.groupby('zone').annual_mean_temperature_c.agg(['count','min','mean','max']).round(3).to_string())
    print(json.dumps(dict(qualified_cases=len(cases),skipped=skipped)),flush=True)
    return cases


if __name__=='__main__':freeze(*dataset())
