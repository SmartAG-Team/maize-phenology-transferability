"""Frozen DWD/ERA5 common outcomes and weather-station-aware source groups."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.phenology_exploration.data import source_units,sample_units
from research.phenology_exploration.core import sha
from .climate import OUT
from .design import dataset

FOLDER=OUT/'weather'


def grouped_source(units,seed=17):
    q=units.copy();parent={str(b):str(b) for b in q.block.unique()}
    def find(x):
        x=str(x)
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for _,g in q.groupby('station'):
        blocks=sorted(g.block.astype(str).unique());root=find(blocks[0])
        for block in blocks[1:]:
            a,b=sorted([find(root),find(block)]);parent[b]=a;root=a
    q['native_block']=q.block;q['block']=q.block.astype(str).map(lambda b:'WEATHER_GROUP_'+find(b))
    sizes=q.groupby('block').site.nunique();rng=np.random.default_rng(seed);jitter={g:rng.random() for g in sorted(sizes.index)}
    loads=np.zeros(3,int);roles={}
    for group in sorted(sizes.index,key=lambda g:(-sizes[g],jitter[g])):
        role=int(loads.argmin());roles[group]=role;loads[role]+=int(sizes[group])
    q['inner']=q.block.map(roles).astype(int)
    return q


def common_inputs():
    dest=FOLDER/'common_inputs.npz';inventory=FOLDER/'common_units.csv.gz';done=FOLDER/'common_qualification.json'
    if done.exists():
        m=json.loads(done.read_text())
        if m['code_sha256']!=sha(__file__):raise ValueError('Common-weather calculation changed')
        for name,value in m['outputs'].items():
            if sha(FOLDER/name)!=value:raise ValueError('Common-weather output changed')
        return pd.read_csv(inventory)
    assert (FOLDER/'era5_land/manifest.json').exists()
    q=pd.read_csv(FOLDER/'dwd_complete_units.csv.gz')
    with np.load(FOLDER/'dwd_native_weather.npz') as z:
        assert np.array_equal(z['u'],q.u);dwd=z['temperature']
    temps=[];dates=[];order=None;inputs={str(FOLDER/'dwd_qualification.json'):sha(FOLDER/'dwd_qualification.json')}
    for year in range(2001,2026):
        path=FOLDER/'era5_land'/f'{year}.npz'
        with np.load(path) as z:
            if order is None:order=z['sites']
            else:assert np.array_equal(order,z['sites'])
            temps.append(z['temperature']);dates.append(z['dates'])
        inputs[str(path)]=sha(path)
    weather=np.concatenate(temps,axis=1);dates=np.concatenate(dates)
    expected=pd.date_range('2001-01-01','2025-12-31').to_numpy(dtype='datetime64[D]').astype('int64')
    assert np.array_equal(dates,expected)
    rows=q.site.map({s:i for i,s in enumerate(order)}).to_numpy(int)
    starts=pd.to_datetime(q.sowing_date).to_numpy(dtype='datetime64[D]').astype('int64')-dates[0]
    index=starts[:,None]+np.arange(367)[None,:];assert index.min()>=0 and index.max()<len(dates)
    era=weather[rows[:,None],index]
    complete=np.isfinite(era).all((1,2))&(era[:,:,0]<=era[:,:,1]).all(1)&(era[:,:,0]>=-80).all(1)&(era[:,:,1]<=65).all(1)
    audit=q.copy();audit['era5_complete']=complete;audit.to_csv(FOLDER/'era5_window_audit.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    q=q.loc[complete].reset_index(drop=True)
    q.to_csv(inventory,index=False,compression={'method':'gzip','mtime':0})
    np.savez_compressed(dest,u=q.u.to_numpy(int),dwd=dwd[complete],era5=era[complete])
    meta=dict(complete=True,code_sha256=sha(__file__),inputs=inputs,
        observations_used_for_weather_selection=False,source_years=[2001,2010],test_years=[2011,2024],
        common_native_units=len(q),common_sites=q.site.nunique(),weather_window_days=367,
        outputs={p.name:sha(p) for p in [dest,inventory,FOLDER/'era5_window_audit.csv.gz']})
    done.write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k not in ['inputs','outputs']}),flush=True)
    return q


def freeze_weather():
    common=common_inputs();data,reference,_=dataset();common=common.set_index('u')
    labels=data.labels[data.labels.u.isin(common.index)].copy();units=data.units[data.units.u.isin(common.index)].copy()
    units['station']=units.u.map(common.station)
    data=replace(data,labels=labels,units=units)
    folder=FOLDER/'paired_design';(folder/'memberships').mkdir(parents=True,exist_ok=True);(folder/'source_pools').mkdir(exist_ok=True)
    design=dict(source_years=[2001,2010],test_years=[2011,2024],sites_per_fit=40,distinct_years_per_site=4,draws=[17,29,43],folds=list(range(5)),
        source_budget_reason='only 49 DWD sites provide eight UTC-qualified source years across the whole network; four-year budget supports spatially held source pools',
        stations='nearest catalogued station within10 km; time-valid station locations and UTC day definitions checked every day',
        outer_split='existing native 50-km outer blocks and25-km cell buffers; source station IDs also excluded if present at target sites',
        inner_and_teacher_groups='connected unions of source native blocks and shared weather-station IDs; whole groups in inner validation and residual teachers',
        context='original climate/sowing/daylength covariates held constant across DWD and ERA5',
        comparison='fixed-parameter weather swaps plus same-budget product-specific refits; same source and target native outcomes',
        interpretation='weather product and station/grid representativeness sensitivity; not an error-free ground-truth-weather claim')
    file=folder/'design.json'
    if file.exists() and json.loads(file.read_text())!=design:raise ValueError('Frozen weather design changed')
    file.write_text(json.dumps(design,indent=2)+'\n');cases=[];skipped=[]
    for fold in range(5):
        pool=source_units(data,fold)
        role=f'fold{fold}_case';target=labels[labels[role].isin(['historical_held','later_held'])]
        target_stations=set(target.u.map(common.station))
        pool=pool[~pool.station.isin(target_stations)]
        eligible=pool.groupby('site').year.nunique();pool=pool[pool.site.isin(eligible[eligible.ge(4)].index)]
        pool=grouped_source(pool);pool.to_csv(folder/'source_pools'/f'fold{fold}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        for draw in [17,29,43]:
            name=f'f{fold}_weather_d{draw}'
            try:sample=sample_units(pool,40,4,'random',draw)
            except ValueError as e:skipped.append(dict(case=name,reason=str(e)));continue
            source=labels[labels.unit.isin(sample.unit)&labels.pre2011_prefix_eligible]
            train=source[source.unit.map(sample.set_index('unit').inner).ne(2)]
            counts=train.groupby('head').u.nunique().reindex(range(7),fill_value=0)
            if counts.min()<30:skipped.append(dict(case=name,reason='inner native-head support <30',head_counts=counts.tolist()));continue
            path=folder/'memberships'/f'{name}.csv.gz'
            if path.exists():
                if not pd.read_csv(path).astype(str).equals(sample.astype(str)):raise ValueError('Frozen weather membership changed')
            else:sample.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
            cases.append(dict(case=name,fold=fold,source_zone=-1,draw=draw,n_sites=40,n_years=4,source_units=len(sample),membership_sha256=sha(path)))
    payload=dict(cases=cases,skipped=skipped)
    file=folder/'cases.json'
    if file.exists() and json.loads(file.read_text())!=payload:raise ValueError('Frozen weather cases changed')
    file.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(dict(weather_cases=len(cases),skipped=skipped)),flush=True)
    return data,common.reset_index(),cases


if __name__=='__main__':freeze_weather()
