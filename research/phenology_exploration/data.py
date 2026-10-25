"""Source-qualified native units, frozen spatial roles and source-only draws."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .core import sha

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/europe_climate_transfer_20261015'
PRIOR=BASE/'combined_refit'
OUT=ROOT/'results/phenology_exploration_20261023/year_matched'
STAGES=('Emergence','Stem elongation','Tasseling','Flowering onset','Milking','Early dough','BBCH 87')


def balanced_counts(groups,n):
    keys=sorted(groups);counts={k:0 for k in keys}
    for _ in range(n):
        available=[k for k in keys if counts[k]<len(groups[k])]
        if not available:raise ValueError('Insufficient source support')
        k=min(available,key=lambda z:(counts[z],z));counts[k]+=1
    return counts


def sample_units(units,n_sites,years_per_site,recipe,seed,source_ez=None):
    if units.unit.duplicated().any():raise ValueError('Duplicate native unit')
    f=units if source_ez is None else units[units.ez.eq(source_ez)]
    eligible=f.groupby('site').year.nunique();eligible=eligible[eligible.ge(years_per_site)].index
    f=f[f.site.isin(eligible)].copy()
    sites=f[['site','ez','inner','block']].drop_duplicates()
    if sites.site.duplicated().any():raise ValueError('Site roles are not constant')
    rng=np.random.default_rng(seed);selected=[]
    # Frozen whole-block roles; equal site budgets in each inner role.
    quotas={False:n_sites-n_sites//3,True:n_sites//3}
    for tune,n in quotas.items():
        q=sites[sites.inner.eq(2).eq(tune)]
        if len(q)<n:raise ValueError('Insufficient site support in inner role')
        if recipe=='random':selected.extend(rng.choice(q.site.to_numpy(),n,replace=False))
        elif recipe=='coverage':
            groups={int(k):v.site.to_numpy() for k,v in q.groupby('ez')}
            for k,count in balanced_counts(groups,n).items():
                selected.extend(rng.choice(groups[k],count,replace=False))
        else:raise ValueError('Unknown source recipe')
    pieces=[]
    for site in selected:
        q=f[f.site.eq(site)].sort_values(['year','unit'])
        years=rng.choice(np.sort(q.year.unique()),years_per_site,replace=False)
        for year in years:
            within=q[q.year.eq(year)]
            pieces.append(within.iloc[[int(rng.integers(len(within)))]])
    result=pd.concat(pieces,ignore_index=True)
    assert result.site.nunique()==n_sites and len(result)==n_sites*years_per_site
    assert result.groupby('site').year.nunique().eq(years_per_site).all()
    assert not result.duplicated(['site','year']).any()
    return result


@dataclass
class Dataset:
    labels:pd.DataFrame
    units:pd.DataFrame
    raw:np.ndarray
    lengths:np.ndarray
    train_lengths:np.ndarray
    context:np.ndarray
    protected:dict


def load():
    portable=ROOT/'data/study_dataset'
    if (portable/'native_inputs.npz').is_file():
        labels=pd.read_csv(portable/'labels.csv.gz',float_precision='round_trip')
        units=pd.read_csv(portable/'units.csv.gz',float_precision='round_trip')
        with np.load(portable/'native_inputs.npz',allow_pickle=False) as z:
            arrays={key:z[key] for key in ['raw','lengths','train_lengths','context']}
        if len(units)!=len(arrays['raw']) or not np.array_equal(units.u,np.arange(len(units))):
            raise ValueError('Portable native-unit array order changed')
        if labels.duplicated(['unit','head']).any():raise ValueError('Conflicting native stage identity')
        protected={str(p):sha(p) for p in sorted(portable.iterdir()) if p.is_file()}
        return Dataset(labels,units,protected=protected,**arrays)
    cols=['label_id','observation_type','head','elapsed_days','canonical_site_id',
          'canonical_site_year_id','source_unit_alias_id','source_environment_alias_id',
          'wi','ez','year','pre2011_prefix_eligible','sowing_date','spatial_block_id']
    cols += [f'fold{k}_case' for k in range(5)]
    path=PRIOR/'combined_observations.csv.gz'
    labels=pd.read_csv(path,usecols=cols)
    labels=labels[labels.observation_type.eq('exact_point')&labels['head'].between(0,6)].copy()
    labels=labels.rename(columns={'canonical_site_id':'site','canonical_site_year_id':'site_year',
        'source_unit_alias_id':'unit','elapsed_days':'observed','spatial_block_id':'block'})
    labels['head']=labels['head'].astype(int);labels['ez']=labels.ez.astype(int)
    if labels.duplicated(['unit','head']).any():raise ValueError('Conflicting native stage identity')
    units=labels[['unit','site','site_year','wi','ez','year','sowing_date','block']].drop_duplicates()
    if units.unit.duplicated().any():raise ValueError('Native-unit weather ambiguity')
    units=units.sort_values('unit').reset_index(drop=True);units['u']=units.index
    labels=labels.merge(units[['unit','u']],on='unit',validate='many_to_one')
    aliases=pd.read_csv(PRIOR/'weather_alias_index.csv.gz').set_index('wi')
    raw=np.empty((len(units),367,3),np.float32);lengths=np.empty(len(units),int)
    weather=aliases.loc[units.wi].reset_index(drop=True)
    files=[path,PRIOR/'weather_alias_index.csv.gz',BASE/'sampling/site_fold_memberships.csv',
           ROOT/'models/maize_clock.pt',ROOT.parent/'maize_phen_model/models/maize_model.json']
    for file,q in weather.groupby('source_alias_weather_file'):
        with np.load(file,allow_pickle=False) as z:
            if int(z['first_DAS'])!=0:raise ValueError('Unexpected native day origin')
            raw[q.index]=z['X'][q.source_alias_weather_index.to_numpy(int)]
            lengths[q.index]=z['lengths'][q.source_alias_weather_index.to_numpy(int)]
        files.append(Path(file))
    if not np.isfinite(raw).all() or (raw[:,:,0]>raw[:,:,1]).any():raise ValueError('Invalid native weather')
    climate=pd.read_csv(BASE/'sampling/site_fold_memberships.csv').set_index('canonical_site_id')
    c=climate.loc[units.site];day=pd.to_datetime(units.sowing_date).dt.dayofyear.to_numpy()
    context=np.column_stack([c.reference_temperature_c,c.reference_dtr_c,
                            np.sin(2*np.pi*day/365.25),np.cos(2*np.pi*day/365.25),raw[:,0,2]])
    if not np.isfinite(context).all():raise ValueError('Missing static environmental context')
    protected={str(f):sha(f) for f in files}
    cutoff=(pd.Timestamp('2011-01-01')-pd.to_datetime(units.sowing_date)).dt.days.to_numpy()
    train_lengths=np.minimum(lengths,np.maximum(cutoff,0)).astype(int)
    labels['horizon']=lengths[labels.u]
    return Dataset(labels,units,raw,lengths,train_lengths,context.astype(np.float32),protected)


def source_units(data,fold):
    portable=ROOT/'data/study_dataset'/f'fold{fold}_source_roles.csv.gz'
    if portable.is_file():
        roles=pd.read_csv(portable)
        if roles.unit.duplicated().any():raise ValueError('Native-unit inner role conflict')
        return data.units.merge(roles,on='unit',validate='one_to_one')
    role=f'fold{fold}_case'
    labels=data.labels[data.labels[role].eq('source_1981_2010')&data.labels.pre2011_prefix_eligible]
    memberships=pd.read_csv(PRIOR/'memberships'/f'fold{fold}_source.csv.gz',usecols=['label_id','inner_partition'])
    joined=labels[['label_id','unit']].merge(memberships,on='label_id',validate='one_to_one')
    roles=joined[['unit','inner_partition']].drop_duplicates().rename(columns={'inner_partition':'inner'})
    if roles.unit.duplicated().any():raise ValueError('Native-unit inner role conflict')
    return data.units.merge(roles,on='unit',validate='one_to_one')


def freeze_design(data):
    OUT.mkdir(parents=True,exist_ok=True);support=[]
    for fold in range(5):
        u=source_units(data,fold)
        for (ez,inner),q in u.groupby(['ez','inner']):
            counts=q.groupby('site').year.nunique()
            support.append(dict(fold=fold,ez=int(ez)+1,inner=int(inner),sites=len(counts),units=len(q),
                                sites_with_8_years=int(counts.ge(8).sum())))
    pd.DataFrame(support).to_csv(OUT/'source_support.csv',index=False)
    design=dict(date='2026-10-23',source_period=[1981,2010],outer_folds=list(range(5)),
        site_budgets=[40,80],years_per_site=8,draw_seeds=[17,29,43],optimizer_seeds=[17,29,43],
        primary_recipes=['random','coverage'],single_zone_site_budget=40,
        temporal_sampling='eight DISTINCT years per site; one randomly chosen native unit per selected site-year',
        single_zone_draw_seeds=[17],minimum_calibration_environments_per_head=30,
        models=['calendar','T','context_T','nonlinear_T','phase_T','residual_RF','residual_MLP','guarded_RF'],
        date_population='source-native German DWD exact event dates',
        inference='retrospective supplied-weather timing; prior target inspection acknowledged',
        uncertainty='whole spatial blocks; source draws and optimizer repeats retained separately',
        protected_inputs=data.protected)
    file=OUT/'design.json'
    if file.exists() and json.loads(file.read_text())!=design:raise ValueError('Frozen design changed')
    file.write_text(json.dumps(design,indent=2)+'\n')
    data.units.to_csv(OUT/'native_units.csv.gz',index=False)
    return design
