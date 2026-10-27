"""Immutable-source assembly for joint European date/summary fitting."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'data/results/europe_climate_transfer_20261015'
HEADS=('NATIVE_678c5e271701bed181fd','NATIVE_799daca32a1553244d95',
       'NATIVE_ebfb24763c3fe6ed82fe','NATIVE_fb0c8e2408357e28866f',
       'NATIVE_a769707a563ced1cbe6a','NATIVE_5319025324f3f9b5ca70',
       'NATIVE_c522e6c7b48a1909fb84','NATIVE_f37e97cb43fe0e287939')
NAMES=('Emergence','Stem elongation','Tasseling','Flowering onset','Milking',
       'Early dough','Maturity (BBCH 87)','Full flowering (native)')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def training_lengths(sowing,lengths):
    days=(pd.Timestamp('2011-01-01')-pd.to_datetime(sowing)).dt.days.to_numpy()
    return np.minimum(np.asarray(lengths),np.maximum(days,0)).astype(int)

def inner_partition(frame,seed):
    blocks=sorted(frame.spatial_block_id.astype(str).unique())
    rng=np.random.default_rng(seed);rng.shuffle(blocks)
    lookup={b:i%3 for i,b in enumerate(blocks)}
    return frame.spatial_block_id.astype(str).map(lookup).to_numpy(int)

def weekly_features(raw):
    """Past-only daily summaries sampled at DAS 0, 7, ... 364."""
    raw=np.asarray(raw,np.float32);n,t,_=raw.shape;days=np.arange(0,t,7)
    low,high,light=np.moveaxis(raw,2,0);mean=(low+high)/2
    gdd=(np.clip(low,8,30)+np.clip(high,8,30))/2-8
    def roll(x,w):
        c=np.pad(x.cumsum(1),((0,0),(1,0)));ends=days+1;starts=np.maximum(ends-w,0)
        return (c[:,ends]-c[:,starts])/(ends-starts)
    values=[low[:,days],high[:,days],light[:,days],mean[:,days],(high-low)[:,days],gdd[:,days],
        np.maximum(high[:,days]-30,0),np.broadcast_to(days[None]/366,(n,len(days))),
        gdd.cumsum(1)[:,days]/1000,mean.cumsum(1)[:,days]/1000,light.cumsum(1)[:,days]/1000,
        roll(gdd,7),roll(gdd,30),roll(mean,7),roll(mean,30),np.zeros((n,len(days))),
        roll(light,7),roll(light,30)]
    return np.stack(values,2).astype(np.float32)

def assemble():
    cols=['label_id','observation_type','native_head_id','elapsed_days','visit_das','bbch_lower','bbch_upper',
        'source_column','native_group_id','canonical_site_id','canonical_site_year_id','source_unit_alias_id',
        'source_environment_alias_id','sowing_date','year','country','calendar_eligible_for_sowing_based_models',
        'source_alias_weather_file','source_alias_weather_index','weather_day_basis_canonical']
    labels=pd.read_csv(BASE/'data/labels.csv.gz',usecols=cols,low_memory=False)
    cases=pd.read_csv(BASE/'sampling/label_case_memberships.csv.gz').drop(columns='canonical_site_id')
    labels=labels.merge(cases,on='label_id',validate='one_to_one')
    sites=pd.read_csv(BASE/'sampling/site_fold_memberships.csv')[['canonical_site_id','spatial_block_id']]
    labels=labels.merge(sites,on='canonical_site_id',validate='many_to_one')
    eligible=labels.calendar_eligible_for_sowing_based_models & labels.year.between(1981,2025)&labels.ez.notna()
    labels=labels[eligible].copy().reset_index(drop=True)
    labels['head']=labels.native_head_id.map({h:i for i,h in enumerate(HEADS)}).fillna(-1).astype(int)
    labels['phase']=np.floor(labels.bbch_lower.fillna(-10)/10).astype(int)
    labels['statistic']=np.where(labels.source_column.eq('CropStageAtAppRaw'),1,0)
    labels['summary_eligible']=(labels.observation_type.eq('ordinal_visit')&labels.phase.isin([1,3,5,6,7,8])
        &labels.phase.eq(np.floor(labels.bbch_upper.fillna(-10)/10)))
    aliases=labels[['source_environment_alias_id','source_alias_weather_file','source_alias_weather_index',
        'sowing_date','canonical_site_id','canonical_site_year_id','year','spatial_block_id']].drop_duplicates()
    if aliases.source_environment_alias_id.duplicated().any():raise ValueError('Weather alias identity conflicts')
    aliases=aliases.sort_values('source_environment_alias_id').reset_index(drop=True)
    labels['wi']=labels.source_environment_alias_id.map(dict(zip(aliases.source_environment_alias_id,aliases.index))).astype(int)
    raw=np.empty((len(aliases),367,3),np.float32);lengths=np.empty(len(aliases),int)
    for file,q in aliases.groupby('source_alias_weather_file'):
        with np.load(file,allow_pickle=False) as z:
            ids=q.source_alias_weather_index.to_numpy(int)
            if int(z['first_DAS'])!=0:raise ValueError('Sowing day must be DAS 0')
            raw[q.index]=z['X'][ids];lengths[q.index]=z['lengths'][ids]
    if not np.isfinite(raw).all() or (raw[:,:,0]>raw[:,:,1]).any():raise ValueError('Invalid native weather')
    train_lengths=training_lengths(aliases.sowing_date,lengths)
    # Cut off labels themselves, in addition to the source weather arrays.
    source_event=np.where(labels.observation_type.eq('exact_point'),labels.elapsed_days,labels.visit_das)
    labels['pre2011_prefix_eligible']=source_event<train_lengths[labels.wi]
    return labels,aliases,raw,lengths,train_lengths
