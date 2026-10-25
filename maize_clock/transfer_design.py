"""Nested calendar splits with environmental blocks removed before selection."""
import hashlib
import numpy as np
import pandas as pd
from .experiment import experiment_meta
from .extension_boosting import add_context_normalizers


def sowing_reference(env,calibration):
    dates=pd.to_datetime(env.sowing_date).dt.dayofyear.astype(float)
    selected=env.loc[calibration,['site_id']].copy();selected['day']=dates[calibration]
    if selected.empty:raise ValueError('No calibration sowing dates')
    medians=selected.groupby('site_id').day.median()
    global_median=float(selected.day.median())
    anomalies=selected.day-selected.site_id.map(medians)
    return {'station_medians':medians.to_dict(),'global_median':global_median,
            'early_threshold':float(anomalies.quantile(.1)),
            'late_threshold':float(anomalies.quantile(.9))}


def sowing_anomaly(env,reference):
    median=env.site_id.map(reference['station_medians'])
    unknown=median.isna().to_numpy()
    doy=pd.to_datetime(env.sowing_date).dt.dayofyear.to_numpy(float)
    return doy-median.fillna(reference['global_median']).to_numpy(),unknown


def withhold_blocks(env,condition):
    keys=pd.MultiIndex.from_frame(env[['climate_component','year']])
    return np.asarray(keys.isin(keys[np.asarray(condition,bool)]))


def nested_masks(env,experiment,warm_years=()):
    year=env.year.to_numpy(int)
    german=env.dataset_id.eq('dwd_annual').to_numpy() if 'dataset_id' in env else np.ones(len(env),bool)
    condition=np.zeros(len(env),bool)
    if experiment.startswith('zone_'):condition=env.regime.eq(int(experiment[-1])).to_numpy()
    elif experiment=='warm_years':condition=np.isin(year,list(warm_years))
    elif experiment in {'early_sowing','late_sowing'}:
        reference=sowing_reference(env,german&(year<=2013))
        anomaly,_=sowing_anomaly(env,reference)
        condition=anomaly<reference['early_threshold'] if experiment=='early_sowing' else anomaly>reference['late_threshold']
    elif experiment!='germany':raise ValueError('Unknown experiment')
    held=withhold_blocks(env,german&condition)
    allowed=german&~held
    masks={'inner_calibration':allowed&(year<=2013),
           'inner_tuning':allowed&(year>=2014)&(year<=2017),
           'final_calibration':allowed&(year<=2017),
           'final_tuning':allowed&(year>=2018)&(year<=2020),
           'test':german&(year>=2021)&(year<=2025)&(held if experiment!='germany' else True),
           'withheld':held&german}
    return masks


def calibration_metadata(a,env,source,calibration,min_support=30):
    calibration=np.asarray(calibration,bool)
    counts=((a['Y'][:,:7]>0)&calibration[:,None]).sum(0)
    if np.any(counts<min_support):raise ValueError('Insufficient primary stage support')
    # Existing helper requires 30; use its semantics directly for production.
    if min_support>=30:
        local=env.copy();local['new_partition']=np.where(calibration,'train','excluded')
        m,_=experiment_meta(a,local,source)
        m.update(retained=list(range(7)),endpoints=source['endpoints'][:7],
                 threshold_gdd=m['threshold_gdd'][:7],threshold_fraction=m['threshold_fraction'][:7],noise_initial=m['noise_initial'][:7])
    else:
        flat=a['X'][calibration][np.arange(a['X'].shape[1])[None,:]<a['lengths'][calibration,None]]
        cum=a['G'].cumsum(1);threshold=[];noise=[]
        for j in range(7):
            ids=np.flatnonzero(calibration&(a['Y'][:,j]>0));values=cum[ids,a['Y'][ids,j]-1]
            median=float(np.median(values));threshold.append(median)
            noise.append(float(np.clip(np.std(np.log(np.maximum(values,1)/median)),.03,.5)))
        m=dict(source,retained=list(range(7)),endpoints=source['endpoints'][:7],threshold_gdd=threshold,
               threshold_fraction=(np.array(threshold)/threshold[-1]).tolist(),maturation_gdd=threshold[-1],
               feature_mean=flat.mean(0).tolist(),feature_std=np.maximum(flat.std(0),1e-3).tolist(),noise_initial=noise)
    if np.any(np.diff(m['threshold_gdd'])<=0):raise ValueError('Nonincreasing calibration thresholds')
    if {'latitude','longitude'}.issubset(env.columns):m=add_context_normalizers(m,env,calibration)
    m['history_mask_geography']=True
    m['calibration_support']=counts.tolist()
    m['calibration_index_sha256']=hashlib.sha256(np.flatnonzero(calibration).astype('<i8').tobytes()).hexdigest()
    return m
