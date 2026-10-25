"""Calendar-complete climate summaries and paired transfer contrasts."""
import calendar
import numpy as np
import pandas as pd


def annual_temperature(dates,tmin,tmax,years):
    d=pd.DatetimeIndex(dates);a=np.asarray(tmin,float);b=np.asarray(tmax,float)
    if d.duplicated().any():raise ValueError('Duplicate weather dates')
    if len(d)!=len(a) or len(d)!=len(b):raise ValueError('Weather alignment mismatch')
    valid=np.isfinite(a)&np.isfinite(b)&(a<=b)&(a>=-80)&(b<=65)
    means=[]
    for year in years:
        expected=pd.date_range(f'{year}-01-01',f'{year}-12-31');index=np.where(d.year==year)[0]
        complete=d[index].sort_values().equals(expected) and valid[index].all()
        if complete:means.append(float(((a[index]+b[index])/2).mean()))
    complete=len(means)==len(years)
    return dict(complete=complete,annual_mean_c=float(np.mean(means)) if complete else np.nan,
        days=len(d),years=len(means),invalid_days=int((~valid).sum()))


def temperature_zones(values,n=5):
    x=np.asarray(values,float)
    if not np.isfinite(x).all() or len(np.unique(x))<n:raise ValueError('Insufficient distinct reference temperatures')
    edges=np.quantile(x,np.arange(1,n)/n)
    if (np.diff(edges)<=0).any():raise ValueError('Insufficient distinct quantile boundaries')
    return np.searchsorted(edges,x,side='right'),edges


def temperature_distance(source,target):
    s=np.asarray(source,float);t=np.asarray(target,float)
    if not len(s) or not np.isfinite(s).all() or not np.isfinite(t).all():raise ValueError('Finite climate support required')
    return dict(signed=t-np.median(s),outside=np.maximum(np.maximum(s.min()-t,t-s.max()),0.))


def transfer_penalties(frame):
    keys=['model','draw','fold','target_zone','site','head']
    diagonal=frame[frame.source_zone.eq(frame.target_zone)][keys+['error']].rename(columns={'error':'diagonal_error'})
    if diagonal.duplicated(keys).any():raise ValueError('Nonunique diagonal target population')
    out=frame.merge(diagonal,on=keys,how='left',validate='many_to_one')
    out['diagonal_available']=out.diagonal_error.notna();out['penalty']=out.error-out.diagonal_error
    return out
