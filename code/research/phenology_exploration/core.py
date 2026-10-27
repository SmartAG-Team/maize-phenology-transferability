"""Numerical operations with explicit native-day and population definitions."""
import hashlib
import numpy as np
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def crossing(progress,threshold,lengths):
    p=np.asarray(progress);n,t=p.shape
    d=np.broadcast_to(np.asarray(threshold,float),(n,))
    lengths=np.asarray(lengths,int)
    hit=(p>=d[:,None])&(np.arange(t)[None,:]<lengths[:,None])
    found=hit.any(1)&np.isfinite(d)&(d>0)&(lengths>0)
    ix=hit.argmax(1);rows=np.arange(n)
    previous=np.where(ix>0,p[rows,np.maximum(ix-1,0)],0.)
    rate=p[rows,ix]-previous
    result=np.maximum(0.,ix-1+(d-previous)/np.maximum(rate,1e-8))
    return np.where(found,result,np.nan)


def phase_progress(rate,anchor):
    rate=np.asarray(rate);anchor=np.asarray(anchor,float)
    c=rate.cumsum(1);n,t=c.shape
    safe=np.nan_to_num(anchor,nan=t).clip(0,t-1)
    left=np.floor(safe).astype(int);right=np.minimum(left+1,t-1)
    rows=np.arange(n);fraction=safe-left
    at_anchor=c[rows,left]*(1-fraction)+c[rows,right]*fraction
    result=np.maximum(0.,c-at_anchor[:,None])
    result[~np.isfinite(anchor)]=0.
    return result


def thermal_rate(raw,alpha=1.):
    z=np.clip(np.asarray(raw)[:,:,:2],8,30)-8
    return (22*np.power(z/22,float(alpha))).mean(2)


def hierarchical_weights(frame):
    if len(frame)==0:return np.empty(0)
    a=frame.groupby('head').site.transform('nunique')
    b=frame.groupby(['head','site']).site_year.transform('nunique')
    c=frame.groupby(['head','site_year']).site_year.transform('size')
    w=(1/(frame['head'].nunique()*a*b*c)).to_numpy(float)
    return w/w.sum()


def macro(frame,values):
    q=frame[['head','site','site_year']].copy();q['value']=np.asarray(values)
    a=q.groupby(['head','site','site_year']).value.mean()
    return float(a.groupby(level=['head','site']).mean().groupby(level='head').mean().mean())


def valid_prediction(frame,prediction,horizon=None):
    pred=np.asarray(prediction,float)
    if horizon is None:horizon=frame.horizon.to_numpy() if 'horizon' in frame else 367.
    return np.isfinite(pred)&(pred>=0)&(pred<np.asarray(horizon))


def metric(frame,prediction,horizon=None):
    pred=np.asarray(prediction,float);ok=valid_prediction(frame,pred,horizon)
    err=np.abs(pred-frame.observed.to_numpy())
    return dict(penalized_mae=macro(frame,np.where(ok,err,366.)),
                resolved_mae=macro(frame.loc[ok],err[ok]) if ok.any() else None,
                completion=float(ok.mean()),labels=len(frame),sites=frame.site.nunique())


def weighted_median(value,weight):
    order=np.argsort(value);v=np.asarray(value)[order];w=np.asarray(weight)[order]
    return float(v[min(len(v)-1,np.searchsorted(np.cumsum(w/w.sum()),.5))])
