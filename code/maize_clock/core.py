"""Physical conventions and discrete event probability utilities."""
import numpy as np

def gdd(low, high, base=8., cap=30.):
    return (np.clip(low,base,cap)+np.clip(high,base,cap))/2-base

def daylength(doy, latitude):
    # Astronomical sunrise-to-sunset, geometric horizon; polar clipping explicit.
    decl=.409*np.sin(2*np.pi*np.asarray(doy)/365.-1.39)
    angle=np.arccos(np.clip(-np.tan(np.deg2rad(latitude))*np.tan(decl),-1.,1.))
    return 24.*angle/np.pi

def valid_prefix(low, high):
    low,high=np.asarray(low),np.asarray(high)
    ok=np.isfinite(low)&np.isfinite(high)&(low<=high)&(low>=-80)&(high<=65)
    bad=np.flatnonzero(~ok)
    return int(bad[0]) if len(bad) else len(low)

def partition(site_partition, year):
    if year<1991 or year>2025:return 'supplement'
    heldout=site_partition not in ('training','train')
    if heldout:return 'test_spatial' if year<=2020 else 'test_joint'
    return 'train' if year<=2017 else ('validation' if year<=2020 else 'test_temporal')

def event_distribution(cdf):
    cdf=np.asarray(cdf,dtype=float)
    if cdf.ndim!=1 or not len(cdf) or np.any(np.diff(cdf)<-1e-8) or cdf.min()<0 or cdf.max()>1:
        raise ValueError('CDF must be a nonempty increasing probability vector')
    return np.diff(np.r_[0.,cdf]),float(1.-cdf[-1])

def quantile_day(cdf,q):
    found=np.flatnonzero(np.asarray(cdf)>=q)
    return int(found[0]+1) if len(found) else None

def crps(cdf,observed_day):
    cdf=np.asarray(cdf)
    return float(np.sum((cdf-(np.arange(1,len(cdf)+1)>=observed_day))**2))
