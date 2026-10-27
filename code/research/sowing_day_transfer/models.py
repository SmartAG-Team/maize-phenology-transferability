"""Parallel flowering models with source-fitted biological demand effects.

Weather is encoded once per native environment, then queried by its plots.
Neither the biological vocabulary nor normalizers need evaluation outcomes.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from research.unified_transfer.models import WeatherEncoder


ENDPOINTS = ("anthesis_das", "silking_das")
FEATURE_NAMES = ("Tmin", "Tmax", "daylength", "mean_temperature", "diurnal_range", "GDD8_30",
    "heat_above30", "elapsed_fraction", "cum_GDD", "cum_temperature", "cum_daylength",
    "past7_GDD", "past30_GDD", "past7_temperature", "past30_temperature", "reserved_zero", "sowing_sine", "sowing_cosine")


def positional_rows(rows, population):
    values = np.asarray(rows)
    if values.dtype == bool:
        if values.shape != (population,):
            raise ValueError("Boolean row mask is not aligned")
        values = np.flatnonzero(values)
    values = values.astype(np.int64)
    if values.ndim != 1 or (values < 0).any() or (values >= population).any() or len(np.unique(values)) != len(values):
        raise ValueError("Unique valid positional rows are required")
    return values


def row_weights(observations, rows=None):
    """Site-year → native weather/sowing environment → hybrid → native plot."""
    ids = np.arange(len(observations)) if rows is None else positional_rows(rows,len(observations))
    frame = observations.iloc[ids].copy().reset_index(drop=True)
    if frame.empty:
        return np.empty(0)
    group = frame.physical_site_id.astype(str) + ":" + frame.year.astype(str)
    native = frame.environment_id.astype(str) if "environment_id" in frame else group
    hybrid = frame.hybrid_id.fillna("").astype(str)
    keyed = pd.DataFrame({"group":group,"native":native,"hybrid":hybrid})
    group_count = keyed.group.nunique()
    native_count = keyed.groupby("group").native.transform("nunique")
    hybrid_count = keyed.groupby(["group","native"]).hybrid.transform("nunique")
    plot_count = keyed.groupby(["group","native","hybrid"]).hybrid.transform("size")
    return (1. / (group_count*native_count*hybrid_count*plot_count)).to_numpy()


def weighted_mae(target, prediction, weights, unresolved_penalty=366.):
    target,prediction=np.asarray(target,float),np.asarray(prediction,float)
    weights=np.asarray(weights,float)
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 2 or weights.shape != (len(target),):
        raise ValueError("Aligned paired targets, predictions and row weights required")
    if not len(target):
        return None
    if not np.isfinite(target).all() or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Finite paired targets and positive weight required")
    errors=np.where(np.isfinite(prediction),np.abs(target-prediction),float(unresolved_penalty))
    return float(np.sum(errors*weights[:,None])/(2*weights.sum()))


def grouped_mae(observations, rows, prediction, unresolved_penalty=366.):
    ids=positional_rows(rows,len(observations))
    return weighted_mae(observations.iloc[ids][list(ENDPOINTS)].to_numpy(float),prediction,
        row_weights(observations,ids),unresolved_penalty)


def _identity(value):
    return "" if pd.isna(value) else str(value).strip()


def fit_vocabulary(observations, rows, tier):
    if tier not in ("weather","biology"):
        raise ValueError("Information tier must be weather or biology")
    source=observations.iloc[positional_rows(rows,len(observations))]
    hybrids=sorted({_identity(v) for v in source.hybrid_id if _identity(v)}) if tier=="biology" else []
    parents=sorted({_identity(v) for column in ("parent1_id","parent2_id") for v in source[column] if _identity(v)}) if tier=="biology" else []
    return {"tier":tier,"hybrids":{v:i+1 for i,v in enumerate(hybrids)},"parents":{v:i+1 for i,v in enumerate(parents)}}


def biological_indices(observations,vocabulary):
    values=np.zeros((len(observations),3),dtype=np.int64)
    if vocabulary["tier"]=="biology":
        for j,column in enumerate(("hybrid_id","parent1_id","parent2_id")):
            mapping=vocabulary["hybrids"] if j==0 else vocabulary["parents"]
            values[:,j]=[mapping.get(_identity(v),0) for v in observations[column]]
    return torch.as_tensor(values,dtype=torch.long)


class BiologicalEffects(nn.Module):
    """Symmetric parental and hybrid log-demand effects; unknown row zero."""
    def __init__(self,vocabulary):
        super().__init__();self.vocabulary=vocabulary
        self.parents=nn.Embedding(len(vocabulary["parents"])+1,2,padding_idx=0)
        self.hybrids=nn.Embedding(len(vocabulary["hybrids"])+1,2,padding_idx=0)
        nn.init.zeros_(self.parents.weight);nn.init.zeros_(self.hybrids.weight)
        if vocabulary["tier"]=="weather":
            self.parents.weight.requires_grad_(False);self.hybrids.weight.requires_grad_(False)

    def forward(self,indices):
        indices=torch.as_tensor(indices,device=self.parents.weight.device,dtype=torch.long)
        return self.hybrids(indices[:,0])+(self.parents(indices[:,1])+self.parents(indices[:,2]))/2

    def penalty(self):
        terms=[p[1:].square().mean() for p in (self.parents.weight,self.hybrids.weight) if len(p)>1]
        return sum(terms) if terms else self.parents.weight.sum()*0


class PhysicalModel(nn.Module):
    """Fixed base 8°C, bounded cardinal response and two independent demands."""
    def __init__(self,kind,vocabulary,initial_demands):
        super().__init__()
        if kind not in ("t","tp"):
            raise ValueError("Physical family must be t or tp")
        self.kind=kind;self.bio=BiologicalEffects(vocabulary)
        self.log_demands=nn.Parameter(torch.log(torch.as_tensor(initial_demands,dtype=torch.float32).clamp_min(1)))
        self.raw_optimum=nn.Parameter(torch.tensor(.9))
        self.raw_maximum=nn.Parameter(torch.tensor(0.))
        self.raw_photo=nn.Parameter(torch.tensor([-3.,-3.]),requires_grad=kind=="tp")

    def cardinal_parameters(self):
        return 20+14*torch.sigmoid(self.raw_optimum),36+8*torch.sigmoid(self.raw_maximum)

    def progress(self,raw):
        temperature=raw[:,:,:2].mean(2)
        optimum,maximum=self.cardinal_parameters()
        alpha=math.log(2)/torch.log((maximum-8)/(optimum-8))
        u=((temperature-8).clamp_min(1e-6)/(optimum-8)).pow(alpha)
        rate=(optimum-8)*(2*u-u.square()).clamp_min(0)
        rate=torch.where((temperature>8)&(temperature<maximum),rate,torch.zeros_like(rate))
        rate=rate[:,:,None].expand(-1,-1,2)
        if self.kind=="tp":
            photo=torch.nn.functional.softplus(self.raw_photo)
            rate=rate/(1+photo[None,None,:]*(raw[:,:,2,None]-12.5).clamp_min(0))
        return rate.cumsum(1)

    def demands(self,indices):
        return torch.exp((self.log_demands[None]+self.bio(indices)).clamp(-5,12))


def first_crossings(progress,demands,lengths):
    progress=np.asarray(progress,float);demands=np.asarray(demands,float);lengths=np.asarray(lengths,int)
    if progress.ndim!=3 or progress.shape[2]!=2 or demands.shape!=(len(progress),2) or lengths.shape!=(len(progress),):
        raise ValueError("Paired progress, demands and native lengths must align")
    valid=np.arange(progress.shape[1])[None,:,None]<lengths[:,None,None]
    hits=(progress>=demands[:,None,:])&valid&np.isfinite(progress)&np.isfinite(demands[:,None,:])
    return np.where(hits.any(1),hits.argmax(1),np.nan)


def interpolated_dates(progress,demands,lengths):
    """Differentiable within-day crossing; extrapolation is training-only."""
    lengths=torch.as_tensor(lengths,device=progress.device,dtype=torch.long)
    if (lengths<1).any():raise ValueError("Positive native weather prefixes required")
    valid=torch.arange(progress.shape[1],device=progress.device)[None,:,None]<lengths[:,None,None]
    hits=(progress>=demands[:,None,:])&valid
    index=hits.to(torch.int64).argmax(1)
    resolved=hits.any(1)
    index=torch.where(resolved,index,(lengths-1)[:,None].expand(-1,2))
    current=progress.gather(1,index[:,None,:]).squeeze(1)
    previous=progress.gather(1,(index-1).clamp_min(0)[:,None,:]).squeeze(1)
    previous=torch.where(index==0,torch.zeros_like(previous),previous)
    fraction=(demands-previous)/(current-previous).clamp_min(.01)
    crossing=(index.to(progress.dtype)+fraction-1).clamp_min(0)
    average=current/lengths[:,None].clamp_min(1)
    fallback=(lengths-1)[:,None]+(demands-current)/average.clamp_min(.1)
    return torch.where(resolved,crossing,fallback)


def _raw_daily_features(env,arrays):
    raw=np.asarray(arrays["X"],float);lengths=np.asarray(arrays["lengths"],int)
    if raw.ndim!=3 or raw.shape[2]!=3 or lengths.shape!=(len(raw),):
        raise ValueError("Native weather arrays must have shape [environment,day,3]")
    features=np.zeros((*raw.shape[:2],18),np.float32)
    if env.weather_index.duplicated().any():raise ValueError("Environment weather indices are not unique")
    nday=raw.shape[1]
    for _,row in env.iterrows():
        i=int(row.weather_index);n=int(lengths[i])
        if n<1 or n>nday:raise ValueError("Invalid native weather prefix length")
        low,high,photo=np.nan_to_num(raw[i,:n],nan=0).T
        mean=(low+high)/2;gdd=(np.clip(low,8,30)+np.clip(high,8,30))/2-8
        def rolling(values,window):
            cumulative=np.r_[0.,np.cumsum(values)];end=np.arange(1,n+1);start=np.maximum(0,end-window)
            return (cumulative[end]-cumulative[start])/(end-start)
        sowing=pd.Timestamp(row.sowing_date);phase=2*np.pi*sowing.dayofyear/365.25
        if float(row.latitude)<0:phase+=np.pi
        features[i,:n]=np.column_stack([low,high,photo,mean,high-low,gdd,np.maximum(high-30,0),
            np.arange(n)/366,np.cumsum(gdd)/1000,np.cumsum(mean)/1000,np.cumsum(photo)/1000,
            rolling(gdd,7),rolling(gdd,30),rolling(mean,7),rolling(mean,30),
            np.zeros(n),np.full(n,np.sin(phase)),np.full(n,np.cos(phase))])
    return features


def prepare_daily_features(env,arrays,fit_environment_indices=None,normalizer=None):
    raw=_raw_daily_features(env,arrays);lengths=np.asarray(arrays["lengths"],int)
    if normalizer is None:
        ids=np.unique(np.asarray(fit_environment_indices,dtype=int))
        if not len(ids):raise ValueError("Source weather is required for normalization")
        means=np.stack([raw[i,:lengths[i]].mean(0) for i in ids])
        second=np.stack([(raw[i,:lengths[i]]**2).mean(0) for i in ids])
        mean=means.mean(0);std=np.sqrt(np.maximum(second.mean(0)-mean**2,0)).clip(.05)
        normalizer={"mean":mean.tolist(),"std":std.tolist(),"fit_environment_indices":ids.tolist(),"weather_repetition":"one_native_environment"}
    mean,std=np.asarray(normalizer["mean"]),np.asarray(normalizer["std"])
    features=((raw-mean)/std).astype(np.float32)
    features[np.arange(raw.shape[1])[None,:]>=lengths[:,None]]=0.
    return features,normalizer


class DirectTransformer(nn.Module):
    def __init__(self,vocabulary,initial_dates,width=32):
        super().__init__();self.weather=WeatherEncoder(width=width,heads=4,dropout=.1)
        self.rate=nn.Linear(width,2);nn.init.zeros_(self.rate.weight);nn.init.constant_(self.rate.bias,.5413248546)
        self.log_demands=nn.Parameter(torch.log(torch.as_tensor(initial_dates,dtype=torch.float32).clamp_min(1)))
        self.bio=BiologicalEffects(vocabulary)

    def progress(self,x,lengths):
        h=self.weather(x,lengths)
        rate=torch.nn.functional.softplus(self.rate(h))+1e-4
        valid=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]
        return torch.where(valid[:,:,None],rate,torch.zeros_like(rate)).cumsum(1)

    def demands(self,indices):
        return torch.exp((self.log_demands[None]+self.bio(indices)).clamp(-5,12))


class ResidualTransformer(nn.Module):
    def __init__(self,vocabulary,width=32):
        super().__init__();self.weather=WeatherEncoder(width=width,heads=4,dropout=.1)
        self.bio=BiologicalEffects(vocabulary);self.stage=nn.Embedding(2,4)
        self.head=nn.Sequential(nn.Linear(width+9,width),nn.GELU(),nn.Linear(width,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def residuals(self,x,lengths,local_environments,base,indices,demand_context):
        valid=torch.isfinite(base)&(base>=0)&(base<lengths[local_environments,None])
        safe=torch.where(valid,base,torch.ones_like(base))
        cut=min(x.shape[1],max(1,int(safe.floor().max().item())+1))
        hidden=self.weather(x[:,:cut],lengths.clamp_max(cut))
        query=safe.floor().long().clamp(0,cut-1)
        history=hidden[local_environments[:,None],query]
        biological=self.bio(indices)[:,:,None]
        demand=demand_context[:,:,None]/1000
        stage=self.stage(torch.arange(2,device=x.device))[None].expand(len(base),-1,-1)
        context=torch.cat([safe[:,:,None]/180,biological,demand,(~valid)[:,:,None].to(x.dtype),torch.ones_like(safe[:,:,None]),stage],2)
        value=self.head(torch.cat([history,context],2)).squeeze(-1)
        return torch.where(valid,value,torch.zeros_like(value))


def model_from_artifact(artifact):
    kind=artifact["kind"]
    if kind in ("t","tp"):
        model=PhysicalModel(kind,artifact["vocabulary"],artifact["initial_demands"])
    elif kind=="transformer":
        model=DirectTransformer(artifact["vocabulary"],artifact["initial_demands"],artifact["width"])
    elif kind=="residual_transformer":
        model=ResidualTransformer(artifact["vocabulary"],artifact["width"])
    else:raise ValueError("A forest has no torch module")
    model.load_state_dict(artifact["state_dict"]);model.eval()
    return model


def forest_features(observations,env,arrays,rows,base,demands):
    """Only source-fitted own-T–P prefixes and prediction-available context."""
    ids=positional_rows(rows,len(observations));selected=observations.iloc[ids]
    raw=np.asarray(arrays["X"],float);lengths=np.asarray(arrays["lengths"],int)
    wi=selected.weather_index.to_numpy(int)
    valid=np.isfinite(base)
    n=np.clip(np.floor(np.where(valid,base,0)).astype(int)+1,1,lengths[wi,None])
    low,high,photo=np.moveaxis(np.nan_to_num(raw,nan=0),2,0)
    mean=(low+high)/2;gdd=(np.clip(low,8,30)+np.clip(high,8,30))/2-8
    daily=np.stack([low,high,photo,mean,gdd,np.maximum(high-30,0)],axis=2)
    cumulative=np.concatenate([np.zeros((len(raw),1,6)),np.cumsum(daily,axis=1)],axis=1)
    total=cumulative[wi[:,None],n]
    averages=total/n[:,:,None]
    previous=cumulative[wi[:,None],np.maximum(0,n-7)]
    trailing=(total-previous)/np.minimum(n,7)[:,:,None]
    phase=2*np.pi*pd.to_datetime(selected.sowing_date).dt.dayofyear.to_numpy()/365.25
    latitudes=env.set_index("weather_index").latitude.reindex(wi).to_numpy(float)
    phase=phase+np.where(latitudes<0,np.pi,0)
    output=np.stack([base/180,demands/1000,averages[:,:,0],averages[:,:,1],averages[:,:,2],averages[:,:,3],
        total[:,:,4]/1000,total[:,:,5]/100,trailing[:,:,3],trailing[:,:,2],
        np.broadcast_to(np.sin(phase)[:,None],base.shape),np.broadcast_to(np.cos(phase)[:,None],base.shape),np.zeros_like(base)],axis=2)
    return np.where(valid[:,:,None],output,0).astype(np.float32)
