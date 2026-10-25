"""Distinct native point heads with sparse, endpoint-equal supervision."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import torch
from torch import nn

from .models import WeatherEncoder,prepare_daily_features,forest_features,positional_rows


def independent_ids(frame):
    if "independent_site_year_id" in frame:return frame.independent_site_year_id.astype(str)
    site="validation_cluster_id" if "validation_cluster_id" in frame else "physical_site_id"
    return frame[site].astype(str)+":"+frame.year.astype(str)


def unit_weights(frame):
    if not len(frame):return np.empty(0)
    t=pd.DataFrame({"group":independent_ids(frame).to_numpy(),"native":frame.environment_id.astype(str).to_numpy(),
        "hybrid":frame.hybrid_id.fillna("").astype(str).to_numpy() if "hybrid_id" in frame else np.repeat("",len(frame))})
    count=t.group.nunique();native=t.groupby("group").native.transform("nunique")
    hybrid=t.groupby(["group","native"]).hybrid.transform("nunique")
    plots=t.groupby(["group","native","hybrid"]).hybrid.transform("size")
    return (1/(count*native*hybrid*plots)).to_numpy(float)


def select_heads(observations,rows,head_columns,*,min_source_environments=30,retained_heads=None,teacher_mode=False):
    columns=list(head_columns)
    if not columns or len(set(columns))!=len(columns) or not set(columns)<=set(observations):
        raise ValueError("Unique declared native head columns required")
    source=observations.iloc[positional_rows(rows,len(observations))]
    support={head:int(independent_ids(source.loc[np.isfinite(source[head])&source[head].ge(1)]).nunique()) for head in columns}
    if retained_heads is None:
        active=[head for head in columns if support[head]>=min_source_environments]
    else:
        retained=list(retained_heads)
        if len(set(retained))!=len(retained) or not set(retained)<=set(columns):raise ValueError("Frozen native heads must match candidate columns")
        if not teacher_mode and any(support[head]<min_source_environments for head in retained):
            raise ValueError("Frozen head has insufficient independent source support")
        active=[head for head in retained if support[head]>=1]
    return active,support


def head_weights(observations,rows,head_columns,mask=None):
    ids=positional_rows(rows,len(observations));frame=observations.iloc[ids]
    truth=frame[list(head_columns)].to_numpy(float);observed=np.isfinite(truth)&(truth>=1)
    if mask is not None:
        mask=np.asarray(mask,bool)
        if mask.shape!=observed.shape:raise ValueError("Head support mask is not aligned")
        observed&=mask
    count=int(observed.any(0).sum());weights=np.zeros_like(truth)
    for j in range(len(head_columns)):
        present=observed[:,j]
        if present.any():weights[present,j]=unit_weights(frame.iloc[np.flatnonzero(present)])/count
    return weights


def masked_mae(observations,rows,head_columns,prediction,*,mask=None,unresolved_penalty=366.):
    ids=positional_rows(rows,len(observations));truth=observations.iloc[ids][list(head_columns)].to_numpy(float)
    prediction=np.asarray(prediction,float)
    if prediction.shape!=truth.shape:raise ValueError("Native head prediction axes disagree")
    weights=head_weights(observations,ids,head_columns,mask)
    if not weights.any():return None
    error=np.where(np.isfinite(prediction),np.abs(np.nan_to_num(truth)-prediction),unresolved_penalty)
    return float((error*weights).sum()/weights.sum())


def _identity(value):return "" if pd.isna(value) else str(value).strip()


def fit_vocabulary(observations,rows,tier):
    if tier not in ("weather","biology"):raise ValueError("Unknown information tier")
    vocabulary={"tier":tier,"hybrids":{},"parents":{}}
    if tier=="weather":return vocabulary
    needed={"hybrid_id","parent1_id","parent2_id","identity_metadata_qualified"}
    if not needed<=set(observations):raise ValueError("Explicit qualified biological identity metadata required")
    source=observations.iloc[positional_rows(rows,len(observations))]
    source=source[source.identity_metadata_qualified.eq(True)]
    hybrids=sorted({_identity(v) for v in source.hybrid_id if _identity(v)})
    parents=sorted({_identity(v) for c in ("parent1_id","parent2_id") for v in source[c] if _identity(v)})
    if not hybrids and not parents:raise ValueError("No qualified source identity metadata")
    vocabulary.update(hybrids={v:i+1 for i,v in enumerate(hybrids)},parents={v:i+1 for i,v in enumerate(parents)})
    return vocabulary


def biological_indices(observations,vocabulary):
    values=np.zeros((len(observations),3),np.int64)
    if vocabulary["tier"]=="biology":
        qualified=observations.identity_metadata_qualified.eq(True).to_numpy()
        for j,c in enumerate(("hybrid_id","parent1_id","parent2_id")):
            mapping=vocabulary["hybrids"] if j==0 else vocabulary["parents"]
            values[:,j]=[mapping.get(_identity(v),0) if qualified[i] else 0 for i,v in enumerate(observations[c])]
    return torch.as_tensor(values,dtype=torch.long)


class BiologicalEffects(nn.Module):
    def __init__(self,vocabulary,nheads):
        super().__init__();self.parents=nn.Embedding(len(vocabulary["parents"])+1,nheads,padding_idx=0)
        self.hybrids=nn.Embedding(len(vocabulary["hybrids"])+1,nheads,padding_idx=0)
        nn.init.zeros_(self.parents.weight);nn.init.zeros_(self.hybrids.weight)
        if vocabulary["tier"]=="weather":self.parents.weight.requires_grad_(False);self.hybrids.weight.requires_grad_(False)
    def forward(self,indices):
        indices=torch.as_tensor(indices,device=self.parents.weight.device,dtype=torch.long)
        return self.hybrids(indices[:,0])+(self.parents(indices[:,1])+self.parents(indices[:,2]))/2
    def penalty(self):
        terms=[p[1:].square().mean() for p in (self.parents.weight,self.hybrids.weight) if len(p)>1]
        return sum(terms) if terms else self.parents.weight.sum()*0


class PhysicalModel(nn.Module):
    def __init__(self,kind,vocabulary,initial_demands):
        super().__init__();self.kind=kind;nheads=len(initial_demands);self.bio=BiologicalEffects(vocabulary,nheads)
        if kind not in ("t","tp") or not nheads:raise ValueError("Physical family and observed native heads required")
        self.log_demands=nn.Parameter(torch.log(torch.as_tensor(initial_demands,dtype=torch.float32).clamp_min(1)))
        self.raw_optimum=nn.Parameter(torch.tensor(.9));self.raw_maximum=nn.Parameter(torch.tensor(0.))
        self.raw_photo=nn.Parameter(torch.full((nheads,),-3.),requires_grad=kind=="tp")
    def cardinal_parameters(self):return 20+14*torch.sigmoid(self.raw_optimum),36+8*torch.sigmoid(self.raw_maximum)
    def progress(self,raw):
        temperature=raw[:,:,:2].mean(2);optimum,maximum=self.cardinal_parameters()
        alpha=math.log(2)/torch.log((maximum-8)/(optimum-8))
        u=((temperature-8).clamp_min(1e-6)/(optimum-8)).pow(alpha)
        rate=(optimum-8)*(2*u-u.square()).clamp_min(0)
        rate=torch.where((temperature>8)&(temperature<maximum),rate,torch.zeros_like(rate))[:,:,None].expand(-1,-1,len(self.log_demands))
        if self.kind=="tp":rate=rate/(1+torch.nn.functional.softplus(self.raw_photo)[None,None,:]*(raw[:,:,2,None]-12.5).clamp_min(0))
        return rate.cumsum(1)
    def demands(self,indices):return torch.exp((self.log_demands[None]+self.bio(indices)).clamp(-5,12))


def first_crossings(progress,demands,lengths):
    progress=np.asarray(progress,float);demands=np.asarray(demands,float);lengths=np.asarray(lengths,int)
    if progress.ndim!=3 or demands.shape!=(len(progress),progress.shape[2]) or lengths.shape!=(len(progress),):raise ValueError("Native head axes must align")
    valid=np.arange(progress.shape[1])[None,:,None]<lengths[:,None,None]
    hits=(progress>=demands[:,None])&valid&np.isfinite(progress)&np.isfinite(demands[:,None])
    return np.where(hits.any(1),hits.argmax(1),np.nan)


def interpolated_dates(progress,demands,lengths):
    lengths=torch.as_tensor(lengths,device=progress.device,dtype=torch.long)
    valid=torch.arange(progress.shape[1],device=progress.device)[None,:,None]<lengths[:,None,None]
    hits=(progress>=demands[:,None])&valid;resolved=hits.any(1);index=hits.long().argmax(1)
    index=torch.where(resolved,index,(lengths-1)[:,None].expand_as(index))
    current=progress.gather(1,index[:,None]).squeeze(1)
    previous=progress.gather(1,(index-1).clamp_min(0)[:,None]).squeeze(1);previous=torch.where(index==0,torch.zeros_like(previous),previous)
    crossing=(index.to(progress.dtype)+(demands-previous)/(current-previous).clamp_min(.01)-1).clamp_min(0)
    fallback=(lengths-1)[:,None]+(demands-current)/(current/lengths[:,None]).clamp_min(.1)
    return torch.where(resolved,crossing,fallback)


class DirectTransformer(nn.Module):
    def __init__(self,vocabulary,initial_dates,width=32):
        super().__init__();nheads=len(initial_dates);self.weather=WeatherEncoder(width=width,heads=4,dropout=.1)
        self.rate=nn.Linear(width,nheads);nn.init.zeros_(self.rate.weight);nn.init.constant_(self.rate.bias,.5413248546)
        self.log_demands=nn.Parameter(torch.log(torch.as_tensor(initial_dates,dtype=torch.float32).clamp_min(1)))
        self.bio=BiologicalEffects(vocabulary,nheads)
    def progress(self,x,lengths):
        rate=torch.nn.functional.softplus(self.rate(self.weather(x,lengths)))+1e-4
        valid=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]
        return torch.where(valid[:,:,None],rate,torch.zeros_like(rate)).cumsum(1)
    def demands(self,indices):return torch.exp((self.log_demands[None]+self.bio(indices)).clamp(-5,12))


class ResidualTransformer(nn.Module):
    def __init__(self,vocabulary,nheads,width=32,correction_heads=None):
        super().__init__();self.tier=vocabulary["tier"];self.weather=WeatherEncoder(width=width,heads=4,dropout=.1)
        self.bio=BiologicalEffects(vocabulary,nheads);self.stage=nn.Embedding(nheads,4)
        self.head=nn.Sequential(nn.Linear(width+9,width),nn.GELU(),nn.Linear(width,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("correction_heads",torch.as_tensor(np.ones(nheads,bool) if correction_heads is None else correction_heads,dtype=torch.bool))
    def residuals(self,x,lengths,local,base,indices,demand_context):
        valid=torch.isfinite(base)&(base>=0)&(base<lengths[local,None])&self.correction_heads[None]
        safe=torch.where(valid,base,torch.ones_like(base));cut=min(x.shape[1],max(1,int(safe.floor().max().item())+1))
        hidden=self.weather(x[:,:cut],lengths.clamp_max(cut));query=safe.floor().long().clamp(0,cut-1)
        history=hidden[local[:,None],query];bio=self.bio(indices)[:,:,None]
        # Weather-only requirements are curve-dependent nuisance scales.
        context=torch.zeros_like(base) if self.tier=="weather" else demand_context
        stage=self.stage(torch.arange(base.shape[1],device=x.device))[None].expand(len(base),-1,-1)
        values=torch.cat([history,safe[:,:,None]/180,bio,context[:,:,None],(~valid)[:,:,None].to(x.dtype),torch.ones_like(bio),stage],2)
        return torch.where(valid,self.head(values).squeeze(-1),torch.zeros_like(base))


def residual_features(observations,env,arrays,rows,base,demand_context,tier):
    context=np.zeros_like(base) if tier=="weather" else np.asarray(demand_context)*1000
    return forest_features(observations,env,arrays,rows,base,context)


def model_from_artifact(artifact):
    kind=artifact["kind"];vocabulary=artifact["vocabulary"]
    if kind in ("t","tp"):model=PhysicalModel(kind,vocabulary,artifact["initial_demands"])
    elif kind=="transformer":model=DirectTransformer(vocabulary,artifact["initial_demands"],artifact["width"])
    elif kind=="residual_transformer":model=ResidualTransformer(vocabulary,len(artifact["active_heads"]),artifact["width"],artifact["correction_heads"])
    else:raise ValueError("Forest artifacts do not have a torch module")
    model.load_state_dict(artifact["state_dict"]);model.eval();return model
