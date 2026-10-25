"""Causal point-date Transformers and a genuine MAE-criterion random forest."""
import math
import numpy as np
import torch
from torch import nn
from research.temperature_cluster_transfer.physical import interpolated_crossings as training_dates
from research.temperature_cluster_transfer.physical import stage_equal_l1 as stage_l1


class WeatherEncoder(nn.Module):
    def __init__(self,width=32,heads=4,layers=2,dropout=.1):
        super().__init__();self.width=width
        self.input=nn.Linear(18,width)
        layer=nn.TransformerEncoderLayer(width,heads,4*width,dropout,
            batch_first=True,norm_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,layers,norm=nn.LayerNorm(width),enable_nested_tensor=False)
        # PyTorch clones an encoder layer: give every layer independent initial weights.
        for block in self.encoder.layers:
            for name,p in block.named_parameters():
                if p.ndim>=2:nn.init.xavier_uniform_(p)
                elif 'weight' in name:nn.init.ones_(p)
                else:nn.init.zeros_(p)

    def forward(self,x,lengths):
        if x.ndim!=3 or x.shape[2]!=18:raise ValueError('Eighteen daily source features required')
        lengths=torch.as_tensor(lengths,device=x.device,dtype=torch.long)
        if lengths.shape!=(len(x),) or (lengths<1).any() or (lengths>x.shape[1]).any():
            raise ValueError('Valid aligned positive weather-prefix lengths required')
        n=x.shape[1];position=torch.arange(n,device=x.device,dtype=x.dtype)[:,None]
        factor=torch.exp(torch.arange(0,self.width,2,device=x.device,dtype=x.dtype)*(-math.log(10000)/self.width))
        encoding=torch.zeros(n,self.width,device=x.device,dtype=x.dtype)
        encoding[:,0::2]=torch.sin(position*factor);encoding[:,1::2]=torch.cos(position*factor)
        causal=torch.triu(torch.ones(n,n,device=x.device,dtype=torch.bool),diagonal=1)
        padding=torch.arange(n,device=x.device)[None,:]>=lengths[:,None]
        return self.encoder(self.input(x)+encoding[None],mask=causal,src_key_padding_mask=padding,is_causal=True)


class DirectTransformer(nn.Module):
    def __init__(self,initial_dates,width=32,dropout=.1):
        super().__init__();initial=np.asarray(initial_dates,float)
        gaps=np.diff(np.r_[0.,initial])
        if initial.shape!=(7,) or not np.isfinite(initial).all() or (gaps<=0).any():
            raise ValueError('Seven strictly increasing source initial dates required')
        self.weather=WeatherEncoder(width=width,dropout=dropout)
        self.rate=nn.Linear(width,1);nn.init.zeros_(self.rate.weight);nn.init.zeros_(self.rate.bias)
        self.raw_gaps=nn.Parameter(torch.tensor(np.where(gaps>20,gaps,np.log(np.expm1(gaps))),dtype=torch.float32))

    def thresholds(self):return torch.nn.functional.softplus(self.raw_gaps).cumsum(0)

    def forward(self,x,lengths):
        lengths=torch.as_tensor(lengths,device=x.device);horizon=x.shape[1]
        thresholds=self.thresholds()
        # Causality permits omitting weather AFTER every predicted stage has resolved.
        # The stopping check uses predictions, never observed target dates.
        horizons=list(dict.fromkeys([min(192,horizon),min(256,horizon),horizon])) if self.training else [horizon]
        for cut in horizons:
            h=self.weather(x[:,:cut],lengths.clamp_max(cut))
            rate=torch.exp(self.rate(h).squeeze(-1).clamp(-12,8))
            prefix=torch.arange(cut,device=x.device)[None,:]<lengths[:,None]
            progress=torch.where(prefix,rate,torch.zeros_like(rate)).cumsum(1)
            if cut==horizon or bool((progress[:,-1]>=thresholds[-1]).all()):break
        self.last_computed_days=cut
        if cut<horizon:progress=torch.cat([progress,progress[:,-1:].expand(-1,horizon-cut)],1)
        return progress,thresholds


class ResidualTransformer(nn.Module):
    def __init__(self,width=32,dropout=.1):
        super().__init__();self.weather=WeatherEncoder(width=width,dropout=dropout)
        self.stage=nn.Embedding(7,8)
        self.head=nn.Sequential(nn.Linear(width+12,width),nn.GELU(),nn.Linear(width,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,x,lengths,base):
        base=base.to(x.device,dtype=x.dtype)
        lengths=torch.as_tensor(lengths,device=x.device)
        valid=torch.isfinite(base)&(base>=1)&(base<=lengths[:,None])
        safe=torch.where(valid,base,torch.ones_like(base))
        cut=max(1,min(x.shape[1],int(safe.floor().max().item())))
        self.last_computed_days=cut
        h=self.weather(x[:,:cut],lengths.clamp_max(cut))
        index=(safe.floor().long()-1).clamp(0,cut-1)
        history=h.gather(1,index[:,:,None].expand(-1,-1,h.shape[2]))
        previous=torch.cat([torch.zeros_like(base[:,:1]),base[:,:-1]],1)
        missing=~torch.isfinite(previous);previous=torch.where(missing,torch.zeros_like(previous),previous)
        context=torch.stack([safe/180,previous/180,(safe-previous)/90,missing.to(x.dtype)],2)
        stage=self.stage(torch.arange(7,device=x.device))[None].expand(len(x),-1,-1)
        raw=self.head(torch.cat([history,context,stage],2)).squeeze(-1)
        return torch.where(valid,raw,torch.full_like(raw,float('nan')))


def balanced_l1(prediction,target,valid,counts,population):
    target=torch.as_tensor(target,device=prediction.device,dtype=prediction.dtype)
    valid=torch.as_tensor(valid,device=prediction.device,dtype=torch.bool)
    counts=torch.as_tensor(counts,device=prediction.device,dtype=prediction.dtype)
    error=torch.abs(torch.where(valid,prediction,torch.zeros_like(prediction))-
                    torch.where(valid,target,torch.zeros_like(target)))
    active=counts>0
    return (error.sum(0)/counts.clamp_min(1))[active].sum()*float(population)/(len(prediction)*int(active.sum()))


def crossing_dates(progress,thresholds,lengths):
    progress=np.asarray(progress,float);thresholds=np.asarray(thresholds,float)
    if progress.ndim!=2 or thresholds.shape!=(7,):raise ValueError('Progress/threshold axes disagree')
    hits=progress[:,:,None]>=thresholds[None,None,:]
    hits&=np.arange(1,progress.shape[1]+1)[None,:,None]<=np.asarray(lengths)[:,None,None]
    return np.where(hits.any(1),hits.argmax(1)+1,np.nan)


def make_forest(seed,min_samples_leaf,n_estimators=300):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=n_estimators,criterion='absolute_error',
        max_depth=12,min_samples_leaf=min_samples_leaf,max_features=.5,
        bootstrap=True,max_samples=.5,random_state=seed,n_jobs=2)
