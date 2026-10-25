"""Matched causal thermal Transformers with/without observable local context."""
import argparse
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from research.europe_climate_transfer.joint_data import weekly_features
from research.sowing_day_transfer.multistage_models import interpolated_dates
from .data import OUT,load
from .models import fit_thermal
from .core import thermal_rate,crossing,hierarchical_weights,metric,sha
from .experiments import row_predictions
from .analysis import site_errors,block_interval
from .provenance import code_signature,validate_receipt

DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
NEURAL_FOLDER='paired_causal_neural'


class ThermalTransformer(nn.Module):
    def __init__(self,thresholds,n_features):
        super().__init__();d=np.asarray(thresholds);gaps=np.diff(np.r_[0.,d])
        if (gaps<0).any() or (d<=0).any():raise ValueError('Source thermal thresholds must be nondecreasing')
        self.log_demands=nn.Parameter(torch.tensor(np.log(d),dtype=torch.float32))
        self.input=nn.Linear(n_features,32)
        layer=nn.TransformerEncoderLayer(32,4,64,dropout=.1,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.rate=nn.Linear(32,1);nn.init.zeros_(self.rate.weight);nn.init.zeros_(self.rate.bias)
    def demands(self):return torch.cummax(self.log_demands.clamp(-5,12).exp(),dim=0).values
    def progress(self,features,rate,lengths):
        n=features.shape[1];position=torch.arange(n,device=features.device,dtype=features.dtype)[:,None]
        factor=torch.exp(torch.arange(0,32,2,device=features.device)*(-math.log(10000)/32))
        enc=torch.zeros(n,32,device=features.device)
        enc[:,0::2]=torch.sin(position*factor);enc[:,1::2]=torch.cos(position*factor)
        causal=torch.triu(torch.ones(n,n,device=features.device,dtype=torch.bool),diagonal=1)
        pad=torch.arange(n,device=features.device)[None,:]*7>=lengths[:,None]
        hidden=self.encoder(self.input(features)+enc[None],mask=causal,src_key_padding_mask=pad,is_causal=True)
        adjustment=self.rate(hidden).squeeze(2).clamp(-3,3).exp()
        dayix=torch.div(torch.arange(rate.shape[1],device=features.device),7,rounding_mode='floor')
        valid=torch.arange(rate.shape[1],device=features.device)[None,:]<lengths[:,None]
        return torch.where(valid,rate*adjustment[:,dayix],torch.zeros_like(rate)).cumsum(1)


def feature_normalizer(features,lengths):
    valid=np.arange(features.shape[1])[None,:]*7<lengths[:,None]
    mean=features[valid].mean(0);scale=features[valid].std(0).clip(.05)
    return dict(mean=mean,scale=scale)


def features(data,ids,context):
    x=weekly_features(data.raw[ids])
    z=np.broadcast_to(data.context[ids,None,:],(len(ids),x.shape[1],5)) if context else np.zeros((len(ids),x.shape[1],5),np.float32)
    x=np.concatenate([x,z],2)
    return x.astype(np.float32)


def predict_network(model,data,ids,norm,context,train=False,device=DEVICE):
    model.eval();output=[];lengths=(data.train_lengths if train else data.lengths)[ids]
    with torch.no_grad():
        for start in range(0,len(ids),128):
            ii=ids[start:start+128];length=lengths[start:start+128]
            x=(features(data,ii,context)-norm['mean'])/norm['scale']
            p=model.progress(torch.tensor(x,device=device),torch.tensor(thermal_rate(data.raw[ii]),device=device),
                             torch.tensor(length,device=device)).cpu().numpy()
            d=model.demands().cpu().numpy()
            output.append(np.column_stack([crossing(p,value,length) for value in d]))
    return np.concatenate(output)


def fit_network(data,source,initial,context,seed=17,tune=None,fixed_steps=None):
    torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    ids=np.sort(source.u.unique());lookup={u:i for i,u in enumerate(ids)}
    length=data.train_lengths[ids];f=features(data,ids,context);norm=feature_normalizer(f,length)
    x=torch.tensor((f-norm['mean'])/norm['scale'],device=DEVICE)
    r=torch.tensor(thermal_rate(data.raw[ids]),device=DEVICE);n=torch.tensor(length,device=DEVICE)
    model=ThermalTransformer(initial,23).to(DEVICE)
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    weights=hierarchical_weights(source);best=None;best_score=float('inf');best_step=0;history=[]
    maximum=320 if fixed_steps is None else fixed_steps
    for step in range(maximum+1):
        if step:
            q=source.iloc[rng.choice(len(source),256,replace=True,p=weights)]
            index=np.array([lookup[u] for u in q.u]);unique,local=np.unique(index,return_inverse=True)
            model.train();progress=model.progress(x[unique],r[unique],n[unique])
            demands=model.demands()[None].expand(len(unique),-1)
            dates=interpolated_dates(progress[:,:,None].expand(-1,-1,7),demands,n[unique])[local]
            values=dates.gather(1,torch.tensor(q['head'].to_numpy(int),device=DEVICE)[:,None]).squeeze(1)
            loss=(values-torch.tensor(q.observed.to_numpy(np.float32),device=DEVICE)).abs().mean()/10
            if not torch.isfinite(loss):raise ValueError('Nonfinite neural training loss')
            optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optim.step()
        if tune is None:
            if step==maximum:best=copy.deepcopy(model.state_dict());best_step=step
        elif step%40==0:
            ti=tune.u.unique();pred=predict_network(model,data,ti,norm,context,train=True)
            score=metric(tune,row_predictions(tune,ti,pred),data.train_lengths[tune.u])['penalized_mae']
            history.append(dict(step=step,mae=score))
            if score<best_score-1e-8:best_score=score;best_step=step;best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best);model.eval()
    return model,norm,best_step,history


def run(limit=None):
    data=load();folder=OUT/NEURAL_FOLDER;folder.mkdir(exist_ok=True)
    design=dict(source_cases='all 80-site primary random/coverage cases; three draws and five folds',
        models=['thermal_transformer','context_thermal_transformer'],encoder='two causal layers, width32, heads4, feedforward64',
        initial_predictor='exactly the source-fitted T model; epoch zero included in selection',
        progress='daily clipped GDD times a positive causal neural correction; shared increasing thermal thresholds',
        features='past-only weather summaries; optional source-known reference climate and sowing/daylength context',
        ablation='identical 23-input architecture and seeded initial weights; five context channels are zero in the ablated model',
        checkpoints=list(range(0,321,40)),batch_labels=256,optimizer_seed=17,
        training_cutoff='actual daily prefixes through 2010-12-31',device=DEVICE,
        status='exploratory matched-budget neural extension; prior target inspection acknowledged')
    dp=folder/'design.json'
    if dp.exists() and json.loads(dp.read_text())!=design:raise ValueError('Neural design changed')
    dp.write_text(json.dumps(design,indent=2)+'\n')
    cases=[c for c in json.loads((OUT/'cases.json').read_text())['cases'] if c['n_sites']==80 and c['source_ez'] is None]
    if limit:cases=cases[:limit]
    code_snapshot=code_signature('neural.py')
    for case in cases:
        name=case['case'];dest=folder/name;dest.mkdir(exist_ok=True)
        signature=dict(code=code_snapshot,inputs=data.protected,case=case,
                       parent_case_sha256=sha(OUT/'cases'/name/'complete.json'),
                       membership_sha256=sha(OUT/'memberships'/f'{name}.csv.gz'),design=design)
        if (dest/'complete.json').exists():
            receipt=json.loads((dest/'complete.json').read_text())
            validate_receipt(receipt,signature,dest)
            continue
        start=time.monotonic();sample=pd.read_csv(OUT/'memberships'/f'{name}.csv.gz')
        source=data.labels[data.labels.unit.isin(sample.unit)&data.labels.pre2011_prefix_eligible]
        inner=sample.set_index('unit').inner;train=source[source.unit.map(inner).ne(2)];tune=source[source.unit.map(inner).eq(2)]
        original=pd.read_csv(OUT/'cases'/name/'predictions.csv.gz')
        target=data.labels.set_index('label_id').loc[original.label_id].reset_index();ids=target.u.unique()
        train_base=fit_thermal(data,train);final_base=fit_thermal(data,source);trials={};selected={}
        exported=original[['label_id','unit','u','site','site_year','block','year','ez','head','observed','case','horizon','T']].copy()
        for context in [False,True]:
            kind='context_thermal_transformer' if context else 'thermal_transformer'
            model,norm,steps,history=fit_network(data,train,train_base['thresholds'],context,tune=tune)
            trials[kind]=history;selected[kind]=steps
            model,norm,_,_=fit_network(data,source,final_base['thresholds'],context,fixed_steps=steps)
            pred=predict_network(model,data,ids,norm,context)
            exported[kind]=row_predictions(target,ids,pred)
            state={k:v.cpu() for k,v in model.state_dict().items()}
            torch.save(dict(state=state,norm=norm,context=context,initial=final_base['thresholds'],selected_steps=steps),dest/f'{kind}.pt')
        exported.to_csv(dest/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        receipt=dict(case=case,signature=signature,selection_trials=trials,selected_steps=selected,
            source_case_sha256=sha(OUT/'cases'/name/'complete.json'),output_sha256=sha(dest/'predictions.csv.gz'),
            elapsed_seconds=time.monotonic()-start,source_and_target_sites_disjoint=not bool(set(source.site)&set(target.site)),
            outputs={p.name:sha(p) for p in dest.iterdir() if p.name!='complete.json'})
        (dest/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(dict(case=name,steps=selected,seconds=round(receipt['elapsed_seconds'],1))),flush=True)
    if code_signature('neural.py')!=code_snapshot:raise ValueError('Neural dependencies changed during execution')
    print(json.dumps(dict(completed_requested_cases=len(cases),protected_inputs_unchanged=all(sha(p)==d for p,d in data.protected.items()))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args();run(args.limit)
