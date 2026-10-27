"""Source-only physical/context fits and spatially cross-fitted corrections."""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from .core import crossing,phase_progress,thermal_rate,hierarchical_weights,weighted_median

MIN_SUPPORT=30


def validate_source(data,labels):
    if (labels.observed.to_numpy()>=data.train_lengths[labels.u]).any():
        raise ValueError('A fitted label lies outside its source weather prefix')


def _calibrate(progress,lengths,target,weight):
    rows=np.arange(len(target));ix=np.asarray(target,int).clip(0,progress.shape[1]-1)
    initial=weighted_median(progress[rows,ix],weight)
    last=progress[rows,np.maximum(lengths-1,0)]
    def loss(d):
        value=crossing(progress,d,lengths)
        fallback=lengths-1+(d-last)/np.maximum(last/np.maximum(lengths,1),.1)
        return float(np.dot(weight,abs(np.where(np.isfinite(value),value,fallback)-target)))
    lower=max(.1,initial*.35);upper=max(lower+.1,initial*2)
    grid=np.linspace(lower,upper,25);values=np.array([loss(x) for x in grid])
    k=int(values.argmin());lo=grid[max(0,k-1)];hi=grid[min(len(grid)-1,k+1)]
    result=minimize_scalar(loss,bounds=(lo,hi),method='bounded',options={'xatol':.01})
    return min([(initial,loss(initial)),(float(result.x),float(result.fun))],key=lambda a:a[1])[0]


def fit_calendar(data,labels):
    validate_source(data,labels)
    days=[];support=[]
    for h in range(7):
        q=labels[labels['head'].eq(h)];n=q.u.nunique();support.append(n)
        days.append(weighted_median(q.observed,hierarchical_weights(q)) if n>=MIN_SUPPORT else np.nan)
    return dict(kind='calendar',days=days,support=support)


def fit_thermal(data,labels,alpha=1.,phase=False,base=None):
    validate_source(data,labels)
    ids=np.sort(labels.u.unique());lookup={u:i for i,u in enumerate(ids)}
    rate=thermal_rate(data.raw[ids],alpha)
    if phase:
        if base is None:raise ValueError('A source-fitted anchor model is required')
        anchor=predict(base,data,ids,train=True)[:,3]
        progress=phase_progress(rate,anchor)
    else:progress=rate.cumsum(1)
    thresholds=[];support=[]
    for h in range(7):
        q=labels[labels['head'].eq(h)];n=q.u.nunique();support.append(n)
        if phase and h<4:thresholds.append(np.nan);continue
        if n<MIN_SUPPORT:thresholds.append(np.nan);continue
        ii=q.u.map(lookup).to_numpy(int)
        thresholds.append(_calibrate(progress[ii],data.train_lengths[q.u],q.observed.to_numpy(),hierarchical_weights(q)))
    unconstrained=list(thresholds);pools=[]
    for h in range(4 if phase else 0,7):
        if not np.isfinite(thresholds[h]):continue
        pools.append(([h],thresholds[h]))
        while len(pools)>1 and pools[-2][1]>pools[-1][1]:
            heads=pools[-2][0]+pools[-1][0];q=labels[labels['head'].isin(heads)]
            ii=q.u.map(lookup).to_numpy(int)
            value=_calibrate(progress[ii],data.train_lengths[q.u],q.observed.to_numpy(),hierarchical_weights(q))
            pools=pools[:-2]+[(heads,value)]
    for heads,value in pools:
        for h in heads:thresholds[h]=value
    return dict(kind='thermal',thresholds=thresholds,unconstrained_thresholds=unconstrained,
                order_pools=[h for h,v in pools],alpha=float(alpha),phase=phase,base=base,support=support)


def fit_context(data,labels,regularization=1.):
    validate_source(data,labels)
    ids=np.unique(labels.u);mean=data.context[ids].mean(0);scale=data.context[ids].std(0).clip(.05)
    coeff=[];intercept=[];support=[]
    for h in range(7):
        q=labels[labels['head'].eq(h)];n=q.u.nunique();support.append(n)
        if n<MIN_SUPPORT:coeff.append([np.nan]*5);intercept.append(np.nan);continue
        ids=q.u.to_numpy(int);c=thermal_rate(data.raw[ids]).cumsum(1)
        demand=c[np.arange(len(q)),q.observed.to_numpy(int)]
        if (demand<=0).any():raise ValueError('Zero thermal demand in source target')
        w=hierarchical_weights(q)
        model=Ridge(alpha=float(regularization)).fit((data.context[ids]-mean)/scale,np.log(demand),sample_weight=w*len(w))
        coeff.append(model.coef_.tolist());intercept.append(float(model.intercept_))
    return dict(kind='context',mean=mean,scale=scale,coeff=np.array(coeff),intercept=np.array(intercept),
                regularization=float(regularization),support=support)


def predict(model,data,ids,train=False):
    ids=np.asarray(ids,int);lengths=(data.train_lengths if train else data.lengths)[ids]
    if model['kind']=='calendar':return np.broadcast_to(model['days'],(len(ids),7)).copy()
    if model['kind']=='context':
        z=(data.context[ids]-model['mean'])/model['scale']
        demand=np.exp(np.clip(z@model['coeff'].T+model['intercept'],0,np.log(10000)))
        progress=thermal_rate(data.raw[ids]).cumsum(1)
        return np.column_stack([crossing(progress,demand[:,h],lengths) for h in range(7)])
    if model['kind']=='thermal':
        rate=thermal_rate(data.raw[ids],model['alpha'])
        if model['phase']:
            result=predict(model['base'],data,ids,train)
            progress=phase_progress(rate,result[:,3])
        else:
            result=np.full((len(ids),7),np.nan);progress=rate.cumsum(1)
        for h in range(4 if model['phase'] else 0,7):
            result[:,h]=crossing(progress,model['thresholds'][h],lengths)
        return result
    raise ValueError('Unknown timing model')


def teacher_predictions(data,labels):
    blocks=sorted(labels.block.unique());groups={b:i%3 for i,b in enumerate(blocks)}
    output=np.full((len(data.raw),7),np.nan);receipts=[]
    for group in range(3):
        held_blocks=[b for b in blocks if groups[b]==group]
        fit=labels[~labels.block.isin(held_blocks)];held=labels[labels.block.isin(held_blocks)]
        if not len(held):continue
        model=fit_thermal(data,fit);ids=held.u.unique()
        output[ids]=predict(model,data,ids,train=True)
        receipts.append(dict(held_blocks=list(map(str,held_blocks)),fit_blocks=list(map(str,fit.block.unique())),
                             fit_units=int(fit.u.nunique()),held_units=len(ids),support=model['support']))
    return output,receipts


def event_features(data,ids,base,train=False):
    ids=np.asarray(ids,int);raw=data.raw[ids];n,t,_=raw.shape
    lengths=(data.train_lengths if train else data.lengths)[ids]
    channels=np.concatenate([raw,raw[:,:,:2].mean(2,keepdims=True)],2)
    cum=np.pad(channels.cumsum(1),((0,0),(1,0),(0,0)))
    square=np.pad((channels**2).cumsum(1),((0,0),(1,0),(0,0)))
    gdd=np.pad(thermal_rate(raw).cumsum(1),((0,0),(1,0)))
    rows=np.arange(n);out=[]
    previous=np.column_stack([np.zeros(n),base[:,:-1]])
    for h in range(7):
        b=base[:,h];end=np.floor(np.nan_to_num(b,nan=0)).astype(int).clip(0,t-1)+1
        values=[data.context[ids],np.column_stack([b/180,np.nan_to_num(previous[:,h])/180,
                    (b-np.nan_to_num(previous[:,h]))/180]),np.broadcast_to(np.eye(7)[h],(n,7))]
        for window in [t,7,30]:
            start=np.maximum(0,end-window);count=(end-start)[:,None]
            mean=(cum[rows,end]-cum[rows,start])/count
            variance=np.maximum(0,(square[rows,end]-square[rows,start])/count-mean**2)
            values.extend([mean,np.sqrt(variance)])
        values.append((gdd[rows,end]/1000)[:,None])
        feature=np.column_stack(values)
        feature[~np.isfinite(b)|(b<0)|(b>=lengths)]=np.nan
        out.append(feature)
    return np.stack(out,1).astype(np.float32)


def fit_residual(data,labels,teacher,kind,seed=17,steps=160,teacher_qualified=None):
    validate_source(data,labels)
    ids=np.unique(labels.u);features=event_features(data,ids,teacher[ids],train=True)
    lookup={u:i for i,u in enumerate(ids)};row=labels.u.map(lookup).to_numpy(int);head=labels['head'].to_numpy(int)
    x=features[row,head];y=labels.observed.to_numpy()-teacher[labels.u,head]
    valid=np.isfinite(x).all(1)&np.isfinite(y)
    qualified=[True]*7 if teacher_qualified is None else list(map(bool,teacher_qualified))
    counts=[int(labels.loc[valid&labels['head'].eq(h),'u'].nunique()) for h in range(7)]
    supported=[bool(counts[h]>=MIN_SUPPORT and qualified[h]) for h in range(7)]
    audit=dict(teacher_head_qualified=qualified,residual_training_environments=counts,
               available_labels_per_head=[int(labels['head'].eq(h).sum()) for h in range(7)],
               invalid_teacher_or_features_per_head=[int((~valid&labels['head'].eq(h)).sum()) for h in range(7)],
               unsupported_head_behavior='zero residual correction; retain complete-source T prediction')
    valid &=np.array(supported)[head]
    if not valid.any():return dict(kind=kind,empty=True,supported=supported,**audit)
    q=labels.loc[valid];x=x[valid];y=y[valid];w=hierarchical_weights(q)
    if kind=='RF':
        from sklearn.ensemble import RandomForestRegressor
        model=RandomForestRegressor(n_estimators=64,max_depth=10,min_samples_leaf=6,
            max_features=.7,criterion='absolute_error',max_samples=max(2,int(.8*len(x))),
            random_state=int(seed),n_jobs=2)
        model.fit(x,y,sample_weight=w*len(w))
        return dict(kind=kind,model=model,supported=supported,empty=False,**audit,
                    samples=len(x),seed=seed,max_samples=model.max_samples)
    if kind!='MLP':raise ValueError('Unknown residual learner')
    import torch
    torch.set_num_threads(2);torch.manual_seed(int(seed));rng=np.random.default_rng(seed)
    mean=x.mean(0);scale=x.std(0).clip(.05);z=torch.tensor((x-mean)/scale,dtype=torch.float32)
    target=torch.tensor(y,dtype=torch.float32)
    network=torch.nn.Sequential(torch.nn.Linear(x.shape[1],48),torch.nn.GELU(),
        torch.nn.Linear(48,32),torch.nn.GELU(),torch.nn.Linear(32,1))
    torch.nn.init.zeros_(network[-1].weight);torch.nn.init.zeros_(network[-1].bias)
    optim=torch.optim.AdamW(network.parameters(),lr=.003,weight_decay=.001)
    network.train()
    for _ in range(steps):
        index=rng.choice(len(x),256,replace=True,p=w)
        loss=(network(z[index]).squeeze(1)-target[index]).abs().mean()
        optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5);optim.step()
    network.eval()
    state={k:v.detach().numpy() for k,v in network.state_dict().items()}
    return dict(kind=kind,state=state,mean=mean,scale=scale,supported=supported,empty=False,**audit,
                samples=len(x),seed=seed,steps=steps)


def residual_prediction(model,data,ids,base,train=False):
    result=np.zeros_like(base)
    if model['empty']:return result
    f=event_features(data,ids,base,train=train);valid=np.isfinite(f).all(2)&np.array(model['supported'])[None,:]
    if not valid.any():return result
    x=f[valid]
    if model['kind']=='RF':result[valid]=model['model'].predict(x)
    else:
        # NumPy replay makes saved inference independent of a live Torch module.
        from scipy.special import erf
        z=(x-model['mean'])/model['scale'];s=model['state']
        for prefix in ['0','2']:
            z=z@s[prefix+'.weight'].T+s[prefix+'.bias'];z=.5*z*(1+erf(z/np.sqrt(2)))
        result[valid]=(z@s['4.weight'].T+s['4.bias']).ravel()
    return result
