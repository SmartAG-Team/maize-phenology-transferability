"""Causal DL hazards and a fixed-window tabular ML reference."""
import copy,json,time,math
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import pandas as pd
import torch
from torch import nn
from .experiment import load_data,experiment_meta,fit_mask,tuning_mask,objective_weights,score_cdf

class CausalConv(nn.Module):
    def __init__(self,hidden,dilation):
        super().__init__();self.conv=nn.Conv1d(hidden,hidden,3,dilation=dilation);self.padding=2*dilation
    def forward(self,x):
        return x+torch.nn.functional.silu(self.conv(torch.nn.functional.pad(x,(self.padding,0))))
class HazardNetwork(nn.Module):
    def __init__(self,stages,mode='lstm',width=16):
        super().__init__();self.mode=mode
        if mode=='lstm':self.net=nn.LSTM(5,width,batch_first=True)
        else:self.net=nn.Sequential(nn.Conv1d(5,width,1),CausalConv(width,1),CausalConv(width,2),CausalConv(width,4),CausalConv(width,8))
        self.head=nn.Linear(width,stages)
    def forward(self,x):
        h=self.net(x)[0] if self.mode=='lstm' else self.net(x.transpose(1,2)).transpose(1,2)
        return self.head(h)

def hazard_cdf(logits):
    logits=logits.to(device='cpu').double()
    return -torch.expm1(torch.cumsum(torch.nn.functional.logsigmoid(-logits),dim=1)).transpose(1,2)

def hazard_loss(logits,y,weights):
    logsurv=torch.cumsum(torch.nn.functional.logsigmoid(-logits),dim=1)
    prev=torch.cat([torch.zeros_like(logsurv[:,:1]),logsurv[:,:-1]],dim=1)
    logpmf=(torch.nn.functional.logsigmoid(logits)+prev).transpose(1,2)
    ix=(y.clamp_min(1)-1).clamp_max(logits.shape[1]-1)
    obs=logpmf.gather(2,ix[:,:,None]).squeeze(2)
    return -(obs*(y>0)*weights).sum()/len(y)

def sequence_features(a,m):
    x=(a['X']-np.asarray(m['feature_mean'],dtype=np.float32))/np.asarray(m['feature_std'],dtype=np.float32)
    days=np.broadcast_to(np.arange(1,367,dtype=np.float32)[None,:,None]/366.,(len(x),366,1))
    gdd=a['G'].cumsum(1)[:,:,None]/m['maturation_gdd']
    return np.concatenate([x,days,gdd],axis=2).astype(np.float32)

def tabular_features(x,g,lengths=None):
    result=[]
    for horizon in [7,14,30,60,90,120,180]:
        z=x[:,:horizon];result.extend([z.mean(1),z.std(1),np.quantile(z,.1,axis=1),np.quantile(z,.9,axis=1),g[:,:horizon].sum(1)[:,None],(z[:,:,1]>=30).sum(1)[:,None]/horizon])
    features=np.concatenate(result,axis=1).astype(np.float32)
    if lengths is not None:
        for j,horizon in enumerate([7,14,30,60,90,120,180]):
            features[np.asarray(lengths)<horizon,j*14:(j+1)*14]=np.nan
    return features

def predict_hazard(model,x,ii,batch=256):
    chunks=[];model.eval()
    with torch.no_grad():
        for k in range(0,len(ii),batch):chunks.append(hazard_cdf(model(torch.as_tensor(x[ii[k:k+batch]]))).numpy())
    return np.concatenate(chunks)

def train_hazard(root,mode,seed=17,epochs=60,patience=8,experiment='main',width=16,batch=64):
    torch.set_num_threads(4);torch.manual_seed(seed);root=Path(root);a,env,meta=load_data(root);m,mask=experiment_meta(a,env,meta,experiment)
    y=a['Y'][:,m['retained']];x=sequence_features(a,m);ii=np.flatnonzero(mask);v=tuning_mask(env,experiment)
    vi=np.flatnonzero(v);weights=objective_weights(y[ii]);model=HazardNetwork(y.shape[1],mode,width)
    with torch.no_grad():
        med=np.array([np.median(y[ii,j][y[ii,j]>0]) for j in range(y.shape[1])]);hazard=np.log(2.)/med
        model.head.bias.copy_(torch.tensor(np.log(hazard/(1-hazard)),dtype=torch.float32))
        model.head.weight.mul_(.01)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4);rng=np.random.default_rng(seed);history=[];best=np.inf;wait=0;tag=f'{experiment}_{mode}_s{seed}'
    output=root/'results/checkpoints';output.mkdir(exist_ok=True,parents=True)
    for epoch in range(1,epochs+1):
        t=time.time();model.train();order=rng.permutation(ii);total=0.
        for k in range(0,len(order),batch):
            ix=order[k:k+batch];yy=torch.as_tensor(y[ix],dtype=torch.long);T=int(yy.max())
            opt.zero_grad(set_to_none=True);logits=model(torch.as_tensor(x[ix,:T]));loss=hazard_loss(logits,yy,weights)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())*len(ix)
        cc=predict_hazard(model,x,vi);vv=[score_cdf(cc[y[vi,j]>0,j],y[vi[y[vi,j]>0],j])['crps'] for j in range(7)]
        val=float(np.mean(vv));row=dict(epoch=epoch,train_nll=total/len(ii),validation_macro_crps=val,seconds=time.time()-t);history.append(row);print(json.dumps(dict(tag=tag,**row)),flush=True)
        if val<best-1e-4:
            best=val;wait=0
            torch.save({'state_dict':copy.deepcopy(model.state_dict()),'metadata':m,'mode':mode,'seed':seed,'width':width,'best_epoch':epoch,'validation_crps':best},output/(tag+'.pt'))
        else:wait+=1
        pd.DataFrame(history).to_csv(output/(tag+'_history.csv'),index=False)
        if wait>=patience:break
    return {'tag':tag,'validation_macro_crps':best,'epochs_run':len(history)}

def xgb_reference(root,seed=17,experiment='main'):
    from xgboost import XGBRegressor
    root=Path(root);a,env,meta=load_data(root);m,mask=experiment_meta(a,env,meta,experiment);y=a['Y'][:,m['retained']]
    X=tabular_features(a['X'],a['G'],a['lengths']);train=np.flatnonzero(mask);valmask=tuning_mask(env,experiment)
    val=np.flatnonzero(valmask);med=np.zeros((len(env),y.shape[1]));residuals=[];choices=[]
    out=root/'results/checkpoints';out.mkdir(exist_ok=True,parents=True)
    for j,e in enumerate(m['endpoints']):
        ti=train[y[train,j]>0];vi=val[y[val,j]>0];models=[]
        if len(vi)<5:
            # No validation evidence for a native auxiliary endpoint: fixed depth 3.
            vi=ti;selection='fixed_depth_due_to_sparse_validation'
        else:selection='validation_mae'
        for depth in ([3] if selection.startswith('fixed') else [3,6]):
            model=XGBRegressor(n_estimators=300,max_depth=depth,learning_rate=.05,subsample=.8,colsample_bytree=.8,reg_lambda=5,min_child_weight=10,n_jobs=4,random_state=seed,objective='reg:squarederror',early_stopping_rounds=20)
            model.fit(X[ti],y[ti,j],eval_set=[(X[vi],y[vi,j])],verbose=False)
            error=float(np.mean(np.abs(model.predict(X[vi])-y[vi,j])));models.append((error,model,depth))
        error,model,depth=min(models,key=lambda z:z[0]);med[:,j]=model.predict(X)
        rv=val[y[val,j]>0];res=(y[rv,j]-med[rv,j]) if len(rv)>=5 else np.array([-10.,0.,10.])
        residuals.append(np.sort(res));choices.append({'endpoint':e,'depth':depth,'validation_mae':error,'residual_n':len(rv),'selection':selection,'best_iteration':int(model.best_iteration)})
        model.save_model(out/f'{experiment}_xgb_s{seed}_{j}.ubj')
    # Empirical additive validation residuals; no tail renormalization.
    np.savez(out/f'{experiment}_xgb_s{seed}_predictions.npz',medians=med,**{f'residual_{j}':r for j,r in enumerate(residuals)})
    (out/f'{experiment}_xgb_s{seed}_metadata.json').write_text(json.dumps({'metadata':m,'seed':seed,'choices':choices},indent=2))
    return choices

def evaluate_reference(root,mode,seed=17,experiment='main'):
    root=Path(root);torch.set_num_threads(4);a,env,meta=load_data(root);tag=f'{experiment}_{mode}_s{seed}';out=root/'results/checkpoints'
    if mode=='xgb':
        b=json.loads((out/(tag+'_metadata.json')).read_text());m=b['metadata'];v=np.load(out/(tag+'_predictions.npz'))
    else:
        b=torch.load(out/(tag+'.pt'),weights_only=False);m=b['metadata'];model=HazardNetwork(len(m['endpoints']),mode,b['width']);model.load_state_dict(b['state_dict']);x=sequence_features(a,m)
    y=a['Y'][:,m['retained']];scores=[];predictions=[]
    cached={k:env[k].to_numpy(copy=True) for k in ['environment_id_model','spatial_group','year','warm_challenge','mean_temperature_120']}
    parts=['train','validation','test_spatial','test_temporal','test_joint'] if experiment=='main' else ['validation','test_spatial','test_temporal','test_joint','warm_train_reserved']
    for part in parts:
        mask=env.new_partition.eq(part).to_numpy(copy=True)
        if part=='warm_train_reserved':mask=env.new_partition.eq('train').to_numpy(copy=True)&env.warm_challenge.to_numpy(bool)
        if experiment=='warm' and part=='validation':mask&=~env.warm_challenge.to_numpy(bool)
        mask&=a['lengths']==366;ii=np.flatnonzero(mask)
        if not len(ii):continue
        if mode=='xgb':
            grid=np.arange(-400.,401.);curves=[]
            for j in range(y.shape[1]):
                # Fixed 1-day Gaussian smoothing prevents zero discrete date likelihoods.
                residual=v[f'residual_{j}'];cdfgrid=ndtr((grid[:,None]-residual[None,:])/1.).mean(1)
                curves.append(np.interp(np.arange(1,367)[None,:]-v['medians'][ii,j,None],grid,cdfgrid,left=0.,right=1.))
            cc=np.stack(curves,axis=1)
        else:cc=predict_hazard(model,x,ii)
        for j,e in enumerate(m['endpoints']):
            ok=y[ii,j]>0
            if j<7:ok &= env.iloc[ii].dataset_id.eq('dwd_annual').to_numpy(copy=True)
            truth=y[ii[ok],j];c=cc[ok,j]
            if not len(truth):continue
            scores.append(dict(tag=tag,mode=mode,seed=seed,experiment=experiment,partition=part,endpoint=e,**score_cdf(c,truth)))
            err=((c-(np.arange(1,367)[None,:]>=truth[:,None]))**2).sum(1)
            for k,i in enumerate(ii[ok]):
                qs={q:float(np.argmax(c[k]>=q)+1) if (c[k]>=q).any() else None for q in [.05,.5,.95]}
                predictions.append(dict(tag=tag,partition=part,environment_id_model=cached['environment_id_model'][i],spatial_group=cached['spatial_group'][i],year=int(cached['year'][i]),endpoint=e,observed_day=int(truth[k]),median_day=qs[.5],lower90_day=qs[.05],upper90_day=qs[.95],crps=float(err[k]),tail_mass=float(1-c[k,-1]),warm_challenge=bool(cached['warm_challenge'][i]),mean_temperature_120=float(cached['mean_temperature_120'][i])))
        print(f'evaluated {tag} {part}: {len(ii)} environments',flush=True)
    out=root/'results/evaluation';out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(scores).to_csv(out/(tag+'_metrics.csv'),index=False);pd.DataFrame(predictions).to_csv(out/(tag+'_predictions.csv.gz'),index=False)
