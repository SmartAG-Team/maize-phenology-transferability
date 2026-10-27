"""Training, retrospective locked evaluation and finite-horizon proper scores."""
import copy,json,time,resource,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from sklearn.cluster import MiniBatchKMeans
from .model import DevelopmentClock,event_loss
from .core import crps

SEEDS=[17,29,43,71,101]

def fit_mask(env,experiment='main'):
    mask=env.new_partition.eq('train').to_numpy(copy=True)
    if experiment=='warm':
        mask &= ~env.warm_challenge.to_numpy(bool)
        if 'weather_days' in env:mask &= env.weather_days.to_numpy(copy=True)>=120
    return mask

def tuning_mask(env,experiment='main'):
    mask=env.new_partition.eq('validation').to_numpy(copy=True)
    if experiment=='warm':
        mask &= ~env.warm_challenge.to_numpy(bool)
        if 'weather_days' in env:mask &= env.weather_days.to_numpy(copy=True)>=120
    return mask

def score_cdf(cdf,observed):
    cdf=np.asarray(cdf);observed=np.asarray(observed,dtype=int);n,H=cdf.shape
    days=np.arange(1,H+1);error=(cdf-(days[None,:]>=observed[:,None]))**2
    med=np.where(np.any(cdf>=.5,1),np.argmax(cdf>=.5,1)+1,np.nan)
    ok=np.isfinite(med);res=med[ok]-observed[ok]
    pmf=cdf[np.arange(n),observed-1]-np.where(observed>1,cdf[np.arange(n),(observed-2).clip(0)],0.)
    d={'n':n,'n_resolved_median':int(ok.sum()),'mae':float(np.abs(res).mean()) if len(res) else np.nan,'rmse':float(np.sqrt(np.mean(res**2))) if len(res) else np.nan,'bias':float(res.mean()) if len(res) else np.nan,'crps':float(error.sum(1).mean()),'nll':float(-np.log(pmf.clip(1e-12)).mean()),'n_zero_event_mass':int((pmf<=0).sum()),'probability_floor_fraction':float((pmf<1e-12).mean()),'unresolved_fraction':float((~ok).mean()),'tail_mass':float((1-cdf[:,-1]).mean())}
    for nominal in [.5,.8,.9,.95]:
        a=(1-nominal)/2;b=1-a
        low=np.where(np.any(cdf>=a,1),np.argmax(cdf>=a,1)+1,np.inf)
        high=np.where(np.any(cdf>=b,1),np.argmax(cdf>=b,1)+1,np.inf)
        closed=np.isfinite(low)&np.isfinite(high)
        d[f'coverage_{int(nominal*100)}']=float(((observed>=low)&(observed<=high)).mean())
        d[f'width_{int(nominal*100)}']=float((high[closed]-low[closed]).mean()) if closed.any() else np.nan
        d[f'open_interval_{int(nominal*100)}']=float((~closed).mean())
    return d

def load_data(root):
    root=Path(root);a=np.load(root/'data/weather_features.npz')
    arrays={k:a[k] for k in a.files};a.close()
    env=pd.read_csv(root/'data/environments.csv.gz',low_memory=False)
    meta=json.loads((root/'data/metadata.json').read_text())
    return arrays,env,meta

def experiment_meta(a,env,meta,experiment='main',refit=False):
    train=np.ones(len(env),bool) if refit else fit_mask(env,experiment)
    y=a['Y'];g=a['G'];x=a['X'];lengths=a['lengths']
    retained=[i for i in range(y.shape[1]) if ((y[:,i]>0)&train).sum()>=30]
    if any(i not in retained for i in range(7)):raise ValueError('Insufficient primary support')
    flat=x[train][np.arange(x.shape[1])[None,:]<lengths[train,None]]
    mean,std=flat.mean(0),np.maximum(flat.std(0),1e-3)
    cumulative=g.cumsum(1);thresholds=[];noise=[]
    for j in retained:
        inds=np.flatnonzero(train&(y[:,j]>0));vals=cumulative[inds,y[inds,j]-1]
        med=np.median(vals);thresholds.append(float(med));noise.append(float(np.clip(np.std(np.log(np.maximum(vals,1)/med)),.03,.5)))
    if np.any(np.diff(thresholds[:7])<=0):raise ValueError('Non-increasing native medians')
    new=dict(meta,retained=retained,endpoints=[meta['endpoints'][j] for j in retained],threshold_gdd=thresholds,threshold_fraction=(np.array(thresholds)/thresholds[6]).tolist(),maturation_gdd=thresholds[6],feature_mean=mean.tolist(),feature_std=std.tolist(),noise_initial=noise,experiment=experiment,production_refit=refit)
    return new,train

def make_model(a,env,meta,mode,seed,width=16,inducing=32,rff=256,refit=False,experiment='main'):
    torch.manual_seed(seed);np.random.seed(seed)
    m,train=experiment_meta(a,env,meta,experiment,refit)
    # Inducing initialization uses observed growing-season training weather only.
    sample=a['X'][train,:180][np.arange(180)[None,:]<a['lengths'][train,None]]
    rng=np.random.default_rng(seed);sample=sample[rng.choice(len(sample),min(20000,len(sample)),replace=False)]
    sample=(sample-np.asarray(m['feature_mean']))/np.asarray(m['feature_std'])
    if mode in ['hybrid','raw_gp','unsafe_kernel','shared_phase','independent_days']:
        centers=MiniBatchKMeans(n_clusters=inducing,random_state=seed,n_init=3,batch_size=1024).fit(sample).cluster_centers_
    else:centers=sample[:inducing]
    model=DevelopmentClock(m['threshold_fraction'],m['feature_mean'],m['feature_std'],torch.tensor(centers,dtype=torch.float32),width=width,rff=rff,mode=mode,sigma=m['noise_initial'])
    return model,m,train

def predictive_cdf(model,a,indices,km,samples=32,batch=32,seed=987,device=None):
    out=[];model.eval()
    device=device or model.thresholds.device
    with torch.no_grad():
        for start in range(0,len(indices),batch):
            ii=indices[start:start+batch]
            # Same sampled functions across chunks, preserving common parameter uncertainty.
            torch.manual_seed(seed)
            x=torch.as_tensor(a['X'][ii],device=device);g=torch.as_tensor(a['G'][ii],device=device)
            z=model(x,g,float(km),samples=samples)
            out.append(model.cdf(z).cpu().numpy())
    return np.concatenate(out,axis=0) if out else np.empty((0,len(model.thresholds),366))

def objective_weights(y):
    counts=(y>0).sum(0);J=y.shape[1]
    alpha=np.array([.85/7]*7+[.15/(J-7)]*(J-7)) if J>7 else np.ones(7)/7
    return torch.tensor(alpha*len(y)/np.maximum(counts,1),dtype=torch.float32)

def train(root,mode,seed=17,width=16,inducing=32,epochs=60,batch=32,patience=8,experiment='main',refit=False,tag=None,rff=256,learning_rate=5e-4,pilot_n=None,threads=4,device=None):
    root=Path(root);torch.set_num_threads(threads)
    a,env,meta=load_data(root)
    model,m,trainmask=make_model(a,env,meta,mode,seed,width,inducing,rff,refit,experiment)
    device=device or ('mps' if len(model.phase_models) and torch.backends.mps.is_available() else 'cpu')
    model=model.to(device)
    inds=np.flatnonzero(trainmask)
    if pilot_n is not None and len(inds)>pilot_n:
        rngpilot=np.random.default_rng(seed)
        auxiliary=inds[(a['Y'][inds,7:]>0).any(1)]
        primary=np.setdiff1d(inds,auxiliary)
        inds=np.r_[auxiliary,rngpilot.choice(primary,pilot_n-len(auxiliary),replace=False)]
        m['pilot_subset']=True;m['full_training_environments']=int(trainmask.sum())
    valmask=tuning_mask(env,experiment)
    validation=np.flatnonzero(valmask);retained=m['retained'];y=a['Y'][:,retained];weights=objective_weights(y[inds])
    opt=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    rng=np.random.default_rng(seed);history=[];best=float('inf');best_state=None;noimprove=0;bestepoch=0
    tag=tag or f'{experiment}_{mode}_s{seed}_w{width}_m{inducing}'
    output=root/'results/checkpoints';output.mkdir(exist_ok=True,parents=True)
    dest=output/(tag+'.pt')
    starttime=time.time();train_samples=2 if len(model.phase_models) else 1
    for epoch in range(1,epochs+1):
        t=time.time();model.train();order=rng.permutation(inds);totloss=0.;n=0
        for step in range(0,len(order),batch):
            ii=order[step:step+batch];yy=torch.as_tensor(y[ii],dtype=torch.long)
            maxday=int(yy.max());x=torch.as_tensor(a['X'][ii,:maxday],device=device);g=torch.as_tensor(a['G'][ii,:maxday],device=device)
            opt.zero_grad(set_to_none=True);z=model(x,g,float(m['maturation_gdd']),train_samples)
            loss=event_loss(z.cpu(),yy,model.thresholds.cpu(),model.sigma.cpu(),weights)+model.kl().cpu()/len(inds)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss: {tag} epoch{epoch} batch{step}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            totloss+=float(loss.detach())*len(ii);n+=len(ii)
        if refit:
            valscore=totloss/n
        else:
            c=predictive_cdf(model,a,validation,m['maturation_gdd'],samples=32 if len(model.phase_models) else 1,batch=batch)
            vv=[]
            for j in range(7):
                ok=y[validation,j]>0
                vv.append(score_cdf(c[ok,j],y[validation[ok],j])['crps'])
            valscore=float(np.mean(vv))
        row={'epoch':epoch,'train_composite_nll_kl':totloss/n,'validation_macro_crps':None if refit else valscore,'seconds':time.time()-t,'kl':float(model.kl().detach())};history.append(row)
        print(json.dumps(dict(tag=tag,**row)),flush=True)
        if refit or valscore<best-1e-4:
            best=valscore;bestepoch=epoch;noimprove=0;best_state=copy.deepcopy(model.state_dict())
            torch.save({'state_dict':best_state,'training_device':device,'metadata':m,'mode':mode,'width':width,'inducing':inducing,'rff':rff,'seed':seed,'best_epoch':bestepoch,'validation_crps':None if refit else best,'train_composite_nll_kl':totloss/n,'training_environments':len(inds),'training_environment_hash':hashlib.sha256('\n'.join(env.iloc[inds].environment_id_model).encode()).hexdigest()},dest)
        else:noimprove+=1
        pd.DataFrame(history).to_csv(output/(tag+'_history.csv'),index=False)
        if noimprove>=patience:break
    report={'tag':tag,'checkpoint':str(dest),'epochs_run':len(history),'best_epoch':bestepoch,'validation_macro_crps':None if refit else best,'final_train_composite_nll_kl':history[-1]['train_composite_nll_kl'],'elapsed_seconds':time.time()-starttime,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'device':device,'likelihood_device':'cpu_float64','batch':batch,'learning_rate':learning_rate,'pilot_n':pilot_n,'training_environments':len(inds)}
    (output/(tag+'_report.json')).write_text(json.dumps(report,indent=2));return report

def restore(path):
    bundle=torch.load(path,map_location='cpu',weights_only=False);m=bundle['metadata'];state=bundle['state_dict']
    inducing=state.get('phase_models.0.inducing',torch.zeros(bundle['inducing'],3))
    model=DevelopmentClock(m['threshold_fraction'],m['feature_mean'],m['feature_std'],inducing,width=bundle['width'],rff=bundle['rff'],mode=bundle['mode'],sigma=m['noise_initial'])
    model.load_state_dict(state);model.eval();return model,bundle

def evaluation_device(has_gp,requested=None):
    return requested or ("mps" if has_gp and torch.backends.mps.is_available() else "cpu")

def evaluate(root,checkpoint,samples=256,tag=None,device=None):
    root=Path(root);torch.set_num_threads(4);a,env,meta=load_data(root);model,b=restore(checkpoint);m=b['metadata'];ret=m['retained'];Y=a['Y'][:,ret]
    device=evaluation_device(bool(len(model.phase_models)),device);model=model.to(device)
    experiment=m['experiment'];tag=tag or Path(checkpoint).stem
    parts=['train','validation','test_spatial','test_temporal','test_joint']
    if experiment=='warm':parts=['validation','test_spatial','test_temporal','test_joint','warm_train_reserved']
    scores=[];preds=[];individual=[]
    cached={k:env[k].to_numpy(copy=True) for k in ['environment_id_model','spatial_group','year','warm_challenge','mean_temperature_120']}
    for part in parts:
        mask=env.new_partition.eq(part).to_numpy(copy=True)
        if part=='warm_train_reserved':mask=env.new_partition.eq('train').to_numpy(copy=True)&env.warm_challenge.to_numpy(bool)
        if experiment=='warm' and part=='validation':mask&=~env.warm_challenge.to_numpy(bool)
        # Full-horizon scores are primary; truncation-specific audit remains in env table.
        mask&=a['lengths']==366
        ii=np.flatnonzero(mask)
        if not len(ii):continue
        c=predictive_cdf(model,a,ii,m['maturation_gdd'],samples=samples if len(model.phase_models) else 1)
        for j,endpoint in enumerate(m['endpoints']):
            ok=Y[ii,j]>0
            if j<7:ok &= env.iloc[ii].dataset_id.eq('dwd_annual').to_numpy(copy=True)
            truth=Y[ii[ok],j];cc=c[ok,j]
            if not len(truth):continue
            result=score_cdf(cc,truth);scores.append(dict(tag=tag,mode=b['mode'],seed=b['seed'],experiment=experiment,partition=part,endpoint=endpoint,**result))
            days=np.arange(1,367);errors=((cc-(days[None,:]>=truth[:,None]))**2).sum(1)
            for k,i in enumerate(ii[ok]):
                q={str(q):float(np.argmax(cc[k]>=q)+1) if np.any(cc[k]>=q) else None for q in [.05,.1,.25,.5,.75,.9,.95]}
                pred={'tag':tag,'partition':part,'environment_id_model':cached['environment_id_model'][i],'spatial_group':cached['spatial_group'][i],'year':int(cached['year'][i]),'endpoint':endpoint,'observed_day':int(truth[k]),'median_day':q['0.5'],'lower90_day':q['0.05'],'upper90_day':q['0.95'],'crps':float(errors[k]),'tail_mass':float(1-cc[k,-1]),'warm_challenge':bool(cached['warm_challenge'][i]),'mean_temperature_120':float(cached['mean_temperature_120'][i])}
                preds.append(pred)
        print(f'evaluated {tag} {part}: {len(ii)} environments',flush=True)
    out=root/'results/evaluation';out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(scores).to_csv(out/(tag+'_metrics.csv'),index=False)
    pd.DataFrame(preds).to_csv(out/(tag+'_predictions.csv.gz'),index=False)
    return scores
