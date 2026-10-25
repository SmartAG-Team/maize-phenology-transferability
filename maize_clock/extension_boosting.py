"""Causal weather-history features and sampled discrete-time boosting hazards."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .experiment import load_data,experiment_meta,score_cdf
from .references import sequence_features

def history_features(a,m,env):
    base=sequence_features(a,m);X=a['X'];G=a['G'];N,T=G.shape
    mid=(X[:,:,0]+X[:,:,1])/2
    total=np.c_[np.zeros(N),np.cumsum(mid,axis=1,dtype=np.float64)]
    square=np.c_[np.zeros(N),np.cumsum(mid.astype(np.float64)**2,axis=1)]
    thermal=np.c_[np.zeros(N),np.cumsum(G,axis=1,dtype=np.float64)]
    end=np.arange(1,T+1);features=[base]
    for window in [7,14,30]:
        start=np.maximum(0,end-window);count=end-start
        mean=(total[:,end]-total[:,start])/count
        var=np.maximum(0,(square[:,end]-square[:,start])/count-mean**2)
        gd=(thermal[:,end]-thermal[:,start])/float(m['maturation_gdd'])
        features.append(np.stack([mean/20.,np.sqrt(var)/10.,gd],axis=2).astype(np.float32))
    doy=pd.to_datetime(env.sowing_date,errors='raise').dt.dayofyear.to_numpy()
    static=np.c_[env.latitude.to_numpy()/90.,env.longitude.to_numpy()/180.,np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25)].astype(np.float32)
    if 'history_static_mean' in m:static=(static-np.asarray(m['history_static_mean']))/np.asarray(m['history_static_std'])
    static=static.astype(np.float32)
    features.append(np.broadcast_to(static[:,None,:],(N,T,4)))
    return np.concatenate(features,axis=2).astype(np.float32)

def add_context_normalizers(m,env,mask):
    doy=pd.to_datetime(env.sowing_date,errors='raise').dt.dayofyear.to_numpy()
    values=np.c_[env.latitude.to_numpy()/90.,env.longitude.to_numpy()/180.,np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25)]
    m['history_static_mean']=values[mask].mean(0).tolist();m['history_static_std']=np.maximum(values[mask].std(0),.01).tolist()
    m['history_feature_definition']='Daily weather, elapsed day, cumulative GDD; past 7/14/30-day midpoint mean/std and GDD; latitude/longitude/sowing-day sine/cosine. Static normalizers use calibration only.'
    return m

def risk_rows(features,ids,observed,seed,negative_days=8):
    rng=np.random.default_rng(seed);observed=np.asarray(observed,dtype=int);ids=np.asarray(ids)
    event=features[ids,observed-1];valid=observed>1;vi=ids[valid];days=observed[valid]
    neg=np.floor(rng.random((len(vi),negative_days))*(days-1)[:,None]).astype(int)
    negative=features[vi[:,None],neg].reshape(-1,features.shape[-1])
    X=np.concatenate([event,negative]);Y=np.r_[np.ones(len(event)),np.zeros(len(negative))]
    weights=np.r_[np.ones(len(event)),np.repeat((days-1)/negative_days,negative_days)]
    return X,Y,weights

def predict_boosting(root,tag,features,ids,stages,batch=128):
    from xgboost import XGBClassifier
    out=Path(root)/'results/extensions/checkpoints';curves=[]
    for j in range(stages):
        model=XGBClassifier();model.load_model(out/f'{tag}_stage{j}.ubj');chunks=[]
        for start in range(0,len(ids),batch):
            ii=ids[start:start+batch];H=features.shape[1]
            hazard=model.predict_proba(features[ii].reshape(-1,features.shape[2]))[:,1].reshape(len(ii),H).astype(np.float64)
            chunks.append(-np.expm1(np.cumsum(np.log1p(-np.clip(hazard,0,1-1e-8)),axis=1)))
        curves.append(np.concatenate(chunks))
    return np.stack(curves,axis=1)

def fit_boosting(root,seed=17,depth=3,trees=250):
    from xgboost import XGBClassifier
    root=Path(root);out=root/'results/extensions/checkpoints';out.mkdir(exist_ok=True,parents=True)
    tag=f'boosted_hazard_d{depth}_s{seed}';path=out/(tag+'_report.json')
    if path.exists():return json.loads(path.read_text())
    tick=time.time();a,env,meta=load_data(root);m,mask=experiment_meta(a,env,meta,'main');m=add_context_normalizers(m,env,mask)
    features=history_features(a,m,env);y=a['Y'][:,m['retained']];train=np.flatnonzero(mask)
    val=np.flatnonzero(env.new_partition.eq('validation').to_numpy()&(a['lengths']==366));details=[];scores=[]
    for j,endpoint in enumerate(m['endpoints']):
        ti=train[y[train,j]>0];X,Y,W=risk_rows(features,ti,y[ti,j],seed+j,8)
        model=XGBClassifier(n_estimators=trees,max_depth=depth,learning_rate=.05,subsample=.8,colsample_bytree=.8,reg_lambda=5,min_child_weight=10,n_jobs=4,random_state=seed,objective='binary:logistic',eval_metric='logloss')
        model.fit(X,Y,sample_weight=W,verbose=False);model.save_model(out/f'{tag}_stage{j}.ubj')
        vi=val[y[val,j]>0]
        if j<7:vi=vi[env.iloc[vi].dataset_id.eq('dwd_annual').to_numpy()]
        h=model.predict_proba(features[vi].reshape(-1,features.shape[2]))[:,1].reshape(len(vi),366).astype(np.float64)
        c=-np.expm1(np.cumsum(np.log1p(-np.clip(h,0,1-1e-8)),axis=1))
        score=score_cdf(c,y[vi,j])['crps'] if len(vi) else None
        if j<7:scores.append(score)
        details.append({'endpoint':endpoint,'calibration_environments':len(ti),'risk_rows':len(Y),'risk_negative_days':8,'validation_crps':score})
        print(json.dumps({'tag':tag,**details[-1]}),flush=True)
    report={'tag':tag,'config':{'name':f'boosted_hazard_d{depth}','output':'boosted_hazard','depth':depth,'trees':trees,'feature_scheme':'history'},'metadata':m,'seed':seed,'validation_crps':float(np.mean(scores)),'stage_results':details,'training_environments':len(train),'training_environment_hash':hashlib.sha256('\n'.join(env.iloc[train].environment_id_model).encode()).hexdigest(),'held_out_read_for_selection':False,'causal_weather_features':True,'elapsed_seconds':time.time()-tick}
    path.write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--seed',type=int,default=17);p.add_argument('--depth',type=int,default=3);args=p.parse_args();fit_boosting(args.root,args.seed,args.depth)


class LazySequenceFeatures:
    """Identical daily features computed for requested environment batches."""
    def __init__(self,a,metadata,environments,history=False):
        self.a=a;self.metadata=metadata;self.environments=environments;self.history=history
        self.shape=(len(environments),366,18 if history else 5)
    def __getitem__(self,key):
        rows,days=key if isinstance(key,tuple) else (key,slice(None))
        ids=np.atleast_1d(np.arange(len(self.environments))[rows])
        weather={k:self.a[k][ids] for k in ['X','G']}
        env=self.environments.iloc[ids]
        values=history_features(weather,self.metadata,env) if self.history else sequence_features(weather,self.metadata)
        if self.history and self.metadata.get('history_mask_geography',False):
            values[:,:,14:16]=0.
        return values[:,days]
