"""Resumable calibration-only fitting of the declared transfer families."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .experiment import objective_weights,score_cdf
from .extensions import SequenceDevelopmentModel,sequence_crps_loss
from .transfer_attention import TransferAttentionModel
from .transfer_artifacts import fit_request,validate_artifact,atomic_torch_save,atomic_json_save


def select_recipe(reports):
    if not reports or not all(np.isfinite(r['tuning_crps']) for r in reports):
        raise ValueError('All attempted candidates must finish with finite tuning scores')
    grouped={}
    for r in reports:
        key=json.dumps(r['config'],sort_keys=True);grouped.setdefault(key,[]).append(r['tuning_crps'])
    return json.loads(min(grouped,key=lambda key:np.mean(grouped[key])))


def check_resume(saved,requested):
    for key in ['config','input_sha256','phase','seed','metadata_sha256','fit_index_sha256','tune_index_sha256','source_code_sha256','batch','patience','maximum_epochs']:
        if saved.get(key)!=requested.get(key):raise ValueError('Resume mismatch: '+key)


def clip_gradients(parameters,max_norm):
    parameters=[p for p in parameters if p.grad is not None]
    norms=[torch.linalg.vector_norm(p.grad.detach().to('cpu').double()) for p in parameters]
    norm=torch.linalg.vector_norm(torch.stack(norms)) if norms else torch.tensor(0.)
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
    scale=min(1.,float(max_norm)/(float(norm)+1e-6))
    for p in parameters:p.grad.mul_(scale)
    return float(norm)


def anchor_schedule(calibration_ids,seed,updates):
    rng=np.random.default_rng(seed)
    panel=rng.choice(np.asarray(calibration_ids),256,replace=False)
    batches=np.stack([rng.choice(panel,64,replace=False) for _ in range(updates)])
    return {'panel':panel,'batches':batches}


class MarginalProcess(TransferAttentionModel):
    def __init__(self,metadata):super().__init__(metadata,mode='process')
    def forward(self,x,g,km=None):
        progress=self.process_progress(x,g)
        z=(progress.clamp_min(1e-12).log()[:,:,None]-self.thresholds().log()[None,None,:])/self.sigma()[None,None,:]
        cdf=torch.where(progress[:,:,None]>0,torch.special.ndtr(z),torch.zeros_like(z)).transpose(1,2)
        return progress,cdf


def make_model(metadata,config):
    mode=config['mode']
    if mode=='standalone_photo':return MarginalProcess(metadata)
    if mode in {'sequential','hazard'}:
        return SequenceDevelopmentModel(metadata['threshold_fraction'][:7],metadata['noise_initial'][:7],
                 output='ordered_markov' if mode=='sequential' else 'hazard',width=32,
                 learn_thresholds=False,input_size=18)
    return TransferAttentionModel(metadata,mode=mode,bound=config.get('bound',1.),position=config.get('position','thermal'))


def restore(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=make_model(saved['metadata'],saved['config']);model.load_state_dict(saved['state_dict']);model.eval()
    return model,saved


def predict(model,features,arrays,ids,metadata,batch=128):
    chunks=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(ids),batch):
            ix=ids[start:start+batch]
            x=torch.as_tensor(features[ix]);g=torch.as_tensor(arrays['G'][ix])
            _,cdf=model(x,g,metadata['maturation_gdd']);chunks.append(cdf.detach().cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty((0,7,arrays['G'].shape[1]))


def fit_run(a,env,metadata,features,fit_ids,tune_ids,config,seed,path,phase,input_sha256,epochs=40,batch=128,patience=8):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(seed)
    fit_ids=np.asarray(fit_ids);tune_ids=np.asarray(tune_ids)
    if len(np.intersect1d(fit_ids,tune_ids)) or not len(fit_ids) or not len(tune_ids):raise ValueError('Invalid fitting/tuning separation')
    digest=lambda value:hashlib.sha256(value).hexdigest()
    requested=fit_request(metadata,fit_ids,tune_ids,config,seed,phase,input_sha256,epochs,batch,patience)
    reportpath=path.with_suffix('.json')
    if reportpath.exists():return validate_artifact(path,requested)
    model=make_model(metadata,config)
    if config['mode']=='hazard':
        with torch.no_grad():
            med=np.array([np.median(a['Y'][fit_ids,j][a['Y'][fit_ids,j]>0]) for j in range(7)])
            p=np.log(2)/med;model.head.bias.copy_(torch.tensor(np.log(p/(1-p)),dtype=torch.float32))
    neural_device=config.get('neural_device','cpu')
    if neural_device!='cpu':
        if config['mode'] not in {'recurrent','shared_attention','stage_attention'}:raise ValueError('Split acceleration supports correction families only')
        model.set_neural_device(neural_device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4)
    rng=np.random.default_rng(seed);best=np.inf;wait=0;history=[];best_epoch=0;start_epoch=1
    lastpath=path.with_name(path.name+'_resume.pt')
    if lastpath.exists():
        saved=torch.load(lastpath,map_location='cpu',weights_only=False);check_resume(saved,requested)
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['rng'];history=saved['history'];best=saved['best'];wait=saved['wait'];best_epoch=saved['best_epoch'];start_epoch=len(history)+1
    y=a['Y'][:,:7];weights=objective_weights(y[fit_ids]);tick=time.time()
    groups=config.get('sampling','uniform')
    if groups=='balanced':
        # All descriptors come from fitting covariates; no observation dates.
        valid=np.arange(a['G'].shape[1])[None,:]<a['lengths'][fit_ids,None]
        thermal=(a['G'][fit_ids]*valid).sum(1)/valid.sum(1).clip(1)
        descriptors=np.c_[thermal,a['X'][fit_ids,0,2],pd.to_datetime(env.iloc[fit_ids].sowing_date).dt.dayofyear.to_numpy()]
        cuts=[np.quantile(descriptors[:,j],[.33,.67]) for j in range(3)]
        labels=np.digitize(descriptors[:,0],cuts[0])*9+np.digitize(descriptors[:,1],cuts[1])*3+np.digitize(descriptors[:,2],cuts[2])
        _,inverse,counts=np.unique(labels,return_inverse=True,return_counts=True)
        probabilities=1/counts[inverse];probabilities/=probabilities.sum()
    for epoch in range(start_epoch,epochs+1) if wait<patience else []:
        model.train();started=time.time()
        order=rng.choice(fit_ids,len(fit_ids),replace=True,p=probabilities) if groups=='balanced' else rng.permutation(fit_ids)
        total=0
        for k in range(0,len(order),batch):
            ids=order[k:k+batch];optimizer.zero_grad(set_to_none=True)
            _,cdf=model(torch.as_tensor(features[ids]),torch.as_tensor(a['G'][ids]),metadata['maturation_gdd'])
            loss=sequence_crps_loss(cdf,torch.as_tensor(y[ids]),weights,torch.as_tensor(a['lengths'][ids]))
            if not torch.isfinite(loss):raise FloatingPointError(str(path)+' invalid loss')
            loss.backward();clip_gradients(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(ids)
        predicted=predict(model,features,a,tune_ids,metadata,batch)
        stage=[]
        for j in range(7):
            keep=y[tune_ids,j]>0
            if not keep.any():raise ValueError('No tuning support at stage '+str(j))
            stage.append(score_cdf(predicted[keep,j],y[tune_ids[keep],j])['crps'])
        value=float(np.mean(stage));history.append({'epoch':epoch,'calibration_crps':total/len(order),'tuning_crps':value,'seconds':time.time()-started})
        if value<best-1e-4:
            best=value;best_epoch=epoch;wait=0
            atomic_torch_save(dict(requested,metadata=metadata,state_dict=copy.deepcopy(model.state_dict()),best_epoch=epoch,tuning_crps=best),path.with_suffix('.pt'))
        else:wait+=1
        pd.DataFrame(history).to_csv(path.with_name(path.name+'_history.csv'),index=False)
        atomic_torch_save(dict(requested,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,
                        history=history,best=best,wait=wait,best_epoch=best_epoch),lastpath)
        print(json.dumps({'fit':str(path),'epoch':epoch,'tuning_crps':value,'seconds':history[-1]['seconds']}),flush=True)
        if wait>=patience:break
    report=dict(requested,training_neural_device=neural_device,tuning_crps=best,best_epoch=best_epoch,epochs_run=len(history),
                training_environments=len(fit_ids),tuning_environments=len(tune_ids),
                actual_updates=len(history)*int(np.ceil(len(fit_ids)/batch)),parameter_count=sum(p.numel() for p in model.parameters()),
                elapsed_seconds=sum(r['seconds'] for r in history),outer_test_read_for_selection=False,
                checkpoint_sha256=digest(path.with_suffix('.pt').read_bytes()))
    atomic_json_save(report,reportpath);validate_artifact(path,requested);lastpath.unlink(missing_ok=True)
    return report
