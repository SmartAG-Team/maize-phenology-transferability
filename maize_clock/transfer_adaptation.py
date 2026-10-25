"""Paired local calibration with identical German weather-anchor exposures."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .transfer_fit import restore,predict,anchor_schedule,clip_gradients
from .extension_boosting import LazySequenceFeatures
from .extensions import sequence_crps_loss
from .experiment import score_cdf
from .transfer_artifacts import dependency_fingerprint,file_sha256,atomic_torch_save,atomic_json_save

COMPATIBLE=['emergence','stem_elongation','tasseling']


def local_masks(env,held_site=None):
    primary=env.collection.eq('station').to_numpy()&env.analysis_role.eq('primary_single_sowing').to_numpy()
    remaining=primary if held_site is None else primary&~env.site_id.eq(held_site).to_numpy()
    masks={'calibration':remaining&(env.year<=2003).to_numpy(),
           'tuning':remaining&env.year.between(2004,2005).to_numpy(),
           'test':primary&(env.year>=2006).to_numpy() if held_site is None else primary&env.site_id.eq(held_site).to_numpy()}
    if held_site is not None and any(masks[k].sum()<n for k,n in [('calibration',6),('tuning',2),('test',3)]):raise ValueError('Unsupported station fold')
    return masks


def observed_targets(env,targets):
    y=np.zeros((len(env),7),int);lookup=dict(zip(env.environment_id_model,env.array_index))
    if len(lookup)!=len(env):raise ValueError('Nonunique external environment IDs')
    for row in targets.itertuples(index=False):
        if row.analysis_role!='primary_single_sowing' or row.endpoint not in COMPATIBLE:continue
        i=lookup[row.environment_id_model];j=COMPATIBLE.index(row.endpoint)
        if y[i,j]!=0 and y[i,j]!=row.elapsed_days:raise ValueError('Conflicting compatible observations')
        if not 0<row.elapsed_days<=366:raise ValueError('Invalid observed day')
        y[i,j]=row.elapsed_days
    return y


def adapt(base_checkpoint,german,german_env,external,external_env,targets,path,seed,update_mode='efficient',held_site=None,epochs=40,patience=8):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if update_mode not in {'efficient','full'}:raise ValueError('Unknown local update mode')
    torch.set_num_threads(1);torch.manual_seed(seed)
    model,saved=restore(base_checkpoint);teacher,_=restore(base_checkpoint);metadata=saved['metadata']
    masks=local_masks(external_env,held_site);y=observed_targets(external_env,targets)
    cal=np.flatnonzero(masks['calibration']);tune=np.flatnonzero(masks['tuning']);test=np.flatnonzero(masks['test'])
    if np.any((y[cal,:3]>0).sum(0)<1):raise ValueError('Missing local calibration support')
    german_ids=np.flatnonzero(german_env.dataset_id.eq('dwd_annual').to_numpy()&(german_env.year<=2017).to_numpy()&(german['lengths']==366))
    schedule=anchor_schedule(german_ids,seed,40)
    panel=schedule['panel'];positions={value:i for i,value in enumerate(panel)}
    draw=np.array([[positions[value] for value in batch] for batch in schedule['batches']])
    gx=LazySequenceFeatures(german,metadata,german_env,history=True)
    anchor_x=gx[panel];anchor_g=german['G'][panel]
    teacher_cdf=predict(teacher,gx,german,panel,metadata).astype(np.float32)
    cx=LazySequenceFeatures(external,metadata,external_env,history=True)
    signature={'source_checkpoint_sha256':hashlib.sha256(Path(base_checkpoint).read_bytes()).hexdigest(),
        'external_weather_sha256':hashlib.sha256(external['X'].tobytes()+external['G'].tobytes()+external['lengths'].tobytes()).hexdigest(),
        'compatible_label_sha256':hashlib.sha256(y.tobytes()).hexdigest(),
        'anchor_schedule_sha256':hashlib.sha256(panel.tobytes()+schedule['batches'].tobytes()).hexdigest(),
        'update_mode':update_mode,'held_site':held_site,'seed':seed,'maximum_epochs':epochs,'patience':patience,
        'source_code_sha256':dependency_fingerprint()+'_'+file_sha256(__file__)}
    reportpath=path.with_suffix('.json')
    if reportpath.exists():
        report=json.loads(reportpath.read_text())
        if report['signature']!=signature:raise ValueError('Changed local-calibration run cannot resume')
        if file_sha256(path.with_suffix('.pt'))!=report['checkpoint_sha256']:raise ValueError('Changed local checkpoint bytes')
        bundle=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)
        if bundle.get('local_calibration')!=signature:raise ValueError('Local checkpoint signature mismatch')
        return report
    if update_mode=='efficient':
        if saved['config']['mode'] in {'standalone_photo','process'}:
            for name,p in model.named_parameters():p.requires_grad_(name in {'raw_optimum','raw_maximum','raw_photo'})
        else:
            for name,p in model.named_parameters():p.requires_grad_(name.startswith(('correction.','query.','stage_embedding.')) or name in {'stage_query','raw_decay'})
    parameters=[p for p in model.parameters() if p.requires_grad]
    lr=.01 if saved['config']['mode'] in {'standalone_photo','process'} else .0001
    optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=1e-4)
    counts=(y[cal,:3]>0).sum(0);weights=torch.tensor(np.r_[len(cal)/counts/3,np.zeros(4)],dtype=torch.float32)
    history=[];best=np.inf;wait=0;best_epoch=0;start=time.time()
    np.savez_compressed(path.with_name(path.name+'_anchors.npz'),panel=panel,batches=schedule['batches'],teacher_cdf=teacher_cdf)
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        _,cdf=model(torch.tensor(cx[cal]),torch.tensor(external['G'][cal]),metadata['maturation_gdd'])
        target_loss=sequence_crps_loss(cdf,torch.tensor(y[cal]),weights,torch.tensor(external['lengths'][cal]))
        pick=draw[epoch-1]
        _,anchored=model(torch.tensor(anchor_x[pick]),torch.tensor(anchor_g[pick]),metadata['maturation_gdd'])
        penalty=(anchored-torch.tensor(teacher_cdf[pick])).square().sum(2).mean()
        loss=target_loss+.25*penalty
        if not torch.isfinite(loss):raise FloatingPointError('Invalid local-calibration objective')
        loss.backward();clip_gradients(parameters,1.);optimizer.step()
        c=predict(model,cx,external,tune,metadata)
        value=float(np.mean([score_cdf(c[y[tune,j]>0,j],y[tune[y[tune,j]>0],j])['crps'] for j in range(3)]))
        history.append({'epoch':epoch,'local_crps':float(target_loss.detach()),'anchor_penalty':float(penalty.detach()),'tuning_crps':value})
        if value<best-1e-4:
            best=value;best_epoch=epoch;wait=0
            bundle=dict(saved,state_dict={key:value.detach().cpu().clone() for key,value in model.state_dict().items()},
                        local_calibration=signature,best_epoch=epoch,tuning_crps=best)
            atomic_torch_save(bundle,path.with_suffix('.pt'))
        else:wait+=1
        pd.DataFrame(history).to_csv(path.with_name(path.name+'_history.csv'),index=False)
        if wait>=patience:break
    report={'signature':signature,'best_epoch':best_epoch,'tuning_crps':best,'epochs_run':len(history),'learning_rate':lr,
        'calibration_environments':len(cal),'tuning_environments':len(tune),'test_environments':len(test),
        'calibration_indices':cal.tolist(),'tuning_indices':tune.tolist(),'test_indices':test.tolist(),
        'anchor_panel':256,'anchors_per_update':64,'anchor_weight':.25,'actual_anchor_exposures':64*len(history),
        'actual_local_exposures':len(cal)*len(history),'updated_parameter_count':sum(p.numel() for p in parameters),
        'test_labels_used_for_selection':False,'elapsed_seconds':time.time()-start,
        'checkpoint_sha256':hashlib.sha256(path.with_suffix('.pt').read_bytes()).hexdigest()}
    atomic_json_save(report,reportpath);return report
