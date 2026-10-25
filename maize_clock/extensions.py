"""Additional causal sequence models; original experiments remain immutable."""
import copy,hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .references import CausalConv,sequence_features,hazard_cdf
from .experiment import load_data,experiment_meta,objective_weights,score_cdf
from .model import event_loss

class SequenceDevelopmentModel(nn.Module):
    def __init__(self,thresholds,sigma,encoder='lstm',output='clock',width=16,learn_thresholds=True,input_size=5,context_thresholds=False,thermal_gate=True):
        super().__init__();self.encoder_name=encoder;self.output=output;self.width=width;self.thermal_gate=thermal_gate
        if encoder=='lstm':self.encoder=nn.LSTM(input_size,width,batch_first=True)
        elif encoder=='gru':self.encoder=nn.GRU(input_size,width,batch_first=True)
        elif encoder=='tcn':self.encoder=nn.Sequential(nn.Conv1d(input_size,width,1),*[CausalConv(width,d) for d in [1,2,4,8]])
        else:raise ValueError('Unknown encoder: '+encoder)
        thresholds=np.asarray(thresholds,dtype=np.float32)
        if len(thresholds)<7 or np.any(np.diff(thresholds[:7])<=0):raise ValueError('Primary thresholds must increase')
        self.register_buffer('fixed_thresholds',torch.tensor(thresholds))
        gaps=np.diff(np.r_[0.,thresholds[:7]])
        self.raw_gaps=nn.Parameter(torch.tensor(np.log(np.expm1(gaps)),dtype=torch.float32),requires_grad=learn_thresholds)
        self.raw_aux=nn.Parameter(torch.log(torch.tensor(thresholds[7:])),requires_grad=learn_thresholds)
        self.learn_thresholds=learn_thresholds
        self.context_thresholds=context_thresholds
        if context_thresholds:
            if input_size!=18 or output!='clock':raise ValueError('Context thresholds require the 18-input clock')
            self.threshold_network=nn.Sequential(nn.Linear(4,16),nn.SiLU(),nn.Linear(16,7))
            with torch.no_grad():self.threshold_network[-1].weight.zero_();self.threshold_network[-1].bias.zero_()
        initial=np.clip((np.asarray(sigma)-.02)/.78,.001,.999)
        self.raw_sigma=nn.Parameter(torch.tensor(np.log(initial/(1-initial)),dtype=torch.float32),requires_grad=output=='clock')
        self.head=nn.Linear(width,1 if output=='clock' else len(thresholds))
        with torch.no_grad():
            self.head.weight.mul_(.01);self.head.bias.zero_()
            if output in ['ordered_markov','independent_rate']:
                initial=np.r_[1/np.diff(np.r_[0.,thresholds[:7]]),1/thresholds[7:]] if output=='ordered_markov' else 1/thresholds
                self.head.bias.copy_(torch.tensor(np.log(initial),dtype=torch.float32))
    def thresholds(self,context=None):
        if not self.learn_thresholds:return self.fixed_thresholds
        gaps=torch.nn.functional.softplus(self.raw_gaps)+1e-5
        if context is not None and self.context_thresholds:
            delta=.3*torch.tanh(self.threshold_network(context))
            gaps=gaps[None,:]*torch.exp(delta)
            primary=torch.cumsum(gaps,1);primary=primary/primary[:,-1:]
            auxiliary=torch.exp(self.raw_aux)[None,:].expand(len(context),-1)
            return torch.cat([primary,auxiliary],1)
        primary=torch.cumsum(gaps,0);primary=primary/primary[-1]
        return torch.cat([primary,torch.exp(self.raw_aux)])
    def sigma(self):return .02+.78*torch.sigmoid(self.raw_sigma)
    def encoder_logits(self,x):
        h=self.encoder(x.transpose(1,2)).transpose(1,2) if self.encoder_name=='tcn' else self.encoder(x)[0]
        return self.head(h)
    def transition_intensities(self,x,g,km):
        if self.output not in ['ordered_markov','independent_rate']:raise ValueError('This output has no thermal-rate transition process')
        thermal=g if self.thermal_gate else torch.full_like(g,8.)
        return torch.exp(self.encoder_logits(x).to('cpu').double().clamp(-12,8))*thermal.to('cpu').double()[:,:,None]/float(km)
    def forward(self,x,g,km):
        logits=self.encoder_logits(x)
        if self.output=='hazard':return None,hazard_cdf(logits)
        if self.output in ['ordered_markov','independent_rate']:
            thermal=g if self.thermal_gate else torch.full_like(g,8.)
            intensity=torch.exp(logits.to('cpu').double().clamp(-12,8))*thermal.to('cpu').double()[:,:,None]/float(km)
            if self.output=='ordered_markov':
                from .ordered_process import ordered_stage_cdf
                main=ordered_stage_cdf(intensity[:,:,:7])
                extra=-torch.expm1(-torch.cumsum(intensity[:,:,7:],1)).transpose(1,2)
                cdf=torch.cat([main,extra],1)
            else:cdf=-torch.expm1(-torch.cumsum(intensity,1)).transpose(1,2)
            return cdf[:,:7].sum(1)/7,cdf
        thermal=g if self.thermal_gate else torch.full_like(g,8.)
        increments=thermal*(2*torch.sigmoid(logits.squeeze(-1)))/float(km)
        state=torch.cumsum(increments,1)
        thresholds=self.thresholds(x[:,0,-4:]) if self.context_thresholds else self.thresholds()[None,:]
        a=(torch.log(state.clamp_min(1e-12))[:,None,:]-torch.log(thresholds)[:,:,None])/self.sigma()[None,:,None]
        cdf=torch.where(state[:,None,:]>0,.5*(1+torch.erf(a/math.sqrt(2.))),torch.zeros_like(a))
        return state,cdf

def sequence_crps_loss(cdf,y,weights,lengths=None):
    days=torch.arange(1,cdf.shape[2]+1,device=cdf.device)
    truth=(days[None,None,:]>=y[:,:,None]).to(cdf.dtype)
    error=(cdf-truth).square()
    if lengths is not None:error=error*(days[None,None,:]<=lengths[:,None,None])
    return (error.sum(2)*(y>0)*weights).sum()/len(y)

def predict(model,x,g,indices,km,batch=128):
    result=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(indices),batch):
            ii=indices[start:start+batch]
            device=next(model.parameters()).device
            _,c=model(torch.as_tensor(x[ii],device=device),torch.as_tensor(g[ii],device=device),km);result.append(np.maximum.accumulate(np.clip(c.cpu().numpy(),0.,1.),axis=2))
    return np.concatenate(result) if result else np.empty((0,len(model.fixed_thresholds),366))

CANDIDATES=[
 dict(name='lstm_clock_fixed_nll',encoder='lstm',output='clock',learn_thresholds=False,objective='nll',width=16),
 dict(name='lstm_clock_learned_nll',encoder='lstm',output='clock',learn_thresholds=True,objective='nll',width=16),
 dict(name='lstm_clock_crps_w16',encoder='lstm',output='clock',learn_thresholds=True,objective='crps',width=16),
 dict(name='lstm_clock_crps_w32',encoder='lstm',output='clock',learn_thresholds=True,objective='crps',width=32),
 dict(name='gru_clock_crps_w16',encoder='gru',output='clock',learn_thresholds=True,objective='crps',width=16),
 dict(name='gru_clock_crps_w32',encoder='gru',output='clock',learn_thresholds=True,objective='crps',width=32),
 dict(name='tcn_clock_crps_w32',encoder='tcn',output='clock',learn_thresholds=True,objective='crps',width=32),
 dict(name='lstm_hazard_crps_w16',encoder='lstm',output='hazard',learn_thresholds=False,objective='crps',width=16),
 dict(name='lstm_hazard_crps_w32',encoder='lstm',output='hazard',learn_thresholds=False,objective='crps',width=32),
 dict(name='gru_hazard_crps_w16',encoder='gru',output='hazard',learn_thresholds=False,objective='crps',width=16),
 dict(name='gru_hazard_crps_w32',encoder='gru',output='hazard',learn_thresholds=False,objective='crps',width=32),
 dict(name='tcn_hazard_crps_w32',encoder='tcn',output='hazard',learn_thresholds=False,objective='crps',width=32),
]

CANDIDATES.extend([dict(name=f'{encoder}_{output}_history_crps_w32',encoder=encoder,output=output,learn_thresholds=output=='clock',objective='crps',width=32,input_size=18,feature_scheme='history') for encoder in ['lstm','gru'] for output in ['clock','hazard']])

CANDIDATES.extend([dict(name='tcn_clock_history_crps_w32',encoder='tcn',output='clock',learn_thresholds=True,objective='crps',width=32,input_size=18,feature_scheme='history')])
CANDIDATES.extend([dict(name=f'{encoder}_clock_structured_crps_w32',encoder=encoder,output='clock',learn_thresholds=True,objective='crps',width=32,input_size=18,feature_scheme='history',context_thresholds=True) for encoder in ['lstm','tcn']])

CANDIDATES.extend([dict(name=f'{encoder}_ordered_{scheme}_crps_w32',encoder=encoder,output='ordered_markov',learn_thresholds=False,objective='crps',width=32,input_size=18 if scheme=='history' else 5,feature_scheme=scheme,thermal_gate=True) for encoder,scheme in [('lstm','daily'),('lstm','history'),('tcn','daily')]])

CANDIDATES.extend([dict(c,name=c['name']+'_lr05',learning_rate=.0005) for c in CANDIDATES if c['output']=='ordered_markov'])

def fit_candidate(root,config,seed=17,epochs=80,patience=10,batch=128,device='cpu',threads=1):
    root=Path(root);out=root/'results/extensions/checkpoints';out.mkdir(parents=True,exist_ok=True)
    tag=config['name']+f'_s{seed}';reportpath=out/(tag+'_report.json')
    if reportpath.exists():return json.loads(reportpath.read_text())
    torch.set_num_threads(threads);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    a,env,meta=load_data(root)
    if config.get('experiment')=='buffered_north':
        from .extension_regions import buffered_region
        env=buffered_region(root,env)
    m,trainmask=experiment_meta(a,env,meta,'main')
    if config.get('feature_scheme')=='history':
        from .extension_boosting import LazySequenceFeatures,add_context_normalizers
        m=add_context_normalizers(m,env,trainmask);x=LazySequenceFeatures(a,m,env,history=True)
        if config.get('mask_geography',False):m['history_mask_geography']=True
    else:x=sequence_features(a,m)
    y=a['Y'][:,m['retained']]
    ii=np.flatnonzero(trainmask);vi=np.flatnonzero(env.new_partition.eq('validation').to_numpy()&(a['lengths']==366))
    model=SequenceDevelopmentModel(m['threshold_fraction'],m['noise_initial'],thermal_gate=config.get('thermal_gate',True),context_thresholds=config.get('context_thresholds',False),input_size=config.get('input_size',5),**{k:config[k] for k in ['encoder','output','width','learn_thresholds']})
    if config['output']=='hazard':
        with torch.no_grad():
            med=np.array([np.median(y[ii,j][y[ii,j]>0]) for j in range(y.shape[1])]);p=np.log(2.)/med
            model.head.bias.copy_(torch.tensor(np.log(p/(1-p)),dtype=torch.float32))
    if device=='mps':torch.mps.set_per_process_memory_fraction(.25)
    model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=config.get('learning_rate',.001),weight_decay=1e-4)
    weights=objective_weights(y[ii]);history=[];best=np.inf;wait=0;best_epoch=0;start=time.time()
    for epoch in range(1,epochs+1):
        model.train();tick=time.time();order=rng.permutation(ii);total=0.
        for k in range(0,len(order),batch):
            ix=order[k:k+batch];yy=torch.as_tensor(y[ix],dtype=torch.long)
            T=366 if config['objective']=='crps' else int(yy.max())
            xx=torch.as_tensor(x[ix,:T],device=device);gg=torch.as_tensor(a['G'][ix,:T],device=device);opt.zero_grad(set_to_none=True)
            z,c=model(xx,gg,m['maturation_gdd'])
            if config['objective']=='crps':loss=sequence_crps_loss(c.cpu(),yy,weights,torch.as_tensor(a['lengths'][ix]))
            elif config['output']=='clock':loss=event_loss(z[None].cpu(),yy,model.thresholds().cpu(),model.sigma().cpu(),weights)
            else:raise ValueError('Undeclared loss')
            if config['output']=='clock' and config['learn_thresholds']:
                loss=loss+.01*(torch.log(model.thresholds()/model.fixed_thresholds).square().mean()).cpu()
            if not torch.isfinite(loss):raise FloatingPointError(f'{tag} epoch {epoch}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())*len(ix)
        c=predict(model,x,a['G'],vi,m['maturation_gdd'])
        scores=[]
        for j in range(7):
            ok=(y[vi,j]>0)&env.iloc[vi].dataset_id.eq('dwd_annual').to_numpy()
            scores.append(score_cdf(c[ok,j],y[vi[ok],j])['crps'])
        val=float(np.mean(scores));row={'epoch':epoch,'training_objective':total/len(ii),'validation_crps':val,'seconds':time.time()-tick}
        history.append(row);print(json.dumps({'tag':tag,**row}),flush=True)
        if val<best-1e-4:
            best=val;wait=0;best_epoch=epoch
            torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'training_device':device,'config':config,'metadata':m,'seed':seed,'best_epoch':epoch,'validation_crps':best},out/(tag+'.pt'))
        else:wait+=1
        pd.DataFrame(history).to_csv(out/(tag+'_history.csv'),index=False)
        if device=='mps':
            torch.mps.synchronize();torch.mps.empty_cache()
        if wait>=patience:break
    report={'tag':tag,'training_device':device,'training_threads':threads,'config':config,'seed':seed,'validation_crps':best,'best_epoch':best_epoch,'epochs_run':len(history),'training_environments':len(ii),'training_environment_hash':hashlib.sha256('\n'.join(env.iloc[ii].environment_id_model).encode()).hexdigest(),'tuning_environments':len(vi),'held_out_read_for_selection':False,'elapsed_seconds':time.time()-start}
    reportpath.write_text(json.dumps(report,indent=2)+'\n');return report

def restore_candidate(path):
    b=torch.load(path,map_location='cpu',weights_only=False);m=b['metadata'];config=b['config']
    model=SequenceDevelopmentModel(m['threshold_fraction'],m['noise_initial'],thermal_gate=config.get('thermal_gate',True),context_thresholds=config.get('context_thresholds',False),input_size=config.get('input_size',5),**{k:config[k] for k in ['encoder','output','width','learn_thresholds']})
    model.load_state_dict(b['state_dict']);model.eval();return model,b

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--candidate');p.add_argument('--seed',type=int,default=17);p.add_argument('--epochs',type=int,default=80);args=p.parse_args()
    configs=[c for c in CANDIDATES if args.candidate is None or c['name']==args.candidate]
    if not configs:raise ValueError('Unknown declared candidate')
    for config in configs:fit_candidate(args.root,config,args.seed,epochs=args.epochs)
