"""Causal stage-specific correction of temperature–photoperiod development.

The process reference and every correction share one parameter set across
all environments. Inputs are calibration-standardized causal weather features.
"""
import math
import numpy as np
import torch
from torch import nn
from .ordered_process import ordered_stage_cdf


def neural_position(position,device):
    # Keep dtype conversion on CPU: a simultaneous CPU-double → MPS-float
    # transfer corrupts the backward copy on the installed Apple backend.
    return position.float().to(device)


class TransferAttentionModel(nn.Module):
    def __init__(self, metadata, mode='stage_attention', bound=1., position='thermal', width=32):
        super().__init__()
        if mode not in {'process','recurrent','shared_attention','stage_attention'}:
            raise ValueError('Unknown correction mode')
        if position not in {'thermal','day'} or not 0<bound<=1:
            raise ValueError('Invalid correction bound or positional basis')
        self.mode,self.bound,self.position=mode,float(bound),position
        self.register_buffer('feature_mean',torch.tensor(metadata['feature_mean'],dtype=torch.float64))
        self.register_buffer('feature_std',torch.tensor(metadata['feature_std'],dtype=torch.float64))
        thresholds=np.asarray(metadata['threshold_gdd'][:7],dtype=float)
        if len(thresholds)!=7 or np.any(np.diff(np.r_[0,thresholds])<=0):
            raise ValueError('Seven positive increasing process thresholds required')
        gaps=np.diff(np.r_[0.,thresholds])/1000
        self.raw_gaps=nn.Parameter(torch.tensor(np.log(np.expm1(gaps)),dtype=torch.float64))
        noise=np.clip((np.asarray(metadata['noise_initial'][:7])-.03)/.77,.001,.999)
        self.raw_sigma=nn.Parameter(torch.tensor(np.log(noise/(1-noise)),dtype=torch.float64))
        self.raw_optimum=nn.Parameter(torch.tensor(0.,dtype=torch.float64))
        self.raw_maximum=nn.Parameter(torch.tensor(0.,dtype=torch.float64))
        self.raw_photo=nn.Parameter(torch.tensor(-3.,dtype=torch.float64))
        if mode!='process':
            self.encoder=nn.LSTM(18,width,batch_first=True)
            self.stage_embedding=nn.Embedding(7,8)
            self.correction=nn.Sequential(nn.Linear(32+width+8,32),nn.SiLU(),nn.Linear(32,1))
            nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)
            self.query=nn.Linear(3,32)
            self.stage_query=nn.Parameter(torch.randn(7,2,16)*.05)
            self.keys=nn.Linear(width,32);self.values=nn.Linear(width,32)
            self.current=nn.Linear(width,32)
            self.raw_decay=nn.Parameter(torch.zeros(7 if mode=='stage_attention' else 1,2))

    def set_neural_device(self,device):
        if self.mode!='process':
            for name in ['encoder','stage_embedding','correction','query','keys','values','current']:
                getattr(self,name).to(device)
            self.stage_query=nn.Parameter(self.stage_query.detach().to(device))
            self.raw_decay=nn.Parameter(self.raw_decay.detach().to(device))
        return self

    def thresholds(self):
        return torch.cumsum(torch.nn.functional.softplus(self.raw_gaps)+1e-8,0)*1000

    def sigma(self): return .03+.77*torch.sigmoid(self.raw_sigma)

    def cardinal_parameters(self):
        return 20+14*torch.sigmoid(self.raw_optimum),36+8*torch.sigmoid(self.raw_maximum)

    def process_progress(self,x,g):
        raw=x[:,:,:3].double()*self.feature_std+self.feature_mean
        temperature=(raw[:,:,0]+raw[:,:,1])/2
        optimum,maximum=self.cardinal_parameters()
        alpha=math.log(2)/torch.log((maximum-8)/(optimum-8))
        u=((temperature-8).clamp_min(1e-8)/(optimum-8))**alpha
        rate=(optimum-8)*(2*u-u.square()).clamp_min(0)
        rate=torch.where((temperature>8)&(temperature<maximum)&(g>0),rate,torch.zeros_like(rate))
        uncorrected=rate.cumsum(1)
        # A predicted progress window, independent of observed stage dates.
        threshold=self.thresholds()
        window=(uncorrected>=threshold[0])&(uncorrected<threshold[3])
        delay=1+torch.nn.functional.softplus(self.raw_photo)*(raw[:,:,2]-12.5).clamp_min(0)*window
        return (rate/delay).cumsum(1)

    def process_intensities(self,x,g):
        progress=self.process_progress(x,g)
        z=(torch.log(progress.clamp_min(1e-12))[:,:,None]-torch.log(self.thresholds())[None,None,:])/self.sigma()[None,None,:]
        log_survival=torch.special.log_ndtr(-z)
        log_survival=torch.where(progress[:,:,None]>0,log_survival,torch.zeros_like(log_survival))
        previous=torch.cat([torch.zeros_like(log_survival[:,:1]),log_survival[:,:-1]],1)
        return (previous-log_survival).clamp(0,20),progress

    def attention_queries(self,x):
        # Astronomical daylength and sowing harmonics are known at sowing.
        context=torch.cat([x[:,0,2:3],x[:,0,16:18]],1)
        query=self.query(context).reshape(len(x),1,2,16)
        if self.mode=='stage_attention':return query+self.stage_query[None,:,:,:]
        return query.expand(-1,7,-1,-1)

    def attention_context(self,x,hidden,progress):
        keys=self.keys(hidden).reshape(len(x),x.shape[1],2,16)
        values=self.values(hidden).reshape(len(x),x.shape[1],2,16)
        score=torch.einsum('njhd,nthd->ntjh',self.attention_queries(x),keys)/4
        position=progress/self.thresholds()[-1] if self.position=='thermal' else x[:,:,3].double()
        decay=torch.sigmoid(self.raw_decay)
        if len(decay)==1:decay=decay.expand(7,-1)
        # exp(-rho*(position_t-position_tau)); the current-position factor
        # cancels in prefix normalization. No future maximum is used.
        score=score+neural_position(position,score.device)[:,:,None,None]*decay[None,None,:,:]
        weight=torch.exp(score.clamp(-20,20))
        weight=weight.permute(0,2,3,1).contiguous()
        value=values.permute(0,2,3,1).contiguous()
        weighted=(weight[:,:,:,None,:]*value[:,None,:,:,:]).cumsum(-1)
        context=weighted/weight.cumsum(-1)[:,:,:,None,:]
        return context.permute(0,4,1,2,3).reshape(len(x),x.shape[1],7,32)

    def transition_intensities(self,x,g,km=None):
        prior,progress=self.process_intensities(x,g)
        if self.mode=='process':return prior
        neural_x=x.float().to(next(self.encoder.parameters()).device)
        hidden=self.encoder(neural_x)[0]
        if self.mode=='recurrent':context=self.current(hidden)[:,:,None,:].expand(-1,-1,7,-1)
        else:context=self.attention_context(neural_x,hidden,progress)
        embedding=self.stage_embedding(torch.arange(7,device=hidden.device))[None,None,:,:]
        linear=self.correction[0]
        value=torch.nn.functional.linear(context,linear.weight[:,:32],linear.bias)
        value=value+torch.nn.functional.linear(hidden,linear.weight[:,32:32+hidden.shape[2]])[:,:,None,:]
        value=value+torch.nn.functional.linear(embedding,linear.weight[:,32+hidden.shape[2]:])
        delta=self.bound*torch.tanh(self.correction[-1](self.correction[1](value)).squeeze(-1))
        return prior*torch.exp(delta.to('cpu').double())

    def forward(self,x,g,km=None):
        cdf=ordered_stage_cdf(self.transition_intensities(x,g,km))
        return cdf.sum(1)/7,cdf
