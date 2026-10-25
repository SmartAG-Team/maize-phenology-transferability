"""Positive phase-dependent clock with correlated sparse GP function draws.

The neural encoder/kernel are point estimates. Variational uncertainty is
conditional on them. RFF prior draws approximate the kernel; conditioning uses
its exact covariance. The likelihood objective is explicitly composite.
"""
import math
from typing import List
import numpy as np
import torch
from torch import nn

@torch.jit.script
def integrate_clock(mult:torch.Tensor,g:torch.Tensor,km:float,emergence:float,flowering:float):
    # mult is samples x batch x days x phases. Function draw persists over time.
    z=torch.zeros_like(mult[:,:,0,0]);values=torch.jit.annotate(List[torch.Tensor],[])
    for t in range(g.size(1)):
        se=torch.sigmoid((z-emergence)/.03);sf=torch.sigmoid((z-flowering)/.03)
        rate=(1.-se)*mult[:,:,t,0]+(se-sf)*mult[:,:,t,1]+sf*mult[:,:,t,2]
        z=z+g[:,t].unsqueeze(0)*rate/km
        values.append(z)
    return torch.stack(values,dim=-1)

class SparseFunctionGP(nn.Module):
    def __init__(self,inducing,feature_dim,raw=True,neural=True,rff=256):
        super().__init__();self.raw=raw;self.neural=neural;self.rff=rff
        self.register_buffer('inducing',inducing.clone())
        dim=(3 if raw else 0)+(feature_dim if neural else 0)
        self.raw_length=nn.Parameter(torch.zeros(dim));self.raw_amplitude=nn.Parameter(torch.tensor(-1.5))
        m=len(inducing);self.q_mean=nn.Parameter(torch.zeros(m))
        rawchol=torch.zeros(m,m);rawchol.diagonal().fill_(math.log(math.expm1(.999)))
        self.q_raw_chol=nn.Parameter(rawchol)
        self.register_buffer('omega',torch.randn(rff,dim))
        self.register_buffer('phase',2*math.pi*torch.rand(rff))
    @property
    def length(self):return .25+1.75*torch.sigmoid(self.raw_length)
    @property
    def amplitude(self):return .1+1.9*torch.sigmoid(self.raw_amplitude)
    @property
    def q_chol(self):return torch.tril(self.q_raw_chol,-1)+torch.diag(torch.nn.functional.softplus(self.q_raw_chol.diagonal())+1e-3)
    def embedded(self,x,encoder):
        pieces=[]
        if self.raw:pieces.append(x)
        if self.neural:pieces.append(encoder(x))
        return torch.cat(pieces,-1)/self.length
    def kernel(self,x,y,encoder):
        a,b=self.embedded(x,encoder),self.embedded(y,encoder)
        d=(a.square().sum(-1,keepdim=True)+b.square().sum(-1).unsqueeze(0)-2*a@b.T).clamp_min(0)
        return self.amplitude.square()*torch.exp(-.5*d)
    def draw(self,x,samples,encoder,mean_only=False):
        fx=self.embedded(x,encoder);fz=self.embedded(self.inducing,encoder)
        dist=(fx.square().sum(-1,keepdim=True)+fz.square().sum(-1).unsqueeze(0)-2*fx@fz.T).clamp_min(0)
        kxz=self.amplitude.square()*torch.exp(-.5*dist)
        dz=(fz.square().sum(-1,keepdim=True)+fz.square().sum(-1).unsqueeze(0)-2*fz@fz.T).clamp_min(0)
        kzz=self.amplitude.square()*torch.exp(-.5*dz)
        chol=torch.linalg.cholesky(kzz+torch.eye(len(fz),device=x.device)*1e-5)
        if mean_only:
            coeff=torch.linalg.solve_triangular(chol.T,self.q_mean[:,None],upper=True)
            return (kxz@coeff).T.expand(samples,-1)
        weights=torch.randn(samples,self.rff,device=x.device)
        phi_x=math.sqrt(2/self.rff)*self.amplitude*torch.cos(fx@self.omega.T+self.phase)
        phi_z=math.sqrt(2/self.rff)*self.amplitude*torch.cos(fz@self.omega.T+self.phase)
        prior_x=weights@phi_x.T;prior_z=weights@phi_z.T
        white=self.q_mean[None,:]+torch.randn(samples,len(fz),device=x.device)@self.q_chol.T
        u=white@chol.T
        coeff=torch.cholesky_solve((u-prior_z).T,chol)
        return prior_x+(kxz@coeff).T
    def draw_independent_days(self,x,samples,encoder,days):
        # Each relative day receives a newly sampled whole response function.
        # The marginal q(f(x)) matches draw(); off-day covariance is exactly absent.
        batch=len(x)//days
        fx=self.embedded(x,encoder);fz=self.embedded(self.inducing,encoder)
        dist=(fx.square().sum(-1,keepdim=True)+fz.square().sum(-1).unsqueeze(0)-2*fx@fz.T).clamp_min(0)
        kxz=(self.amplitude.square()*torch.exp(-.5*dist)).reshape(batch,days,-1)
        dz=(fz.square().sum(-1,keepdim=True)+fz.square().sum(-1).unsqueeze(0)-2*fz@fz.T).clamp_min(0)
        kzz=self.amplitude.square()*torch.exp(-.5*dz)
        chol=torch.linalg.cholesky(kzz+torch.eye(len(fz),device=x.device)*1e-5)
        weights=torch.randn(samples,days,self.rff,device=x.device)
        phi_x=(math.sqrt(2/self.rff)*self.amplitude*torch.cos(fx@self.omega.T+self.phase)).reshape(batch,days,self.rff)
        phi_z=math.sqrt(2/self.rff)*self.amplitude*torch.cos(fz@self.omega.T+self.phase)
        prior_x=torch.einsum('stf,btf->sbt',weights,phi_x)
        prior_z=torch.einsum('stf,mf->stm',weights,phi_z)
        white=self.q_mean[None,None,:]+torch.randn(samples,days,len(fz),device=x.device)@self.q_chol.T
        u=white@chol.T
        coeff=torch.cholesky_solve((u-prior_z).reshape(-1,len(fz)).T,chol).T.reshape(samples,days,len(fz))
        return (prior_x+torch.einsum('stm,btm->sbt',coeff,kxz)).reshape(samples,-1)
    def kl(self):
        l=self.q_chol
        return .5*(l.square().sum()+self.q_mean.square().sum()-len(self.q_mean)-2*torch.log(l.diagonal()).sum())

class DevelopmentClock(nn.Module):
    def __init__(self,thresholds,mean,std,inducing,width=16,rff=256,mode='hybrid',sigma=None):
        super().__init__();self.mode=mode
        self.emergence_anchor=float(thresholds[0]);self.flowering_anchor=float(thresholds[3])
        self.register_buffer('thresholds',torch.tensor(thresholds,dtype=torch.float32))
        self.register_buffer('feature_mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('feature_std',torch.tensor(std,dtype=torch.float32))
        self.encoder=nn.Sequential(nn.Linear(3,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,4),nn.Tanh())
        self.correction=nn.Linear(4,3);nn.init.normal_(self.correction.weight,std=1e-3);nn.init.zeros_(self.correction.bias)
        if mode in ['hybrid','raw_gp','unsafe_kernel','shared_phase','independent_days']:
            self.phase_models=nn.ModuleList([SparseFunctionGP(inducing,4,raw=mode!='unsafe_kernel',neural=mode!='raw_gp',rff=rff) for _ in range(1 if mode=='shared_phase' else 3)])
        else:self.phase_models=nn.ModuleList()
        init=np.array(sigma if sigma is not None else [0.12]*len(thresholds));init=np.clip((init-.02)/.78,.001,.999)
        self.raw_sigma=nn.Parameter(torch.tensor(np.log(init/(1-init)),dtype=torch.float32))
        self.we_opt=nn.Parameter(torch.tensor(0.));self.we_max=nn.Parameter(torch.tensor(0.));self.photo=nn.Parameter(torch.tensor(0.))
    @property
    def sigma(self):return .02+.78*torch.sigmoid(self.raw_sigma)
    def forward(self,x,g,km:float,samples=2,mean_only=False):
        raw=x;norm=(x-self.feature_mean)/self.feature_std;batch,days,_=norm.shape
        if self.mode=='gdd':return (torch.cumsum(g,dim=1)/km).unsqueeze(0).expand(samples,-1,-1)
        if self.mode=='wang_engel':
            t=(raw[:,:,0]+raw[:,:,1])/2;opt=20.+12.*torch.sigmoid(self.we_opt);maximum=opt+5.+12.*torch.sigmoid(self.we_max)
            alpha=torch.log(torch.tensor(2.,device=x.device))/torch.log((maximum-8.)/(opt-8.))
            a=((t-8.).clamp_min(0)/(opt-8.)).pow(alpha)
            response=(2*a-a*a).clamp_min(0);response=torch.where(t<maximum,response,torch.zeros_like(response))
            phot=torch.exp(.15*torch.tanh(self.photo)*(raw[:,:,2]-14.))
            return torch.cumsum(response*(opt-8.)*phot,dim=1).unsqueeze(0).expand(samples,-1,-1)/km
        if self.mode=='neural':
            values=self.correction(self.encoder(norm.reshape(-1,3))).unsqueeze(0).expand(samples,-1,-1)
        else:
            draws=[p.draw_independent_days(norm.reshape(-1,3),samples,self.encoder,days) if self.mode=='independent_days' and not mean_only else p.draw(norm.reshape(-1,3),samples,self.encoder,mean_only) for p in self.phase_models]
            if len(draws)==1:draws=draws*3
            values=torch.stack(draws,dim=-1)
        mult=2.*torch.sigmoid(values.reshape(samples,batch,days,3))
        return integrate_clock(mult,g,km,self.emergence_anchor,self.flowering_anchor)
    def cdf(self,state):
        state=state.to(device='cpu').double()
        thresholds=self.thresholds.to(device='cpu').double()
        sigma=self.sigma.to(device='cpu').double()
        a=(torch.log(state.clamp_min(1e-12))[:,:,None,:]-torch.log(thresholds)[None,None,:,None])/sigma[None,None,:,None]
        return torch.special.ndtr(a).mean(0)
    def kl(self):
        return sum([p.kl() for p in self.phase_models],torch.tensor(0.,device=self.thresholds.device))

def event_loss(state,y,thresholds,sigma,weights):
    # Expected composite negative log-likelihood across function samples.
    samples,batch,days=state.shape
    index=(y.clamp_min(1)-1).clamp_max(days-1)
    current=state.gather(2,index[None,:,:].expand(samples,-1,-1))
    previous=torch.cat([torch.zeros_like(state[:,:,:1]),state[:,:,:-1]],dim=-1).gather(2,index[None,:,:].expand(samples,-1,-1))
    # Float64 at observed dates avoids subtracting nearly equal float32 CDFs.
    current,previous=current.double(),previous.double()
    thresholds,sigma=thresholds.double(),sigma.double()
    b=(torch.log(current.clamp_min(1e-12))-torch.log(thresholds)[None,None,:])/sigma[None,None,:]
    a=(torch.log(previous.clamp_min(1e-12))-torch.log(thresholds)[None,None,:])/sigma[None,None,:]
    logb=torch.special.log_ndtr(b);loga=torch.special.log_ndtr(a)
    upper=torch.special.log_ndtr(-a);lower=torch.special.log_ndtr(-b)
    # Choose the tail avoiding cancellation of nearly-one CDFs.
    maxlog=torch.where(b<0,logb,upper);minlog=torch.where(b<0,loga,lower)
    diff=(minlog-maxlog).clamp_max(0.)
    finite_logprob=maxlog+torch.log(-torch.expm1(diff.clamp_max(-1e-15)))
    # Exact equal states have zero event mass; the explicit score floor is constant.
    logprob=torch.where(diff<0,finite_logprob,torch.full_like(diff,math.log(1e-12))).clamp_min(math.log(1e-12))
    mask=y>0
    return -(logprob.mean(0)*mask*weights[None,:]).sum()/batch
