"""Vectorized sequential-stage probabilities with thermal-gated transitions."""
import torch

def ordered_stage_cdf(intensity):
    """Seven forward states, one transition per day, final state absorbing.

    Log-space prefix sums solve the exact daily forward recursion. Finite
    sentinels underflow to zero and avoid undefined gradients for empty prefixes.
    """
    intensity=intensity.double()
    log_stay=-intensity
    log_event=torch.where(intensity>0,torch.log((-torch.expm1(-intensity)).clamp_min(1e-30)),torch.full_like(intensity,-1e30))
    previous=torch.cumsum(log_stay[:,:,0],1);states=[previous]
    for j in range(1,intensity.shape[2]+1):
        initial=torch.zeros_like(previous[:,:1]) if j==1 else torch.full_like(previous[:,:1],-1e30)
        shifted=torch.cat([initial,previous[:,:-1]],1)
        incoming=shifted+log_event[:,:,j-1]
        if j<intensity.shape[2]:
            survival=torch.cumsum(log_stay[:,:,j],1)
            previous=survival+torch.logcumsumexp(incoming-survival,1)
        else:previous=torch.logcumsumexp(incoming,1)
        states.append(previous)
    probability=torch.exp(torch.stack(states,2))
    probability=probability/probability.sum(2,keepdim=True)
    cdf=torch.flip(torch.cumsum(torch.flip(probability[:,:,1:],dims=[2]),2),dims=[2])
    return cdf.transpose(1,2).clamp(0.,1.)


def sample_forward_dates(intensity,draws=512,seed=52):
    """Sample the same one-transition-per-day process; zero marks unreached stages."""
    import numpy as np
    intensity=np.asarray(intensity,float)
    if intensity.ndim!=3 or intensity.shape[2]!=7 or np.any(intensity<0) or not np.isfinite(intensity).all():raise ValueError('Expected finite nonnegative N×T×7 intensity')
    n,t,_=intensity.shape;probability=-np.expm1(-intensity);rng=np.random.default_rng(seed)
    state=np.zeros((n,draws),np.uint8);dates=np.zeros((n,draws,7),np.int16);environments=np.arange(n)[:,None]
    for day in range(t):
        p=probability[environments,day,np.minimum(state,6)]
        advance=(state<7)&(rng.random((n,draws))<p)
        i,j=np.nonzero(advance);dates[i,j,state[i,j]]=day+1;state[i,j]+=1
    return dates
