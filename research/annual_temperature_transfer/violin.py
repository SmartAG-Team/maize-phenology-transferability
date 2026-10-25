"""Head-balanced distributions of paired local-history gains, without refitting."""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def local_gain_distribution(sites,contrasts,models):
    keys=['model','head','site','block']
    f=sites[sites.model.isin(models)&sites.budget.isin([0,5])].copy()
    if f.duplicated(keys+['budget']).any():raise ValueError('Repeated site/endpoint/budget')
    zero=f[f.budget.eq(0)][keys+['error']].rename(columns={'error':'error_zero'})
    five=f[f.budget.eq(5)][keys+['error']].rename(columns={'error':'error_five'})
    paired=zero.merge(five,on=keys,how='outer',validate='one_to_one',indicator=True)
    if not paired['_merge'].eq('both').all():raise ValueError('Local-history supports differ')
    paired=paired.drop(columns='_merge');paired['gain_days']=paired.error_zero-paired.error_five
    rows=[];densities=[];identities=None
    for model in models:
        q=paired[paired.model.eq(model)].copy()
        if q['head'].nunique()!=7 or not np.isfinite(q.gain_days).all():raise ValueError('Incomplete head support')
        support=set(map(tuple,q[['head','site','block']].to_numpy()))
        if identities is not None and support!=identities:raise ValueError('Models use different local targets')
        identities=support
        # Equal total weight for each head, then equal site weight within head.
        weights=1/(7*q.groupby('head').gain_days.transform('size').to_numpy())
        paired.loc[q.index,'density_weight']=weights
        values=q.gain_days.to_numpy();mean=float(np.dot(values,weights))
        reference=contrasts[contrasts.model.eq(model)&contrasts.budget.eq(5)&contrasts.comparison.eq('adapted_minus_zero')]
        if len(reference)!=1:raise ValueError('Missing paired mean/interval')
        r=reference.iloc[0]
        if abs(mean+r.difference)>1e-10:raise ValueError('Distribution mean disagrees with paired study contrast')
        order=np.argsort(values);cdf=np.cumsum(weights[order])-weights[order]/2
        median=float(np.interp(.5,cdf,values[order]))
        grid=np.linspace(values.min(),values.max(),601)
        density=gaussian_kde(values,weights=weights,bw_method='scott')(grid)
        rows.append(dict(model=model,mean_gain_days=mean,ci_low_days=-r.ci_high,ci_high_days=-r.ci_low,
            weighted_median_days=median,weighted_improved_fraction=float(weights[values>0].sum()),
            weighted_worsened_fraction=float(weights[values<0].sum()),site_endpoint_pairs=len(q),
            unique_sites=q.site.nunique(),native_heads=7,min_gain_days=float(values.min()),max_gain_days=float(values.max())))
        densities.append(pd.DataFrame(dict(model=model,gain_days=grid,density_per_day=density,
            displayed_halfwidth=.34*density/density.max())))
    paired['panel']='(b)'
    return paired,pd.DataFrame(rows),pd.concat(densities,ignore_index=True)
