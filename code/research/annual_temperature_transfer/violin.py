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


def weather_difference_distribution(sites,contrasts,models):
    """Average source draws before descriptive density; retain frozen paired CIs."""
    comparisons=['fixed_era5_swap_dwd','fixed_dwd_swap_era5','matched_dwd_minus_era5']
    raw=sites[sites.period.eq('later_held')&sites.model.isin(models)&sites.comparison.isin(comparisons)].copy()
    keys=['model','comparison','head','site','block','fold']
    if raw.duplicated(keys+['draw']).any():raise ValueError('Repeated weather site/endpoint/draw')
    if not raw.groupby(keys).draw.nunique().eq(3).all():raise ValueError('Incomplete source-draw support')
    paired=raw.groupby(keys,as_index=False).error.mean().rename(columns={'error':'difference_days'})
    paired['panel']=np.where(paired.comparison.eq('matched_dwd_minus_era5'),'(b)','(a)')
    rows=[];curves=[];identity=None
    for comparison in comparisons:
        for model in models:
            q=paired[paired.model.eq(model)&paired.comparison.eq(comparison)]
            support=set(map(tuple,q[['head','site','block','fold']].to_numpy()))
            if identity is not None and support!=identity:raise ValueError('Weather comparison supports differ')
            identity=support
            if q['head'].nunique()!=7 or not np.isfinite(q.difference_days).all():raise ValueError('Incomplete weather head support')
            weights=1/(7*q.groupby('head').difference_days.transform('size').to_numpy())
            paired.loc[q.index,'density_weight']=weights
            values=q.difference_days.to_numpy();mean=float(np.dot(values,weights))
            ref=contrasts[contrasts.period.eq('later_held')&contrasts.model.eq(model)&contrasts.comparison.eq(comparison)]
            if len(ref)!=1 or abs(mean-ref.iloc[0].difference)>1e-10:raise ValueError('Weather density mean disagrees with frozen contrast')
            order=np.argsort(values);cdf=np.cumsum(weights[order])-weights[order]/2
            grid=np.linspace(values.min(),values.max(),601)
            density=gaussian_kde(values,weights=weights,bw_method='scott')(grid)
            rows.append(dict(model=model,comparison=comparison,weighted_mean_days=mean,
                weighted_median_days=float(np.interp(.5,cdf,values[order])),
                site_endpoint_pairs=len(q),minimum_days=float(values.min()),maximum_days=float(values.max())))
            curves.append(pd.DataFrame(dict(model=model,comparison=comparison,panel=q.panel.iloc[0],
                difference_days=grid,density_per_day=density,relative_density=density/density.max())))
    summary=contrasts[contrasts.period.eq('later_held')].merge(pd.DataFrame(rows),on=['model','comparison'],validate='one_to_one')
    summary['panel']=np.where(summary.comparison.eq('matched_dwd_minus_era5'),'(b)','(a)')
    return paired,summary,pd.concat(curves,ignore_index=True)
