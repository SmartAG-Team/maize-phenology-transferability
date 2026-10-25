"""Full-horizon scores and treatment contrasts with explicit sampling units."""
import numpy as np
import pandas as pd


def score_rows(cdf,truth,environments,endpoints,mode=None,seed=None):
    cdf=np.asarray(cdf,dtype=float);truth=np.asarray(truth)
    if cdf.ndim!=3 or cdf.shape[:2]!=truth.shape or len(environments)!=len(cdf):raise ValueError('Prediction/truth shape mismatch')
    if not np.isfinite(cdf).all() or np.any((cdf<0)|(cdf>1)) or np.any(np.diff(cdf,axis=2)<-1e-7):raise ValueError('Invalid CDF')
    days=np.arange(1,cdf.shape[2]+1);rows=[]
    for j,endpoint in enumerate(endpoints):
        ids=np.flatnonzero(truth[:,j]>0);curve=cdf[ids,j]
        observed=truth[ids,j]
        if np.any(observed>cdf.shape[2]):raise ValueError('Observed day beyond weather horizon')
        quantile=lambda p:np.where((curve>=p).any(1),np.argmax(curve>=p,1)+1,np.nan)
        low,median,high=quantile(.05),quantile(.5),quantile(.95)
        # Unresolved upper intervals extend beyond the analysis horizon.
        coverage=(observed>=np.where(np.isnan(low),np.inf,low))&(observed<=np.where(np.isnan(high),np.inf,high))
        base=environments.iloc[ids].reset_index(drop=True).copy()
        base['endpoint']=endpoint;base['observed_day']=observed;base['median_day']=median
        base['crps']=((curve-(days[None,:]>=observed[:,None]))**2).sum(1)
        base['bias']=median-observed;base['absolute_error']=np.abs(median-observed)
        base['covered90']=coverage.astype(float);base['interval_width90']=high-low
        base['open_interval90']=~(np.isfinite(low)&np.isfinite(high))
        base['tail_probability']=1-curve[:,-1];base['unresolved_median']=np.isnan(median)
        if mode is not None:base['mode']=mode
        if seed is not None:base['seed']=seed
        rows.append(base)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def sowing_contrasts(predictions):
    rows=[]
    for key,group in predictions.groupby(['mode','seed','year','endpoint']):
        group=group.sort_values('sowing_day');earliest=group.iloc[0]
        if group.sowing_day.duplicated().any():raise ValueError('Repeated sowing-treatment pairing')
        for _,row in group.iloc[1:].iterrows():
            observed=row.observed_day-earliest.observed_day
            predicted=row.predicted_day-earliest.predicted_day
            rows.append(dict(zip(['mode','seed','year','endpoint'],key),treatment=row.treatment,
                             sowing_delay=row.sowing_day-earliest.sowing_day,observed_change=observed,
                             predicted_change=predicted,absolute_contrast_error=abs(predicted-observed)))
    contrasts=pd.DataFrame(rows)
    peryear=contrasts.groupby(['mode','seed','year','endpoint']).absolute_contrast_error.mean().groupby(['mode','seed','year']).mean()
    summary=peryear.groupby(['mode','seed']).agg(contrast_mae='mean',independent_years='size').reset_index()
    counts=contrasts.groupby(['mode','seed']).absolute_contrast_error.agg(total_contrasts='size',resolved_contrasts='count').reset_index()
    counts['unresolved_fraction']=1-counts.resolved_contrasts/counts.total_contrasts
    summary=summary.merge(counts,on=['mode','seed'],validate='one_to_one')
    return contrasts,summary


def paired_bootstrap(scores,model,reference,replicates=2000,seed=712):
    keys=['environment_id_model','endpoint']
    d=scores[scores['mode'].isin([model,reference])]
    if d.duplicated(keys+['mode','seed']).any():raise ValueError('Duplicated predictions require conflict checking')
    pivot=d.groupby(keys+['mode']).crps.mean().unstack('mode')
    if pivot[[model,reference]].isna().any().any():raise ValueError('Unpaired primary targets')
    context=d[keys+['spatial_group','year']].drop_duplicates()
    if context.duplicated(keys).any():raise ValueError('Conflicting sampling units')
    joined=pivot[[model,reference]].reset_index().merge(context,on=keys,validate='one_to_one')
    stages=sorted(joined.endpoint.unique());groups=sorted(joined.spatial_group.unique());years=sorted(joined.year.unique())
    gi=pd.Categorical(joined.spatial_group,categories=groups).codes;yi=pd.Categorical(joined.year,categories=years).codes
    delta=(joined[model]-joined[reference]).to_numpy();sj=pd.Categorical(joined.endpoint,categories=stages).codes
    def statistic(weights):
        sums=np.bincount(sj,weights=weights*delta,minlength=len(stages));mass=np.bincount(sj,weights=weights,minlength=len(stages))
        return float(np.mean(sums/mass)) if (mass>0).all() else np.nan
    point=statistic(np.ones(len(joined)));rng=np.random.default_rng(seed);boot=[]
    for _ in range(replicates):
        spatial=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)))
        annual=rng.multinomial(len(years),np.full(len(years),1/len(years)))
        boot.append(statistic(spatial[gi]*annual[yi]))
    finite=np.asarray(boot);finite=finite[np.isfinite(finite)]
    if not len(finite):raise ValueError('No evaluable bootstrap replicates')
    return {'model':model,'reference':reference,'difference':point,'low':float(np.quantile(finite,.025)),
            'high':float(np.quantile(finite,.975)),'years':len(years),'spatial_groups':len(groups),
            'requested_replicates':replicates,'valid_replicates':len(finite),'seed':seed,
            'uncertainty_scope':'Paired product spatial-group × observed-year weights; conditional on fitted model recipes'}


def field_interval_score(cdf,lower,upper,length):
    curve=np.asarray(cdf,dtype=float)[:int(length)]
    if not 0<=lower<upper<=len(curve):raise ValueError('Interval beyond available weather')
    if not np.isfinite(curve).all() or np.any(np.diff(curve)<-1e-7):raise ValueError('Invalid field CDF')
    median=float(np.argmax(curve>=.5)+1) if np.any(curve>=.5) else np.nan
    return dict(interval_probability=float(curve[int(upper)-1]-(curve[int(lower)-1] if lower>0 else 0.)),
        median_day=median,distance_to_interval=max(lower+1-median,median-upper,0.) if np.isfinite(median) else np.nan,
        unresolved_median=bool(not np.isfinite(median)),weather_horizon_days=len(curve))


def paired_sowing_comparison(contrasts,model,reference):
    keys=['seed','year','endpoint','treatment']
    selected=contrasts[contrasts['mode'].isin([model,reference])]
    if selected.duplicated(keys+['mode']).any():raise ValueError('Duplicated sowing contrasts')
    wide=selected.pivot(index=keys,columns='mode',values='absolute_contrast_error')
    if model not in wide or reference not in wide:raise ValueError('Missing comparison family')
    common=wide[[model,reference]].dropna()
    def stage_year_equal(column):
        if common.empty:return np.nan
        return float(common[column].groupby(['seed','year','endpoint']).mean().groupby(['seed','year']).mean().groupby('seed').mean().mean())
    return dict(model=model,reference=reference,total_contrasts=len(wide),paired_contrasts=len(common),
        model_resolved_contrasts=int(wide[model].notna().sum()),reference_resolved_contrasts=int(wide[reference].notna().sum()),
        common_resolved_fraction=len(common)/len(wide) if len(wide) else np.nan,
        model_common_mae=stage_year_equal(model),reference_common_mae=stage_year_equal(reference),
        conditional_on_common_resolved_medians=True)


def archive_predictions(path,cdf,truth,environments,lengths,endpoints,provenance):
    import json
    from pathlib import Path
    from .transfer_artifacts import file_sha256,atomic_json_save
    path=Path(path);ids=environments.environment_id_model.astype(str).to_numpy(dtype=str)
    if len(set(ids))!=len(ids):raise ValueError('Nonunique prediction keys')
    cdf=np.asarray(cdf,dtype=np.float32);truth=np.asarray(truth);lengths=np.asarray(lengths,dtype=int)
    if cdf.shape[:2]!=truth.shape or len(ids)!=len(cdf) or len(lengths)!=len(ids):raise ValueError('Prediction archive alignment failed')
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,cdf=cdf,observed=truth,environment_ids=ids,weather_lengths=lengths,endpoints=np.asarray(endpoints,dtype=str))
    temporary.replace(path)
    atomic_json_save(dict(provenance,prediction_sha256=file_sha256(path),environments=len(ids),scores_reconstructed_from_saved_cdf=True),path.with_suffix('.json'))
    with np.load(path) as z:
        if not np.array_equal(ids,z['environment_ids']) or not np.array_equal(lengths,z['weather_lengths']):raise ValueError('Saved prediction keys changed')
        return z['cdf'],z['observed']
