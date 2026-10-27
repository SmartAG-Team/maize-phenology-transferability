"""Fixed foreign applications; no foreign outcome selects German recipes."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .run_study import ROOT,OUT,SEEDS,load,tag,checked_fit
from maize_clock.transfer_fit import restore,predict
from maize_clock.transfer_adaptation import adapt,local_masks,observed_targets
from maize_clock.transfer_evaluate import score_rows,sowing_contrasts,field_interval_score,paired_sowing_comparison,archive_predictions
from maize_clock.transfer_artifacts import file_sha256,atomic_json_save
from maize_clock.extension_boosting import LazySequenceFeatures

CHINA=ROOT/'data/results/agronomic_transfer/china'


def compatible_external(a,env,annotations):
    labels=np.zeros((len(env),7),int)
    # Qualified common stages; no source-maturity → BBCH87 conversion.
    gucheng=set(annotations.loc[annotations.site_id.eq('gucheng'),'environment_id_model'])
    northeast=set(annotations.loc[annotations.region.eq('Northeast China'),'environment_id_model'])
    selected=[]
    for i,row in env.iterrows():
        allowed=[0,2,4] if row.environment_id_model in gucheng else [0] if row.environment_id_model in northeast or row.dataset_id=='USDA_Bushland' else []
        if allowed and a['lengths'][i]==366:
            labels[i,allowed]=a['Y'][i,allowed]
            if (labels[i]>0).any():selected.append(i)
    return labels,np.array(selected,dtype=int),gucheng


def evaluate_model(model,saved,mode,seed,german,ger,external,env,y,directory,annotations,checkpoint):
    metadata=saved['metadata']
    provenance={'checkpoint_sha256':file_sha256(checkpoint),
        'inputs':json.loads((OUT/'run_manifest.json').read_text())['input_hashes'],
        'model':mode,'seed':seed,'source_code_sha256':saved['source_code_sha256']}
    def archive(cohort,curves,truth,cohort_env,lengths):
        return archive_predictions(directory/f'{mode}_s{seed}_{cohort}_cdf.npz',curves,truth,cohort_env,lengths,metadata['endpoints'][:7],provenance)
    cx=LazySequenceFeatures(external,metadata,env,history=True)
    masks=local_masks(env);ids=np.flatnonzero(masks['test']&(external['lengths']==366))
    cdf=predict(model,cx,external,ids,metadata)
    cdf,truth=archive('station',cdf,y[ids],env.iloc[ids],external['lengths'][ids])
    station=score_rows(cdf,truth,env.iloc[ids],metadata['endpoints'][:7],mode,seed);station['cohort']='North China later years'
    scores=[station]
    # Original-domain retention uses later German years, outside anchor fitting.
    gids=np.flatnonzero(ger.dataset_id.eq('dwd_annual').to_numpy()&ger.year.between(2021,2025).to_numpy()&(german['lengths']==366))
    gx=LazySequenceFeatures(german,metadata,ger,history=True)
    gcdf=predict(model,gx,german,gids,metadata)
    gcdf,truth=archive('retention',gcdf,german['Y'][gids,:7],ger.iloc[gids],german['lengths'][gids])
    retention=score_rows(gcdf,truth,ger.iloc[gids],metadata['endpoints'][:7],mode,seed)
    retention['cohort']='German later years';scores.append(retention)
    mainy,foreign_ids,gucheng=compatible_external(german,ger,annotations)
    fcdf=predict(model,gx,german,foreign_ids,metadata)
    fcdf,truth=archive('other_foreign',fcdf,mainy[foreign_ids],ger.iloc[foreign_ids],german['lengths'][foreign_ids])
    foreign=score_rows(fcdf,truth,ger.iloc[foreign_ids],metadata['endpoints'][:7],mode,seed)
    foreign['cohort']=np.where(foreign.environment_id_model.isin(gucheng),'Gucheng summer trials',
                              np.where(foreign.dataset_id.eq('USDA_Bushland'),'US emergence','Northeast spring maize'))
    scores.append(foreign)
    treatments=foreign[foreign.environment_id_model.isin(gucheng)].copy()
    treatments['sowing_day']=pd.to_datetime(treatments.sowing_date).dt.dayofyear
    treatments['predicted_day']=treatments.median_day
    treatments['treatment']=treatments.groupby(['year','endpoint']).sowing_day.rank(method='dense').map({1.:'A',2.:'B',3.:'C',4.:'D'})
    contrast,summary=sowing_contrasts(treatments)
    contrasts=directory/f'{mode}_s{seed}_contrasts.csv';contrast.to_csv(contrasts,index=False)
    summary.to_csv(directory/f'{mode}_s{seed}_contrast_summary.csv',index=False)
    # Field observations give intervals, not single-day validation truth.
    intervals=pd.read_csv(CHINA/'field_stage_intervals.csv');fields=env.index[env.collection.eq('field')].to_numpy()
    fieldcdf=predict(model,cx,external,fields,metadata);lookup={env.iloc[i].environment_id_model:k for k,i in enumerate(fields)}
    fieldcdf,_=archive('field',fieldcdf,np.zeros((len(fields),7),int),env.iloc[fields],external['lengths'][fields])
    intervals.to_csv(directory/f'{mode}_s{seed}_field_observed_intervals.csv',index=False)
    intervalrows=[]
    for row in intervals.itertuples(index=False):
        if row.endpoint not in ['emergence','tasseling','milking']:continue
        j=metadata['endpoints'].index(row.endpoint);i=lookup[row.environment_id_model];curve=fieldcdf[i,j]
        lower,upper=row.lower_day_exclusive,row.upper_day_inclusive
        if upper>external['lengths'][fields[i]]:continue
        metrics=field_interval_score(curve,lower,upper,external['lengths'][fields[i]])
        intervalrows.append(dict(environment_id_model=row.environment_id_model,endpoint=row.endpoint,mode=mode,seed=seed,
            lower_day_exclusive=lower,upper_day_inclusive=upper,**metrics,
            independent_farm_years=1,observed_order_conflict=row.later_preceding_state_reported))
    pd.DataFrame(intervalrows).to_csv(directory/f'{mode}_s{seed}_field_intervals.csv',index=False)
    result=pd.concat(scores,ignore_index=True);result.to_csv(directory/f'{mode}_s{seed}_scores.csv.gz',index=False)
    return result


def run():
    torch.set_num_threads(1)
    selection=json.loads((OUT/'selection/germany.json').read_text())
    a,ger,source,design=load()
    with np.load(CHINA/'external_weather_features.npz') as z:ca={k:z[k] for k in z.files}
    env=pd.read_csv(CHINA/'external_environments.csv');targets=pd.read_csv(CHINA/'station_targets.csv')
    if not np.array_equal(env.array_index.to_numpy(),np.arange(len(env))):raise ValueError('Weather/environment alignment changed')
    y=observed_targets(env,targets)
    annotations=pd.read_csv(ROOT/'data/results/country_transfer/cropping_system_audit/environment_season_annotations.csv')
    out=OUT/'foreign';out.mkdir(parents=True,exist_ok=True);all_scores=[]
    # All German-selected primary comparisons retain foreign outcomes.
    for mode,config in selection['selected'].items():
        for seed in SEEDS:
            base=OUT/'fits/germany/final'/(tag(config)+f'_s{seed}.pt')
            checked_fit('germany','final',config,seed,(a,ger,source,design))
            model,saved=restore(base)
            all_scores.append(evaluate_model(model,saved,mode,seed,a,ger,ca,env,y,out,annotations,base))
    loso=[]
    for mode in ['standalone_photo','stage_attention']:
        config=selection['selected'][mode]
        for seed in SEEDS:
            base=OUT/'fits/germany/final'/(tag(config)+f'_s{seed}.pt')
            for held_site in [None,'Luancheng','Shangqiu','Yucheng']:
                for update in ['efficient','full']:
                    fold='temporal' if held_site is None else 'hold_'+held_site
                    name=f'{mode}_{update}_{fold}'
                    path=OUT/'adaptation'/f'{name}_s{seed}'
                    adapt(base,a,ger,ca,env,targets,path,seed,update_mode=update,held_site=held_site)
                    model,saved=restore(path.with_suffix('.pt'))
                    if held_site is None:
                        all_scores.append(evaluate_model(model,saved,name,seed,a,ger,ca,env,y,out,annotations,path.with_suffix('.pt')))
                    else:
                        masks=local_masks(env,held_site);ids=np.flatnonzero(masks['test']&(ca['lengths']==366))
                        cx=LazySequenceFeatures(ca,saved['metadata'],env,history=True);cdf=predict(model,cx,ca,ids,saved['metadata'])
                        cdf,truth=archive_predictions(out/f'{name}_s{seed}_cdf.npz',cdf,y[ids],env.iloc[ids],ca['lengths'][ids],saved['metadata']['endpoints'][:7],
                            {'checkpoint_sha256':file_sha256(path.with_suffix('.pt')),'inputs':json.loads((OUT/'run_manifest.json').read_text())['input_hashes'],'mode':name,'seed':seed})
                        scores=score_rows(cdf,truth,env.iloc[ids],saved['metadata']['endpoints'][:7],name,seed)
                        scores['held_site']=held_site;loso.append(scores)
    result=pd.concat(all_scores,ignore_index=True);result.to_csv(out/'all_scores.csv.gz',index=False)
    metrics=['crps','absolute_error','bias','covered90','interval_width90','open_interval90','tail_probability']
    result.groupby(['cohort','mode','seed','endpoint'])[metrics].mean().reset_index().to_csv(out/'stage_summary.csv',index=False)
    pd.concat(loso,ignore_index=True).to_csv(out/'station_holdout_scores.csv.gz',index=False)
    summaries=pd.concat([pd.read_csv(path) for path in out.glob('*_contrast_summary.csv')],ignore_index=True)
    summaries.to_csv(out/'sowing_contrast_summary.csv',index=False)
    contrasts=pd.concat([pd.read_csv(path) for path in out.glob('*_contrasts.csv')],ignore_index=True)
    comparisons=[]
    for model in sorted(contrasts['mode'].unique()):
        if model!='standalone_photo':comparisons.append(paired_sowing_comparison(contrasts,model,'standalone_photo'))
    pd.DataFrame(comparisons).to_csv(out/'paired_sowing_contrasts.csv',index=False)
    atomic_json_save({'complete':True,'foreign_labels_used_for_German_recipe_selection':False,
        'local_labels':['emergence','stem_elongation','tasseling'],'German_anchor_panels_matched_by_seed':True,
        'Gucheng_independent_site_years':3,'field_independent_farm_years':1,'Northeast_evidence_descriptive':True},out/'status.json')


if __name__=='__main__':run()
