"""Fixed-parameter swaps and paired weather-specific recalibration contrasts."""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from research.phenology_exploration.models import predict,residual_prediction
from research.phenology_exploration.neural import ThermalTransformer,predict_network,DEVICE
from research.phenology_exploration.experiments import row_predictions
from research.phenology_exploration.support_blend import support_weight,blend_dates
from research.phenology_exploration.core import valid_prediction,sha
from research.phenology_exploration.analysis import site_errors,block_interval
from .analysis import MODELS
from .weather_design import freeze_weather,FOLDER,grouped_source
from .weather_fit import product_data


def weather_interval(frame,stations):
    geography=frame[['site','block']].drop_duplicates().merge(stations[['site','station']].drop_duplicates(),on='site',validate='one_to_one')
    groups=grouped_source(geography).set_index('site').block
    connected=frame.copy();connected['block']=connected.site.map(groups)
    lo,hi=block_interval(connected);native_lo,native_hi=block_interval(frame)
    return dict(ci_low=lo,ci_high=hi,native_block_ci_low=native_lo,native_block_ci_high=native_hi,
        target_clusters=connected.block.nunique(),native_blocks=frame.block.nunique())


def swap_predictions(folder,data,members,reference):
    with (folder/'fits.pkl').open('rb') as f:saved=pickle.load(f)
    models=saved['models'];ids=reference.u.unique();base=predict(models['T'],data,ids)
    physical=predict(models['nonlinear_T'],data,ids)
    neural=base+saved['mlp_choice']['strength']*residual_prediction(models['MLP'],data,ids,base)
    rf=base+saved['rf_choice']['strength']*residual_prediction(models['RF'],data,ids,base)
    source=members.drop_duplicates('site');w,_=support_weight(data.context[source.u,:2],data.context[ids,:2])
    blended=blend_dates(neural,physical,w[:,None],data.lengths[ids,None])
    cp=torch.load(folder/'thermal_transformer.pt',map_location='cpu',weights_only=False)
    network=ThermalTransformer(cp['initial'],23).to(DEVICE);network.load_state_dict(cp['state'])
    transformer=predict_network(network,data,ids,cp['norm'],False)
    out=reference.copy()
    for model,values in [('T',base),('nonlinear_T',physical),('residual_RF',rf),('residual_MLP',neural),('thermal_transformer',transformer),('support_blend',blended)]:out[model]=row_predictions(reference,ids,values)
    return out


def run():
    data,common,cases=freeze_weather();assert (FOLDER/'refit_verification.json').exists()
    weather={provider:product_data(data,common,provider) for provider in ['dwd','era5']}
    output=FOLDER/'analysis';output.mkdir(exist_ok=True);tables=[];contrasts=[];identities={}
    for case in cases:
        name=case['case'];members=pd.read_csv(FOLDER/'paired_design/memberships'/f'{name}.csv.gz');predictions={}
        for train in ['dwd','era5']:
            folder=FOLDER/'refits'/train/'cases'/name
            receipt=json.loads((folder/'complete.json').read_text())
            for file,digest in receipt['outputs'].items():assert sha(folder/file)==digest
            for file,digest in receipt['signature']['code'].items():assert sha(file)==digest
            for file,digest in receipt['signature']['inputs'].items():assert sha(file)==digest
            reference=pd.read_csv(folder/'predictions.csv.gz')
            if case['fold'] in identities:assert tuple(sorted(reference.label_id))==identities[case['fold']]
            else:identities[case['fold']]=tuple(sorted(reference.label_id))
            for test in ['dwd','era5']:
                f=reference if test==train else swap_predictions(folder,weather[test],members,reference)
                f=f.merge(common[['u','max_distance_km','station']],on='u',validate='many_to_one')
                f['training_weather']=train;f['prediction_weather']=test
                dest=output/'predictions'/train/test;dest.mkdir(parents=True,exist_ok=True)
                f.to_csv(dest/f'{name}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
                predictions[train,test]=f
                for period,q in f.groupby('period'):
                    for model in MODELS:
                        error=np.where(valid_prediction(q,q[model]),abs(q[model]-q.observed),366.)
                        s=site_errors(q,error).assign(period=period,model=model,training_weather=train,prediction_weather=test,fold=case['fold'],draw=case['draw'])
                        s['completion']=site_errors(q,valid_prediction(q,q[model]).astype(float)).error
                        tables.append(s)
        pairs=[('fixed_era5_swap_dwd',('era5','dwd'),('era5','era5')),
               ('fixed_dwd_swap_era5',('dwd','era5'),('dwd','dwd')),
               ('matched_dwd_minus_era5',('dwd','dwd'),('era5','era5'))]
        for comparison,a,b in pairs:
            f=predictions[a];ref=predictions[b].set_index('label_id').loc[f.label_id]
            for period,q in f.groupby('period'):
                rr=ref.loc[q.label_id]
                for model in MODELS:
                    first=np.where(valid_prediction(q,q[model]),abs(q[model]-q.observed),366.)
                    second=np.where(valid_prediction(rr,rr[model]),abs(rr[model]-rr.observed),366.)
                    diff=site_errors(q,first-second).assign(period=period,model=model,comparison=comparison,fold=case['fold'],draw=case['draw'])
                    contrasts.append(diff)
    site=pd.concat(tables,ignore_index=True);diff=pd.concat(contrasts,ignore_index=True)
    site.to_csv(output/'site_errors.csv.gz',index=False,compression={'method':'gzip','mtime':0});diff.to_csv(output/'paired_site_differences.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    summary=[];paired=[]
    for keys,q in site.groupby(['period','model','training_weather','prediction_weather']):
        period,model,train,test=keys;a=q.groupby(['head','site','block'])[['error','completion']].mean().reset_index()
        interval=weather_interval(a,common[['site','station']])
        summary.append(dict(period=period,model=model,training_weather=train,prediction_weather=test,mae=float(a.groupby('head').error.mean().mean()),
            **interval,completion=float(a.groupby('head').completion.mean().mean()),sites=a.site.nunique()))
    for keys,q in diff.groupby(['period','model','comparison']):
        period,model,comparison=keys;a=q.groupby(['head','site','block']).error.mean().reset_index()
        interval=weather_interval(a,common[['site','station']])
        paired.append(dict(period=period,model=model,comparison=comparison,difference=float(a.groupby('head').error.mean().mean()),**interval,sites=a.site.nunique()))
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False);pd.DataFrame(paired).to_csv(output/'paired_contrasts.csv',index=False)
    # Each site/calendar day appears once despite overlapping native sowing windows.
    with np.load(FOLDER/'common_inputs.npz') as z:dwd=z['dwd'];era=z['era5'];assert np.array_equal(z['u'],common.u)
    starts=pd.to_datetime(common.sowing_date).to_numpy(dtype='datetime64[D]').astype('int64')
    daily=pd.DataFrame(dict(site=np.repeat(common.site.to_numpy(),367),station=np.repeat(common.station.to_numpy(),367),
        day=(starts[:,None]+np.arange(367)[None,:]).ravel(),dwd_min=dwd[:,:,0].ravel(),dwd_max=dwd[:,:,1].ravel(),era5_min=era[:,:,0].ravel(),era5_max=era[:,:,1].ravel()))
    daily=daily.drop_duplicates(['site','day']).reset_index(drop=True)
    daily['dwd_mean']=(daily.dwd_min+daily.dwd_max)/2;daily['era5_mean']=(daily.era5_min+daily.era5_max)/2
    daily['dwd_dtr']=daily.dwd_max-daily.dwd_min;daily['era5_dtr']=daily.era5_max-daily.era5_min
    daily.to_csv(output/'unique_site_daily_weather.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    agreement=[]
    for feature in ['min','max','mean','dtr']:
        delta=daily['era5_'+feature]-daily['dwd_'+feature]
        f=daily[['site']].assign(bias=delta,absolute=abs(delta),square=delta**2).groupby('site').mean()
        agreement.append(dict(variable=feature,era5_minus_dwd_bias_c=float(f.bias.mean()),mae_c=float(f.absolute.mean()),rmse_c=float(np.sqrt(f.square.mean())),
            sites=len(f),unique_site_days=len(daily),dwd_station_days=daily[['station','day']].drop_duplicates().shape[0]))
    pd.DataFrame(agreement).to_csv(output/'weather_agreement.csv',index=False)
    verification=dict(complete=True,cases_per_provider=len(cases),same_targets_across_weather_and_model=True,
        comparison='same complete UTC native windows; fixed parameters in swaps; identical membership and budgets in refits',
        attribution='weather-product plus station/grid representativeness effect; no error-free DWD-field-weather assumption',
        source_and_target_periods_separate=True,source_year_budget=4,source_site_budget=40,
        uncertainty='connected target weather-station/native-block clusters; native-block-only intervals retained as sensitivity',
        files={p.name:sha(p) for p in output.glob('*.csv*')})
    (output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(pd.DataFrame(summary).query('period == "later_held"').round(3).to_string(index=False))
    print(pd.DataFrame(paired).query('period == "later_held"').round(3).to_string(index=False))


if __name__=='__main__':run()
