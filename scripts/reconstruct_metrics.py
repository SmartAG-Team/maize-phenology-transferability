"""Reconstruct grouped means and conditional paired intervals without refitting."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from prepare_data import prepare
from research.phenology_exploration.analysis import block_interval
from research.annual_temperature_transfer.weather_analysis import weather_interval

E=ROOT/'evidence';O=ROOT/'outputs/metrics';O.mkdir(parents=True,exist_ok=True)
checks=[]

def estimate(frame,column='error'):
    q=frame.groupby(['head','site','block'],observed=True)[column].mean().reset_index().rename(columns={column:'error'})
    if q['head'].nunique()!=7 or not np.isfinite(q.error).all():return np.nan,np.nan,np.nan
    lo,hi=block_interval(q);return float(q.groupby('head').error.mean().mean()),lo,hi

def check(actual,expected,label):
    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=0,equal_nan=True);checks.append(label)

with threadpool_limits(limits=1):
    # Hierarchy is already preserved in the released draw-averaged site/head tables.
    for filename,reference,groups in [('benchmark_sites.csv.gz','benchmark.csv',['study','period','recipe','model']),('local_sites.csv.gz','local_summary.csv',['budget','model'])]:
        data=pd.read_csv(E/'insights'/filename);ref=pd.read_csv(E/'insights'/reference);rows=[]
        for key,q in data.groupby(groups):
            row=dict(zip(groups,key if isinstance(key,tuple) else (key,)));v=estimate(q)
            match=ref
            for k,x in row.items():match=match[match[k].eq(x)]
            r=match.iloc[0];check(v,[r.mae,r.ci_low,r.ci_high],str(row));rows.append(dict(**row,mae=v[0],ci_low=v[1],ci_high=v[2]))
        pd.DataFrame(rows).to_csv(O/reference,index=False)
    data=pd.read_csv(E/'insights/correction_sites.csv.gz');ref=pd.read_csv(E/'insights/correction_summary.csv');rows=[]
    for (recipe,model),q in data.groupby(['recipe','model']):
        r=ref[ref.recipe.eq(recipe)&ref.model.eq(model)].iloc[0];row=dict(recipe=recipe,model=model)
        for metric in ['benefit','harm','net_gain','improved_fraction','harmed_fraction']:
            v=estimate(q,metric);check(v,[r[metric],r[metric+'_ci_low'],r[metric+'_ci_high']],recipe+'/'+model+'/'+metric)
            row.update({metric:v[0],metric+'_ci_low':v[1],metric+'_ci_high':v[2]})
        assert abs(row['benefit']-row['harm']-row['net_gain'])<1e-10;rows.append(row)
    pd.DataFrame(rows).to_csv(O/'correction_summary.csv',index=False)
    stations=pd.read_csv(E/'weather/site_station_mapping.csv')
    for source,reference,keys,value in [('site_errors.csv.gz','summary.csv',['period','model','training_weather','prediction_weather'],'mae'),('paired_site_differences.csv.gz','paired_contrasts.csv',['period','model','comparison'],'difference')]:
        data=pd.read_csv(E/'weather/analysis'/source);ref=pd.read_csv(E/'weather/analysis'/reference);rows=[]
        for key,q in data.groupby(keys):
            row=dict(zip(keys,key));a=q.groupby(['head','site','block']).error.mean().reset_index();interval=weather_interval(a,stations)
            actual=float(a.groupby('head').error.mean().mean());match=ref
            for k,x in row.items():match=match[match[k].eq(x)]
            r=match.iloc[0];check([actual,interval['ci_low'],interval['ci_high']],[r[value],r.ci_low,r.ci_high],str(row));rows.append(dict(**row,**{value:actual},**interval))
        pd.DataFrame(rows).to_csv(O/('weather_'+reference),index=False)
    # Preserve each source/target/native-head/site block before averaging source draws.
    keys=['period','model','source_zone','target_zone','head','site','block'];parts=[]
    for q in pd.read_csv(prepare(),chunksize=150000,usecols=keys+['error','penalty']):
        grouped=q.groupby(keys,observed=True).agg(error_sum=('error','sum'),penalty_sum=('penalty','sum'),n=('error','size')).reset_index();parts.append(grouped)
    q=pd.concat(parts,ignore_index=True).groupby(keys,observed=True)[['error_sum','penalty_sum','n']].sum().reset_index()
    q['error']=q.error_sum/q.n;q['penalty']=q.penalty_sum/q.n
    ref=pd.read_csv(E/'matrix.csv');rows=[]
    for key,a in q.groupby(['period','model','source_zone','target_zone'],observed=True):
        row=dict(zip(['period','model','source_zone','target_zone'],key));match=ref
        for k,x in row.items():match=match[match[k].eq(x)]
        r=match.iloc[0]
        for metric in ['error','penalty']:
            v=estimate(a,metric);check(v,[r[metric],r[metric+'_ci_low'],r[metric+'_ci_high']],str(row)+'/'+metric)
            row.update({metric:v[0],metric+'_ci_low':v[1],metric+'_ci_high':v[2]})
        rows.append(row)
    pd.DataFrame(rows).to_csv(O/'annual_transfer_matrix.csv',index=False)
(O/'verification.json').write_text(json.dumps({'verified':True,'grouped_mean_interval_checks':len(checks),'no_model_refitting':True},indent=2)+'\n')
print(json.dumps({'verified':True,'grouped_mean_interval_checks':len(checks),'outputs':str(O)}))
