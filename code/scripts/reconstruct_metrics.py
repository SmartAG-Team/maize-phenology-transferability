"""Reconstruct grouped means and conditional paired intervals without refitting."""
from pathlib import Path
import sys,json
PROJECT=Path(__file__).resolve().parents[2]
ROOT=PROJECT
sys.path.insert(0,str(PROJECT/'code'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from prepare_data import prepare
from research.phenology_exploration.analysis import block_interval
from research.annual_temperature_transfer.weather_analysis import weather_interval
from research.phenology_exploration.core import sha
from research.full_source_benchmark import analysis as full_source

E=ROOT/'data/evidence';O=ROOT/'outputs/metrics'
checks=[]

def estimate(frame,column='error'):
    q=frame.groupby(['head','site','block'],observed=True)[column].mean().reset_index().rename(columns={column:'error'})
    if q['head'].nunique()!=7 or not np.isfinite(q.error).all():return np.nan,np.nan,np.nan
    lo,hi=block_interval(q);return float(q.groupby('head').error.mean().mean()),lo,hi

def check(actual,expected,label):
    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=0,equal_nan=True);checks.append(label)

def run_legacy():
    O.mkdir(parents=True,exist_ok=True)
    checks.clear()
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
    return len(checks)


FULL_SOURCE_RESULT_KEYS={
    'summary.csv':['study','recipe','period','model'],
    'stage_performance.csv':['study','recipe','period','model','head'],
    'benchmark_sites.csv.gz':['study','recipe','period','model','head','site','block'],
    'correction_summary.csv':['study','recipe','period','model'],
    'correction_sites.csv.gz':['study','recipe','period','model','head','site','block'],
    'paired_contrasts.csv':['study','recipe','period','left','right','head'],
    'keyed_predictions.csv.gz':['label_id'],
    'pooled_cells.csv':['head','model','weather_cell'],
    'pooled_map.csv':['panel','head','model','weather_cell'],
    'spatial_summary.csv':['panel'],
}


def require_full_source_evidence(folder):
    folder=Path(folder);receipt=folder/'verification.json'
    if not receipt.is_file():
        raise FileNotFoundError(f'Missing canonical full-source evidence: {receipt}')
    verification=json.loads(receipt.read_text())
    if not verification.get('complete'):
        raise ValueError('Incomplete canonical full-source evidence')
    files=verification.get('output_sha256',{})
    csvs=sorted(name for name in files if name.endswith(('.csv','.csv.gz')))
    if set(csvs)!=set(FULL_SOURCE_RESULT_KEYS):
        raise ValueError('Incomplete or unsupported canonical full-source result CSVs')
    for name in csvs:
        path=folder/name
        if not path.is_file():
            raise FileNotFoundError(f'Missing canonical full-source evidence: {path}')
        if sha(path)!=files[name]:
            raise ValueError(f'Changed canonical full-source evidence: {name}')
    return csvs


def compare_result_csvs(actual_folder,expected_folder,filenames):
    """Compare every numerical column after aligning the native result keys."""
    actual_folder,expected_folder=Path(actual_folder),Path(expected_folder)
    column_count=value_count=0
    for name in filenames:
        actual=pd.read_csv(actual_folder/name);expected=pd.read_csv(expected_folder/name)
        if list(actual.columns)!=list(expected.columns):
            raise AssertionError(f'Result schema differs: {name}')
        keys=[key for key in FULL_SOURCE_RESULT_KEYS[name] if key in expected]
        if not keys or actual.duplicated(keys).any() or expected.duplicated(keys).any():
            raise AssertionError(f'Duplicate or missing native support keys: {name}')
        actual=actual.sort_values(keys).reset_index(drop=True)
        expected=expected.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[keys],expected[keys],check_dtype=False,
            obj=f'Native support differs for {name}')
        for column in expected:
            if pd.api.types.is_numeric_dtype(expected[column]):
                np.testing.assert_allclose(actual[column].to_numpy(float),expected[column].to_numpy(float),
                    atol=1e-8,rtol=0,equal_nan=True,err_msg=f'{name}/{column}')
                column_count+=1;value_count+=len(expected)
            else:
                pd.testing.assert_series_equal(actual[column],expected[column],check_dtype=False,
                    obj=f'{name}/{column}')
    return dict(files=len(filenames),numerical_columns=column_count,numerical_values=value_count,
        filenames=list(filenames),absolute_tolerance=1e-8,relative_tolerance=0,native_support_exact=True)


def reconstruct_full_source():
    canonical=E/'full_source_benchmark'
    filenames=require_full_source_evidence(canonical)
    destination=O/'full_source_benchmark'
    with threadpool_limits(limits=1):
        full_source.run(input_folder=ROOT/'data/results/full_source_benchmark_20261026',
            output_folder=destination,
            reference_folder=ROOT/'data/results/phenology_exploration_20261023/year_matched',
            coordinates_path=E/'site_zones.csv')
    comparison=compare_result_csvs(destination,canonical,filenames)
    comparison.update(verified=True,no_model_refitting=True,canonical_evidence=str(canonical),
        canonical_receipt_sha256=sha(canonical/'verification.json'),
        reconstructed_receipt_sha256=sha(destination/'verification.json'))
    (destination/'reconstruction_verification.json').write_text(json.dumps(comparison,indent=2)+'\n')
    return comparison


def main():
    # Require both archives before creating any reconstructed outputs.
    require_full_source_evidence(E/'full_source_benchmark')
    ordinary=E/'mae_bias_revision_20261026'
    if not (ordinary/'verification.json').is_file():
        raise FileNotFoundError(f'Missing canonical ordinary-MAE evidence: {ordinary}')
    from research.mae_bias_revision.analysis import verify as verify_ordinary
    with threadpool_limits(limits=1):
        active=verify_ordinary(ordinary,report_path=O/'ordinary_mae_verification.json')
    current=reconstruct_full_source()
    legacy_checks=run_legacy()
    if legacy_checks!=924:
        raise ValueError(f'Expected all 924 archived reconstruction checks, got {legacy_checks}')
    verification=dict(verified=True,grouped_mean_interval_checks=legacy_checks,
        ordinary_mae_and_bias_checks=active,active_evidence='data/evidence/mae_bias_revision_20261026',
        full_source_numerical_csv_checks=current,legacy_checks_role='Archived original-score audit',no_model_refitting=True)
    (O/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(dict(verification,outputs=str(O))))
    return verification


if __name__=='__main__':
    main()
