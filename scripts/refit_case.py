"""Refit a frozen annual-transfer or weather case in a new output directory."""
from pathlib import Path
import argparse,json,shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import pandas as pd
from research.annual_temperature_transfer import fit,weather_fit
from research.annual_temperature_transfer.design import dataset

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['annual','dwd','era5'],default='annual')
    p.add_argument('--case');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    base=ROOT/'results/annual_temperature_transfer_20261024'
    data,_,_=dataset()
    if a.cohort=='annual':source=base
    else:
        common=pd.read_csv(base/'weather/common_units.csv.gz')
        labels=data.labels[data.labels.u.isin(common.u)].copy();units=data.units[data.units.u.isin(common.u)].copy()
        units['station']=units.u.map(common.set_index('u').station)
        data=replace(data,labels=labels,units=units)
        source=base/'weather/refits'/a.cohort
        data=weather_fit.product_data(data,common,a.cohort)
    cases=json.loads((source/'cases.json').read_text())['cases']
    case=next((c for c in cases if c['case']==a.case),None) if a.case else cases[0]
    if case is None:raise ValueError('Unknown frozen case')
    if a.cohort!='annual':data=weather_fit.case_data(data,case['fold'])
    dest=ROOT/'outputs/refits'/a.cohort;dest.mkdir(parents=True,exist_ok=True)
    (dest/'memberships').mkdir(exist_ok=True)
    for old,new in [(source/'design.json',dest/'design.json'),(source/'memberships'/f'{case["case"]}.csv.gz',dest/'memberships'/f'{case["case"]}.csv.gz')]:
        if new.exists() and old.read_bytes()!=new.read_bytes():raise ValueError('Changed refit design or membership')
        shutil.copy2(old,new)
    fit.OUT=dest
    members=pd.read_csv(dest/'memberships'/f'{case["case"]}.csv.gz')
    assert fit.sha(dest/'memberships'/f'{case["case"]}.csv.gz')==case['membership_sha256']
    source_labels=data.labels[data.labels.unit.isin(members.unit)&data.labels.pre2011_prefix_eligible]
    role=f'fold{case["fold"]}_case';target=data.labels[data.labels[role].isin(['historical_held','later_held'])]
    assert not set(source_labels.site)&set(target.site)
    inner=members.set_index('unit').inner
    assert not set(source_labels[source_labels.unit.map(inner).ne(2)].block)&set(source_labels[source_labels.unit.map(inner).eq(2)].block)
    if not a.prepare_only:fit.run_case(data,case)
    print(json.dumps({'cohort':a.cohort,'case':case['case'],'source_units':len(members),'source_labels':len(source_labels),'target_labels':len(target),'prepare_only':a.prepare_only,'output':str(dest.relative_to(ROOT))}))

if __name__=='__main__':main()
