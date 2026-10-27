"""Qualified observation years, annual-temperature zones and cohort geography."""
import json
import pandas as pd
from research.phenology_exploration.core import sha
from .climate import ROOT,OUT

FOLDER=OUT/'spatial'


def run():
    FOLDER.mkdir(parents=True,exist_ok=True);inputs={}
    def record(path):inputs[str(path)]=sha(path)
    path=OUT/'site_zones.csv';record(path);coordinates=pd.read_csv(path)
    labels=[];prior=ROOT/'data/results/phenology_exploration_20261023/year_matched'
    for fold in range(5):
        folder=prior/'cases'/f'f{fold}_random_n80_d17';receipt=json.loads((folder/'complete.json').read_text())
        path=folder/'predictions.csv.gz';assert sha(path)==receipt['outputs'][path.name];record(path)
        labels.append(pd.read_csv(path,usecols=['label_id','unit','site','site_year','year','case']))
    labels=pd.concat(labels,ignore_index=True);assert labels.label_id.is_unique and len(labels)==207325
    count=labels.groupby('site').agg(observed_years=('year','nunique'),native_labels=('label_id','nunique'),native_units=('unit','nunique'),first_year=('year','min'),last_year=('year','max')).reset_index()
    data=count.merge(coordinates,left_on='site',right_on='canonical_site_id',how='left',validate='one_to_one')
    assert len(data)==2232 and data.weather_cell.is_unique and data.zone.notna().all()
    data['later_evaluation']=data.site.isin(labels.loc[labels['case'].eq('later_held'),'site'])
    path=OUT/'insights/local_sites.csv.gz';record(path);local=set(pd.read_csv(path,usecols=['site']).site)
    path=OUT/'weather/common_units.csv.gz';record(path);weather=pd.read_csv(path)
    weather=set(weather.loc[weather.year.gt(2010),'site'])
    assert len(local)==822 and len(weather)==268 and set(data.loc[data.later_evaluation,'site'])>=local|weather
    data['local_cohort']=data.site.isin(local);data['weather_cohort']=data.site.isin(weather)
    data['availability']='not_in_later_cohort'
    data.loc[data.later_evaluation&~data.local_cohort&~data.weather_cohort,'availability']='neither'
    data.loc[data.later_evaluation&data.local_cohort&~data.weather_cohort,'availability']='local_only'
    data.loc[data.later_evaluation&~data.local_cohort&data.weather_cohort,'availability']='weather_only'
    data.loc[data.later_evaluation&data.local_cohort&data.weather_cohort,'availability']='both'
    data['annual_zone']=data.zone.astype(int)+1
    assert data.later_evaluation.sum()==1029 and data.observed_years.between(1,45).all()
    data.to_csv(FOLDER/'study_network.csv',index=False)
    summary=dict(qualified_sites=len(data),native_labels=int(data.native_labels.sum()),later_sites=int(data.later_evaluation.sum()),
        local_sites=822,weather_later_sites=268,availability=data[data.later_evaluation].availability.value_counts().to_dict(),
        annual_zone_counts=data.annual_zone.value_counts().sort_index().to_dict(),observation_year_range=[int(data.observed_years.min()),int(data.observed_years.max())])
    (FOLDER/'study_network_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    receipt=dict(complete=True,input_sha256=inputs,code_sha256=sha(__file__),native_labels_unique=True,availability_categories_disjoint=True,
        coordinates='recorded native weather-cell centres; one conservative monitoring-site group per cell in this qualified network',
        period=[1981,2025],output_sha256={name:sha(FOLDER/name) for name in ['study_network.csv','study_network_summary.json']})
    (FOLDER/'study_network_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
