"""Acquire a full-calendar, equal-year 1981–2010 ERA5-Land reference."""
import calendar
import json
import os
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from research.phenology_exploration.core import sha

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/annual_temperature_transfer_20261024'
COLLECTION='ECMWF/ERA5_LAND/DAILY_AGGR'
TRANSFORM=[.1,0,-180.05,0,-.1,90.05]


def run():
    import ee
    folder=OUT/'climate';folder.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/europe_climate_transfer_20261015/data/sites.csv';original=sha(source)
    if (folder/'manifest.json').exists():
        m=json.loads((folder/'manifest.json').read_text())
        if m['input_sha256']!=original or m['code_sha256']!=sha(__file__):raise ValueError('Annual climate calculation changed')
        for name,digest in m['output_sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('Annual climate output changed')
        print('Reusing qualified annual climate',flush=True);return
    sites=pd.read_csv(source);ee.Initialize(project=os.environ.get('PHENOLOGY_EE_PROJECT','ee-gangzhaomodel'))
    collection=ee.ImageCollection(COLLECTION).filterDate('1981-01-01','2011-01-01').select(['temperature_2m_min','temperature_2m_max'])
    counts=ee.Dictionary({'images':collection.size(),'unique_dates':collection.aggregate_count_distinct('system:time_start'),
        'years':ee.List([collection.filterDate(f'{y}-01-01',f'{y+1}-01-01').size() for y in range(1981,2011)])}).getInfo()
    expected=[366 if calendar.isleap(y) else 365 for y in range(1981,2011)]
    if counts['images']!=sum(expected) or counts['unique_dates']!=sum(expected) or counts['years']!=expected:raise ValueError('Incomplete/duplicated reference calendar')
    annual=[]
    for year,days in zip(range(1981,2011),expected):
        c=collection.filterDate(f'{year}-01-01',f'{year+1}-01-01')
        def pair(im):
            low=im.select('temperature_2m_min');high=im.select('temperature_2m_max')
            valid=low.lte(high).And(low.gte(193.15)).And(high.lte(338.15))
            return low.add(high).divide(2).subtract(273.15).rename('mat_c').addBands(high.subtract(low).rename('dtr_c')).addBands(valid.unmask(0).rename('valid_days'))
        daily=c.map(pair);valid=daily.select('valid_days').sum()
        annual.append(daily.select(['mat_c','dtr_c']).mean().addBands(valid.eq(days).rename('complete_year')).addBands(valid.rename('valid_days')))
    years=ee.ImageCollection.fromImages(annual)
    image=years.select(['mat_c','dtr_c']).mean().addBands(years.select('complete_year').sum().rename('complete_years')).addBands(years.select('valid_days').sum().rename('valid_days'))
    cells=sites[['weather_cell','weather_latitude','weather_longitude']].drop_duplicates()
    if cells.weather_cell.duplicated().any():raise ValueError('Ambiguous native weather-cell coordinates')
    features=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.weather_longitude),float(r.weather_latitude)]),{'weather_cell':str(r.weather_cell)}) for r in cells.itertuples()])
    print(json.dumps(dict(requested_cells=len(cells),reference_days=sum(expected),reference_years=30)),flush=True)
    response=image.sampleRegions(collection=features,projection=ee.Projection('EPSG:4326',TRANSFORM),tileScale=8,geometries=False).getInfo()
    (folder/'earth_engine_response.json').write_text(json.dumps(response,separators=(',',':'))+'\n')
    records=pd.DataFrame([x['properties'] for x in response['features']]);assert records.weather_cell.is_unique
    f=sites[['canonical_site_id','weather_cell','weather_latitude','weather_longitude']].merge(records,on='weather_cell',how='left',validate='many_to_one')
    f['annual_climate_complete']=f.complete_years.eq(30)&f.valid_days.eq(sum(expected))&f.mat_c.notna()&f.dtr_c.ge(0)
    f['annual_mean_temperature_c']=f.mat_c.where(f.annual_climate_complete)
    f['annual_mean_dtr_c']=f.dtr_c.where(f.annual_climate_complete)
    f.to_csv(folder/'site_annual_climate.csv',index=False)
    assert sha(source)==original
    manifest=dict(reference_period=[1981,2010],months=list(range(1,13)),expected_days=sum(expected),calendar_counts=counts,
        collection=COLLECTION,project=os.environ.get('PHENOLOGY_EE_PROJECT','ee-gangzhaomodel'),definition='equal-year mean of complete Jan–Dec daily (Tmin+Tmax)/2; no clipping',
        source_units='K',output_units='degrees Celsius',day_basis='ERA5-Land UTC daily aggregates',projection='EPSG:4326',transform=TRANSFORM,
        source_sites=len(sites),source_cells=len(cells),returned_cells=len(records),complete_sites=int(f.annual_climate_complete.sum()),
        target_outcomes_used=False,retrieved_utc=datetime.now(timezone.utc).isoformat(),input_sha256=original,code_sha256=sha(__file__),
        output_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'})
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(complete_sites=int(f.annual_climate_complete.sum()),requested_sites=len(sites))),flush=True)


if __name__=='__main__':run()
