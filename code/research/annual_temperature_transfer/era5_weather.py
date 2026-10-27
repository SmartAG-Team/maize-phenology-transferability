"""Independent UTC ERA5-Land daily forcing on DWD-qualified native cells."""
import calendar
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from research.phenology_exploration.core import sha
from .climate import OUT,COLLECTION,TRANSFORM


def run():
    import ee
    folder=OUT/'weather/era5_land';folder.mkdir(parents=True,exist_ok=True)
    input_file=OUT/'weather/dwd_complete_units.csv.gz';units=pd.read_csv(input_file)
    sites=pd.read_csv(OUT/'site_zones.csv').set_index('canonical_site_id').loc[sorted(units.site.unique())].reset_index()
    signature=dict(input_sha256=sha(input_file),sites_sha256=sha(OUT/'site_zones.csv'),code_sha256=sha(__file__))
    ee.Initialize(project=os.environ.get('PHENOLOGY_EE_PROJECT','ee-gangzhaomodel'))
    points=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.weather_longitude),float(r.weather_latitude)]),{'site':str(r.canonical_site_id)}) for r in sites.itertuples()])
    collection=ee.ImageCollection(COLLECTION)
    years=list(range(2001,2026))
    def acquire(year):
        meta=folder/f'{year}.json';dest=folder/f'{year}.npz';raw=folder/f'{year}_response.json'
        if meta.exists():
            m=json.loads(meta.read_text())
            if m['signature']!=signature:raise ValueError('ERA5 forcing signature changed')
            if sha(dest)!=m['npz_sha256'] or sha(raw)!=m['response_sha256']:raise ValueError('ERA5 forcing output changed')
            return m
        c=collection.filterDate(f'{year}-01-01',f'{year+1}-01-01').sort('system:time_start').select(['temperature_2m_min','temperature_2m_max'])
        dates=pd.date_range(f'{year}-01-01',f'{year}-12-31');expected=len(dates)
        counts=ee.Dictionary({'n':c.size(),'unique':c.aggregate_count_distinct('system:time_start')}).getInfo()
        if counts['n']!=expected or counts['unique']!=expected:raise ValueError(f'Incomplete ERA5 calendar: {year}')
        image=c.toArray().rename('temperature_k')
        result=image.sampleRegions(collection=points,projection=ee.Projection('EPSG:4326',TRANSFORM),tileScale=8,geometries=False).getInfo()
        raw.write_text(json.dumps(result,separators=(',',':'))+'\n')
        records={r['properties']['site']:np.asarray(r['properties']['temperature_k'],np.float32) for r in result['features']}
        site_ids=sites.canonical_site_id.to_numpy(str);temperature=np.full((len(site_ids),expected,2),np.nan,np.float32)
        for i,site in enumerate(site_ids):
            if site in records:
                if records[site].shape!=(expected,2):raise ValueError('ERA5 forcing array/date mismatch')
                temperature[i]=records[site]-273.15
        valid=np.isfinite(temperature).all(2)&(temperature[:,:,0]<=temperature[:,:,1])&(temperature[:,:,0]>=-80)&(temperature[:,:,1]<=65)
        np.savez_compressed(dest,sites=site_ids,dates=dates.to_numpy(dtype='datetime64[D]').astype('int64'),temperature=temperature,valid=valid)
        m=dict(signature=signature,year=year,days=expected,requested_sites=len(sites),returned_sites=len(records),complete_sites=int(valid.all(1).sum()),
            units='degrees Celsius',source_units='K',collection=COLLECTION,day_basis='00–24 UTC native daily aggregates',transform=TRANSFORM,
            response_sha256=sha(raw),npz_sha256=sha(dest),retrieved_utc=datetime.now(timezone.utc).isoformat())
        meta.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(dict(year=year,sites=len(sites),complete_sites=m['complete_sites'])),flush=True)
        return m
    metadata=list(ThreadPoolExecutor(max_workers=2).map(acquire,years))
    (folder/'manifest.json').write_text(json.dumps(dict(complete=True,signature=signature,years=years,annual_receipts_sha256={str(y):sha(folder/f'{y}.json') for y in years}),indent=2)+'\n')


if __name__=='__main__':run()
