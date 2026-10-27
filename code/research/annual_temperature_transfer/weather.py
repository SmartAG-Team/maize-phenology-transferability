"""DWD historical station weather with explicit native day-window qualification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import re
import urllib.request
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from research.phenology_exploration.core import sha
from .climate import OUT
from .design import dataset

BASE='https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/'
FOLDER=OUT/'weather'


def utc_flags(parameters,dates):
    dates=pd.DatetimeIndex(dates);masks=[]
    for field in ['TNK','TXK']:
        mask=np.zeros(len(dates),bool)
        for _,row in parameters[parameters.Parameter.str.strip().eq(field)].iterrows():
            if not re.search(r'00:00\s*-\s*24:00\s*UTC',str(row.get('Zusatz-Info',''))):continue
            start=pd.to_datetime(str(row.Von_Datum).strip(),format='%Y%m%d',errors='coerce')
            end=pd.to_datetime(str(row.Bis_Datum).strip(),format='%Y%m%d',errors='coerce')
            if pd.isna(end):end=pd.Timestamp('2099-12-31')
            mask|=(dates>=start)&(dates<=end)
        masks.append(mask)
    return masks[0]&masks[1]


def distance_km(lat,lon,latitude,longitude):
    a,b,c,d=np.deg2rad([lat,lon,latitude,longitude]) if np.ndim(latitude)==0 else (np.deg2rad(lat),np.deg2rad(lon),np.deg2rad(latitude),np.deg2rad(longitude))
    return 6371.*2*np.arcsin(np.sqrt(np.clip(np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2,0,1)))


def weather_window(daily,dates,latitude,longitude,max_distance=10.):
    if daily.index.duplicated().any():raise ValueError('Duplicate station weather dates')
    f=daily.reindex(pd.DatetimeIndex(dates));temperature=f[['TNK','TXK']].to_numpy(float)
    valid=np.isfinite(temperature).all(1)&(temperature[:,0]<=temperature[:,1])&(temperature[:,0]>=-80)&(temperature[:,1]<=65)
    clock=f.utc_extremes.fillna(False).to_numpy(bool);quality=f.QN_4.eq(10).to_numpy()
    distances=distance_km(latitude,longitude,f.latitude.to_numpy(float),f.longitude.to_numpy(float))
    geographic=np.isfinite(distances)&(distances<=max_distance)
    return dict(complete=bool((valid&clock&quality&geographic).all()),temperature=temperature,
        max_distance_km=float(np.nanmax(distances)) if np.isfinite(distances).any() else np.nan,
        invalid_days=int((~valid).sum()),non_utc_days=int((~clock).sum()),unfinished_qc_days=int((~quality).sum()),
        geographic_mismatch_days=int((~geographic).sum()))


def station_daily(path):
    with ZipFile(path) as archive:
        assert archive.testzip() is None
        product=next(n for n in archive.namelist() if n.startswith('produkt_klima_tag_'))
        f=pd.read_csv(io.BytesIO(archive.read(product)),sep=';',encoding='latin1',skipinitialspace=True)
        f.columns=f.columns.str.strip();f.index=pd.to_datetime(f.MESS_DATUM.astype(str),format='%Y%m%d')
        parameters=next(n for n in archive.namelist() if n.startswith('Metadaten_Parameter_') and n.endswith('.txt'))
        p=pd.read_csv(io.BytesIO(archive.read(parameters)),sep=';',encoding='latin1',dtype=str).fillna('');p.columns=p.columns.str.strip()
        f['utc_extremes']=utc_flags(p,f.index)
        name=next(n for n in archive.namelist() if n.startswith('Metadaten_Geographie_') and n.endswith('.txt'))
        geography=pd.read_csv(io.BytesIO(archive.read(name)),sep=';',encoding='latin1',dtype=str).fillna('');geography.columns=geography.columns.str.strip()
        f['latitude']=np.nan;f['longitude']=np.nan;f['station_height_m']=np.nan
        for _,row in geography.iterrows():
            start=pd.to_datetime(str(row.von_datum).strip(),format='%Y%m%d',errors='coerce')
            end=pd.to_datetime(str(row.bis_datum).strip(),format='%Y%m%d',errors='coerce')
            if pd.isna(start):continue
            if pd.isna(end):end=pd.Timestamp('2099-12-31')
            active=(f.index>=start)&(f.index<=end)
            if f.loc[active,'latitude'].notna().any():raise ValueError('Overlapping station geography')
            f.loc[active,['latitude','longitude','station_height_m']]=[float(row['Geogr.Breite']),float(row['Geogr.Laenge']),float(row.Stationshoehe)]
    return f[['TNK','TXK','QN_4','utc_extremes','latitude','longitude','station_height_m']].sort_index()


def run():
    FOLDER.mkdir(exist_ok=True);raw=FOLDER/'raw';raw.mkdir(exist_ok=True);archives=raw/'stations';archives.mkdir(exist_ok=True)
    done=FOLDER/'dwd_qualification.json'
    if done.exists():
        meta=json.loads(done.read_text())
        if meta['code_sha256']!=sha(__file__):raise ValueError('DWD qualification code changed')
        for name,digest in meta['output_sha256'].items():
            if sha(FOLDER/name)!=digest:raise ValueError('DWD qualification output changed')
        print('Reusing qualified DWD windows',flush=True);return
    data,reference,_=dataset()
    rows=[]
    for line in (raw/'stations.txt').read_text(encoding='latin1').splitlines()[2:]:
        parts=line.split(maxsplit=6)
        if len(parts)>=6:rows.append(dict(station=int(parts[0]),start=parts[1],end=parts[2],height_m=float(parts[3]),latitude=float(parts[4]),longitude=float(parts[5])))
    stations=pd.DataFrame(rows).set_index('station')
    links={}
    for station,start,end in set(re.findall(r'tageswerte_KL_(\d{5})_(\d{8})_(\d{8})_hist.zip',(raw/'historical_index.html').read_text())):
        if start<='20070101' and end>='20251231':links[int(station)]=f'tageswerte_KL_{station}_{start}_{end}_hist.zip'
    stations=stations.loc[sorted(set(links)&set(stations.index))]
    tree=BallTree(np.deg2rad(stations[['latitude','longitude']].to_numpy()),metric='haversine')
    distance,index=tree.query(np.deg2rad(reference[['weather_latitude','weather_longitude']].to_numpy()),k=1)
    reference['station']=stations.index.to_numpy()[index.ravel()];reference['current_station_distance_km']=distance.ravel()*6371
    matched=reference[reference.current_station_distance_km.le(10)].copy()
    matched.to_csv(FOLDER/'station_matches.csv',index=False)
    chosen=sorted(matched.station.unique());print(json.dumps(dict(candidate_station_matches=len(matched),weather_stations=len(chosen),distance_limit_km=10)),flush=True)
    def download(station):
        dest=archives/links[station]
        if not dest.exists():
            error=None
            for attempt in range(3):
                try:content=urllib.request.urlopen(BASE+links[station],timeout=50).read();dest.write_bytes(content);error=None;break
                except Exception as e:error=e
            if error:raise error
        with ZipFile(dest) as z:assert z.testzip() is None
        return station,dest
    files=dict(ThreadPoolExecutor(max_workers=4).map(download,chosen))
    print(json.dumps(dict(downloaded_stations=len(files))),flush=True)
    by_site=matched.set_index('canonical_site_id');eligible=data.units[data.units.site.isin(by_site.index)&data.units.year.between(2001,2024)]
    outputs=[];arrays=[];audit=[];cache={}
    for station,group in eligible.groupby(eligible.site.map(by_site.station)):
        weather=station_daily(files[station])
        for unit in group.itertuples():
            site=by_site.loc[unit.site];key=(unit.site,unit.sowing_date)
            if key not in cache:
                dates=pd.date_range(unit.sowing_date,periods=367)
                cache[key]=weather_window(weather,dates,float(site.weather_latitude),float(site.weather_longitude))
            q=cache[key];info=dict(unit=unit.unit,u=int(unit.u),site=unit.site,year=int(unit.year),sowing_date=unit.sowing_date,station=int(station),
                **{k:v for k,v in q.items() if k not in ['temperature']})
            audit.append(info)
            if q['complete']:outputs.append(info);arrays.append(q['temperature'].astype(np.float32))
    a=pd.DataFrame(audit);q=pd.DataFrame(outputs);a.to_csv(FOLDER/'dwd_unit_audit.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    q.to_csv(FOLDER/'dwd_complete_units.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    np.savez_compressed(FOLDER/'dwd_native_weather.npz',u=q.u.to_numpy(int),temperature=np.stack(arrays))
    source=q[q.year.le(2010)].groupby('site').year.nunique();target=q[q.year.ge(2011)].groupby('site').year.nunique()
    summary=dict(complete=True,distance_basis='10 km from native phenology weather-cell centre to time-valid DWD station coordinates; not field GPS',
        day_basis='both TNK/TXK explicitly 00–24 UTC in native parameter metadata',quality='historical product; QN_4=10 every day',
        source_year_candidates=[2001,2010],target_year_candidates=[2011,2024],weather_window_days=367,
        matched_sites=len(matched),stations=len(files),candidate_units=len(a),complete_units=len(q),complete_sites=q.site.nunique(),
        source_sites_4_years=int(source.ge(4).sum()),source_sites_8_years=int(source.ge(8).sum()),later_sites_3_years=int(target.ge(3).sum()),
        code_sha256=sha(__file__),raw_sha256={str(p.relative_to(FOLDER)):sha(p) for p in files.values()},
        protected_inputs_unchanged=all(sha(p)==d for p,d in data.protected.items()),
        output_sha256={name:sha(FOLDER/name) for name in ['station_matches.csv','dwd_unit_audit.csv.gz','dwd_complete_units.csv.gz','dwd_native_weather.npz']})
    done.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['raw_sha256','output_sha256']}),flush=True)


if __name__=='__main__':run()
