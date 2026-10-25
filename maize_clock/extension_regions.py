"""Declared north-region holdout, buffered from parameter fitting and tuning."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_KM=6371.0088

def unit_coordinates(latitude,longitude):
    lat=np.radians(np.asarray(latitude,float));lon=np.radians(np.asarray(longitude,float))
    return np.c_[np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]

def buffered_region(root,env):
    """North-Germany latitude>=53°, 25-km exclusion, original years retained."""
    root=Path(root);out=root/'results/extensions';out.mkdir(exist_ok=True,parents=True)
    frame=env.copy();north=frame.dataset_id.eq('dwd_annual').to_numpy()&(frame.latitude.to_numpy()>=53.)
    if not north.any():raise ValueError('No withheld north-region environments')
    coordinates=unit_coordinates(frame.latitude,frame.longitude)
    tree=cKDTree(np.unique(coordinates[north],axis=0));chord=tree.query(coordinates)[0]
    distance=2*EARTH_KM*np.arcsin(np.minimum(chord/2,1))
    safe=(distance>=25)&~north
    groups=set(frame.loc[north,'spatial_group'])
    safe &= ~frame.spatial_group.isin(groups).to_numpy()
    original=frame.new_partition.copy();frame['new_partition']='excluded'
    frame.loc[safe&original.eq('train').to_numpy(),'new_partition']='train'
    frame.loc[safe&original.eq('validation').to_numpy(),'new_partition']='validation'
    frame.loc[north&(frame.year<=2020).to_numpy(),'new_partition']='test_region_historical'
    frame.loc[north&(frame.year>=2021).to_numpy(),'new_partition']='test_region_future'
    audit=frame[['environment_id_model','spatial_group','dataset_id','latitude','longitude','year','new_partition']].copy();audit['distance_to_withheld_region_km']=distance
    descriptor={'north_boundary_latitude':53.,'buffer_km':25.,'selection':'North DWD locations omitted from parameter fitting and epoch selection; spatial groups shared with the region also omitted','historical_test_years':'1991–2020','future_test_years':'2021–2025','counts':frame.new_partition.value_counts().to_dict(),'minimum_calibration_distance_km':float(distance[frame.new_partition.eq('train')].min()),'environment_partition_sha256':hashlib.sha256('\n'.join(frame.environment_id_model+' '+frame.new_partition).encode()).hexdigest(),'architecture_selection':'Exploratory regional refitting of main tuning-selected architectures; main architecture development included observations from this region','fits_or_scores_used_to_change_architecture':False}
    dest=out/'buffered_region_design.json'
    if dest.exists():
        previous=json.loads(dest.read_text())
        if previous['environment_partition_sha256']!=descriptor['environment_partition_sha256']:raise ValueError('Buffered regional partition changed')
    else:
        audit.to_csv(out/'buffered_region_environments.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        dest.write_text(json.dumps(descriptor,indent=2)+'\n')
    return frame
