"""Paired, identically sampled DWD/ERA5 refits with station-aware teachers."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from research.phenology_exploration.core import sha
from .climate import OUT
from .weather_design import freeze_weather,FOLDER
from . import fit


def product_data(data,common,provider):
    with np.load(FOLDER/'common_inputs.npz') as z:
        assert np.array_equal(z['u'],common.u)
        raw=data.raw.copy();raw[common.u.to_numpy(int),:,:2]=z[provider]
    protected=dict(data.protected)
    for path in [FOLDER/'common_inputs.npz',FOLDER/'common_units.csv.gz',FOLDER/'common_qualification.json',
                 Path(__file__),Path(__file__).parent/'weather_design.py',Path(__file__).parent/'weather.py']:
        protected[str(path)]=sha(path)
    return replace(data,raw=raw,protected=protected)


def case_data(data,fold):
    path=FOLDER/'paired_design/source_pools'/f'fold{fold}.csv.gz'
    pool=pd.read_csv(path);mapping=pool.set_index('unit').block
    labels=data.labels.copy();mask=labels.unit.isin(mapping.index)
    labels.loc[mask,'block']=labels.loc[mask,'unit'].map(mapping).to_numpy()
    protected=dict(data.protected);protected[str(path)]=sha(path)
    return replace(data,labels=labels,protected=protected)


def run():
    data,common,cases=freeze_weather();original_out=fit.OUT
    parent=FOLDER/'paired_design';design=json.loads((parent/'design.json').read_text())
    for provider in ['dwd','era5']:
        folder=FOLDER/'refits'/provider;(folder/'memberships').mkdir(parents=True,exist_ok=True)
        record=dict(**design,training_weather=provider,parent_design_sha256=sha(parent/'design.json'))
        path=folder/'design.json'
        if path.exists() and json.loads(path.read_text())!=record:raise ValueError('Weather provider design changed')
        path.write_text(json.dumps(record,indent=2)+'\n')
        (folder/'cases.json').write_text((parent/'cases.json').read_text())
        for case in cases:shutil.copy2(parent/'memberships'/f'{case["case"]}.csv.gz',folder/'memberships'/f'{case["case"]}.csv.gz')
        weather=product_data(data,common,provider);fit.OUT=folder
        for case in cases:fit.run_case(case_data(weather,case['fold']),case)
        assert all(sha(p)==value for p,value in weather.protected.items())
        print(json.dumps(dict(training_weather=provider,complete_cases=len(cases),protected_inputs_unchanged=True)),flush=True)
        fit.OUT=original_out
    (FOLDER/'refit_verification.json').write_text(json.dumps(dict(complete=True,providers=['dwd','era5'],cases_per_provider=len(cases),
        identical_memberships=True,common_native_units=len(common),source_station_groups_preserved=True,
        source_year_budget=4,site_budget=40,code_sha256=sha(__file__)),indent=2)+'\n')


if __name__=='__main__':run()
