import sys
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
"""Replay saved ERA5-trained models under ERA5 and DWD forcing on24native units."""
from pathlib import Path
from types import SimpleNamespace
import json,pickle
import numpy as np
import pandas as pd
import torch
from research.phenology_exploration.models import predict,residual_prediction
from research.phenology_exploration.neural import ThermalTransformer,predict_network
from research.phenology_exploration.support_blend import support_weight,blend_dates

folder=Path(__file__).resolve().parents[1]/'data/example'
with (folder/'fits.pkl').open('rb') as f:saved=pickle.load(f)
checkpoint=torch.load(folder/'thermal_transformer.pt',weights_only=False,map_location='cpu')
with np.load(folder/'native_inputs.npz') as z:inputs={name:z[name] for name in z.files}
ids=np.arange(len(inputs['original_u']));lookup={int(u):i for i,u in enumerate(inputs['original_u'])};results={}
for product in ['era5','dwd']:
    data=SimpleNamespace(raw=inputs[product],context=inputs['context'],lengths=inputs['lengths'],train_lengths=inputs['lengths'])
    fits=saved['models'];base=predict(fits['T'],data,ids);thermal=predict(fits['nonlinear_T'],data,ids)
    neural=base+saved['mlp_choice']['strength']*residual_prediction(fits['MLP'],data,ids,base)
    forest=base+saved['rf_choice']['strength']*residual_prediction(fits['RF'],data,ids,base)
    w,_=support_weight(inputs['source_climate'],data.context[:,:2]);blend=blend_dates(neural,thermal,w[:,None],data.lengths[:,None])
    net=ThermalTransformer(checkpoint['initial'],23);net.load_state_dict(checkpoint['state'])
    transformer=predict_network(net,data,ids,checkpoint['norm'],False,device='cpu')
    ref=pd.read_csv(folder/f'{product}_reference.csv.gz');row=ref.u.map(lookup).to_numpy(int);head=ref['head'].to_numpy(int);differences={}
    for model,values in [('T',base),('nonlinear_T',thermal),('residual_RF',forest),('residual_MLP',neural),('thermal_transformer',transformer),('support_blend',blend)]:
        a=values[row,head];b=ref[model].to_numpy();tolerance=.02 if model=='thermal_transformer' else 1e-5
        assert np.allclose(a,b,atol=tolerance,equal_nan=True),(product,model)
        differences[model]=float(np.nanmax(abs(a-b)))
    results[product]=differences
print(json.dumps(dict(replay_passed=True,native_units=len(ids),products=results),indent=2))
