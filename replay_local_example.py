"""Replay saved nonlinear T and historical local updating at three held sites."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from research.phenology_exploration.models import predict
from research.annual_temperature_transfer.insights import historical_local_update

folder=Path(__file__).parent/'local_example'
model=json.loads((folder/'model.json').read_text())
with np.load(folder/'native_inputs.npz') as z:inputs={name:z[name] for name in z.files}
history=pd.read_csv(folder/'history.csv.gz');reference=pd.read_csv(folder/'targets.csv.gz')
for which,frame in [('history',history),('target',reference[reference.budget.eq(0)].copy())]:
    lengths=inputs[which+'_lengths'];raw=inputs[which+'_raw']
    data=SimpleNamespace(raw=raw,lengths=lengths,train_lengths=lengths)
    predictions=predict(model,data,np.arange(len(raw)),train=(which=='history'))
    lookup={int(u):i for i,u in enumerate(inputs[which+'_u'])}
    values=predictions[frame.u.map(lookup).to_numpy(int),frame['head'].to_numpy(int)]
    assert np.allclose(values,frame.nonlinear_T,atol=1e-4,equal_nan=True)
    if which=='history':history['nonlinear_T']=values
    else:target=frame;target['nonlinear_T']=values
assert history.year.le(2010).all() and target.year.gt(2010).all()
assert not set(history.label_id)&set(target.label_id)
differences={}
for budget in [0,1,3,5]:
    values=historical_local_update(history,target,budget,inputs['kappa'])
    saved=reference[reference.budget.eq(budget)].set_index('label_id').loc[target.label_id,'nonlinear_T'].to_numpy()
    assert np.allclose(values,saved,atol=1e-4,equal_nan=True)
    differences[budget]=float(np.nanmax(abs(values-saved)))
print(json.dumps(dict(replay_passed=True,sites=int(target.site.nunique()),historical_labels=len(history),later_labels=len(target),differences_days=differences),indent=2))
