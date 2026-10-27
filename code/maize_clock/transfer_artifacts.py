"""Verify scientific inputs, dependency versions and completed checkpoint bytes."""
import hashlib,json,math,platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dependency_fingerprint(paths=None):
    base=Path(__file__).parent
    paths=paths or [base/name for name in ['transfer_artifacts.py','transfer_fit.py','transfer_attention.py',
        'transfer_design.py','ordered_process.py','experiment.py','extensions.py','extension_boosting.py','references.py','core.py']]
    payload={'files':{str(Path(p).name):file_sha256(p) for p in paths},
             'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def verify_inputs(expected,root=None):
    for name,digest in expected.items():
        path=Path(name) if root is None else Path(root)/name
        if not path.is_file() or file_sha256(path)!=digest:raise ValueError('Input bytes changed: '+str(path))


def fit_request(metadata,fit_ids,tune_ids,config,seed,phase,input_sha256,epochs=40,batch=128,patience=8):
    digest=lambda b:hashlib.sha256(b).hexdigest()
    return dict(config=config,seed=seed,phase=phase,input_sha256=input_sha256,
        metadata_sha256=digest(json.dumps(metadata,sort_keys=True).encode()),
        fit_index_sha256=digest(np.asarray(fit_ids,dtype='<i8').tobytes()),
        tune_index_sha256=digest(np.asarray(tune_ids,dtype='<i8').tobytes()),
        source_code_sha256=dependency_fingerprint(),maximum_epochs=epochs,batch=batch,patience=patience)


def validate_artifact(stem,expected):
    stem=Path(stem);reportpath=stem.with_suffix('.json');checkpoint=stem.with_suffix('.pt')
    if not reportpath.is_file() or not checkpoint.is_file():raise ValueError('Unfinished fit: '+str(stem))
    report=json.loads(reportpath.read_text())
    for key,value in expected.items():
        if report.get(key)!=value:raise ValueError('Completed fit mismatch: '+key)
    if file_sha256(checkpoint)!=report.get('checkpoint_sha256'):raise ValueError('Checkpoint bytes changed: '+str(checkpoint))
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    for key,value in expected.items():
        if saved.get(key)!=value:raise ValueError('Checkpoint signature mismatch: '+key)
    actual=hashlib.sha256(json.dumps(saved['metadata'],sort_keys=True).encode()).hexdigest()
    if actual!=expected['metadata_sha256']:raise ValueError('Checkpoint metadata bytes changed')
    if not 1<=report['best_epoch']<=report['epochs_run']<=expected.get('maximum_epochs',report['epochs_run']):raise ValueError('Invalid fitting history')
    if not math.isfinite(report['tuning_crps']):raise ValueError('Nonfinite tuning score')
    return report


def atomic_torch_save(value,path):
    path=Path(path);temporary=path.with_name(path.name+'.tmp');torch.save(value,temporary);temporary.replace(path)


def atomic_json_save(value,path):
    path=Path(path);temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
