"""Calculation signatures and fail-closed reuse of research artifacts."""
from pathlib import Path
from .core import sha


def code_signature(*extra):
    folder=Path(__file__).parent
    names=['core.py','data.py','models.py','experiments.py','provenance.py']+list(extra)
    files=[folder/name for name in dict.fromkeys(names)]
    if 'neural.py' in extra:
        files.extend([folder.parent/'europe_climate_transfer/joint_data.py',
                      folder.parent/'sowing_day_transfer/multistage_models.py'])
    return {str(p):sha(p) for p in files}


def validate_receipt(receipt,expected,folder):
    if receipt.get('signature')!=expected:raise ValueError('Completed calculation signature changed')
    for name,digest in receipt.get('outputs',{}).items():
        path=Path(folder)/name
        if not path.is_file() or sha(path)!=digest:raise ValueError(f'Completed output changed: {name}')
    if not receipt.get('outputs'):raise ValueError('Completed output hashes are missing')
