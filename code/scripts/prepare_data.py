"""Reassemble the byte-exact large transfer table from checked Git-sized chunks."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def prepare():
    metadata=json.loads((ROOT/'data/evidence/climate_transfer_parts/parts.json').read_text());target=ROOT/'outputs/cache/site_transfer_errors.csv.gz'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and sha(target)==metadata['sha256']:return target
    temporary=target.with_suffix(target.suffix+'.assembling')
    with temporary.open('wb') as output:
        for record in metadata['parts']:
            path=ROOT/'data'/record['path'];assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],record['path']
            with path.open('rb') as source:
                for block in iter(lambda:source.read(1024*1024),b''):output.write(block)
    assert temporary.stat().st_size==metadata['bytes'] and sha(temporary)==metadata['sha256']
    temporary.replace(target);return target

if __name__=='__main__':print(prepare())
