"""Check every tracked release file against the scientific snapshot manifest."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

if __name__=='__main__':
    manifest=json.loads((ROOT/'release_manifest.json').read_text())
    for record in manifest['files']:
        path=ROOT/record['path'];assert path.is_file() and path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],record['path']
    print(json.dumps({'verified':True,'files':len(manifest['files']),'version':manifest['version']}))
