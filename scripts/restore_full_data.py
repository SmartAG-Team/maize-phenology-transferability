"""Verify and restore versioned GitHub Release assets without path traversal."""
from pathlib import Path
import argparse, hashlib, json, tarfile, urllib.request
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path);p.add_argument('--download',action='store_true');a=p.parse_args()
    manifest=json.loads((ROOT/'full_data_assets.json').read_text())
    cache=a.assets or ROOT/'outputs/downloads';cache.mkdir(parents=True,exist_ok=True)
    for record in manifest['assets']:
        path=cache/record['name']
        if not path.exists():
            if not a.download:raise FileNotFoundError(f'{path}; use --download or --assets DIR')
            temp=path.with_suffix(path.suffix+'.download')
            urllib.request.urlretrieve(record['url'],temp)
            if sha(temp)!=record['sha256']:raise ValueError('Downloaded asset hash mismatch')
            temp.replace(path)
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise ValueError(f'Changed asset: {path.name}')
        with tarfile.open(path) as t:
            members=t.getmembers()
            for member in members:
                dest=(ROOT/member.name).resolve()
                if not dest.is_relative_to(ROOT) or not (member.isfile() or member.isdir()):
                    raise ValueError(f'Unsafe archive member: {member.name}')
                if dest.exists() and member.isfile():
                    stream=t.extractfile(member);h=hashlib.file_digest(stream,'sha256').hexdigest()
                    if sha(dest)!=h:raise ValueError(f'Refusing to overwrite changed local file: {member.name}')
            t.extractall(ROOT,filter='data')
        print(json.dumps({'restored':path.name,'sha256':record['sha256'],'members':len(members)}),flush=True)

if __name__=='__main__':main()
