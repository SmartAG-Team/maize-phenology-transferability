"""Refit the 30 matched 80-site benchmark cases without changing frozen results."""
from pathlib import Path
import argparse,json,shutil,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
import pandas as pd
from research.phenology_exploration import data as native,experiments,response_followup,neural,support_blend,adaptation,local_extensions
from research.phenology_exploration.core import sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    source=ROOT/'data/results/phenology_exploration_20261023/year_matched';dest=ROOT/'outputs/refits/benchmark'
    cases=[c for c in json.loads((source/'cases.json').read_text())['cases'] if c['n_sites']==80 and c['source_ez'] is None]
    assert len(cases)==30
    dest.mkdir(parents=True,exist_ok=True);(dest/'memberships').mkdir(exist_ok=True)
    design=json.loads((source/'design.json').read_text())
    (dest/'design.json').write_text(json.dumps(design,indent=2)+'\n');(dest/'cases.json').write_text(json.dumps(dict(cases=cases),indent=2)+'\n')
    data=native.load()
    for case in cases:
        old=source/'memberships'/f'{case["case"]}.csv.gz';new=dest/'memberships'/old.name
        assert sha(old)==case['membership_sha256'];shutil.copy2(old,new)
        members=pd.read_csv(new);labels=data.labels[data.labels.unit.isin(members.unit)&data.labels.pre2011_prefix_eligible]
        target=data.labels[data.labels[f'fold{case["fold"]}_case'].isin(['historical_held','later_held'])]
        assert members.site.nunique()==80 and len(members)==640 and not set(labels.site)&set(target.site)
        inner=members.set_index('unit').inner
        assert not set(labels[labels.unit.map(inner).ne(2)].block)&set(labels[labels.unit.map(inner).eq(2)].block)
    if not args.prepare_only:
        for module in [experiments,response_followup,neural,support_blend,adaptation,local_extensions]:module.OUT=dest
        for case in cases:experiments.fit_case(data,case,design)
        response_followup.run();neural.run();support_blend.run();adaptation.run();local_extensions.run()
    print(json.dumps(dict(cases=len(cases),prepare_only=args.prepare_only,output=str(dest.relative_to(ROOT)),frozen_inputs_unchanged=True)))

if __name__=='__main__':main()
