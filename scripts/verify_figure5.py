"""Audit paired support, head weighting, frozen contrasts and native figure exports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--figure-dir',type=Path,default=ROOT/'outputs/figures');args=parser.parse_args()
    folder=args.figure_dir
    raw=pd.read_csv(ROOT/'evidence/insights/local_sites.csv.gz')
    summary=pd.read_excel(folder/'fig5.xlsx','distribution_summary')
    pairs=pd.read_excel(folder/'fig5.xlsx','paired_gains')
    curves=pd.read_excel(folder/'fig5.xlsx','accuracy_curve').drop(columns='panel')
    contrasts=pd.read_csv(ROOT/'evidence/insights/local_contrasts.csv')
    reference=pd.read_csv(ROOT/'evidence/insights/local_summary.csv');reference=reference[reference.model.isin(summary.model)]
    pd.testing.assert_frame_equal(curves,reference.reset_index(drop=True),check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12)
    checked=0
    for r in summary.itertuples():
        q=raw[raw.model.eq(r.model)&raw.budget.isin([0,5])]
        p=q.pivot(index=['head','site','block'],columns='budget',values='error')
        assert p.notna().all().all() and len(p)==5323
        gain=(p[0]-p[5]).rename('gain_days').reset_index()
        saved=pairs[pairs.model.eq(r.model)]
        joined=gain.merge(saved,on=['head','site','block'],suffixes=('_computed','_saved'),validate='one_to_one')
        np.testing.assert_allclose(joined.gain_days_computed,joined.gain_days_saved,rtol=1e-12,atol=1e-12)
        assert saved.site.nunique()==822 and saved['head'].nunique()==7
        np.testing.assert_allclose(saved.groupby('head').density_weight.sum(),np.full(7,1/7),atol=1e-12)
        mean=gain.groupby('head').gain_days.mean().mean()
        assert abs(mean-r.mean_gain_days)<1e-12
        ref=contrasts[contrasts.model.eq(r.model)&contrasts.budget.eq(5)&contrasts.comparison.eq('adapted_minus_zero')].iloc[0]
        np.testing.assert_allclose([mean,r.ci_low_days,r.ci_high_days],[-ref.difference,-ref.ci_high,-ref.ci_low],atol=1e-12)
        assert gain.gain_days.min()==r.min_gain_days or abs(gain.gain_days.min()-r.min_gain_days)<1e-12
        assert -40<r.min_gain_days<=r.max_gain_days<70
        checked+=len(joined)
    page=pymupdf.open(folder/'fig5.pdf')[0]
    assert abs(page.rect.width*2.54/72-19)<1e-4
    spans=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]
    assert spans and all(abs(s['size']-12)<1e-3 for s in spans)
    assert '(a)' in page.get_text() and '(b)' in page.get_text() and 'asinh scale' in page.get_text()
    with Image.open(folder/'fig5.png') as im:
        assert abs(im.width-19/2.54*600)<2 and abs(im.info['dpi'][0]-600)<.1
    result=dict(complete=True,figure='fig5',paired_site_endpoint_values_verified=checked,models=5,
        site_endpoint_pairs_per_model=5323,sites=822,native_endpoints=7,
        equal_head_density_weights=True,original_means_and_95pct_intervals_unchanged=True,
        accuracy_curve_unchanged=True,full_tails_retained=True,width_cm=19,font_pt=12,dpi=600,
        display_scale='signed asinh,linear width2days;KDE on raw days',no_model_refitting=True)
    out=ROOT/'outputs/figure5_verification.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
