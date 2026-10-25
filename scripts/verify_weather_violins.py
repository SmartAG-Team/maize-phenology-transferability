"""Independently verify Figure 10 paired values, weighting and original intervals."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--figure-dir',type=Path,default=ROOT/'outputs/figures');a=parser.parse_args();folder=a.figure_dir
    raw=pd.read_csv(ROOT/'evidence/weather/analysis/paired_site_differences.csv.gz')
    raw=raw[raw.period.eq('later_held')]
    ref=pd.read_csv(ROOT/'evidence/weather/analysis/paired_contrasts.csv');ref=ref[ref.period.eq('later_held')]
    pairs=pd.read_excel(folder/'fig10.xlsx','paired_values');summary=pd.read_excel(folder/'fig10.xlsx','plotted_data');curves=pd.read_excel(folder/'fig10.xlsx','density_curves')
    keys=['model','comparison','head','site','block','fold']
    assert not raw.duplicated(keys+['draw']).any() and raw.groupby(keys).draw.nunique().eq(3).all()
    computed=raw.groupby(keys,as_index=False).error.mean().rename(columns={'error':'recomputed'})
    q=pairs.merge(computed,on=keys,validate='one_to_one');assert len(q)==len(pairs)==30564
    np.testing.assert_allclose(q.difference_days,q.recomputed,atol=1e-12,rtol=1e-12)
    maximum=0.
    for r in summary.itertuples():
        p=pairs[pairs.model.eq(r.model)&pairs.comparison.eq(r.comparison)]
        assert len(p)==1698 and p.site.nunique()==268 and p['head'].nunique()==7
        np.testing.assert_allclose(p.groupby('head').density_weight.sum(),np.full(7,1/7),atol=1e-12)
        mean=p.groupby('head').difference_days.mean().mean()
        original=ref[ref.model.eq(r.model)&ref.comparison.eq(r.comparison)].iloc[0]
        np.testing.assert_allclose([mean,r.difference,r.ci_low,r.ci_high],[original.difference,original.difference,original.ci_low,original.ci_high],atol=1e-12)
        maximum=max(maximum,abs(mean-original.difference))
        density=curves[curves.model.eq(r.model)&curves.comparison.eq(r.comparison)]
        assert len(density)==601 and np.isfinite(density.density_per_day).all() and density.density_per_day.ge(0).all()
        np.testing.assert_allclose([density.difference_days.min(),density.difference_days.max()],[p.difference_days.min(),p.difference_days.max()],atol=1e-12)
        order=p.sort_values('difference_days');cdf=order.density_weight.cumsum()-order.density_weight/2
        np.testing.assert_allclose(np.interp(.5,cdf,order.difference_days),r.weighted_median_days,atol=1e-12)
    assert len(summary)==18
    with pymupdf.open(folder/'fig10.pdf') as pdf:
        page=pdf[0];assert abs(page.rect.width*2.54/72-19)<1e-4
        spans=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]
        assert spans and all(abs(s['size']-12)<1e-3 for s in spans)
        for s in spans:
            box=pymupdf.Rect(s['bbox']);assert box.x0>=-.01 and box.y0>=-.01 and box.x1<=page.rect.width+.01 and box.y1<=page.rect.height+.01
    with Image.open(folder/'fig10.png') as im:assert abs(im.width-19/2.54*600)<2 and abs(im.info['dpi'][0]-600)<.1
    result=dict(complete=True,paired_site_endpoint_values=30564,distributions=18,sites=268,pairs_per_distribution=1698,source_draws_averaged=3,equal_head_weights=True,original_means_and_95pct_intervals_preserved=True,maximum_mean_difference_days=maximum,full_observed_tails_retained=True,kde_scale='raw days',display_scale='signed asinh; linear width 2 days',width_cm=19,font_pt=12,dpi=600)
    out=ROOT/'outputs/weather_violin_verification.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
