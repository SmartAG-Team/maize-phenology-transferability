"""Render all released figures using exact tables and original scientific styles."""
from pathlib import Path
import sys,json
PROJECT=Path(__file__).resolve().parents[2]
ROOT=PROJECT
sys.path.insert(0,str(PROJECT/'code'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from research.annual_temperature_transfer import figures as f,spatial_figures as spatial
from research.phenology_exploration.core import sha

original_save=f.save

def save(fig,stem,tables,caption,layout=True):
    caption=(ROOT/'paper/figures'/f'{stem}_caption.txt').read_text().strip()
    return original_save(fig,stem,tables,caption,layout)


def weather_agreement():
    data=pd.read_excel(ROOT/'paper/figures/fig9.xlsx',sheet_name='plotted_data')
    fig,axes=plt.subplots(1,2,figsize=(19/2.54,10/2.54));images=[]
    for ax,variable,mark in zip(axes,['mean','dtr'],['(a)','(b)']):
        q=data[data.variable.eq(variable)];x=np.sort(q.dwd_c.unique());y=np.sort(q.era5_c.unique())
        h=q.pivot(index='dwd_c',columns='era5_c',values='site_days').reindex(index=x,columns=y).to_numpy()
        xx=np.r_[x-.5,x[-1]+.5];yy=np.r_[y-.5,y[-1]+.5]
        images.append((ax,h,xx,yy));ax.plot([xx[0],xx[-1]],[yy[0],yy[-1]],'--',color='.3',linewidth=.8)
        name='daily mean' if variable=='mean' else 'diurnal range'
        ax.set_xlabel(f'DWD {name} (°C)');ax.set_ylabel(f'ERA5 {name} (°C)');f.panel(ax,mark)
    maximum=max(float(h.max()) for _,h,_,_ in images)
    for ax,h,x,y in images:im=ax.pcolormesh(x,y,np.ma.masked_where(h.T==0,h.T),cmap='viridis',norm=LogNorm(1,maximum),rasterized=True)
    fig.subplots_adjust(left=.1,right=.86,bottom=.18,top=.93,wspace=.4)
    bar=fig.colorbar(im,cax=fig.add_axes([.9,.22,.025,.65]));bar.set_label('Unique site-days')
    ticks=[10**k for k in range(int(np.floor(np.log10(maximum)))+1)];bar.set_ticks(ticks,labels=[f'{v:,}' for v in ticks])
    agreement=pd.read_excel(ROOT/'paper/figures/fig9.xlsx',sheet_name='agreement')
    save(fig,'fig9',{'plotted_data':data,'agreement':agreement},'',layout=False)


def require_primary_evidence():
    folder=ROOT/'data/evidence/full_source_benchmark'
    receipt=folder/'verification.json'
    if not receipt.is_file():
        raise FileNotFoundError(f'Missing canonical full-source primary figure evidence: {receipt}')
    verification=json.loads(receipt.read_text())
    if not verification.get('complete'):
        raise ValueError('Incomplete canonical full-source primary figure evidence')
    for name in ['stage_performance.csv','correction_summary.csv','pooled_map.csv','pooled_cells.csv']:
        path=folder/name
        if not path.is_file():
            raise FileNotFoundError(f'Missing canonical full-source primary figure evidence: {path}')
        if verification.get('output_sha256',{}).get(name)!=sha(path):
            raise ValueError(f'Changed canonical full-source primary figure evidence: {name}')
    return folder


def configure():
    f.OUT=ROOT/'data/evidence';f.DEST=ROOT/'outputs';f.FIG=f.DEST/'figures'
    spatial.OUT=f.OUT;spatial.FOLDER=f.OUT/'spatial'
    f.save=save;f.weather_agreement=weather_agreement


def main(figure=None):
    require_primary_evidence()
    from research.mae_bias_revision.figures import render
    result=render(ROOT/'outputs/figures',ROOT/'data/evidence/mae_bias_revision_20261026',only=figure)
    f.FIG=result

    print(f.FIG)
    return f.FIG


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--figure',choices=['fig'+str(i) for i in range(1,11)]+['figs1']);args=parser.parse_args()
    main(args.figure)
