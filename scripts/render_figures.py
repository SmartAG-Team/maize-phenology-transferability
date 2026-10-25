"""Render all released figures using exact tables and original scientific styles."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from research.annual_temperature_transfer import figures as f,spatial_figures as spatial

f.OUT=ROOT/'evidence';f.DEST=ROOT/'outputs';f.FIG=f.DEST/'figures'
spatial.OUT=f.OUT;spatial.FOLDER=f.OUT/'spatial'
original_save=f.save

def save(fig,stem,tables,caption,layout=True):
    caption=(ROOT/'plot_data'/f'{stem}_caption.txt').read_text().strip()
    return original_save(fig,stem,tables,caption,layout)

f.save=save

def weather_agreement():
    data=pd.read_excel(ROOT/'plot_data/fig9.xlsx',sheet_name='plotted_data')
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
    agreement=pd.read_excel(ROOT/'plot_data/fig9.xlsx',sheet_name='agreement')
    save(fig,'fig9',{'plotted_data':data,'agreement':agreement},'',layout=False)

f.weather_agreement=weather_agreement
if __name__=='__main__':
    f.run()
    print(f.FIG)
