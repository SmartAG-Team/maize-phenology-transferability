"""Native-size publication figures with distinct questions and exact plot data."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,Normalize,LogNorm
from matplotlib.transforms import Bbox
from .climate import OUT
from .analysis import MODELS

DEST=OUT/'report';FIG=DEST/'figures'
LABELS={'T':'T','nonlinear_T':'Nonlinear T','residual_RF':'RF','residual_MLP':'MLP','thermal_transformer':'Thermal Transformer','support_blend':'Support blend'}
COLORS={'T':'#666666','nonlinear_T':'#E69F00','residual_RF':'#009E73','residual_MLP':'#0072B2','thermal_transformer':'#56B4E9','support_blend':'#882255'}
CAPTIONS={}
MAIN_FIGURES=['fig'+str(i) for i in range(1,11)]
SUPPLEMENTARY_FIGURES=['figs1']


def panel(ax,text):ax.text(0,1.025,text,transform=ax.transAxes,ha='left',va='bottom',fontsize=12)


def save(fig,stem,tables,caption,layout=True):
    if getattr(fig,'_suptitle',None) is not None and fig._suptitle.get_text().strip():
        raise ValueError('Figure titles belong in captions: '+stem)
    if any(ax.get_title(loc=loc).strip() for ax in fig.axes for loc in ['left','center','right']):
        raise ValueError('Panel titles are not permitted: '+stem)
    FIG.mkdir(parents=True,exist_ok=True);width=19/2.54;pad=.025
    for _ in range(12):
        if layout:fig.tight_layout(pad=.5)
        fig.canvas.draw();tight=fig.get_tightbbox(fig.canvas.get_renderer());delta=width-2*pad-tight.width
        if abs(delta)<.001:break
        fig.set_size_inches(fig.get_figwidth()+delta,fig.get_figheight(),forward=True)
    center=tight.x0+tight.width/2;bounds=Bbox.from_bounds(center-width/2,tight.y0-pad,width,tight.height+2*pad)
    for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{stem}.{ext}',bbox_inches=bounds,dpi=600)
    with pd.ExcelWriter(FIG/f'{stem}.xlsx',engine='openpyxl') as writer:
        for name,data in tables.items():data.to_excel(writer,sheet_name=name[:31],index=False)
        pd.DataFrame([dict(field='width',value=19,unit='cm'),dict(field='font',value=12,unit='pt'),dict(field='raster',value=600,unit='dpi'),
            dict(field='all_target_invalid_date_loss',value=366,unit='days'),dict(field='error_and_penalty_columns',value='days',unit='days'),
            dict(field='temperature_columns',value='degrees Celsius',unit='degrees Celsius'),dict(field='failure_and_completion',value='fraction',unit='dimensionless'),
            dict(field='zone_order',value='0=coldest,4=warmest;figure ticks1..5',unit='index')]).to_excel(writer,sheet_name='metadata',index=False)
    (FIG/f'{stem}_caption.txt').write_text(caption+'\n');CAPTIONS[stem]=caption;plt.close(fig)


def matrix_figure(penalty=True):
    f=pd.read_csv(OUT/'matrix.csv');f=f[f.period.eq('later_held')&f.source_zone.ge(0)].copy()
    value='penalty' if penalty else 'error';f['panel']=f.model.map(dict(zip(MODELS,'abcdef')))
    vmin=min(-.5,float(np.floor(f[value].min()))) if penalty else float(np.floor(f[value].min()))
    vmax=float(np.ceil(f[value].max()));norm=TwoSlopeNorm(0,vmin,vmax) if penalty else Normalize(vmin,vmax)
    fig,axes=plt.subplots(2,3,figsize=(19/2.54,15/2.54),sharex=True,sharey=True)
    fig.subplots_adjust(left=.075,right=.985,top=.96,bottom=.18,wspace=.18,hspace=.32)
    for i,(ax,model) in enumerate(zip(axes.flat,MODELS)):
        a=f[f.model.eq(model)].pivot(index='source_zone',columns='target_zone',values=value).reindex(index=range(5),columns=range(5))
        im=ax.imshow(a,cmap='RdBu_r' if penalty else 'viridis',norm=norm,aspect='auto')
        for r in range(5):
            for c in range(5):
                x=a.iloc[r,c];display=0. if np.isfinite(x) and abs(x)<.05 else x
                light_text=np.isfinite(x) and ((norm(x)>.78 or norm(x)<.19) if penalty else norm(x)<.5)
                ax.text(c,r,f'{display:.1f}' if np.isfinite(display) else 'NA',ha='center',va='center',fontsize=12,
                    color='white' if light_text else '#222222')
        ax.set_xticks(range(5),range(1,6));ax.set_yticks(range(5),range(1,6));panel(ax,'('+chr(97+i)+')')
        if i>=3:ax.set_xlabel('Target zone')
        if i%3==0:ax.set_ylabel('Source zone')
    cax=fig.add_axes([.18,.065,.68,.025]);bar=fig.colorbar(im,cax=cax,orientation='horizontal')
    bar.set_label('Excess transfer loss (days)' if penalty else 'Penalised MAE (days)')
    caption=('Fig. 6. Annual-temperature source-to-target transfer penalties. ' if penalty else 'Fig. S1. Absolute errors underlying the annual-temperature transfer experiment. ')
    caption+=('(a) T. (b) Nonlinear T. (c) Residual RF. (d) Residual MLP. (e) Causal thermal Transformer. (f) Climate-support blend. '
        'Zones 1–5 are cold-to-warm quintiles of complete 1981–2010 January–December temperature within the German exact-date network. '
        'Each source fit uses 40 sites and eight distinct years/site ending by 2010; three source draws and five spatial folds share identical 2011–2025 target observations. ')
    caption+=('A cell is the paired error difference between training in its source zone and training in the target zone itself, evaluated on the same native target heads and site-years. Diagonals are zero by construction. Positive values indicate excess transfer loss. ' if penalty else 'Cells show equal-head, site-year-balanced errors in days. These values describe absolute accuracy; Fig. 6 separately adjusts for the target zone’s own trained-model error. ')
    caption+='All seven native heads receive equal weight. Invalid dates receive a 366-day loss. Exact native definitions remain unchanged; these are retrospective German-network comparisons, not causal temperature responses. Confidence intervals, stage support and source sampling details accompany the plotting data.'
    save(fig,'fig6' if penalty else 'figs1',{'plotted_data':f},caption,layout=False)


def curves():
    f=pd.read_csv(OUT/'temperature_distance_curves.csv');f=f[f.period.eq('later_held')].copy()
    fig,axes=plt.subplots(3,1,figsize=(19/2.54,17.5/2.54),sharex=True,gridspec_kw={'height_ratios':[.55,1,1]})
    support=f[f.model.eq('nonlinear_T')].sort_values('signed_mat_distance_c')
    axes[0].plot(support.signed_mat_distance_c,support.error_sites,'o-',color='.4',markersize=4)
    for row in support.itertuples():axes[0].text(row.signed_mat_distance_c,row.error_sites*1.18,str(int(row.error_sites)),ha='center',va='bottom',fontsize=12)
    axes[0].set_yscale('log');axes[0].set_yticks([10,100,1000],['10','100','1000']);axes[0].set_ylim(6,1600);axes[0].set_ylabel('Target sites');panel(axes[0],'(a)')
    for ax,value,mark in zip(axes[1:],['error','penalty'],['(b)','(c)']):
        for model in MODELS:
            a=f[f.model.eq(model)].sort_values('signed_mat_distance_c')
            ax.plot(a.signed_mat_distance_c,a[value],'-o',markersize=4,color=COLORS[model],label=LABELS[model])
            ax.fill_between(a.signed_mat_distance_c,a[value+'_ci_low'],a[value+'_ci_high'],color=COLORS[model],alpha=.08,linewidth=0)
        ax.grid(axis='y',color='.92',linewidth=.6);ax.axvline(0,color='.6',linestyle='--',linewidth=.7);panel(ax,mark)
    axes[1].set_ylabel('Penalised MAE (days)');axes[2].set_ylabel('Excess loss (days)');axes[2].axhline(0,color='.5',linestyle='--',linewidth=.7)
    axes[2].set_xlabel('Target minus source median annual temperature (°C)')
    handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.065),ncol=3,frameon=False)
    caption=('Fig. 8. Transfer along the annual-temperature gradient. (a) Unique target-site support. (b) Penalised MAE. (c) Excess loss relative to target-zone training. '
        'Negative horizontal differences denote colder targets. Off-diagonal comparisons use fixed distance bins and actual within-bin mean distances. '
        'Native outcomes are year/site/head balanced; source draws and conditions are averaged within target-site/head/bin before site averaging. '
        'Shading gives conditional 95% target-block intervals; incomplete seven-head support remains undefined. '
        'A site can enter multiple bins, so counts are not additive. The far-cold bin contains nine sites. Invalid dates incur 366-day loss. '
        'The two directions retain different target populations and do not identify causal warming or cooling effects.')
    save(fig,'fig8',{'plotted_data':f,'site_support':support},caption)


def weather_effects():
    from .violin import weather_difference_distribution
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    raw=pd.read_csv(OUT/'weather/analysis/paired_site_differences.csv.gz')
    reference=pd.read_csv(OUT/'weather/analysis/paired_contrasts.csv')
    paired,summary,densities=weather_difference_distribution(raw,reference,MODELS)
    fig,axes=plt.subplots(2,1,figsize=(19/2.54,15.8/2.54),sharex=True,sharey=True)
    fig.subplots_adjust(left=.15,right=.985,bottom=.11,top=.82,hspace=.22)
    cases=[('fixed_era5_swap_dwd',-.18,'#0072B2'),('fixed_dwd_swap_era5',.18,'#D55E00')]
    for comparison,offset,color in cases:
        for i,model in enumerate(MODELS):
            q=densities[densities.model.eq(model)&densities.comparison.eq(comparison)]
            r=summary[summary.model.eq(model)&summary.comparison.eq(comparison)].iloc[0]
            center=i+offset;width=.14*q.relative_density
            axes[0].fill_betweenx(q.difference_days,center-width,center+width,color=color,alpha=.30,edgecolor=color,linewidth=.7)
            axes[0].errorbar(center+.035,r.difference,yerr=[[r.difference-r.ci_low],[r.ci_high-r.difference]],fmt='o',color='#222222',capsize=2,markersize=4,zorder=4)
            axes[0].plot(center-.035,r.weighted_median_days,'D',markerfacecolor='white',markeredgecolor='#222222',markersize=3.5,zorder=4)
    for i,model in enumerate(MODELS):
        q=densities[densities.model.eq(model)&densities.comparison.eq('matched_dwd_minus_era5')]
        r=summary[summary.model.eq(model)&summary.comparison.eq('matched_dwd_minus_era5')].iloc[0]
        width=.30*q.relative_density
        axes[1].fill_betweenx(q.difference_days,i-width,i+width,color=COLORS[model],alpha=.34,edgecolor=COLORS[model],linewidth=.7)
        axes[1].errorbar(i+.055,r.difference,yerr=[[r.difference-r.ci_low],[r.ci_high-r.difference]],fmt='o',color='#222222',capsize=3,markersize=4,zorder=4)
        axes[1].plot(i-.055,r.weighted_median_days,'D',markerfacecolor='white',markeredgecolor='#222222',markersize=4,zorder=4)
    for ax,mark in zip(axes,['(a)','(b)']):
        panel(ax,mark);ax.set_yscale('asinh',linear_width=2);ax.set_ylim(-400,400)
        ticks=[-100,-30,-10,-3,0,3,10,30,100];ax.set_yticks(ticks,[str(t).replace('-','−') for t in ticks])
        ax.axhline(0,color='.45',linestyle='--',linewidth=.75);ax.grid(axis='y',color='.93',linewidth=.6);ax.set_axisbelow(True)
        ax.set_ylabel('Paired MAE difference (days)')
    axes[1].set_xticks(range(6),['T','Nonlinear\nT','RF','MLP','Thermal\nTransformer','Support\nblend']);axes[1].set_xlim(-.55,5.55)
    fig.legend(handles=[Patch(facecolor='#0072B2',alpha=.30,label='ERA5 fit → DWD input'),Patch(facecolor='#D55E00',alpha=.30,label='DWD fit → ERA5 input')],loc='upper center',bbox_to_anchor=(.5,1.005),ncol=2,frameon=False)
    fig.legend(handles=[Line2D([],[],color='#222222',marker='o',markersize=4,label='Mean and 95% CI'),Line2D([],[],color='none',marker='D',markerfacecolor='white',markeredgecolor='#222222',markersize=4,label='Median')],loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    caption=('Fig. 10. Distributions of weather-product effects on common maize dates. '
        '(a) Paired error changes under fixed-parameter weather swaps, relative to each fit’s own product. '
        '(b) DWD-minus-ERA5 errors after product-specific refitting; negative values favour the DWD-specific fit. '
        'Each distribution contains 1,698 site–endpoint pairs at 268 sites, after averaging three source draws. '
        'Density weights are equal across seven native endpoints, then across sites within endpoints. Gaussian kernels use Scott bandwidth in untransformed days and equal maximum widths. '
        'The signed asinh axis (2-day linear width) retains all tails; ticks are days. Black circles/whiskers retain the mean contrasts and conditional 95% intervals from connected station/spatial-block resampling; white diamonds show weighted medians. '
        'Matched source budgets are 40 sites × four 2001–2010 years in five folds; targets span 2011–2024. '
        'DWD station observations and ERA5-Land reanalysis (0.1°) have different spatial supports. DWD is not assumed to be exact field weather.')
    save(fig,'fig10',{'plotted_data':summary,'paired_values':paired,'density_curves':densities,'original_contrasts':reference[reference.period.eq('later_held')]},caption,layout=False)


def weather_agreement():
    f=pd.read_csv(OUT/'weather/analysis/unique_site_daily_weather.csv.gz');rows=[]
    fig,axes=plt.subplots(1,2,figsize=(19/2.54,10/2.54));images=[]
    counts=[]
    for ax,variable,mark in zip(axes,['mean','dtr'],['(a)','(b)']):
        x=f['dwd_'+variable];y=f['era5_'+variable];lo=float(np.floor(min(x.min(),y.min())));hi=float(np.ceil(max(x.max(),y.max())))
        edges=np.arange(lo,hi+1,1.);h,xx,yy=np.histogram2d(x,y,bins=[edges,edges]);counts.append(h)
        images.append((ax,h,xx,yy));ax.plot([lo,hi],[lo,hi],'--',color='.3',linewidth=.8)
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):rows.append(dict(panel=mark,variable=variable,dwd_c=(xx[i]+xx[i+1])/2,era5_c=(yy[j]+yy[j+1])/2,site_days=int(h[i,j])))
        name='daily mean' if variable=='mean' else 'diurnal range'
        ax.set_xlabel(f'DWD {name} (°C)');ax.set_ylabel(f'ERA5 {name} (°C)');panel(ax,mark)
    maximum=max(float(h.max()) for h in counts)
    for ax,h,x,y in images:im=ax.pcolormesh(x,y,np.ma.masked_where(h.T==0,h.T),cmap='viridis',norm=LogNorm(1,maximum),rasterized=True)
    fig.subplots_adjust(left=.1,right=.86,bottom=.18,top=.93,wspace=.4)
    cax=fig.add_axes([.9,.22,.025,.65]);bar=fig.colorbar(im,cax=cax);bar.set_label('Unique site-days')
    ticks=[10**k for k in range(int(np.floor(np.log10(maximum)))+1)]
    bar.set_ticks(ticks,labels=[f'{v:,}' for v in ticks])
    caption=('Fig. 9. Comparison of station-observed DWD weather with model-based gridded ERA5-Land reanalysis weather (0.1°). '
        '(a) Daily (Tmin+Tmax)/2. (b) Diurnal temperature range (Tmax−Tmin). Colours show counts in 1 °C bins; dashed lines denote equality. '
        'A native weather-cell/calendar-day pair is counted once despite overlapping sowing-origin windows. Different cells can share a DWD station, so cells are not independent meteorological replicates. '
        'DWD extremes are quality-controlled, explicitly 00–24 UTC and geographically qualified; ERA5-Land uses its native UTC grid aggregates. '
        'Temperature differences include station/grid representativeness. Native day-level comparison data and exact binned counts are retained separately.')
    save(fig,'fig9',{'plotted_data':pd.DataFrame(rows),'agreement':pd.read_csv(OUT/'weather/analysis/weather_agreement.csv')},caption,layout=False)


def stage_and_corrections():
    folder=OUT/'insights'
    stage=pd.read_csv(folder/'stage_performance.csv')
    stage=stage[stage.study.eq('prior_pooled_80')&stage.period.eq('later_held')&stage.recipe.eq('random')&stage.model.isin(MODELS)].copy()
    stage['panel']='(a)'
    correction=pd.read_csv(folder/'correction_summary.csv');correction=correction[correction.recipe.eq('random')].copy();correction['panel']='(b)'
    fig=plt.figure(figsize=(19/2.54,18.0/2.54))
    top=fig.add_axes([.14,.57,.72,.38]);gross=fig.add_axes([.14,.14,.43,.27]);net=fig.add_axes([.74,.14,.22,.27])
    a=stage.pivot(index='head',columns='model',values='mae').reindex(index=range(7),columns=MODELS)
    im=top.imshow(a,cmap='viridis',vmin=0,vmax=22,aspect='auto')
    for i in range(7):
        for j in range(6):top.text(j,i,f'{a.iloc[i,j]:.1f}',ha='center',va='center',fontsize=12,color='white' if a.iloc[i,j]<11 else '#111111')
    top.set_yticks(range(7),[10,31,53,61,75,83,87]);top.set_ylabel('Native BBCH endpoint')
    top.set_xticks(range(6),['T','Nonlinear\nT','RF','MLP','Thermal\nTransformer','Support\nblend']);panel(top,'(a)')
    position=top.get_position();cax=fig.add_axes([.90,position.y0,.025,position.height]);bar=fig.colorbar(im,cax=cax);bar.set_label('Penalised MAE (days)')
    names=['residual_RF','residual_MLP','thermal_transformer','support_blend']
    for i,model in enumerate(names):
        r=correction[correction.model.eq(model)].iloc[0]
        gross.barh(i,r.benefit,height=.58,color=COLORS[model],edgecolor=COLORS[model])
        gross.barh(i,-r.harm,height=.58,color=COLORS[model],edgecolor='#333333',alpha=.45,hatch='///',linewidth=.5)
        gross.errorbar(r.benefit,i,xerr=[[r.benefit-r.benefit_ci_low],[r.benefit_ci_high-r.benefit]],fmt='none',color='#222222',capsize=3)
        gross.errorbar(-r.harm,i,xerr=[[r.harm_ci_high-r.harm],[r.harm-r.harm_ci_low]],fmt='none',color='#222222',capsize=3)
        net.errorbar(r.net_gain,i,xerr=[[r.net_gain-r.net_gain_ci_low],[r.net_gain_ci_high-r.net_gain]],
                     fmt='o',color=COLORS[model],capsize=3,markersize=5)
    gross.set_xlim(-2.25,2.25);gross.set_xticks([-2,-1,0,1,2]);gross.set_xlabel('Gross effect (days)')
    gross.set_yticks(range(4),['RF','MLP','Thermal\nTransformer','Support\nblend'])
    net.set_xlim(-.30,.30);net.set_xticks([-.2,0,.2],['−0.2','0','0.2']);net.set_xlabel('Net gain (days)');net.set_yticks(range(4),['']*4)
    for ax,label in [(gross,'(b)'),(net,'(c)')]:
        ax.set_ylim(3.65,-.65);ax.axvline(0,color='.4',linewidth=.8,linestyle='--');ax.grid(axis='x',color='.92',linewidth=.6);ax.set_axisbelow(True);panel(ax,label)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor='#557C91',label='Error reduction'),Patch(facecolor='#B2C4CD',edgecolor='#333333',hatch='///',label='Error increase')],
               loc='lower center',bbox_to_anchor=(.52,.01),ncol=2,frameon=False)
    caption=('Fig. 3. Native-stage performance and the opposing effects of learned predictions. '
        '(a) Penalised errors for seven source-native endpoints under the earlier 80-site random-source design. '
        '(b) Gross error reduction (positive, solid) and gross error increase (negative, hatched) relative to nonlinear T on identical individual observations. '
        '(c) Net error reduction, equal to gross benefit minus gross harm, with a separate signed difference scale; positive values favour the candidate. '
        'Whiskers in (b) and (c) are paired 95% target-block intervals after averaging source draws. '
        'The models in (b) and (c) are compared with nonlinear T; RF/MLP corrections themselves are trained from T teachers. '
        'All panels use identical 2011–2025 targets, five held spatial folds, eight source years/site and three source draws. '
        'Observations are averaged within site-years, years within sites and sites within heads; (b) and (c) weight seven heads equally. '
        'Invalid dates retain a 366-day loss. Native-stage definitions remain separate, including BBCH 87. Table 2 supplies overall accuracy; (a) shows its stage distribution.')
    net_data=correction.copy();net_data['panel']='(c)'
    save(fig,'fig3',{'plotted_data':pd.concat([stage,correction,net_data],ignore_index=True),'stage_errors':stage,'benefit_harm':correction,'net_gain':net_data},caption,layout=False)


def local_calibration():
    from .violin import local_gain_distribution
    from matplotlib.lines import Line2D
    folder=OUT/'insights';f=pd.read_csv(folder/'local_summary.csv')
    names=['T','nonlinear_T','residual_RF','residual_MLP','thermal_transformer'];f=f[f.model.isin(names)].copy();f['panel']='(a)'
    contrasts=pd.read_csv(folder/'local_contrasts.csv')
    paired,summary,densities=local_gain_distribution(pd.read_csv(folder/'local_sites.csv.gz'),contrasts,names)
    fig,axes=plt.subplots(2,1,figsize=(19/2.54,16.5/2.54),gridspec_kw={'height_ratios':[1,1.12]})
    fig.subplots_adjust(left=.20,right=.98,bottom=.10,top=.87,hspace=.58)
    ax=axes[0]
    for model in names:
        q=f[f.model.eq(model)].sort_values('budget')
        ax.plot(q.budget,q.mae,'-o',color=COLORS[model],label=LABELS[model],markersize=5)
        ax.fill_between(q.budget,q.ci_low,q.ci_high,color=COLORS[model],alpha=.1,linewidth=0)
    ax.set_xticks([0,1,3,5]);ax.set_xlabel('Historical local years (ending by 2010)');ax.set_ylabel('Penalised MAE (days)')
    ax.grid(axis='y',color='.92',linewidth=.6);panel(ax,'(a)')
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.005),ncol=3,frameon=False)
    ax=axes[1]
    for i,model in enumerate(names):
        q=densities[densities.model.eq(model)];r=summary[summary.model.eq(model)].iloc[0]
        ax.fill_between(q.gain_days,i-q.displayed_halfwidth,i+q.displayed_halfwidth,
            color=COLORS[model],alpha=.36,edgecolor=COLORS[model],linewidth=.75)
        ax.errorbar(r.mean_gain_days,i-.10,
            xerr=[[r.mean_gain_days-r.ci_low_days],[r.ci_high_days-r.mean_gain_days]],
            fmt='o',color='#222222',markersize=4,capsize=3,linewidth=1.1,zorder=4)
        ax.plot(r.weighted_median_days,i+.10,'D',markerfacecolor='white',markeredgecolor='#222222',markersize=4,zorder=4)
    ax.set_yticks(range(len(names)),['T','Nonlinear T','RF','MLP','Thermal\nTransformer'])
    ax.set_ylim(len(names)-.45,-.55);ax.set_xscale('asinh',linear_width=2)
    ax.set_xlim(-40,70);ax.set_xticks([-30,-10,-3,0,3,10,30,60],['−30','−10','−3','0','3','10','30','60'])
    ax.set_xlabel('MAE reduction with 5 local years (days; asinh scale)')
    ax.axvline(0,color='.45',linestyle='--',linewidth=.8,zorder=0);ax.grid(axis='x',color='.93',linewidth=.6);ax.set_axisbelow(True);panel(ax,'(b)')
    ax.legend(handles=[Line2D([],[],color='#222222',marker='o',markersize=4,label='Mean and 95% CI'),
        Line2D([],[],color='none',marker='D',markerfacecolor='white',markeredgecolor='#222222',markersize=4,label='Median')],
        loc='lower right',frameon=False,bbox_to_anchor=(1.,1.02),ncol=2,handlelength=1.4,columnspacing=1.)
    caption=('Fig. 5. Accuracy and distributions of gains from historical local observations. '
        '(a) MAE at 0, 1, 3 and 5 pre-2011 local years; shading gives 95% target-block intervals. '
        '(b) Violin distributions of paired MAE reduction from zero to five historical years; positive values indicate improvement. '
        'Each model uses 5,323 matched site–endpoint pairs at 822 sites. Density weights are equal across seven native endpoints, then across sites within each endpoint. '
        'Gaussian kernels use Scott bandwidth in untransformed days, equal maximum widths and the full observed ranges. '
        'The signed asinh horizontal scale (2-day linear width) expands the centre while retaining all tails; ticks remain in days. '
        'Black circles/whiskers show equal-head mean reductions and 95% target-block intervals; white diamonds show weighted medians. '
        'All models share 2011–2025 targets, source-derived shrinkage and historical-year budgets; missing heads receive no update. '
        'Five spatial folds use 80 coverage-selected source sites, eight years/site and draw 17. Intervals condition on fitted models; densities do not imply independent site–endpoint replicates.')
    save(fig,'fig5',{'plotted_data':pd.concat([f,paired],ignore_index=True),'accuracy_curve':f,
        'paired_gains':paired,'distribution_summary':summary,'density_curves':densities,'paired_contrasts':contrasts},caption,layout=False)


def run():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,'xtick.labelsize':12,'ytick.labelsize':12,
        'legend.fontsize':12,'axes.titlesize':12,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    # Obsolete supplementary weather display moved to main Fig. 6; snapshots preserve the preceding version.
    for path in FIG.glob('figs2*'):
        if path.is_file():path.unlink()
    from .workflow import draw as workflow
    workflow(save)
    stage_and_corrections();local_calibration();matrix_figure(True);matrix_figure(False);curves();weather_agreement();weather_effects()
    from .spatial_figures import run as spatial_maps
    spatial_maps(save)
    (FIG/'captions.json').write_text(json.dumps(CAPTIONS,indent=2)+'\n')


if __name__=='__main__':run()
