"""Publication figures for ordinary MAE and signed bias on valid predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from research.annual_temperature_transfer import figures as legacy, spatial_figures as spatial
from research.full_source_benchmark.figures import build_map_figure, STYLE
from research.annual_temperature_transfer.workflow import draw as workflow
from research.phenology_exploration.data import ROOT

EVIDENCE=ROOT/'data/evidence/mae_bias_revision_20261026'
_BASE_SAVE=legacy.save


def save(fig,stem,tables,caption='',layout=True):
    from matplotlib.text import Text
    for ax in fig.axes:
        if ax.axison:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(.7)
    for item in fig.findobj(Text):
        value=item.get_text().replace('Penalised MAE','MAE').replace('Penalized MAE','MAE')
        value=value.replace('Excess transfer loss','ΔMAE').replace('Excess loss','ΔMAE')
        value=value.replace('Source zone','Training zone').replace('source median','training median')
        value=value.replace('Native BBCH endpoint','BBCH stage').replace('Native endpoint','Stage')
        value=value.replace('native endpoints','stages').replace('site–endpoint','site–stage')
        item.set_text(value)
    caption=(ROOT/'paper/figures'/f'{stem}_caption.txt').read_text().strip()
    _BASE_SAVE(fig,stem,tables,caption,layout)
    meta=pd.DataFrame([
        dict(field='width',value=19,unit='cm'),dict(field='font',value=12,unit='pt'),
        dict(field='raster',value=300,unit='dpi'),dict(field='evaluation',value='Ordinary MAE and signed bias',unit='days'),
        dict(field='failed_predictions',value='Excluded from MAE and bias; separately counted',unit='observations'),
        dict(field='paired_comparisons',value='Common valid observations before aggregation',unit='matched observations'),
        dict(field='hierarchy',value='Observations within site-year; years within site; sites within stage; equal stages',unit=''),
        dict(field='intervals',value='Conditional95% intervals from2000 grouped resamples',unit='days'),
        dict(field='temperature_columns',value='degrees Celsius',unit='degrees Celsius')])
    with pd.ExcelWriter(legacy.FIG/f'{stem}.xlsx',engine='openpyxl',mode='a',if_sheet_exists='replace') as writer:
        meta.to_excel(writer,sheet_name='metadata',index=False)


def stage_figure(folder):
    table=pd.read_csv(folder/'primary/stage_performance.csv').query("period == 'later_held'")
    fig=plt.figure(figsize=(19/2.54,18/2.54))
    for i,(metric,bar_label,y0) in enumerate([('mae','MAE (days)',.57),('bias','Bias (days)',.12)]):
        ax=fig.add_axes([.12,y0,.73,.34]);values=table.pivot(index='head',columns='model',values=metric).reindex(index=range(7),columns=legacy.MODELS)
        bound=float(np.ceil(np.nanmax(abs(values.to_numpy()))))
        if metric=='mae':im=ax.imshow(values,cmap='viridis',vmin=0,vmax=bound,aspect='auto')
        else:im=ax.imshow(values,cmap='RdBu_r',norm=TwoSlopeNorm(0,-bound,bound),aspect='auto')
        for r in range(7):
            for c in range(6):
                value=values.iloc[r,c]
                white=(value<bound*.48) if metric=='mae' else abs(value)>bound*.56
                shown=0.0 if abs(value)<.05 else value
                ax.text(c,r,f'{shown:.1f}',ha='center',va='center',color='white' if white else '#17252C',fontsize=12)
        ax.set_yticks(range(7),[10,31,53,61,75,83,87]);ax.set_ylabel('BBCH stage')
        ax.set_xticks(range(6),['T','Nonlinear\nT','RF','MLP','Thermal\nTransformer','Support\nblend'])
        legacy.panel(ax,'('+chr(97+i)+')')
        color=fig.colorbar(im,cax=fig.add_axes([.89,y0,.025,.34]));color.set_label(bar_label)
    save(fig,'fig3',{'plotted_data':table,'stage_metrics':table},layout=False)


def local_figure(folder):
    table=pd.read_csv(folder/'insights/local_summary.csv')
    if 'period' in table:table=table[table.period.eq('later_held')]
    models=['T','nonlinear_T','residual_RF','residual_MLP','thermal_transformer']
    table=table[table.model.isin(models)].copy()
    assert set(table.model)==set(models), 'Missing local-calibration model data'
    fig,axes=plt.subplots(2,1,figsize=(19/2.54,14.7/2.54),sharex=True)
    fig.subplots_adjust(left=.13,right=.985,top=.81,bottom=.13,hspace=.24)
    markers={'T':'o','nonlinear_T':'s','residual_RF':'P','residual_MLP':'^','thermal_transformer':'D'}
    for ax,metric,mark in zip(axes,['mae','bias'],['(a)','(b)']):
        for model in markers:
            q=table[table.model.eq(model)].sort_values('budget');color=legacy.COLORS[model]
            lo='ci_low' if metric=='mae' else 'bias_ci_low';hi='ci_high' if metric=='mae' else 'bias_ci_high'
            ax.plot(q.budget,q[metric],marker=markers[model],color=color,linewidth=1.2,markersize=4,label=legacy.LABELS[model])
            ax.fill_between(q.budget,q[lo],q[hi],color=color,alpha=.09,linewidth=0)
        ax.set_ylabel('MAE (days)' if metric=='mae' else 'Bias (days)');legacy.panel(ax,mark)
        ax.grid(axis='y',color='.92',linewidth=.6)
        if metric=='bias':ax.axhline(0,color='.5',linewidth=.7,linestyle='--')
    axes[1].set_xticks([0,1,3,5]);axes[1].set_xlabel('Years of local observations')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.985),ncol=3,frameon=False)
    save(fig,'fig5',{'plotted_data':table,'local_metrics':table,'paired_contrasts':pd.read_csv(folder/'insights/local_contrasts.csv')},layout=False)


def distance_figure(folder):
    table=pd.read_csv(folder/'temperature_distance_curves.csv').query("period == 'later_held'")
    fig,axes=plt.subplots(3,1,figsize=(19/2.54,17.5/2.54),sharex=True,gridspec_kw={'height_ratios':[.6,1,1]})
    fig.subplots_adjust(left=.13,right=.985,top=.86,bottom=.11,hspace=.30)
    support=table[table.model.eq('nonlinear_T')].sort_values('signed_mat_distance_c')
    count='error_valid_sites' if 'error_valid_sites' in support else 'error_sites'
    axes[0].plot(support.signed_mat_distance_c,support[count],'o-',color='.4',markersize=4)
    for row in support.to_dict('records'):axes[0].text(row['signed_mat_distance_c'],row[count]*1.18,str(int(row[count])),ha='center',va='bottom',fontsize=12)
    axes[0].set_yscale('log');axes[0].set_ylim(6,1600);axes[0].set_yticks([10,100,1000],['10','100','1000']);axes[0].set_ylabel('Sites');legacy.panel(axes[0],'(a)')
    for ax,metric,mark in zip(axes[1:],['error','bias'],['(b)','(c)']):
        for model in legacy.MODELS:
            q=table[table.model.eq(model)].sort_values('signed_mat_distance_c');color=legacy.COLORS[model]
            ax.plot(q.signed_mat_distance_c,q[metric],'-o',color=color,markersize=4,label=legacy.LABELS[model])
            ax.fill_between(q.signed_mat_distance_c,q[metric+'_ci_low'],q[metric+'_ci_high'],color=color,alpha=.08,linewidth=0)
        ax.set_ylabel('MAE (days)' if metric=='error' else 'Bias (days)');ax.axvline(0,color='.5',linewidth=.7,linestyle='--');ax.grid(axis='y',color='.93',linewidth=.6);legacy.panel(ax,mark)
    axes[2].axhline(0,color='.5',linewidth=.7,linestyle='--');axes[2].set_xlabel('Target minus training median annual temperature (°C)')
    handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.995),ncol=3,frameon=False)
    save(fig,'fig8',{'plotted_data':table,'site_support':support},layout=False)


def weather_figure(folder):
    table=pd.read_csv(folder/'weather/analysis/paired_contrasts.csv').query("period == 'later_held'")
    fig,axes=plt.subplots(2,2,figsize=(19/2.54,18.6/2.54),sharey=True)
    fig.subplots_adjust(left=.19,right=.985,top=.89,bottom=.12,hspace=.40,wspace=.24)
    fixed=[('fixed_era5_swap_dwd',-.16,'#0072B2'),('fixed_dwd_swap_era5',.16,'#D55E00')]
    for index,(ax,kind,metric) in enumerate([(axes[0,0],'fixed','difference'),(axes[0,1],'fixed','bias'),(axes[1,0],'matched','difference'),(axes[1,1],'matched','bias')]):
        for i,model in enumerate(legacy.MODELS):
            cases=fixed if kind=='fixed' else [('matched_dwd_minus_era5',0,legacy.COLORS[model])]
            for comparison,offset,color in cases:
                row=table[table.model.eq(model)&table.comparison.eq(comparison)].iloc[0]
                low=row.ci_low if metric=='difference' else row.bias_ci_low;high=row.ci_high if metric=='difference' else row.bias_ci_high
                y=i+offset;ax.hlines(y,low,high,color=color,linewidth=1.1);ax.vlines([low,high],y-.06,y+.06,color=color,linewidth=1)
                ax.plot(row[metric],y,'o',color=color,markersize=4)
        ax.set_ylim(5.6,-.6);ax.set_yticks(range(6),['T','Nonlinear T','RF','MLP','Thermal\nTransformer','Support blend'])
        ax.set_xlabel('ΔMAE (days)' if metric=='difference' else 'Δbias (days)');ax.axvline(0,color='.5',linewidth=.7,linestyle='--');ax.grid(axis='x',color='.93',linewidth=.6);ax.xaxis.set_major_locator(MaxNLocator(nbins=4));legacy.panel(ax,'('+chr(97+index)+')')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],marker='o',color='#0072B2',label='ERA5 fit → DWD input'),Line2D([],[],marker='o',color='#D55E00',label='DWD fit → ERA5 input')],loc='upper center',bbox_to_anchor=(.58,.995),ncol=2,frameon=False)
    save(fig,'fig10',{'plotted_data':table,'paired_contrasts':table,'paired_site_values':pd.read_csv(folder/'weather/analysis/paired_site_differences.csv.gz')},layout=False)


def render(destination,evidence=EVIDENCE,only=None):
    folder=Path(evidence);receipt=folder/'verification.json'
    if not receipt.is_file() or not json.loads(receipt.read_text()).get('complete'):
        raise FileNotFoundError('Missing complete ordinary-MAE evaluation evidence: '+str(receipt))
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(STYLE):
        legacy.FIG=destination;legacy.OUT=folder;spatial.OUT=folder;spatial.FOLDER=folder/'spatial'
        requested=set(['fig'+str(i) for i in range(1,11)]+['figs1']) if only is None else {only}
        if 'fig1' in requested:workflow(save)
        if 'fig2' in requested:spatial.run(save,only='fig2')
        if 'fig3' in requested:stage_figure(folder)
        if 'fig4' in requested:
            fig,tables=build_map_figure(pd.read_csv(folder/'primary/pooled_map.csv'),pd.read_csv(folder/'primary/pooled_cells.csv'));save(fig,'fig4',tables,layout=False)
        if 'fig5' in requested:local_figure(folder)
        old=legacy.save;legacy.save=save
        try:
            if 'fig6' in requested:legacy.matrix_figure(True)
            if 'figs1' in requested:legacy.matrix_figure(False)
        finally:legacy.save=old
        if 'fig7' in requested:spatial.run(save,only='fig7')
        if 'fig8' in requested:distance_figure(folder)
        if 'fig9' in requested:
            from scripts import render_figures as renderer
            old=renderer.save;renderer.save=save
            try:renderer.weather_agreement()
            finally:renderer.save=old
        if 'fig10' in requested:weather_figure(folder)
    captions={stem:(ROOT/'paper/figures'/f'{stem}_caption.txt').read_text().strip() for stem in ['fig'+str(i) for i in range(1,11)]+['figs1']}
    (destination/'captions.json').write_text(json.dumps(captions,indent=2)+'\n')
    return destination
