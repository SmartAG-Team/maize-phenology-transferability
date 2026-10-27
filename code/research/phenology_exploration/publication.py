"""Source-bound figures and a separate formal article candidate."""
import argparse
import html
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.transforms import Bbox
from .data import ROOT,OUT,PRIOR,STAGES
from .core import sha,valid_prediction
from .analysis import site_errors,block_interval
from .neural import NEURAL_FOLDER

DEST=ROOT/'paper/phenology_exploration_20261023'
FIG=DEST/'figures'
LABELS={'T':'T','calendar':'Calendar','relaxed_T':'Nonlinear T','context_T':'Context T',
        'residual_RF':'RF','residual_MLP':'MLP','thermal_transformer':'Thermal Transformer',
        'context_thermal_transformer':'Context Transformer','conditional_local_T':'Conditional local T',
        'local_thermal_T':'Local thermal T','relaxed_phase_T':'Phase T','climate_blend':'Support blend'}
COLORS={'T':'#666666','calendar':'#999999','relaxed_T':'#E69F00','context_T':'#8C564B',
        'residual_RF':'#009E73','residual_MLP':'#0072B2','thermal_transformer':'#56B4E9',
        'context_thermal_transformer':'#AA4499','conditional_local_T':'#D55E00','local_thermal_T':'#333333',
        'relaxed_phase_T':'#CC79A7','climate_blend':'#882255'}
CAPTIONS={}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,
        'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,'axes.titlesize':12,
        'figure.titlesize':12,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,
        'lines.linewidth':1.6,'savefig.dpi':600})


def panel(ax,label):ax.text(0,1.025,label,transform=ax.transAxes,ha='left',va='bottom',fontsize=12)


def save_figure(fig,stem,tables,caption,layout=True):
    FIG.mkdir(parents=True,exist_ok=True);target=19/2.54;pad=.025
    for _ in range(12):
        if layout:fig.tight_layout(pad=.45)
        fig.canvas.draw();tight=fig.get_tightbbox(fig.canvas.get_renderer())
        delta=target-2*pad-tight.width
        if abs(delta)<.001:break
        fig.set_size_inches(fig.get_figwidth()+delta,fig.get_figheight(),forward=True)
    center=tight.x0+tight.width/2
    bounds=Bbox.from_bounds(center-target/2,tight.y0-pad,target,tight.height+2*pad)
    for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{stem}.{ext}',bbox_inches=bounds,dpi=600)
    with pd.ExcelWriter(FIG/f'{stem}.xlsx',engine='openpyxl') as writer:
        for name,data in tables.items():data.to_excel(writer,sheet_name=name[:31],index=False)
        days={'mae','ci_low','ci_high','difference_vs_T','difference_ci_low','difference_ci_high','gain',
              'reference_mae','source_draw_sd','mae_change_days'}
        variables=[]
        for name,data in tables.items():
            for column in data.columns:
                unit='days' if column in days else 'dimensionless'
                if column in {'n_sites','sites'}:unit='connected site groups'
                elif column in {'count','labels','native_site_head_pairs'}:unit='count'
                elif column=='budget':unit='historical years'
                elif column=='bbch':unit='BBCH code'
                variables.append(dict(sheet=name,variable=column,unit=unit,
                    definition='caption panel' if column=='panel' else column.replace('_',' ')))
        pd.DataFrame(variables).to_excel(writer,sheet_name='variables',index=False)
        pd.DataFrame([dict(field='physical_width',value=19,unit='cm'),dict(field='font_size',value=12,unit='pt'),
            dict(field='raster_resolution',value=600,unit='dpi'),dict(field='error_unit',value='days',unit='day'),
            dict(field='unresolved_penalty',value=366,unit='days')]).to_excel(writer,sheet_name='metadata',index=False)
    (FIG/f'{stem}_caption.txt').write_text(caption+'\n');CAPTIONS[stem]=caption
    plt.close(fig)


def figure1():
    base=pd.read_csv(OUT/'summary.csv');wide=pd.read_csv(OUT/'response_followup/summary.csv')
    models=['T','relaxed_T','residual_RF','residual_MLP'];frames=[]
    for model in models:
        source=wide if model=='relaxed_T' else base
        frames.append(source[source.source_ez.eq(-1)&source.period.eq('later_held')&source.model.eq(model)])
    f=pd.concat(frames,ignore_index=True)
    f['panel']=f.recipe.map({'random':'a','coverage':'b'})
    fig,axes=plt.subplots(1,2,figsize=(19/2.54,10/2.54),sharey=True)
    for ax,recipe,mark in zip(axes,['random','coverage'],['(a)','(b)']):
        for model in models:
            q=f[f.recipe.eq(recipe)&f.model.eq(model)].sort_values('n_sites')
            ax.errorbar(q.n_sites,q.mae,yerr=np.array([q.mae-q.ci_low,q.ci_high-q.mae]),
                marker='o',markersize=5,color=COLORS[model],capsize=3,label=LABELS[model])
        ax.set_xticks([40,80]);ax.set_xlim(34,86);ax.set_xlabel('Source sites');panel(ax,mark)
        ax.grid(axis='y',color='.9',linewidth=.6)
    axes[0].set_ylabel('MAE (days)');axes[0].set_ylim(6.7,max(f.ci_high)+.35)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.01),ncol=4,frameon=False)
    caption=('Fig. 1. Later-period held-site error under matched source budgets. '
        '(a) Random whole-site sampling. (b) Capped thermal-zone-coverage sampling. Each selected site contributes eight distinct source years and one native unit per year. '
        'Points average three source draws over five fixed spatial folds; seven native event heads receive equal weight after averaging observations within site-years and years within sites. '
        'Whiskers are 95% intervals from 2,000 target spatial-block bootstrap resamples conditional on the fitted predictors. '
        'Invalid or unresolved dates receive a 366-day loss. Nonlinear T uses the expanded source-validation response grid. Sampling variability is retained separately in the numerical data.')
    save_figure(fig,'fig1',{'plotted_data':f},caption)


def figure2():
    f=pd.read_csv(OUT/NEURAL_FOLDER/'integrated_stage_summary.csv');f=f[f.period.eq('later_held')]
    average=f.groupby(['model','head']).mae.mean().unstack(0)
    models=['relaxed_T','residual_MLP','context_thermal_transformer'];rows=[]
    fig,axes=plt.subplots(2,1,figsize=(19/2.54,15/2.54),sharex=True)
    for model in models:
        gain=average['T']-average[model]
        axes[0].plot(np.arange(7),gain,'o-',color=COLORS[model],label=LABELS[model])
        rows.extend(dict(panel='a',head=h,bbch=[10,31,53,61,75,83,87][h],model=model,
                         mae=float(average.loc[h,model]),reference_mae=float(average.loc[h,'T']),gain=float(gain.loc[h])) for h in range(7))
    for recipe,marker in [('random','o'),('coverage','s')]:
        q=f[f.recipe.eq(recipe)].pivot(index='head',columns='model',values='mae')
        gain=q.thermal_transformer-q.context_thermal_transformer
        axes[1].plot(np.arange(7),gain,marker=marker,color='#AA4499' if recipe=='coverage' else '#56B4E9',label=recipe.capitalize())
        rows.extend(dict(panel='b',head=h,bbch=[10,31,53,61,75,83,87][h],model=recipe,gain=float(gain.loc[h])) for h in range(7))
    for ax,mark in zip(axes,['(a)','(b)']):
        panel(ax,mark);ax.axhline(0,color='.5',linewidth=.7,linestyle='--');ax.grid(axis='y',color='.92',linewidth=.6)
        ax.set_ylabel('MAE reduction (days)');ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(1.01,1))
    axes[1].set_xticks(range(7),['10','31','53','61','75','83','87']);axes[1].set_xlabel('Native BBCH-aligned endpoint')
    caption=('Fig. 2. Stage-specific sources of accuracy improvement at the 80-site budget. '
        '(a) Reduction in later-period MAE relative to T, averaged equally over random and coverage sampling and three source draws. '
        '(b) Reduction obtained by adding observable local context to the thermal Transformer, shown separately for the two sampling recipes. '
        'Both neural variants have the same 23-input architecture, seeded initial parameters and optimization budget; the five context channels are zero in the ablated variant. '
        'Positive differences indicate improvement. Native endpoints are field emergence (10), stem elongation (31), tasseling (53), male flowering onset (61), milk ripening (75), early dough (83) and the native BBCH-87 endpoint. '
        'BBCH 87 retains its source criterion and is distinct from harvest and a general physiological-R6 label. Curves show stage-specific point estimates.')
    save_figure(fig,'fig2',{'plotted_data':pd.DataFrame(rows),'stage_errors':f},caption)


def response_matrix():
    rows=[];cases=json.loads((OUT/'cases.json').read_text())['cases']
    for case in cases:
        if case['source_ez'] is None:continue
        f=pd.read_csv(OUT/'response_followup'/case['case']/'predictions.csv.gz')
        f=f[f['case'].eq('later_held')]
        for ez,q in f.groupby('ez'):
            a=np.where(valid_prediction(q,q.relaxed_T),abs(q.relaxed_T-q.observed),366.)
            b=np.where(valid_prediction(q,q['T']),abs(q['T']-q.observed),366.)
            s=site_errors(q,a-b);s['target_ez']=int(ez)+1;s['source_ez']=case['source_ez']+1
            rows.append(s)
    f=pd.concat(rows,ignore_index=True);result=[]
    for (s,t),q in f.groupby(['source_ez','target_ez']):
        result.append(dict(source_ez=s,target_ez=t,model='relaxed_T',difference_vs_T=float(q.groupby('head').error.mean().mean()),sites=q.site.nunique()))
    return pd.DataFrame(result)


def figure3():
    f=pd.read_csv(OUT/'zone_matrix_summary.csv');f=f[f.period.eq('later_held')]
    blend=pd.read_csv(OUT/'support_blend/zone_matrix.csv')
    blend=blend[blend.period.eq('later_held')&blend.model.eq('climate_blend')]
    f=pd.concat([f[f.model.isin(['residual_RF','residual_MLP'])],response_matrix(),blend],ignore_index=True)
    f['gain']=-f.difference_vs_T
    f['panel']=f.model.map({'relaxed_T':'a','residual_RF':'b','residual_MLP':'c','climate_blend':'d'})
    limit=max(1,float(np.ceil(np.nanmax(abs(f.gain)))))
    fig,axes=plt.subplots(2,2,figsize=(19/2.54,16/2.54))
    fig.subplots_adjust(left=.12,right=.98,top=.94,bottom=.19,wspace=.38,hspace=.42)
    models=['relaxed_T','residual_RF','residual_MLP','climate_blend'];image=None
    for ax,model,mark in zip(axes.flat,models,['(a)','(b)','(c)','(d)']):
        values=f[f.model.eq(model)].pivot(index='source_ez',columns='target_ez',values='gain').reindex(index=range(1,5),columns=range(1,6))
        image=ax.imshow(values,cmap='PuOr',norm=TwoSlopeNorm(0,-limit,limit),aspect='auto')
        for i in range(4):
            for j in range(5):ax.text(j,i,f'{values.iloc[i,j]:.1f}',ha='center',va='center',fontsize=12,
                color='white' if abs(values.iloc[i,j])>.62*limit else '#222222')
        ax.set_xticks(range(5),['1','2','3','4','5*']);ax.set_yticks(range(4),['1','2','3','4'])
        ax.set_xlabel('Target EZ');ax.set_ylabel('Source EZ');panel(ax,mark)
    cax=fig.add_axes([.19,.065,.68,.025]);bar=fig.colorbar(image,cax=cax,orientation='horizontal')
    bar.set_label('MAE reduction relative to T (days)')
    caption=('Fig. 3. Source-zone dependence of later-period prediction gains. '
        '(a) Nonlinear T. (b) RF residual correction. (c) Feedforward MLP residual correction. (d) Source-climate-support blend of MLP and nonlinear T. '
        'Every source fit uses 40 sites and eight distinct years per site; a fixed source draw is evaluated in the same five spatial folds. '
        'Cells show the reduction in seven-head, site-balanced MAE relative to a T model trained on the same source records; positive values indicate improvement. '
        'Comparisons down a column retain the same target population. Source EZ5 lacks the required 40 independent source sites and has no fabricated row. '
        'The asterisk marks the sparse warm target stratum, with 15 later-period exact-date sites; its estimates describe limited support. '
        'These event-date results apply to the German observational network, not to all European countries represented by summary visits.')
    save_figure(fig,'fig3',{'plotted_data':f},caption,layout=False)


def figure4():
    f=pd.read_csv(OUT/'paired_local_extensions/summary.csv')
    models=['local_thermal_T','conditional_local_T','residual_RF','residual_MLP','context_thermal_transformer']
    fig,ax=plt.subplots(figsize=(19/2.54,10/2.54))
    for model,marker in zip(models,['s','D','^','o','v']):
        q=f[f.model.eq(model)].sort_values('budget')
        ax.errorbar(q.budget,q.mae,yerr=np.array([q.mae-q.ci_low,q.ci_high-q.mae]),color=COLORS[model],
            marker=marker,capsize=3,label=LABELS[model],linestyle='--' if model=='local_thermal_T' else '-')
    ax.set_xticks([0,1,3,5]);ax.set_xlabel('Historical local calibration years');ax.set_ylabel('MAE (days)')
    ax.grid(axis='y',color='.9',linewidth=.6);ax.set_xlim(-.2,5.2)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2,frameon=False)
    caption=('Fig. 4. Value of historical local information on a fixed later-period cohort. '
        'The same held-site groups with at least five historical observation years are evaluated at local budgets of zero, one, three and five most recent years ending by 2010. '
        'Local thermal T updates shared thermal thresholds. Conditional local T updates environment-conditioned thermal demands using source-variance-informed shrinkage of local log-demand residuals. '
        'RF, MLP and Context Transformer use native-stage date corrections with the same source-derived shrinkage rule. '
        'Native rows are averaged within site, head and calibration year before the number of informative head-years determines shrinkage. '
        'Historical calibration predictions are recomputed using weather prefixes ending in 2010. Future observations enter evaluation only. '
        'Points are equal-head, site-balanced all-target errors; whiskers are 95% target spatial-block intervals conditional on fitted source models. '
        'All methods and budgets share the same later-period target identifiers.')
    save_figure(fig,'fig4',{'plotted_data':f[f.model.isin(models)],'all_local_controls':f},caption)


def supplement_figures():
    a=pd.read_csv(OUT/'response_followup/selected_alpha.csv');fig,axes=plt.subplots(1,2,figsize=(19/2.54,10/2.54),sharey=True)
    grid=[1/64,1/32,1/16,1/8,1/4,.5,.75,1.,1.5];rows=[]
    for ax,selected,mark in zip(axes,[a.source_ez.isna(),a.source_ez.notna()],['(a)','(b)']):
        for model,offset,color,label in [('relaxed_T',-.17,'#E69F00','Whole-season'),('relaxed_phase_T',.17,'#CC79A7','Post-flowering')]:
            q=a[selected&a.model.eq(model)];counts=q.alpha.value_counts().reindex(grid,fill_value=0)
            ax.bar(np.arange(9)+offset,counts,width=.32,color=color,label=label)
            rows.extend(dict(panel=mark,model=model,alpha=v,count=int(counts.loc[v])) for v in grid)
        ax.set_xticks([0,2,4,6,8],['1/64','1/16','1/4','0.75','1.5'],rotation=35,ha='right')
        ax.set_xlabel('Source-selected response exponent');panel(ax,mark)
    axes[0].set_ylabel('Fitted source cases');axes[1].legend(frameon=False,loc='upper left')
    caption=('Fig. S1. Source-selected exponents in the expanded temperature-response sensitivity. '
        '(a) Pooled-zone source sampling. (b) Single-zone source sampling. The whole-season exponent modifies the shared thermal-progress function; '
        'the post-flowering exponent acts after model-predicted flowering. Values below one increase low-temperature contributions relative to the linear reference. '
        'Selections use source spatial-block validation and do not identify cultivar-specific physiological response curves.')
    save_figure(fig,'figs1',{'plotted_data':pd.DataFrame(rows),'selection_records':a},caption)
    t=pd.read_csv(OUT/NEURAL_FOLDER/'temporal_shift_summary.csv')
    models=['T','relaxed_T','context_T','residual_RF','residual_MLP','thermal_transformer','context_thermal_transformer']
    fig,ax=plt.subplots(figsize=(19/2.54,11/2.54));average=t.groupby('model')[['mae_change_days','ci_low','ci_high']].mean().reindex(models)
    ax.barh(np.arange(len(models)),average.mae_change_days,color=[COLORS[m] for m in models])
    ax.set_yticks(range(len(models)),[LABELS[m] for m in models]);ax.invert_yaxis();ax.set_xlabel('Error in paired duration change (days)')
    ax.grid(axis='x',color='.9',linewidth=.6);ax.set_axisbelow(True)
    caption=('Fig. S2. Error in reproducing paired historical-to-later changes in sowing-to-stage duration. '
        'Each cohort retains the same connected site and native endpoint, at least five historical years and three later years, and a joint-valid prediction population across all compared methods. '
        'Native observations are averaged within years before period means are contrasted. Bars average the two source-sampling recipes and three source draws. '
        'Changes are observational period contrasts and do not isolate a causal warming effect. The underlying cohort-level changes and separate recipe results accompany the figure.')
    save_figure(fig,'figs2',{'plotted_data':average.reset_index(),'recipe_results':t},caption)


def figures(early=False):
    DEST.mkdir(parents=True,exist_ok=True);style();figure1();figure3()
    if not early:figure2();figure4();supplement_figures()
    (FIG/'captions.json').write_text(json.dumps(CAPTIONS,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--early-figures',action='store_true');args=p.parse_args();figures(args.early_figures)
