"""Study workflow with a shared model/input foundation and parallel comparisons."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.colors import to_rgb

MODEL_COLORS={'T':'#666666','nonlinear_T':'#E69F00','residual_RF':'#009E73','residual_MLP':'#0072B2','thermal_transformer':'#56B4E9','support_blend':'#882255'}
CAPTION=('Fig. 1. Workflow for evaluating phenology transferability. '
    '(a) Pooled model performance. (b) Local updating from historical observations. '
    '(c) Transfer among annual-temperature zones, including the 40-site pooled control. '
    '(d) DWD/ERA5-Land substitutions and product-specific refits. '
    'Fitting, selection, residual teachers and normalization use pre-2011 data. Paired targets are common within comparisons; budgets and cohorts differ between comparisons. '
    'Local updating uses five models and source draw 17. Annual temperature defines zones; growing-season climate defines blend support. '
    'Intervals condition on fitted models. Predictions assume supplied weather. Node/edge data accompany the figure. '
    'Layout code was assisted by OpenAI Codex (GPT-6) and rendered with Matplotlib.')


def draw(save):
    fig=plt.figure(figsize=(19/2.54,(18.5/19*17.25)/2.54))
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,19),ylim=(.2,17.45));ax.axis('off')
    ink='#223B4A';line='#81929D';nodes=[];edges=[];labels=[];containers=[]
    def text(key,x,y,label,ha='left',bold=False,container=None,**kw):
        item=ax.text(x,y,label,ha=ha,va='center',fontsize=12,fontweight='bold' if bold else 'normal',
                     color=ink,linespacing=1.3,**kw)
        labels.append((key,item,container))
        return item
    def box(key,x,y,w,h,fill='#F4F7F8',stroke='#B8C5CD',kind='shared'):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.10',
            facecolor=fill,edgecolor=stroke,linewidth=.8,zorder=2)
        patch.set_gid(key);ax.add_patch(patch);containers.append((key,patch))
        nodes.append(dict(node=key,x_layout=x,y_layout=y,width_layout=w,height_layout=h,
            kind=kind,parent_node='models' if kind=='model formulation' else '',fill=fill,stroke=stroke,coordinate_basis='arbitrary diagram layout coordinates; not measured or geographic',
            display='conceptual study design; no measured performance values'))
        return patch
    def edge(source,target,points,arrow=True,draw=True):
        edges.append(dict(source=source,target=target,path=json.dumps(points),direction='source to target'))
        if not draw:return
        if len(points)>2:ax.plot(*np.array(points[:-1]).T,color=line,linewidth=.85,zorder=1)
        start,end=points[-2:]
        if arrow:ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,
            color=line,linewidth=.85,shrinkA=0,shrinkB=0,zorder=1))
        else:ax.plot([start[0],end[0]],[start[1],end[1]],color=line,linewidth=.85,zorder=1)

    left,right,width=.4,9.9,8.7
    data=box('qualified_data',left,14.4,width,2.85)
    text('data_heading',left+.4,16.73,'Data and provenance',bold=True,container=data)
    text('data_body',left+.4,15.55,'Observed maize event dates\nDaily weather at native cells\nSeven BBCH endpoints',container=data)
    control=box('separation',right,14.4,width,2.85)
    text('control_heading',right+.4,16.73,'Calibration and testing',bold=True,container=control)
    text('control_body',right+.4,15.55,'Fitting data end by 2010\n5 folds · 50-km spatial blocks\n25-km cell-centre buffer',container=control)
    for source,center in [('qualified_data',4.75),('separation',14.25)]:
        edge(source,'shared_inputs',[(center,14.4),(center,14.1),(9.5,14.1)],arrow=False)
    edge('shared_inputs','models',[(9.5,14.1),(9.5,13.9)],arrow=False)

    models=box('models',.4,11.6,18.2,2.30,fill='#FBFCFC')
    text('models_heading',.8,13.50,'Six model formulations',bold=True,container=models)
    names=[('T','T'),('nonlinear_T','Nonlinear\nT'),('residual_RF','Residual\nRF'),
           ('residual_MLP','Residual\nMLP'),('thermal_transformer','Thermal\nTransformer'),('support_blend','Support\nblend')]
    for i,(key,label) in enumerate(names):
        x=.67+i*2.97;color=MODEL_COLORS[key];tint=tuple(.90+.10*v for v in to_rgb(color))
        chip=box(key,x,11.75,2.78,1.24,fill=tint,stroke=color,kind='model formulation')
        text('model_'+key,x+1.39,12.37,label,ha='center',container=chip)
    branches=[
      ('benchmark','(a)','Model performance',left,7.55,'#EEF3F7',
       '80 sites × 8 years · 3 draws\nRandom / coverage sampling\nStage errors; benefit and harm'),
      ('local','(b)','Local information',right,7.55,'#EFF5F1',
       '0, 1, 3 or 5 past local years\n80-site coverage fit · 1 draw\nFive models; shared date offsets'),
      ('transfer','(c)','Environmental transfer',left,3.7,'#F8F3E9',
       '40 sites × 8 years · 3 draws\n5 temperature zones + pooled\nPaired transfer loss and distance'),
      ('weather','(d)','Weather inputs',right,3.7,'#F3EFF7',
       '40 sites × 4 years · 3 draws\nDWD station / ERA5-Land inputs\nWeather swaps and refitting')]
    for key,mark,heading,x,y,fill,body in branches:
        card=box(key,x,y,width,3.1,fill=fill,kind='parallel comparison')
        text(key+'_panel',x+.4,y+2.57,mark,bold=True,container=card)
        text(key+'_heading',x+1.15,y+2.57,heading,bold=True,container=card)
        text(key+'_body',x+.4,y+1.32,body,container=card)
        nodes[-1].update(panel=mark,label=heading,details=body)
        center=x+width/2
        if key in ['benchmark','local']:
            edge('models',key,[(9.5,11.6),(9.5,11.1),(center,11.1),(center,y+3.18)])
        else:
            side=.12 if x==left else 18.88
            edge('models',key,[(9.5,11.6),(9.5,11.1),(side,11.1),(side,7.15),(center,7.15),(center,y+3.18)])
        inside=x+width if x==left else x
        edge(key,'paired_comparisons',[(inside,y+1.55),(9.5,y+1.55),(9.5,3.55)],arrow=False,draw=False)
        ax.plot([inside,9.5],[y+1.55,y+1.55],color=line,linewidth=.85,zorder=1)
    ax.plot([9.5,9.5],[9.10,3.55],color=line,linewidth=.85,zorder=1)
    ax.plot([9.5,9.5],[9.10,5.25],'o',color=line,markersize=2.5,zorder=1)
    edge('paired_comparisons','evaluation',[(9.5,3.55),(9.5,3.08)])
    evaluation=box('evaluation',.4,.4,18.2,2.6,fill='#EAF1F4',stroke='#9EAFBA')
    text('evaluation_heading',.8,2.53,'Paired evaluation under specified conditions',bold=True,container=evaluation)
    text('evaluation_body',.8,1.40,
        '2011–2025 outcomes; weather cohort: 2011–2024\nPenalised MAE; equal endpoint weights · Invalid dates: 366 d\n95% conditional intervals · Spatial / station groups',container=evaluation)

    fig.canvas.draw();renderer=fig.canvas.get_renderer();checks=[]
    for key,item,container in labels:
        bounds=item.get_window_extent(renderer)
        if container is not None:
            frame=container.get_window_extent(renderer)
            fits=(bounds.x0>=frame.x0+2 and bounds.x1<=frame.x1-2 and bounds.y0>=frame.y0+2 and bounds.y1<=frame.y1-2)
            if not fits:print('CARD_OVERFLOW',key,'text',bounds.bounds,'card',frame.bounds)
        else:fits=True
        checks.append(dict(label=key,text=item.get_text(),font_pt=12,font_weight=item.get_fontweight(),horizontal_alignment=item.get_ha(),container_node=container.get_gid() if container is not None else '',within_container=bool(fits),
            x_layout=item.get_position()[0],y_layout=item.get_position()[1]))
    if not all(r['within_container'] for r in checks):raise ValueError('Workflow card/text geometry failed')
    for key,x,y in [('shared_inputs',9.5,14.1),('paired_comparisons',9.5,3.55)]:
        nodes.append(dict(node=key,x_layout=x,y_layout=y,width_layout=0,height_layout=0,kind='connector junction',coordinate_basis='arbitrary diagram layout coordinates; not measured or geographic',display='conceptual connection'))
    node_data=pd.DataFrame(nodes)
    save(fig,'fig1',{'plotted_data':node_data,'nodes':node_data,'edges':pd.DataFrame(edges),
        'labels':pd.DataFrame(checks),'encodings':pd.DataFrame([dict(model=k,color=v) for k,v in MODEL_COLORS.items()])},CAPTION,layout=False)
