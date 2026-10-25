"""Conceptual study workflow; node and edge data accompany the figure."""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch


def draw(save):
    fig,ax=plt.subplots(figsize=(19/2.54,17.4/2.54))
    fig.subplots_adjust(left=.015,right=.985,bottom=.015,top=.985)
    ax.set(xlim=(0,19),ylim=(0,17.4));ax.axis('off')
    nodes=[];edges=[]
    def box(key,x,y,width,height,text,fill,stroke='#87949E'):
        patch=FancyBboxPatch((x,y),width,height,boxstyle='round,pad=0.06,rounding_size=0.12',
                             facecolor=fill,edgecolor=stroke,linewidth=.9)
        ax.add_patch(patch)
        ax.text(x+width/2,y+height/2,text,ha='center',va='center',fontsize=12,linespacing=1.25,color='#192D3C')
        nodes.append(dict(node=key,x_cm=x,y_cm=y,width_cm=width,height_cm=height,label=text,display='conceptual workflow; no measured result'))
    def arrow(source,target,start,end):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,color='#738390',linewidth=.9))
        edges.append(dict(source=source,target=target,start_x_cm=start[0],start_y_cm=start[1],end_x_cm=end[0],end_y_cm=end[1]))
    box('qualified_data',.18,15.6,18.64,1.5,
        'Qualified native maize dates and daily weather\nSeven BBCH endpoints; original definitions and provenance retained','#EDF3F6')
    box('separation',.18,12.9,18.64,1.95,
        'Source labels and fitting-weather prefixes end by 2010\nFive spatial folds: 50-km blocks; 25-km cell-centre buffer\nSource-only model selection, residual teachers and normalization','#E3EDF3')
    arrow('qualified_data','separation',(9.5,15.52),(9.5,14.93))
    box('models',.18,10.35,18.64,1.8,
        'Six core model families\nT · nonlinear T · residual RF · residual MLP\nThermal Transformer · source-support blend','#F3F0E7')
    arrow('separation','models',(9.5,12.82),(9.5,12.23))
    branches=[
        ('benchmark','Pooled benchmark\n80 sites × 8 years\nRandom / coverage\n3 draws × 5 folds\n\nStage errors; gross\nand net corrections','#EAF1F6'),
        ('local','Local calibration\n0/1/3/5 past years\n80-site source fit\n1 draw × 5 folds\n\nFive models; shared\ndate-offset updates','#EDF3EC'),
        ('transfer','Climate transfer\n40 sites × 8 years\n5 zones + pooled\n3 draws × 5 folds\n\nExcess loss; distance\nand source support','#F6F0E3'),
        ('weather','Weather comparison\n40 sites × 4 years\nDWD / ERA5-Land\n3 draws × 5 folds\n\nFixed-input swaps;\nown-product refits','#F1EDF5')]
    centers=[]
    for i,(key,label,fill) in enumerate(branches):
        x=.18+i*4.76;width=4.36;center=x+width/2;centers.append(center)
        box(key,x,4.85,width,4.55,label,fill)
        ax.plot([center,center],[9.48,9.85],color='#738390',linewidth=.9)
        arrow('models',key,(center,9.85),(center,9.48))
        edges[-1]['via']='shared branch connector at y=9.85 cm'
    ax.plot([centers[0],centers[-1]],[9.85,9.85],color='#738390',linewidth=.9)
    ax.plot([9.5,9.5],[10.27,9.85],color='#738390',linewidth=.9)
    box('evaluation',.18,1.35,18.64,2.35,
        'Later held outcomes: 2011–2025; weather cohort: 2011–2024\nIdentical native targets within each paired experiment\nHead-balanced penalised MAE; invalid dates receive 366 d\nConditional 95% intervals from paired spatial/station-block resampling','#E5EEF0')
    for center,(key,_,_) in zip(centers,branches):
        arrow(key,'evaluation',(center,4.77),(center,3.78))
    ax.text(9.5,.52,'Experimental cohorts and source budgets remain distinct.',ha='center',va='center',fontsize=12,color='#344D5B')
    caption=('Fig. 1. Overall study workflow. Qualified native dates and weather enter spatially separated, source-only fitting. '
        'Four branches address pooled performance, local history, annual-temperature transfer and weather sensitivity. '
        'The transfer branch also supplies the 40-site pooled benchmark control. Local calibration uses five date-offset models, the frozen 80-site coverage fit and draw 17. '
        'Weather fits use identical product-specific memberships with four source years/site and preserve shared-station groups. '
        'Paired contrasts retain common outcomes within each experiment; source budgets and target cohorts differ between branches. '
        'Annual reference temperature defines transfer zones; growing-season climate defines blend support. Intervals condition on fitted models. Predictions assume supplied weather. Node and edge data accompany the figure.')
    save(fig,'fig1',{'plotted_data':pd.DataFrame(nodes),'nodes':pd.DataFrame(nodes),'edges':pd.DataFrame(edges)},caption,layout=False)
