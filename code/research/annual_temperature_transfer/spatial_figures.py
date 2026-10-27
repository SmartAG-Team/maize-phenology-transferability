"""Geographic held-prediction maps at native weather-cell centres."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm,ListedColormap,BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.patches import Polygon,PathPatch
from matplotlib.path import Path as PlotPath
from pyproj import Transformer
import shapefile
from .climate import OUT
from .spatial import FOLDER,BASEMAP

PROJECTION=Transformer.from_crs('EPSG:4326','EPSG:3035',always_xy=True)


def geometry():
    polygons=[];german=[]
    with shapefile.Reader(str(BASEMAP)) as reader:
        for record in reader.iterShapeRecords():
            shape=record.shape;x0,y0,x1,y1=shape.bbox
            if x1<0 or x0>22 or y1<43 or y0>59:continue
            country=record.record.as_dict().get('ADMIN','')
            points=np.asarray(shape.points);parts=list(shape.parts)+[len(points)]
            for start,end in zip(parts[:-1],parts[1:]):
                ring=points[start:end]
                if len(ring)<3:continue
                x,y=PROJECTION.transform(ring[:,0],ring[:,1]);vertices=np.c_[x,y]
                polygons.append((country,vertices))
                if country=='Germany':german.append(vertices)
    if not german:raise ValueError('Germany boundary absent from archived basemap')
    country,xlim,ylim,_=germany_view(polygons)
    return country,xlim,ylim


def base(ax,polygons,xlim,ylim):
    ax.set(xlim=xlim,ylim=ylim,aspect='equal');ax.set_facecolor('#F5F8FA');ax.set_xticks([]);ax.set_yticks([])
    for country,vertices in polygons:
        ax.add_patch(Polygon(vertices,facecolor='#FFFFFF' if country=='Germany' else '#EDF0F1',
                             edgecolor='#78848B' if country=='Germany' else '#B9C2C6',linewidth=.5 if country=='Germany' else .35,zorder=0))
    for spine in ax.spines.values():spine.set_visible(False)


def germany_view(polygons):
    country=[(name,vertices) for name,vertices in polygons if name=='Germany']
    if not country:raise ValueError('Germany boundary absent from archived basemap')
    vertices=np.vstack([ring for _,ring in country])
    low=vertices.min(axis=0)-20000;high=vertices.max(axis=0)+20000
    boundary=PlotPath.make_compound_path(*[PlotPath(ring,closed=True) for _,ring in country])
    return country,(low[0],high[0]),(low[1],high[1]),boundary


def shared_scale(ax,xlim,baseline=.18):
    # Horizontal coordinates retain map units after native-width export resizing.
    transform=blended_transform_factory(ax.transData,ax.figure.transFigure)
    x=xlim[0]+.035*(xlim[1]-xlim[0]);y=baseline;tick=.006
    ax.plot([x,x+200000],[y,y],transform=transform,color='#20272B',linewidth=2,clip_on=False)
    for v in [x,x+100000,x+200000]:
        ax.plot([v,v],[y-tick,y+tick],transform=transform,color='#20272B',linewidth=.7,clip_on=False)
    ax.text(x+100000,y-.033,'200 km',transform=transform,ha='center',va='top',fontsize=12,clip_on=False)


def overview(save,polygons,xlim,ylim):
    polygons,xlim,ylim,boundary=germany_view(polygons)
    data=pd.read_csv(FOLDER/'study_network.csv')
    x,y=PROJECTION.transform(data.weather_longitude,data.weather_latitude)
    points=np.c_[x,y]
    data['displayed_in_germany']=np.logical_or.reduce([PlotPath(ring).contains_points(points) for _,ring in polygons])
    shown=data[data.displayed_in_germany]
    fig,axes=plt.subplots(1,3,figsize=(19/2.54,11/2.54))
    fig.subplots_adjust(left=.025,right=.985,top=.97,bottom=.30,wspace=.025)
    palette=['#3F648B','#729CA7','#A7BFAA','#D6BA79','#B96C42']
    maps=[data.copy(),data.copy(),data[data.later_evaluation].copy()]
    for ax,letter in zip(axes,'abc'):
        base(ax,polygons,xlim,ylim);ax.text(0,1.015,'('+letter+')',transform=ax.transAxes,ha='left',va='bottom',fontsize=12)
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_visible(True);spine.set_edgecolor('#7D8891');spine.set_linewidth(.7)
    def scatter(ax,x,y,**kwargs):
        marks=ax.scatter(x,y,**kwargs)
        marks.set_clip_path(PathPatch(boundary,transform=ax.transData))
    x,y=PROJECTION.transform(shown.weather_longitude,shown.weather_latitude)
    scatter(axes[0],x,y,c=shown.observed_years,norm=Normalize(1,45),cmap='viridis',marker='s',s=4,edgecolors='none',zorder=3)
    norm=BoundaryNorm(np.arange(.5,6),5);cmap=ListedColormap(palette)
    scatter(axes[1],x,y,c=shown.annual_zone,norm=norm,cmap=cmap,marker='s',s=4,edgecolors='none',zorder=3)
    groups=[('local_only','#4D867E','Local only'),('weather_only','#C58B4E','Weather only'),('both','#8D6E9C','Both'),('neither','#BCC3C8','Neither')]
    for key,color,label in groups:
        q=shown[shown.later_evaluation&shown.availability.eq(key)];x,y=PROJECTION.transform(q.weather_longitude,q.weather_latitude)
        scatter(axes[2],x,y,color=color,marker='s',s=5.5,edgecolors='none',zorder=3)
    shared_scale(axes[2],xlim,baseline=.25)
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(1,45),cmap='viridis'),cax=fig.add_axes([.07,.17,.25,.022]),orientation='horizontal')
    bar.set_label('Observed years');bar.set_ticks([1,15,30,45])
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.40,.17,.25,.022]),orientation='horizontal')
    bar.set_label('Annual T zone');bar.set_ticks([1,2,3,4,5])
    handles=[Line2D([],[],marker='s',linestyle='none',color=color,label=label,markersize=5) for _,color,label in groups]
    fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.69,.10),ncol=2,frameon=False,labelspacing=.2,handletextpad=.3,columnspacing=.8)
    tables=[]
    for letter,q in zip('abc',maps):q['panel']='('+letter+')';tables.append(q)
    caption=('Fig. 2. Geographic support of the qualified maize phenology study. '
        '(a) Distinct observed calendar years per native monitoring/weather cell during 1981–2025; repeated stages do not add years. '
        '(b) Five cold-to-warm annual-temperature zones from complete 1981–2010 reference climate. '
        '(c) Availability of the local-history and paired-weather cohorts among the 1,029 later-period evaluation sites. '
        'Local only, weather only, both and neither are disjoint eligibility classes, not model-performance categories. '
        'The qualified network contains 2,232 German monitoring-site groups; the display retains only native cell centres inside Germany; local and weather cohorts contain 822 and 268 later sites, respectively. '
        'Points are recorded native weather-cell centres rather than surveyed field positions. No interpolation fills unobserved areas. '
        'All panels share ETRS89-LAEA Europe (EPSG:3035), extent and one geographic scale bar. Boundaries: Natural Earth 1:50m. '
        'Observed-year counts and ranges, native-label/unit counts and overlapping cohort memberships accompany the plotting data.')
    panels=pd.concat(tables,ignore_index=True)
    geometry_data=pd.DataFrame([dict(panel=panel,country='Germany',projection_epsg=3035,
        x_min_m=xlim[0],x_max_m=xlim[1],y_min_m=ylim[0],y_max_m=ylim[1],scale_length_km=200,
        full_analysis_sites=len(q),displayed_sites=int(q.displayed_in_germany.sum()),panel_frame=True,
        font_pt=12,raster_dpi=300,display_mask='Native cell centre inside Germany; markers clipped to outline')
        for panel,q in panels.groupby('panel')])
    save(fig,'fig2',{'plotted_data':panels[panels.displayed_in_germany],'qualified_network':data,
        'display_geometry':geometry_data},caption,layout=False)


def run(save,only=None):
    polygons,xlim,ylim=geometry()
    if only is None or only=='fig2':overview(save,polygons,xlim,ylim)
    for name,stem in [('pooled','fig4'),('transfer','fig7')]:
        if only is not None and only!=stem:continue
        data=pd.read_csv(FOLDER/f'{name}_map.csv')
        view_polygons,view_xlim,view_ylim=polygons,xlim,ylim
        view_polygons,view_xlim,view_ylim,boundary=germany_view(polygons)
        x,y=PROJECTION.transform(data.weather_longitude,data.weather_latitude)
        points=np.c_[x,y]
        data['displayed_in_germany']=np.logical_or.reduce([PlotPath(ring).contains_points(points) for _,ring in view_polygons])
        fig,axes=plt.subplots(2,3,figsize=(19/2.54,19.2/2.54))
        fig.subplots_adjust(left=.025,right=.985,top=.97,bottom=.16 if name=='transfer' else .155,wspace=.025,hspace=.10)
        for ax,letter in zip(axes.flat,'abcdef'):
            base(ax,view_polygons,view_xlim,view_ylim);q=data[data.panel.eq('('+letter+')')]
            ax.set_facecolor('white')
            for patch in ax.patches:patch.set_facecolor('#F8FAFB')
            for spine in ax.spines.values():
                spine.set_visible(True);spine.set_edgecolor('#7D8891');spine.set_linewidth(.7)
            q=q[q.displayed_in_germany]
            if q.empty:raise ValueError('Empty geographic result panel')
            delta=name=='transfer' or q.metric.iloc[0]=='difference'
            norm=TwoSlopeNorm(0,-30,30) if name=='transfer' else (TwoSlopeNorm(0,-10,10) if delta else Normalize(0,30))
            x,y=PROJECTION.transform(q.weather_longitude,q.weather_latitude)
            marks=ax.scatter(x,y,c=q.value,cmap='RdBu_r' if delta else 'viridis',norm=norm,marker='s',s=5.5,edgecolors='#72797D',linewidths=.1,zorder=3)
            if boundary is not None:marks.set_clip_path(PathPatch(boundary,transform=ax.transData))
            missing=q.any_unresolved.to_numpy(bool)
            if missing.any():
                crosses=ax.scatter(np.asarray(x)[missing],np.asarray(y)[missing],marker='x',s=14,color='#101010',linewidth=.6,zorder=4)
                if boundary is not None:crosses.set_clip_path(PathPatch(boundary,transform=ax.transData))
            ax.text(0,1.015,'('+letter+')',transform=ax.transAxes,ha='left',va='bottom',fontsize=12)
        if name=='pooled':
            shared_scale(axes[1,0],view_xlim,baseline=.16)
            for position,norm,cmap,label,extend,ticks in [([.08,.085,.38,.018],Normalize(0,30),'viridis','Penalised MAE (days)','max',[0,10,20,30]),
                    ([.57,.085,.35,.018],TwoSlopeNorm(0,-10,10),'RdBu_r','MLP − nonlinear T (days)','both',[-10,0,10])]:
                bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes(position),orientation='horizontal',extend=extend)
                bar.set_label(label);bar.set_ticks(ticks)
            caption=('Fig. 4. Geographic distribution of held-site prediction errors. '
                '(a, b) T and nonlinear-T MAE for flowering onset (BBCH 61). (c) MLP-minus-nonlinear-T error at BBCH 61. '
                '(d, e) T and nonlinear-T MAE at BBCH 87. (f) MLP-minus-nonlinear-T error at BBCH 87. '
                'Negative differences favour MLP. Each square is a recorded native weather-cell centre; monitoring-site/head errors sharing a cell receive equal weight after within-site year averaging. '
                'All models share the same native targets and the earlier 80-site random-source design, eight historical years/site and three source draws in five spatial folds. '
                'Endpoints retain their own observed support. No interpolation fills unsampled areas. Black crosses indicate invalid or unresolved predictions contributing a 366-day loss. '
                'Colour saturates above 30 d for MAE and beyond ±10 d for differences; exact values and observed-year/label support accompany the plotting data. '
                'All panels share ETRS89-LAEA Europe (EPSG:3035) and geographic extent. Boundaries: Natural Earth 1:50m. Positions do not represent surveyed crop-field GPS.')
        else:
            shared_scale(axes[1,0],view_xlim,baseline=.085)
            bar=fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(0,-30,30),cmap='RdBu_r'),cax=fig.add_axes([.27,.085,.65,.018]),orientation='horizontal',extend='both')
            bar.set_label('Excess transfer loss (days)');bar.set_ticks([-30,-15,0,15,30])
            caption=('Fig. 7. Geographic distribution of BBCH-87 transfer penalties. '
                '(a–c) Coldest-zone source fits for nonlinear T, MLP and the support blend, respectively. '
                '(d–f) Warmest-zone source fits for the same three specifications. '
                'Each value subtracts that model’s target-zone-trained error from its source-zone-trained error on identical native outcomes, then averages the three source draws and equally weights monitoring sites within a native weather cell. '
                'All source fits use 40 sites and eight historical years/site in five spatial folds; targets are native 2011–2025 BBCH-87 observations. '
                'Positive values indicate excess loss, and negative values favour foreign-zone training. Black crosses identify invalid source or target-reference dates included with 366-day loss. '
                'Colours saturate beyond ±30 d; the numerical files retain exact penalties, completion and native observed-year/label support. '
                'Points are recorded native weather-cell centres; blank areas are not interpolated. All panels share ETRS89-LAEA Europe (EPSG:3035) and geographic extent. '
                'Boundaries: Natural Earth 1:50m. These are descriptive sampled-cell patterns, not tests of significant geographic clusters or causal temperature effects.')
        data['projection_epsg']=3035;data['coordinate_basis']='native weather-cell centre';data['geographic_scale_bar']=True
        tables={'plotted_data':data,'displayed_data':data[data.displayed_in_germany],
            'native_cell_results':pd.read_csv(FOLDER/f'{name}_cells.csv')}
        tables['display_geometry']=pd.DataFrame([dict(panel=panel,country='Germany',projection_epsg=3035,
                x_min_m=view_xlim[0],x_max_m=view_xlim[1],y_min_m=view_ylim[0],y_max_m=view_ylim[1],
                scale_length_km=200,full_analysis_cells=len(q),displayed_cells=int(q.displayed_in_germany.sum()),
                panel_frame=True,frame_linewidth_pt=.7,font_pt=12,raster_dpi=300,
                display_mask='native cell centre within Natural Earth Germany outline; marker footprints clipped to outline')
                for panel,q in data.groupby('panel')])
        save(fig,stem,tables,caption,layout=False)
