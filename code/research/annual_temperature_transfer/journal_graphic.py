"""Evidence-linked graphical abstract with native 12-pt, PNG-only export.

Crop development is schematic. Response curves use source-selected exponents;
arrow widths encode transfer loss rather than event-date advance or delay.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon

ROOT = Path(__file__).resolve().parents[3]
WIDTH_CM, HEIGHT_CM = 19, 7.6
INK, GREY = '#203D4E', '#666666'
ORANGE, BLUE = '#E69F00', '#0072B2'  # Same model colours as the manuscript.
GREEN, LEAF, GOLD = '#386B51', '#6B9568', '#C3A557'
COOL = '#3F648B'
DATA_TABLES = ['conceptual_data', 'transfer_evidence', 'phenology_evidence',
               'primary_contrast', 'response_curves', 'text_layout', 'illustration_geometry']


def label(ax, x, y, text, **kwargs):
    style = dict(fontsize=12, color=INK, ha='center', va='center')
    style.update(kwargs)
    return ax.text(x, y, text, **style)


def arrow(ax, start, end, color=INK, width=1.2, scale=11):
    p = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=scale,
                        linewidth=width, color=color)
    ax.add_patch(p)


def leaf(ax, x, y, dx, dy, color, width=.12):
    vertices = [(x, y), (x + dx*.30, y + dy*.65 + width),
                (x + dx*.83, y + dy + width), (x + dx, y + dy),
                (x + dx*.62, y + dy*.70 - width),
                (x + dx*.18, y + dy*.14 - width), (x, y)]
    ax.add_patch(PathPatch(MplPath(vertices, [MplPath.MOVETO] + [MplPath.CURVE4]*6),
                           facecolor=color, edgecolor='none', zorder=4))
    ax.plot([x, x + dx*.80], [y, y + dy*.80], color=GREEN, linewidth=.35, alpha=.45, zorder=5)


def maize(ax, x, ground, height, stage='flowering'):
    mature = stage == 'mature'
    stalk = GOLD if mature else GREEN
    top = ground + height*.86
    ax.plot([x, x], [ground, top], color=stalk,
            linewidth=1.4 if height > 1 else 1.0, zorder=4)
    if stage == 'emergence':
        for side in [-1, 1]:
            leaf(ax, x, ground + height*.28, side*height*.48, height*.38,
                 LEAF, width=height*.055)
        return
    for i, f in enumerate([.17, .29, .41, .54, .67, .78]):
        direction = -1 if i % 2 == 0 else 1
        tint = GOLD if mature and i < 4 else LEAF
        leaf(ax, x, ground + height*f, direction*height*(.39 - .025*i),
             height*(.13 - .007*i), tint, width=height*.038)
        ax.plot([x-height*.021,x+height*.021],[ground+height*f]*2,
                color=stalk,linewidth=.65,zorder=5)
    for direction in [-1, 0, 1]:
        tip = (x + direction*height*.13, ground + height*(.98 if direction == 0 else .95))
        ax.plot([x, tip[0]], [top, tip[1]], color=GOLD, linewidth=.7, zorder=5)
        for f in [.35, .60, .82]:
            xx, yy = x + (tip[0] - x)*f, top + (tip[1] - top)*f
            ax.plot([xx - height*.024, xx + height*.024],
                    [yy - height*.014, yy + height*.014], color=GOLD, linewidth=.7, zorder=5)
    if stage in ['dough','mature']:
        ax.add_patch(Ellipse((x + height*.065, ground + height*.50), height*.095,
                            height*.27, angle=-17, facecolor=ORANGE if mature else '#DBC56C',
                            edgecolor=GOLD, linewidth=.5, zorder=6))
        leaf(ax, x, ground + height*.38, height*.15, height*.22,
             '#A59C58' if mature else GREEN, width=height*.035)


def thermometer(ax,x,y,warm=False):
    color=ORANGE if warm else COOL
    ax.add_patch(FancyBboxPatch((x-.075,y-.18),.15,.80,
        boxstyle='round,pad=0,rounding_size=.07',facecolor='white',edgecolor=color,linewidth=.8,zorder=4))
    ax.plot([x,x],[y-.18,y+(.43 if warm else .08)],color=color,linewidth=2.1,
            solid_capstyle='round',zorder=5)
    ax.add_patch(Circle((x,y-.24),.15,facecolor=color,edgecolor='white',linewidth=.6,zorder=5))
    for yy in [y+.03,y+.20,y+.37,y+.54]:
        ax.plot([x+.16,x+.25],[yy,yy],color=color,linewidth=.55,zorder=4)


def sun(ax, x, y, radius=.24):
    ax.add_patch(Circle((x, y), radius, facecolor='#F0BD51', edgecolor='none', zorder=2))
    for t in np.linspace(0, 2*np.pi, 8, endpoint=False):
        ax.plot([x + radius*1.35*np.cos(t), x + radius*1.7*np.cos(t)],
                [y + radius*1.35*np.sin(t), y + radius*1.7*np.sin(t)],
                color='#DEA53D', linewidth=.8, zorder=2)


def snowflake(ax, x, y, radius=.28):
    for t in np.linspace(0, np.pi, 3, endpoint=False):
        v, normal = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
        ax.plot([x - radius*v[0], x + radius*v[0]],
                [y - radius*v[1], y + radius*v[1]], color=COOL, linewidth=.8)
        for sign in [-1, 1]:
            base = np.array([x, y]) + sign*radius*.60*v
            for side in [-1, 1]:
                tip = base - sign*radius*.23*v + side*radius*.20*normal
                ax.plot([base[0], tip[0]], [base[1], tip[1]], color=COOL, linewidth=.7)


def climate_field(ax, x, y, warm=False):
    # Landscapes are climate vignettes, not geographical maps.
    sky, soil = ('#FCF3DF', '#CDBB86') if warm else ('#EAF2F8', '#9EBAB1')
    ax.add_patch(FancyBboxPatch((x - 1.58, y), 3.16, 2.12,
        boxstyle='round,pad=0,rounding_size=.30', facecolor=sky, edgecolor='none', zorder=0))
    ground = y + .25
    vertices = [(x - 1.58, ground), (x - .5, ground + .17),
                (x + .72, ground + .14), (x + 1.58, ground),
                (x + 1.58, y + .04), (x - 1.58, y + .04), (x - 1.58, ground)]
    ax.add_patch(PathPatch(MplPath(vertices,
        [MplPath.MOVETO] + [MplPath.CURVE4]*3 + [MplPath.LINETO]*3),
        facecolor=soil, edgecolor='none', zorder=2))
    for dx in [-.95, -.40, .15, .70]:
        ax.plot([x + dx*.72, x + dx], [ground + .02, y + .05],
                color='white', alpha=.55, linewidth=.7, zorder=3)
    # Same stages in both zones: no invented measured stage differences.
    maize(ax, x - .68, ground + .02, 1.20)
    maize(ax, x + .28, ground + .02, 1.38, 'mature')
    thermometer(ax,x+1.04,y+1.36,warm)


def calendar(ax, x, y, color, warning=False):
    ax.add_patch(FancyBboxPatch((x - .36, y - .33), .72, .66,
        boxstyle='round,pad=0,rounding_size=.07', facecolor='white',
        edgecolor=color, linewidth=1.1, zorder=5))
    ax.plot([x - .34, x + .34], [y + .12, y + .12], color=color, linewidth=1.1, zorder=6)
    for dx in [-.18, .18]:
        ax.plot([x + dx, x + dx], [y + .27, y + .40],
                color=color, linewidth=1.2, solid_capstyle='round', zorder=6)
    if warning:
        ax.add_patch(Polygon([(x - .17, y - .22), (x + .17, y - .22), (x, y + .05)],
                             facecolor='#D55E00', edgecolor='none', zorder=6))
        ax.plot([x, x], [y - .06, y - .12], color='white', linewidth=.8, zorder=7)
        ax.add_patch(Circle((x, y - .17), .013, facecolor='white', edgecolor='none', zorder=7))
    else:
        for xx, yy in [(-.17, -.06), (0, -.06), (.17, -.06), (-.17, -.21), (0, -.21), (.17, -.21)]:
            ax.add_patch(Circle((x + xx, y + yy), .025, facecolor=color, edgecolor='none', zorder=6))


def neural_network(ax, x, y):
    layers = [[(x, y - .48), (x, y), (x, y + .48)],
              [(x + .60, y - .66), (x + .60, y - .22), (x + .60, y + .22), (x + .60, y + .66)],
              [(x + 1.20, y - .33), (x + 1.20, y + .33)]]
    for left, right in zip(layers[:-1], layers[1:]):
        for a in left:
            for b in right:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, alpha=.35, linewidth=.65, zorder=2)
    for layer in layers:
        for p in layer:
            ax.add_patch(Circle(p, .092, facecolor='white', edgecolor=BLUE, linewidth=1.0, zorder=3))


def evidence_tables():
    """Load the two distinct experiments and derive only displayed quantities."""
    from research.phenology_exploration.core import thermal_rate, sha
    primary = ROOT / 'data/evidence/mae_bias_revision_20261026/primary'
    matrix_path = ROOT / 'data/evidence/mae_bias_revision_20261026/matrix.csv'
    matrix = pd.read_csv(matrix_path)
    transfer = matrix.loc[matrix.period.eq('later_held') & matrix.model.isin(['nonlinear_T', 'residual_MLP']) &
        (((matrix.source_zone == 0) & (matrix.target_zone == 4)) |
         ((matrix.source_zone == 4) & (matrix.target_zone == 0)))].copy()
    if len(transfer) != 4 or transfer.duplicated(['model','source_zone','target_zone']).any():
        raise ValueError('Incomplete directional transfer evidence')
    transfer['panel'] = '(b)'
    transfer['arrow_tail_width_pt'] = transfer.penalty * 1.4
    transfer['arrow_id'] = 'transfer_'+transfer.model+'_'+transfer.source_zone.astype(str)+'_'+transfer.target_zone.astype(str)
    stage = pd.read_csv(primary / 'stage_performance.csv')
    stage = stage[stage.period.eq('later_held')]
    values = stage.pivot(index='head',columns='model',values='mae').sort_index()
    support = stage[stage.model.eq('T')].set_index('head').sort_index()
    phenology = support[['bbch','stage','labels','sites']].copy()
    for model in ['T','nonlinear_T','residual_MLP']:
        phenology[model+'_mae_days'] = values[model]
    phenology['thermal_gain_days'] = values['T']-values.nonlinear_T
    phenology['thermal_gain_fraction'] = phenology.thermal_gain_days/phenology.thermal_gain_days.sum()
    phenology['late_endpoint'] = phenology.bbch.isin([83,87])
    phenology['panel'] = '(a)'
    phenology = phenology.reset_index()
    correction = pd.read_csv(primary / 'correction_summary.csv')
    correction = correction[correction.period.eq('later_held') & correction.model.eq('residual_MLP')].copy()
    if len(correction)!=1 or len(phenology)!=7:
        raise ValueError('Incomplete full-source phenology evidence')
    correction['panel']='(a)'
    alphas=[]; hashes={str(matrix_path.relative_to(ROOT)):sha(matrix_path)}
    for fold in range(5):
        path=ROOT/'data/results/full_source_benchmark_20261026'/f'fold{fold}'/'complete.json'
        alphas.append(float(json.loads(path.read_text())['selections']['alpha']['alpha']))
        hashes[str(path.relative_to(ROOT))]=sha(path)
    curves=[];temperature=np.linspace(8.,30.,201)
    weather=np.stack([temperature,temperature],axis=-1)[:,None,:]
    for alpha in sorted(set([1.]+alphas)):
        rate=thermal_rate(weather,alpha)[:,0]/22.
        curves.extend(dict(panel='(a)',temperature_c=float(t),normalized_progress=float(r),
            alpha=alpha,model='T' if alpha==1 else 'nonlinear_T',
            selected_source_folds=5 if alpha==1 else alphas.count(alpha),
            basis='Model formulation under constant Tmin = Tmax; not measured physiological rates')
            for t,r in zip(temperature,rate))
    hashes['code/research/phenology_exploration/core.py']=sha(ROOT/'code/research/phenology_exploration/core.py')
    for name in ['stage_performance.csv','correction_summary.csv']:
        path=primary/name;hashes[str(path.relative_to(ROOT))]=sha(path)
    return transfer,phenology,correction,pd.DataFrame(curves),alphas,hashes


def draw(destination, data_destination=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    data_destination = Path(data_destination or ROOT / 'outputs/graphical_abstract_data')
    data_destination.mkdir(parents=True, exist_ok=True)
    transfer,phenology,contrast,curves,alphas,hashes=evidence_tables()
    late_share=phenology.loc[phenology.late_endpoint,'thermal_gain_fraction'].sum()
    additional_gain=float(contrast.iloc[0].net_gain)
    points=transfer.set_index(['model','source_zone','target_zone'])
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':12,
                         'svg.fonttype':'none','pdf.fonttype':42}):
        fig=plt.figure(figsize=(WIDTH_CM/2.54,HEIGHT_CM/2.54),dpi=300)
        ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,WIDTH_CM),ylim=(0,HEIGHT_CM));ax.axis('off')
        label(ax,.40,7.18,'(a)')
        ground=4.78
        ax.plot([.45,7.94],[ground,ground],color='#CBD7CD',linewidth=.8)
        for x,height,stage,bbch in [(1.15,.95,'emergence',10),(3.13,1.78,'flowering',61),
                                    (5.13,1.82,'dough',83),(7.18,1.82,'mature',87)]:
            maize(ax,x,ground,height,stage)
            label(ax,x,4.37,f'BBCH {bbch}' if bbch==10 else str(bbch))
        # Only the two endpoints represented by the reported 78% are bracketed.
        ax.plot([4.43,4.43,7.86,7.86],[6.87,6.99,6.99,6.87],color=ORANGE,linewidth=1.1)
        label(ax,6.04,7.28,f'{late_share:.0%} of thermal gain')
        # Actual model formulations, with the source-selected nonlinear range.
        x0,x1,y0,y1=1.03,3.62,1.87,2.96
        arrow(ax,(x0,y0),(x1+.18,y0),color='#8E9BA3',width=.7,scale=7)
        arrow(ax,(x0,y0),(x0,y1+.13),color='#8E9BA3',width=.7,scale=7)
        xp=lambda t:x0+(t-8)/22*(x1-x0)
        response=curves.pivot(index='temperature_c',columns='alpha',values='normalized_progress')
        lower=response[min(alphas)];upper=response[max(alphas)]
        ax.fill_between(xp(response.index),y0+(y1-y0)*lower,y0+(y1-y0)*upper,
                        color=ORANGE,alpha=.22,linewidth=0)
        for alpha in sorted(set(alphas)):
            ax.plot(xp(response.index),y0+(y1-y0)*response[alpha],color=ORANGE,linewidth=1.35)
        ax.plot(xp(response.index),y0+(y1-y0)*response[1.],color=GREY,linewidth=1.05,linestyle=(0,(4,3)))
        for x,t in [(x0,'8'),(x1,'30')]:label(ax,x,1.57,t)
        label(ax,2.40,1.03,'Temperature (°C)')
        label(ax,.43,2.43,'Progress',rotation=90)
        ax.plot([.93,1.36],[3.79,3.79],color=ORANGE,linewidth=2)
        label(ax,2.66,3.79,'Nonlinear T')
        ax.plot([.93,1.36],[3.32,3.32],color=GREY,linewidth=1.05,linestyle=(0,(4,3)))
        label(ax,2.27,3.32,'Linear T')
        # The residual learner corrects linear-T event dates, not the rate curve.
        label(ax,6.11,3.79,'Residual MLP',color=BLUE)
        label(ax,4.67,2.58,'T +',color=GREY)
        neural_network(ax,5.34,2.61)
        arrow(ax,(6.78,2.61),(7.12,2.61),color=BLUE,width=.8,scale=7)
        calendar(ax,7.58,2.61,BLUE)
        label(ax,6.11,1.48,f'{additional_gain:.2f} d gain',color=BLUE)
        label(ax,6.11,1.00,'vs nonlinear T')
        # Climate-domain transfer: equal horizontal distance, quantitative widths.
        ax.plot([8.36,8.36],[.82,7.37],color='#DCE3E6',linewidth=.7)
        label(ax,8.88,7.18,'(b)')
        left,right=10.76,17.10
        label(ax,left,7.16,'Cold zone');label(ax,right,7.16,'Warm zone')
        climate_field(ax,left,4.77);climate_field(ax,right,4.77,warm=True)
        label(ax,13.98,4.42,'Excess MAE (d)')
        for x in [left,right]:
            ax.plot([x,x],[.91,4.70],color='#DCE3E6',linewidth=.65,linestyle=(0,(2,3)),zorder=0)
        for y,model,source,target,color in [(3.58,'nonlinear_T',0,4,ORANGE),
            (2.85,'residual_MLP',0,4,BLUE),(2.06,'nonlinear_T',4,0,ORANGE),
            (1.20,'residual_MLP',4,0,BLUE)]:
            row=points.loc[(model,source,target)]
            tail=float(row.arrow_tail_width_pt)/10
            start,end=(left,right) if source==0 else (right,left)
            patch=FancyArrowPatch((start,y),(end,y),
                arrowstyle=f'simple,tail_width={tail},head_width={max(.62,tail*1.65)},head_length={max(.58,tail*1.4)}',
                mutation_scale=10,linewidth=0,facecolor=color,edgecolor='none',zorder=3,
                shrinkA=0,shrinkB=0)
            patch.set_gid(str(row.arrow_id))
            ax.add_patch(patch)
            label(ax,13.98,y+.31+float(row.arrow_tail_width_pt)/72*2.54/2,f'+{row.penalty:.2f}')
        ax.plot([.20,18.80],[.65,.65],color='#DCE3E6',linewidth=.7)
        label(ax,4.15,.29,'All eligible training data')
        label(ax,13.98,.29,'40 sites × 8 years')
        fig.canvas.draw();renderer=fig.canvas.get_renderer();text_layout=[];boxes=[]
        for text in ax.texts:
            box=text.get_window_extent(renderer)
            if text.get_fontsize()!=12 or not(0<=box.x0 and 0<=box.y0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height):
                raise ValueError('Invalid graphical-abstract text geometry: '+text.get_text())
            for other,otherbox in boxes:
                if box.x0<otherbox.x1 and box.x1>otherbox.x0 and box.y0<otherbox.y1 and box.y1>otherbox.y0:
                    raise ValueError('Overlapping graphical-abstract labels: '+text.get_text()+' / '+other)
            boxes.append((text.get_text(),box))
            text_layout.append(dict(text=text.get_text(),font_pt=text.get_fontsize(),
                x_cm=text.get_position()[0],y_cm=text.get_position()[1],
                x_min_cm=box.x0/fig.dpi*2.54,y_min_cm=box.y0/fig.dpi*2.54,
                x_max_cm=box.x1/fig.dpi*2.54,y_max_cm=box.y1/fig.dpi*2.54))
        fig.savefig(destination/'graphical_abstract.png',dpi=300,facecolor='white')
        for ext in ['pdf','svg']:(destination/f'graphical_abstract.{ext}').unlink(missing_ok=True)
        geometry=[]
        for i,artist in enumerate([*ax.lines,*ax.patches]):
            path=artist.get_path().transformed(artist.get_transform()).transformed(ax.transData.inverted())
            for j,(x,y) in enumerate(path.vertices):
                geometry.append(dict(element=i,primitive=type(artist).__name__,gid=artist.get_gid(),vertex=j,x_cm=float(x),y_cm=float(y),
                    path_code=int(path.codes[j]) if path.codes is not None else None,
                    coordinate_role='Illustration layout; transfer tail widths encode observed excess MAE'))
        concepts=[dict(panel='(a)',label='Late phenology',finding='BBCH 83 and 87 supply 78% of the all-source linear-to-nonlinear thermal gain',
            evidence='data/evidence/mae_bias_revision_20261026/primary/stage_performance.csv',representation='Bracket covers BBCH 83 and 87 only; crop forms are schematic'),
            dict(panel='(a)',label='Learned residual',finding='All-source residual MLP adds a paired 0.50-day gain over nonlinear T',
            evidence='data/evidence/mae_bias_revision_20261026/primary/correction_summary.csv',representation='Network adds signed date corrections to linear T; no date-shift direction is depicted'),
            dict(panel='(b)',label='Directional transfer',finding='Warmest-to-coldest penalties exceed the reverse-direction point estimates',
            evidence='data/evidence/mae_bias_revision_20261026/matrix.csv',representation='Arrow direction is training to evaluation; tail width is proportional to excess MAE')]
        metadata=dict(width_cm=WIDTH_CM,height_cm=HEIGHT_CM,font_pt=12,raster_dpi=300,png_only=True,
            primary_source='All eligible 1981–2010 source observations in five frozen spatial folds',
            primary_target='56,447 observed stage dates at 1,029 held sites, 2011–2025',
            late_thermal_gain_fraction=float(late_share),additional_MLP_gain_days=additional_gain,
            gain_share_definition='Sum of BBCH83 and87 equal-head MAE improvements divided by the sum over all seven observed stages',
            variable_units={'temperature_c':'degrees Celsius','normalized_progress':'dimensionless','penalty':'days','net_gain':'days','thermal_gain_days':'days','thermal_gain_fraction':'fraction','arrow_tail_width_pt':'points','illustration_coordinates':'cm; conceptual positions'},
            selected_source_exponents_by_fold=alphas,response_curve_basis='Existing thermal_rate under constant Tmin=Tmax, divided by22; empirical model formulation, not identified physiology',
            zone_definition='Quintiles of 1981–2010 annual mean temperature in the German network',zone_indices='0=coldest;4=warmest',
            transfer_source_sites=40,transfer_source_years_per_site=8,transfer_source_draws=3,transfer_outer_folds=5,
            penalty_unit='days',penalty_definition='Ordinary MAE minus target-zone-trained MAE on shared valid observations within each contrast',
            transfer_tail_width_pt_per_day=1.4,transfer_width_scale='Proportional with zero intercept; arrowhead size has a visibility minimum',
            uncertainty='Conditional 95% target-block intervals retained in primary_contrast and transfer_evidence; graphic labels are point estimates',
            illustration='Crop forms, landscapes and unscaled thermometers are conceptual; identical crop stages in both zone vignettes',
            rendering='Matplotlib primitives exported only as PNG; no generative image tool',standalone_title=False,
            text_collision_check=True,source_sha256=hashes,data_tables=DATA_TABLES)
        (data_destination/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        tables=dict(conceptual_data=pd.DataFrame(concepts),transfer_evidence=transfer,phenology_evidence=phenology,
            primary_contrast=contrast,response_curves=curves,text_layout=pd.DataFrame(text_layout),illustration_geometry=pd.DataFrame(geometry))
        for name in DATA_TABLES:tables[name].to_csv(data_destination/(name+'.csv'),index=False)
        plt.close(fig)
    caption=('Graphical abstract. (a) Maize development and model structure. BBCH 83 and 87 account for 78% of the linear-to-nonlinear T gain. All-source MLP improves on nonlinear T by 0.50 d (95% CI: 0.37–0.63 d). The orange envelope shows normalized progress at source-selected exponents 0.125–0.25. '
        '(b) Transfer between coldest and warmest zones. Arrows denote training-to-evaluation direction; tail width scales with additional MAE (d) relative to target-zone training. Zone fits use 40 sites × 8 historical years. Targets are paired within contrasts; directions use different target populations. Crop forms and landscapes are schematic. Metrics use shared valid predictions; conditional 95% intervals and failures accompany the plotting data. ')
    (destination/'graphical_abstract_caption.txt').write_text(caption+'\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', nargs='?', type=Path, default=ROOT / 'outputs/paper')
    parser.add_argument('--data-destination', type=Path, default=ROOT / 'outputs/graphical_abstract_data')
    args = parser.parse_args()
    draw(args.destination, args.data_destination)
