"""Primary all-source figures using the shared native-size export policy."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd

from research.phenology_exploration.core import sha
from research.phenology_exploration.data import ROOT
from research.annual_temperature_transfer import figures as common
from research.annual_temperature_transfer.spatial_figures import geometry, germany_view, base, shared_scale, PROJECTION
from .analysis import DEFAULT_INPUT, MODELS, CORRECTIONS

STYLE = {'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.labelsize': 12,
         'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
         'axes.titlesize': 12, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
         'axes.spines.top': False, 'axes.spines.right': False}


def interval(ax, low, high, y, color='#222222'):
    """Draw exact percentile endpoints even if an estimate lies outside them."""
    ax.hlines(y, low, high, color=color, linewidth=1.1, zorder=4)
    ax.vlines([low, high], y - .065, y + .065, color=color, linewidth=1.1, zorder=4)


def build_stage_figure(stage, correction):
    """Stage heatmap and paired gross/net changes, with data-driven limits."""
    stage, correction = stage.copy(), correction.copy()
    if stage.duplicated(['head', 'model']).any() or correction.model.duplicated().any():
        raise ValueError('Duplicate native-stage/correction figure data')
    errors = stage.pivot(index='head', columns='model', values='mae').reindex(index=range(7), columns=MODELS)
    if errors.isna().any().any() or set(correction.model) != set(CORRECTIONS):
        raise ValueError('Incomplete primary figure model/native-head support')
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(19 / 2.54, 18.0 / 2.54))
        top = fig.add_axes([.14, .57, .72, .38])
        gross = fig.add_axes([.14, .14, .43, .27])
        net = fig.add_axes([.74, .14, .22, .27])
        maximum = max(22., float(np.ceil(errors.to_numpy().max())))
        im = top.imshow(errors, cmap='viridis', vmin=0, vmax=maximum, aspect='auto')
        for i in range(7):
            for j in range(6):
                value = errors.iloc[i, j]
                top.text(j, i, f'{value:.1f}', ha='center', va='center', fontsize=12,
                         color='white' if value < maximum * .5 else '#111111')
        top.set_yticks(range(7), [10, 31, 53, 61, 75, 83, 87])
        top.set_ylabel('Native BBCH endpoint')
        top.set_xticks(range(6), ['T', 'Nonlinear\nT', 'RF', 'MLP', 'Thermal\nTransformer', 'Support\nblend'])
        common.panel(top, '(a)')
        position = top.get_position()
        bar = fig.colorbar(im, cax=fig.add_axes([.90, position.y0, .025, position.height]))
        bar.set_label('Penalised MAE (days)')
        for i, model in enumerate(CORRECTIONS):
            row = correction[correction.model.eq(model)].iloc[0]
            color = common.COLORS[model]
            gross.barh(i, row.benefit, height=.58, color=color, edgecolor=color)
            gross.barh(i, -row.harm, height=.58, color=color, edgecolor='#333333', alpha=.45, hatch='///', linewidth=.5)
            interval(gross, row.benefit_ci_low, row.benefit_ci_high, i)
            interval(gross, -row.harm_ci_high, -row.harm_ci_low, i)
            interval(net, row.net_gain_ci_low, row.net_gain_ci_high, i, color)
            net.plot(row.net_gain, i, 'o', color=color, markersize=5, zorder=5)
        gross_extent = max(float(correction.benefit_ci_high.max()), float(correction.harm_ci_high.max()),
                           float(correction.benefit.max()), float(correction.harm.max()), .1)
        gross.set_xlim(-gross_extent * 1.12, gross_extent * 1.12)
        gross.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        gross.set_xlabel('Gross effect (days)')
        gross.set_yticks(range(4), ['RF', 'MLP', 'Thermal\nTransformer', 'Support\nblend'])
        low = min(0., float(correction.net_gain_ci_low.min()), float(correction.net_gain.min()))
        high = max(0., float(correction.net_gain_ci_high.max()), float(correction.net_gain.max()))
        span = max(high - low, .1)
        net.set_xlim(low - span * .12, high + span * .12)
        net.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        net.set_xlabel('Net gain (days)')
        net.set_yticks(range(4), [''] * 4)
        for ax, label in [(gross, '(b)'), (net, '(c)')]:
            ax.set_ylim(3.65, -.65)
            ax.axvline(0, color='.4', linewidth=.8, linestyle='--')
            ax.grid(axis='x', color='.92', linewidth=.6)
            ax.set_axisbelow(True)
            common.panel(ax, label)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor='#557C91', label='Error reduction'),
                            Patch(facecolor='#B2C4CD', edgecolor='#333333', hatch='///', label='Error increase')],
                   loc='lower center', bbox_to_anchor=(.52, .01), ncol=2, frameon=False)
    stage['panel'] = '(a)'
    correction['panel'] = '(b)'
    net_data = correction.assign(panel='(c)')
    return fig, {'plotted_data': pd.concat([stage, correction, net_data], ignore_index=True),
                 'stage_errors': stage, 'benefit_harm': correction, 'net_gain': net_data,
                 'display_limits': pd.DataFrame([
                     dict(panel='(a)', minimum=0, maximum=maximum, unit='days'),
                     dict(panel='(b)', minimum=gross.get_xlim()[0], maximum=gross.get_xlim()[1], unit='days'),
                     dict(panel='(c)', minimum=net.get_xlim()[0], maximum=net.get_xlim()[1], unit='days')])}


def build_map_figure(data, cells):
    """Six paired native-cell maps, with one common projection and scale bar."""
    data = data.copy()
    if set(data.panel) != {'(' + c + ')' for c in 'abcdef'}:
        raise ValueError('Incomplete primary pooled-map panels')
    polygons, xlim, ylim = geometry()
    polygons,xlim,ylim,boundary=germany_view(polygons)
    x,y=PROJECTION.transform(data.weather_longitude,data.weather_latitude)
    points=np.c_[x,y]
    data['displayed_in_germany']=np.logical_or.reduce([PlotPath(ring).contains_points(points) for _,ring in polygons])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(19 / 2.54, 19.2 / 2.54))
        fig.subplots_adjust(left=.025, right=.985, top=.97, bottom=.18, wspace=.025, hspace=.10)
        for ax, letter in zip(axes.flat, 'abcdef'):
            base(ax, polygons, xlim, ylim)
            ax.set_facecolor('white')
            for patch in ax.patches:patch.set_facecolor('#F8FAFB')
            for spine in ax.spines.values():
                spine.set_visible(True);spine.set_edgecolor('#7D8891');spine.set_linewidth(.7)
            q = data[data.panel.eq('(' + letter + ')')]
            delta = q.metric.iloc[0] == 'difference'
            q=q[q.displayed_in_germany]
            norm = TwoSlopeNorm(0, -10, 10) if delta else Normalize(0, 30)
            x, y = PROJECTION.transform(q.weather_longitude, q.weather_latitude)
            marks=ax.scatter(x, y, c=q.value, cmap='RdBu_r' if delta else 'viridis', norm=norm,
                       marker='s', s=5.5, edgecolors='#72797D', linewidths=.1, zorder=3)
            marks.set_clip_path(PathPatch(boundary,transform=ax.transData))
            unresolved = q.any_unresolved.to_numpy(bool)
            if unresolved.any():
                crosses=ax.scatter(np.asarray(x)[unresolved], np.asarray(y)[unresolved], marker='x', s=14,
                           color='#101010', linewidth=.6, zorder=4)
                crosses.set_clip_path(PathPatch(boundary,transform=ax.transData))
            ax.text(0, 1.015, '(' + letter + ')', transform=ax.transAxes, ha='left', va='bottom', fontsize=12)
        shared_scale(axes[1, 0], xlim, baseline=.16)
        for position, norm, cmap, label, extend, ticks in [
            ([.08, .075, .38, .018], Normalize(0, 30), 'viridis', 'Penalised MAE (days)', 'max', [0, 10, 20, 30]),
            ([.57, .075, .35, .018], TwoSlopeNorm(0, -10, 10), 'RdBu_r', 'MLP − nonlinear T (days)', 'both', [-10, 0, 10])]:
            bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes(position),
                               orientation='horizontal', extend=extend)
            bar.set_label(label)
            bar.set_ticks(ticks)
    data['projection_epsg'] = 3035
    data['coordinate_basis'] = 'native weather-cell centre'
    data['geographic_scale_bar'] = True
    display = pd.DataFrame([dict(panel=panel,country='Germany', projection_epsg=3035, x_min_m=xlim[0], x_max_m=xlim[1],
        y_min_m=ylim[0], y_max_m=ylim[1], shared_scale_length_km=200,
        scale_length_km=200,full_analysis_cells=len(q),displayed_cells=int(q.displayed_in_germany.sum()),
        panel_frame=True,frame_linewidth_pt=.7,font_pt=12,raster_dpi=300,
        display_mask='Native cell centre inside Germany; markers clipped to outline',
        coordinate_basis='native weather-cell centre', interpolation=False) for panel,q in data.groupby('panel')])
    return fig, {'plotted_data': data,'displayed_data':data[data.displayed_in_germany],
                 'native_cell_results': cells, 'display_geometry': display}


def render_primary_figures(destination, analysis_folder=DEFAULT_INPUT / 'analysis'):
    """Render Fig. 3/4 and exact workbooks with the common save implementation.

    Captions are supplied by the canonical manuscript figure files. Destination is
    the figure directory itself; the shared module's global destination is restored.
    """
    destination, analysis_folder = Path(destination), Path(analysis_folder)
    receipt = json.loads((analysis_folder / 'verification.json').read_text())
    if not receipt.get('complete'):
        raise ValueError('Incomplete full-source analysis receipt')
    required = ['stage_performance.csv', 'correction_summary.csv', 'pooled_map.csv', 'pooled_cells.csv']
    for name in required:
        if receipt['output_sha256'].get(name) != sha(analysis_folder / name):
            raise ValueError(f'Changed primary figure input: {name}')
    stage = pd.read_csv(analysis_folder / 'stage_performance.csv')
    correction = pd.read_csv(analysis_folder / 'correction_summary.csv')
    stage = stage[stage.period.eq('later_held') & stage.study.eq('full_source') & stage.recipe.eq('all_eligible')].copy()
    correction = correction[correction.period.eq('later_held') & correction.study.eq('full_source') & correction.recipe.eq('all_eligible')].copy()
    captions = {stem: (ROOT / 'paper/figures' / f'{stem}_caption.txt').read_text().strip() for stem in ['fig3', 'fig4']}
    previous = common.FIG
    try:
        common.FIG = destination
        fig, tables = build_stage_figure(stage, correction)
        common.save(fig, 'fig3', tables, captions['fig3'], layout=False)
        fig, tables = build_map_figure(pd.read_csv(analysis_folder / 'pooled_map.csv'), pd.read_csv(analysis_folder / 'pooled_cells.csv'))
        common.save(fig, 'fig4', tables, captions['fig4'], layout=False)
    finally:
        common.FIG = previous
    return {stem: destination / f'{stem}.png' for stem in ['fig3', 'fig4']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=DEFAULT_INPUT / 'analysis')
    parser.add_argument('--destination', type=Path, default=ROOT / 'paper/figures')
    args = parser.parse_args()
    render_primary_figures(args.destination, args.analysis)
