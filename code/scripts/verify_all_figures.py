"""Verify every workbook sheet, ordinary-score source frame and native figure."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from figure_export import verify_png

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'data/evidence/mae_bias_revision_20261026'
STEMS=['fig'+str(i) for i in range(1,11)]+['figs1']
LOCAL_MODELS=['T','nonlinear_T','residual_RF','residual_MLP','thermal_transformer']


def same_frame(actual,expected,label,source_columns=False):
    if source_columns:
        assert set(expected.columns)<=set(actual.columns),label+' missing source columns'
        actual=actual[expected.columns]
    pd.testing.assert_frame_equal(actual.reset_index(drop=True).fillna(''),
        expected.reset_index(drop=True).fillna(''),check_dtype=False,check_exact=False,
        atol=1e-10,rtol=1e-10,obj=label)


def source_frames(stem,evidence):
    """Independent frozen numerical inputs for each result display."""
    read=lambda name:pd.read_csv(evidence/name)
    if stem=='fig2':
        return {'qualified_network':read('spatial/study_network.csv')}
    if stem=='fig3':
        data=read('primary/stage_performance.csv').query("period == 'later_held'")
        return {'plotted_data':data,'stage_metrics':data}
    if stem in ['fig4','fig7']:
        prefix='primary/pooled' if stem=='fig4' else 'spatial/transfer'
        data=read(prefix+'_map.csv')
        return {'plotted_data':data,'displayed_data':data[data.in_germany],
            'native_cell_results':read(prefix+'_cells.csv')}
    if stem=='fig5':
        data=read('insights/local_summary.csv')
        if 'period' in data:data=data[data.period.eq('later_held')]
        data=data[data.model.isin(LOCAL_MODELS)]
        return {'plotted_data':data,'local_metrics':data,'paired_contrasts':read('insights/local_contrasts.csv')}
    if stem in ['fig6','figs1']:
        data=read('matrix.csv');data=data[data.period.eq('later_held')&data.source_zone.ge(0)]
        return {'plotted_data':data}
    if stem=='fig8':
        data=read('temperature_distance_curves.csv').query("period == 'later_held'")
        return {'plotted_data':data,'site_support':data[data.model.eq('nonlinear_T')].sort_values('signed_mat_distance_c')}
    if stem=='fig9':
        return {'agreement':read('weather/analysis/weather_agreement.csv')}
    if stem=='fig10':
        data=read('weather/analysis/paired_contrasts.csv').query("period == 'later_held'")
        return {'plotted_data':data,'paired_contrasts':data,
            'paired_site_values':read('weather/analysis/paired_site_differences.csv.gz')}
    return {}


def verify(figure_folder,canonical_folder,evidence):
    figure_folder,canonical_folder,evidence=map(Path,[figure_folder,canonical_folder,evidence])
    receipt=json.loads((evidence/'verification.json').read_text())
    assert receipt.get('complete') and receipt.get('evaluation_only'),'Missing complete ordinary-evaluation evidence'
    records=[];map_geometry=[]
    for stem in STEMS:
        source=canonical_folder/(stem+'.xlsx');new=figure_folder/(stem+'.xlsx')
        with pd.ExcelFile(source) as expected,pd.ExcelFile(new) as actual:
            assert expected.sheet_names==actual.sheet_names,(stem,expected.sheet_names,actual.sheet_names)
            for sheet in expected.sheet_names:
                same_frame(pd.read_excel(actual,sheet),pd.read_excel(expected,sheet),stem+'/'+sheet)
            source_checks=[]
            for sheet,frame in source_frames(stem,evidence).items():
                assert sheet in actual.sheet_names,(stem,sheet)
                same_frame(pd.read_excel(actual,sheet),frame,stem+'/'+sheet+'/ordinary source',True)
                source_checks.append(sheet)
            metadata=pd.read_excel(actual,'metadata').set_index('field')
            assert float(metadata.loc['width','value'])==19
            assert float(metadata.loc['font','value'])==12
            assert float(metadata.loc['raster','value'])==300
            if stem in ['fig2','fig4','fig7']:
                geometry=pd.read_excel(actual,'display_geometry')
                assert len(geometry)==(3 if stem=='fig2' else 6)
                assert geometry.country.eq('Germany').all() and geometry.panel_frame.all()
                assert geometry.projection_epsg.eq(3035).all()
                assert geometry.font_pt.eq(12).all() and geometry.raster_dpi.eq(300).all()
                assert geometry.scale_length_km.eq(200).all()
                map_geometry.append(geometry[['projection_epsg','x_min_m','x_max_m','y_min_m','y_max_m']].to_numpy(float))
                plotted=pd.read_excel(actual,'plotted_data')
                if stem=='fig2':
                    assert geometry.full_analysis_sites.tolist()==[2232,2232,1029]
                    assert geometry.displayed_sites.tolist()==[2204,2204,1016]
                    assert plotted.displayed_in_germany.all()
                else:
                    full=[962]*3+[841]*3 if stem=='fig4' else [841]*6
                    shown=[949]*3+[829]*3 if stem=='fig4' else [829]*6
                    assert geometry.full_analysis_cells.tolist()==full
                    assert geometry.displayed_cells.tolist()==shown
                    displayed=pd.read_excel(actual,'displayed_data')
                    assert displayed.displayed_in_germany.all()
                    assert len(plotted)==sum(full) and len(displayed)==sum(shown)
        verify_png(figure_folder,stem)
        caption=(figure_folder/(stem+'_caption.txt')).read_text().strip()
        assert caption==(canonical_folder/(stem+'_caption.txt')).read_text().strip()
        assert not re.search(r'OpenAI\s+Codex|Codex[- ]assisted|GPT[- ]?6|Native endpoint',caption,re.I)
        records.append(dict(figure=stem,all_sheets_match=True,sheets=expected.sheet_names,
            ordinary_source_sheets=source_checks,width_cm=19,font_pt=12,dpi=300,png_only=True))
    common=map_geometry[0][0]
    for geometry in map_geometry:
        np.testing.assert_allclose(geometry,np.tile(common,(len(geometry),1)),atol=1e-8,rtol=0)
    return dict(complete=True,figures=records,ordinary_mae_and_bias_sources=True,
        common_germany_map_geometry=True,full_map_support_retained=True,
        active_evidence='data/evidence/mae_bias_revision_20261026')


def main():
    result=verify(ROOT/'outputs/figures',ROOT/'paper/figures',EVIDENCE)
    (ROOT/'outputs/all_figures_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
