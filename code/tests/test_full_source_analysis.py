"""Scientific-contract tests for the all-source held-target evaluation."""
import numpy as np
import pandas as pd
import pytest


def api():
    from research.full_source_benchmark import analysis
    return analysis


def frame():
    return pd.DataFrame(dict(
        label_id=['a0','a1','a2','a3','b0','c0'],
        unit=['u0','u1','u2','u3','u4','u5'], u=range(6),
        head=[0,0,0,0,0,1], site=['a','a','a','a','b','c'],
        site_year=['a|2011','a|2011','a|2011','a|2012','b|2011','c|2011'],
        block=['x','x','x','x','y','x'],year=[2011,2011,2011,2012,2011,2011],
        period=['later_held']*6,observed=[10.]*6,horizon=[100]*6,
        nonlinear_T=[10.]*6,residual_MLP=[10.,10.,10.,18.,16.,30.]))


def test_aggregation_balances_site_years_then_sites_and_native_heads():
    q=api().model_sites(frame(),'residual_MLP')
    # Site a averages (0 + 8)/2=4; head 0 averages sites a=4,b=6 ->5;
    # head 1 has c=20; equal endpoint averaging gives 12.5, not row MAE.
    assert q.set_index(['head','site']).loc[(0,'a'),'error']==pytest.approx(4)
    assert api().summarize(q,replicates=100)['value']==pytest.approx(12.5)
    assert q.set_index(['head','site']).loc[(0,'a'),'n_native_labels']==4
    assert q.set_index(['head','site']).loc[(0,'a'),'n_observed_years']==2


def test_invalid_negative_and_prefix_boundary_predictions_keep_366_loss():
    f=frame();f['residual_MLP']=[np.nan,-.1,100.,np.inf,10.,0.]
    q=api().model_sites(f,'residual_MLP').set_index(['head','site'])
    assert q.loc[(0,'a'),'error']==366.
    assert q.loc[(0,'a'),'completion']==0.
    assert q.loc[(0,'a'),'failure14']==1.
    assert q.loc[(1,'c'),'error']==10.
    assert q.loc[(1,'c'),'completion']==1.


def test_support_check_rejects_dropped_targets_changed_outcomes_and_duplicates():
    f=frame()
    api().validate_targets(f,f.copy())
    with pytest.raises(ValueError,match='support'):
        api().validate_targets(f.iloc[:-1],f)
    altered=f.copy();altered.loc[0,'observed']=11
    with pytest.raises(ValueError,match='observed'):
        api().validate_targets(altered,f)
    with pytest.raises(ValueError,match='Duplicate'):
        api().validate_targets(pd.concat([f,f.iloc[:1]]),f)


def test_correction_components_pair_native_outcomes_before_averaging():
    f=frame();f['nonlinear_T']=[14.,14.,14.,14.,14.,14.]
    q=api().correction_components(f,'nonlinear_T','residual_MLP')
    # a: benefit=(4+0)/2=2, harm=(0+4)/2=2;
    # b: harm=2; c: harm=16. Gross macro benefit=.5, harm=9.
    assert api().summarize(q,'benefit',replicates=100)['value']==pytest.approx(.5)
    assert api().summarize(q,'harm',replicates=100)['value']==pytest.approx(9.)
    assert np.allclose(q.benefit-q.harm,q.net_gain)


def test_conditional_block_resampling_preserves_pairing_and_all_heads():
    f=pd.DataFrame(dict(head=[0,1,0,1],site=['a','a','b','b'],
                        block=['x','x','y','y'],error=[0.,0.,10.,10.]))
    a=api().conditional_samples(f,replicates=1000,seed=17)
    b=api().conditional_samples(f.assign(error=f.error+7),replicates=1000,seed=17)
    assert np.allclose(b-a,7)
    assert set(np.unique(a))=={0.,5.,10.}
    assert np.array_equal(a,api().conditional_samples(f.iloc[::-1],replicates=1000,seed=17))
    # Losing either exclusive block loses a native head; never silently average
    # the remaining head and narrow its interval.
    exclusive=pd.DataFrame(dict(head=[0,1],site=['a','b'],block=['x','y'],error=[0.,10.]))
    result=api().summarize(exclusive,replicates=1000,seed=17)
    assert np.isnan(result['ci_low']) and np.isnan(result['ci_high'])


def test_missing_fold_or_incomplete_receipt_cannot_be_analysed(tmp_path):
    with pytest.raises(ValueError,match='Incomplete'):
        api().load_folds(tmp_path,reference_folder=tmp_path/'archive')


def test_fold_loader_checks_archived_metadata_and_receipt_hashes(tmp_path):
    import hashlib
    import json
    f=frame()
    # One small test fold contains all seven heads and both fixed periods.
    f=pd.concat([f.iloc[[0]].assign(label_id=f'id{h}',unit=f'u{h}',u=h,head=h) for h in range(7)],ignore_index=True)
    f=pd.concat([f, f.assign(label_id=lambda q:q.label_id+'h',unit=lambda q:q.unit+'h',
        year=2009,site_year='a|2009',period='historical_held')],ignore_index=True)
    for model in ['T','nonlinear_T','residual_RF','residual_MLP','thermal_transformer','support_blend']:
        f[model]=12.
    for folder,table in [(tmp_path/'fold0',f),(tmp_path/'archive/cases/f0_random_n80_d17',f.rename(columns={'period':'case'}))]:
        folder.mkdir(parents=True)
        path=folder/'predictions.csv.gz';table.to_csv(path,index=False)
        (folder/'complete.json').write_text(json.dumps({'complete':True,'outputs':{'predictions.csv.gz':hashlib.sha256(path.read_bytes()).hexdigest()}}))
    result,_,_=api().load_folds(tmp_path,tmp_path/'archive',folds=[0])
    assert len(result)==14
    path=tmp_path/'fold0/predictions.csv.gz'
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Changed'):
        api().load_folds(tmp_path,tmp_path/'archive',folds=[0])


def test_map_cells_average_native_sites_and_retain_label_support():
    f=frame(); f['head']=[3,3,3,3,3,6]
    for model in ['T','nonlinear_T','residual_RF','residual_MLP','thermal_transformer','support_blend']:
        if model not in f:f[model]=f.residual_MLP
    geo=pd.DataFrame(dict(canonical_site_id=['a','b','c'],weather_cell=['cell1','cell1','cell2'],
        weather_latitude=[50.,50.,51.],weather_longitude=[10.,10.,11.]))
    # A cell cannot straddle frozen blocks. Sites a and b share one block here.
    f.loc[f.site.eq('b'),'block']='x'
    sites=pd.concat([api().model_sites(f,m).assign(model=m,period='later_held') for m in ['T','nonlinear_T','residual_MLP']])
    tables=api().pooled_map_tables(sites,f,geo)
    cells=tables['pooled_cells.csv']
    q=cells[cells['head'].eq(3)&cells.model.eq('residual_MLP')].iloc[0]
    assert q.error==pytest.approx(5.)
    assert q.n_native_labels==5 and q.n_native_units==5 and q.n_sites==2
    assert set(tables['pooled_map.csv'].panel)=={'(a)','(b)','(c)','(d)','(e)','(f)'}


def test_stage_figure_axes_retain_large_new_gains_and_intervals():
    from research.full_source_benchmark.figures import build_stage_figure
    models=['T','nonlinear_T','residual_RF','residual_MLP','thermal_transformer','support_blend']
    stage=pd.DataFrame([dict(head=h,model=m,mae=float(h+2)) for h in range(7) for m in models])
    correction=pd.DataFrame([dict(model=m,benefit=12.,benefit_ci_low=11.,benefit_ci_high=14.,
        harm=1.,harm_ci_low=.5,harm_ci_high=2.,net_gain=11.,net_gain_ci_low=9.,net_gain_ci_high=13.) for m in models[2:]])
    fig,tables=build_stage_figure(stage,correction)
    _,gross,net=fig.axes[:3]
    assert gross.get_xlim()[0]<-2 and gross.get_xlim()[1]>14
    assert net.get_xlim()[0]<=0 and net.get_xlim()[1]>13
    assert len(tables['stage_errors'])==42
    from matplotlib.text import Text
    assert all(t.get_fontsize()==12 for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip())
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_complete_tables_report_hierarchy_weighted_completion_and_failure():
    original=frame(); pieces=[]
    for head in range(6):
        q=original.iloc[:5].copy().assign(head=head)
        q['label_id']=q.label_id+f'h{head}'
        q['residual_MLP']=[np.nan,np.nan,np.nan,18.,16.]
        pieces.append(q)
    pieces.append(original.iloc[[5]].assign(head=6))
    f=pd.concat(pieces,ignore_index=True)
    for model in ['T','residual_RF','thermal_transformer','support_blend']:
        f[model]=f.nonlinear_T
    tables=api().build_tables(f,replicates=20,seed=17)
    summary=tables['summary.csv'].set_index('model')
    assert summary.loc['residual_MLP','completion']==pytest.approx(11/14)
    assert summary.loc['residual_MLP','failure14']==pytest.approx(5/14)
    assert summary.loc['residual_MLP','mae']==pytest.approx(599/7)
    assert summary.loc['residual_MLP','labels']==31
    assert len(tables['stage_performance.csv'])==42
    contrast=tables['paired_contrasts.csv'].query('left == "residual_MLP" and right == "nonlinear_T" and head == "all"').iloc[0]
    assert contrast.difference==pytest.approx(599/7)


def test_primary_exports_have_native_width_300dpi_and_exact_workbook_data(tmp_path):
    import json
    from PIL import Image
    from research.full_source_benchmark import figures
    from research.phenology_exploration.core import sha
    models=api().MODELS
    stage=pd.DataFrame([dict(study='full_source',recipe='all_eligible',period='later_held',head=h,model=m,
        mae=float(h+2),ci_low=float(h+1),ci_high=float(h+3)) for h in range(7) for m in models])
    correction=pd.DataFrame([dict(study='full_source',recipe='all_eligible',period='later_held',model=m,
        benefit=2.,benefit_ci_low=1.,benefit_ci_high=3.,harm=1.,harm_ci_low=.5,harm_ci_high=2.,
        net_gain=1.,net_gain_ci_low=-.5,net_gain_ci_high=2.) for m in models[2:]])
    pred=frame();pred['head']=[3,3,3,3,3,6];pred.loc[pred.site.eq('b'),'block']='x'
    pred['T']=pred.residual_MLP
    sites=pd.concat([api().model_sites(pred,m).assign(model=m,period='later_held') for m in ['T','nonlinear_T','residual_MLP']])
    geo=pd.DataFrame(dict(canonical_site_id=['a','b','c'],weather_cell=['cell1','cell1','cell2'],
        weather_latitude=[50.,50.,51.],weather_longitude=[10.,10.,11.]))
    data=api().pooled_map_tables(sites,pred,geo)
    data.update({'stage_performance.csv':stage,'correction_summary.csv':correction})
    folder=tmp_path/'analysis';folder.mkdir()
    for name,table in data.items():table.to_csv(folder/name,index=False)
    (folder/'verification.json').write_text(json.dumps({'complete':True,'output_sha256':{name:sha(folder/name) for name in data}}))
    outputs=figures.render_primary_figures(tmp_path/'figures',folder)
    for stem,path in outputs.items():
        with Image.open(path) as image:
            assert image.info['dpi'][0]==pytest.approx(300.,abs=.01)
            assert image.width/image.info['dpi'][0]*2.54==pytest.approx(19.,abs=.02)
        metadata=pd.read_excel(path.with_suffix('.xlsx'),sheet_name='metadata').set_index('field')
        assert float(metadata.loc['font','value'])==12
    plotted=pd.read_excel(outputs['fig3'].with_suffix('.xlsx'),sheet_name='stage_errors')
    pd.testing.assert_frame_equal(plotted[stage.columns],stage,check_dtype=False)
    plotted=pd.read_excel(outputs['fig4'].with_suffix('.xlsx'),sheet_name='native_cell_results')
    pd.testing.assert_frame_equal(plotted,data['pooled_cells.csv'],check_dtype=False)
