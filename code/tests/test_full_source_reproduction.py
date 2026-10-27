"""Entry-point guards and numerical reconstruction comparisons."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'


def module(monkeypatch,name):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec=importlib.util.spec_from_file_location(f'test_import_{name}',SCRIPTS/f'{name}.py')
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_reconstruction_import_does_not_read_or_reconstruct_data(monkeypatch):
    def unexpected_read(*args,**kwargs):
        raise AssertionError('Import attempted data reconstruction')
    monkeypatch.setattr(pd,'read_csv',unexpected_read)
    module(monkeypatch,'reconstruct_metrics')


def test_numerical_csv_comparison_aligns_native_keys_and_detects_changes(monkeypatch,tmp_path):
    script=module(monkeypatch,'reconstruct_metrics')
    actual=tmp_path/'actual';expected=tmp_path/'expected';actual.mkdir();expected.mkdir()
    data=pd.DataFrame(dict(period=['later_held']*2,model=['T','nonlinear_T'],mae=[2.,1.],completion=[.5,1.]))
    data.to_csv(expected/'summary.csv',index=False)
    data.iloc[::-1].to_csv(actual/'summary.csv',index=False)
    result=script.compare_result_csvs(actual,expected,['summary.csv'])
    assert result['files']==1 and result['numerical_columns']==2 and result['numerical_values']==4
    changed=data.copy();changed.loc[0,'mae']=3.;changed.to_csv(actual/'summary.csv',index=False)
    with pytest.raises(AssertionError,match='summary.csv'):
        script.compare_result_csvs(actual,expected,['summary.csv'])
    changed=data.copy();changed.loc[0,'model']='unmatched';changed.to_csv(actual/'summary.csv',index=False)
    with pytest.raises(AssertionError,match='support'):
        script.compare_result_csvs(actual,expected,['summary.csv'])


def test_missing_canonical_evidence_stops_reconstruction_before_outputs(monkeypatch,tmp_path):
    script=module(monkeypatch,'reconstruct_metrics')
    monkeypatch.setattr(script,'E',tmp_path/'missing_evidence')
    monkeypatch.setattr(script,'O',tmp_path/'unwritten_outputs')
    with pytest.raises(FileNotFoundError,match='full.source'):
        script.main()
    assert not script.O.exists()


def test_missing_canonical_primary_data_stops_figure_entrypoint(monkeypatch,tmp_path):
    script=module(monkeypatch,'render_figures')
    monkeypatch.setattr(script,'ROOT',tmp_path)
    with pytest.raises(FileNotFoundError,match='full.source'):
        script.main()
    assert not (tmp_path/'outputs/figures').exists()
