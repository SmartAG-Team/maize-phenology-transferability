import numpy as np
import pandas as pd
import pytest


def api():
    from research.annual_temperature_transfer import insights
    return insights


def test_benefit_harm_uses_native_year_site_weights_and_keeps_invalid_dates():
    f=pd.DataFrame(dict(label_id=range(5),head=[0]*5,site=['a','a','a','b','c'],
                        site_year=['a1','a1','a2','b1','c1'],block=['x','x','x','y','z'],
                        observed=[0.]*5,horizon=[100]*5,reference=[10.]*5,prediction=[0.,0.,20.,0.,np.nan]))
    s=api().correction_components(f,'reference','prediction').set_index('site')
    assert s.loc['a','benefit']==pytest.approx(5.)
    assert s.loc['a','harm']==pytest.approx(5.)
    assert s.loc['a','improved_fraction']==pytest.approx(.5)
    assert s.loc['b','net_gain']==pytest.approx(10.)
    assert s.loc['c','harm']==pytest.approx(356.)
    assert np.allclose(s.benefit-s.harm,s.net_gain)


def test_benefit_harm_rejects_duplicate_native_labels():
    f=pd.DataFrame(dict(label_id=['same','same'],head=[0,0],site=['a','a'],site_year=['a1','a1'],
                        block=['x','x'],observed=[0.,0.],horizon=[100,100],reference=[10.,10.],prediction=[0.,0.]))
    with pytest.raises(ValueError,match='Duplicate'):
        api().correction_components(f,'reference','prediction')


def local_example():
    h=pd.DataFrame(dict(label_id=['h1','h2','h3','future'],site=['a']*4,head=[0]*4,year=[2008,2008,2009,2012],
                        observed=[10.,14.,14.,90.],horizon=[100]*4,nonlinear_T=[10.]*4))
    t=pd.DataFrame(dict(label_id=['target'],site=['a'],head=[0],year=[2015],nonlinear_T=[10.]))
    return h,t


def test_local_nonlinear_update_counts_years_and_excludes_future_history():
    h,t=local_example()
    assert api().historical_local_update(h,t,3,np.full(7,2.))[0]==pytest.approx(11.5)


def test_local_nonlinear_zero_budget_exactly_preserves_saved_prior():
    h,t=local_example()
    assert np.array_equal(api().historical_local_update(h,t,0,np.full(7,2.)),t.nonlinear_T)


def test_local_nonlinear_update_rejects_observed_dates_outside_historical_weather_prefix():
    h,t=local_example();h.loc[h.label_id.eq('h3'),'horizon']=14
    with pytest.raises(ValueError,match='prefix'):
        api().historical_local_update(h,t,3,np.full(7,2.))


def test_local_nonlinear_update_requires_later_disjoint_targets():
    h,t=local_example();t['year']=2010
    with pytest.raises(ValueError,match='later'):
        api().historical_local_update(h,t,1,np.full(7,2.))
