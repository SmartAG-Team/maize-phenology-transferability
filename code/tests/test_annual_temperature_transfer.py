import numpy as np
import pandas as pd
import pytest


def api():
    from research.annual_temperature_transfer import core
    return core


def test_annual_temperature_uses_equal_complete_years_and_leap_days():
    dates=pd.date_range('1984-01-01','1985-12-31')
    mean=np.where(dates.year==1984,1.,3.)
    out=api().annual_temperature(dates,mean-2,mean+2,[1984,1985])
    assert out['complete'] and out['days']==731 and out['years']==2
    assert out['annual_mean_c']==pytest.approx(2.)


def test_missing_winter_day_cannot_be_replaced_by_seasonal_mean():
    dates=pd.date_range('1984-01-02','1984-12-31')
    out=api().annual_temperature(dates,np.zeros(len(dates)),np.ones(len(dates)),[1984])
    assert not out['complete'] and np.isnan(out['annual_mean_c'])


def test_duplicate_dates_are_rejected_even_with_expected_total_count():
    dates=pd.date_range('1985-01-01','1985-12-31').to_list();dates[-1]=dates[-2]
    with pytest.raises(ValueError,match='Duplicate'):
        api().annual_temperature(dates,np.zeros(365),np.ones(365),[1985])


def test_inverted_extreme_disqualifies_reference():
    dates=pd.date_range('1985-01-01','1985-12-31');low=np.zeros(365);low[0]=2.
    out=api().annual_temperature(dates,low,np.ones(365),[1985])
    assert not out['complete'] and out['invalid_days']==1


def test_temperature_zones_preserve_ties_and_order():
    x=np.repeat(np.arange(10.),3)
    zones,edges=api().temperature_zones(x,5)
    assert set(zones)==set(range(5)) and np.all(np.diff(zones)>=0)
    assert all(len(set(zones[x==v]))==1 for v in np.unique(x))
    assert len(edges)==4


def test_insufficient_distinct_temperatures_do_not_fabricate_five_zones():
    with pytest.raises(ValueError,match='distinct'):
        api().temperature_zones(np.ones(30),5)


def test_source_distance_distinguishes_centre_distance_and_extrapolation():
    out=api().temperature_distance(np.array([8.,9.,10.]),np.array([7.,9.5,12.]))
    assert np.allclose(out['signed'],[-2.,.5,3.])
    assert np.allclose(out['outside'],[1.,0.,2.])


def test_transfer_penalty_pairs_target_sites_and_native_heads():
    f=pd.DataFrame(dict(model=['M']*4,draw=[17]*4,fold=[0]*4,source_zone=[0,1,0,1],
        target_zone=[1]*4,site=['a','a','b','b'],head=[0,0,0,0],error=[10.,3.,4.,6.]))
    q=api().transfer_penalties(f)
    assert q.loc[q.source_zone.eq(1),'penalty'].eq(0).all()
    assert q.loc[q.source_zone.eq(0),'penalty'].to_list()==[7.,-2.]


def test_missing_diagonal_remains_undefined_without_dropping_target():
    f=pd.DataFrame(dict(model=['M'],draw=[17],fold=[0],source_zone=[0],target_zone=[1],site=['a'],head=[0],error=[10.]))
    q=api().transfer_penalties(f)
    assert len(q)==1 and q.penalty.isna().all() and not q.diagonal_available.any()


def test_historical_and_later_transfer_diagonals_cannot_be_mixed():
    from research.annual_temperature_transfer.analysis import paired_period_penalties
    f=pd.DataFrame(dict(model=['M']*4,draw=[17]*4,fold=[0]*4,source_zone=[0,1,0,1],target_zone=[1]*4,
        site=['a']*4,head=[0]*4,period=['historical_held']*2+['later_held']*2,error=[10.,3.,4.,6.]))
    q=paired_period_penalties(f)
    assert q.loc[q.source_zone.eq(0),'penalty'].to_list()==[7.,-2.]


def test_dwd_utc_qualification_requires_both_extremes_for_each_day():
    from research.annual_temperature_transfer.weather import utc_flags
    params=pd.DataFrame(dict(Parameter=['TNK','TXK','TNK','TXK'],Von_Datum=['20000101','20000101','20070102','20070103'],
        Bis_Datum=['20070101','20070102','',''],**{'Zusatz-Info':['21:30 VT. - 21:30 MEZ']*2+['00:00 - 24:00 UTC gemessen']*2}))
    dates=pd.date_range('2007-01-01',periods=4)
    assert utc_flags(params,dates).tolist()==[False,False,True,True]


def daily_fixture():
    dates=pd.date_range('2010-04-01',periods=4)
    return pd.DataFrame(dict(TNK=[5.]*4,TXK=[15.]*4,QN_4=[10]*4,utc_extremes=[True]*4,
        latitude=[50.]*4,longitude=[10.]*4),index=dates)


def test_dwd_window_preserves_actual_temperatures_without_gdd_clipping():
    from research.annual_temperature_transfer.weather import weather_window
    f=daily_fixture();out=weather_window(f,f.index,50.,10.)
    assert out['complete'] and np.allclose(out['temperature'],[[5.,15.]]*4)


def test_dwd_missing_day_and_native_evening_window_are_not_imputed():
    from research.annual_temperature_transfer.weather import weather_window
    f=daily_fixture();assert not weather_window(f.iloc[:-1],f.index,50.,10.)['complete']
    f.loc[f.index[1],'utc_extremes']=False
    assert not weather_window(f,f.index,50.,10.)['complete']


def test_dwd_sentinel_or_unfinished_quality_control_disqualifies_window():
    from research.annual_temperature_transfer.weather import weather_window
    f=daily_fixture();f.iloc[0,f.columns.get_loc('TNK')]=-999
    assert not weather_window(f,f.index,50.,10.)['complete']
    f=daily_fixture();f.loc[f.index[0],'QN_4']=3
    assert not weather_window(f,f.index,50.,10.)['complete']


def test_dwd_time_valid_station_distance_is_checked_after_relocation():
    from research.annual_temperature_transfer.weather import weather_window
    f=daily_fixture();f.loc[f.index[-1],'latitude']=50.2
    out=weather_window(f,f.index,50.,10.)
    assert not out['complete'] and out['max_distance_km']>20


def test_weather_teacher_groups_keep_shared_station_and_native_block_whole():
    from research.annual_temperature_transfer.weather_design import grouped_source
    f=pd.DataFrame(dict(unit=['u1','u2','u3','u4','u5'],site=['a','b','c','d','e'],
        block=['A','B','B','C','D'],station=[1,1,2,3,4],inner=[0,1,1,2,2]))
    q=grouped_source(f)
    assert q.loc[:2,'block'].nunique()==1 and q.loc[:2,'inner'].nunique()==1
    assert q.native_block.to_list()==f.block.to_list()
    again=grouped_source(f.sample(frac=1,random_state=8)).set_index('unit').sort_index()
    assert q.set_index('unit').sort_index()[['block','inner']].equals(again[['block','inner']])


def test_weather_intervals_resample_shared_station_inputs_together():
    from research.annual_temperature_transfer.weather_analysis import weather_interval
    sites=[f's{i}' for i in range(10)]
    frame=pd.DataFrame(dict(head=[0]*10,site=sites,block=[f'b{i}' for i in range(10)],error=[-1.]*5+[1.]*5))
    stations=pd.DataFrame(dict(site=sites,station=[1]*5+[2]*5))
    out=weather_interval(frame,stations)
    assert out['target_clusters']==2 and out['native_blocks']==10
    assert out['ci_low']==pytest.approx(-1.) and out['ci_high']==pytest.approx(1.)
    assert out['native_block_ci_low']>-1 and out['native_block_ci_high']<1
