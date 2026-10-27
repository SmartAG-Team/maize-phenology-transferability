"""Scientific regression tests for evaluation-only ordinary-error reconstruction."""
import numpy as np
import pandas as pd
import pytest

from research.mae_bias_revision.analysis import (
    model_sites, paired_sites, summarize, correction_components,
    conditional_samples, connected_weather_blocks, verify,
)


def observations():
    return pd.DataFrame(dict(
        label_id=list('abcde'), head=[0] * 5, site=['s'] * 5,
        block=['b'] * 5, site_year=['s1', 's1', 's2', 's2', 's3'],
        year=[1, 1, 2, 2, 3], unit=list('abcde'), observed=[10.] * 5,
        horizon=[30.] * 5, a=[12., np.nan, 8., 30., -1.],
        b=[np.nan, 16., 9., 15., 0.]))


def test_ordinary_errors_keep_failures_out_of_mae_and_bias_but_audit_all():
    q = model_sites(observations(), 'a').iloc[0]
    assert q.error == 2
    assert q.bias == 0
    assert q.n_native_labels == 5
    assert q.n_valid_labels == 2
    assert q.n_failed_labels == 3
    assert np.isclose(q.completion, (0.5 + 0.5 + 0) / 3)


def test_pairwise_common_observations_before_year_and_site_means():
    q = paired_sites(observations(), 'a', 'b').iloc[0]
    # Only observation c is valid for both. Marginal means cannot recover this.
    assert q.error == 1
    assert q.bias == -1
    assert q.left_error == 2
    assert q.right_error == 1
    assert q.n_common_valid_labels == 1
    assert q.n_excluded_labels == 4


def test_benefit_and_harm_are_paired_on_same_observations():
    q = correction_components(observations(), 'a', 'b').iloc[0]
    assert q.benefit == 1
    assert q.harm == 0
    assert q.net_gain == q.benefit - q.harm
    assert q.improved_fraction == 1
    assert q.n_common_valid_labels == 1


def test_empty_stage_support_is_explicitly_undefined():
    q = model_sites(observations().assign(a=np.nan), 'a')
    assert len(q) == 1
    assert np.isnan(q.error.iloc[0])
    r = summarize(q, required_heads=[0], replicates=50)
    assert np.isnan(r['value'])
    assert r['complete_bootstrap_fraction'] == 0
    assert r['valid_sites'] == 0


def test_absent_required_stage_cannot_change_equal_stage_estimand():
    q = model_sites(observations(), 'a')
    r = summarize(q, required_heads=range(7), replicates=50)
    assert np.isnan(r['value'])
    assert np.isnan(r['ci_low'])


def test_years_are_equal_after_averaging_observations_within_year():
    f = observations().iloc[:3].copy()
    f['a'] = [10., 20., 30.]
    f['horizon'] = 40.
    s = model_sites(f, 'a').iloc[0]
    assert s.error == 12.5  # (mean(0, 10) + mean(20))/2
    assert s.bias == 12.5


def test_pairing_aligns_label_ids_before_errors_and_bias():
    f = observations()
    shuffled = f.iloc[::-1].reset_index(drop=True)
    a = paired_sites(f, 'a', 'b')
    b = paired_sites(f, 'a', 'b', shuffled)
    pd.testing.assert_frame_equal(a, b)


def test_bootstrap_replications_missing_a_stage_stay_undefined():
    f = pd.DataFrame(dict(head=[0, 1], site=['s0', 's1'], block=['b0', 'b1'], error=[2., 4.]))
    samples = conditional_samples(f, required_heads=[0, 1], replicates=2000)
    assert 0.45 < np.isfinite(samples).mean() < 0.55
    assert np.unique(samples[np.isfinite(samples)]).tolist() == [3.]
    result = summarize(f, required_heads=[0, 1], replicates=2000)
    assert result['value'] == 3
    assert np.isnan(result['ci_low'])


def test_weather_resamples_transitive_station_block_connections():
    f = pd.DataFrame(dict(site=['a', 'b', 'c', 'd'], block=['A', 'B', 'B', 'C']))
    stations = pd.DataFrame(dict(site=['a', 'b', 'c', 'd'], station=[1, 1, 2, 2]))
    linked = connected_weather_blocks(f, stations)
    assert linked.block.nunique() == 1


def test_verification_report_cannot_modify_frozen_evidence(tmp_path):
    with pytest.raises(ValueError, match='outside the frozen evidence'):
        verify(tmp_path/'evidence', report_path=tmp_path/'evidence'/'report.json')
    assert not (tmp_path/'evidence').exists()
