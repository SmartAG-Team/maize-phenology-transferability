"""Full-source orchestration keeps native labels, frozen roles and replayable fits."""
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
NAMES = ('T', 'nonlinear_T', 'residual_RF', 'residual_MLP', 'thermal_transformer', 'support_blend')


def implementation():
    path = ROOT / 'code/research/full_source_benchmark/fit.py'
    assert path.is_file(), 'Full-source fitting orchestration is missing'
    return importlib.import_module('research.full_source_benchmark.fit')


@pytest.fixture(scope='module')
def cohort():
    # Two native units/site-year intentionally detect accidental one-unit sampling.
    n, days = 187, 28
    u = np.arange(n)
    low = 15 + np.sin(np.arange(days)[None, :] / 10) + (u % 8)[:, None] / 12
    raw = np.stack([low, low + 9, np.broadcast_to(13 + np.arange(days)[None, :] / 200, low.shape)], 2).astype('float32')
    units = pd.DataFrame([dict(unit=f'u{i}', u=i,
        site=f's{i // 2}' if i < 180 else f's{i}',
        site_year=f's{i // 2}|2000' if i < 180 else f's{i}|{2000 if i < 184 else 2015}',
        block=f'b{i // 10}' if i < 180 else f'b{i}',
        year=2000 if i < 184 else 2015,
        inner=(i // 10) % 3, ez=0) for i in u])
    labels = []
    for row in units.to_dict('records'):
        for head, observed in enumerate([1, 3, 6, 9, 12, 16, 20]):
            labels.append(dict(**{k:v for k,v in row.items() if k != 'inner'},
                label_id=f'{row["unit"]}|{head}', head=head, observed=float(observed + (row['u'] % 3) / 4),
                pre2011_prefix_eligible=row['u'] != 180, horizon=days,
                fold0_case='source_1981_2010' if row['u'] <= 180 else ('historical_held' if row['u'] < 184 else 'later_held')))
    data = SimpleNamespace(raw=raw, context=np.column_stack([15 + u / 187, 9 + (u % 7) / 10, u / 100, np.ones(n), np.zeros(n)]).astype('float32'),
        lengths=np.full(n, days), train_lengths=np.full(n, days), units=units,
        labels=pd.DataFrame(labels), protected={})
    design = dict(source_period=[1981, 2010], folds=[0], models=list(NAMES),
        source_counts=[dict(fold=0, eligible_source_sites=90, eligible_source_labels=1260, eligible_source_units=180)],
        alpha_grid=[.5, 1.], rf_strengths=[0., 1.], mlp_strengths=[0., .5, 1.], initial_mlp_steps=[1, 2],
        neural_extension=dict(budgets=[], minimum_improvement_days=.01), optimizer_seed=17,
        artifact_contract=dict(prediction_columns=['label_id', 'unit', 'u', 'site', 'site_year', 'block', 'year', 'head', 'observed', 'horizon', 'period', *NAMES]))
    return data, units[units.u.lt(180)].copy(), design


def test_complete_cohort_keeps_every_eligible_native_unit_and_head(cohort):
    data, members, design = cohort
    prepared = implementation().prepare_fold(data, 0, design, members=members)
    assert len(prepared.source) == 1260
    assert len(prepared.members) == 180
    assert prepared.source.groupby(['site', 'site_year']).unit.nunique().eq(2).all()
    assert set(prepared.source.unit) == set(members.unit)
    assert len(prepared.train) == 840 and len(prepared.tune) == 420
    assert set(prepared.target.label_id) == {f'u{i}|{h}' for i in range(181, 187) for h in range(7)}


@pytest.mark.parametrize('fault', ['missing_unit', 'inner_overlap', 'target_site', 'late_source', 'prefix'])
def test_source_leakage_and_missing_membership_are_rejected_before_fitting(cohort, fault):
    data, members, design = deepcopy(cohort)
    if fault == 'missing_unit':
        members = members.iloc[:-1]
    elif fault == 'inner_overlap':
        members.loc[members.index[0], 'inner'] = 2
    elif fault == 'target_site':
        data.labels.loc[data.labels.u.eq(181), 'site'] = 's0'
    elif fault == 'late_source':
        data.labels.loc[data.labels.u.eq(0), 'year'] = 2011
    else:
        data.train_lengths[0] = 10
    with pytest.raises(ValueError):
        implementation().prepare_fold(data, 0, design, members=members)


@pytest.mark.parametrize('trials, expected', [
    ([dict(steps=80, strength=1., mae=5.), dict(steps=320, strength=1., mae=4.98)], True),
    ([dict(steps=80, strength=1., mae=5.), dict(steps=320, strength=1., mae=4.995)], False),
    ([dict(steps=80, strength=1., mae=5.), dict(steps=320, strength=0., mae=4.)], False),
    ([dict(steps=80, strength=1., mae=4.), dict(steps=320, strength=1., mae=5.)], False),
])
def test_neural_extension_requires_nonzero_boundary_winner_and_real_improvement(trials, expected):
    assert implementation().should_extend(trials, 'steps', .01) is expected


@pytest.fixture(scope='module')
def fitted(cohort, tmp_path_factory):
    data, members, design = cohort
    module = implementation()
    module.set_device('cpu')
    output = tmp_path_factory.mktemp('full-source-smoke')
    receipt = module.run_fold(data, 0, output, design=design, members=members)
    return module, data, members, design, output, receipt


def test_small_real_fit_exports_six_independent_models_with_identical_target_support(fitted):
    from model.api import load, predict
    module, data, members, design, output, receipt = fitted
    folder = output / 'fold0'
    predictions = pd.read_csv(folder / 'predictions.csv.gz')
    assert len(predictions) == 42 and predictions.label_id.nunique() == 42
    assert list(predictions) == design['artifact_contract']['prediction_columns']
    assert len(pd.read_csv(folder / 'source_memberships.csv.gz')) == 180
    ids = predictions.u.unique()
    rows = {int(u): i for i, u in enumerate(ids)}
    for name in NAMES:
        artifact = load(folder / 'models' / f'{name}.pkl', model=name)
        assert artifact.metadata['input_channels'] == 3
        assert artifact.metadata['source_units'] == 180
        assert artifact.metadata['source_labels'] == 1260
        assert artifact.metadata['source_only_selection']
        assert artifact.metadata['selected'] == receipt['selections']
        replay = predict(artifact, data, ids)
        values = replay[predictions.u.map(rows).to_numpy(int), predictions['head'].to_numpy(int)]
        np.testing.assert_allclose(values, predictions[name], atol=1e-5, rtol=0, equal_nan=True)
    assert len(receipt['teachers']['inner']) == 3
    assert len(receipt['teachers']['complete_source']) == 3
    assert receipt['source_counts']['source_labels'] == 1260
    assert receipt['source_counts']['inner_fit_labels'] == 840
    assert receipt['source_counts']['inner_validation_labels'] == 420
    assert receipt['source_target_sites_disjoint']
    assert receipt['source_target_blocks_disjoint']
    assert module.run_fold(data, 0, output, design=design, members=members) == receipt


def test_resume_rejects_changed_output_and_source_configuration(fitted):
    module, data, members, design, output, receipt = fitted
    changed = deepcopy(design)
    changed['optimizer_seed'] = 29
    with pytest.raises(ValueError, match='signature'):
        module.run_fold(data, 0, output, design=changed, members=members)
    path = output / 'fold0/predictions.csv.gz'
    original = path.read_bytes()
    path.write_bytes(original + b'changed')
    try:
        with pytest.raises(ValueError, match='output|Output'):
            module.run_fold(data, 0, output, design=design, members=members)
    finally:
        path.write_bytes(original)


def test_target_horizon_must_match_native_weather_length(cohort):
    data, members, design = deepcopy(cohort)
    data.labels.loc[data.labels.u.eq(181), 'horizon'] = 27
    with pytest.raises(ValueError, match='horizon'):
        implementation().prepare_fold(data, 0, design, members=members)


def test_transformer_extension_retains_source_selected_intermediate_checkpoint(cohort, tmp_path, monkeypatch):
    module = implementation()
    data, members, design = deepcopy(cohort)
    design['neural_extension']['budgets'] = [640, 1280]
    prepared = module.prepare_fold(data, 0, design, members=members)
    initial = module.fit_thermal(data, prepared.train)['thresholds']
    # Replace only expensive optimization; source selection and cache logic stay real.
    def candidate(data, source, initial, context, seed=17, tune=None, fixed_steps=None):
        network = module.neural.ThermalTransformer(initial, 23)
        ids = source.u.unique()
        norm = module.neural.feature_normalizer(module.neural.features(data, ids, False), data.train_lengths[ids])
        if fixed_steps is None:
            return network, norm, 320, [dict(step=0, mae=10.), dict(step=280, mae=8.), dict(step=320, mae=7.)]
        return network, norm, 480, [dict(step=0, mae=10.), dict(step=280, mae=8.), dict(step=320, mae=7.),
                                   dict(step=360, mae=6.), dict(step=480, mae=5.), dict(step=640, mae=6.)]
    monkeypatch.setattr(module.neural, 'fit_network', candidate)
    steps, trials = module._select_transformer(data, prepared, initial, prepared.tune.u.unique(), design, tmp_path)
    assert steps == 480
    assert max(row['step'] for row in trials) == 640


def test_target_outcomes_cannot_change_source_selection_or_fitted_parameters(fitted, tmp_path):
    from model.api import load
    module, original, members, design, first_output, first_receipt = fitted
    data = deepcopy(original)
    data.labels.loc[data.labels.u.ge(181), 'observed'] += 1000
    second = module.run_fold(data, 0, tmp_path, design=design, members=members)
    assert second['selections'] == first_receipt['selections']
    first = load(first_output / 'fold0/models/residual_MLP.pkl')
    other = load(tmp_path / 'fold0/models/residual_MLP.pkl')
    for key in first.payload['residual']['state']:
        np.testing.assert_array_equal(first.payload['residual']['state'][key], other.payload['residual']['state'][key])
    first = load(first_output / 'fold0/models/thermal_transformer.pkl')
    other = load(tmp_path / 'fold0/models/thermal_transformer.pkl')
    for key in first.payload['state']:
        np.testing.assert_array_equal(first.payload['state'][key], other.payload['state'][key])


def test_transformer_checkpoint_selection_preserves_archived_tolerance():
    module = implementation()
    trials = [dict(step=280, mae=5.), dict(step=320, mae=5.-5e-9)]
    assert module._choice(trials, 'step')['step'] == 280
    assert not module.should_extend(trials, 'step', .01)
