"""Complete-source fitting with frozen source roles and independent target evaluation.

Run from the repository root with ``PYTHONPATH=code .venv/bin/python -m
research.full_source_benchmark.fit --fold 0``. Scientific estimators are imported
unchanged; this module only coordinates selection, refitting and export.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

from research.annual_temperature_transfer.design import dataset
from research.phenology_exploration import neural
from research.phenology_exploration.core import metric, sha
from research.phenology_exploration.data import source_units
from research.phenology_exploration.experiments import row_predictions
from research.phenology_exploration.models import (
    fit_residual, fit_thermal, predict, residual_prediction, teacher_predictions,
    validate_source,
)
from research.phenology_exploration.provenance import code_signature, validate_receipt
from research.phenology_exploration.support_blend import support_weight

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model.api import MODEL_NAMES, ModelArtifact, save, predict as artifact_predict

DESIGN_PATH = Path(__file__).with_name('design.json')
DEFAULT_OUTPUT = ROOT / 'outputs/full_source_benchmark_20261026'
MEMBERSHIP_COLUMNS = ['unit', 'u', 'site', 'site_year', 'block', 'year', 'inner']


def progress(step, **values):
    print(json.dumps(dict(step=step, **values), sort_keys=True), flush=True)


def set_device(device):
    """Set the existing neural runtime, including its captured inference default."""
    if device not in ('cpu', 'mps'):
        raise ValueError('Device must be cpu or mps')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is not available')
    neural.DEVICE = device
    # fit_network resolves this function globally but omits the device argument.
    # The archived default was bound at import, so update that runtime default too.
    defaults = neural.predict_network.__defaults__
    neural.predict_network.__defaults__ = (*defaults[:-1], device)
    return device


def load_design():
    return json.loads(DESIGN_PATH.read_text())


@dataclass
class PreparedFold:
    members: pd.DataFrame
    source: pd.DataFrame
    train: pd.DataFrame
    tune: pd.DataFrame
    target: pd.DataFrame
    counts: dict


def prepare_fold(data, fold, design=None, *, members=None):
    """Construct the full eligible cohort; fail before fitting on a role mismatch."""
    design = load_design() if design is None else design
    if fold not in design['folds']:
        raise ValueError(f'Fold {fold} is absent from the frozen design')
    role = f'fold{fold}_case'
    eligible = data.labels[role].eq('source_1981_2010') & data.labels.pre2011_prefix_eligible
    source = data.labels.loc[eligible].copy()
    if source.empty or source.label_id.duplicated().any() or source.duplicated(['unit', 'head']).any():
        raise ValueError('Source labels must be nonempty unique native-stage observations')
    lo, hi = design['source_period']
    if not source.year.between(lo, hi).all():
        raise ValueError('Source years exceed the declared historical period')
    members = source_units(data, fold) if members is None else members.copy()
    if not set(MEMBERSHIP_COLUMNS).issubset(members) or members.unit.duplicated().any():
        raise ValueError('Frozen source memberships are incomplete or duplicated')
    if set(members.unit) != set(source.unit):
        raise ValueError('Frozen source memberships do not cover every eligible native unit exactly')
    if members.inner.isna().any() or not members.inner.isin([0, 1, 2]).all():
        raise ValueError('Every source unit requires a frozen inner role')
    for column in ('u', 'site', 'site_year', 'block', 'year'):
        expected = source[['unit', column]].drop_duplicates()
        if expected.unit.duplicated().any():
            raise ValueError(f'Conflicting source {column} membership')
        check = expected.merge(members[['unit', column]], on='unit', validate='one_to_one', suffixes=('_label', '_member'))
        if not check[f'{column}_label'].equals(check[f'{column}_member']):
            raise ValueError(f'Source membership {column} differs from labels')
    for grouping in ('site', 'block'):
        if members.groupby(grouping).inner.nunique().gt(1).any():
            raise ValueError(f'Source inner roles split a {grouping}')
    inner = members.set_index('unit').inner
    train = source[source.unit.map(inner).ne(2)].copy()
    tune = source[source.unit.map(inner).eq(2)].copy()
    if train.empty or tune.empty:
        raise ValueError('Both source inner fitting and tuning cohorts are required')
    for column in ('u', 'unit', 'site', 'site_year', 'block'):
        if set(train[column]) & set(tune[column]):
            raise ValueError(f'Source train/tune {column} overlap')
    validate_source(data, source)
    target = data.labels[data.labels[role].isin(['historical_held', 'later_held'])].copy()
    target['period'] = target[role]
    if target.empty or target.label_id.duplicated().any():
        raise ValueError('Independent target labels must be nonempty and unique')
    if not np.array_equal(target.horizon.to_numpy(), data.lengths[target.u.to_numpy(int)]):
        raise ValueError('Target horizon differs from its native weather length')
    for column in ('label_id', 'unit', 'u', 'site', 'site_year', 'block'):
        if set(source[column]) & set(target[column]):
            raise ValueError(f'Source/target {column} overlap')
    counts = dict(source_sites=int(source.site.nunique()), source_units=int(source.unit.nunique()), source_labels=len(source),
                  inner_fit_sites=int(train.site.nunique()), inner_fit_units=int(train.unit.nunique()), inner_fit_labels=len(train),
                  inner_validation_sites=int(tune.site.nunique()), inner_validation_units=int(tune.unit.nunique()), inner_validation_labels=len(tune),
                  source_year_min=int(source.year.min()), source_year_max=int(source.year.max()),
                  target_labels=len(target), target_units=int(target.unit.nunique()), target_sites=int(target.site.nunique()),
                  target_labels_by_period={str(k):int(v) for k,v in target.groupby('period').size().items()},
                  source_labels_by_head={str(k):int(v) for k,v in source.groupby('head').size().items()})
    frozen = next((row for row in design['source_counts'] if row['fold'] == fold), None)
    if frozen is None:
        raise ValueError('Frozen source counts are missing')
    for expected, actual in [('eligible_source_sites', 'source_sites'), ('eligible_source_units', 'source_units'), ('eligible_source_labels', 'source_labels'), ('years_min', 'source_year_min'), ('years_max', 'source_year_max')]:
        if expected in frozen and counts[actual] != frozen[expected]:
            raise ValueError(f'Complete-source count mismatch: {actual}={counts[actual]}, expected {frozen[expected]}')
    return PreparedFold(members.sort_values('u').reset_index(drop=True), source, train, tune, target, counts)


def _choice(trials, key):
    if key == 'step':
        # Preserve the archived checkpoint improvement tolerance and earliest tie.
        best = None
        for trial in sorted(trials, key=lambda q: q[key]):
            if best is None or trial['mae'] < best['mae'] - 1e-8:
                best = trial
        return best
    return min(trials, key=lambda q: (q['mae'], q.get('strength', 1.), q[key]))


def should_extend(trials, budget_key, improvement=.01):
    """Continue only a nonzero largest-budget winner with a meaningful gain."""
    selected = _choice(trials, budget_key)
    maximum = max(q[budget_key] for q in trials)
    smaller = [q['mae'] for q in trials if q[budget_key] < maximum]
    return bool(maximum > 0 and selected[budget_key] == maximum and selected.get('strength', 1.) > 0
                and smaller and min(smaller) - selected['mae'] > improvement)


def _frame_sha(frame):
    return hashlib.sha256(frame.to_csv(index=False, float_format='%.17g').encode()).hexdigest()


def signature(data, prepared, fold, design):
    code = code_signature('neural.py', 'support_blend.py', 'analysis.py')
    extra = [Path(__file__), DESIGN_PATH, ROOT/'model/api.py',
             ROOT/'code/research/annual_temperature_transfer/design.py',
             ROOT/'code/research/annual_temperature_transfer/core.py',
             ROOT/'code/research/annual_temperature_transfer/climate.py']
    code.update({str(path):sha(path) for path in extra})
    return dict(study=design.get('study', 'full-source'), fold=int(fold), code=code,
                inputs=dict(data.protected), design=design, design_sha256=sha(DESIGN_PATH),
                source_membership_sha256=_frame_sha(prepared.members),
                source_labels_sha256=_frame_sha(prepared.source), target_labels_sha256=_frame_sha(prepared.target),
                device=neural.DEVICE, estimator_seed=int(design.get('optimizer_seed', 17)))


def _json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def _pickle(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _cached(folder, name, calculate):
    """Resume only a hashed completed stage under the verified fold signature."""
    cache = folder/'cache'; cache.mkdir(exist_ok=True)
    path = cache/f'{name}.pkl'; receipt = cache/f'{name}.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if not path.is_file() or sha(path) != saved['sha256']:
            raise ValueError(f'Cached output changed: {name}')
        progress('cache_reuse', fold=folder.name, component=name)
        with path.open('rb') as stream:
            return pickle.load(stream)
    progress('fit_start', fold=folder.name, component=name)
    started = time.monotonic()
    result = calculate()
    _pickle(path, result)
    _json(receipt, dict(sha256=sha(path), elapsed_seconds=time.monotonic()-started))
    progress('fit_complete', fold=folder.name, component=name, seconds=round(time.monotonic()-started, 2))
    return result


def _teacher(data, labels):
    teacher, receipts = teacher_predictions(data, labels)
    if len(receipts) != 3:
        raise ValueError('Three whole-source-block teachers are required')
    qualified = np.asarray([r['support'] for r in receipts]).min(0) >= 30
    return teacher, receipts, qualified


def _score(data, labels, ids, predictions):
    return metric(labels, row_predictions(labels, ids, predictions), data.train_lengths[labels.u])['penalized_mae']


def _safe(values, lengths):
    return np.where(np.isfinite(values) & (values >= 0) & (values < lengths[:, None]), values, np.nan)


def _select_mlp(data, prepared, teacher, qualified, base_tune, ids, design, folder):
    trials = []; seed = int(design.get('optimizer_seed', 17))
    def evaluate(steps):
        model = _cached(folder, f'inner_MLP_{steps}', lambda: fit_residual(data, prepared.train, teacher, 'MLP', seed, steps, teacher_qualified=qualified))
        correction = residual_prediction(model, data, ids, base_tune, train=True)
        for strength in design['mlp_strengths']:
            trials.append(dict(steps=int(steps), strength=float(strength), mae=_score(data, prepared.tune, ids, base_tune + strength*correction)))
        progress('source_tune', fold=folder.name, model='residual_MLP', steps=steps, best=_choice(trials, 'steps'))
    for steps in design['initial_mlp_steps']:
        evaluate(int(steps))
    extension = design['neural_extension']
    for steps in extension['budgets']:
        if not should_extend(trials, 'steps', extension['minimum_improvement_days']):
            break
        evaluate(int(steps))
    return _choice(trials, 'steps'), trials


def _select_transformer(data, prepared, initial, ids, design, folder):
    seed = int(design.get('optimizer_seed', 17))
    def initial_selection():
        _, _, steps, trials = neural.fit_network(data, prepared.train, initial, False, seed=seed, tune=prepared.tune)
        return int(steps), trials
    selected, trials = _cached(folder, 'inner_transformer_checkpoints', initial_selection)
    trials = [dict(q) for q in trials]
    if selected != _choice(trials, 'step')['step']:
        raise ValueError('Archived Transformer checkpoint selection differs from source trials')
    extension = design['neural_extension']
    for steps in extension['budgets']:
        if not should_extend(trials, 'step', extension['minimum_improvement_days']):
            break
        def evaluate(steps=int(steps)):
            _, _, selected, history = neural.fit_network(data, prepared.train, initial, False,
                seed=seed, tune=prepared.tune, fixed_steps=steps)
            return int(selected), history
        selected, history = _cached(folder, f'inner_transformer_{steps}', evaluate)
        if selected != _choice(history, 'step')['step']:
            raise ValueError('Extended Transformer source checkpoint selection differs from trials')
        # The seeded fit recreates the same optimization prefix. Preserve each
        # previously scored checkpoint once and retain every new 40-step score.
        previous_max = max(q['step'] for q in trials)
        trials.extend(dict(q) for q in history if q['step'] > previous_max)
        progress('source_tune', fold=folder.name, model='thermal_transformer', steps=steps, best=_choice(trials, 'step'))
    return int(_choice(trials, 'step')['step']), trials


def _fit_all(data, prepared, design, folder):
    train, source, tune = prepared.train, prepared.source, prepared.tune
    ids = tune.u.unique(); seed = int(design.get('optimizer_seed', 17))
    inner_T = _cached(folder, 'inner_T', lambda: fit_thermal(data, train))
    final_T = _cached(folder, 'full_T', lambda: fit_thermal(data, source))
    alpha_trials = []
    for alpha in design['alpha_grid']:
        candidate = inner_T if float(alpha) == 1. else _cached(folder, f'inner_alpha_{float(alpha):g}', lambda alpha=alpha: fit_thermal(data, train, alpha))
        value = _score(data, tune, ids, predict(candidate, data, ids, train=True))
        alpha_trials.append(dict(alpha=float(alpha), mae=value))
        progress('source_tune', fold=folder.name, model='nonlinear_T', alpha=float(alpha), mae=value)
    alpha_choice = min(alpha_trials, key=lambda q: (q['mae'], abs(q['alpha']-1)))
    nonlinear_T = final_T if alpha_choice['alpha'] == 1. else _cached(folder, 'full_nonlinear_T', lambda: fit_thermal(data, source, alpha_choice['alpha']))
    teacher, inner_receipts, qualified = _cached(folder, 'inner_teachers', lambda: _teacher(data, train))
    base_tune = predict(inner_T, data, ids, train=True)
    rf = _cached(folder, 'inner_RF', lambda: fit_residual(data, train, teacher, 'RF', seed, teacher_qualified=qualified))
    correction = residual_prediction(rf, data, ids, base_tune, train=True)
    rf_trials = [dict(strength=float(strength), mae=_score(data, tune, ids, base_tune + strength*correction)) for strength in design['rf_strengths']]
    rf_choice = min(rf_trials, key=lambda q: (q['mae'], q['strength']))
    progress('source_tune', fold=folder.name, model='residual_RF', best=rf_choice)
    mlp_choice, mlp_trials = _select_mlp(data, prepared, teacher, qualified, base_tune, ids, design, folder)
    del teacher
    full_teacher, full_receipts, full_qualified = _cached(folder, 'full_teachers', lambda: _teacher(data, source))
    full_RF = _cached(folder, 'full_RF', lambda: fit_residual(data, source, full_teacher, 'RF', seed, teacher_qualified=full_qualified))
    full_MLP = _cached(folder, 'full_MLP', lambda: fit_residual(data, source, full_teacher, 'MLP', seed, mlp_choice['steps'], teacher_qualified=full_qualified))
    del full_teacher
    for label, base in [('inner source', inner_T), ('complete source', final_T)]:
        if not np.isfinite(base['thresholds']).all():
            raise ValueError(f'Transformer requires >=30 source environments/head in {label}')
    nn_steps, nn_trials = _select_transformer(data, prepared, inner_T['thresholds'], ids, design, folder)
    def full_network():
        network, norm, _, _ = neural.fit_network(data, source, final_T['thresholds'], False, seed=seed, fixed_steps=nn_steps)
        return dict(state={k:v.detach().cpu() for k,v in network.state_dict().items()}, norm=norm, initial=final_T['thresholds'], context=False)
    network_payload = _cached(folder, 'full_transformer', full_network)
    source_climate = data.context[prepared.members.drop_duplicates('site').u.to_numpy(int), :2].copy()
    _, geometry = support_weight(source_climate, source_climate)
    selections = dict(alpha=alpha_choice, rf=rf_choice, mlp=mlp_choice, transformer_updates=nn_steps)
    trials = dict(nonlinear_T=alpha_trials, residual_RF=rf_trials, residual_MLP=mlp_trials, thermal_transformer=nn_trials)
    teachers = dict(inner=inner_receipts, complete_source=full_receipts)
    common = dict(input_channels=int(data.raw.shape[2]), **prepared.counts, seed=seed,
                  source_only_selection=True, refit='all eligible source fitting plus tuning labels',
                  selected=selections, selection_trials=trials, teachers=teachers,
                  neural_extension=design['neural_extension'], source_period=design['source_period'])
    residual_payload = dict(base=final_T, residual=full_MLP, strength=mlp_choice['strength'])
    artifacts = {
        'T':ModelArtifact('T', dict(physical=final_T), dict(common)),
        'nonlinear_T':ModelArtifact('nonlinear_T', dict(physical=nonlinear_T), dict(common)),
        'residual_RF':ModelArtifact('residual_RF', dict(base=final_T, residual=full_RF, strength=rf_choice['strength']), dict(common)),
        'residual_MLP':ModelArtifact('residual_MLP', dict(residual_payload), dict(common)),
        'thermal_transformer':ModelArtifact('thermal_transformer', network_payload, dict(common)),
        'support_blend':ModelArtifact('support_blend', dict(**residual_payload, physical=nonlinear_T, source_climate=source_climate), dict(common, support_geometry=geometry)),
    }
    audit = dict(selections=selections, selection_trials=trials, teachers=teachers,
                 correction_support=dict(residual_RF=full_RF['supported'], residual_MLP=full_MLP['supported']),
                 support_geometry=geometry)
    return artifacts, audit


def _predictions(data, prepared, artifacts, folder):
    """Evaluate only after source fitting and selection have finished."""
    target = prepared.target; ids = target.u.unique()
    progress('target_prediction_start', fold=folder.name, target_units=len(ids), target_labels=len(target))
    columns = ['label_id', 'unit', 'u', 'site', 'site_year', 'block', 'year', 'head', 'observed', 'horizon', 'period']
    out = target[columns].copy()
    for name in MODEL_NAMES:
        # The public replay path masks invalid dates consistently and evaluates
        # Transformer inference on CPU, independently of its fitting device.
        values = artifact_predict(artifacts[name], data, ids)
        out[name] = row_predictions(target, ids, values)
        progress('target_prediction_complete', fold=folder.name, model=name)
    return out


def run_fold(data, fold, output=DEFAULT_OUTPUT, *, design=None, members=None, prepare_only=False):
    design = load_design() if design is None else design
    prepared = prepare_fold(data, fold, design, members=members)
    folder = Path(output)/f'fold{fold}'
    sig = signature(data, prepared, fold, design)
    if not all(Path(path).is_file() and sha(path) == digest for path,digest in data.protected.items()):
        raise ValueError('Protected input signature changed before fitting')
    completed = folder/'complete.json'
    if completed.exists():
        receipt = json.loads(completed.read_text())
        validate_receipt(receipt, sig, folder)
        progress('fold_reuse', fold=fold, source_counts=prepared.counts)
        return receipt
    state_path = folder/'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('signature') != sig:
            raise ValueError('Incomplete calculation signature changed; use a fresh output directory')
    elif folder.exists() and any(folder.iterdir()):
        raise ValueError('Unverified existing fold outputs; use a fresh output directory')
    folder.mkdir(parents=True, exist_ok=True)
    if not state_path.exists():
        _json(state_path, dict(signature=sig))
    membership_path = folder/'source_memberships.csv.gz'
    if membership_path.exists():
        existing = pd.read_csv(membership_path, float_precision='round_trip')
        if not existing.astype(str).equals(prepared.members.astype(str)):
            raise ValueError('Saved complete-source membership changed')
    else:
        prepared.members.to_csv(membership_path, index=False, compression=dict(method='gzip', mtime=0))
    progress('fold_prepared', fold=fold, device=neural.DEVICE, **prepared.counts)
    if prepare_only:
        return dict(prepared=True, fold=fold, source_counts=prepared.counts, signature=sig)
    started = time.monotonic()
    artifacts, audit = _fit_all(data, prepared, design, folder)
    for name in MODEL_NAMES:
        save(artifacts[name], folder/'models'/f'{name}.pkl')
    progress('models_exported', fold=fold, models=list(MODEL_NAMES))
    predictions = _predictions(data, prepared, artifacts, folder)
    required = design.get('artifact_contract', {}).get('prediction_columns', list(predictions))
    if list(predictions) != required:
        raise ValueError('Prediction column contract differs from the frozen design')
    predictions.to_csv(folder/'predictions.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    if signature(data, prepared, fold, design) != sig:
        raise ValueError('Fitting code or design signature changed during execution')
    if not all(sha(path) == digest for path,digest in data.protected.items()):
        raise ValueError('Protected input signature changed during fitting')
    receipt = dict(signature=sig, fold=int(fold), source_counts=prepared.counts, **audit,
        source_target_sites_disjoint=True, source_target_blocks_disjoint=True,
        source_only_selection=True, target_labels_evaluation_only=True,
        input_channels=int(data.raw.shape[2]), elapsed_seconds=time.monotonic()-started,
        membership_sha256=sha(membership_path),
        outputs={str(path.relative_to(folder)):sha(path) for path in sorted(folder.rglob('*'))
                 if path.is_file() and path != completed and not path.name.endswith('.tmp')})
    _json(completed, receipt)
    progress('fold_complete', fold=fold, seconds=round(receipt['elapsed_seconds'], 2), selections=receipt['selections'])
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'mps'])
    args = parser.parse_args()
    previous_device = neural.DEVICE
    previous_defaults = neural.predict_network.__defaults__
    try:
        if args.device is not None:
            set_device(args.device)
        progress('dataset_load_start', fold=args.fold)
        data, _, _ = dataset()
        progress('dataset_loaded', fold=args.fold, units=len(data.units), labels=len(data.labels))
        run_fold(data, args.fold, args.output, prepare_only=args.prepare_only)
    finally:
        neural.DEVICE = previous_device
        neural.predict_network.__defaults__ = previous_defaults


if __name__ == '__main__':
    main()
