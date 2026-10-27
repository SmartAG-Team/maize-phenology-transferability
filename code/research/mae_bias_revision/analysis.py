"""Reconstruct ordinary MAE/bias from immutable saved observation predictions.

No fitting, model selection, or prediction is performed. Failure counts retain all
observed labels; dates outside their recorded weather horizon have no error sign.
Every comparison intersects valid labels before hierarchical aggregation.
"""
import argparse
from functools import lru_cache
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from research.phenology_exploration.core import sha, valid_prediction

ROOT = Path(__file__).resolve().parents[3]
ANNUAL = ROOT / 'data/results/annual_temperature_transfer_20261024'
PRIOR = ROOT / 'data/results/phenology_exploration_20261023/year_matched'
PRIMARY = ROOT / 'data/results/full_source_benchmark_20261026'
DEFAULT_OUTPUT = ROOT / 'outputs/mae_bias_revision_20261026/evidence'
MODELS = ['T', 'nonlinear_T', 'residual_RF', 'residual_MLP', 'thermal_transformer', 'support_blend']
SITE_KEYS = ['head', 'site', 'block']
STAGES = ['Emergence', 'Stem elongation', 'Tasseling', 'Flowering onset', 'Milk stage', 'Early dough', 'Physiological maturity']
BBCH = [10, 31, 53, 61, 75, 83, 87]
SEED = 20261023
REPLICATES = 2000
EDGES = [-np.inf, -3, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 3, np.inf]
INPUTS = {}


def record(path):
    path = Path(path)
    digest = sha(path)
    if str(path) in INPUTS and INPUTS[str(path)] != digest:
        raise ValueError(f'Input changed during reconstruction: {path}')
    INPUTS[str(path)] = digest
    return digest


def read_csv(path, receipt=None, **kwargs):
    path = Path(path)
    digest = record(path)
    if receipt is not None:
        receipt = Path(receipt)
        record(receipt)
        j = json.loads(receipt.read_text())
        if j.get('complete') is False:
            raise ValueError(f'Incomplete archived receipt: {receipt}')
        declared = j.get('outputs', j.get('output_sha256', j.get('files', {})))
        if declared.get(path.name) != digest:
            raise ValueError(f'Archived prediction hash mismatch: {path}')
    return pd.read_csv(path, **kwargs)


def fitted(folder):
    folder = Path(folder)
    f = read_csv(folder / 'predictions.csv.gz', folder / 'complete.json')
    if f.label_id.isna().any() or f.label_id.duplicated().any():
        raise ValueError('Missing/duplicate observed label')
    if 'period' not in f and 'case' in f:
        f = f.rename(columns={'case': 'period'})
    return f


def write_tables(folder, tables):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        compression = {'method': 'gzip', 'mtime': 0} if name.endswith('.gz') else None
        table.to_csv(folder / name, index=False, compression=compression)


def site_values(frame, values):
    """Observations -> site-years -> sites; undefined years have no valid error."""
    q = frame[SITE_KEYS + ['site_year']].copy()
    for name, value in values.items():
        q[name] = np.asarray(value, float)
    columns = list(values)
    yearly = q.groupby(SITE_KEYS + ['site_year'], observed=True)[columns].mean()
    return yearly.groupby(level=SITE_KEYS, observed=True)[columns].mean().reset_index()


def support_counts(frame, valid, prefix='valid'):
    q = frame[SITE_KEYS + ['label_id', 'unit', 'site_year', 'year']].copy()
    q['_valid'] = np.asarray(valid, bool)
    all_counts = q.groupby(SITE_KEYS, observed=True).agg(
        n_native_labels=('label_id', 'size'), n_native_units=('unit', 'nunique'),
        n_observed_years=('year', 'nunique'), n_site_years=('site_year', 'nunique'),
        year_min=('year', 'min'), year_max=('year', 'max')).reset_index()
    valid_counts = q[q._valid].groupby(SITE_KEYS, observed=True).agg(
        n_valid_labels=('label_id', 'size'), n_valid_units=('unit', 'nunique'),
        n_valid_site_years=('site_year', 'nunique')).reset_index()
    result = all_counts.merge(valid_counts, on=SITE_KEYS, how='left', validate='one_to_one')
    for column in ['n_valid_labels', 'n_valid_units', 'n_valid_site_years']:
        result[column] = result[column].fillna(0).astype(int)
    result['n_failed_labels'] = result.n_native_labels - result.n_valid_labels
    if prefix != 'valid':
        result = result.rename(columns={c: c.replace('n_valid', f'n_{prefix}') for c in result if c.startswith('n_valid')})
        result = result.rename(columns={'n_failed_labels': 'n_excluded_labels'})
    return result


def model_sites(frame, model):
    pred = frame[model].to_numpy(float)
    valid = valid_prediction(frame, pred)
    delta = np.where(valid, pred - frame.observed.to_numpy(float), np.nan)
    result = site_values(frame, dict(error=abs(delta), bias=delta,
        completion=valid.astype(float), failure14=np.where(valid, (abs(delta) > 14).astype(float), np.nan)))
    return result.merge(support_counts(frame, valid), on=SITE_KEYS, validate='one_to_one')


def aligned_reference(frame, reference):
    """Align exact native targets; metadata changes are an error."""
    if frame.label_id.duplicated().any() or reference.label_id.duplicated().any():
        raise ValueError('Duplicate observed label in paired comparison')
    if set(frame.label_id) != set(reference.label_id):
        raise ValueError('Different observed targets in paired comparison')
    rr = reference.set_index('label_id').loc[frame.label_id].reset_index()
    for col in ['site', 'site_year', 'block', 'head', 'year', 'observed', 'horizon']:
        if not np.array_equal(frame[col].to_numpy(), rr[col].to_numpy(), equal_nan=False):
            raise ValueError(f'Changed paired target metadata: {col}')
    return rr


def paired_sites(frame, left, right, reference=None):
    reference = frame if reference is None else aligned_reference(frame, reference)
    a, b = frame[left].to_numpy(float), reference[right].to_numpy(float)
    av, bv = valid_prediction(frame, a), valid_prediction(reference, b)
    common = av & bv
    observed = frame.observed.to_numpy(float)
    ae, be = np.where(common, abs(a - observed), np.nan), np.where(common, abs(b - observed), np.nan)
    ab, bb = np.where(common, a - observed, np.nan), np.where(common, b - observed, np.nan)
    result = site_values(frame, dict(error=ae-be, bias=ab-bb, left_error=ae, right_error=be,
        left_bias=ab, right_bias=bb, common_valid_fraction=common.astype(float),
        left_failure14=np.where(common, (ae > 14).astype(float), np.nan),
        right_failure14=np.where(common, (be > 14).astype(float), np.nan),
        completion=av.astype(float), reference_completion=bv.astype(float)))
    result = result.merge(support_counts(frame, common, 'common_valid'), on=SITE_KEYS, validate='one_to_one')
    counts = frame[SITE_KEYS].assign(_left=av.astype(int), _right=bv.astype(int))
    counts = counts.groupby(SITE_KEYS)[['_left', '_right']].sum().reset_index().rename(
        columns={'_left': 'n_left_valid_labels', '_right': 'n_right_valid_labels'})
    return result.merge(counts, on=SITE_KEYS, validate='one_to_one')


def correction_components(frame, reference, prediction):
    pair = paired_sites(frame, reference, prediction)
    a, b = frame[reference].to_numpy(float), frame[prediction].to_numpy(float)
    common = valid_prediction(frame, a) & valid_prediction(frame, b)
    gain = np.where(common, abs(a-frame.observed.to_numpy()) - abs(b-frame.observed.to_numpy()), np.nan)
    metrics = dict(benefit=np.maximum(gain, 0), harm=np.maximum(-gain, 0), net_gain=gain,
        improved_fraction=np.where(common, (gain > 0).astype(float), np.nan),
        harmed_fraction=np.where(common, (gain < 0).astype(float), np.nan))
    return pair.merge(site_values(frame, metrics), on=SITE_KEYS, validate='one_to_one')


@lru_cache(maxsize=12)
def bootstrap_weights(blocks, replicates, seed):
    if not blocks:
        return np.empty((replicates, 0))
    return np.random.default_rng(seed).multinomial(len(blocks), np.full(len(blocks), 1/len(blocks)), size=replicates).astype(float)


def conditional_samples(frame, column='error', required_heads=range(7), replicates=REPLICATES, seed=SEED):
    if frame.duplicated(SITE_KEYS).any():
        raise ValueError('Duplicate stage/site in conditional bootstrap')
    heads = list(required_heads)
    blocks = tuple(sorted(frame.block.unique()))
    draw = bootstrap_weights(blocks, int(replicates), int(seed))
    totals, counts = [], []
    for head in heads:
        q = frame[frame['head'].eq(head) & np.isfinite(frame[column])]
        stats = q.groupby('block')[column].agg(['sum', 'count']).reindex(blocks).fillna(0)
        totals.append(stats['sum'].to_numpy())
        counts.append(stats['count'].to_numpy())
    numerator, denominator = draw @ np.asarray(totals).T, draw @ np.asarray(counts).T
    complete = (denominator > 0).all(axis=1)
    out = np.full(replicates, np.nan)
    out[complete] = (numerator[complete] / denominator[complete]).mean(axis=1)
    return out


def summarize(frame, column='error', required_heads=range(7), replicates=REPLICATES, seed=SEED):
    finite = np.isfinite(frame[column])
    means = frame.loc[finite].groupby('head')[column].mean().reindex(list(required_heads))
    value = float(means.mean()) if means.notna().all() else np.nan
    draws = conditional_samples(frame, column, required_heads, replicates, seed)
    valid = np.isfinite(draws)
    interval = np.quantile(draws[valid], [.025, .975]) if valid.mean() >= .95 else [np.nan, np.nan]
    return dict(value=value, ci_low=float(interval[0]), ci_high=float(interval[1]),
        complete_bootstrap_fraction=float(valid.mean()), heads=int(means.notna().sum()),
        sites=int(frame.site.nunique()), valid_sites=int(frame.loc[finite, 'site'].nunique()),
        site_stages=len(frame), valid_site_stages=int(finite.sum()))


def metric_summary(frame, replicates=REPLICATES, seed=SEED, required_heads=range(7)):
    error = summarize(frame, 'error', required_heads, replicates, seed)
    bias = summarize(frame, 'bias', required_heads, replicates, seed)
    result = dict(mae=error.pop('value'), **error, bias=bias['value'], bias_ci_low=bias['ci_low'],
        bias_ci_high=bias['ci_high'], bias_complete_bootstrap_fraction=bias['complete_bootstrap_fraction'])
    for metric in ['completion', 'failure14', 'common_valid_fraction']:
        if metric in frame:
            result[metric] = summarize(frame, metric, required_heads, replicates, seed)['value']
    for col in ['n_native_labels', 'n_valid_labels', 'n_failed_labels', 'n_common_valid_labels', 'n_excluded_labels']:
        if col in frame:
            result[col] = float(frame[col].sum())
    return result


def average_draws(frame, dimensions, values=None):
    keys = list(dimensions) + SITE_KEYS
    if values is None:
        values = [c for c in frame.select_dtypes(include='number') if c not in keys + ['draw', 'fold', 'budget']]
    result = frame.groupby(keys, observed=True)[values].mean().reset_index()
    draws = frame.assign(_valid=np.isfinite(frame.error)).groupby(keys, observed=True).agg(
        n_source_draws=('draw', 'nunique'), n_valid_source_draws=('_valid', 'sum')).reset_index()
    return result.merge(draws, on=keys, validate='one_to_one')


def contrast_rows(frame, info, replicates=REPLICATES, seed=SEED, stages=True):
    rows = []
    groups = [('all', frame)] + ([(str(h), q) for h, q in frame.groupby('head')] if stages else [])
    for head, q in groups:
        required = range(7) if head == 'all' else [int(head)]
        v = metric_summary(q, replicates, seed, required)
        v['difference'] = v.pop('mae')
        add_bias_difference(v)
        rows.append(dict(**info, head=head, **v))
    return rows


def add_bias_difference(row):
    row.update(bias_difference=row['bias'], bias_difference_ci_low=row['bias_ci_low'],
        bias_difference_ci_high=row['bias_ci_high'])


def completion_audit(frame, models, info):
    rows = []
    for period, targets in frame.groupby('period'):
        for model in models:
            v = valid_prediction(targets, targets[model])
            q = targets[['head', 'site', 'year', 'horizon']].assign(valid=v)
            for head, g in q.groupby('head'):
                pred = targets.loc[g.index, model].to_numpy(float)
                rows.append(dict(**info, period=period, model=model, head=int(head), labels=len(g),
                    valid_labels=int(g.valid.sum()), failed_labels=int((~g.valid).sum()),
                    nonfinite=int((~np.isfinite(pred)).sum()), negative=int((np.isfinite(pred)&(pred<0)).sum()),
                    beyond_horizon=int((np.isfinite(pred)&(pred>=g.horizon.to_numpy())).sum()),
                    sites=int(g.site.nunique()), valid_sites=int(g.loc[g.valid, 'site'].nunique())))
    return rows


def primary_tables(replicates=REPLICATES, seed=SEED):
    predictions, sites, summaries, stages, contrasts, pairs, components, corrections, audit = [], [], [], [], [], [], [], [], []
    from research.full_source_benchmark.analysis import validate_targets
    for fold in range(5):
        f = fitted(PRIMARY / f'fold{fold}')
        expected = fitted(PRIOR / 'cases' / f'f{fold}_random_n80_d17')
        validate_targets(f, expected)
        predictions.append(f.assign(fold=fold))
    frame = pd.concat(predictions, ignore_index=True)
    if frame.label_id.duplicated().any():
        raise ValueError('Duplicate cross-fold primary observations')
    later = frame[frame.period.eq('later_held')]
    if len(later) != 56447 or later.site.nunique() != 1029:
        raise ValueError('Changed full-source later target population')
    for period, targets in frame.groupby('period'):
        info = dict(study='full_source', recipe='all_eligible', period=period)
        audit.extend(completion_audit(targets, MODELS, dict(study='full_source')))
        for model in MODELS:
            s = model_sites(targets, model).assign(model=model, **info)
            sites.append(s)
            summaries.append(dict(**info, model=model, labels=len(targets), units=targets.unit.nunique(), **metric_summary(s, replicates, seed)))
            for head, h in s.groupby('head'):
                stages.append(dict(**info, model=model, head=int(head), bbch=BBCH[head], stage=STAGES[head],
                    labels=int(h.n_native_labels.sum()), **metric_summary(h, replicates, seed, [head])))
        for right, left in itertools.combinations(MODELS, 2):
            pair = paired_sites(targets, left, right).assign(left=left, right=right, **info)
            pairs.append(pair)
            contrasts.extend(contrast_rows(pair, dict(**info, left=left, right=right), replicates, seed))
        for model in MODELS[2:]:
            q = correction_components(targets, 'nonlinear_T', model).assign(model=model, **info)
            components.append(q)
            row = dict(**info, model=model, reference='nonlinear_T')
            for metric in ['benefit', 'harm', 'net_gain', 'improved_fraction', 'harmed_fraction']:
                v = summarize(q, metric, range(7), replicates, seed)
                row.update({metric: v['value'], metric+'_ci_low': v['ci_low'], metric+'_ci_high': v['ci_high']})
            corrections.append(row)
    return dict(zip(['summary.csv', 'stage_performance.csv', 'benchmark_sites.csv.gz', 'paired_contrasts.csv',
        'paired_sites.csv.gz', 'correction_sites.csv.gz', 'correction_summary.csv', 'support_audit.csv', 'keyed_predictions.csv.gz'],
        [pd.DataFrame(summaries), pd.DataFrame(stages), pd.concat(sites, ignore_index=True), pd.DataFrame(contrasts),
         pd.concat(pairs, ignore_index=True), pd.concat(components, ignore_index=True), pd.DataFrame(corrections), pd.DataFrame(audit), frame]))


def configuration(path):
    record(path)
    return json.loads(Path(path).read_text())['cases']


def estimate_columns(q, columns, replicates=REPLICATES, seed=SEED, required_heads=range(7)):
    # Repeated source draws are averaged only after each draw's label hierarchy.
    a = q.groupby(SITE_KEYS, observed=True)[columns].mean().reset_index()
    row = {}
    for column in columns:
        v = summarize(a, column, required_heads, replicates, seed)
        row.update({column if key == 'value' else column+'_'+key: value for key, value in v.items()})
    return row


def zone_tables(replicates=REPLICATES, seed=SEED):
    cases = configuration(ANNUAL / 'cases.json')
    if len(cases) != 90:
        raise ValueError('Expected all 90 temperature-zone cases')
    specifications = pd.DataFrame(cases)
    tables, audit, identities = [], [], {}
    for (fold, draw), group in specifications.groupby(['fold', 'draw']):
        frames = {int(c.source_zone): fitted(ANNUAL / 'cases' / c.case) for c in group.itertuples()}
        for f in frames.values():
            ids = tuple(sorted(f.label_id))
            if fold in identities and identities[fold] != ids:
                raise ValueError('Changed held observations across source zones or draws')
            identities[fold] = ids
        diagonal = pd.concat([f[f.target_zone.eq(z)] for z, f in frames.items() if z >= 0], ignore_index=True)
        if diagonal.label_id.duplicated().any():
            raise ValueError('Duplicate diagonal observations')
        for source, f in frames.items():
            info = dict(fold=int(fold), draw=int(draw), source_zone=int(source))
            reference = aligned_reference(f, diagonal)
            f = f.copy()
            f['distance_bin'] = pd.cut(f.signed_mat_distance_c, EDGES, labels=False, right=False)
            audit.extend(completion_audit(f, MODELS, info))
            for period, q in f.groupby('period'):
                rr = reference.set_index('label_id').loc[q.label_id].reset_index()
                extras = q[SITE_KEYS + ['target_zone', 'mat_c', 'signed_mat_distance_c', 'outside_mat_range_c', 'distance_bin']].drop_duplicates()
                if extras.duplicated(SITE_KEYS).any():
                    raise ValueError('Climate metadata changes within a target site')
                for model in MODELS:
                    pair = paired_sites(q, model, model, rr).rename(columns={
                        'error': 'penalty', 'bias': 'bias_difference', 'left_error': 'error',
                        'right_error': 'diagonal_error', 'left_bias': 'bias', 'right_bias': 'diagonal_bias',
                        'left_failure14': 'failure14', 'right_failure14': 'diagonal_failure14'})
                    marginal = model_sites(q, model)[SITE_KEYS + ['error', 'bias', 'failure14', 'n_valid_labels', 'n_failed_labels']].rename(
                        columns={'error': 'marginal_error', 'bias': 'marginal_bias', 'failure14': 'marginal_failure14'})
                    s = pair.merge(marginal, on=SITE_KEYS, validate='one_to_one').merge(extras, on=SITE_KEYS, validate='one_to_one')
                    s['diagonal_available'] = s.diagonal_error.notna()
                    tables.append(s.assign(model=model, period=period, **info))
        print(f'Zone rescoring fold {fold}, draw {draw}', flush=True)
    paired = pd.concat(tables, ignore_index=True)
    metric = ['error', 'penalty', 'failure14', 'completion', 'bias', 'common_valid_fraction', 'marginal_error', 'marginal_bias', 'bias_difference']
    matrix, direction, curves, stages, summary = [], [], [], [], []
    for keys, q in paired.groupby(['period', 'model', 'source_zone', 'target_zone']):
        period, model, source, target = keys
        info = dict(period=period, model=model, source_zone=int(source), target_zone=int(target))
        row = dict(**info, **estimate_columns(q, metric, replicates, seed))
        row['mean_signed_mat_distance_c'] = float(q.groupby('site').signed_mat_distance_c.mean().mean())
        matrix.append(row)
        for head, h in q.groupby('head'):
            v = estimate_columns(h, ['error', 'penalty', 'bias'], replicates, seed, [head])
            stages.append(dict(**info, head=int(head), bbch=BBCH[head], stage=STAGES[head],
                stage_error=v['error'], stage_penalty=v['penalty'], stage_bias=v['bias'], **v))
    off = paired[paired.source_zone.ge(0) & paired.source_zone.ne(paired.target_zone)].copy()
    off['direction'] = np.where(off.target_zone.gt(off.source_zone), 'cold_to_warm', 'warm_to_cold')
    for keys, q in off.groupby(['period', 'model', 'direction']):
        period, model, name = keys
        direction.append(dict(period=period, model=model, direction=name, **estimate_columns(q, metric, replicates, seed)))
    for keys, q in off.groupby(['period', 'model', 'distance_bin']):
        period, model, distance_bin = keys
        curves.append(dict(period=period, model=model, distance_bin=int(distance_bin),
            signed_mat_distance_c=float(q.groupby(['head', 'site']).signed_mat_distance_c.mean().groupby('head').mean().mean()),
            **estimate_columns(q, metric, replicates, seed)))
    for (period, model), q in paired.groupby(['period', 'model']):
        for name, mask in [('all_single_zone', q.source_zone.ge(0)),
            ('off_diagonal', q.source_zone.ge(0)&q.source_zone.ne(q.target_zone)),
            ('diagonal', q.source_zone.eq(q.target_zone)), ('pooled_source', q.source_zone.eq(-1))]:
            summary.append(dict(period=period, model=model, comparison=name,
                **estimate_columns(q.loc[mask], metric, replicates, seed)))
    diagonal = paired[paired.source_zone.eq(paired.target_zone)]
    if not np.allclose(diagonal.penalty.dropna(), 0):
        raise ValueError('Diagonal ordinary MAE difference is not zero')
    return {'site_transfer_errors.csv.gz': paired, 'matrix.csv': pd.DataFrame(matrix),
        'directional_summary.csv': pd.DataFrame(direction), 'temperature_distance_curves.csv': pd.DataFrame(curves),
        'stage_matrix.csv': pd.DataFrame(stages), 'summary.csv': pd.DataFrame(summary), 'support_audit.csv': pd.DataFrame(audit)}


def old_predictions(case):
    name = case['case']
    f = fitted(PRIOR / 'cases' / name).rename(columns={'nonlinear_T': 'restricted_nonlinear_T'})
    for folder, columns in [('response_followup', ['relaxed_T', 'relaxed_phase_T']),
        ('paired_causal_neural', ['thermal_transformer', 'context_thermal_transformer']),
        ('support_blend', ['climate_blend'])]:
        path = PRIOR / folder / name
        if not path.exists():
            if folder == 'paired_causal_neural' and case['n_sites'] != 80:
                continue
            raise ValueError(f'Missing archived prediction extension: {path}')
        other = fitted(path)
        # Some followups retain fewer metadata columns; exact IDs and targets in
        # the base archive remain authoritative.
        if set(other.label_id) != set(f.label_id):
            raise ValueError('Changed target IDs in prior prediction extension')
        f = f.merge(other[['label_id'] + columns], on='label_id', validate='one_to_one')
    return f.rename(columns={'relaxed_T': 'nonlinear_T', 'relaxed_phase_T': 'phase_nonlinear_T', 'climate_blend': 'support_blend'})


def prior_models(frame):
    names = MODELS + ['calendar', 'context_T', 'restricted_nonlinear_T', 'phase_T', 'phase_nonlinear_T',
        'guarded_RF', 'context_thermal_transformer', 'residual_MLP_init29', 'residual_MLP_init43']
    return [m for m in names if m in frame]


def relevant_pairs(models):
    pairs = [('nonlinear_T', 'T'), ('residual_MLP', 'nonlinear_T'), ('thermal_transformer', 'nonlinear_T'),
        ('residual_RF', 'nonlinear_T'), ('support_blend', 'nonlinear_T'), ('nonlinear_T', 'calendar'),
        ('context_thermal_transformer', 'thermal_transformer'), ('nonlinear_T', 'phase_nonlinear_T'),
        ('context_T', 'T'), ('phase_T', 'T'), ('guarded_RF', 'residual_RF'),
        ('residual_MLP_init29', 'residual_MLP'), ('residual_MLP_init43', 'residual_MLP')]
    return [(a, b) for a, b in pairs if a in models and b in models]


def benchmark_outputs(native, paired, components, audit, replicates=REPLICATES, seed=SEED):
    dimensions = ['study', 'period', 'recipe', 'source_zone']
    averaged = average_draws(native, dimensions + ['model'])
    summary, stages, draws = [], [], []
    for keys, q in averaged.groupby(dimensions + ['model']):
        info = dict(zip(dimensions + ['model'], keys))
        summary.append(dict(**info, **metric_summary(q, replicates, seed)))
        for head, h in q.groupby('head'):
            stages.append(dict(**info, head=int(head), stage=STAGES[head], bbch=BBCH[head], **metric_summary(h, replicates, seed, [head])))
    for keys, q in native.groupby(dimensions + ['model', 'draw']):
        info = dict(zip(dimensions + ['model', 'draw'], keys))
        # No bootstrap is used for the three source-draw point estimates.
        draws.append(dict(**info, mae=summarize(q, replicates=1)['value'], bias=summarize(q, 'bias', replicates=1)['value']))
    p = average_draws(paired, dimensions + ['left', 'right'])
    pair_group_keys = dimensions + ['left', 'right']
    draw_points = paired.groupby(pair_group_keys+['draw', 'head']).error.mean().groupby(level=pair_group_keys+['draw']).mean()
    draw_grouped = {key: group.droplevel(pair_group_keys) for key, group in draw_points.groupby(level=pair_group_keys)}
    contrasts = []
    structural = []
    for keys, q in p.groupby(dimensions + ['left', 'right']):
        info = dict(zip(dimensions + ['left', 'right'], keys))
        contrasts.extend(contrast_rows(q, info, replicates, seed))
        v = metric_summary(q, replicates, seed)
        draw_means = draw_grouped[keys]
        left_error = summarize(q, 'left_error', replicates=1)['value']
        right_error = summarize(q, 'right_error', replicates=1)['value']
        structural.append(dict(**info, source_sites=int(info['study'].rsplit('_', 1)[-1]), source_years_per_site=8,
            left_error=left_error, right_error=right_error, difference=v['mae'], ci_low=v['ci_low'], ci_high=v['ci_high'],
            bias_difference=v['bias'], bias_ci_low=v['bias_ci_low'], bias_ci_high=v['bias_ci_high'],
            source_draw_sd=float(draw_means.std()), source_draw_min=float(draw_means.min()), source_draw_max=float(draw_means.max()),
            left_relative_reduction_percent=float(100*(right_error-left_error)/right_error),
            sites=q.site.nunique(), heads=v['heads'], blocks=q.block.nunique(),
            interval_excludes_zero=bool(v['ci_high'] < 0 or v['ci_low'] > 0)))
    c = average_draws(components, dimensions + ['model'])
    correction = []
    for keys, q in c.groupby(dimensions + ['model']):
        row = dict(zip(dimensions + ['model'], keys))
        for metric in ['benefit', 'harm', 'net_gain', 'improved_fraction', 'harmed_fraction']:
            v = summarize(q, metric, range(7), replicates, seed)
            row.update({metric: v['value'], metric+'_ci_low': v['ci_low'], metric+'_ci_high': v['ci_high']})
        correction.append(row)
    draw_table = pd.DataFrame(draws)
    variation = draw_table.groupby(dimensions+['model']).agg(source_draw_sd=('mae', 'std'),
        source_draw_min=('mae', 'min'), source_draw_max=('mae', 'max'), source_draw_bias_sd=('bias', 'std')).reset_index()
    summary_table = pd.DataFrame(summary).merge(variation, on=dimensions+['model'], validate='one_to_one')
    return {'benchmark.csv': summary_table, 'stage_performance.csv': pd.DataFrame(stages),
        'benchmark_sites.csv.gz': averaged, 'benchmark_draw_sites.csv.gz': native,
        'benchmark_draws.csv': draw_table, 'contrasts.csv': pd.DataFrame(contrasts),
        'paired_sites.csv.gz': p, 'paired_draw_sites.csv.gz': paired,
        'correction_summary.csv': pd.DataFrame(correction), 'correction_sites.csv.gz': c,
        'support_audit.csv': pd.DataFrame(audit), 'structural_contrasts.csv': pd.DataFrame(structural)}


def insight_tables(replicates=REPLICATES, seed=SEED):
    cases = configuration(PRIOR / 'cases.json')
    sites, paired, components, audit = [], [], [], []
    config = pd.DataFrame(cases)
    for (fold, draw, budget), group in config.groupby(['fold', 'draw_seed', 'n_sites']):
        frames = []
        for record_case in group.to_dict('records'):
            # JSON null becomes NaN in the intermediate DataFrame.
            source = record_case['source_ez']
            source = -1 if pd.isna(source) else int(source)
            f = old_predictions(record_case)
            models = prior_models(f)
            study = f'prior_pooled_{budget}' if source < 0 else f'prior_geographic_{budget}'
            info = dict(study=study, recipe=record_case['recipe'], draw=int(draw), fold=int(fold), source_zone=source)
            audit.extend(completion_audit(f, models, info))
            for period, q in f.groupby('period'):
                metadata = dict(**info, period=period)
                for model in models:
                    sites.append(model_sites(q, model).assign(model=model, **metadata))
                for left, right in relevant_pairs(models):
                    paired.append(paired_sites(q, left, right).assign(left=left, right=right, **metadata))
                for model in MODELS[2:]:
                    if model in q:
                        components.append(correction_components(q, 'nonlinear_T', model).assign(model=model, **metadata))
            if source < 0:
                frames.append((record_case['recipe'], f))
        if len(frames) == 2:
            recipes = dict(frames)
            first, second = recipes['coverage'], recipes['random']
            for period, q in first.groupby('period'):
                rr = second[second.period.eq(period)]
                info = dict(study=f'prior_pooled_{budget}', period=period, recipe='coverage_minus_random',
                    draw=int(draw), fold=int(fold), source_zone=-1)
                for model in prior_models(q):
                    paired.append(paired_sites(q, model, model, rr).assign(left=model, right=model, **info))
        print(f'Prior rescoring fold {fold}, draw {draw}, source sites {budget}', flush=True)
    # The annual pooled-source benchmark is marginal ordinary MAE, separate from
    # its common-valid source-zone/target-zone comparison.
    annual_cases = [c for c in configuration(ANNUAL / 'cases.json') if c['source_zone'] == -1]
    for case in annual_cases:
        f = fitted(ANNUAL / 'cases' / case['case'])
        info = dict(study='annual_pooled_40', recipe='random', draw=case['draw'], fold=case['fold'], source_zone=-1)
        for period, q in f.groupby('period'):
            metadata = dict(**info, period=period)
            for model in MODELS:
                sites.append(model_sites(q, model).assign(model=model, **metadata))
            for left, right in relevant_pairs(MODELS):
                paired.append(paired_sites(q, left, right).assign(left=left, right=right, **metadata))
            for model in MODELS[2:]:
                components.append(correction_components(q, 'nonlinear_T', model).assign(model=model, **metadata))
        audit.extend(completion_audit(f, MODELS, info))
    return benchmark_outputs(pd.concat(sites, ignore_index=True), pd.concat(paired, ignore_index=True),
        pd.concat(components, ignore_index=True), audit, replicates, seed)


def local_tables(replicates=REPLICATES, seed=SEED):
    f = read_csv(ANNUAL / 'insights/local_predictions.csv.gz', ANNUAL / 'insights/verification.json')
    f['period'] = 'later_held'
    models = [m for m in ['T', 'nonlinear_T', 'context_T', 'phase_T', 'residual_RF', 'residual_MLP',
        'thermal_transformer', 'context_thermal_transformer', 'local_thermal_T'] if m in f]
    if f.duplicated(['label_id', 'budget']).any() or set(f.budget) != {0, 1, 3, 5}:
        raise ValueError('Invalid archived local-history support')
    zero = f[f.budget.eq(0)].copy()
    sites, summary, stages, contrasts, pair_sites, draws, audit = [], [], [], [], [], [], []
    for budget, q in f.groupby('budget'):
        if len(q) != 49462 or q.site.nunique() != 822 or not q.year.gt(2010).all():
            raise ValueError('Changed local-history target population')
        aligned_reference(q, zero)
        audit.extend(completion_audit(q, models, dict(budget=int(budget))))
        for model in models:
            s = model_sites(q, model).assign(budget=int(budget), model=model)
            sites.append(s)
            summary.append(dict(budget=int(budget), model=model, labels=len(q), **metric_summary(s, replicates, seed)))
            for head, h in s.groupby('head'):
                stages.append(dict(budget=int(budget), model=model, head=int(head), bbch=BBCH[head], stage=STAGES[head],
                    **metric_summary(h, replicates, seed, [head])))
            if budget:
                pair = paired_sites(q, model, model, zero).assign(budget=int(budget), model=model, comparison='adapted_minus_zero')
                pair_sites.append(pair)
                v = metric_summary(pair, replicates, seed)
                v['difference'] = v.pop('mae')
                add_bias_difference(v)
                contrasts.append(dict(model=model, budget=int(budget), comparison='adapted_minus_zero', **v))
                samples = conditional_samples(pair, 'error', range(7), replicates, seed)
                draws.append(pd.DataFrame(dict(model=model, budget=int(budget), comparison='adapted_minus_zero',
                    bootstrap_draw=np.arange(replicates), difference=samples,
                    bias_difference=conditional_samples(pair, 'bias', range(7), replicates, seed))))
        for model in models:
            if model == 'nonlinear_T':
                continue
            pair = paired_sites(q, model, 'nonlinear_T').assign(budget=int(budget), model=model, comparison='model_minus_nonlinear_T')
            pair_sites.append(pair)
            v = metric_summary(pair, replicates, seed)
            v['difference'] = v.pop('mae')
            add_bias_difference(v)
            contrasts.append(dict(model=model, budget=int(budget), comparison='model_minus_nonlinear_T', **v))
    paired = pd.concat(pair_sites, ignore_index=True)
    gain = paired[paired.comparison.eq('adapted_minus_zero') & paired.budget.eq(5)].copy()
    gain = gain.rename(columns={'left_error': 'error_five', 'right_error': 'error_zero'})
    gain['gain_days'] = -gain.error
    return {'local_sites.csv.gz': pd.concat(sites, ignore_index=True), 'local_summary.csv': pd.DataFrame(summary),
        'local_stage_performance.csv': pd.DataFrame(stages), 'local_contrasts.csv': pd.DataFrame(contrasts),
        'local_paired_sites.csv.gz': paired, 'local_gain_pairs.csv.gz': gain,
        'local_draws.csv.gz': pd.concat(draws, ignore_index=True), 'local_support_audit.csv': pd.DataFrame(audit)}


def connected_weather_blocks(frame, stations):
    """Connected components of shared native blocks and weather stations."""
    geography = frame[['site', 'block']].drop_duplicates().merge(stations[['site', 'station']].drop_duplicates(),
        on='site', how='left', validate='one_to_one')
    if geography.station.isna().any():
        raise ValueError('Target site lacks weather station')
    parent = {str(b): str(b) for b in geography.block.unique()}
    def find(x):
        x = str(x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for _, group in geography.groupby('station'):
        blocks = sorted(group.block.astype(str).unique())
        root = find(blocks[0])
        for block in blocks[1:]:
            a, b = sorted([find(root), find(block)])
            parent[b] = a
            root = a
    mapping = geography.assign(connected=lambda x: x.block.astype(str).map(lambda b: 'WEATHER_GROUP_'+find(b)))
    result = frame.copy()
    result['block'] = result.site.map(mapping.set_index('site').connected)
    return result


def weather_estimate(frame, stations, replicates=REPLICATES, seed=SEED):
    native = metric_summary(frame, replicates, seed)
    connected = connected_weather_blocks(frame, stations)
    result = metric_summary(connected, replicates, seed)
    result.update(native_block_ci_low=native['ci_low'], native_block_ci_high=native['ci_high'],
        native_block_bias_ci_low=native['bias_ci_low'], native_block_bias_ci_high=native['bias_ci_high'],
        target_clusters=connected.block.nunique(), native_blocks=frame.block.nunique())
    return result


def weather_tables(replicates=REPLICATES, seed=SEED):
    cases = configuration(ANNUAL / 'weather/paired_design/cases.json')
    stations = read_csv(ANNUAL / 'weather/common_units.csv.gz')[['site', 'station']].drop_duplicates()
    sites, paired, audit = [], [], []
    for case in cases:
        predictions = {}
        for train, test in itertools.product(['dwd', 'era5'], repeat=2):
            # Saved fixed-parameter swaps are used directly; no network is loaded.
            path = ANNUAL / 'weather/analysis/predictions' / train / test / f'{case["case"]}.csv.gz'
            f = read_csv(path)
            if f.label_id.duplicated().any():
                raise ValueError('Duplicate weather prediction target')
            # Matched products additionally retain a complete fit receipt.
            if train == test:
                saved = fitted(ANNUAL / 'weather/refits' / train / 'cases' / case['case'])
                rr = aligned_reference(f, saved)
                for model in MODELS:
                    if not np.array_equal(f[model].to_numpy(), rr[model].to_numpy(), equal_nan=True):
                        raise ValueError('Saved matched-weather predictions changed')
            predictions[train, test] = f
            info = dict(training_weather=train, prediction_weather=test, fold=case['fold'], draw=case['draw'])
            audit.extend(completion_audit(f, MODELS, info))
            for period, q in f.groupby('period'):
                for model in MODELS:
                    sites.append(model_sites(q, model).assign(period=period, model=model, **info))
        comparisons = [('fixed_era5_swap_dwd', ('era5', 'dwd'), ('era5', 'era5')),
            ('fixed_dwd_swap_era5', ('dwd', 'era5'), ('dwd', 'dwd')),
            ('matched_dwd_minus_era5', ('dwd', 'dwd'), ('era5', 'era5'))]
        for name, left, right in comparisons:
            first, second = predictions[left], predictions[right]
            for period, q in first.groupby('period'):
                rr = second[second.period.eq(period)]
                for model in MODELS:
                    paired.append(paired_sites(q, model, model, rr).assign(period=period, model=model,
                        comparison=name, fold=case['fold'], draw=case['draw']))
        print(f'Weather rescoring {case["case"]}', flush=True)
    sites, paired = pd.concat(sites, ignore_index=True), pd.concat(paired, ignore_index=True)
    averaged = average_draws(sites, ['period', 'model', 'training_weather', 'prediction_weather'])
    paired_average = average_draws(paired, ['period', 'model', 'comparison'])
    summaries, contrasts = [], []
    for keys, q in averaged.groupby(['period', 'model', 'training_weather', 'prediction_weather']):
        info = dict(zip(['period', 'model', 'training_weather', 'prediction_weather'], keys))
        summaries.append(dict(**info, **weather_estimate(q, stations, replicates, seed)))
    for keys, q in paired_average.groupby(['period', 'model', 'comparison']):
        info = dict(zip(['period', 'model', 'comparison'], keys))
        row = weather_estimate(q, stations, replicates, seed)
        row['difference'] = row.pop('mae')
        add_bias_difference(row)
        contrasts.append(dict(**info, **row))
    tables = {'site_errors.csv.gz': sites, 'paired_site_differences.csv.gz': paired,
        'averaged_site_errors.csv.gz': averaged, 'averaged_paired_site_differences.csv.gz': paired_average,
        'summary.csv': pd.DataFrame(summaries), 'paired_contrasts.csv': pd.DataFrame(contrasts),
        'support_audit.csv': pd.DataFrame(audit)}
    # Meteorological agreement never used a failure loss; its frozen daily input
    # diagnostic remains valid and is preserved with its original provenance.
    tables['weather_agreement.csv'] = read_csv(ANNUAL/'weather/analysis/weather_agreement.csv',
        ANNUAL/'weather/analysis/verification.json')
    return tables


def map_coordinates():
    import shapefile
    from matplotlib.path import Path as PolygonPath
    geo = read_csv(ANNUAL / 'site_zones.csv')
    basemap = ROOT / 'code/research/unified_transfer/assets/naturalearth/ne_50m_admin_0_countries.shp'
    for path in basemap.parent.glob(basemap.stem+'.*'):
        record(path)
    locations = geo[['weather_longitude', 'weather_latitude']].to_numpy()
    winding = np.zeros(len(geo), dtype=int)
    found = False
    with shapefile.Reader(str(basemap)) as reader:
        for item in reader.iterShapeRecords():
            if item.record.as_dict().get('ADMIN') != 'Germany':
                continue
            found = True
            points = np.asarray(item.shape.points)
            parts = list(item.shape.parts)+[len(points)]
            for start, end in zip(parts[:-1], parts[1:]):
                ring = points[start:end]
                # ESRI outer rings are clockwise; holes are counterclockwise.
                area = np.sum(ring[:-1, 0]*ring[1:, 1]-ring[1:, 0]*ring[:-1, 1])
                inside = PolygonPath(ring, closed=True).contains_points(locations, radius=1e-10)
                winding += (-1 if area > 0 else 1)*inside
    if not found:
        raise ValueError('Archived Germany boundary missing')
    geo['in_germany'] = winding != 0
    geo['map_display_country'] = 'Germany'
    return geo


def map_cells(frame, coordinates, dimensions=()):
    dimensions = list(dimensions)
    keys = dimensions + ['head', 'model', 'site']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate site/stage in map aggregation')
    geo_columns = ['canonical_site_id', 'weather_cell', 'weather_latitude', 'weather_longitude', 'in_germany']
    q = frame.merge(coordinates[geo_columns], left_on='site', right_on='canonical_site_id', how='left', validate='many_to_one')
    if q.weather_cell.isna().any():
        raise ValueError('Missing map coordinates')
    group = dimensions + ['head', 'model', 'weather_cell', 'weather_latitude', 'weather_longitude', 'block', 'in_germany']
    metrics = [c for c in ['error', 'bias', 'penalty', 'diagonal_error', 'diagonal_bias', 'completion',
        'reference_completion', 'common_valid_fraction'] if c in q]
    means = q.groupby(group, observed=True)[metrics].mean().reset_index()
    counts = q.assign(_valid=np.isfinite(q.error)).groupby(group, observed=True).agg(
        n_sites=('site', 'nunique'), n_valid_sites=('_valid', 'sum')).reset_index()
    completeness = [c for c in ['completion', 'reference_completion'] if c in q]
    failure = q.groupby(group, observed=True)[completeness].min().lt(1-1e-12).any(axis=1).rename('any_unresolved').reset_index()
    return means.merge(counts, on=group, validate='one_to_one').merge(failure, on=group, validate='one_to_one')


def native_map_support(predictions, coordinates):
    q = predictions[predictions.period.eq('later_held') & predictions['head'].isin([3, 6])]
    q = q.merge(coordinates[['canonical_site_id', 'weather_cell']], left_on='site', right_on='canonical_site_id', validate='many_to_one')
    return q.groupby(['head', 'weather_cell']).agg(n_native_labels=('label_id', 'nunique'),
        n_native_units=('unit', 'nunique'), n_observed_years=('year', 'nunique'),
        support_sites=('site', 'nunique'), year_min=('year', 'min'), year_max=('year', 'max')).reset_index()


def pooled_map_tables(sites, pairs, predictions, coordinates):
    selected = sites[sites.period.eq('later_held') & sites['head'].isin([3, 6]) & sites.model.isin(['T', 'nonlinear_T', 'residual_MLP'])]
    support = native_map_support(predictions, coordinates)
    cells = map_cells(selected, coordinates).merge(support, on=['head', 'weather_cell'], validate='many_to_one')
    pair = pairs[pairs.period.eq('later_held') & pairs['head'].isin([3, 6]) & pairs.left.eq('residual_MLP') & pairs.right.eq('nonlinear_T')].copy()
    pair['model'] = 'residual_MLP'
    differences = map_cells(pair, coordinates).merge(support, on=['head', 'weather_cell'], validate='many_to_one')
    panels = []
    for row, head in enumerate([3, 6]):
        for column, model in enumerate(['T', 'nonlinear_T']):
            q = cells[cells['head'].eq(head) & cells.model.eq(model)].copy()
            panels.append(q.assign(value=q.error, panel='('+chr(97+row*3+column)+')', metric='mae', reference_model='',
                bbch=BBCH[head], display_min=0., display_max=30.))
        q = differences[differences['head'].eq(head)].copy()
        panels.append(q.assign(value=q.error, difference=q.error, panel='('+chr(99+row*3)+')', metric='difference',
            reference_model='nonlinear_T', bbch=BBCH[head], display_min=-10., display_max=10.))
    result = pd.concat(panels, ignore_index=True)
    return {'pooled_cells.csv': cells, 'pooled_map.csv': result, 'spatial_summary.csv': map_summary([('pooled', result)])}


def map_summary(frames):
    rows = []
    for name, frame in frames:
        for panel, q in frame.groupby('panel'):
            rows.append(dict(map=name, panel=panel, model=q.model.iloc[0], head=int(q['head'].iloc[0]),
                cells=len(q), valid_cells=int(q.value.notna().sum()), sites=int(q.n_sites.sum()),
                german_cells=int(q.in_germany.sum()), minimum=q.value.min(), p10=q.value.quantile(.1),
                median=q.value.median(), p90=q.value.quantile(.9), maximum=q.value.max(),
                negative_cells=int(q.value.lt(0).sum()), positive_cells=int(q.value.gt(0).sum()),
                unresolved_cells=int(q.any_unresolved.sum()),
                clipped_low_cells=int(q.value.lt(q.display_min).sum()), clipped_high_cells=int(q.value.gt(q.display_max).sum())))
    return pd.DataFrame(rows)


def spatial_tables(output):
    output = Path(output)
    geo = map_coordinates()
    primary = {name: pd.read_csv(output/'primary'/name) for name in ['benchmark_sites.csv.gz', 'paired_sites.csv.gz', 'keyed_predictions.csv.gz']}
    p = pooled_map_tables(primary['benchmark_sites.csv.gz'], primary['paired_sites.csv.gz'], primary['keyed_predictions.csv.gz'], geo)
    write_tables(output/'primary', p)
    old_sites = pd.read_csv(output/'insights/benchmark_sites.csv.gz')
    old_pairs = pd.read_csv(output/'insights/paired_sites.csv.gz')
    old_sites = old_sites[old_sites.study.eq('prior_pooled_80') & old_sites.recipe.eq('random')]
    old_pairs = old_pairs[old_pairs.study.eq('prior_pooled_80') & old_pairs.recipe.eq('random')]
    old_predictions = pd.concat([fitted(PRIOR/'cases'/f'f{fold}_random_n80_d17') for fold in range(5)], ignore_index=True)
    result = pooled_map_tables(old_sites, old_pairs, old_predictions, geo)
    del result['spatial_summary.csv']
    zone = pd.read_csv(output/'site_transfer_errors.csv.gz')
    q = zone[zone.period.eq('later_held') & zone['head'].eq(6) & zone.source_zone.isin([0, 4]) &
        zone.model.isin(['nonlinear_T', 'residual_MLP', 'support_blend'])]
    values = ['error', 'bias', 'penalty', 'diagonal_error', 'diagonal_bias', 'completion', 'reference_completion', 'common_valid_fraction']
    q = q.groupby(['source_zone', 'head', 'model', 'site', 'block'])[values].mean().reset_index()
    cells = map_cells(q, geo, ['source_zone'])
    cells = cells.merge(native_map_support(old_predictions, geo), on=['head', 'weather_cell'], validate='many_to_one')
    panels = []
    for row, zone in enumerate([0, 4]):
        for col, model in enumerate(['nonlinear_T', 'residual_MLP', 'support_blend']):
            g = cells[cells.source_zone.eq(zone) & cells.model.eq(model)]
            panels.append(g.assign(value=g.penalty, panel='('+chr(97+row*3+col)+')', metric='penalty', bbch=87,
                display_min=-30., display_max=30.))
    result['transfer_cells.csv'] = cells
    result['transfer_map.csv'] = pd.concat(panels, ignore_index=True)
    result['summary.csv'] = map_summary([('pooled', result['pooled_map.csv']), ('transfer', result['transfer_map.csv'])])
    result['site_zones.csv'] = geo
    return result


def copy_study_network(output):
    source = ROOT/'data/evidence/spatial/study_network.csv'
    record(source)
    destination = Path(output)/'spatial/study_network.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def structural_tables(output, insight=None):
    """Original manuscript's 24 structural contrasts, plus their ordinary sites."""
    output = Path(output)
    if insight is None:
        contrast = pd.read_csv(output/'insights/structural_contrasts.csv')
        sites = pd.read_csv(output/'insights/benchmark_sites.csv.gz', usecols=
            ['study', 'period', 'recipe', 'model', 'head', 'site', 'block', 'error', 'bias'])
    else:
        contrast, sites = insight['structural_contrasts.csv'], insight['benchmark_sites.csv.gz']
    included = []
    for study, extra in [('prior_pooled_80', ('nonlinear_T', 'phase_nonlinear_T')),
        ('annual_pooled_40', ('support_blend', 'nonlinear_T'))]:
        pairs = [('nonlinear_T', 'T'), ('residual_MLP', 'nonlinear_T'), ('thermal_transformer', 'nonlinear_T'), extra]
        for left, right in pairs:
            included.append(contrast[contrast.study.eq(study) & contrast.recipe.isin(['random', 'coverage']) &
                contrast.left.eq(left) & contrast.right.eq(right)])
    result = pd.concat(included, ignore_index=True)
    if len(result) != 24:
        raise ValueError('Incomplete manuscript structural comparisons')
    site = sites[sites.study.isin(['prior_pooled_80', 'annual_pooled_40']) & sites.recipe.isin(['random', 'coverage']) &
        sites.model.isin(['T', 'nonlinear_T', 'phase_nonlinear_T', 'residual_MLP', 'thermal_transformer', 'support_blend'])].copy()
    site['source_sites'] = site.study.map({'prior_pooled_80': 80, 'annual_pooled_40': 40})
    return {'structural_contrasts.csv': result, 'structural_site_errors.csv.gz': site}


def verification(output, sections, replicates=REPLICATES, seed=SEED):
    for path, expected in INPUTS.items():
        if sha(path) != expected:
            raise ValueError(f'Archived input changed: {path}')
    output = Path(output)
    def portable(path):
        path = Path(path).resolve()
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    files = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file() and p.name != 'verification.json'}
    receipt = dict(complete=True, evaluation_only=True, sections=sections,
        predictions_regenerated=False, fitting=False, source_selection_changed=False,
        experiment_counts=dict(primary_folds=5, primary_models=6, annual_temperature_cases=90,
            prior_cases=80, prior_80_site_paired_neural_cases=30, local_folds=5, local_budgets=[0, 1, 3, 5],
            weather_cases_per_provider=15, frozen_weather_fits=30, weather_saved_prediction_combinations=60, new_fits=0),
        source_selection_objective='Frozen original source training/tuning/selection, including its original invalid-date penalty where applicable',
        metric='Ordinary MAE and signed predicted-minus-observed bias over valid dates only; no substituted failure loss',
        valid_prediction='finite, nonnegative, strictly less than the observation-specific weather horizon',
        failure14='Fraction of valid dates with absolute error greater than 14 days; unresolved dates counted separately',
        weighting='observations within site-year; valid years within site; sites within stage; equal seven stages; source draws averaged after within-draw grouping',
        comparison_support='intersection of valid native observation IDs for both candidates before site-year and site aggregation',
        zero_valid_policy='retain all observed support and failure counts; error and bias undefined for empty valid stage/site support',
        interval=dict(replicates=int(replicates), seed=int(seed), confidence=.95, conditional_on_frozen_fits=True,
            whole_target_blocks=True, paired=True, source_draws_averaged_before_resampling=True,
            weather='Connected shared target station/native-block clusters; native-block sensitivity retained',
            missing_stage_replicate='undefined; interval undefined if fewer than 95% replicates retain all required stages'),
        map_policy='Full observed raw cohort retained with in_germany flags; figures display Germany only',
        support_counts='Single-fit counts are exact; counts after source-draw averaging are mean counts per draw, with exact counts retained in support_audit and draw-site files',
        input_sha256={portable(p): digest for p, digest in INPUTS.items()}, output_sha256=files,
        code_sha256={portable(p): sha(p) for p in [Path(__file__), Path(__file__).parent/'__init__.py',
            ROOT/'code/research/phenology_exploration/core.py', ROOT/'code/research/full_source_benchmark/analysis.py']})
    (output/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def source_manifest():
    """Validate receipted frozen prediction sources without recomputing results."""
    def prediction(folder):
        folder = Path(folder)
        receipt = folder/'complete.json'
        record(receipt)
        digest = json.loads(receipt.read_text())['outputs']['predictions.csv.gz']
        if record(folder/'predictions.csv.gz') != digest:
            raise ValueError(f'Changed archived prediction source: {folder}')
    for fold in range(5):
        prediction(PRIMARY/f'fold{fold}')
    for case in configuration(PRIOR/'cases.json'):
        for folder in ['cases', 'response_followup', 'support_blend']:
            prediction(PRIOR/folder/case['case'])
        if case['n_sites'] == 80:
            prediction(PRIOR/'paired_causal_neural'/case['case'])
    for case in configuration(ANNUAL/'cases.json'):
        prediction(ANNUAL/'cases'/case['case'])
    for case in configuration(ANNUAL/'weather/paired_design/cases.json'):
        for train, test in itertools.product(['dwd', 'era5'], repeat=2):
            record(ANNUAL/'weather/analysis/predictions'/train/test/f'{case["case"]}.csv.gz')
            if train == test:
                prediction(ANNUAL/'weather/refits'/train/'cases'/case['case'])
    path = ANNUAL/'insights/local_predictions.csv.gz'
    receipt = ANNUAL/'insights/verification.json'
    record(receipt)
    if record(path) != json.loads(receipt.read_text())['output_sha256'][path.name]:
        raise ValueError('Changed local-history predictions')
    for path in [ANNUAL/'weather/common_units.csv.gz', ANNUAL/'site_zones.csv',
        ANNUAL/'weather/analysis/weather_agreement.csv', ANNUAL/'weather/analysis/verification.json',
        ROOT/'data/evidence/spatial/study_network.csv']:
        record(path)
    basemap = ROOT/'code/research/unified_transfer/assets/naturalearth/ne_50m_admin_0_countries.shp'
    for path in basemap.parent.glob(basemap.stem+'.*'):
        record(path)


def audit_outputs(output, replicates=REPLICATES, seed=SEED):
    """Numerical identities and all-observation audits on completed new evidence."""
    output = Path(output)
    results = {}
    primary = pd.read_csv(output/'primary/keyed_predictions.csv.gz')
    summary = pd.read_csv(output/'primary/summary.csv')
    largest, largest_interval = 0., 0.
    for period, q in primary.groupby('period'):
        for model in MODELS:
            pred = q[model].to_numpy()
            valid = np.isfinite(pred)&(pred>=0)&(pred<q.horizon.to_numpy())
            raw = q[SITE_KEYS+['site_year']].copy()
            raw['mae'] = np.where(valid, abs(pred-q.observed.to_numpy()), np.nan)
            raw['bias'] = np.where(valid, pred-q.observed.to_numpy(), np.nan)
            # An independent groupby route reconstructs the complete hierarchy.
            years = raw.groupby(['head', 'site', 'block', 'site_year'])[['mae', 'bias']].mean()
            sites = years.groupby(level=['head', 'site', 'block']).mean()
            point = sites.groupby('head').mean().mean()
            row = summary[summary.period.eq(period)&summary.model.eq(model)].iloc[0]
            largest = max(largest, abs(point.mae-row.mae), abs(point.bias-row.bias))
            native_sites = sites.reset_index()
            for column, low, high in [('mae', 'ci_low', 'ci_high'), ('bias', 'bias_ci_low', 'bias_ci_high')]:
                samples = conditional_samples(native_sites, column, range(7), replicates, seed)
                complete = np.isfinite(samples)
                interval = np.quantile(samples[complete], [.025, .975]) if complete.mean() >= .95 else [np.nan, np.nan]
                if not np.allclose(interval, [row[low], row[high]], atol=1e-10, rtol=0, equal_nan=True):
                    raise ValueError('Primary conditional interval does not reconstruct from ordinary errors')
                largest_interval = max(largest_interval, float(np.nanmax(abs(np.asarray(interval)-[row[low], row[high]]))))
            if row.n_native_labels != len(q) or row.n_valid_labels != valid.sum() or row.n_failed_labels != (~valid).sum():
                raise ValueError('Primary observed/valid/failure counts do not reconstruct')
    if largest > 1e-10:
        raise ValueError('Primary ordinary MAE/bias reconstruction failed')
    results['primary_max_reconstruction_error_days'] = largest
    results['primary_max_interval_reconstruction_error_days'] = largest_interval
    zone = pd.read_csv(output/'site_transfer_errors.csv.gz', usecols=['source_zone', 'target_zone', 'error', 'diagonal_error',
        'penalty', 'bias', 'diagonal_bias', 'bias_difference', 'n_native_labels', 'n_common_valid_labels', 'n_excluded_labels'])
    for left, right, difference in [('error', 'diagonal_error', 'penalty'), ('bias', 'diagonal_bias', 'bias_difference')]:
        error = (zone[left]-zone[right]-zone[difference]).abs().max()
        if error > 1e-10:
            raise ValueError('Paired zone identities failed')
        results[f'zone_{difference}_algebra_max_error_days'] = error
    if not np.allclose(zone[zone.source_zone.eq(zone.target_zone)].penalty.dropna(), 0):
        raise ValueError('Nonzero diagonal common-valid MAE difference')
    if not zone.n_native_labels.eq(zone.n_common_valid_labels+zone.n_excluded_labels).all():
        raise ValueError('Zone support counts lose failed observations')
    results['zone_site_stage_source_model_rows'] = len(zone)
    results['zone_undefined_common_support_rows'] = int(zone.error.isna().sum())
    for folder, filename in [('primary', 'correction_sites.csv.gz'), ('insights', 'correction_sites.csv.gz')]:
        q = pd.read_csv(output/folder/filename, usecols=['benefit', 'harm', 'net_gain'])
        delta = (q.benefit-q.harm-q.net_gain).abs().max()
        if delta > 1e-10:
            raise ValueError('Gross benefit/harm does not reconstruct net gain')
        results[folder+'_benefit_harm_max_error_days'] = delta
    for folder, filename in [('primary', 'support_audit.csv'), ('', 'support_audit.csv'),
        ('insights', 'support_audit.csv'), ('insights', 'local_support_audit.csv'), ('weather/analysis', 'support_audit.csv')]:
        q = pd.read_csv(output/folder/filename)
        if not q.labels.eq(q.valid_labels+q.failed_labels).all():
            raise ValueError('Completion audit silently loses observed targets')
    results['all_observed_targets_retained_in_failure_audits'] = True
    def raw_pair_check(first, second, model_a, model_b, serialized):
        b = second.set_index('label_id').loc[first.label_id]
        aa, bb = first[model_a].to_numpy(), b[model_b].to_numpy()
        common = (np.isfinite(aa)&(aa>=0)&(aa<first.horizon.to_numpy()) &
            np.isfinite(bb)&(bb>=0)&(bb<b.horizon.to_numpy()))
        observed = first.observed.to_numpy()
        raw = first[SITE_KEYS+['site_year']].copy()
        raw['error'] = np.where(common, abs(aa-observed)-abs(bb-observed), np.nan)
        raw['bias'] = np.where(common, aa-bb, np.nan)
        raw['n_common_valid_labels'] = common.astype(int)
        expected = raw.groupby(SITE_KEYS+['site_year'])[['error', 'bias']].mean().groupby(level=SITE_KEYS).mean()
        counts = raw.groupby(SITE_KEYS).n_common_valid_labels.sum()
        actual = serialized.set_index(SITE_KEYS).sort_index()
        expected = expected.sort_index()
        if not actual.index.equals(expected.index):
            raise ValueError('Serialized paired sites lose observed support')
        for name in ['error', 'bias']:
            if not np.allclose(expected[name], actual[name], equal_nan=True, atol=1e-10, rtol=0):
                raise ValueError('Paired observation intersection failed independent reconstruction')
        if not counts.reindex(actual.index).eq(actual.n_common_valid_labels).all():
            raise ValueError('Common-valid label counts differ from original observed labels')
        return float(np.nanmax(abs(expected[['error', 'bias']].to_numpy()-actual[['error', 'bias']].to_numpy())))
    primary_pairs = pd.read_csv(output/'primary/paired_sites.csv.gz')
    a = primary[primary.period.eq('historical_held')]
    p = primary_pairs[primary_pairs.period.eq('historical_held') & primary_pairs.left.eq('residual_MLP') & primary_pairs.right.eq('nonlinear_T')]
    results['primary_common_valid_raw_pair_max_error_days'] = raw_pair_check(a, a, 'residual_MLP', 'nonlinear_T', p)
    local = read_csv(ANNUAL/'insights/local_predictions.csv.gz')
    p = pd.read_csv(output/'insights/local_paired_sites.csv.gz')
    p = p[p.budget.eq(5) & p.model.eq('residual_MLP') & p.comparison.eq('adapted_minus_zero')]
    results['local_common_valid_raw_pair_max_error_days'] = raw_pair_check(local[local.budget.eq(5)],
        local[local.budget.eq(0)], 'residual_MLP', 'residual_MLP', p)
    first = read_csv(ANNUAL/'weather/analysis/predictions/era5/dwd/f0_weather_d17.csv.gz')
    second = read_csv(ANNUAL/'weather/analysis/predictions/era5/era5/f0_weather_d17.csv.gz')
    p = pd.read_csv(output/'weather/analysis/paired_site_differences.csv.gz')
    p = p[p.fold.eq(0) & p.draw.eq(17) & p.period.eq('later_held') & p.model.eq('T') & p.comparison.eq('fixed_era5_swap_dwd')]
    results['weather_common_valid_raw_pair_max_error_days'] = raw_pair_check(first[first.period.eq('later_held')],
        second[second.period.eq('later_held')], 'T', 'T', p)
    return results


def finalize(output=DEFAULT_OUTPUT, replicates=REPLICATES, seed=SEED):
    copy_study_network(output)
    source_manifest()
    checks = audit_outputs(output, replicates, seed)
    receipt = verification(output, ['primary', 'zones', 'insights', 'local', 'weather', 'spatial'], replicates, seed)
    receipt['numerical_verification'] = checks
    (Path(output)/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def verify(evidence_folder, report_path=None, replicates=REPLICATES, seed=SEED):
    """Read-only verification of a frozen evidence tree and its archived sources.

    Returns the report in memory. An optional report destination must be outside
    the evidence tree, so verification cannot alter a canonical frozen archive.
    """
    evidence = Path(evidence_folder).resolve()
    destination = Path(report_path).resolve() if report_path is not None else None
    if destination is not None and destination.is_relative_to(evidence):
        raise ValueError('Verification report must be outside the frozen evidence tree')
    receipt_path = evidence/'verification.json'
    receipt = json.loads(receipt_path.read_text())
    if not receipt.get('complete') or not receipt.get('evaluation_only'):
        raise ValueError('Incomplete ordinary-evaluation receipt')
    for category, paths in [('input_sha256', receipt['input_sha256']), ('code_sha256', receipt['code_sha256'])]:
        for path, digest in paths.items():
            resolved = Path(path) if Path(path).is_absolute() else ROOT/path
            if sha(resolved) != digest:
                raise ValueError(f'Changed frozen {category}: {path}')
    for relative, digest in receipt['output_sha256'].items():
        if sha(evidence/relative) != digest:
            raise ValueError(f'Changed frozen ordinary evaluation: {relative}')
    source_manifest()
    interval = receipt['interval']
    if interval['replicates'] != replicates or interval['seed'] != seed:
        raise ValueError('Requested verification bootstrap differs from frozen evaluation')
    checks = audit_outputs(evidence, replicates, seed)
    report = dict(complete=True, evidence_folder=str(evidence), evidence_receipt_sha256=sha(receipt_path),
        verification_writes_evidence=False, numerical_verification=checks,
        frozen_sources_checked=len(receipt['input_sha256']), frozen_outputs_checked=len(receipt['output_sha256']),
        evaluation_code_checked=len(receipt['code_sha256']), intervals=dict(replicates=replicates, seed=seed))
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2)+'\n')
    return report


def run(output_folder=DEFAULT_OUTPUT, section='all', replicates=REPLICATES, seed=SEED, report_path=None):
    output = Path(output_folder)
    if section == 'verify':
        return verify(output, report_path, replicates, seed)
    if section == 'finalize':
        return finalize(output, replicates, seed)
    selected = ['primary', 'zones', 'insights', 'local', 'weather', 'spatial'] if section == 'all' else [section]
    builders = {'primary': (primary_tables, 'primary'), 'zones': (zone_tables, ''),
        'insights': (insight_tables, 'insights'), 'local': (local_tables, 'insights'),
        'weather': (weather_tables, 'weather/analysis')}
    for name in selected:
        if name == 'spatial':
            write_tables(output/'spatial', spatial_tables(output))
            copy_study_network(output)
        else:
            build, destination = builders[name]
            tables = build(replicates, seed)
            write_tables(output/destination, tables)
            if name == 'insights':
                write_tables(output, structural_tables(output, tables))
        print(f'Completed ordinary MAE/bias section: {name}', flush=True)
    return finalize(output, replicates, seed) if section == 'all' else verification(output, selected, replicates, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--section', choices=['all', 'primary', 'zones', 'insights', 'local', 'weather', 'spatial', 'verify', 'finalize'], default='all')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--replicates', type=int, default=REPLICATES)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    run(args.output, args.section, args.replicates, args.seed, args.report)
