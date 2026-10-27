"""Complete-target evaluation of one all-eligible-source fit per frozen fold.

The estimand follows the archived observation/site-year/site/native-head hierarchy.
Uncertainty conditions on the five fitted models and resamples whole target blocks.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.phenology_exploration.analysis import site_errors
from research.phenology_exploration.core import sha, valid_prediction
from research.phenology_exploration.data import ROOT, OUT as PRIOR, STAGES
from research.annual_temperature_transfer.climate import OUT as ANNUAL
from research.annual_temperature_transfer.spatial import cell_results, cell_difference, HEADS

DEFAULT_INPUT = ROOT / 'outputs/full_source_benchmark_20261026'
MODELS = ['T', 'nonlinear_T', 'residual_RF', 'residual_MLP', 'thermal_transformer', 'support_blend']
CORRECTIONS = MODELS[2:]
METADATA = ['label_id', 'unit', 'u', 'site', 'site_year', 'block', 'year', 'head', 'observed', 'horizon', 'period']
SITE_KEYS = ['head', 'site', 'block']
SEED = 20261023
REPLICATES = 2000


def label_digest(frame):
    """Stable digest of the complete, sorted native observation identity list."""
    return hashlib.sha256('\n'.join(sorted(frame.label_id.astype(str))).encode()).hexdigest()


def validate_targets(frame, expected):
    """Reject changed support or metadata, including an unscored native endpoint."""
    for name, data in [('prediction', frame), ('archive', expected)]:
        missing = set(METADATA) - set(data)
        if missing:
            raise ValueError(f'Missing {name} target metadata: {sorted(missing)}')
        if data.label_id.isna().any() or data.label_id.duplicated().any():
            raise ValueError(f'Duplicate or missing native label IDs in {name}')
    if set(frame.label_id) != set(expected.label_id):
        raise ValueError('Target support differs from frozen archived observations')
    a = frame.set_index('label_id').sort_index()
    b = expected.set_index('label_id').sort_index()
    for name in METADATA[1:]:
        if name in ['u', 'year', 'head', 'observed', 'horizon']:
            equal = np.isclose(a[name].to_numpy(float), b[name].to_numpy(float), atol=1e-9, rtol=0)
        else:
            equal = a[name].eq(b[name]).to_numpy()
        if not equal.all():
            raise ValueError(f'Frozen target {name} changed')


def checked_receipt(folder, inputs):
    path = folder / 'complete.json'
    if not path.is_file() or not (folder / 'predictions.csv.gz').is_file():
        raise ValueError(f'Incomplete frozen fold/case: {folder}')
    receipt = json.loads(path.read_text())
    if receipt.get('complete') is False:
        raise ValueError(f'Incomplete fold receipt: {folder}')
    outputs = receipt.get('outputs', {})
    if 'predictions.csv.gz' not in outputs:
        raise ValueError(f'Incomplete prediction hash receipt: {folder}')
    inputs[str(path)] = sha(path)
    for name, digest in outputs.items():
        output = folder / name
        if not output.is_file() or sha(output) != digest:
            raise ValueError(f'Changed or missing receipted output: {output}')
        inputs[str(output)] = digest
    return receipt


def load_folds(input_folder, reference_folder=PRIOR, folds=range(5)):
    """Load all folds only after exact per-label archived-target verification."""
    input_folder, reference_folder = Path(input_folder), Path(reference_folder)
    frames, receipts, inputs = [], [], {}
    for fold in folds:
        folder = input_folder / f'fold{fold}'
        receipt = checked_receipt(folder, inputs)
        frame = pd.read_csv(folder / 'predictions.csv.gz')
        missing = set(MODELS) - set(frame)
        if missing:
            raise ValueError(f'Missing model predictions in fold {fold}: {sorted(missing)}')
        for model in MODELS:
            frame[model] = pd.to_numeric(frame[model], errors='raise')
        archive = reference_folder / 'cases' / f'f{fold}_random_n80_d17'
        checked_receipt(archive, inputs)
        expected = pd.read_csv(archive / 'predictions.csv.gz', usecols=[c if c != 'period' else 'case' for c in METADATA])
        expected = expected.rename(columns={'case': 'period'})
        validate_targets(frame, expected)
        if set(frame.period) != {'historical_held', 'later_held'}:
            raise ValueError(f'Missing frozen evaluation period in fold {fold}')
        if set(frame['head']) != set(range(7)):
            raise ValueError(f'Missing native endpoint in fold {fold}')
        frame = frame[METADATA + MODELS].assign(fold=int(fold))
        frames.append(frame)
        receipts.append(dict(fold=int(fold), predictions_sha256=sha(folder / 'predictions.csv.gz'),
                             receipt_sha256=sha(folder / 'complete.json'), labels=len(frame),
                             target_label_sha256=label_digest(frame),
                             periods={p: len(q) for p, q in frame.groupby('period')}))
    if not frames:
        raise ValueError('No complete full-source folds')
    frame = pd.concat(frames, ignore_index=True)
    if frame.label_id.duplicated().any():
        raise ValueError('Duplicate held target label IDs across folds')
    if frame.groupby('site').fold.nunique().gt(1).any() or frame.groupby('block').fold.nunique().gt(1).any():
        raise ValueError('A held target site or frozen block crosses outer folds')
    later = frame[frame.period.eq('later_held')]
    if tuple(folds) == tuple(range(5)) and (len(later) != 56447 or later.site.nunique() != 1029):
        raise ValueError('Later target support is not the frozen 56,447 labels / 1,029 sites')
    return frame.sort_values(['fold', 'period', 'label_id']).reset_index(drop=True), inputs, receipts


def native_support(frame):
    """Native counts are never multiplied by models or source draws."""
    return frame.groupby(SITE_KEYS).agg(
        n_native_labels=('label_id', 'nunique'), n_native_units=('unit', 'nunique'),
        n_observed_years=('year', 'nunique'), n_site_years=('site_year', 'nunique'),
        year_min=('year', 'min'), year_max=('year', 'max')).reset_index()


def losses(frame, model):
    prediction = frame[model].to_numpy(float)
    valid = valid_prediction(frame, prediction)
    return np.where(valid, abs(prediction - frame.observed.to_numpy(float)), 366.), valid


def model_sites(frame, model):
    """Keep every outcome, assigning 366 d to each invalid/unresolved date."""
    error, valid = losses(frame, model)
    result = site_errors(frame, error)
    result['completion'] = site_errors(frame, valid.astype(float)).error.to_numpy()
    result['failure14'] = site_errors(frame, (error > 14).astype(float)).error.to_numpy()
    return result.merge(native_support(frame), on=SITE_KEYS, validate='one_to_one')


def conditional_samples(frame, column='error', replicates=REPLICATES, seed=SEED):
    """Paired whole-block draws; replicates missing a native head stay undefined.

    Identical block populations and seeds produce identical multiplicities across
    all candidates, losses and correction components. No models are refitted.
    """
    if frame.empty or replicates < 1:
        raise ValueError('Conditional bootstrap requires targets and positive replicates')
    if frame.duplicated(SITE_KEYS).any():
        raise ValueError('Duplicate native site/head in conditional bootstrap')
    if not np.isfinite(frame[column].to_numpy(float)).all():
        raise ValueError('Nonfinite scored target metric')
    blocks, heads = sorted(frame.block.unique()), sorted(frame['head'].unique())
    totals, counts = [], []
    for head in heads:
        grouped = frame[frame['head'].eq(head)].groupby('block')[column].agg(['sum', 'count']).reindex(blocks).fillna(0)
        totals.append(grouped['sum'].to_numpy())
        counts.append(grouped['count'].to_numpy())
    draw = np.random.default_rng(seed).multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=replicates)
    numerator, denominator = draw @ np.array(totals).T, draw @ np.array(counts).T
    complete = (denominator > 0).all(axis=1)
    values = np.full(replicates, np.nan)
    values[complete] = (numerator[complete] / denominator[complete]).mean(axis=1)
    return values


def summarize(frame, column='error', replicates=REPLICATES, seed=SEED):
    draws = conditional_samples(frame, column, replicates, seed)
    complete = np.isfinite(draws)
    interval = np.quantile(draws[complete], [.025, .975]) if complete.mean() >= .95 else [np.nan, np.nan]
    return dict(value=float(frame.groupby('head')[column].mean().mean()),
                ci_low=float(interval[0]), ci_high=float(interval[1]),
                complete_bootstrap_fraction=float(complete.mean()))


def correction_components(frame, reference, prediction):
    """Decompose paired individual-date error changes before any aggregation."""
    if frame.label_id.duplicated().any():
        raise ValueError('Duplicate native labels in correction comparison')
    gain = losses(frame, reference)[0] - losses(frame, prediction)[0]
    values = dict(benefit=np.maximum(gain, 0), harm=np.maximum(-gain, 0), net_gain=gain,
                  improved_fraction=(gain > 0).astype(float), harmed_fraction=(gain < 0).astype(float))
    result = None
    for name, value in values.items():
        table = site_errors(frame, value).rename(columns={'error': name})
        result = table if result is None else result.merge(table, on=SITE_KEYS, validate='one_to_one')
    return result


def paired_site_difference(frame, left, right):
    a = frame[frame.model.eq(left)][SITE_KEYS + ['error']]
    b = frame[frame.model.eq(right)][SITE_KEYS + ['error']]
    result = a.merge(b, on=SITE_KEYS, how='outer', validate='one_to_one', indicator=True, suffixes=('_left', '_right'))
    if not result['_merge'].eq('both').all():
        raise ValueError('Unmatched native model support in paired contrast')
    result['error'] = result.error_left - result.error_right
    return result[SITE_KEYS + ['error']]


def build_tables(frame, replicates=REPLICATES, seed=SEED):
    """Construct absolute metrics, paired contrasts and benefit/harm summaries."""
    sites, summaries, stages, components, corrections, contrasts = [], [], [], [], [], []
    for period, targets in frame.groupby('period'):
        if set(targets['head']) != set(range(7)):
            raise ValueError('All seven native heads are required for the primary estimand')
        info = dict(study='full_source', recipe='all_eligible', period=period)
        period_sites = []
        counts = dict(sites=int(targets.site.nunique()), labels=len(targets), units=int(targets.unit.nunique()), heads=7)
        for model in MODELS:
            table = model_sites(targets, model).assign(model=model, **info)
            period_sites.append(table)
            value = summarize(table, replicates=replicates, seed=seed)
            summaries.append(dict(**info, model=model, mae=value['value'], ci_low=value['ci_low'], ci_high=value['ci_high'],
                                  completion=float(table.groupby('head').completion.mean().mean()),
                                  failure14=float(table.groupby('head').failure14.mean().mean()), **counts))
            for head, group in table.groupby('head'):
                value = summarize(group, replicates=replicates, seed=seed)
                stages.append(dict(**info, model=model, head=int(head), bbch=[10,31,53,61,75,83,87][int(head)],
                                   stage=STAGES[int(head)], mae=value['value'], ci_low=value['ci_low'], ci_high=value['ci_high'],
                                   completion=float(group.completion.mean()), failure14=float(group.failure14.mean()),
                                   sites=int(group.site.nunique()), labels=int(group.n_native_labels.sum()),
                                   units=int(group.n_native_units.sum())))
        period_sites = pd.concat(period_sites, ignore_index=True)
        sites.append(period_sites)
        for left, right in itertools.combinations(MODELS, 2):
            # Later-listed candidates minus earlier-listed comparators.
            difference = paired_site_difference(period_sites, right, left)
            for head, group in [('all', difference)] + [(str(h), g) for h, g in difference.groupby('head')]:
                value = summarize(group, replicates=replicates, seed=seed)
                contrasts.append(dict(**info, left=right, right=left, head=head, difference=value['value'],
                                      ci_low=value['ci_low'], ci_high=value['ci_high'], sites=int(group.site.nunique())))
        for model in CORRECTIONS:
            component = correction_components(targets, 'nonlinear_T', model).assign(model=model, **info)
            components.append(component)
            row = dict(**info, model=model, reference='nonlinear_T')
            for metric in ['benefit', 'harm', 'net_gain', 'improved_fraction', 'harmed_fraction']:
                value = summarize(component, metric, replicates, seed)
                row.update({metric: value['value'], metric + '_ci_low': value['ci_low'], metric + '_ci_high': value['ci_high']})
            if not np.isclose(row['benefit'] - row['harm'], row['net_gain'], atol=1e-10):
                raise ValueError('Benefit minus harm does not reconstruct paired net gain')
            corrections.append(row)
    return {'summary.csv': pd.DataFrame(summaries), 'stage_performance.csv': pd.DataFrame(stages),
            'benchmark_sites.csv.gz': pd.concat(sites, ignore_index=True),
            'correction_summary.csv': pd.DataFrame(corrections), 'correction_sites.csv.gz': pd.concat(components, ignore_index=True),
            'paired_contrasts.csv': pd.DataFrame(contrasts)}


def pooled_map_tables(sites, predictions, coordinates):
    """BBCH 61/87 maps retain native sites and recorded weather-cell centres."""
    native = predictions[predictions.period.eq('later_held') & predictions['head'].isin(HEADS)].copy()
    selected = sites[sites.period.eq('later_held') & sites['head'].isin(HEADS) & sites.model.isin(['T', 'nonlinear_T', 'residual_MLP'])].copy()
    if set(map(tuple, native[['head', 'site']].drop_duplicates().to_numpy())) != set(map(tuple, selected[['head', 'site']].drop_duplicates().to_numpy())):
        raise ValueError('Native map and held-target site/head support differ')
    cells = cell_results(selected, coordinates)
    native = native.merge(coordinates[['canonical_site_id', 'weather_cell']], left_on='site', right_on='canonical_site_id', validate='many_to_one')
    support = native.groupby(['head', 'weather_cell']).agg(n_native_labels=('label_id', 'nunique'),
        n_native_units=('unit', 'nunique'), n_observed_years=('year', 'nunique'),
        year_min=('year', 'min'), year_max=('year', 'max'), support_sites=('site', 'nunique')).reset_index()
    cells = cells.merge(support, on=['head', 'weather_cell'], validate='many_to_one')
    if not cells.n_sites.eq(cells.support_sites).all():
        raise ValueError('Native map cell support differs between models')
    panels = []
    for row, head in enumerate(HEADS):
        group = cells[cells['head'].eq(head)]
        for column, model in enumerate(['T', 'nonlinear_T']):
            q = group[group.model.eq(model)].copy().assign(value=lambda f: f.error,
                panel='(' + chr(97 + row * 3 + column) + ')', metric='mae', reference_model='',
                bbch=HEADS[head], display_min=0., display_max=30.)
            panels.append(q)
        q = cell_difference(group, 'residual_MLP', 'nonlinear_T').assign(value=lambda f: f.difference,
            panel='(' + chr(99 + row * 3) + ')', metric='difference', reference_model='nonlinear_T',
            bbch=HEADS[head], display_min=-10., display_max=10.)
        panels.append(q)
    result = pd.concat(panels, ignore_index=True)
    summary = []
    for panel, group in result.groupby('panel'):
        summary.append(dict(panel=panel, model=group.model.iloc[0], head=int(group['head'].iloc[0]), cells=len(group),
            sites=int(group.n_sites.sum()), minimum=float(group.value.min()), maximum=float(group.value.max()),
            median=float(group.value.median()), unresolved_cells=int(group.any_unresolved.sum()),
            clipped_low_cells=int(group.value.lt(group.display_min).sum()), clipped_high_cells=int(group.value.gt(group.display_max).sum())))
    return {'pooled_cells.csv': cells, 'pooled_map.csv': result, 'spatial_summary.csv': pd.DataFrame(summary)}


def run(input_folder=DEFAULT_INPUT, output_folder=None, reference_folder=PRIOR,
        coordinates_path=ANNUAL / 'site_zones.csv', replicates=REPLICATES, seed=SEED):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder) if output_folder else input_folder / 'analysis'
    frame, inputs, receipts = load_folds(input_folder, reference_folder)
    coordinates_path = Path(coordinates_path)
    inputs[str(coordinates_path)] = sha(coordinates_path)
    tables = build_tables(frame, replicates, seed)
    tables.update(pooled_map_tables(tables['benchmark_sites.csv.gz'], frame, pd.read_csv(coordinates_path)))
    tables['keyed_predictions.csv.gz'] = frame
    output_folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        compression = {'method': 'gzip', 'mtime': 0} if name.endswith('.gz') else None
        table.to_csv(output_folder / name, index=False, compression=compression)
    verification = dict(complete=True, study='full_source', recipe='all_eligible', folds=receipts,
        models=MODELS, target_labels=len(frame), target_label_sha256=label_digest(frame),
        later_labels=int(frame.period.eq('later_held').sum()), later_sites=int(frame[frame.period.eq('later_held')].site.nunique()),
        exact_archived_target_metadata=True, all_targets_scored=True, invalid_prediction_loss_days=366,
        weighting='observations within site-year; years within site; sites within native head; equal seven heads',
        completion_and_failure_weighting='same native observation/site-year/site/head hierarchy as MAE',
        intervals=dict(replicates=int(replicates), seed=int(seed), confidence=.95,
            scope='conditional whole-target-block sampling; fixed one full-source fit per outer fold',
            paired=True, source_draw_averaging=False, refitting=False,
            missing_native_head_policy='undefined replicate; CI undefined below 95% complete replicates'),
        native_map_support=True, coordinate_basis='recorded native weather-cell centres', projection='EPSG:3035',
        input_sha256=inputs, code_sha256={str(Path(__file__)): sha(__file__)},
        output_sha256={name: sha(output_folder / name) for name in tables})
    (output_folder / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(tables['summary.csv'].to_string(index=False))
    return tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.input, args.output)
