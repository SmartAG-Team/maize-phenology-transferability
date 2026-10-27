"""Paired structural contrasts from frozen fits; no new training or selection."""
import json
import numpy as np
import pandas as pd
from research.phenology_exploration.analysis import block_interval
from research.phenology_exploration.core import sha
from .climate import ROOT, OUT

PRIOR = ROOT / 'data/results/phenology_exploration_20261023/year_matched/paired_causal_neural'
KEYS = ['head', 'site', 'block', 'draw_seed', 'fold']
PAIRS = [('nonlinear_T', 'T'), ('residual_MLP', 'nonlinear_T'),
         ('thermal_transformer', 'nonlinear_T')]


def paired_contrast(frame, left, right):
    a = frame.loc[frame.model.eq(left), KEYS + ['error']]
    b = frame.loc[frame.model.eq(right), KEYS + ['error']]
    q = a.merge(b, on=KEYS, how='outer', validate='one_to_one',
                suffixes=('_left', '_right'), indicator=True)
    if not q['_merge'].eq('both').all() or not np.isfinite(q[['error_left', 'error_right']]).all().all():
        raise ValueError('Structural comparisons require identical finite target supports')
    if set(q.draw_seed) != {17, 29, 43} or set(q.fold) != set(range(5)):
        raise ValueError('Incomplete frozen source draws/folds')
    if not q.groupby(['head', 'site', 'block']).size().eq(3).all():
        raise ValueError('A target is missing a source draw')
    q['difference'] = q.error_left - q.error_right
    average = q.groupby(['head', 'site', 'block'])[['error_left', 'error_right', 'difference']].mean().reset_index()
    if average['head'].nunique() != 7:
        raise ValueError('Incomplete native-head population')
    left_error, right_error, difference = average.groupby('head')[['error_left', 'error_right', 'difference']].mean().mean()
    lo, hi = block_interval(average.rename(columns={'difference': 'error'}))
    draw_means = q.groupby(['draw_seed', 'head']).difference.mean().groupby('draw_seed').mean()
    return dict(left=left, right=right, left_error=float(left_error), right_error=float(right_error),
                difference=float(difference), ci_low=float(lo), ci_high=float(hi),
                left_relative_reduction_percent=float(100 * (right_error-left_error) / right_error),
                interval_excludes_zero=bool(hi < 0 or lo > 0), source_draw_sd=float(draw_means.std(ddof=1)),
                source_draw_min=float(draw_means.min()), source_draw_max=float(draw_means.max()),
                sites=int(q.site.nunique()), blocks=int(q.block.nunique()), heads=7,
                paired_case_site_head_rows=len(q))


def run():
    old_path = PRIOR / 'integrated_site_errors.csv.gz'
    new_path = OUT / 'site_transfer_errors.csv.gz'
    old_receipt = json.loads((PRIOR / 'verification.json').read_text())
    new_receipt = json.loads((OUT / 'analysis_verification.json').read_text())
    assert old_receipt['complete'] and new_receipt['complete']
    assert sha(old_path) == old_receipt['files'][old_path.name]
    assert sha(new_path) == new_receipt['files'][new_path.name]
    old = pd.read_csv(old_path)
    assert old.n_sites.eq(80).all()
    old = old[old.model.isin(['T', 'relaxed_T', 'relaxed_phase_T', 'residual_MLP', 'thermal_transformer'])].copy()
    old.model = old.model.replace({'relaxed_T': 'nonlinear_T', 'relaxed_phase_T': 'phase_nonlinear_T'})
    new = pd.read_csv(new_path)
    new = new[new.source_zone.eq(-1)].rename(columns={'draw': 'draw_seed'})
    new['recipe'] = 'random'
    rows = []; target_tables = []
    for study, budget, frame in [('prior_pooled_80', 80, old), ('annual_pooled_40', 40, new)]:
        pairs = PAIRS + ([('nonlinear_T', 'phase_nonlinear_T')] if budget == 80 else [('support_blend', 'nonlinear_T')])
        for (period, recipe), group in frame.groupby(['period', 'recipe']):
            target_tables.append(group.groupby(['head', 'site', 'block', 'model']).error.mean().reset_index().assign(
                study=study, source_sites=budget, period=period, recipe=recipe))
            for left, right in pairs:
                rows.append(dict(study=study, source_sites=budget, source_years_per_site=8,
                                 period=period, recipe=recipe, **paired_contrast(group, left, right)))
    result = pd.DataFrame(rows)
    # Independent agreement with archived aggregate scores (the paired CI is newly derived).
    prior_summary = pd.read_csv(PRIOR / 'integrated_summary.csv')
    current_summary = pd.read_csv(OUT / 'summary.csv')
    largest = 0.
    for r in result.itertuples():
        for model, value in [(r.left, r.left_error), (r.right, r.right_error)]:
            if r.study == 'prior_pooled_80':
                name = {'nonlinear_T': 'relaxed_T', 'phase_nonlinear_T': 'relaxed_phase_T'}.get(model, model)
                reference = prior_summary[prior_summary.period.eq(r.period) & prior_summary.recipe.eq(r.recipe) & prior_summary.model.eq(name)].mae.item()
            else:
                reference = current_summary[current_summary.period.eq(r.period) & current_summary.model.eq(model) & current_summary.comparison.eq('pooled_source')].error.item()
            largest = max(largest, abs(value-reference))
    assert largest < 1e-9
    result.to_csv(OUT / 'structural_contrasts.csv', index=False)
    pd.concat(target_tables, ignore_index=True).to_csv(OUT / 'structural_site_errors.csv.gz', index=False,
                                                     compression={'method': 'gzip', 'mtime': 0})
    receipt = dict(complete=True, fitted_models_changed=False, contrasts=len(result),
                   input_sha256={str(p): sha(p) for p in [old_path, new_path, PRIOR/'integrated_summary.csv', OUT/'summary.csv']},
                   code_sha256=sha(__file__), output_sha256={name: sha(OUT/name) for name in ['structural_contrasts.csv', 'structural_site_errors.csv.gz']},
                   paired_target_supports_identical=True, max_aggregate_reconstruction_difference_days=largest,
                   metric='all-target penalised MAE; observations/site-year, years/site, sites/head, seven heads',
                   sign='left minus right; negative values favour the left specification',
                   intervals='2000 paired whole target-block draws after averaging three source draws; conditional on frozen fits',
                   inference='exploratory unadjusted 95% intervals; no equivalence, physiological or prospective claim',
                   source_draw_variation='SD and range of three paired point differences, separate from target-block intervals',
                   archived_design='80 sites x eight historical years; random and capped coverage-balanced source sampling; 30 frozen cases',
                   current_design='40 sites x eight historical years; pooled-source subset of 90 annual-transfer cases')
    (OUT / 'structural_verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(result[result.period.eq('later_held')][['study', 'recipe', 'left', 'right', 'difference', 'ci_low', 'ci_high']].round(4).to_string(index=False))


if __name__ == '__main__':
    run()
