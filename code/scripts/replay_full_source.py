"""Replay every held prediction from the six public saved artifacts in each fold."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT)]
from model.api import load
from research.annual_temperature_transfer.design import dataset
from research.full_source_benchmark.fit import artifact_predict, row_predictions
from research.full_source_benchmark.analysis import checked_receipt, MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/results/full_source_benchmark_20261026')
    args = parser.parse_args()
    data, _, _ = dataset()
    rows = []
    for fold in range(5):
        folder = args.input / f'fold{fold}'
        checked_receipt(folder, {})
        reference = pd.read_csv(folder / 'predictions.csv.gz', float_precision='round_trip')
        ids = np.sort(reference.u.unique())
        for name in MODELS:
            artifact = load(folder / 'models' / f'{name}.pkl', model=name)
            actual = row_predictions(reference, ids, artifact_predict(artifact, data, ids))
            saved = reference[name].to_numpy()
            np.testing.assert_allclose(actual, saved, rtol=0, atol=1e-8, equal_nan=True)
            valid = np.isfinite(saved)
            rows.append(dict(fold=fold, model=name, labels=len(reference),
                unresolved=int((~valid).sum()), maximum_difference_days=float(np.max(abs(actual[valid]-saved[valid])))))
        print(json.dumps(dict(fold=fold, complete=True, labels=len(reference))), flush=True)
    report = dict(complete=True, artifacts=len(rows), scored_predictions=sum(r['labels'] for r in rows),
                  public_model_api=True, all_held_predictions_replayed=True, records=rows)
    (ROOT / 'outputs/full_source_saved_model_replay.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))


if __name__ == '__main__':
    main()
