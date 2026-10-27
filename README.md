# Predictive skill and environmental transferability of maize phenology models

Scientific code, callable models, qualified study inputs, frozen fitted artifacts, numerical evidence and figures supporting the maize phenology transferability study. Version **v1.1.0** adds the complete-source primary comparison (30 fitted artifacts across five frozen spatial folds), ordinary valid-date MAE and signed-bias evaluation, and the corresponding revised figures and plotting workbooks.

The study uses 207,325 qualified stage dates from 47,008 seasonal records in the German exact-date network. Evaluation is retrospective with supplied weather. Prediction failures are counted separately; paired comparisons intersect valid observations before site-year, site and equal-stage averaging. Saved fits and their original training/selection objectives remain unchanged. These comparisons do not establish operational forecasting skill or generalization outside the observed network.

## Files

| Path | Contents |
| --- | --- |
| `code/` | Scientific fitting, inference, evaluation, figure rendering, replay and tests. |
| `model/` | Six callable model interfaces with training, prediction, saving and loading. |
| `data/study_dataset/` | Complete qualified inputs, daily weather arrays and frozen source-fold roles, restored from the Release. |
| `data/results/` | Earlier fitted models and checkpoints, the 30 primary full-source artifacts, keyed predictions, memberships and source receipts, restored from the Release. |
| `data/evidence/` | Frozen numerical evidence for metrics and figures, restored from the Release. |
| `data/example/`, `data/local_example/` | Small saved-model replay inputs and reference predictions. |
| `paper/figures/` | Eleven PNG figures at 300 dpi, Excel plotting data and captions. |
| `data/graphical_abstract/` | Graphical-abstract evidence, geometry and layout metadata. |
| `outputs/` | Locally generated reconstructions, figures and checks; excluded from Git. |

Unpublished manuscript text, author administration, review records and submission documents remain outside this public distribution. The `paper/` directory contains graphical evidence only. Historical provenance records retain their original source descriptions and paths.

## Restore the complete data

Large data are published as five Release assets, each smaller than 2 GiB. `release_assets.json` records their paths, byte sizes, SHA-256 hashes and download URLs. Download them from the [v1.1.0 Release](https://github.com/SmartAG-Team/maize-phenology-transferability/releases/tag/v1.1.0), or use GitHub CLI from the repository root:

```sh
mkdir -p outputs/release_assets
gh release download v1.1.0 --repo SmartAG-Team/maize-phenology-transferability --pattern '*.tar.gz' --pattern SHA256SUMS --dir outputs/release_assets
(cd outputs/release_assets && shasum -a 256 -c SHA256SUMS)
for asset in outputs/release_assets/*.tar.gz; do tar -xzf "$asset"; done
```

Proceed with extraction only when every checksum passes. Each archive restores its recorded `data/` subdirectories relative to the repository root. The `manifest.json` inventory records every distributed file, including files restored from these archives. Provider credentials and access to the original workstation are unnecessary for saved-result reproduction.

## Offline numerical reproduction

Python 3.12 is the recorded environment. Direct dependencies are pinned in `requirements.txt`; `requirements.lock.txt` records the macOS ARM transitive environment. After restoring the Release assets:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p outputs
.venv/bin/python code/replay_weather_example.py
.venv/bin/python code/replay_local_example.py
.venv/bin/python code/scripts/replay_full_source.py
.venv/bin/python code/scripts/reconstruct_metrics.py
.venv/bin/python code/scripts/render_figures.py
.venv/bin/python code/scripts/verify_workflow.py
.venv/bin/python code/scripts/verify_all_figures.py
```

The replay commands check saved predictions, including every held prediction from the 30 full-source primary artifacts. Numerical reconstruction checks ordinary MAE, signed bias, paired contrasts and conditional intervals, and retains the earlier 924 grouped numerical audit checks. Figure reconstruction uses frozen evidence and checks plotting workbooks independently. Figures use the same 19 cm final width, 12 pt text, 300 dpi PNG output and consistent model colours.

The source-isolation test compares complete selection receipts. Run tests with a sequential joblib backend to avoid parallel random-forest summation differences in that synthetic fixture; published fitting settings remain unchanged:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=code:. .venv/bin/python - <<'PY_TEST'
import sys, pytest
from joblib import parallel_config
with parallel_config(backend='sequential'):
    sys.exit(pytest.main(['code/tests', '-q', '-p', 'no:cacheprovider']))
PY_TEST
```

File integrity can be checked independently with the Python standard library:

```sh
python3 - <<'PY_VERIFY'
from pathlib import Path
import hashlib, json
records = json.loads(Path('manifest.json').read_text())['files']
for record in records:
    path = Path(record['path'])
    assert path.is_file() and not path.is_symlink(), path
    assert path.stat().st_size == record['bytes'], path
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    assert digest == record['sha256'], path
print(f"Verified {len(records)} files")
PY_VERIFY
```

## Call or train individual models

The six model interfaces are `T`, `nonlinear_T`, `residual_RF`, `residual_MLP`, `thermal_transformer` and `support_blend`. Input schemas, callable examples and a small training example are documented in [`model/README.md`](model/README.md).

```sh
.venv/bin/python -m model.nonlinear_T predict --inputs data/example/native_inputs.npz --weather dwd --artifact data/example --output outputs/nonlinear_T_dwd.npy
.venv/bin/python -m model.residual_MLP train --inputs SOURCE.npz --labels SOURCE.csv --output outputs/residual_MLP.pkl
```

The primary comparison uses every eligible pre-2011 source record in each frozen spatial fold: 1,052–1,112 sites and 73,381–79,834 stage dates. Held sites, buffer sites and later years remain excluded from training. The 40/80-site experiments remain fixed-budget controls. To refit one primary fold (repeat for folds 0–4):

```sh
PYTHONPATH=code:. .venv/bin/python -m research.full_source_benchmark.fit --fold 0 --output outputs/full_source_refits --device cpu
```

The primary refit has no source-site or source-year cap. On supported Apple hardware, `--device mps` enables Transformer fitting on the GPU. New fits retain their own code/input signatures. Hardware and library differences can affect fitted weights; replay of frozen predictions is separate from optimizer reruns.

The earlier frozen annual-temperature, DWD and ERA5 cases and fixed-budget benchmark can also be refitted:

```sh
.venv/bin/python code/scripts/refit_case.py --cohort annual --all-cases --prepare-only
.venv/bin/python code/scripts/refit_case.py --cohort dwd --all-cases --prepare-only
.venv/bin/python code/scripts/refit_case.py --cohort era5 --all-cases --prepare-only
.venv/bin/python code/scripts/refit_benchmark.py --prepare-only
```

Remove `--prepare-only` to run optimization. Newly fitted outputs use `outputs/`; published fitted artifacts remain frozen.

## Sources and reproduction limits

`DATA_SOURCES.md` records DWD, Copernicus/ECMWF and Natural Earth attribution and source-specific reuse terms. No blanket software licence is assigned. DWD station observations, ERA5-Land UTC reanalysis and AgERA5 local-day products retain their distinct definitions and provenance. Native BBCH 87 remains distinct from harvest or generic physiological maturity.

Qualified 1981–2010 reference-climate aggregates, Earth Engine responses and calendar/source receipts are included. Complete historical daily reference grids were not archived. Recreating provider aggregates requires separately authorized live provider access; daily weather used for study fitting and prediction is included. Optional Earth Engine reacquisition uses the user's own credentials and `PHENOLOGY_EE_PROJECT`. No authentication material is distributed.

Earlier releases remain available with their original evidence and scoring definitions. The active v1.1.0 summaries are under `data/evidence/mae_bias_revision_20261026/`; primary full-source fits are under `data/results/full_source_benchmark_20261026/`.
