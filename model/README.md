# Callable and trainable phenology models

Each study model has its own package and command-line entry point:

| Package | Estimator and required components |
| --- | --- |
| [`T/`](T/) | Linear clipped thermal accumulation with seven fitted stage thresholds |
| [`nonlinear_T/`](nonlinear_T/) | Source-selected temperature exponent and seven fitted thresholds |
| [`residual_RF/`](residual_RF/) | Linear T plus a spatially cross-fitted random-forest correction |
| [`residual_MLP/`](residual_MLP/) | Linear T plus a spatially cross-fitted MLP correction |
| [`thermal_transformer/`](thermal_transformer/) | Causal thermal-rate Transformer initialized from linear T |
| [`support_blend/`](support_blend/) | Source-climate-support blend of residual MLP and nonlinear T |

The adapters call the existing implementations in `code/research/phenology_exploration/`. The fitting and selection defaults follow `code/research/annual_temperature_transfer/fit.py`. Fitting one package fits its required components. It does not train all six models.

Run commands from the repository root after installing `requirements.txt`. No editable package installation or network access is needed. The examples use `.venv/bin/python`.

## Small runnable training example

```bash
.venv/bin/python -m model.make_example --output outputs/model_example
.venv/bin/python -m model.residual_MLP train \
  --inputs outputs/model_example/inputs.npz --labels outputs/model_example/labels.csv \
  --output outputs/model_example/residual_MLP.pkl --mlp-steps 2
.venv/bin/python -m model.residual_MLP predict \
  --inputs outputs/model_example/inputs.npz --artifact outputs/model_example/residual_MLP.pkl \
  --output outputs/model_example/residual_MLP_dates.npy
```

The generated cohort contains 90 synthetic units, seven stages, six train blocks and three disjoint tune blocks. Its support is sufficient to exercise nonempty residual learners. The two-update fit is an execution check using fabricated weather and phenology, without scientific validation. Use the same inputs with any other package; for a short Transformer run add `--transformer-fixed-steps 2`.

## Call supplied models immediately

```bash
.venv/bin/python -m model.T predict \
  --inputs data/example/native_inputs.npz --weather dwd \
  --artifact data/example --output outputs/T_dwd.npy

.venv/bin/python -m model.residual_MLP predict \
  --inputs data/example/native_inputs.npz --weather era5 \
  --artifact data/example --output outputs/residual_MLP_era5.npy

.venv/bin/python -m model.thermal_transformer predict \
  --inputs data/example/native_inputs.npz --weather dwd \
  --artifact data/example --output outputs/transformer_dwd.npy
```

Replace the package name with any of the six names above. The equivalent shared entry point is `python -m model predict --model T ...`. `--artifact` accepts either a newly saved artifact or a supplied archive directory containing `fits.pkl`, `thermal_transformer.pt` and, for the blend, `native_inputs.npz` with `source_climate`.

```python
import numpy as np
from model import T, residual_MLP, load_inputs

data = load_inputs("data/example/native_inputs.npz", weather="dwd")
thermal = T.load("data/example")
mlp = residual_MLP.load("data/example")
thermal_dates = T.predict(thermal, data)       # (24, 7)
mlp_dates = residual_MLP.predict(mlp, data, ids=[0, 4, 9])  # (3, 7)
```

Rows follow NPZ array order. `data/example/native_inputs.npz` includes `original_u` for mapping the 24 example rows back to the supplied reference CSV identifiers. These archived models were fitted with ERA5 weather; changing `--weather` changes inference forcing, without retraining.

## Primary full-source saved models

The six primary models for each frozen outer fold are supplied as standalone artifacts under `data/results/full_source_benchmark_20261026/fold0/models/` (folds 0–4). Load them through the same API:

```python
from model import residual_MLP
artifact = residual_MLP.load("data/results/full_source_benchmark_20261026/fold0/models/residual_MLP.pkl")
dates = residual_MLP.predict(artifact, data)
```

`data` follows the input schema below. The saved fold memberships identify every source record; predictions on the supplied held cohorts can be verified with `code/scripts/replay_full_source.py`. The full-source six-model training recipe and source-only budget extensions are documented in the root README.

## Input schema

The Python API accepts an object, such as `types.SimpleNamespace`, with these arrays. The CLI reads the same names from one NPZ file with `allow_pickle=False`.

| Array | Shape | Meaning |
| --- | --- | --- |
| `raw` | `(N,D,3)` | Daily minimum temperature (°C), maximum temperature (°C), daylength (h), in this order; elapsed day zero is sowing day |
| `context` | `(N,5)` | Reference April–September mean temperature (°C), reference diurnal range (°C), sine of sowing day of year, cosine of sowing day of year, sowing-day daylength (h), in this order |
| `lengths` | `(N,)` | Number of usable daily rows at inference, from zero through `D` |
| `train_lengths` | `(N,)` | Number of source-eligible daily rows; must not exceed `lengths` |

The study reference-climate period is 1981–2010. Reference climate is a known covariate; it is not calculated from target phenology. Source-site reference climate must be consistent across that site's years, since blend support uses one climate pair per site. The sowing sine/cosine convention must match the training data when calling an existing fit. The frozen study arrays preserve that convention.

All weather and context values, including padded daily rows, must be finite. Padding is ignored through the length arrays; it must preserve minimum temperature ≤ maximum temperature. `train_lengths` defaults to `lengths` if absent, so a historical cutoff must be supplied explicitly when required. No source/target year cutoff is inferred from filenames or weather values. `--train-prefix` evaluates predictions using `train_lengths`; ordinary prediction uses `lengths`.

Thermal and freshly fitted residual models also accept `(N,D,2)` minimum/maximum weather. This produces a residual feature set without daylength and is a new fit configuration, distinct from the study's 40-feature residual models. A saved residual model requires the same channel count used for fitting. The Transformer and supplied residual fits require all three channels; daylength is never fabricated. All models retain the five-column context schema.

The seven output columns are: 0 emergence, 1 stem elongation, 2 tasseling, 3 flowering onset, 4 milking, 5 early dough, 6 BBCH 87. Outputs are floating-point elapsed days after sowing, with fractional crossing times. Unsupported stages, missing thermal crossings and predictions outside the usable weather horizon are `NaN`. Residual predictions are stagewise corrections and do not impose a new chronological-order constraint. Climate blending uses a valid component when the other is unresolved.

## Train one model

Training labels are a pandas DataFrame with `u,head,observed,site,site_year,block`. `u` indexes the input-array rows; `head` is an integer from 0 to 6; `observed` is a nonnegative elapsed-day observation strictly below the unit's `train_lengths`. `site`, `site_year` and `block` identify the weighting hierarchy and spatial groups. Membership must be consistent within a unit; a site must belong to one block. The CLI additionally requires `role`, containing only `train` or `tune`. Target rows are rejected by the training CLI.

Train and tune labels must have disjoint unit, site, site-year and block identifiers. The supplied tune labels are part of the source cohort: selection uses inner train/tune roles, followed by refitting on their union. Target evaluation labels must remain outside both roles. The API receives train and tune DataFrames separately; cohort eligibility and weather provenance remain the caller's responsibility.

```python
import pandas as pd
from model import FitOptions, residual_RF, load_inputs

data = load_inputs("source_inputs.npz")
labels = pd.read_csv("source_labels.csv")
train = labels[labels.role.eq("train")]
tune = labels[labels.role.eq("tune")]

artifact = residual_RF.fit(data, train, tune, seed=17)
residual_RF.save(artifact, "outputs/residual_RF.pkl")
reloaded = residual_RF.load("outputs/residual_RF.pkl")
dates = residual_RF.predict(reloaded, data)
print(artifact.metadata["selected"])
```

```bash
.venv/bin/python -m model.residual_RF train \
  --inputs source_inputs.npz --labels source_labels.csv \
  --output outputs/residual_RF.pkl --seed 17

.venv/bin/python -m model.residual_RF predict \
  --inputs source_inputs.npz --artifact outputs/residual_RF.pkl \
  --output outputs/residual_RF_dates.npy
```

`T.fit(data, train)` can omit tuning because T has no selected hyperparameter. Other packages require a nonempty tune set. Passing tune labels to T includes them in the final source fit after validating separation.

The supplied study inputs can provide an actual training cohort without inventing new identifiers. For example, this prepares explicit roles for fold zero; running a full-cohort neural fit can be expensive:

```python
from pathlib import Path
import numpy as np
import pandas as pd

base = Path("data/study_dataset")
labels = pd.read_csv(base / "labels.csv.gz")
roles = pd.read_csv(base / "fold0_source_roles.csv.gz")
source = labels[labels.fold0_case.eq("source_1981_2010")
                & labels.pre2011_prefix_eligible].merge(roles, on="unit", validate="many_to_one")
source["role"] = np.where(source.inner.eq(2), "tune", "train")
Path("outputs").mkdir(exist_ok=True)
source.to_csv("outputs/fold0_source_labels.csv", index=False)
```

```bash
.venv/bin/python -m model.T train \
  --inputs data/study_dataset/native_inputs.npz \
  --labels outputs/fold0_source_labels.csv --output outputs/fold0_T.pkl
```

This example uses the complete eligible fold-zero source pool. Exact study-case replication requires the corresponding frozen sampled memberships and the existing `refit_case.py` workflow.

## Selection budgets and support

Nonlinear T selects alpha on the nine-value study grid. RF selects nine correction strengths. MLP selects 80, 160 or 320 optimizer updates and strengths 0, 0.5 or 1. Model selection retains the study's original hierarchical training objective and tie-breaking rules, including its failure penalty. Current manuscript evaluation uses ordinary MAE and signed bias on valid predictions and counts failures separately. RF and MLP teachers and final backbones use **linear T**, independently of nonlinear-T selection. Teachers exclude complete spatial blocks in three source folds. A fitted artifact retains selection trials, selected settings, source counts and teacher receipts in `metadata`.

Thermal stage calibration requires at least 30 distinct source environments per stage. Residual learning additionally requires at least 30 eligible correction environments per stage and qualifying thermal teachers in every participating source fold. Unsupported residual stages receive zero correction and retain complete-source T predictions. The Transformer requires all seven initial stage thresholds to be supported in both inner and complete-source fits, and reports a validation error otherwise. Sufficient total label count alone does not establish these conditions.

The default Transformer selects checkpoints at 0, 40, …, 320 updates using source tune labels and then refits for the selected length. A declared fixed training budget is available for short smoke runs or separately specified experiments:

```python
from model import FitOptions, thermal_transformer

artifact = thermal_transformer.fit(
    data, train, tune,
    options=FitOptions(transformer_fixed_steps=2),
)
```

```bash
.venv/bin/python -m model.thermal_transformer train \
  --inputs source_inputs.npz --labels source_labels.csv \
  --output outputs/transformer_smoke.pkl --transformer-fixed-steps 2
```

The fixed budget bypasses Transformer checkpoint selection and is recorded in metadata. Two updates verify execution; they do not reproduce a study fit. MLP budgets can be set with `FitOptions(mlp_steps=(2,))` or `--mlp-steps 2`; correction strength remains source-selected. Other custom grids are available through `FitOptions` in Python.

## Saved-artifact and numerical limits

Only load trusted pickle and Torch files. They contain Python/scikit-learn/Torch state and require compatible dependencies. Newly saved artifacts require a `.pkl` or `.pickle` suffix, including Transformer artifacts; `.pt` is reserved for loading supplied Torch checkpoints. Each new artifact records its model name and input channel count; wrapper and loader checks reject a different requested model. The loaders also accept supplied `fits.pkl` and `thermal_transformer.pt` directly with an explicit model name. For a blend stored elsewhere, pass a source-climate array of shape `(n_source_sites,2)` to `model.load(..., model="support_blend", source_climate=...)` or keep the neighboring `native_inputs.npz`.

Inference uses CPU Transformer replay. Transformer input weather is normalized to float32, including integer and float64 inputs, for compatibility with the scientific trainer's available-device choice. New optimizer runs may differ numerically by platform and library version. Supplied saved-model replay is checked against the archived dates at `1e-5` d absolute tolerance for the five non-Transformer models and `0.02` d for the Transformer. These interfaces provide retrospective timing under supplied weather; they do not add weather forecasting or establish transferability beyond the study network.

```bash
PYTHONPATH=code:$PWD .venv/bin/python -m pytest code/tests/test_model_interface.py -q
```

The tests replay all six models on both archived 24-unit weather examples, run real small fits for all six models, verify nonempty supported residual learners, check save/load equivalence, exercise the CLI, and reject source-role overlap and invalid inputs. The small neural fits use an explicitly declared two-update budget.
