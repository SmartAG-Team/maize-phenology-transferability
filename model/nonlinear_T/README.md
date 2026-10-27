# nonlinear_T

Nonlinear thermal accumulation selects the temperature-response exponent using disjoint source train/tune blocks, then refits seven stage thresholds on all source labels. The default grid is 1/64, 1/32, 1/16, 1/8, 1/4, 0.5, 0.75, 1 and 1.5.

The implementation wraps the existing scientific estimator. See [the shared interface guide](../README.md) for the exact input arrays, label roles, support requirements, selection budgets and trusted-artifact handling.

```python
from model import nonlinear_T, load_inputs

data = load_inputs("data/example/native_inputs.npz", weather="dwd")
artifact = nonlinear_T.load("data/example")
dates = nonlinear_T.predict(artifact, data)  # (24, 7), elapsed days; NaN if unresolved

# For a new source cohort with the documented arrays and label schema:
# artifact = nonlinear_T.fit(data, train_labels, tune_labels, seed=17)
# nonlinear_T.save(artifact, "outputs/nonlinear_T.pkl")
```

```bash
.venv/bin/python -m model.nonlinear_T predict \
  --inputs data/example/native_inputs.npz --weather dwd \
  --artifact data/example --output outputs/nonlinear_T_dwd.npy

.venv/bin/python -m model.nonlinear_T train \
  --inputs source_inputs.npz --labels source_labels.csv \
  --output outputs/nonlinear_T.pkl
```
