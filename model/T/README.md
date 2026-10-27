# T

Linear clipped thermal accumulation uses daily minimum and maximum temperatures clipped to 8–30 °C and seven source-fitted stage thresholds. No hyperparameter selection is required. Unsupported stages retain NaN predictions.

The implementation wraps the existing scientific estimator. See [the shared interface guide](../README.md) for the exact input arrays, label roles, support requirements, selection budgets and trusted-artifact handling.

```python
from model import T, load_inputs

data = load_inputs("data/example/native_inputs.npz", weather="dwd")
artifact = T.load("data/example")
dates = T.predict(artifact, data)  # (24, 7), elapsed days; NaN if unresolved

# For a new source cohort with the documented arrays and label schema:
# artifact = T.fit(data, train_labels, tune_labels, seed=17)
# T.save(artifact, "outputs/T.pkl")
```

```bash
.venv/bin/python -m model.T predict \
  --inputs data/example/native_inputs.npz --weather dwd \
  --artifact data/example --output outputs/T_dwd.npy

.venv/bin/python -m model.T train \
  --inputs source_inputs.npz --labels source_labels.csv \
  --output outputs/T.pkl
```
