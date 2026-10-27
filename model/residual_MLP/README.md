# residual_MLP

The 48/32-unit GELU MLP learns observed-day residuals from spatially cross-fitted linear-T teachers. Source tuning selects optimizer updates and correction strength. The complete-source linear T model remains the final backbone; an unsupported residual stage receives zero correction.

The implementation wraps the existing scientific estimator. See [the shared interface guide](../README.md) for the exact input arrays, label roles, support requirements, selection budgets and trusted-artifact handling.

```python
from model import residual_MLP, load_inputs

data = load_inputs("data/example/native_inputs.npz", weather="dwd")
artifact = residual_MLP.load("data/example")
dates = residual_MLP.predict(artifact, data)  # (24, 7), elapsed days; NaN if unresolved

# For a new source cohort with the documented arrays and label schema:
# artifact = residual_MLP.fit(data, train_labels, tune_labels, seed=17)
# residual_MLP.save(artifact, "outputs/residual_MLP.pkl")
```

```bash
.venv/bin/python -m model.residual_MLP predict \
  --inputs data/example/native_inputs.npz --weather dwd \
  --artifact data/example --output outputs/residual_MLP_dwd.npy

.venv/bin/python -m model.residual_MLP train \
  --inputs source_inputs.npz --labels source_labels.csv \
  --output outputs/residual_MLP.pkl
```
