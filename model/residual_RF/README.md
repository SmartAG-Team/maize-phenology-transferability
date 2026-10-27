# residual_RF

The random forest learns observed-day residuals from spatially cross-fitted linear-T teachers. Source tuning selects the correction strength. The complete-source linear T model remains the final backbone; an unsupported residual stage receives zero correction.

The implementation wraps the existing scientific estimator. See [the shared interface guide](../README.md) for the exact input arrays, label roles, support requirements, selection budgets and trusted-artifact handling.

```python
from model import residual_RF, load_inputs

data = load_inputs("data/example/native_inputs.npz", weather="dwd")
artifact = residual_RF.load("data/example")
dates = residual_RF.predict(artifact, data)  # (24, 7), elapsed days; NaN if unresolved

# For a new source cohort with the documented arrays and label schema:
# artifact = residual_RF.fit(data, train_labels, tune_labels, seed=17)
# residual_RF.save(artifact, "outputs/residual_RF.pkl")
```

```bash
.venv/bin/python -m model.residual_RF predict \
  --inputs data/example/native_inputs.npz --weather dwd \
  --artifact data/example --output outputs/residual_RF_dwd.npy

.venv/bin/python -m model.residual_RF train \
  --inputs source_inputs.npz --labels source_labels.csv \
  --output outputs/residual_RF.pkl
```
