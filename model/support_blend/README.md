# support_blend

The blend combines the source-selected residual MLP with nonlinear T. Its MLP component uses the linear-T backbone. The neural weight is min(1, radius / nearest-source-distance) squared in source-standardized two-feature climate space; radius is the 95th percentile of source-site leave-one-out nearest-neighbor distances. A valid component is retained when the other is unresolved. No target outcome selects the blend weight.

The implementation wraps the existing scientific estimator. See [the shared interface guide](../README.md) for the exact input arrays, label roles, support requirements, selection budgets and trusted-artifact handling.

```python
from model import support_blend, load_inputs

data = load_inputs("data/example/native_inputs.npz", weather="dwd")
artifact = support_blend.load("data/example")
dates = support_blend.predict(artifact, data)  # (24, 7), elapsed days; NaN if unresolved

# For a new source cohort with the documented arrays and label schema:
# artifact = support_blend.fit(data, train_labels, tune_labels, seed=17)
# support_blend.save(artifact, "outputs/support_blend.pkl")
```

```bash
.venv/bin/python -m model.support_blend predict \
  --inputs data/example/native_inputs.npz --weather dwd \
  --artifact data/example --output outputs/support_blend_dwd.npy

.venv/bin/python -m model.support_blend train \
  --inputs source_inputs.npz --labels source_labels.csv \
  --output outputs/support_blend.pkl
```
