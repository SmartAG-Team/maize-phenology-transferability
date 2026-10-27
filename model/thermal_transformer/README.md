# thermal_transformer

The causal Transformer adjusts daily thermal progress with positive neural multipliers. Two encoder layers have width 32, four attention heads and feedforward width 64. Linear T initializes its seven demands. The default source-only selection compares checkpoints at 0, 40, …, 320 updates, followed by complete-source refitting. Three weather channels, including daylength, and support for all seven stages are required.

The implementation wraps the existing scientific estimator. See [the shared interface guide](../README.md) for the exact input arrays, label roles, support requirements, selection budgets and trusted-artifact handling.

```python
from model import thermal_transformer, load_inputs

data = load_inputs("data/example/native_inputs.npz", weather="dwd")
artifact = thermal_transformer.load("data/example")
dates = thermal_transformer.predict(artifact, data)  # (24, 7), elapsed days; NaN if unresolved

# For a new source cohort with the documented arrays and label schema:
# artifact = thermal_transformer.fit(data, train_labels, tune_labels, seed=17)
# thermal_transformer.save(artifact, "outputs/thermal_transformer.pkl")
```

```bash
.venv/bin/python -m model.thermal_transformer predict \
  --inputs data/example/native_inputs.npz --weather dwd \
  --artifact data/example --output outputs/thermal_transformer_dwd.npy

.venv/bin/python -m model.thermal_transformer train \
  --inputs source_inputs.npz --labels source_labels.csv \
  --output outputs/thermal_transformer.pkl
```
