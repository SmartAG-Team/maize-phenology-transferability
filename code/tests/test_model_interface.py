"""Public interfaces must preserve scientific replay and prevent source leakage."""
from pathlib import Path
from types import SimpleNamespace
import importlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
NAMES = ("T", "nonlinear_T", "residual_RF", "residual_MLP", "thermal_transformer", "support_blend")


def api():
    # A missing implementation is an assertion failure during the initial red run.
    assert (ROOT / "model" / "api.py").is_file(), "The callable model interface is missing"
    return importlib.import_module("model")


@pytest.fixture(scope="module")
def source_data():
    n, days = 90, 84
    unit = np.arange(n)
    day = np.arange(days)
    seasonal = np.sin(day / 18)[None, :]
    low = 15 + seasonal + (unit % 10)[:, None] / 8
    high = low + 9
    light = np.broadcast_to(13 + day[None, :] / 200, (n, days))
    data = SimpleNamespace(raw=np.stack([low, high, light], 2).astype(np.float32),
                           context=np.column_stack([15 + unit / 90, 9 + (unit % 7) / 10,
                                                    unit / 100, np.ones(n), np.zeros(n)]).astype(np.float32),
                           lengths=np.full(n, days), train_lengths=np.full(n, days))
    labels = pd.DataFrame([dict(u=int(u), head=h, observed=float(d + (u % 3)),
                                site=f"s{u}", site_year=f"s{u}|2000", block=f"b{u // 10}")
                           for u in unit for h, d in enumerate([5, 12, 22, 31, 41, 52, 63])])
    return data, labels[labels.u.lt(60)].copy(), labels[labels.u.ge(60)].copy()


def test_six_public_wrappers_expose_callable_training_and_prediction():
    assert tuple(api().MODEL_NAMES) == NAMES
    for name in NAMES:
        module = importlib.import_module(f"model.{name}")
        assert callable(module.fit) and callable(module.predict) and callable(module.load)


@pytest.mark.parametrize("product", ["era5", "dwd"])
@pytest.mark.parametrize("name", NAMES)
def test_saved_models_preserve_24_unit_reference_dates(name, product):
    m = api()
    folder = ROOT / "data/example"
    with np.load(folder / "native_inputs.npz") as z:
        data = SimpleNamespace(raw=z[product], context=z["context"], lengths=z["lengths"],
                               train_lengths=z["lengths"])
        original = z["original_u"]
    artifact = m.load(folder, model=name)
    values = m.predict(artifact, data)
    reference = pd.read_csv(folder / f"{product}_reference.csv.gz")
    lookup = {int(u): i for i, u in enumerate(original)}
    actual = values[reference.u.map(lookup).to_numpy(int), reference["head"].to_numpy(int)]
    assert values.shape == (24, 7)
    np.testing.assert_allclose(actual, reference[name], rtol=0,
                               atol=.02 if name == "thermal_transformer" else 1e-5, equal_nan=True)


@pytest.mark.parametrize("name", NAMES)
def test_each_model_trains_saves_and_predicts_from_source_only(name, source_data, tmp_path):
    m = api()
    data, train, tune = source_data
    options = m.FitOptions(alpha_grid=(.5, 1.), mlp_steps=(2,), transformer_fixed_steps=2)
    artifact = m.fit(name, data, train, tune, options=options)
    values = m.predict(artifact, data, ids=[0, 64, 89])
    assert values.shape == (3, 7) and np.isfinite(values).all()
    assert np.all((values >= 0) & (values < 84))
    assert artifact.metadata["source_units"] == 90
    if name in {"residual_RF", "residual_MLP", "support_blend"}:
        assert not artifact.payload["residual"]["empty"]
        assert all(artifact.payload["residual"]["supported"])
        assert artifact.payload["base"]["alpha"] == 1.
    path = tmp_path / f"{name}.pkl"
    m.save(artifact, path)
    loaded = m.load(path, model=name)
    np.testing.assert_allclose(m.predict(loaded, data, ids=[0, 64, 89]), values, rtol=0, atol=1e-5)
    with pytest.raises(ValueError, match="model"):
        m.load(path, model=NAMES[(NAMES.index(name) + 1) % len(NAMES)])


@pytest.mark.parametrize("column", ["u", "site", "site_year", "block"])
def test_source_tune_overlap_is_rejected_before_training(column, source_data):
    data, train, tune = source_data
    tune = tune.copy()
    tune.loc[tune.index[0], column] = train.iloc[0][column]
    with pytest.raises(ValueError, match="overlap|consistent"):
        api().fit("nonlinear_T", data, train, tune)


def test_source_labels_outside_weather_prefix_are_rejected(source_data):
    data, train, tune = source_data
    broken = train.copy()
    broken.loc[broken.index[0], "observed"] = 84
    with pytest.raises(ValueError, match="prefix"):
        api().fit("T", data, broken, tune)


def test_invalid_weather_and_unit_ids_are_rejected(source_data):
    data, train, _ = source_data
    artifact = api().fit("T", data, train)
    with pytest.raises(ValueError, match="ids"):
        api().predict(artifact, data, ids=[-.5])
    broken = SimpleNamespace(**vars(data))
    broken.raw = data.raw.copy()
    broken.raw[0, 0, 0] = 100
    with pytest.raises(ValueError, match="minimum|Tmin"):
        api().predict(artifact, broken)


def test_unresolved_thermal_head_remains_nan(source_data):
    data, train, _ = source_data
    artifact = api().fit("T", data, train[train["head"].eq(0)])
    values = api().predict(artifact, data, ids=[0])
    assert np.isfinite(values[0, 0]) and np.isnan(values[0, 1:]).all()


def test_transformer_requires_real_daylength_channel(source_data):
    data, train, tune = source_data
    two_channel = SimpleNamespace(**vars(data))
    two_channel.raw = data.raw[:, :, :2]
    with pytest.raises(ValueError, match="daylength"):
        api().fit("thermal_transformer", two_channel, train, tune)


def test_cli_trains_only_requested_model_and_writes_native_prediction_matrix(source_data, tmp_path):
    api()
    data, train, tune = source_data
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, **vars(data))
    labels = tmp_path / "labels.csv"
    pd.concat([train.assign(role="train"), tune.assign(role="tune")]).to_csv(labels, index=False)
    artifact = tmp_path / "T.pkl"
    command = [sys.executable, "-m", "model.T", "train", "--inputs", str(inputs),
               "--labels", str(labels), "--output", str(artifact)]
    trained = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert trained.returncode == 0, trained.stderr
    output = tmp_path / "predictions.npy"
    predicted = subprocess.run([sys.executable, "-m", "model.T", "predict", "--inputs", str(inputs),
                                "--artifact", str(artifact), "--output", str(output)],
                               cwd=ROOT, capture_output=True, text=True)
    assert predicted.returncode == 0, predicted.stderr
    assert np.load(output).shape == (90, 7)
    assert api().load(artifact).model == "T"


def test_cli_rejects_target_roles_in_training_labels(source_data, tmp_path):
    api()
    data, train, _ = source_data
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, **vars(data))
    labels = tmp_path / "labels.csv"
    train.assign(role="target").to_csv(labels, index=False)
    result = subprocess.run([sys.executable, "-m", "model", "train", "--model", "T", "--inputs", str(inputs),
                             "--labels", str(labels), "--output", str(tmp_path / "bad.pkl")],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and "role" in result.stderr
    assert not (tmp_path / "bad.pkl").exists()


def test_save_rejects_torch_suffix_before_writing_pickle(source_data, tmp_path):
    data, train, _ = source_data
    artifact = api().fit("T", data, train)
    with pytest.raises(ValueError, match="pkl|pickle"):
        api().save(artifact, tmp_path / "misleading.pt")
    assert not (tmp_path / "misleading.pt").exists()


def test_cli_rejects_torch_suffix_for_new_pickle_artifact(source_data, tmp_path):
    api()
    data, train, _ = source_data
    inputs, labels, output = tmp_path / "inputs.npz", tmp_path / "labels.csv", tmp_path / "T.pt"
    np.savez(inputs, **vars(data))
    train.assign(role="train").to_csv(labels, index=False)
    result = subprocess.run([sys.executable, "-m", "model.T", "train", "--inputs", str(inputs),
                             "--labels", str(labels), "--output", str(output)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and "pickle" in result.stderr
    assert not output.exists()


def test_small_example_generator_creates_a_trainable_source_cohort(tmp_path):
    api()
    result = subprocess.run([sys.executable, "-m", "model.make_example", "--output", str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    labels = pd.read_csv(tmp_path / "labels.csv")
    data = api().load_inputs(tmp_path / "inputs.npz")
    assert data.raw.shape == (90, 84, 3)
    train, tune = labels[labels.role.eq("train")], labels[labels.role.eq("tune")]
    assert train.u.nunique() == 60 and tune.u.nunique() == 30
    assert not set(train.block) & set(tune.block)
    artifact = api().fit("T", data, train, tune)
    assert np.isfinite(api().predict(artifact, data)).all()


@pytest.mark.parametrize("dtype", [np.float64, np.int32])
def test_transformer_accepts_numeric_weather_without_device_dtype_failure(dtype, source_data):
    data, train, tune = source_data
    numeric = SimpleNamespace(**vars(data))
    numeric.raw = data.raw.astype(dtype)
    artifact = api().fit("thermal_transformer", numeric, train, tune,
                         options=api().FitOptions(transformer_fixed_steps=1))
    assert np.isfinite(api().predict(artifact, numeric, ids=[0, 89])).all()
