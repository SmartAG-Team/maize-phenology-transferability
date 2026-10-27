"""Adapters around the archived scientific estimators, without estimator rewrites."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
import pickle
import sys

import numpy as np
import pandas as pd

# The repository is runnable directly and does not require an editable install.
_CODE = str(Path(__file__).resolve().parents[1] / "code")
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

MODEL_NAMES = ("T", "nonlinear_T", "residual_RF", "residual_MLP", "thermal_transformer", "support_blend")
_LABEL_COLUMNS = ("u", "head", "observed", "site", "site_year", "block")


@dataclass(frozen=True)
class FitOptions:
    """Source-selection budgets; defaults match the annual-temperature study recipe."""
    alpha_grid: tuple = (1/64, 1/32, 1/16, 1/8, 1/4, .5, .75, 1., 1.5)
    rf_strengths: tuple = tuple(np.linspace(0., 1., 9))
    mlp_strengths: tuple = (0., .5, 1.)
    mlp_steps: tuple = (80, 160, 320)
    # A declared fixed budget bypasses Transformer checkpoint selection. None
    # preserves the source-only 0,40,...,320 checkpoint-selection recipe.
    transformer_fixed_steps: int | None = None


@dataclass
class ModelArtifact:
    model: str
    payload: dict
    metadata: dict = field(default_factory=dict)
    schema_version: int = 1


def _name(name):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}")
    return name


def _integer_array(value, name):
    a = np.asarray(value)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError(f"{name} must be a one-dimensional array of finite integers")
    return a.astype(np.int64)


def _data(data, model, channels=None):
    try:
        raw = np.asarray(data.raw)
        context = np.asarray(data.context)
        lengths = _integer_array(data.lengths, "lengths")
        train_lengths = _integer_array(getattr(data, "train_lengths", lengths), "train_lengths")
    except AttributeError as exc:
        raise ValueError("data requires raw, context and lengths") from exc
    if raw.ndim != 3 or raw.shape[2] not in (2, 3) or not raw.shape[0] or not raw.shape[1]:
        raise ValueError("raw must have shape (N,D,2) or (N,D,3)")
    if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
        raise ValueError("raw weather must be finite numeric values, including padding")
    if (raw[:, :, 0] > raw[:, :, 1]).any():
        raise ValueError("Daily minimum temperature (Tmin) exceeds maximum temperature")
    if context.shape != (len(raw), 5) or not np.issubdtype(context.dtype, np.number) or not np.isfinite(context).all():
        raise ValueError("context must contain five finite numeric covariates per unit")
    if lengths.shape != (len(raw),) or train_lengths.shape != lengths.shape:
        raise ValueError("lengths and train_lengths must have one entry per unit")
    if (lengths < 0).any() or (lengths > raw.shape[1]).any() or (train_lengths < 0).any() or (train_lengths > lengths).any():
        raise ValueError("Weather prefix lengths must satisfy 0 <= train_lengths <= lengths <= D")
    if model == "thermal_transformer" and raw.shape[2] != 3:
        raise ValueError("thermal_transformer requires raw channel 3: observed/computed daylength in hours")
    if channels is not None and model in {"residual_RF", "residual_MLP", "thermal_transformer", "support_blend"} and raw.shape[2] != channels:
        raise ValueError(f"Input has {raw.shape[2]} channels; saved model requires {channels}")
    if model == "thermal_transformer":
        # The scientific trainer creates its thermal-rate tensor without an
        # explicit dtype. MPS cannot accept a float64 rate from integer/float64
        # NumPy weather, so normalize this public numeric boundary to float32.
        raw = raw.astype(np.float32, copy=False)
    return SimpleNamespace(raw=raw, context=context, lengths=lengths, train_lengths=train_lengths)


def _labels(labels, data, role):
    if not isinstance(labels, pd.DataFrame) or not set(_LABEL_COLUMNS).issubset(labels.columns):
        raise ValueError(f"{role} labels require columns {_LABEL_COLUMNS}")
    q = labels.copy().reset_index(drop=True)
    if q.empty or q[list(_LABEL_COLUMNS)].isna().any().any():
        raise ValueError(f"{role} labels must be nonempty and complete")
    q["u"] = _integer_array(q.u, "label u")
    q["head"] = _integer_array(q["head"], "label head")
    if (q.u < 0).any() or (q.u >= len(data.raw)).any() or not q["head"].between(0, 6).all():
        raise ValueError("Label u must index raw and head must be in 0,...,6")
    observed = q.observed.to_numpy()
    if not np.issubdtype(observed.dtype, np.number) or not np.isfinite(observed).all() or (observed < 0).any():
        raise ValueError("observed must contain finite nonnegative elapsed days")
    if (observed >= data.train_lengths[q.u]).any():
        raise ValueError("A source label lies outside its training weather prefix")
    for group, column in [("u", "site"), ("u", "site_year"), ("u", "block"), ("site", "block"), ("site_year", "site")]:
        if q.groupby(group)[column].nunique().gt(1).any():
            raise ValueError(f"Inconsistent {column} membership within {group}")
    return q


def _options(options):
    o = FitOptions() if options is None else options
    if not isinstance(o, FitOptions):
        raise ValueError("options must be FitOptions")
    for name in ("alpha_grid", "rf_strengths", "mlp_strengths", "mlp_steps"):
        values = np.asarray(getattr(o, name))
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a nonempty finite sequence")
    if min(o.alpha_grid) <= 0 or any(not 0 <= x <= 1 for x in (*o.rf_strengths, *o.mlp_strengths)):
        raise ValueError("alpha must be positive and correction strengths must be in [0,1]")
    for value in (*o.mlp_steps, o.transformer_fixed_steps):
        if value is not None and (not isinstance(value, (int, np.integer)) or value < 0):
            raise ValueError("Training steps must be nonnegative integers")
    return o


def _score(labels, values, data):
    from research.phenology_exploration.core import metric
    ids = np.sort(labels.u.unique())
    lookup = {int(u): i for i, u in enumerate(ids)}
    row = labels.u.map(lookup).to_numpy(int)
    return metric(labels, values[row, labels["head"].to_numpy(int)], data.train_lengths[labels.u])["penalized_mae"]


def _teacher(data, labels):
    from research.phenology_exploration.models import teacher_predictions
    teacher, receipts = teacher_predictions(data, labels)
    if not receipts:
        raise ValueError("Residual learning requires source spatial blocks")
    qualified = np.asarray([r["support"] for r in receipts]).min(0) >= 30
    return teacher, receipts, qualified


def fit(model, data, train_labels, tune_labels=None, *, seed=17, options=None):
    """Select on disjoint source blocks, then refit this model on train plus tune.

    T may omit tuning. Other models require explicit tune labels. No target label
    parameter exists. Only the selected model and its necessary components fit.
    """
    from research.phenology_exploration.models import fit_thermal, fit_residual, predict as physical_predict, residual_prediction
    model = _name(model)
    data = _data(data, model)
    options = _options(options)
    train = _labels(train_labels, data, "train")
    tune = None if tune_labels is None else _labels(tune_labels, data, "tune")
    if tune is None and model != "T":
        raise ValueError(f"{model} requires separate source tune labels")
    if tune is not None:
        for column in ("u", "site", "site_year", "block"):
            if set(train[column]) & set(tune[column]):
                raise ValueError(f"Source train/tune {column} overlap is forbidden")
    source = train if tune is None else pd.concat([train, tune], ignore_index=True)
    source = _labels(source, data, "complete source")
    metadata = dict(input_channels=data.raw.shape[2], source_units=int(source.u.nunique()),
                    train_units=int(train.u.nunique()), tune_units=0 if tune is None else int(tune.u.nunique()),
                    seed=int(seed), options=asdict(options), selection_trials={}, selected={},
                    source_only_selection=True, refit="train+tune source labels")
    payload = {}
    final_T = fit_thermal(data, source)
    if model == "T":
        payload["physical"] = final_T
        return ModelArtifact(model, payload, metadata)
    ids = np.sort(tune.u.unique())
    if model in {"nonlinear_T", "support_blend"}:
        trials = []
        for alpha in options.alpha_grid:
            candidate = fit_thermal(data, train, alpha=alpha)
            value = _score(tune, physical_predict(candidate, data, ids, train=True), data)
            trials.append(dict(alpha=float(alpha), mae=value))
        best = min(trials, key=lambda q: (q["mae"], abs(q["alpha"] - 1)))
        metadata["selection_trials"]["nonlinear_T"] = trials
        metadata["selected"]["alpha"] = best["alpha"]
        payload["physical"] = fit_thermal(data, source, alpha=best["alpha"])
        if model == "nonlinear_T":
            return ModelArtifact(model, payload, metadata)
    train_T = fit_thermal(data, train)
    if model == "thermal_transformer":
        from research.phenology_exploration.neural import fit_network
        for description, base in [("inner source", train_T), ("complete source", final_T)]:
            if not np.isfinite(base["thresholds"]).all():
                raise ValueError(f"Transformer requires >=30 source environments for every head in {description}")
        if options.transformer_fixed_steps is None:
            _, _, steps, history = fit_network(data, train, train_T["thresholds"], False, seed=int(seed), tune=tune)
            metadata["selection_trials"]["thermal_transformer"] = history
        else:
            steps = options.transformer_fixed_steps
            metadata["transformer_budget"] = "declared fixed steps; no checkpoint selection"
        network, norm, _, _ = fit_network(data, source, final_T["thresholds"], False, seed=int(seed), fixed_steps=steps)
        payload.update(state={k: v.detach().cpu() for k, v in network.state_dict().items()},
                       norm=norm, initial=final_T["thresholds"], context=False)
        metadata["selected"]["transformer_steps"] = int(steps)
        return ModelArtifact(model, payload, metadata)
    kind = "RF" if model == "residual_RF" else "MLP"
    teacher, receipts, qualified = _teacher(data, train)
    base_tune = physical_predict(train_T, data, ids, train=True)
    trials = []
    budgets = (160,) if kind == "RF" else options.mlp_steps
    strengths = options.rf_strengths if kind == "RF" else options.mlp_strengths
    for steps in budgets:
        residual = fit_residual(data, train, teacher, kind, seed=int(seed), steps=int(steps), teacher_qualified=qualified)
        delta = residual_prediction(residual, data, ids, base_tune, train=True)
        for strength in strengths:
            value = _score(tune, base_tune + strength * delta, data)
            trials.append(dict(steps=int(steps), strength=float(strength), mae=value))
    best = min(trials, key=lambda q: (q["mae"], q["strength"], q["steps"]))
    full_teacher, full_receipts, full_qualified = _teacher(data, source)
    residual = fit_residual(data, source, full_teacher, kind, seed=int(seed), steps=best["steps"], teacher_qualified=full_qualified)
    payload.update(base=final_T, residual=residual, strength=best["strength"])
    metadata["selection_trials"][f"residual_{kind}"] = trials
    metadata["selected"].update(residual_strength=best["strength"], residual_steps=best["steps"])
    metadata["teachers"] = dict(inner=receipts, complete_source=full_receipts)
    if model == "support_blend":
        from research.phenology_exploration.support_blend import support_weight
        source_sites = source.drop_duplicates("site").u.to_numpy(int)
        payload["source_climate"] = data.context[source_sites, :2].copy()
        _, geometry = support_weight(payload["source_climate"], payload["source_climate"])
        metadata["support_geometry"] = geometry
    return ModelArtifact(model, payload, metadata)


def predict(artifact, data, ids=None, *, train=False):
    """Return (len(ids),7) elapsed-day predictions; invalid dates become NaN."""
    from research.phenology_exploration.models import predict as physical_predict, residual_prediction
    _artifact(artifact)
    data = _data(data, artifact.model, artifact.metadata.get("input_channels"))
    ids = np.arange(len(data.raw)) if ids is None else _integer_array(ids, "ids")
    if (ids < 0).any() or (ids >= len(data.raw)).any():
        raise ValueError("ids must index raw weather units")
    if not len(ids):
        return np.empty((0, 7), float)
    p = artifact.payload
    if artifact.model in {"T", "nonlinear_T"}:
        result = physical_predict(p["physical"], data, ids, train=train)
    elif artifact.model == "thermal_transformer":
        from research.phenology_exploration.neural import ThermalTransformer, predict_network
        network = ThermalTransformer(p["initial"], 23)
        network.load_state_dict(p["state"])
        result = predict_network(network, data, ids, p["norm"], p["context"], train=train, device="cpu")
    else:
        base = physical_predict(p["base"], data, ids, train=train)
        result = base + p["strength"] * residual_prediction(p["residual"], data, ids, base, train=train)
        if artifact.model == "support_blend":
            from research.phenology_exploration.support_blend import support_weight, blend_dates
            weight, _ = support_weight(p["source_climate"], data.context[ids, :2])
            thermal = physical_predict(p["physical"], data, ids, train=train)
            length = (data.train_lengths if train else data.lengths)[ids, None]
            result = blend_dates(result, thermal, weight[:, None], length)
    length = (data.train_lengths if train else data.lengths)[ids, None]
    return np.where(np.isfinite(result) & (result >= 0) & (result < length), result, np.nan)


def _artifact(artifact, expected=None):
    if not isinstance(artifact, ModelArtifact) or artifact.schema_version != 1:
        raise ValueError("Unsupported model artifact/schema version")
    _name(artifact.model)
    if expected is not None and artifact.model != _name(expected):
        raise ValueError(f"Saved model {artifact.model!r} does not match requested model {expected!r}")


def save(artifact, path):
    """Save an artifact as a trusted Python pickle."""
    _artifact(artifact)
    path = Path(path)
    if path.suffix.lower() not in {".pkl", ".pickle"}:
        raise ValueError("New model artifacts are pickle files and require a .pkl or .pickle suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load(path, *, model=None, source_climate=None):
    """Load a new artifact or one selected model from a supplied archived folder.

    Pickle and Torch checkpoint files must come from a trusted source. Archived
    support_blend also needs native_inputs.npz/source_climate (or its argument).
    """
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    if path.is_dir():
        if model is None:
            raise ValueError("Choose model when loading an archived model directory")
        path = folder / ("thermal_transformer.pt" if model == "thermal_transformer" else "fits.pkl")
    if model is not None:
        _name(model)
    if path.suffix == ".pt":
        if model not in (None, "thermal_transformer"):
            raise ValueError("Torch checkpoint is a thermal_transformer model")
        import torch
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        artifact = ModelArtifact("thermal_transformer", {key: checkpoint[key] for key in ("state", "norm", "initial", "context")},
                                 dict(input_channels=3, archived_from=str(path), selected_steps=checkpoint.get("selected_steps")))
    else:
        with path.open("rb") as stream:
            saved = pickle.load(stream)
        if isinstance(saved, ModelArtifact):
            artifact = saved
        elif isinstance(saved, dict) and "models" in saved:
            if model is None:
                raise ValueError("Choose model when loading archived fits.pkl")
            if model == "thermal_transformer":
                return load(folder / "thermal_transformer.pt", model=model)
            fits = saved["models"]
            payload = {}
            if model in {"T", "nonlinear_T"}:
                payload["physical"] = fits[model]
            else:
                kind = "RF" if model == "residual_RF" else "MLP"
                payload.update(base=fits["T"], residual=fits[kind], strength=saved[f"{kind.lower()}_choice"]["strength"])
                if model == "support_blend":
                    if source_climate is None:
                        with np.load(folder / "native_inputs.npz", allow_pickle=False) as inputs:
                            source_climate = inputs["source_climate"].copy()
                    payload.update(physical=fits["nonlinear_T"], source_climate=np.asarray(source_climate))
            artifact = ModelArtifact(model, payload, dict(input_channels=3, archived_from=str(path)))
        else:
            raise ValueError("File does not contain a recognized model artifact")
    _artifact(artifact, model)
    return artifact


def load_inputs(path, *, weather="raw"):
    """Read NPZ weather/context/lengths with no pickled array loading."""
    with np.load(path, allow_pickle=False) as inputs:
        required = {weather, "context", "lengths"}
        if not required.issubset(inputs.files):
            raise ValueError(f"Input NPZ requires arrays {sorted(required)}")
        lengths = inputs["lengths"].copy()
        return SimpleNamespace(raw=inputs[weather].copy(), context=inputs["context"].copy(), lengths=lengths,
                               train_lengths=inputs["train_lengths"].copy() if "train_lengths" in inputs else lengths.copy())
