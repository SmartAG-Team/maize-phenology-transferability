"""Direct signed date-residual learners with stage-equal MAE selection.

Only calibration residuals fit these models.  Tuning inputs choose an MLP
stopping epoch on the fixed physical starting dates; no test inputs are used.
Artifacts are ordinary dictionaries and contain no open files or device state.
"""

from __future__ import annotations

import copy

import numpy as np


N_STAGES = 7
HORIZON = 366.0


def _dates(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != N_STAGES:
        raise ValueError(f"{name} shape must be [N, {N_STAGES}]")
    return value


def _features(value, name, n_features=None):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or value.shape[1] != N_STAGES or value.shape[2] < 1:
        raise ValueError(f"{name} shape must be [N, {N_STAGES}, P] with P > 0")
    if n_features is not None and value.shape[2] != n_features:
        raise ValueError(f"{name} shape uses an incompatible feature axis")
    return value


def macro_mae(truth, dates, base=None):
    """Average stage MAEs, with a fixed penalty for unresolved point dates.

    Zero or nonfinite truth is missing.  Supplying ``base`` restricts support
    to observed targets whose original physical prediction resolved in the
    366-day horizon.  Missing/invalid corrections on that support cost 366
    days; they never improve the score by disappearing from its denominator.
    """
    truth = _dates(truth, "truth")
    dates = _dates(dates, "dates")
    if truth.shape != dates.shape:
        raise ValueError("truth and dates shape must match")
    support = np.isfinite(truth) & (truth > 0)
    if base is not None:
        base = _dates(base, "base")
        if base.shape != truth.shape:
            raise ValueError("base and truth shape must match")
        support &= np.isfinite(base) & (base > 0) & (base <= HORIZON)
    resolved = np.isfinite(dates) & (dates > 0) & (dates <= HORIZON)
    errors = np.full(truth.shape, HORIZON)
    paired = support & resolved
    errors[paired] = np.abs(dates[paired] - truth[paired])
    stage_scores = [errors[support[:, j], j].mean() for j in range(N_STAGES)
                    if support[:, j].any()]
    return float(np.mean(stage_scores)) if stage_scores else float("nan")


def correct_dates(base, residual, shrinkage):
    """Apply a signed correction, preserving identity and baseline missingness.

    Nonzero corrections receive the shared fixed order projection.  Values
    beyond the weather horizon remain numeric for unresolved-date accounting.
    """
    base = _dates(base, "base")
    residual = _dates(residual, "residual")
    if base.shape != residual.shape:
        raise ValueError("base and residual shape must match")
    shrinkage = float(shrinkage)
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be finite and in [0, 1]")
    if shrinkage == 0:
        return base.copy()
    from .features import project_dates

    with np.errstate(invalid="ignore", over="ignore"):
        raw = base + shrinkage * residual
    raw[~np.isfinite(base)] = np.nan
    projected = project_dates(raw)
    projected[~np.isfinite(base)] = np.nan
    return projected


def _normalization(points, n_features):
    if len(points) == 0:
        return {"mean": np.zeros(n_features), "scale": np.ones(n_features)}
    mean = points.mean(axis=0)
    scale = points.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {"mean": mean, "scale": scale}


def _network(n_features, config):
    import torch

    layers = []
    n_inputs = n_features + N_STAGES
    for _ in range(config["hidden_layers"]):
        layers.extend([torch.nn.Linear(n_inputs, config["width"]), torch.nn.ReLU()])
        n_inputs = config["width"]
    layers.append(torch.nn.Linear(n_inputs, 1))
    return torch.nn.Sequential(*layers)


def _network_inputs(points, stages, normalization):
    normalized = (points - normalization["mean"]) / normalization["scale"]
    one_hot = np.eye(N_STAGES)[stages]
    return np.concatenate([normalized, one_hot], axis=1).astype(np.float32)


def _stage_balanced_l1(prediction, target, stages, stage_counts):
    """Unbiased minibatch estimate of the calibration stage-equal L1 loss.

    ``stage_counts`` counts finite labelled calibration points, not minibatch
    observations.  Global weights prevent common stages dominating training.
    Missing targets have zero contribution, including zero gradient.
    """
    import torch

    valid = torch.isfinite(target)
    if not valid.any():
        return prediction[valid].sum()
    counts = stage_counts.to(device=prediction.device, dtype=prediction.dtype)
    active = (counts > 0).sum()
    selected_counts = counts[stages[valid]]
    if (selected_counts <= 0).any():
        raise ValueError("stage_counts must include every observed training stage")
    weights = counts.sum() / (active * selected_counts)
    return (torch.abs(prediction[valid] - target[valid]) * weights).mean()


def _mlp_predict(network, features, artifact):
    import torch

    valid = np.isfinite(features).all(axis=2)
    valid &= np.asarray(artifact["fit_counts"]) > 0
    result = np.full(features.shape[:2], np.nan)
    rows, stages = np.nonzero(valid)
    if len(rows) == 0:
        return result
    inputs = _network_inputs(features[rows, stages], stages, artifact["normalization"])
    batch_size = int(artifact["config"]["batch_size"])
    network.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            end = start + batch_size
            values = network(torch.from_numpy(inputs[start:end])).squeeze(-1).numpy()
            values[~np.isfinite(values)] = np.nan
            result[rows[start:end], stages[start:end]] = values
    return result


def _fit_mlp(artifact, features, targets, tuning_features, tuning_truth, tuning_base, mask):
    import torch

    torch.set_num_threads(1)
    config = artifact["config"]
    if not mask.any():
        artifact["state"] = None
        return artifact
    if not np.isfinite(macro_mae(tuning_truth, tuning_base, tuning_base)):
        raise ValueError("MLP stopping selection requires observed tuning targets on valid base support")
    rows, stages = np.nonzero(mask)
    inputs = torch.from_numpy(_network_inputs(features[rows, stages], stages, artifact["normalization"]))
    labels = torch.from_numpy(targets[rows, stages].astype(np.float32))
    stage_tensor = torch.from_numpy(stages)
    counts = torch.from_numpy(artifact["fit_counts"].astype(np.float32))
    rng = np.random.default_rng(artifact["seed"])
    best_score = float("inf")
    remaining = config["patience"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(artifact["seed"])
        network = _network(artifact["n_features"], config)
        optimizer = torch.optim.Adam(network.parameters(), lr=config["learning_rate"])
        for epoch in range(1, config["max_epochs"] + 1):
            network.train()
            order = rng.permutation(len(rows))
            for start in range(0, len(order), config["batch_size"]):
                batch = torch.from_numpy(order[start:start + config["batch_size"]])
                optimizer.zero_grad(set_to_none=True)
                prediction = network(inputs[batch]).squeeze(-1)
                loss = _stage_balanced_l1(prediction, labels[batch], stage_tensor[batch], counts)
                loss.backward()
                optimizer.step()
            network.eval()
            with torch.no_grad():
                training_loss = float(_stage_balanced_l1(network(inputs).squeeze(-1), labels, stage_tensor, counts))
            residual = _mlp_predict(network, tuning_features, artifact)
            dates = correct_dates(tuning_base, residual, config["shrinkage"])
            score = macro_mae(tuning_truth, dates, tuning_base)
            artifact["history"].append({"epoch": epoch, "training_l1": training_loss, "tuning_mae": score})
            if score < best_score:
                best_score = score
                artifact["best_epoch"] = epoch
                artifact["state"] = {key: value.detach().numpy().copy()
                                     for key, value in network.state_dict().items()}
                remaining = config["patience"]
            else:
                remaining -= 1
                if remaining == 0:
                    break
    artifact["best_tuning_mae"] = best_score
    return artifact


def fit_model(kind, features, residual_targets, tuning_features, tuning_truth,
              tuning_base, config, seed):
    """Fit bias, stage-specific absolute-error trees, or a shared L1 MLP.

    Residual labels are observed minus out-of-fit physical elapsed dates.
    Bias estimates use every finite residual independently for each stage.
    Learned feature responses require both finite labels and finite features.
    """
    if kind not in {"bias", "trees", "mlp"}:
        raise ValueError("kind must be 'bias', 'trees', or 'mlp'")
    features = _features(features, "features")
    targets = _dates(residual_targets, "residual_targets")
    if targets.shape != features.shape[:2]:
        raise ValueError("residual_targets and features shape must match")
    tuning_features = _features(tuning_features, "tuning_features", features.shape[2])
    tuning_truth = _dates(tuning_truth, "tuning_truth")
    tuning_base = _dates(tuning_base, "tuning_base")
    if tuning_truth.shape != tuning_features.shape[:2] or tuning_base.shape != tuning_truth.shape:
        raise ValueError("tuning arrays shape must match")
    config = copy.deepcopy(dict(config))
    finite_features = np.isfinite(features).all(axis=2)
    finite_labels = np.isfinite(targets)
    mask = finite_features & finite_labels
    counts = finite_labels.sum(axis=0) if kind == "bias" else mask.sum(axis=0)
    artifact = {"kind": kind, "seed": int(seed), "config": config,
                "n_features": features.shape[2], "fit_counts": counts,
                "calibration_label_counts": finite_labels.sum(axis=0),
                "missing_feature_counts": (finite_labels & ~finite_features).sum(axis=0),
                "normalization": _normalization(features[mask], features.shape[2]),
                "history": [], "best_epoch": None}
    if kind == "bias":
        artifact["bias"] = np.array([np.median(targets[finite_labels[:, j], j])
                                     if counts[j] else np.nan for j in range(N_STAGES)])
    elif kind == "trees":
        from sklearn.ensemble import HistGradientBoostingRegressor
        from threadpoolctl import threadpool_limits

        for key, value in {"max_iter": 120, "max_leaf_nodes": 7, "l2_regularization": 1.0}.items():
            config.setdefault(key, value)
        artifact["models"] = []
        with threadpool_limits(limits=1):
            for j in range(N_STAGES):
                if counts[j] == 0:
                    artifact["models"].append(None)
                    continue
                tree = HistGradientBoostingRegressor(
                    loss="absolute_error", max_iter=int(config["max_iter"]),
                    max_leaf_nodes=int(config["max_leaf_nodes"]),
                    l2_regularization=float(config["l2_regularization"]),
                    early_stopping=False, random_state=int(seed))
                tree.fit(features[mask[:, j], j], targets[mask[:, j], j])
                artifact["models"].append(tree)
    else:
        defaults = {"width": 32, "hidden_layers": 2, "learning_rate": 0.001,
                    "max_epochs": 80, "patience": 10, "batch_size": 1024, "shrinkage": 1.0}
        for key, value in defaults.items():
            config.setdefault(key, value)
        for key in ["width", "hidden_layers", "max_epochs", "patience", "batch_size"]:
            config[key] = int(config[key])
            if config[key] < 1:
                raise ValueError(f"{key} must be positive")
        config["learning_rate"] = float(config["learning_rate"])
        if not np.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
            raise ValueError("learning_rate must be finite and positive")
        config["shrinkage"] = float(config["shrinkage"])
        if not np.isfinite(config["shrinkage"]) or not 0 <= config["shrinkage"] <= 1:
            raise ValueError("shrinkage must be finite and in [0, 1]")
        return _fit_mlp(artifact, features, targets, tuning_features, tuning_truth, tuning_base, mask)
    return artifact


def predict_residual(model, features):
    """Return signed residual days, with missing features/stages left unresolved."""
    features = _features(features, "features", model["n_features"])
    valid = np.isfinite(features).all(axis=2)
    result = np.full(features.shape[:2], np.nan)
    if model["kind"] == "bias":
        values = np.broadcast_to(model["bias"], result.shape)
        result[valid] = values[valid]
    elif model["kind"] == "trees":
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            for j, tree in enumerate(model["models"]):
                if tree is not None and valid[:, j].any():
                    result[valid[:, j], j] = tree.predict(features[valid[:, j], j])
    elif model["kind"] == "mlp":
        if model["state"] is None:
            return result
        import torch

        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            network = _network(model["n_features"], model["config"])
        network.load_state_dict({key: torch.from_numpy(value) for key, value in model["state"].items()})
        return _mlp_predict(network, features, model)
    else:
        raise ValueError("unknown model kind")
    return result
