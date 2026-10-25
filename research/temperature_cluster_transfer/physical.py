"""Source-only MAE calibration of the shared temperature–photoperiod model.

Optimization uses a piecewise-linear threshold-crossing-date L1 surrogate.
Day zero is sowing with zero accumulated progress; stored weather starts on
day one. Unreached training crossings extend beyond the supplied horizon at
the mean positive daily progress rate. A 0.001 progress-unit/day denominator
floor keeps zero-rate training examples finite. This numerical extension is
neither a predicted event date nor a discrete-MAE gradient. Inference uses
the unchanged process and exact first daily crossings; unreached dates remain
NaN and cost 366 days in the source-tuning checkpoint selection score.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from maize_clock.transfer_artifacts import (
    atomic_json_save,
    atomic_torch_save,
    dependency_fingerprint,
    file_sha256,
)
from maize_clock.transfer_fit import clip_gradients, make_model, restore

N_STAGES = 7
HORIZON = 366
RATE_FLOOR = 1e-3
CONFIG = {"mode": "standalone_photo", "learning_rate": 0.01}
OBJECTIVE = "stage_equal_L1_interpolated_threshold_crossing_days"
SELECTION_OBJECTIVE = "stage_equal_discrete_median_date_MAE"


def interpolated_crossings(progress: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """Return locally differentiable dates, including training-only extensions.

    The first crossing interval is selected discretely. Gradients within that
    interval pass through its two cumulative-progress values and threshold.
    The mean positive increment, and hence the unreached extension, also has
    local gradients; positive-increment membership is discrete.
    """
    if progress.ndim != 2 or progress.shape[1] < 1 or not progress.is_floating_point():
        raise ValueError("progress must be a floating [N, H] tensor with H > 0")
    if thresholds.ndim != 1 or len(thresholds) != N_STAGES:
        raise ValueError("Seven thresholds required")
    thresholds = thresholds.to(device=progress.device, dtype=progress.dtype)
    if not torch.isfinite(progress).all() or (progress < 0).any() or (torch.diff(progress, dim=1) < 0).any():
        raise ValueError("Finite nonnegative nondecreasing progress required")
    if not torch.isfinite(thresholds).all() or (thresholds <= 0).any() or (torch.diff(thresholds) <= 0).any():
        raise ValueError("Finite positive increasing thresholds required")
    n, horizon = progress.shape
    target = thresholds[None, :].expand(n, -1).contiguous()
    first = torch.searchsorted(progress.contiguous(), target, right=False)
    origin = torch.cat((torch.zeros_like(progress[:, :1]), progress), dim=1)
    index = first.clamp_max(horizon - 1)
    before = origin.gather(1, index)
    after = origin.gather(1, index + 1)
    slope = (after - before).clamp_min(torch.finfo(progress.dtype).eps)
    within = index.to(progress.dtype) + (target - before) / slope
    increments = torch.diff(origin, dim=1)
    positive = increments > 0
    mean_rate = torch.where(positive, increments, torch.zeros_like(increments)).sum(1)
    mean_rate = mean_rate / positive.sum(1).clamp_min(1)
    beyond = horizon + (target - progress[:, -1:]) / mean_rate.clamp_min(RATE_FLOOR)[:, None]
    return torch.where(first < horizon, within, beyond)


def stage_equal_l1(predicted, truth, stage_counts=None, population_size=None):
    """Stage-equal crossing-date L1 with full-calibration minibatch weights.

    With global counts C_j and population N, a batch of B environments uses
    N/(B J) * sum_j sum_i |date_ij - truth_ij| / C_j. Its epoch aggregate
    preserves equal stage weight despite missing observations and uneven
    minibatch support. Without global counts the current tensor is the full
    population. Missing or out-of-horizon truth contributes zero gradient.
    """
    if predicted.ndim != 2 or predicted.shape[1] != N_STAGES:
        raise ValueError("predicted shape must be [N, 7]")
    truth = torch.as_tensor(truth, dtype=predicted.dtype, device=predicted.device)
    if truth.shape != predicted.shape or not torch.isfinite(predicted).all():
        raise ValueError("Matching truth and finite predictions required")
    valid = torch.isfinite(truth) & (truth > 0) & (truth <= HORIZON)
    safe_truth = torch.where(valid, truth, predicted.detach())
    errors = torch.where(valid, (predicted - safe_truth).abs(), torch.zeros_like(predicted))
    if stage_counts is None:
        counts = valid.sum(0).to(predicted.dtype)
        population_size = len(predicted)
    else:
        counts = torch.as_tensor(stage_counts, dtype=predicted.dtype, device=predicted.device)
        if population_size is None or population_size < len(predicted):
            raise ValueError("Full calibration population size required with stage counts")
    if counts.shape != (N_STAGES,) or not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Seven finite nonnegative stage counts required")
    active = counts > 0
    if not active.any() or not len(predicted):
        return errors.sum()
    per_stage = errors.sum(0) / counts.clamp_min(1)
    return per_stage[active].sum() * (float(population_size) / len(predicted) / int(active.sum()))


def stage_equal_mae(truth, dates):
    """Exact-date stage macro MAE; invalid predictions incur 366 days."""
    truth = np.asarray(truth, dtype=float)
    dates = np.asarray(dates, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != N_STAGES or dates.shape != truth.shape:
        raise ValueError("truth and dates must have matching [N, 7] shapes")
    support = np.isfinite(truth) & (truth > 0) & (truth <= HORIZON)
    resolved = np.isfinite(dates) & (dates >= 1) & (dates <= HORIZON)
    errors = np.full(truth.shape, float(HORIZON))
    keep = support & resolved
    errors[keep] = np.abs(dates[keep] - truth[keep])
    stages = [errors[support[:, j], j].mean() for j in range(N_STAGES) if support[:, j].any()]
    return float(np.mean(stages)) if stages else float("nan")


def _normalizers(metadata):
    mean = np.asarray(metadata["feature_mean"], np.float32)
    std = np.asarray(metadata["feature_std"], np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Three finite source means and positive source scales required")
    return mean, std


def predict_points(model, metadata, arrays, ids):
    """Use the existing exact first-crossing predictor without tail extension."""
    from research.residual_learning.study import predict_tp

    _normalizers(metadata)
    return predict_tp(model, metadata, arrays, ids)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _array_digest(arrays, ids):
    digest = hashlib.sha256()
    for name in ("X", "G", "Y", "lengths"):
        selected = arrays[name][ids]
        if name == "Y":
            selected = selected[:, :N_STAGES]
        selected = np.ascontiguousarray(selected)
        digest.update(name.encode())
        digest.update(selected.dtype.str.encode())
        digest.update(json.dumps(selected.shape).encode())
        digest.update(selected.tobytes())
    return digest.hexdigest()


def _indices(ids, size, name):
    ids = np.asarray(ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not len(ids):
        raise ValueError(f"Nonempty integer {name} ids required")
    if (ids < 0).any() or (ids >= size).any() or len(np.unique(ids)) != len(ids):
        raise ValueError(f"Unique in-range {name} ids required")
    return ids.astype(np.int64, copy=False)


def _validate_fit_inputs(arrays, env, metadata, fit_ids, tune_ids, epochs, batch, patience):
    if any(not isinstance(v, (int, np.integer)) or v < 1 for v in (epochs, batch, patience)):
        raise ValueError("Positive integer epoch, batch and patience settings required")
    n = len(env)
    if arrays["X"].shape != (n, HORIZON, 3) or arrays["G"].shape != (n, HORIZON):
        raise ValueError("Source X/G shapes must be [N, 366, 3]/[N, 366]")
    if arrays["Y"].ndim != 2 or arrays["Y"].shape[0] != n or arrays["Y"].shape[1] < N_STAGES or arrays["lengths"].shape != (n,):
        raise ValueError("Source target or weather-length shape mismatch")
    fit_ids = _indices(fit_ids, n, "fit")
    tune_ids = _indices(tune_ids, n, "tune")
    if np.intersect1d(fit_ids, tune_ids).size:
        raise ValueError("Fitting and source tuning ids overlap")
    mean, std = _normalizers(metadata)
    selected = np.r_[fit_ids, tune_ids]
    lengths = arrays["lengths"][selected]
    if not np.isfinite(lengths).all() or (lengths < 1).any() or (lengths > HORIZON).any() or (lengths != np.floor(lengths)).any():
        raise ValueError("Valid source weather prefix lengths required")
    prefix = np.arange(HORIZON)[None, :] < lengths[:, None]
    weather = arrays["X"][selected][prefix]
    gdd = arrays["G"][selected][prefix]
    if not np.isfinite(weather).all() or not np.isfinite(gdd).all() or (gdd < 0).any():
        raise ValueError("Finite source-prefix weather and nonnegative GDD required")
    if not np.isfinite((weather - mean) / std).all():
        raise ValueError("Nonfinite source weather normalization")
    supports = []
    for ids in (fit_ids, tune_ids):
        truth = arrays["Y"][ids, :N_STAGES]
        counts = (np.isfinite(truth) & (truth > 0) & (truth <= HORIZON)).sum(0)
        if (counts == 0).any():
            raise ValueError("Every primary stage requires source fitting and tuning support")
        supports.append(counts)
    # The existing constructor also validates seven positive increasing gaps.
    make_model(metadata, CONFIG)
    _digest(metadata)
    return fit_ids, tune_ids, supports


def _request(arrays, env, metadata, fit_ids, tune_ids, seed, phase, input_sha256, epochs, batch, patience):
    from maize_clock import transfer_attention, transfer_artifacts, transfer_fit, ordered_process
    from research.residual_learning import study

    paths = [Path(__file__), *(Path(module.__file__) for module in
                             (transfer_attention, transfer_artifacts, transfer_fit, ordered_process, study))]
    identity = env["environment_id_model"].astype(str).to_numpy() if "environment_id_model" in env else env.index.astype(str).to_numpy()
    return dict(
        config=dict(CONFIG), seed=int(seed), phase=str(phase), input_sha256=input_sha256,
        metadata_sha256=_digest(metadata), objective=OBJECTIVE, selection_objective=SELECTION_OBJECTIVE,
        fit_index_sha256=hashlib.sha256(fit_ids.astype("<i8").tobytes()).hexdigest(),
        tune_index_sha256=hashlib.sha256(tune_ids.astype("<i8").tobytes()).hexdigest(),
        fit_array_sha256=_array_digest(arrays, fit_ids), tune_array_sha256=_array_digest(arrays, tune_ids),
        fit_environment_sha256=_digest(identity[fit_ids].tolist()),
        tune_environment_sha256=_digest(identity[tune_ids].tolist()),
        source_code_sha256=dependency_fingerprint(paths), maximum_epochs=int(epochs), batch=int(batch),
        patience=int(patience), frozen_parameters=["raw_sigma"], test_labels_used=False,
    )


def _check_signature(saved, requested):
    for key, value in requested.items():
        if saved.get(key) != value:
            raise ValueError("Point-fit signature mismatch: " + key)
    if "metadata" in saved and _digest(saved["metadata"]) != requested["metadata_sha256"]:
        raise ValueError("Point-fit metadata bytes changed")


def _validate_completed(path, requested, arrays, tune_ids):
    report = json.loads(path.with_suffix(".json").read_text())
    _check_signature(report, requested)
    checkpoint = path.with_suffix(".pt")
    if not checkpoint.is_file() or file_sha256(checkpoint) != report.get("checkpoint_sha256"):
        raise ValueError("Point checkpoint bytes changed")
    model, saved = restore(checkpoint)
    _check_signature(saved, requested)
    history = report["history"]
    if not history or [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("Invalid point-fitting history")
    if report["epochs_run"] != len(history) or not 1 <= report["best_epoch"] <= len(history) <= requested["maximum_epochs"]:
        raise ValueError("Invalid point-fitting epochs")
    if not all(math.isfinite(row["tuning_mae"]) and math.isfinite(row["calibration_l1_surrogate"]) for row in history):
        raise ValueError("Nonfinite point-fitting history")
    best = min(history, key=lambda row: row["tuning_mae"])
    if best["epoch"] != report["best_epoch"] or saved["best_epoch"] != report["best_epoch"] or saved["tuning_mae"] != report["tuning_mae"]:
        raise ValueError("Point-fit best-checkpoint history mismatch")
    observed_score = stage_equal_mae(arrays["Y"][tune_ids, :N_STAGES], predict_points(model, saved["metadata"], arrays, tune_ids))
    if not math.isfinite(observed_score) or not math.isclose(observed_score, report["tuning_mae"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Point checkpoint tuning MAE mismatch")
    return report


def fit_point(arrays, env, metadata, fit_ids, tune_ids, seed, path, phase, input_sha256,
              epochs=40, batch=128, patience=8):
    """Fit source crossings and stop by source-only exact discrete stage MAE.

    Complete reports are immutable and reused only after signature, checkpoint
    checksum and source-tuning reconstruction checks. Epoch resume checkpoints
    include optimizer/RNG state and have separate checksum-bound sidecars.
    """
    fit_ids, tune_ids, supports = _validate_fit_inputs(arrays, env, metadata, fit_ids, tune_ids, epochs, batch, patience)
    requested = _request(arrays, env, metadata, fit_ids, tune_ids, seed, phase, input_sha256, epochs, batch, patience)
    path = Path(path)
    if path.with_suffix(".json").exists():
        return _validate_completed(path, requested, arrays, tune_ids)
    resume_path = path.with_name(path.name + "_resume.pt")
    resume_sidecar = path.with_name(path.name + "_resume.json")
    if resume_path.exists() != resume_sidecar.exists():
        raise ValueError("Incomplete point-fit resume checksum sidecar")
    if path.with_suffix(".pt").exists():
        _check_signature(torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=False), requested)
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    model = make_model(metadata, CONFIG)
    model.raw_sigma.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=CONFIG["learning_rate"])
    rng = np.random.default_rng(seed)
    history, best, wait, best_epoch = [], float("inf"), 0, 0
    if resume_path.exists():
        sidecar = json.loads(resume_sidecar.read_text())
        _check_signature(sidecar, requested)
        if file_sha256(resume_path) != sidecar.get("checkpoint_sha256"):
            raise ValueError("Resume checkpoint bytes changed")
        if not path.with_suffix(".pt").is_file() or file_sha256(path.with_suffix(".pt")) != sidecar.get("best_checkpoint_sha256"):
            raise ValueError("Best point checkpoint bytes changed during resume")
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        _check_signature(saved, requested)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"])
        history, best, wait, best_epoch = saved["history"], saved["best"], saved["wait"], saved["best_epoch"]
        if history != sidecar["history"]:
            raise ValueError("Resume history mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    mean, std = _normalizers(metadata)
    counts = torch.as_tensor(supports[0], dtype=torch.float64)
    for epoch in range(len(history) + 1, epochs + 1) if wait < patience else []:
        started = time.perf_counter()
        model.train()
        order = rng.permutation(fit_ids)
        total, maximum_gradient = 0.0, 0.0
        for start in range(0, len(order), batch):
            ids = order[start:start + batch]
            prefix = np.arange(HORIZON)[None, :] < arrays["lengths"][ids, None]
            x = np.where(prefix[:, :, None], arrays["X"][ids], mean)
            x = (x - mean) / std
            g = np.where(prefix, arrays["G"][ids], 0.0)
            optimizer.zero_grad(set_to_none=True)
            progress = model.process_progress(torch.as_tensor(x), torch.as_tensor(g))
            dates = interpolated_crossings(progress, model.thresholds())
            loss = stage_equal_l1(dates, arrays["Y"][ids, :N_STAGES], counts, population_size=len(fit_ids))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite point-calibration loss")
            loss.backward()
            maximum_gradient = max(maximum_gradient, clip_gradients(trainable, 1.0))
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        score = stage_equal_mae(arrays["Y"][tune_ids, :N_STAGES], predict_points(model, metadata, arrays, tune_ids))
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite source-tuning point MAE")
        history.append(dict(epoch=epoch, calibration_l1_surrogate=total / len(order), tuning_mae=score,
                            max_gradient_norm=maximum_gradient, seconds=time.perf_counter() - started))
        if score < best:
            best, best_epoch, wait = score, epoch, 0
            atomic_torch_save(dict(requested, metadata=copy.deepcopy(metadata), state_dict=copy.deepcopy(model.state_dict()),
                                  best_epoch=best_epoch, tuning_mae=best), path.with_suffix(".pt"))
        else:
            wait += 1
        atomic_torch_save(dict(requested, metadata=copy.deepcopy(metadata), state_dict=model.state_dict(),
                              optimizer=optimizer.state_dict(), rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                              history=history, best=best, wait=wait, best_epoch=best_epoch), resume_path)
        atomic_json_save(dict(requested, metadata=metadata, history=history,
                              checkpoint_sha256=file_sha256(resume_path),
                              best_checkpoint_sha256=file_sha256(path.with_suffix(".pt"))), resume_sidecar)
        print(json.dumps({"point_fit": str(path), "epoch": epoch, "source_tuning_mae": score,
                          "seconds": history[-1]["seconds"]}), flush=True)
        if wait >= patience:
            break
    report = dict(requested, metadata=metadata, history=history, tuning_mae=best, best_epoch=best_epoch,
                  epochs_run=len(history), fit_complete=True, training_environments=len(fit_ids),
                  tuning_environments=len(tune_ids), fit_stage_support=supports[0].tolist(),
                  tune_stage_support=supports[1].tolist(), actual_updates=len(history) * math.ceil(len(fit_ids) / batch),
                  parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                  trainable_parameter_count=sum(parameter.numel() for parameter in trainable),
                  elapsed_seconds=sum(row["seconds"] for row in history),
                  crossing_surrogate="Adjacent-day linear interpolation from sowing day zero; training-only mean-positive-rate tail extension",
                  extension_rate_floor=RATE_FLOOR, discrete_unresolved_penalty_days=HORIZON,
                  checkpoint_sha256=file_sha256(path.with_suffix(".pt")))
    atomic_json_save(report, path.with_suffix(".json"))
    verified = _validate_completed(path, requested, arrays, tune_ids)
    resume_path.unlink(missing_ok=True)
    resume_sidecar.unlink(missing_ok=True)
    return verified
