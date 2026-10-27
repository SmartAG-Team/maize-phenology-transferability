"""Command-line entry points shared by the six model directories."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .api import MODEL_NAMES, FitOptions, fit, load, load_inputs, predict, save


def main(pinned_model=None):
    parser = argparse.ArgumentParser(description="Train or call one maize phenology model")
    commands = parser.add_subparsers(dest="command", required=True)
    for operation in ("train", "predict"):
        command = commands.add_parser(operation)
        if pinned_model is None:
            command.add_argument("--model", choices=MODEL_NAMES, required=True)
        command.add_argument("--inputs", required=True, type=Path)
        command.add_argument("--weather", default="raw", help="NPZ weather array name: raw, era5 or dwd")
        command.add_argument("--output", required=True, type=Path)
        if operation == "train":
            command.add_argument("--labels", required=True, type=Path, help="CSV with explicit role=train/tune")
            command.add_argument("--seed", type=int, default=17)
            command.add_argument("--mlp-steps", nargs="+", type=int, default=list(FitOptions().mlp_steps))
            command.add_argument("--transformer-fixed-steps", type=int, default=None,
                                 help="Declared fixed budget; bypass source checkpoint selection")
        else:
            command.add_argument("--artifact", required=True, type=Path)
            command.add_argument("--train-prefix", action="store_true")
    args = parser.parse_args()
    name = pinned_model or args.model
    try:
        data = load_inputs(args.inputs, weather=args.weather)
        if args.command == "train":
            if args.output.suffix.lower() not in {".pkl", ".pickle"}:
                raise ValueError("New model artifacts are pickle files and require a .pkl or .pickle suffix")
            labels = pd.read_csv(args.labels)
            if "role" not in labels or labels.role.isna().any() or not labels.role.isin(["train", "tune"]).all():
                raise ValueError("Training CSV role must be explicitly train or tune; target rows are forbidden")
            train = labels[labels.role.eq("train")].copy()
            tune = labels[labels.role.eq("tune")].copy()
            artifact = fit(name, data, train, None if tune.empty else tune, seed=args.seed,
                           options=FitOptions(mlp_steps=tuple(args.mlp_steps), transformer_fixed_steps=args.transformer_fixed_steps))
            save(artifact, args.output)
            result = dict(model=name, artifact=str(args.output), selected=artifact.metadata["selected"],
                          source_units=artifact.metadata["source_units"])
        else:
            artifact = load(args.artifact, model=name)
            values = predict(artifact, data, train=args.train_prefix)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            result = dict(model=name, predictions=str(args.output), shape=list(values.shape),
                          resolved=int(np.isfinite(values).sum()))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(result))
