"""Generate a small synthetic source cohort for runnable interface examples."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/model_example"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    unit, day = np.arange(90), np.arange(84)
    low = 15 + np.sin(day / 18)[None, :] + (unit % 10)[:, None] / 8
    high = low + 9
    light = np.broadcast_to(13 + day[None, :] / 200, low.shape)
    context = np.column_stack([15 + unit / 90, 9 + (unit % 7) / 10,
                               unit / 100, np.sqrt(1 - (unit / 100) ** 2), light[:, 0]])
    np.savez_compressed(args.output / "inputs.npz", raw=np.stack([low, high, light], 2).astype(np.float32),
                        context=context.astype(np.float32), lengths=np.full(90, 84), train_lengths=np.full(90, 84))
    labels = pd.DataFrame([dict(u=int(u), head=h, observed=float(d + (u % 3)),
                                site=f"synthetic_site_{u}", site_year=f"synthetic_site_{u}|2000",
                                block=f"synthetic_block_{u // 10}", role="train" if u < 60 else "tune")
                           for u in unit for h, d in enumerate([5, 12, 22, 31, 41, 52, 63])])
    labels.to_csv(args.output / "labels.csv", index=False)
    metadata = dict(synthetic=True, scientific_validation=False, units=90, days=84, heads=7,
                    train_units=60, tune_units=30, train_blocks=6, tune_blocks=3,
                    purpose="Interface execution only; fabricated weather and phenology, no study observations")
    (args.output / "example.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(**metadata, inputs=str(args.output / "inputs.npz"), labels=str(args.output / "labels.csv"))))


if __name__ == "__main__":
    main()
