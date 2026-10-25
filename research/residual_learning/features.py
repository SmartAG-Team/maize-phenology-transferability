"""Truth-independent features and ordered point dates for residual learning."""

import numpy as np
import pandas as pd


STAGES7 = [
    "emergence", "stem_elongation", "tasseling", "flowering_onset",
    "milking", "dough", "maturity_bbch87",
]

_WINDOWS = [("prefix", None), ("trailing_7d", 7), ("trailing_14d", 14), ("trailing_30d", 30)]
_CHANNELS = [("tmin", "c"), ("tmax", "c"), ("daylength", "h"),
             ("tmean", "c"), ("diurnal_range", "c")]
_WEATHER_NAMES = [
    f"{channel}_{stat}_{unit}"
    for channel, unit in _CHANNELS for stat in ("mean", "std")
] + ["gdd_sum_cday", "tmax_ge30_fraction", "tmin_le8_fraction", "tmin_le0_fraction"]
FEATURE_NAMES = ["tp_day", "tp_preceding_day", "tp_gap_days", "tp_preceding_missing",
                 "sowing_doy_sin", "sowing_doy_cos"] + [f"stage_{stage}" for stage in STAGES7] + [
    f"{window}_{name}" for window, _ in _WINDOWS for name in _WEATHER_NAMES
]


def build_features(arrays, env, baseline_days):
    """Build float32 environment × seven-stage predictors without observations.

    ``X[..., 0:3]`` contains raw Tmin (°C), Tmax (°C) and daylength (h);
    ``G`` contains native daily GDD, and ``lengths`` is the usable prefix.
    Index zero is day one after sowing. Weather windows include days one
    through ``floor(own T–P day)``; trailing windows contain at most 7/14/30
    days and shorten at sowing. Standard deviations use population weighting.
    Extreme fractions use Tmax >= 30°C, Tmin <= 8°C and Tmin <= 0°C.

    Current and immediately preceding T–P dates, their signed gap, known
    sowing day-of-year harmonics (365.25-day period) and stage identity supply
    context. The first stage's preceding date is sowing (zero). An unresolved
    preceding date supplies zero date/gap plus a missing indicator. No later
    T–P stage, observed date, year identifier or location field is read.

    Own dates outside [1, usable weather length], or with nonfinite prefix
    weather, have all-NaN feature vectors. A fractional date beyond the usable
    length remains unresolved even when its floor fits. Inputs are read-only;
    weather summaries retain native units without normalization or refitting.
    """
    try:
        weather = np.asarray(arrays["X"])
        thermal = np.asarray(arrays["G"])
        lengths = np.asarray(arrays["lengths"], dtype=np.float64)
    except KeyError as error:
        raise ValueError("arrays require X, G and lengths") from error
    baseline = np.asarray(baseline_days, dtype=np.float64)
    if weather.ndim != 3 or weather.shape[2] != 3:
        raise ValueError("X must contain three daily weather channels")
    n, horizon, _ = weather.shape
    if thermal.shape != (n, horizon):
        raise ValueError("G and X must have matching environment/day axes")
    if lengths.shape != (n,) or not np.all(np.isfinite(lengths)) or np.any(
        (lengths < 0) | (lengths > horizon) | (lengths != np.floor(lengths))
    ):
        raise ValueError("lengths must be integer usable weather prefixes aligned with X")
    if baseline.shape != (n, len(STAGES7)):
        raise ValueError("baseline_days must have seven columns aligned with X")
    try:
        sowing = pd.to_datetime(pd.Series(env["sowing_date"]), errors="coerce")
    except KeyError as error:
        raise ValueError("known sowing_date is required") from error
    if len(sowing) != n or sowing.isna().any():
        raise ValueError("known sowing dates must be valid and aligned with X")
    angle = 2 * np.pi * sowing.dt.dayofyear.to_numpy(dtype=float) / 365.25
    resolved = np.isfinite(baseline) & (baseline >= 1) & (baseline <= lengths[:, None])
    preceding = np.zeros_like(baseline)
    preceding[:, 1:] = np.where(resolved[:, :-1], baseline[:, :-1], 0.)
    preceding_missing = np.zeros_like(resolved)
    preceding_missing[:, 1:] = ~resolved[:, :-1]
    gap = np.where(preceding_missing, 0., baseline - preceding)
    features = np.empty((n, len(STAGES7), len(FEATURE_NAMES)), dtype=np.float32)
    features[:, :, 0] = baseline
    features[:, :, 1] = preceding
    features[:, :, 2] = gap
    features[:, :, 3] = preceding_missing
    features[:, :, 4] = np.sin(angle)[:, None]
    features[:, :, 5] = np.cos(angle)[:, None]
    features[:, :, 6:13] = np.eye(len(STAGES7), dtype=np.float32)

    # Batches bound temporary float64 prefix tables independently of population.
    for first in range(0, n, 2048):
        last = min(first + 2048, n)
        x = np.asarray(weather[first:last], dtype=np.float64)
        g = np.asarray(thermal[first:last], dtype=np.float64)
        valid_weather = np.isfinite(x).all(axis=2) & np.isfinite(g)
        x = np.where(np.isfinite(x), x, 0.)
        g = np.where(np.isfinite(g), g, 0.)
        current = resolved[first:last].copy()
        end = np.where(current, np.floor(baseline[first:last]), 0).astype(np.int64)
        row = np.arange(last - first)[:, None]
        bad_prefix = np.column_stack([np.zeros(last - first), np.cumsum(~valid_weather, axis=1)])
        current &= bad_prefix[row, end] == 0
        starts = [np.zeros_like(end) if size is None else np.maximum(0, end - size)
                  for _, size in _WINDOWS]
        counts = [np.maximum(1, end - start) for start in starts]
        channels = [x[:, :, 0], x[:, :, 1], x[:, :, 2],
                    (x[:, :, 0] + x[:, :, 1]) / 2., x[:, :, 1] - x[:, :, 0]]
        chunk = features[first:last]
        for channel_index, values in enumerate(channels):
            total = _prefix_sum(values)
            square = _prefix_sum(values ** 2)
            for window_index, (start, count) in enumerate(zip(starts, counts)):
                mean = (total[row, end] - total[row, start]) / count
                variance = np.maximum(0., (square[row, end] - square[row, start]) / count - mean ** 2)
                offset = 13 + window_index * len(_WEATHER_NAMES) + 2 * channel_index
                chunk[:, :, offset] = mean
                chunk[:, :, offset + 1] = np.sqrt(variance)
        aggregates = [g, x[:, :, 1] >= 30., x[:, :, 0] <= 8., x[:, :, 0] <= 0.]
        for aggregate_index, values in enumerate(aggregates):
            total = _prefix_sum(values)
            for window_index, (start, count) in enumerate(zip(starts, counts)):
                value = total[row, end] - total[row, start]
                if aggregate_index:
                    value = value / count
                offset = 13 + window_index * len(_WEATHER_NAMES) + 10 + aggregate_index
                chunk[:, :, offset] = value
        chunk[~current] = np.nan
    return features, list(FEATURE_NAMES)


def _prefix_sum(values):
    return np.column_stack([np.zeros(values.shape[0]), np.cumsum(values, axis=1, dtype=np.float64)])


def project_dates(dates):
    """Return the equal-weight least-squares nondecreasing date projection.

    Pool adjacent violators over finite stage positions, then impose the
    elapsed-day lower bound of one. Missing positions remain missing; infinite
    inputs denote unresolved dates and become NaN. There is no upper weather
    horizon bound or artificial spacing between adjacent stage dates. An
    already ordered row whose dates are all at least one is preserved exactly.
    """
    values = np.asarray(dates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(STAGES7):
        raise ValueError("dates must be a matrix with seven stage columns")
    output = values.copy()
    output[~np.isfinite(output)] = np.nan
    for row in output:
        positions = np.flatnonzero(np.isfinite(row))
        if positions.size == 0:
            continue
        sequence = row[positions]
        if sequence[0] >= 1 and np.all(np.diff(sequence) >= 0):
            continue
        means, counts = [], []
        for value in sequence:
            means.append(float(value))
            counts.append(1)
            while len(means) >= 2 and means[-2] > means[-1]:
                count = counts[-2] + counts[-1]
                mean = (means[-2] * counts[-2] + means[-1] * counts[-1]) / count
                means[-2:] = [mean]
                counts[-2:] = [count]
        row[positions] = np.maximum(np.repeat(means, counts), 1.)
    return output
