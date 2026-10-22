from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.baselines import persistence_forecast
from cwfc.metrics import mae


TARGETS = ("active_vm_count", "aggregate_cpu_utilization")
OUTPUT_COLUMNS = (
    "timestamp",
    "active_vm_count",
    "aggregate_cpu_utilization",
)


def build_targets(cpu: pd.DataFrame) -> pd.DataFrame:
    """Build the two required aggregate target series from CPU readings."""
    required = {"timestamp", "vm_id", "avg_cpu"}
    missing = required.difference(cpu.columns)
    if missing:
        raise ValueError(f"CPU data is missing required columns: {sorted(missing)}")

    if cpu.empty:
        raise ValueError("CPU data must not be empty")

    if cpu[["timestamp", "vm_id", "avg_cpu"]].isna().any().any():
        raise ValueError("Required CPU fields contain missing values")

    if cpu.duplicated(["vm_id", "timestamp"]).any():
        raise ValueError("Duplicate VM/timestamp readings found")

    cpu_values = pd.to_numeric(cpu["avg_cpu"], errors="raise").to_numpy(float)
    if not np.isfinite(cpu_values).all():
        raise ValueError("avg_cpu contains non-finite values")
    if ((cpu_values < 0) | (cpu_values > 100)).any():
        raise ValueError("avg_cpu values must be within [0, 100]")

    targets = (
        cpu.groupby("timestamp", sort=True)
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reset_index()
    )

    if targets["timestamp"].duplicated().any():
        raise ValueError("Target timestamps must be unique")

    return targets


def _ar1_forecast(train_values: np.ndarray, horizon: int) -> np.ndarray:
    """Fit a first-order autoregression and forecast recursively."""
    if len(train_values) < 3:
        return np.full(horizon, float(train_values[-1]))

    x = train_values[:-1]
    y = train_values[1:]
    slope, intercept = np.polyfit(x, y, 1)

    forecast = []
    previous = float(train_values[-1])

    for _ in range(horizon):
        next_value = float(intercept + slope * previous)
        forecast.append(next_value)
        previous = next_value

    return np.asarray(forecast, dtype=float)


def _forecast(train: pd.DataFrame, template: pd.DataFrame, model: str,
              target: str) -> np.ndarray:
    values = train[target].to_numpy(dtype=float)
    horizon = len(template)

    if model == "persistence":
        return persistence_forecast(train, template, target).to_numpy(float)

    if model == "training_mean":
        return np.full(horizon, float(values.mean()))

    if model == "autoregressive_1":
        return _ar1_forecast(values, horizon)

    raise ValueError(f"Unknown forecasting model: {model}")


def select_cpu_model(train: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Choose a CPU model using expanding-window validation within train."""
    target = "aggregate_cpu_utilization"
    horizon = min(24, max(1, len(train) // 4))
    initial_train_size = max(3, len(train) - 4 * horizon)
    step = max(1, horizon // 2)

    origins = list(range(
        initial_train_size,
        len(train) - horizon + 1,
        step,
    ))

    if not origins:
        return "persistence", pd.DataFrame(
            [{"model": "persistence", "mean_fold_MAE": np.nan}]
        )

    rows = []
    for fold, origin in enumerate(origins, start=1):
        fold_train = train.iloc[:origin].reset_index(drop=True)
        valid = train.iloc[origin:origin + horizon].reset_index(drop=True)
        y_true = valid[target].to_numpy(dtype=float)

        for model in ("persistence", "training_mean", "autoregressive_1"):
            pred = _forecast(fold_train, valid, model, target)
            rows.append({
                "fold": fold,
                "model": model,
                "MAE": mae(y_true, pred),
            })

    details = pd.DataFrame(rows)
    summary = (
        details.groupby("model", as_index=False)
        .agg(mean_fold_MAE=("MAE", "mean"))
        .sort_values(["mean_fold_MAE", "model"])
        .reset_index(drop=True)
    )
    return str(summary.iloc[0]["model"]), summary


def make_predictions(cpu: pd.DataFrame) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    targets = build_targets(cpu)

    # The submission contract requires exactly 24 holdout observations.
    holdout_size = 24
    if len(targets) <= holdout_size + 2:
        raise ValueError(
            "At least 27 timestamps are required: "
            "24 holdout timestamps and at least 3 historical observations"
        )

    train = targets.iloc[:-holdout_size].reset_index(drop=True)
    holdout = targets.iloc[-holdout_size:][["timestamp"]].reset_index(drop=True)

    model, validation_summary = select_cpu_model(train)

    predictions = pd.DataFrame({"timestamp": holdout["timestamp"]})
    for target in TARGETS:
        predictions[target] = _forecast(train, holdout, "persistence", target)

    predictions["aggregate_cpu_utilization"] = _forecast(
        train, holdout, model, "aggregate_cpu_utilization"
    )

    predictions["active_vm_count"] = np.maximum(
        0, predictions["active_vm_count"]
    )
    predictions["aggregate_cpu_utilization"] = predictions[
        "aggregate_cpu_utilization"
    ].clip(0, 100)

    predictions = predictions.loc[:, OUTPUT_COLUMNS]

    if len(predictions) != holdout_size:
        raise ValueError("Prediction row count does not match holdout size")
    if predictions["timestamp"].duplicated().any():
        raise ValueError("Prediction timestamps must be unique")
    if not predictions["timestamp"].is_monotonic_increasing:
        raise ValueError("Prediction timestamps must be chronological")
    if not np.isfinite(
        predictions[["active_vm_count", "aggregate_cpu_utilization"]].to_numpy(float)
    ).all():
        raise ValueError("Predictions contain non-finite values")

    return predictions, model, validation_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast aggregate cloud workload")
    parser.add_argument("--data", type=Path, default=Path("challenge_data"))
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    cpu_path = args.data / "vm_cpu_readings.csv.gz"
    if not cpu_path.exists():
        raise FileNotFoundError(f"CPU readings file not found: {cpu_path}")

    cpu = pd.read_csv(cpu_path)
    predictions, model, summary = make_predictions(cpu)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    print("Validation mean fold MAE:")
    print(summary.to_string(index=False))
    print(f"\nSelected CPU model: {model}")
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
