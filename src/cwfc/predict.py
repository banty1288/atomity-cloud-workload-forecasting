from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.baselines import persistence_forecast, seasonal_naive_forecast
from cwfc.metrics import mae, rmse, smape


TARGETS = ("active_vm_count", "aggregate_cpu_utilization")
CPU_TARGET = "aggregate_cpu_utilization"
HOLDOUT_SIZE = 24
SEASON_LENGTH = 12
MOVING_AVERAGE_WINDOW = 6
RIDGE_LAGS = (1, 2, 3, 6, 12)
RIDGE_ROLLING_WINDOW = 6
RIDGE_ALPHA = 1.0
CANDIDATE_MODELS = (
    "persistence", "training_mean", "autoregressive_1",
    "seasonal_naive", "moving_average", "ridge_lags",
)
HORIZON_BUCKETS = ((1, 6, "1-6"), (7, 12, "7-12"), (13, 18, "13-18"), (19, 24, "19-24"))
OUTPUT_COLUMNS = ("timestamp", "active_vm_count", "aggregate_cpu_utilization")


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
    """Fit AR(1) on historical consecutive pairs and forecast recursively."""
    if len(train_values) < 3:
        return np.full(horizon, float(train_values[-1]))
    slope, intercept = np.polyfit(train_values[:-1], train_values[1:], 1)
    previous, forecast = float(train_values[-1]), []
    for _ in range(horizon):
        previous = float(intercept + slope * previous)
        forecast.append(previous)
    return np.asarray(forecast, dtype=float)


def _ridge_lag_features(history: np.ndarray) -> np.ndarray:
    """Use only already available observations: lags 1/2/3/6/12 and prior-six mean.

    At a forecast origin, every value comes from observed history. At later
    recursive steps, earlier predictions replace unavailable actual future values.
    """
    values = np.asarray(history, dtype=float)
    if len(values) < 12:
        raise ValueError("ridge_lags needs at least 12 historical observations")
    return np.asarray([values[-lag] for lag in RIDGE_LAGS]
                      + [float(np.mean(values[-RIDGE_ROLLING_WINDOW:]))])


def _ridge_lags_forecast(train_values: np.ndarray, horizon: int) -> np.ndarray:
    """Fit Ridge with a NumPy closed-form solution and forecast recursively."""
    values = np.asarray(train_values, dtype=float)
    if len(values) < 13:
        return np.full(horizon, float(values[-1]))
    x = np.asarray([_ridge_lag_features(values[:t]) for t in range(12, len(values))])
    y = values[12:]
    x_mean, x_scale, y_mean = x.mean(axis=0), x.std(axis=0), float(y.mean())
    x_scale[x_scale == 0] = 1.0
    xs, yc = (x - x_mean) / x_scale, y - y_mean
    beta = np.linalg.solve(xs.T @ xs + RIDGE_ALPHA * np.eye(xs.shape[1]), xs.T @ yc)
    history, forecast = values.tolist(), []
    for _ in range(horizon):
        features = (_ridge_lag_features(np.asarray(history)) - x_mean) / x_scale
        value = float(y_mean + features @ beta)
        forecast.append(value)
        history.append(value)
    return np.asarray(forecast, dtype=float)


def _forecast(train: pd.DataFrame, template: pd.DataFrame, model: str,
              target: str) -> np.ndarray:
    """Forecast without reading target values from template/future timestamps.

    persistence repeats the last observed point. training_mean uses observed
    history only. AR(1) and Ridge recurse on prior predictions. seasonal_naive
    repeats the last 12 observed points (one hour), falling back to persistence
    when history is shorter. moving_average repeats the mean of the last six
    observed values. Ridge uses lag 1/2/3/6/12 and rolling mean of the previous
    six values; each feature is strictly available before its supervised target.
    """
    values = train[target].to_numpy(dtype=float)
    horizon = len(template)
    if horizon == 0:
        return np.asarray([], dtype=float)
    if not len(values):
        raise ValueError("train must not be empty")
    if model == "persistence":
        return persistence_forecast(train, template, target).to_numpy(float)
    if model == "training_mean":
        return np.full(horizon, float(values.mean()))
    if model == "autoregressive_1":
        return _ar1_forecast(values, horizon)
    if model == "seasonal_naive":
        if len(values) < SEASON_LENGTH:
            return persistence_forecast(train, template, target).to_numpy(float)
        return seasonal_naive_forecast(train, template, target, season_length=SEASON_LENGTH).to_numpy(float)
    if model == "moving_average":
        return np.full(horizon, float(np.mean(values[-MOVING_AVERAGE_WINDOW:])))
    if model == "ridge_lags":
        return _ridge_lags_forecast(values, horizon)
    raise ValueError(f"Unknown forecasting model: {model}")


def evaluate_candidates(train: pd.DataFrame, target: str = CPU_TARGET,
                        horizon: int | None = None, step: int | None = None) -> pd.DataFrame:
    """Evaluate all candidates on expanding folds contained in the training window.

    With 110 points, origins are 36, 48, 60, 72 and 84; each fold forecasts
    the next 24 points, stepping 12. The final 24 challenge targets are never
    used here for fitting, selecting, tuning, or scoring.
    """
    if target not in train.columns or train.empty:
        raise ValueError(f"Training data must contain a non-empty {target} series")
    values = train[target].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Training targets must be finite")
    horizon = min(24, max(1, len(train) // 4)) if horizon is None else int(horizon)
    step = max(1, horizon // 2) if step is None else int(step)
    if horizon <= 0 or step <= 0:
        raise ValueError("horizon and step must be positive")
    initial = max(3, min(horizon + step, len(train) - horizon), len(train) - 4 * horizon)
    origins = list(range(initial, len(train) - horizon + 1, step))
    if not origins:
        raise ValueError("Insufficient observations for expanding-window validation")

    rows = []
    for fold, origin in enumerate(origins, 1):
        fit = train.iloc[:origin].reset_index(drop=True)
        valid = train.iloc[origin:origin + horizon].reset_index(drop=True)
        actual = valid[target].to_numpy(float)
        peak_threshold = float(np.quantile(fit[target].to_numpy(float), .75))
        for model in CANDIDATE_MODELS:
            predicted = _forecast(fit, valid, model, target)
            if len(predicted) != len(valid) or not np.isfinite(predicted).all():
                raise ValueError(f"Model {model} emitted invalid fold predictions")
            for i, (truth, pred) in enumerate(zip(actual, predicted), 1):
                denominator = max(abs(float(truth)) + abs(float(pred)), 1e-8)
                rows.append({
                    "fold": fold, "origin_index": origin, "train_observations": origin,
                    "horizon_step": i,
                    "timestamp": valid.iloc[i - 1]["timestamp"] if "timestamp" in valid else i,
                    "model": model, "actual": float(truth), "prediction": float(pred),
                    "abs_error": abs(float(pred) - float(truth)),
                    "signed_error": float(pred) - float(truth),
                    "sMAPE_percent": 200 * abs(float(pred) - float(truth)) / denominator,
                    "peak_threshold": peak_threshold, "is_peak": bool(float(truth) >= peak_threshold),
                })
    return pd.DataFrame(rows)


def _metric_row(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {"MAE": mae(actual, predicted), "RMSE": rmse(actual, predicted),
            "sMAPE_percent": smape(actual, predicted)}


def summarize_validation(evaluation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize candidate performance as means of fold-level metrics."""
    fold_rows = []
    for (fold, model), group in evaluation.groupby(["fold", "model"], sort=True):
        actual, predicted = group.actual.to_numpy(float), group.prediction.to_numpy(float)
        peak = group.is_peak.to_numpy(bool)
        fold_rows.append({
            "fold": int(fold), "model": model, "origin_index": int(group.origin_index.iloc[0]),
            "train_observations": int(group.train_observations.iloc[0]),
            **_metric_row(actual, predicted),
            "bias_pred_minus_actual": float(np.mean(predicted - actual)),
            "peak_period_MAE": mae(actual[peak], predicted[peak]) if peak.any() else np.nan,
        })
    fold_metrics = pd.DataFrame(fold_rows)
    summary = fold_metrics.groupby("model", as_index=False).agg(
        mean_fold_MAE=("MAE", "mean"), mean_fold_RMSE=("RMSE", "mean"),
        mean_fold_sMAPE_percent=("sMAPE_percent", "mean"),
        mean_fold_bias=("bias_pred_minus_actual", "mean"),
        mean_peak_period_MAE=("peak_period_MAE", "mean"),
    )
    baseline = float(summary.loc[summary.model == "persistence", "mean_fold_MAE"].iloc[0])
    summary["MAE_relative_to_persistence"] = summary.mean_fold_MAE / baseline if baseline else np.nan
    summary["MAE_improvement_vs_persistence_percent"] = (1 - summary.MAE_relative_to_persistence) * 100
    summary = summary.sort_values(["mean_fold_MAE", "model"]).reset_index(drop=True)

    horizon_rows = []
    max_horizon = int(evaluation.horizon_step.max())
    for low, high, _ in HORIZON_BUCKETS:
        if low > max_horizon:
            continue
        high = min(high, max_horizon)
        label = f"{low}-{high}"
        subset = evaluation.loc[evaluation.horizon_step.between(low, high)]
        for (fold, model), group in subset.groupby(["fold", "model"], sort=True):
            horizon_rows.append({"fold": int(fold), "model": model, "horizon_bucket": label,
                                 **_metric_row(group.actual.to_numpy(float), group.prediction.to_numpy(float))})
    horizon_detail = pd.DataFrame(horizon_rows)
    horizon_summary = horizon_detail.groupby(["model", "horizon_bucket"], as_index=False).agg(
        mean_fold_MAE=("MAE", "mean"), mean_fold_RMSE=("RMSE", "mean"),
        mean_fold_sMAPE_percent=("sMAPE_percent", "mean"))
    return summary, horizon_summary, fold_metrics


def build_prediction_intervals(evaluation: pd.DataFrame, selected_model: str,
                               predictions: pd.DataFrame, quantile: float = .90) -> pd.DataFrame:
    """Build empirical P90 absolute-error bands pooled in six-step horizon buckets.

    Bands are point prediction plus/minus the pooled P90 absolute error, clipped
    to [0,100]. They are diagnostics from overlapping validation folds, not
    calibrated prediction intervals or coverage guarantees.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    selected = evaluation.loc[evaluation.model == selected_model].copy()
    if selected.empty:
        raise ValueError(f"No validation errors found for {selected_model}")
    selected["horizon_bucket"] = pd.cut(selected.horizon_step, [0, 6, 12, 18, 24],
        labels=["1-6", "7-12", "13-18", "19-24"], include_lowest=True).astype(str)
    widths = selected.groupby("horizon_bucket", observed=False).abs_error.agg(
        pooled_validation_points="count",
        p90_abs_error_percentage_points=lambda v: float(np.quantile(v, quantile))).reset_index()
    out = predictions.copy().reset_index(drop=True)
    if len(out) != HOLDOUT_SIZE:
        raise ValueError(f"Expected {HOLDOUT_SIZE} prediction rows to build error bands")
    out["horizon_step"] = np.arange(1, len(out) + 1)
    out["horizon_bucket"] = pd.cut(out.horizon_step, [0, 6, 12, 18, 24],
        labels=["1-6", "7-12", "13-18", "19-24"], include_lowest=True).astype(str)
    out = out.merge(widths, on="horizon_bucket", how="left", validate="many_to_one")
    out["selected_model"] = selected_model
    out["interval_quantile"] = quantile
    out["point_forecast"] = out[CPU_TARGET].astype(float)
    out["interval_lower"] = np.maximum(0, out.point_forecast - out.p90_abs_error_percentage_points)
    out["interval_upper"] = np.minimum(100, out.point_forecast + out.p90_abs_error_percentage_points)
    return out[["timestamp", "horizon_step", "horizon_bucket", "selected_model", "point_forecast",
                "pooled_validation_points", "p90_abs_error_percentage_points",
                "interval_lower", "interval_upper", "interval_quantile"]]


def select_cpu_model(train: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Select by lowest mean fold MAE using only the supplied training observations."""
    evaluation = evaluate_candidates(train)
    summary, _, _ = summarize_validation(evaluation)
    return str(summary.iloc[0].model), summary


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

    # Counts should be emitted as integers; baseline helpers use floats for a
    # shared forecast interface.
    predictions["active_vm_count"] = np.maximum(
        0, np.rint(predictions["active_vm_count"])
    ).astype(int)
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
    if not np.isfinite(predictions[["active_vm_count", "aggregate_cpu_utilization"]].to_numpy(float)).all():
        raise ValueError("Predictions contain non-finite values")
    if (predictions["active_vm_count"] < 0).any() or not predictions["aggregate_cpu_utilization"].between(0, 100).all():
        raise ValueError("Predictions are outside their valid numeric ranges")

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
