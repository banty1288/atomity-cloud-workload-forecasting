"""Reproduce workload analysis and validation artifacts for the fixed data-v1 release.

All actual-value statistics, segment summaries, fold scoring, and peak/low
analysis use only the first 110 aggregate observations. The last 24 timestamps
are retained only to verify output row/timestamp alignment and build forecast
interval rows; their actual target values are not used here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cwfc.predict import (  # noqa: E402
    HOLDOUT_SIZE, build_prediction_intervals, build_targets,
    evaluate_candidates, select_cpu_model, summarize_validation,
)


def verify_checksums(data_dir: Path) -> dict[str, bool]:
    """Verify every entry in the release's SHA256SUMS manifest."""
    manifest = data_dir / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing checksum manifest: {manifest}")
    results = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.match(r"^([A-Fa-f0-9]{64})\s+\*?(.+)$", line.strip())
        if not match:
            raise ValueError(f"Invalid checksum entry: {line}")
        expected, name = match.group(1).lower(), match.group(2)
        digest = hashlib.sha256()
        with (data_dir / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        results[name] = digest.hexdigest() == expected
    if not results or not all(results.values()):
        raise ValueError(f"Checksum verification failed: {results}")
    return results


def summarize_segments(cpu_train: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Describe training-window VM CPU readings by the supplied metadata groups."""
    frame = cpu_train.merge(
        metadata[["vm_id", "vm_category", "virtual_core_bucket", "memory_gb_bucket"]],
        on="vm_id", how="left", validate="many_to_one",
    )
    rows = []
    for dimension in ("vm_category", "virtual_core_bucket", "memory_gb_bucket"):
        for label, group in frame.groupby(dimension, dropna=False, observed=False):
            rows.append({
                "dimension": dimension, "segment": str(label),
                "vm_count": int(group.vm_id.nunique()),
                "observations_first_110": int(len(group)),
                "mean_reading_cpu_percent": float(group.avg_cpu.mean()),
                "median_reading_cpu_percent": float(group.avg_cpu.median()),
                "p95_reading_cpu_percent": float(group.avg_cpu.quantile(.95)),
                "p95_vm_mean_cpu_percent": float(group.groupby("vm_id").avg_cpu.mean().quantile(.95)),
            })
    return pd.DataFrame(rows)


def peak_low_table(train: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    cols = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    high = train.nlargest(count, "aggregate_cpu_utilization")[cols].copy()
    high["period_group"] = "highest"
    low = train.nsmallest(count, "aggregate_cpu_utilization")[cols].copy()
    low["period_group"] = "lowest"
    return pd.concat([high, low], ignore_index=True).sort_values(
        ["period_group", "aggregate_cpu_utilization"]
    ).reset_index(drop=True)


def create_figures(train: pd.DataFrame, cpu_train: pd.DataFrame,
                   segments: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(train.timestamp / 3600, train.aggregate_cpu_utilization, linewidth=1.7)
    ax.axhline(train.aggregate_cpu_utilization.mean(), linestyle="--",
               label=f"Training mean = {train.aggregate_cpu_utilization.mean():.3f}%")
    ax.set(title="Aggregate CPU utilization: first 110 observations",
           xlabel="Elapsed time from trace start (hours)", ylabel="Mean VM avg_cpu (%)")
    ax.grid(True, alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(figure_dir / "aggregate_training_series.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(cpu_train.avg_cpu.to_numpy(float), bins=np.linspace(0, 100, 51), density=True, alpha=.85)
    ax.set(title="VM-level avg_cpu distribution (training window)",
           xlabel="CPU utilization (%)", ylabel="Density")
    ax.grid(True, axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(figure_dir / "vm_cpu_distribution.svg"); plt.close(fig)

    categories = segments.loc[segments.dimension == "vm_category"].sort_values("mean_reading_cpu_percent")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.barh(categories.segment, categories.mean_reading_cpu_percent)
    ax.set(title="Mean VM-level CPU by category (training window)",
           xlabel="Mean avg_cpu (%)", ylabel="VM category")
    ax.grid(True, axis="x", alpha=.25)
    for bar, (_, row) in zip(bars, categories.iterrows()):
        ax.text(bar.get_width() + .1, bar.get_y() + bar.get_height()/2,
                f"{row.mean_reading_cpu_percent:.1f}% (n={int(row.vm_count):,})",
                va="center", fontsize=8)
    ax.set_xlim(0, float(categories.mean_reading_cpu_percent.max()) * 1.35)
    fig.tight_layout(); fig.savefig(figure_dir / "cpu_by_vm_category.svg"); plt.close(fig)


def _json_records(frame: pd.DataFrame) -> list[dict]:
    result = []
    for record in frame.to_dict(orient="records"):
        clean = {}
        for key, value in record.items():
            if pd.isna(value):
                clean[key] = None
            elif isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[key] = float(value)
            elif isinstance(value, (np.bool_,)):
                clean[key] = bool(value)
            else:
                clean[key] = value
        result.append(clean)
    return result


def run_analysis(data_dir: Path, output_dir: Path, figure_dir: Path,
                 predictions_path: Path) -> dict:
    checksums = verify_checksums(data_dir)
    cpu = pd.read_csv(data_dir / "vm_cpu_readings.csv.gz")
    metadata = pd.read_csv(data_dir / "vm_metadata.csv.gz")
    targets = build_targets(cpu)
    if len(targets) <= HOLDOUT_SIZE + 2:
        raise ValueError("At least 27 timestamps are required.")

    # The task's target series is constructed first, then split at 110/24.
    train = targets.iloc[:-HOLDOUT_SIZE].reset_index(drop=True)
    holdout_timestamps = targets.iloc[-HOLDOUT_SIZE:].timestamp.astype(int).tolist()
    cpu_train = cpu.loc[cpu.timestamp <= train.timestamp.max()].copy()

    evaluation = evaluate_candidates(train)
    summary, horizon_summary, fold_metrics = summarize_validation(evaluation)
    selected_model, _ = select_cpu_model(train)
    if selected_model != str(summary.iloc[0].model):
        raise AssertionError("Analysis and prediction model selection disagree")

    predictions = pd.read_csv(predictions_path)
    if len(predictions) != HOLDOUT_SIZE or predictions.timestamp.astype(int).tolist() != holdout_timestamps:
        raise ValueError("predictions.csv does not match the required holdout timestamps")
    intervals = build_prediction_intervals(evaluation, selected_model, predictions)
    segments = summarize_segments(cpu_train, metadata)
    peak_low = peak_low_table(train)

    timestamps = np.sort(cpu_train.timestamp.unique())
    per_vm_count = cpu_train.groupby("vm_id").timestamp.count()
    per_ts_count = cpu_train.groupby("timestamp").vm_id.nunique()
    y = train.aggregate_cpu_utilization.to_numpy(float)
    raw_cpu = cpu_train.avg_cpu.to_numpy(float)
    centered = y - y.mean()
    denom = float(np.sum(centered ** 2))
    acf = {}
    for lag in (1, 2, 3, 6, 12, 18, 24):
        acf[str(lag)] = float(np.sum(centered[:-lag] * centered[lag:]) / denom) if denom else None

    data_quality = {
        "cpu_rows_first_110_timestamps": int(len(cpu_train)),
        "distinct_vm_ids_first_110": int(cpu_train.vm_id.nunique()),
        "training_timestamps": int(len(timestamps)),
        "timestamp_start_seconds": int(timestamps.min()),
        "timestamp_end_seconds": int(timestamps.max()),
        "cadence_values_seconds": [int(v) for v in sorted(set(np.diff(timestamps).tolist()))],
        "timestamp_gaps": int(np.sum(np.diff(timestamps) != 300)),
        "duplicate_vm_timestamp_pairs": int(cpu_train.duplicated(["vm_id", "timestamp"]).sum()),
        "missing_required_cpu_fields": int(cpu_train[["timestamp", "vm_id", "avg_cpu"]].isna().sum().sum()),
        "vm_observation_count_min": int(per_vm_count.min()),
        "vm_observation_count_max": int(per_vm_count.max()),
        "active_vm_counts_per_training_timestamp": [int(v) for v in sorted(per_ts_count.unique())],
        "training_vm_ids_present_in_metadata": bool(set(cpu_train.vm_id).issubset(set(metadata.vm_id))),
        "release_checksums_verified": checksums,
    }
    training_stats = {
        "training_observations": int(len(train)),
        "aggregate_cpu_mean_percent": float(y.mean()),
        "aggregate_cpu_std_percentage_points": float(y.std(ddof=1)),
        "aggregate_cpu_min_percent": float(y.min()),
        "aggregate_cpu_max_percent": float(y.max()),
        "aggregate_cpu_quantiles_percent": {str(q): float(np.quantile(y, q)) for q in (.05, .25, .5, .75, .95)},
        "mean_absolute_consecutive_change_percentage_points": float(np.mean(np.abs(np.diff(y)))),
        "autocorrelation": acf,
        "vm_reading_quantiles_percent": {str(q): float(np.quantile(raw_cpu, q)) for q in (.05, .25, .5, .75, .9, .95, .99)},
        "fraction_vm_readings_at_least_70_percent": float(np.mean(raw_cpu >= 70)),
        "fraction_vm_readings_at_least_90_percent": float(np.mean(raw_cpu >= 90)),
        "peak_timestamp_training_only": int(train.loc[train.aggregate_cpu_utilization.idxmax(), "timestamp"]),
        "low_timestamp_training_only": int(train.loc[train.aggregate_cpu_utilization.idxmin(), "timestamp"]),
        "active_vm_count_unique_training_values": [int(v) for v in sorted(train.active_vm_count.unique())],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "validation_metrics.csv", index=False)
    horizon_summary.to_csv(output_dir / "validation_by_horizon.csv", index=False)
    fold_metrics.to_csv(output_dir / "validation_folds.csv", index=False)
    segments.to_csv(output_dir / "segment_summary.csv", index=False)
    peak_low.to_csv(output_dir / "peak_low_periods.csv", index=False)
    intervals.to_csv(output_dir / "prediction_intervals.csv", index=False)

    selected_rows = evaluation.loc[evaluation.model == selected_model]
    threshold_values = selected_rows.groupby("fold").peak_threshold.first()
    interval_summary = intervals.groupby("horizon_bucket", as_index=False).agg(
        pooled_validation_points=("pooled_validation_points", "first"),
        p90_abs_error_percentage_points=("p90_abs_error_percentage_points", "first"),
    )
    results = {
        "data_quality_first_110_only": data_quality,
        "training_descriptive_statistics": training_stats,
        "selected_cpu_model": selected_model,
        "validation_design": {
            "training_points": int(len(train)),
            "origins_zero_based": [int(v) for v in sorted(evaluation.origin_index.unique())],
            "horizon": int(evaluation.horizon_step.max()),
            "folds": int(evaluation.fold.nunique()),
            "step": 12 if int(evaluation.horizon_step.max()) == 24 else max(1, int(evaluation.horizon_step.max()) // 2),
            "mean_training_only_q75_peak_threshold_percent": float(threshold_values.mean()),
            "fold_windows_overlap": True,
            "final_24_actual_targets_used_for_selection_or_scoring": False,
        },
        "validation_metrics": _json_records(summary),
        "validation_by_horizon": _json_records(horizon_summary),
        "prediction_error_bands": _json_records(interval_summary),
        "forecast_timestamps": holdout_timestamps,
    }
    (output_dir / "analysis_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    create_figures(train, cpu_train, segments, figure_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("challenge_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--predictions", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()
    result = run_analysis(args.data, args.output_dir, args.figure_dir, args.predictions)
    print(f"Verified {len(result['data_quality_first_110_only']['release_checksums_verified'])} release checksums.")
    print(f"Selected model: {result['selected_cpu_model']}")
    print(pd.DataFrame(result["validation_metrics"]).to_string(index=False))


if __name__ == "__main__":
    main()
