from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_committed_predictions_csv_matches_submission_contract():
    predictions = pd.read_csv(ROOT / "predictions.csv")

    assert predictions.columns.tolist() == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
    assert len(predictions) == 24
    assert predictions["timestamp"].is_unique
    assert predictions["timestamp"].is_monotonic_increasing
    assert predictions["timestamp"].tolist() == list(range(33000, 39901, 300))

    counts = predictions["active_vm_count"].to_numpy(float)
    cpu = predictions["aggregate_cpu_utilization"].to_numpy(float)
    assert np.isfinite(counts).all()
    assert np.isfinite(cpu).all()
    assert (counts >= 0).all()
    assert np.equal(counts, np.floor(counts)).all()
    assert ((cpu >= 0) & (cpu <= 100)).all()


def test_prediction_interval_artifact_matches_forecast_horizon():
    path = ROOT / "reports/generated/prediction_intervals.csv"
    assert path.exists(), "Run analysis/analyze_workload.py to generate intervals"
    intervals = pd.read_csv(path)
    assert len(intervals) == 24
    assert intervals["timestamp"].tolist() == list(range(33000, 39901, 300))
    assert (intervals["interval_lower"] <= intervals["point_forecast"]).all()
    assert (intervals["point_forecast"] <= intervals["interval_upper"]).all()
