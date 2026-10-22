import numpy as np
import pandas as pd
import pytest

from cwfc.predict import build_targets, main, make_predictions


def make_cpu_data(n_timestamps=40, n_vms=3):
    rows = []

    for vm in range(n_vms):
        for t in range(n_timestamps):
            rows.append({
                "timestamp": t * 300,
                "vm_id": f"vm-{vm}",
                "avg_cpu": 10.0 + vm + 0.1 * np.sin(t / 3),
            })

    return pd.DataFrame(rows)


def test_build_targets_constructs_expected_aggregates():
    cpu = pd.DataFrame({
        "timestamp": [0, 0, 300],
        "vm_id": ["vm-a", "vm-b", "vm-a"],
        "avg_cpu": [10.0, 20.0, 30.0],
    })

    targets = build_targets(cpu)

    assert targets["active_vm_count"].tolist() == [2, 1]
    assert targets["aggregate_cpu_utilization"].tolist() == [15.0, 30.0]


def test_build_targets_rejects_duplicate_vm_timestamp():
    cpu = pd.DataFrame({
        "timestamp": [0, 0],
        "vm_id": ["vm-a", "vm-a"],
        "avg_cpu": [10.0, 11.0],
    })

    with pytest.raises(ValueError, match="Duplicate"):
        build_targets(cpu)


def test_make_predictions_returns_valid_small_dataset_output():
    cpu = make_cpu_data()
    predictions, model, summary = make_predictions(cpu)

    assert list(predictions.columns) == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
    assert len(predictions) == 24
    assert predictions["timestamp"].tolist() == list(range(16 * 300, 40 * 300, 300))
    assert predictions["timestamp"].is_unique
    assert predictions["timestamp"].is_monotonic_increasing

    numeric = predictions[
        ["active_vm_count", "aggregate_cpu_utilization"]
    ].to_numpy(dtype=float)

    assert np.isfinite(numeric).all()
    assert (predictions["active_vm_count"] >= 0).all()
    assert np.equal(
        predictions["active_vm_count"].to_numpy(float),
        np.floor(predictions["active_vm_count"].to_numpy(float)),
    ).all()
    assert predictions["aggregate_cpu_utilization"].between(0, 100).all()
    assert model in {"persistence", "training_mean", "autoregressive_1"}
    assert not summary.empty


def test_make_predictions_rejects_too_little_history():
    cpu = make_cpu_data(n_timestamps=3)

    with pytest.raises(ValueError, match="At least 27 timestamps are required"):
        make_predictions(cpu)

def test_cli_writes_predictions_csv_without_index(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cpu_path = data_dir / "vm_cpu_readings.csv.gz"
    make_cpu_data().to_csv(cpu_path, index=False, compression="gzip")

    output_path = tmp_path / "generated" / "predictions.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "cwfc.predict",
            "--data",
            str(data_dir),
            "--output",
            str(output_path),
        ],
    )

    main()

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert list(saved.columns) == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
    assert len(saved) == 24
    assert saved["timestamp"].is_monotonic_increasing
    assert saved["timestamp"].is_unique
    assert np.issubdtype(saved["active_vm_count"].dtype, np.integer)
    assert "Unnamed: 0" not in saved.columns
    values = saved[["active_vm_count", "aggregate_cpu_utilization"]].to_numpy(float)
    assert np.isfinite(values).all()
    assert saved["aggregate_cpu_utilization"].between(0, 100).all()
