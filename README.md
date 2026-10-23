# Cloud Workload Forecasting Challenge

AI/ML take-home using the fixed data-v1 release, a curated subset of the Microsoft Azure Public Dataset V2 VM trace.

## Setup

Python 3.11 or newer is required. From the repository root:

~~~powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,analysis]"
~~~

The core forecasting pipeline depends only on NumPy and pandas. The optional analysis extra installs Matplotlib; the dev extra installs pytest.

## Dataset

Download data-v1.tar.gz from the [official challenge release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1) and extract it into a local challenge_data folder:

~~~powershell
New-Item -ItemType Directory -Force challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
~~~

The folder should contain vm_cpu_readings.csv.gz, vm_metadata.csv.gz, DATA_CARD.md, and SHA256SUMS. Dataset files are excluded from Git. Verify checksums from PowerShell:

~~~powershell
Set-Location challenge_data
foreach ($line in Get-Content .\SHA256SUMS) {
    if ($line -notmatch '^([A-Fa-f0-9]{64})\s+\*?(.+)$') { throw "Invalid checksum entry: $line" }
    $expected = $Matches[1].ToLowerInvariant()
    $file = $Matches[2]
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "Checksum mismatch: $file" }
    Write-Output ("{0}: OK" -f $file)
}
Set-Location ..
~~~

## Forecasting task and method

The release contains 10,000 VM traces, each with 134 five-minute observations. The pipeline aggregates raw readings into active_vm_count (distinct VMs represented) and aggregate_cpu_utilization (mean VM-level avg_cpu). The first 110 aggregate observations are used for model selection; the final 24 timestamps are the two-hour forecast horizon.

Six candidates are evaluated using the same expanding-window validation loop: persistence, training mean, AR(1), seasonal naive (12-point/one-hour season), moving average (last six points), and Ridge regression using a NumPy closed-form solution on lags 1, 2, 3, 6, 12 plus the previous-six-point mean. The official fold origins are 36, 48, 60, 72, 84; each validation horizon is 24 and the step is 12. Selection is by mean fold MAE. The final 24 target values are never used for fitting, feature selection, tuning, or model selection.

## Generate predictions

After installing the package and extracting the fixed release, run this CLI unchanged:

~~~powershell
python -m cwfc.predict --data challenge_data --output predictions.csv
~~~

The output remains exactly 24 rows with columns timestamp, active_vm_count, aggregate_cpu_utilization, chronological holdout timestamps, finite numeric values, integer non-negative VM counts, and no index column.

## Reproduce analysis, error bands, and report artifacts

Run the prediction command first, then:

~~~powershell
python analysis/analyze_workload.py --data challenge_data --output-dir reports/generated --figure-dir reports/figures --predictions predictions.csv
~~~

The analysis uses shared validation functions from cwfc.predict and regenerates candidate metrics, fold metrics, horizon-bucket metrics, segment summaries, peak/low periods, JSON diagnostics, figures, and reports/generated/prediction_intervals.csv. That separate 24-row file gives each point forecast a horizon-bucket empirical P90 absolute-error band; it does not change predictions.csv.

The short, executed [EDA notebook](notebooks/01_eda.ipynb) covers data-quality checks, aggregate CPU series and ACF, VM-level distribution, segment comparisons, peak/low periods, validation scores, and empirical error bands. The report is [reports/final_report.md](reports/final_report.md).

## Tests

~~~powershell
python -m pytest -q
~~~

Tests cover target construction and validation, all six candidates, Ridge future-value isolation, the seasonal-naive fallback, the CLI smoke dataset, and the committed prediction/interval contracts.

## Important limitations

All 10,000 VMs appear at every timestamp in this curated release, so active_vm_count is constant. This sample does not demonstrate VM-churn forecasting. Validation metrics are not scores on the final 24-point holdout. AR(1) wins on mean fold MAE but can lose to persistence on the selected peak-period subset. P90 bands are descriptive estimates from five overlapping validation folds, not calibrated prediction guarantees. Longer traces and infrastructure-specific backtesting are required before operational automation.

## Assignment documents

Start with [ASSIGNMENT.md](ASSIGNMENT.md), [DATASET.md](DATASET.md), and [SUBMISSION.md](SUBMISSION.md). They define the task, schema, holdout rule, and exact output contract.
