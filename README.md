# Cloud Workload Forecasting Challenge

AI/ML take-home challenge built from the **Microsoft Azure Public Dataset V2** VM trace.

## Start here

1. Read [`ASSIGNMENT.md`](ASSIGNMENT.md).
2. Read [`DATASET.md`](DATASET.md).
3. Download the **`data-v1` GitHub Release** dataset and extract it as `challenge_data/`.
4. Read [`SUBMISSION.md`](SUBMISSION.md).
5. Run the forecasting pipeline using the command in the Forecasting and submission section.

The challenge dataset is distributed as a versioned GitHub Release rather than committed to the repository. The `data-v1` release contains the fixed curated dataset used for this assignment.

## Environment setup

Python **3.11 or newer** is required.

Create a virtual environment:

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the installation:

```bash
python -m pytest -q
```

The test suite covers aggregate target construction, duplicate-reading detection, prediction output validation, insufficient history, and a command-line integration test.

## Dataset download

The hiring dataset is distributed as the fixed `data-v1` release of this repository.

**Download:** [Azure Dataset v1 Release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)

Download `data-v1.tar.gz` from the release and extract it into the repository as `challenge_data/`:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

The resulting layout is:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

Verify the dataset before starting the assignment. From the repository root, choose the command for your shell.

#### Linux / macOS / Git Bash

```bash
cd challenge_data
sha256sum -c SHA256SUMS
```

#### Windows PowerShell

```powershell
Set-Location challenge_data
foreach ($line in Get-Content .\SHA256SUMS) {
    if ($line -notmatch '^([A-Fa-f0-9]{64})\s+\*?(.+)$') {
        throw "Invalid checksum entry: $line"
    }
    $expectedHash = $Matches[1].ToLowerInvariant()
    $fileName = $Matches[2]
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $fileName).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw "Checksum mismatch: $fileName"
    }
    Write-Output ("{0}: OK" -f $fileName)
}
```

You should see:

```text
vm_cpu_readings.csv.gz: OK
vm_metadata.csv.gz: OK
DATA_CARD.md: OK
```

The release is a real, curated subset of the Microsoft Azure Public Dataset V2. It is not synthetic data.

The dataset contains 10,000 VMs with 134 CPU measurements per VM at five-minute intervals, for a total of 1,340,000 CPU observations.

Use the exact `data-v1` release provided for this assignment. Do not substitute another dataset or a different version of the Azure Public Dataset.

## What you are solving

The underlying data contains anonymized VM workload telemetry sampled every five minutes. Your job is to understand the workload and build a defensible forecasting pipeline for near-term aggregate demand.

The challenge is intentionally open-ended. We evaluate reasoning, time-series methodology, leakage prevention, error analysis, reproducibility, and the quality of your operational conclusions—not just model complexity.

## Expected environment

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Model dependencies are intentionally not prescribed. Add only what you need and document them.

## Source

Microsoft Azure Public Dataset V2:
https://github.com/Azure/AzurePublicDataset

The official V2 documentation describes a 30-day VM trace with five-minute CPU readings, VM metadata, and encrypted subscription/deployment/VM identifiers. The full upstream trace is much larger than this challenge release.

## Submission

Submit your work as a repository or archive according to [`SUBMISSION.md`](SUBMISSION.md).

## Forecasting and submission

### Generate predictions

After installing the project dependencies and extracting the challenge release into `challenge_data/`, run:

```powershell
python -m cwfc.predict --data challenge_data --output predictions.csv
```

The command constructs the aggregate targets from the CPU readings, evaluates candidate CPU forecasting methods using chronological validation on the training window, selects the method with the lowest mean validation MAE, and writes the predictions to `predictions.csv`.

The output contains exactly these columns:

- `timestamp`
- `active_vm_count`
- `aggregate_cpu_utilization`

For the official challenge release, the output contains 24 predictions for the final 24 timestamps. The observed target values for that holdout are not used for model selection or fitting.

### Run tests

```powershell
python -m pytest -q
```

The tests cover target construction, duplicate-reading detection, forecasting output validation, insufficient history, and a command-line integration test.

### Data limitation

In the supplied challenge release, every observed timestamp contains readings for all 10,000 VMs. Consequently, the derived `active_vm_count` target is constant throughout this dataset. Its forecast reflects this property of the supplied trace; it should not be interpreted as evidence that the model can predict VM arrivals or departures.

The CPU forecasting model is selected using chronological validation on the training window. The reported validation MAE is not a score on the final 24 holdout observations.
