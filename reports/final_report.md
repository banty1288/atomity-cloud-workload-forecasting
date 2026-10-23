# Cloud Workload Forecasting and Operational Analysis

**Atomity AI/ML take-home** · Fixed release data-v1 · Two-hour forecast horizon

## Executive summary

This submission builds a reproducible pipeline for two aggregate targets from the curated Microsoft Azure Public Dataset V2 VM trace: distinct VM count and mean VM-level average CPU utilization. The release has 1,340,000 CPU readings from 10,000 VMs over 134 five-minute timestamps. Data checks show regular cadence, complete traces, no duplicate VM/timestamp pairs, no missing CPU fields in the training window, and matching CPU/metadata VM IDs.

The first 110 aggregate observations are used for model selection and validation; the final 24 timestamps are the two-hour forecast horizon. Six candidates are evaluated on the same five expanding-window folds. AR(1) is selected by mean fold MAE (0.1968 percentage points), about 15.4% lower than persistence (0.2326). Ridge lags perform much worse on this short series, demonstrating why model choice remains metric-driven. These are rolling-validation results, **not** accuracy measured on the final holdout.

Aggregate CPU is smooth relative to the spread of VM-level readings. AR(1) has the lowest overall MAE, while persistence has lower error on the selected peak-period subset. The operational recommendation is to use the forecast as a refreshed planning signal with horizon-specific empirical P90 error bands and per-VM/segment monitoring—not as a standalone autoscaling controller.

## 1. Data and target definitions

The fixed data-v1 release is described in [DATASET.md](../DATASET.md), with provenance in [DATA_RELEASE.md](../DATA_RELEASE.md) and [SOURCES.md](../SOURCES.md). It is a curated subset of [Microsoft Azure Public Dataset V2](https://github.com/Azure/AzurePublicDataset), not synthetic data.

| Check | Result |
|---|---:|
| CPU rows in full release | 1,340,000 |
| Distinct VM IDs | 10,000 |
| Distinct timestamps | 134 |
| Timestamp values | 0–39,900 seconds |
| Cadence | 300 seconds |
| Readings per VM | 134 |
| Duplicate VM/timestamp pairs | 0 |
| Missing CPU fields / timestamp gaps | 0 / 0 |
| Release checksums verified | 3 of 3 |

At each timestamp, active_vm_count counts distinct VM IDs with a valid reading. aggregate_cpu_utilization is the mean of VM-level avg_cpu percentages, not total fleet CPU capacity consumed. The pipeline validates required columns, nulls, duplicate VM/timestamp keys, non-finite values, and CPU readings outside [0,100].

The full target grid is constructed first, then split chronologically. The first 110 observations end at timestamp 32,700 seconds. The final 24 output timestamps are 33,000–39,900 seconds. Descriptive target statistics, workload segment comparisons, and validation metrics use the first 110 points. The final 24 target values are not used for fitting, feature construction, tuning, selection, or validation scoring.

## 2. Workload analysis

### 2.1 Aggregate behavior

In the first 110 observations, mean aggregate CPU is **9.414%**, standard deviation **0.194 percentage points**, minimum **9.112%**, and maximum **9.912%**. The 5th–95th percentile range is approximately 9.132%–9.829%, and the mean absolute change between adjacent five-minute points is 0.121 percentage points. Lag-1 autocorrelation is 0.667, lag 6 is 0.477, and lag 12 is 0.367. Recent observations therefore contain short-horizon signal. Correlation at lags 18 and 24 is near zero or slightly negative, so this short trace cannot establish a daily or weekly cycle.

![Aggregate CPU series, first 110 observations](figures/aggregate_training_series.svg)

### 2.2 VM-level CPU readings

Training-window VM-level avg_cpu has median **5.12%**, 95th percentile **37.29%**, and 99th percentile **87.41%**. Approximately 1.56% of VM-level readings are at least 70%, and 0.88% are at least 90%. These are fractions of five-minute VM readings, not counts of unique machines or the amount of time each VM stays saturated.

![VM-level CPU distribution](figures/vm_cpu_distribution.svg)

The long upper tail matters operationally: a low aggregate mean does not mean every VM is lightly loaded. Aggregate-only monitoring can conceal individual hotspots.

### 2.3 Workload segment comparison

All segment summaries use only the first 110 timestamps. Interactive VMs have mean CPU reading **11.51%**, Delay-insensitive VMs **8.26%**, and Unknown-category VMs **10.35%**. The Unknown category's reading-level P95 is highest at about 52.50%. These are descriptive associations based on source metadata labels, not causal effects.

| VM category | VMs | Mean CPU reading | Reading P95 |
|---|---:|---:|---:|
| Delay-insensitive | 6,180 | 8.26% | 25.94% |
| Interactive | 3,091 | 11.51% | 49.15% |
| Unknown | 729 | 10.35% | 52.50% |

![Mean CPU by workload category](figures/cpu_by_vm_category.svg)

The 24-core group has mean CPU around 11.43% and P95 52.51%, but includes just 62 VMs. The dominant 2-core group contains 7,534 VMs, with mean 9.76% and P95 39.21%. For memory buckets, the 4-GB group has mean 11.46% and P95 58.77%, compared with 7.81% and 29.51% for the 32-GB group. Small groups should not be ranked confidently, and these associations do not establish that resource size causes utilization differences. Full results are in [segment_summary.csv](generated/segment_summary.csv); ten highest and lowest training periods are in [peak_low_periods.csv](generated/peak_low_periods.csv).

### 2.4 VM count and churn limitation

Every one of the 10,000 VMs appears at all 134 timestamps, with 134 observations per VM. active_vm_count is therefore **10,000 at every timestamp**. Persistence provides the required forecast for this constant target, but this sample cannot demonstrate VM-arrival or deletion prediction. The short trace also limits claims about seasonality or generalization to other periods and fleets.

## 3. Forecast candidates and validation

### 3.1 Candidate models and leakage-safe features

The six candidates are persistence, training mean, AR(1), seasonal naive, moving average, and Ridge lags.

- **Persistence** repeats the last observed value.
- **Training mean** repeats the mean of the available training history.
- **AR(1)** fits the relationship between consecutive historical values and recursively predicts future steps.
- **Seasonal naive** repeats the last 12 observed values (one hour), falling back to persistence when the history is shorter than 12 points.
- **Moving average** repeats the mean of the last six observed values.
- **Ridge lags** uses NumPy's closed-form Ridge solution on lags 1, 2, 3, 6, and 12 plus the rolling mean of the prior six points. Each supervised feature uses observations strictly earlier than its target. For recursive multi-step forecasting, earlier predictions are fed into subsequent features; no actual future value is read.

All candidates use the same validation loop. The core pipeline has only NumPy and pandas dependencies; Matplotlib is isolated in the analysis extra.

### 3.2 Expanding-window validation

For the 110-point training window, origins are **36, 48, 60, 72, 84**, the forecast horizon is 24 points, and the step is 12. The first fold fits 36 observations and evaluates the next 24; later folds expand the training history. The windows overlap by 12 points, so the scores are useful comparative estimates but not independent replicates. No final-holdout target is used in any fold.

### 3.3 Candidate comparison

The table reports means of fold-level metrics. MAE and RMSE are in percentage points. sMAPE floors its denominator to prevent division by zero. Bias is prediction minus actual, so positive values mean over-prediction. MAE gain is relative to persistence: positive values improve on the baseline and negative values are worse.

| Model | MAE ↓ | RMSE ↓ | sMAPE ↓ | Bias (pp) | Peak MAE ↓ | MAE gain vs persistence |
|---|---:|---:|---:|---:|---:|---:|
| **AR(1), selected** | **0.1968** | **0.2362** | **2.075%** | -0.0191 | 0.3069 | **+15.43%** |
| Training mean | 0.2014 | 0.2409 | 2.124% | -0.0277 | 0.3235 | +13.44% |
| Persistence | 0.2326 | 0.2681 | 2.444% | +0.0768 | **0.2609** | 0.00% |
| Moving average (6) | 0.2454 | 0.2901 | 2.577% | +0.0535 | 0.3090 | -5.47% |
| Seasonal naive (12) | 0.2467 | 0.2894 | 2.590% | +0.0264 | 0.3247 | -6.06% |
| Ridge lags | 0.5723 | 0.7247 | 5.522% | +0.3745 | 0.5196 | -146.01% |

AR(1) improves mean fold MAE by approximately **15.4%** and RMSE by **11.9%** compared with persistence. Its MAE gain over training mean is modest, around 2.3%. Ridge lags perform much worse on this short, smooth series. That is a useful result: added model complexity does not guarantee better forecasts.

### 3.4 Horizon and peak errors

AR(1) mean fold MAE by six-step horizon bucket is 0.1379 percentage points for steps 1–6, 0.2025 for 7–12, 0.1969 for 13–18, and 0.2497 for 19–24. The error tends to be larger later in the two-hour horizon; refreshing the forecast when new observations arrive is preferable to relying on one fixed projection.

A peak is defined within each fold as an actual validation value at or above the 75th percentile of that fold's training-only aggregate CPU history. The average threshold is **9.518%**. AR(1)'s mean fold peak MAE is **0.3069 percentage points**, worse than persistence at **0.2609**. So AR(1) wins the full-horizon MAE criterion but is not the best model on this selected peak subset. See [validation_metrics.csv](generated/validation_metrics.csv), [validation_folds.csv](generated/validation_folds.csv), and [validation_by_horizon.csv](generated/validation_by_horizon.csv).

## 4. Empirical P90 error bands and capacity recommendation

For the selected model, absolute validation errors are pooled by four six-step horizon buckets. The P90 absolute error for each bucket is added to and subtracted from the corresponding point forecast, then clipped to [0,100]. Each bucket pools 5 folds × 6 steps = 30 error rows. Because the folds overlap, the bands are **empirical error bands—not calibrated 90% prediction intervals or coverage guarantees**.

| Forecast steps | Lead time | Pooled errors | P90 absolute error (pp) |
|---|---|---:|---:|
| 1–6 | 5–30 minutes | 30 | 0.3272 |
| 7–12 | 35–60 minutes | 30 | 0.4558 |
| 13–18 | 65–90 minutes | 30 | 0.3884 |
| 19–24 | 95–120 minutes | 30 | 0.4570 |

The per-timestamp bounds are stored separately in [prediction_intervals.csv](generated/prediction_intervals.csv). They do not add columns to the required predictions.csv.

Use the forecast as a frequently refreshed capacity-planning signal, not a standalone autoscaling controller. For an initial pilot, display the point forecast and the matching horizon-bucket interval together, and use the upper endpoint as a conservative aggregate planning signal. Apply any additional safety margin only after relating it to real pool capacity, service objectives, and the cost of under-provisioning. Track per-VM and per-segment P95/maximum utilization in parallel, because the aggregate mean can hide hot machines. Escalate when an upper band approaches a capacity or SLO threshold, or when per-VM tails rise even while the aggregate mean looks stable.

These margins are descriptive estimates from five overlapping folds; they are not production thresholds. More headroom can reduce saturation risk but waste allocated capacity. Less headroom can improve utilization but increase delayed-scaling and service-degradation risk. Before automation, backtest the policy on a longer trace containing real VM churn, sustained peaks, and recorded scaling outcomes.

## 5. Reproducibility and conclusion

After downloading and extracting the specified release into a local challenge_data folder, run:

~~~powershell
python -m pip install -e ".[dev,analysis]"
python -m pytest -q
python -m cwfc.predict --data challenge_data --output predictions.csv
python analysis/analyze_workload.py --data challenge_data --output-dir reports/generated --figure-dir reports/figures --predictions predictions.csv
~~~

The short, executed [EDA notebook](../notebooks/01_eda.ipynb) covers training-window quality checks, aggregate CPU and ACF, VM-level distribution, segment comparisons, peak/low periods, candidate scores, and P90 error bands. The analysis imports the shared validation/evaluation functions from cwfc.predict.

**Conclusion.** AR(1) remains selected because it has the lowest mean fold MAE among the six candidates on the required folds. Ridge, moving average, and seasonal naive lose to simpler alternatives on this release. Constant VM count, the VM-level upper tail, a short dataset, overlapping validation folds, and worse AR(1) peak error limit operational claims. Reported scores are validation—not final holdout accuracy.

## References

- Assignment requirements and rubric: [ASSIGNMENT.md](../ASSIGNMENT.md).
- Target definitions and holdout protocol: [DATASET.md](../DATASET.md).
- Output and repository rules: [SUBMISSION.md](../SUBMISSION.md).
- Release provenance: [DATA_RELEASE.md](../DATA_RELEASE.md) and [SOURCES.md](../SOURCES.md).
- Upstream source: [Microsoft Azure Public Dataset](https://github.com/Azure/AzurePublicDataset).
