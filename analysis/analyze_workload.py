"""Reproducible workload diagnostics for the fixed cloud workload challenge.

Run: python analysis/analyze_workload.py --data challenge_data
Uses first 110 aggregate points for model validation/statistics; reserves the
final 24 target points for the required forecast horizon.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cwfc.metrics import mae, rmse, smape
from cwfc.predict import _forecast, build_targets, select_cpu_model

MODELS = ("persistence", "training_mean", "autoregressive_1")
HOLDOUT = 24


def verify_checksums(data_dir: Path) -> dict[str, bool]:
    manifest = data_dir / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing checksum manifest: {manifest}")
    result = {}
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
        result[name] = digest.hexdigest() == expected
    if not result or not all(result.values()):
        raise ValueError(f"Checksum verification failed: {result}")
    return result


def metrics(actual, predicted):
    return {"MAE": mae(actual, predicted), "RMSE": rmse(actual, predicted),
            "sMAPE_percent": smape(actual, predicted),
            "bias_pred_minus_actual": float(np.mean(predicted - actual))}


def validate(train):
    target = "aggregate_cpu_utilization"
    horizon = min(24, max(1, len(train) // 4))
    step = max(1, horizon // 2)
    initial = max(3, min(horizon + step, len(train) - horizon), len(train) - 4 * horizon)
    origins = list(range(initial, len(train) - horizon + 1, step))
    if not origins:
        raise ValueError("Insufficient observations for temporal validation.")
    rows, horizon_rows, errors, peak_errors, thresholds = [], [], [], [], []
    for fold, origin in enumerate(origins, 1):
        fit = train.iloc[:origin].reset_index(drop=True)
        valid = train.iloc[origin:origin+horizon].reset_index(drop=True)
        actual = valid[target].to_numpy(float)
        threshold = float(np.quantile(fit[target].to_numpy(float), .75))
        peak = actual >= threshold
        thresholds.append(threshold)
        for model in MODELS:
            pred = np.asarray(_forecast(fit, valid, model, target), dtype=float)
            rows.append({"fold":fold,"origin_index":origin,"train_observations":origin,
                         "validation_horizon":len(valid),"training_q75_peak_threshold":threshold,
                         "peak_observations":int(peak.sum()),"model":model,**metrics(actual,pred),
                         "peak_period_MAE":mae(actual[peak],pred[peak]) if peak.any() else np.nan})
            for label, start in (("1-6",0),("7-12",6),("13-18",12),("19-24",18)):
                end = min(start+6,len(actual))
                if start < len(actual):
                    horizon_rows.append({"fold":fold,"model":model,"horizon_steps":label,
                                         "n_points":end-start,**metrics(actual[start:end],pred[start:end])})
            if model == "autoregressive_1":
                errors.extend(np.abs(pred-actual).tolist())
                peak_errors.extend(np.abs(pred[peak]-actual[peak]).tolist())
    details = pd.DataFrame(rows)
    summary = details.groupby("model",as_index=False).agg(
        mean_fold_MAE=("MAE","mean"), mean_fold_RMSE=("RMSE","mean"),
        mean_fold_sMAPE_percent=("sMAPE_percent","mean"),
        mean_fold_bias=("bias_pred_minus_actual","mean"),
        mean_peak_period_MAE=("peak_period_MAE","mean"))
    base = float(summary.loc[summary.model=="persistence","mean_fold_MAE"].iloc[0])
    summary["MAE_relative_to_persistence"] = summary.mean_fold_MAE/base
    summary["MAE_improvement_vs_persistence_percent"] = (1-summary.MAE_relative_to_persistence)*100
    summary = summary.sort_values(["mean_fold_MAE","model"]).reset_index(drop=True)
    horizon_df = pd.DataFrame(horizon_rows).groupby(["model","horizon_steps"],as_index=False).agg(
        mean_fold_MAE=("MAE","mean"),mean_fold_RMSE=("RMSE","mean"),
        mean_fold_sMAPE_percent=("sMAPE_percent","mean"))
    design = {"fold_count":len(origins),"origins_zero_based":origins,
              "validation_horizon":horizon,"initial_training_size":initial,"step":step,
              "mean_fold_training_q75_peak_threshold":float(np.mean(thresholds)),
              "ar1_absolute_error_p90_percentage_points":float(np.quantile(errors,.90)),
              "ar1_absolute_error_p95_percentage_points":float(np.quantile(errors,.95)),
              "ar1_peak_absolute_error_p90_percentage_points":float(np.quantile(peak_errors,.90)),
              "absolute_error_count_with_overlap":len(errors),"validation_windows_overlap":step<horizon}
    return summary,horizon_df,design


def summarize_segments(cpu_train, metadata):
    frame=cpu_train.merge(metadata[["vm_id","vm_category","virtual_core_bucket","memory_gb_bucket"]],
                           on="vm_id",how="left",validate="many_to_one")
    rows=[]
    for dimension in ("vm_category","virtual_core_bucket","memory_gb_bucket"):
        for label,group in frame.groupby(dimension,dropna=False,observed=False):
            rows.append({"dimension":dimension,"segment":str(label),"vm_count":int(group.vm_id.nunique()),
                "observations_first_110":int(len(group)),"mean_reading_cpu_percent":float(group.avg_cpu.mean()),
                "median_reading_cpu_percent":float(group.avg_cpu.median()),
                "p95_reading_cpu_percent":float(group.avg_cpu.quantile(.95)),
                "p95_vm_mean_cpu_percent":float(group.groupby("vm_id").avg_cpu.mean().quantile(.95))})
    return pd.DataFrame(rows)


def plot_artifacts(train, cpu_train, segment_table, folder):
    folder.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,4.6))
    ax.plot(train.timestamp/3600,train.aggregate_cpu_utilization,linewidth=1.8,label="Training aggregate CPU")
    ax.axhline(train.aggregate_cpu_utilization.mean(),linestyle="--",label="Training mean")
    ax.set(title="Aggregate CPU utilization: first 110 observations",
           xlabel="Elapsed time from trace start (hours)",ylabel="Mean VM avg_cpu (%)")
    ax.grid(True,alpha=.25);ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(folder/"aggregate_training_series.svg");plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.5,4.5))
    ax.hist(cpu_train.avg_cpu,bins=np.linspace(0,100,51),density=True,alpha=.82)
    ax.set(title="VM-level average CPU reading distribution (training window)",
           xlabel="VM avg_cpu per five-minute interval (%)",ylabel="Density")
    ax.grid(True,axis="y",alpha=.25);fig.tight_layout()
    fig.savefig(folder/"vm_cpu_distribution.svg");plt.close(fig)

    cat=segment_table[segment_table.dimension=="vm_category"].sort_values("mean_reading_cpu_percent")
    fig,ax=plt.subplots(figsize=(8.5,4.6))
    bars=ax.barh(cat.segment,cat.mean_reading_cpu_percent)
    ax.set(title="Mean VM-level CPU by category (training window)",xlabel="Mean avg_cpu (%)",ylabel="VM category")
    ax.grid(True,axis="x",alpha=.25)
    for bar,(_,row) in zip(bars,cat.iterrows()):
        ax.text(bar.get_width()+.1,bar.get_y()+bar.get_height()/2,
                f"{row.mean_reading_cpu_percent:.1f}% (n={int(row.vm_count):,})",va="center",fontsize=8)
    ax.set_xlim(0,float(cat.mean_reading_cpu_percent.max())*1.35)
    fig.tight_layout();fig.savefig(folder/"cpu_by_vm_category.svg");plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("challenge_data"))
    parser.add_argument("--output-dir",type=Path,default=Path("reports/generated"))
    parser.add_argument("--figure-dir",type=Path,default=Path("reports/figures"))
    args=parser.parse_args()
    data_dir,out_dir,fig_dir=args.data.resolve(),args.output_dir.resolve(),args.figure_dir.resolve()
    out_dir.mkdir(parents=True,exist_ok=True)
    checks=verify_checksums(data_dir)
    cpu=pd.read_csv(data_dir/"vm_cpu_readings.csv.gz")
    meta=pd.read_csv(data_dir/"vm_metadata.csv.gz")
    targets=build_targets(cpu)
    if len(targets)<=HOLDOUT+2:
        raise ValueError("Need at least 27 timestamp observations to reserve the 24-point horizon.")
    train=targets.iloc[:-HOLDOUT].reset_index(drop=True)
    holdout=targets.iloc[-HOLDOUT:].timestamp.astype(int).tolist()
    cpu_train=cpu.loc[cpu.timestamp<=train.timestamp.max()].copy()
    seg=summarize_segments(cpu_train,meta)
    score,horizon,cv=validate(train)
    selected,_=select_cpu_model(train)
    vals=train.aggregate_cpu_utilization.to_numpy(float)
    raw=cpu_train.avg_cpu.to_numpy(float)
    ts=np.sort(cpu.timestamp.unique())
    per_vm=cpu.groupby("vm_id").timestamp.count()
    per_ts=cpu.groupby("timestamp").vm_id.nunique()
    ac={}
    for lag in (1,2,3,6,12,18,24):
        a,b=vals[:-lag],vals[lag:]
        ac[str(lag)]=float(np.corrcoef(a,b)[0,1]) if np.std(a) and np.std(b) else None
    dq={"cpu_rows":int(len(cpu)),"unique_vm_ids":int(cpu.vm_id.nunique()),
        "unique_timestamps":int(len(ts)),"timestamp_min_seconds":int(ts.min()),
        "timestamp_max_seconds":int(ts.max()),
        "timestamp_step_unique_values_seconds":[int(v) for v in sorted(set(np.diff(ts).tolist()))],
        "timestamp_gap_count":int(np.sum(np.diff(ts)!=300)),
        "duplicate_vm_timestamp_pairs":int(cpu.duplicated(["vm_id","timestamp"]).sum()),
        "duplicate_full_rows":int(cpu.duplicated().sum()),
        "cpu_missing_values":{str(k):int(v) for k,v in cpu.isna().sum().items()},
        "vm_observations_count_min":int(per_vm.min()),"vm_observations_count_max":int(per_vm.max()),
        "unique_vm_counts_per_timestamp":[int(v) for v in sorted(per_ts.unique())],
        "vm_ids_match_metadata":bool(set(cpu.vm_id)==set(meta.vm_id)),
        "metadata_rows":int(len(meta)),"metadata_unique_vm_ids":int(meta.vm_id.nunique()),
        "metadata_missing_values":{str(k):int(v) for k,v in meta.isna().sum().items()},
        "metadata_deletions_at_or_before_last_observed_timestamp":int((meta.timestamp_vm_deleted<=ts.max()).sum()),
        "metadata_creation_timestamp_unique_values":int(meta.timestamp_vm_created.nunique()),
        "checksum_verification":checks}
    stats={"training_observations":int(len(train)),"training_start_timestamp":int(train.timestamp.min()),
        "training_end_timestamp":int(train.timestamp.max()),"cadence_seconds":300,
        "aggregate_cpu_mean_percent":float(vals.mean()),"aggregate_cpu_std_percentage_points":float(vals.std(ddof=1)),
        "aggregate_cpu_min_percent":float(vals.min()),"aggregate_cpu_max_percent":float(vals.max()),
        "aggregate_cpu_quantiles_percent":{str(q):float(np.quantile(vals,q)) for q in (.05,.25,.5,.75,.95)},
        "aggregate_cpu_mean_absolute_step_percentage_points":float(np.mean(np.abs(np.diff(vals)))),
        "aggregate_cpu_autocorrelation":ac,
        "vm_reading_avg_cpu_quantiles_percent":{str(q):float(np.quantile(raw,q)) for q in (.05,.25,.5,.75,.9,.95,.99)},
        "vm_reading_fraction_ge_70_percent":float(np.mean(raw>=70)),
        "vm_reading_fraction_ge_90_percent":float(np.mean(raw>=90)),
        "training_peak_timestamp_seconds":int(train.loc[train.aggregate_cpu_utilization.idxmax(),"timestamp"]),
        "training_low_timestamp_seconds":int(train.loc[train.aggregate_cpu_utilization.idxmin(),"timestamp"]),
        "active_vm_count_unique_training_values":[int(v) for v in sorted(train.active_vm_count.unique())],
        "holdout_timestamps":holdout}
    score.to_csv(out_dir/"validation_metrics.csv",index=False)
    horizon.to_csv(out_dir/"validation_by_horizon.csv",index=False)
    seg.to_csv(out_dir/"segment_summary.csv",index=False)
    result={"data_quality":dq,"training_descriptive_statistics":stats,
        "validation_metrics":score.to_dict(orient="records"),
        "validation_by_horizon":horizon.to_dict(orient="records"),"validation_design":cv,
        "selected_model":selected,"official_holdout_actuals_used_for_model_selection_or_validation":False,
        "forecast_timestamps":holdout}
    (out_dir/"analysis_results.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    plot_artifacts(train,cpu_train,seg,fig_dir)
    print(f"Verified {len(checks)} checksums; {len(cpu):,} CPU rows; {cpu.vm_id.nunique():,} VMs; {len(ts)} timestamps.")
    print(f"Training={len(train)}; forecast horizon={len(holdout)}; selected model={selected}.")
    print(score.to_string(index=False))
    print(f"Wrote tables to {out_dir} and figures to {fig_dir}.")


if __name__=="__main__":
    main()
