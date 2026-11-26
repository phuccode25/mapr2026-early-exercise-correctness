import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_camera_ready.metrics import safe_divide, write_metric_outputs


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "paper-tables"
SEEDS = [42, 2024, 3407]
MAIN_MODELS = ["pa_mttcn", "tcn_c", "lstm_c", "lstm_mt"]
PREFIXES = [0.25, 0.50, 0.75, 1.00]
GATE_THRESHOLDS = np.round(np.arange(0.50, 0.951, 0.05), 2)


def sample_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def load_predictions(predictions_dir, model, seed, split):
    return pd.read_csv(predictions_dir / f"{model}_seed_{seed}_{split}_predictions.csv")


def load_metrics_by_prefix(output_dir, model, seed):
    return pd.read_csv(output_dir / "raw_metrics" / f"{model}_seed_{seed}_metrics_by_prefix.csv")


def summarize_metric(df, group_cols, metrics):
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["num_seeds"] = int(sub["seed"].nunique()) if "seed" in sub.columns else len(sub)
        for metric in metrics:
            values = sub[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = sample_std(values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def format_percent(mean, std):
    return f"{mean * 100:.2f} +/- {std * 100:.2f}"


def build_table1(output_dir):
    rows = []
    for seed in SEEDS:
        rows.append(load_metrics_by_prefix(output_dir, "pa_mttcn", seed))
    df = pd.concat(rows, ignore_index=True)
    summary = summarize_metric(
        df,
        ["prefix"],
        ["action_macro_f1", "correctness_macro_f1", "correctness_acc"],
    )
    table = summary.copy()
    table["observation"] = (table["prefix"] * 100).round().astype(int).astype(str) + "%"
    table["Action macro-F1"] = table.apply(
        lambda r: format_percent(r["action_macro_f1_mean"], r["action_macro_f1_std"]), axis=1
    )
    table["Correctness macro-F1"] = table.apply(
        lambda r: format_percent(r["correctness_macro_f1_mean"], r["correctness_macro_f1_std"]), axis=1
    )
    table["Correctness accuracy"] = table.apply(
        lambda r: format_percent(r["correctness_acc_mean"], r["correctness_acc_std"]), axis=1
    )
    table.to_csv(output_dir / "tables" / "table1_pa_mttcn_multiseed_val_selected.csv", index=False)
    return table


def build_table2(output_dir):
    rows = []
    for model in MAIN_MODELS:
        for seed in SEEDS:
            df = load_metrics_by_prefix(output_dir, model, seed)
            df["model"] = model
            rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    summary = summarize_metric(df, ["model", "prefix"], ["correctness_macro_f1"])
    summary["observation"] = (summary["prefix"] * 100).round().astype(int).astype(str) + "%"
    summary["Correctness macro-F1"] = summary.apply(
        lambda r: format_percent(r["correctness_macro_f1_mean"], r["correctness_macro_f1_std"]), axis=1
    )
    summary.to_csv(output_dir / "tables" / "table2_baseline_multiseed_val_selected.csv", index=False)
    return summary


def build_table3(output_dir):
    rows = []
    for model in ["pa_mttcn", "mask_pa_mttcn"]:
        for seed in SEEDS:
            df = load_metrics_by_prefix(output_dir, model, seed)
            df["model"] = model
            rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    summary = summarize_metric(df, ["model", "prefix"], ["correctness_macro_f1", "correctness_acc"])
    wide = summary.pivot(index="prefix", columns="model", values="correctness_macro_f1_mean").reset_index()
    wide["delta_mask_minus_zero"] = wide["mask_pa_mttcn"] - wide["pa_mttcn"]
    wide.to_csv(output_dir / "tables" / "table3_mask_ablation_val_selected.csv", index=False)
    return wide


def build_readiness(output_dir, predictions_dir):
    rows = []
    for seed in SEEDS:
        path = output_dir / "raw_metrics" / f"pa_mttcn_seed_{seed}_per_exercise_metrics.csv"
        rows.append(pd.read_csv(path))
    df = pd.concat(rows, ignore_index=True)
    mean_df = (
        df.groupby(["exercise_name", "prefix"])["correctness_macro_f1"]
        .agg(["mean", lambda s: s.std(ddof=1)])
        .reset_index()
        .rename(columns={"mean": "mean_correctness_macro_f1", "<lambda_0>": "std_correctness_macro_f1"})
    )
    mean_df.to_csv(output_dir / "tables" / "table5_per_exercise_readiness_seed_prefix_means.csv", index=False)

    def state(earliest, best_f1):
        if not math.isnan(earliest) and earliest <= 0.5:
            return "Early-ready"
        if not math.isnan(earliest):
            return "Delayed-ready" if earliest <= 0.75 else "Monitor"
        if best_f1 >= 0.60:
            return "Monitor"
        return "Uncertain"

    test = load_predictions(predictions_dir, "pa_mttcn", SEEDS[0], "test")
    support = test.drop_duplicates("sample_id").groupby("exercise_name")["true_correctness"].agg(["size", "sum"])
    readiness_rows = []
    for exercise, sub in mean_df.groupby("exercise_name"):
        sub = sub.sort_values("prefix")
        best_idx = sub["mean_correctness_macro_f1"].idxmax()
        best_f1 = float(sub.loc[best_idx, "mean_correctness_macro_f1"])
        best_prefix = float(sub.loc[best_idx, "prefix"])
        hits = sub[sub["mean_correctness_macro_f1"] >= 0.80]
        earliest = float(hits["prefix"].min()) if len(hits) else math.nan
        readiness_rows.append(
            {
                "exercise_name": exercise,
                "total": int(support.loc[exercise, "size"]),
                "correct": int(support.loc[exercise, "sum"]),
                "incorrect": int(support.loc[exercise, "size"] - support.loc[exercise, "sum"]),
                "best_mean_correctness_macro_f1": best_f1,
                "best_prefix": best_prefix,
                "earliest_prefix_tau_0.80": earliest,
                "state_tau_0.80_gamma_0.60": state(earliest, best_f1),
            }
        )
    readiness = pd.DataFrame(readiness_rows).sort_values("exercise_name")
    readiness.to_csv(output_dir / "tables" / "table5_per_exercise_readiness_val_selected.csv", index=False)

    sensitivity_rows = []
    for tau in [0.75, 0.80, 0.85, 0.90]:
        counts = {"Early-ready": 0, "Delayed-ready": 0, "Monitor": 0, "Uncertain": 0}
        states = []
        for exercise, sub in mean_df.groupby("exercise_name"):
            sub = sub.sort_values("prefix")
            best_f1 = float(sub["mean_correctness_macro_f1"].max())
            hits = sub[sub["mean_correctness_macro_f1"] >= tau]
            earliest = float(hits["prefix"].min()) if len(hits) else math.nan
            st = state(earliest, best_f1)
            counts[st] += 1
            states.append(f"{exercise}: {st}")
        sensitivity_rows.append({"tau": tau, **counts, "exercise_states": "; ".join(states)})
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(output_dir / "tables" / "table4_readiness_sensitivity_val_selected.csv", index=False)
    return readiness, sensitivity


def gate_metrics(df, q):
    y = df["true_correctness"].astype(int).to_numpy()
    pred = (df["correctness_probability"].astype(float).to_numpy() >= q).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "FPR": safe_divide(fp, fp + tn),
        "FNR": safe_divide(fn, fn + tp),
        "positive_feedback_coverage": safe_divide(tp + fp, len(df)),
        "selective_precision": safe_divide(tp, tp + fp),
    }


def build_table6(output_dir, predictions_dir):
    df = pd.concat([load_predictions(predictions_dir, "pa_mttcn", seed, "test") for seed in SEEDS], ignore_index=True)
    df = df[np.isclose(df["prefix"].astype(float), 0.75)].copy()
    rows = []
    for q in GATE_THRESHOLDS:
        rows.append(
            {
                "model": "pa_mttcn",
                "prefix": 0.75,
                "aggregation": "pooled_three_seed_test_predictions",
                "N": int(len(df)),
                "threshold_q": q,
                **gate_metrics(df, q),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "tables" / "table6_posthoc_confidence_sweep.csv", index=False)
    return table


def build_table7(output_dir, predictions_dir):
    rows = []
    for seed in SEEDS:
        val = load_predictions(predictions_dir, "pa_mttcn", seed, "validation")
        test = load_predictions(predictions_dir, "pa_mttcn", seed, "test")
        for prefix in PREFIXES:
            val_sub = val[np.isclose(val["prefix"].astype(float), prefix)]
            test_sub = test[np.isclose(test["prefix"].astype(float), prefix)]
            scored = [(q, gate_metrics(val_sub, q)) for q in GATE_THRESHOLDS]
            feasible = [(q, m) for q, m in scored if m["FPR"] <= 0.10]
            if feasible:
                q_star, val_m = max(feasible, key=lambda item: (item[1]["positive_feedback_coverage"], -item[0]))
                rule = "max_seed_validation_coverage_subject_to_FPR<=0.10"
            else:
                q_star, val_m = min(
                    scored,
                    key=lambda item: (item[1]["FPR"], -item[1]["positive_feedback_coverage"], item[0]),
                )
                rule = "no_q_met_FPR<=0.10; selected_min_seed_validation_FPR"
            test_m = gate_metrics(test_sub, q_star)
            rows.append(
                {
                    "seed": seed,
                    "prefix": prefix,
                    "selected_q": q_star,
                    "selection_rule": rule,
                    **{f"validation_{k}": v for k, v in val_m.items()},
                    **{f"test_{k}": v for k, v in test_m.items()},
                }
            )
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output_dir / "tables" / "table7_validation_gated_by_seed.csv", index=False)
    summary = summarize_metric(
        per_seed,
        ["prefix"],
        ["selected_q", "test_FPR", "test_FNR", "test_positive_feedback_coverage", "test_selective_precision"],
    )
    summary.to_csv(output_dir / "tables" / "table7_validation_gated_summary.csv", index=False)
    return per_seed, summary


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(prob):
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
    return np.log(prob / (1.0 - prob))


def nll(y_true, prob):
    prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
    return float(-(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob)).mean())


def fit_temperature(y_true, prob):
    logits = logit(prob)
    candidates = np.round(np.arange(0.50, 5.001, 0.01), 2)
    losses = [(temp, nll(y_true, sigmoid(logits / temp))) for temp in candidates]
    return min(losses, key=lambda item: item[1])[0]


def brier(y_true, prob):
    return float(np.mean((prob - y_true) ** 2))


def ece(y_true, prob, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(prob)
    value = 0.0
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        mask = (prob >= left) & (prob <= right) if i == bins - 1 else (prob >= left) & (prob < right)
        count = int(mask.sum())
        if count:
            value += (count / total) * abs(float(prob[mask].mean()) - float(y_true[mask].mean()))
    return float(value)


def build_table8(output_dir, predictions_dir, calibration):
    validation = {seed: load_predictions(predictions_dir, "pa_mttcn", seed, "validation") for seed in SEEDS}
    if calibration == "pooled":
        pooled = pd.concat(validation.values(), ignore_index=True)
        pooled_temperature = fit_temperature(pooled["true_correctness"].to_numpy(), pooled["correctness_probability"].to_numpy())
    rows = []
    for seed, val in validation.items():
        test = load_predictions(predictions_dir, "pa_mttcn", seed, "test")
        for prefix in PREFIXES:
            val_sub = val[np.isclose(val["prefix"].astype(float), prefix)]
            test_sub = test[np.isclose(test["prefix"].astype(float), prefix)]
            val_y = val_sub["true_correctness"].astype(int).to_numpy()
            val_prob = val_sub["correctness_probability"].astype(float).to_numpy()
            test_y = test_sub["true_correctness"].astype(int).to_numpy()
            test_prob = test_sub["correctness_probability"].astype(float).to_numpy()
            temperature = pooled_temperature if calibration == "pooled" else fit_temperature(val_y, val_prob)
            scaled = sigmoid(logit(test_prob) / temperature)
            rows.append(
                {
                    "seed": seed,
                    "prefix": prefix,
                    "temperature": temperature,
                    "test_ECE_before": ece(test_y, test_prob, bins=10),
                    "test_ECE_after": ece(test_y, scaled, bins=10),
                    "test_Brier_before": brier(test_y, test_prob),
                    "test_Brier_after": brier(test_y, scaled),
                    "validation_num_predictions": len(pooled) if calibration == "pooled" else len(val_sub),
                    "test_num_predictions": int(len(test_sub)),
                }
            )
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output_dir / "tables" / "table8_calibration_per_seed_prefix.csv", index=False)
    summary = summarize_metric(
        per_seed,
        ["prefix"],
        ["temperature", "test_ECE_before", "test_ECE_after", "test_Brier_before", "test_Brier_after"],
    )
    summary.to_csv(output_dir / "tables" / "table8_calibration_summary.csv", index=False)
    return per_seed, summary


def write_markdown_preview(output_dir, table1, table2, table3):
    def simple_markdown(df):
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
        return "\n".join(lines)

    lines = ["# Camera-ready validation-selected result preview", ""]
    lines.append("## Table I")
    lines.append(simple_markdown(table1[["observation", "Action macro-F1", "Correctness macro-F1", "Correctness accuracy"]]))
    lines.append("")
    lines.append("## Table II")
    lines.append(simple_markdown(table2[["model", "observation", "Correctness macro-F1"]]))
    lines.append("")
    lines.append("## Table III")
    lines.append(simple_markdown(table3))
    lines.append("")
    (output_dir / "summaries" / "tables_1_3_preview.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Rebuild Tables I?VIII from validation and test predictions.")
    parser.add_argument("--predictions-dir", type=Path, default=PROJECT_ROOT / "results" / "predictions")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration", choices=["per-seed-prefix", "pooled"], default="per-seed-prefix")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    predictions_dir = args.predictions_dir.resolve()
    for directory in ("raw_metrics", "tables", "summaries"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)

    for model in [*MAIN_MODELS, "mask_pa_mttcn"]:
        for seed in SEEDS:
            predictions = load_predictions(predictions_dir, model, seed, "test")
            write_metric_outputs(predictions, model, seed, output_dir)
    table1 = build_table1(output_dir)
    table2 = build_table2(output_dir)
    table3 = build_table3(output_dir)
    build_readiness(output_dir, predictions_dir)
    build_table6(output_dir, predictions_dir)
    build_table7(output_dir, predictions_dir)
    build_table8(output_dir, predictions_dir, args.calibration)
    write_markdown_preview(output_dir, table1, table2, table3)
    print("Saved camera-ready tables to:", output_dir / "tables")


if __name__ == "__main__":
    main()
