"""Classification metrics shared by training and table generation."""

import math

import numpy as np
import pandas as pd


MULTITASK_MODELS = {"pa_mttcn", "lstm_mt", "mask_pa_mttcn"}


def safe_divide(a, b):
    return float(a) / float(b) if b else 0.0


def multiclass_macro_f1(y_true, y_pred, num_classes=6):
    scores = []
    for cls in range(num_classes):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        scores.append(safe_divide(2 * precision * recall, precision + recall))
    return float(np.mean(scores))


def binary_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def binary_macro_f1_from_counts(tp, fp, tn, fn):
    precision_pos = safe_divide(tp, tp + fp)
    recall_pos = safe_divide(tp, tp + fn)
    f1_pos = safe_divide(2 * precision_pos * recall_pos, precision_pos + recall_pos)
    precision_neg = safe_divide(tn, tn + fn)
    recall_neg = safe_divide(tn, tn + fp)
    f1_neg = safe_divide(2 * precision_neg * recall_neg, precision_neg + recall_neg)
    return (f1_pos + f1_neg) / 2.0, f1_pos, f1_neg


def metrics_for_df(df, include_action):
    y_true = df["true_correctness"].astype(int).to_numpy()
    y_pred = df["predicted_correctness"].astype(int).to_numpy()
    tp, fp, tn, fn = binary_counts(y_true, y_pred)
    macro_f1, f1_correct, f1_incorrect = binary_macro_f1_from_counts(tp, fp, tn, fn)
    row = {
        "num_samples": int(len(df)),
        "correctness_acc": float((y_true == y_pred).mean()) if len(df) else math.nan,
        "correctness_macro_f1": macro_f1,
        "f1_correct": f1_correct,
        "f1_incorrect": f1_incorrect,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "FPR": safe_divide(fp, fp + tn),
        "FNR": safe_divide(fn, fn + tp),
    }
    if include_action:
        action_true = df["true_action"].astype(int).to_numpy()
        action_pred = df["predicted_action"].astype(int).to_numpy()
        row["action_acc"] = float((action_true == action_pred).mean()) if len(df) else math.nan
        row["action_macro_f1"] = multiclass_macro_f1(action_true, action_pred, num_classes=6)
    else:
        row["action_acc"] = math.nan
        row["action_macro_f1"] = math.nan
    return row


def write_metric_outputs(pred_df, model_name, seed, output_dir):
    include_action = model_name in MULTITASK_MODELS
    prefix_rows = []
    for prefix, sub in pred_df.groupby("prefix"):
        row = {"model": model_name, "seed": seed, "prefix": float(prefix)}
        row.update(metrics_for_df(sub, include_action))
        prefix_rows.append(row)
    prefix_df = pd.DataFrame(prefix_rows).sort_values("prefix")
    prefix_df.to_csv(output_dir / "raw_metrics" / f"{model_name}_seed_{seed}_metrics_by_prefix.csv", index=False)

    exercise_rows = []
    for (exercise, prefix), sub in pred_df.groupby(["exercise_name", "prefix"]):
        row = {"model": model_name, "seed": seed, "exercise_name": exercise, "prefix": float(prefix)}
        row.update(metrics_for_df(sub, include_action))
        exercise_rows.append(row)
    pd.DataFrame(exercise_rows).sort_values(["exercise_name", "prefix"]).to_csv(
        output_dir / "raw_metrics" / f"{model_name}_seed_{seed}_per_exercise_metrics.csv",
        index=False,
    )

    confusion_cols = ["model", "seed", "prefix", "num_samples", "TP", "FP", "TN", "FN", "FPR", "FNR"]
    prefix_df[confusion_cols].to_csv(
        output_dir / "raw_metrics" / f"{model_name}_seed_{seed}_confusion_by_prefix.csv",
        index=False,
    )
