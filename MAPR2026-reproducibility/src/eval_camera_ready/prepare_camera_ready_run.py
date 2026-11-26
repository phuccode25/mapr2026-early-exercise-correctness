import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "camera-ready"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "rehab24" / "metadata.csv"
DEFAULT_SAMPLES_DIR = DEFAULT_METADATA.parent / "samples"

TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7]
VALIDATION_SUBJECTS = [8]
TEST_SUBJECTS = [9, 10]
PREFIXES = [0.25, 0.50, 0.75, 1.00]
SEEDS = [42, 2024, 3407]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def ensure_dirs(output_dir):
    for name in [
        "configs",
        "checkpoints",
        "logs",
        "predictions",
        "raw_metrics",
        "tables",
    ]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def split_df(df, subjects):
    return df[df["subject_id"].astype(int).isin(subjects)].copy()


def unique_samples(df):
    return df.drop_duplicates("sample_id").copy()


def check_split_integrity(train_df, val_df, test_df):
    rows = []
    split_map = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    sample_sets = {name: set(unique_samples(df)["sample_id"]) for name, df in split_map.items()}
    subject_sets = {name: set(unique_samples(df)["subject_id"].astype(int)) for name, df in split_map.items()}

    for name, df in split_map.items():
        unique = unique_samples(df)
        rows.append(
            {
                "check": f"{name}_counts",
                "status": "PASS",
                "details": (
                    f"rows={len(df)}, unique_samples={len(unique)}, "
                    f"subjects={sorted(subject_sets[name])}, "
                    f"correct={int(unique['correctness'].sum())}, "
                    f"incorrect={int((1 - unique['correctness']).sum())}"
                ),
            }
        )

    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        sample_overlap = sorted(sample_sets[left] & sample_sets[right])
        subject_overlap = sorted(subject_sets[left] & subject_sets[right])
        rows.append(
            {
                "check": f"{left}_{right}_sample_overlap",
                "status": "PASS" if not sample_overlap else "FAIL",
                "details": ",".join(sample_overlap[:10]) if sample_overlap else "none",
            }
        )
        rows.append(
            {
                "check": f"{left}_{right}_subject_overlap",
                "status": "PASS" if not subject_overlap else "FAIL",
                "details": ",".join(map(str, subject_overlap)) if subject_overlap else "none",
            }
        )

    expected_prefixes = {float(p) for p in PREFIXES}
    for name, df in split_map.items():
        bad_prefix = []
        inconsistent = []
        for sample_id, group in df.groupby("sample_id"):
            prefixes = {round(float(p), 2) for p in group["prefix"]}
            if prefixes != expected_prefixes:
                bad_prefix.append(sample_id)
            stable_cols = ["correctness", "exercise_id", "exercise_name", "subject_id"]
            for col in stable_cols:
                if group[col].nunique(dropna=False) != 1:
                    inconsistent.append(f"{sample_id}:{col}")
        rows.append(
            {
                "check": f"{name}_all_samples_have_four_prefixes",
                "status": "PASS" if not bad_prefix else "FAIL",
                "details": ",".join(bad_prefix[:10]) if bad_prefix else "all samples have 0.25/0.50/0.75/1.00",
            }
        )
        rows.append(
            {
                "check": f"{name}_labels_consistent_across_prefixes",
                "status": "PASS" if not inconsistent else "FAIL",
                "details": ",".join(inconsistent[:10]) if inconsistent else "stable labels across prefixes",
            }
        )

    return pd.DataFrame(rows)


def write_split_files(output_dir, train_df, val_df, test_df):
    split_dir = output_dir / "configs"
    train_df.to_csv(split_dir / "train_metadata.csv", index=False)
    val_df.to_csv(split_dir / "validation_metadata.csv", index=False)
    test_df.to_csv(split_dir / "test_metadata.csv", index=False)
    for split_name, split in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        for prefix in PREFIXES:
            sub = split[split["prefix"].astype(float).round(2) == round(prefix, 2)].copy()
            sub.to_csv(split_dir / f"{split_name}_prefix_{prefix}.csv", index=False)


def write_manifest(output_dir, metadata_path, samples_dir):
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "git_commit": git_commit(),
        "dataset_paths": {
            "metadata_csv": str(metadata_path),
            "processed_samples_dir": str(samples_dir),
        },
        "split_definition": {
            "train_subjects": TRAIN_SUBJECTS,
            "validation_subjects": VALIDATION_SUBJECTS,
            "test_subjects": TEST_SUBJECTS,
            "prefixes": PREFIXES,
        },
        "seeds": SEEDS,
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dimension": 128,
        "dilations": [1, 2, 4, 8],
        "lambda": 1.0,
        "checkpoint_selection_metric": "lowest_validation_loss",
        "output_directory": str(output_dir),
        "script_names": {
            "prepare": "src/eval_camera_ready/prepare_camera_ready_run.py",
            "train": "src/train_camera_ready/train_val_selected_models.py",
            "tables": "src/eval_camera_ready/build_camera_ready_results.py",
        },
        "run_status": {
            "split_integrity": "created",
            "pa_mttcn": "pending",
            "tcn_c": "pending",
            "lstm_c": "pending",
            "lstm_mt": "pending",
            "mask_pa_mttcn": "pending",
        },
    }
    with (output_dir / "RUN_MANIFEST.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def verify_samples(samples_dir):
    checksum_file = DEFAULT_METADATA.with_name("samples.sha256")
    for line in checksum_file.read_text(encoding="ascii").splitlines():
        expected, filename = line.split("  ", maxsplit=1)
        sample = samples_dir / filename
        if not sample.is_file():
            raise SystemExit(f"Missing processed sample {sample}. See the README for the required inputs.")
        if hashlib.sha256(sample.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"SHA-256 mismatch for {sample}. Use the original processed arrays.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    df = pd.read_csv(args.metadata)
    samples_dir = args.samples_dir.resolve()
    if args.metadata.resolve() == DEFAULT_METADATA:
        verify_samples(samples_dir)
    if samples_dir != DEFAULT_SAMPLES_DIR:
        df["path"] = df["path"].map(lambda path: str(samples_dir / Path(path).name))
    train_df = split_df(df, TRAIN_SUBJECTS)
    val_df = split_df(df, VALIDATION_SUBJECTS)
    test_df = split_df(df, TEST_SUBJECTS)

    integrity = check_split_integrity(train_df, val_df, test_df)
    ensure_dirs(output_dir)
    write_split_files(output_dir, train_df, val_df, test_df)
    integrity.to_csv(output_dir / "tables" / "split_integrity_check.csv", index=False)

    report_lines = ["Split integrity report", ""]
    for _, row in integrity.iterrows():
        report_lines.append(f"{row['status']}: {row['check']} - {row['details']}")
    (output_dir / "logs" / "split_integrity_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    write_manifest(output_dir, args.metadata.resolve(), samples_dir)
    if (integrity["status"] == "FAIL").any():
        raise SystemExit("Split integrity failed; do not train.")

    print("Prepared camera-ready run directory:", output_dir)
    print("Manifest:", output_dir / "RUN_MANIFEST.json")
    print("Split integrity:", output_dir / "tables" / "split_integrity_check.csv")


if __name__ == "__main__":
    main()
