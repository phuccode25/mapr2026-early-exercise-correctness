import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_camera_ready.metrics import MULTITASK_MODELS, safe_divide
from src.models.gru import CorrectnessOnlyRNN, MultiTaskRNN
from src.models.tcn import MultiTaskTCN
from src.models.tcn_correctness_only import CorrectnessOnlyTCN


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "camera-ready"

DEFAULT_SEEDS = [42, 2024, 3407]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_pose_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def apply_prefix_zero(pose, prefix):
    keep_len = max(1, int(pose.shape[0] * float(prefix)))
    output = np.zeros_like(pose, dtype=np.float32)
    output[:keep_len] = pose[:keep_len]
    return output


def apply_prefix_with_mask(pose, prefix):
    keep_len = max(1, int(pose.shape[0] * float(prefix)))
    output_xyz = np.zeros_like(pose, dtype=np.float32)
    output_xyz[:keep_len] = pose[:keep_len]
    mask = np.zeros((pose.shape[0], pose.shape[1], 1), dtype=np.float32)
    mask[:keep_len, :, 0] = 1.0
    return np.concatenate([output_xyz, mask], axis=-1)


class CameraReadyDataset(Dataset):
    def __init__(self, csv_path, mask_aware=False):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.mask_aware = mask_aware

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        pose = np.load(resolve_pose_path(row["path"]), allow_pickle=False).astype(np.float32)
        if pose.shape != (64, 26, 3):
            raise ValueError(f"Expected pose shape (64, 26, 3), got {pose.shape}")
        prefix = float(row["prefix"])
        if self.mask_aware:
            pose = apply_prefix_with_mask(pose, prefix).reshape(64, 104)
        else:
            pose = apply_prefix_zero(pose, prefix).reshape(64, 78)
        return {
            "x": torch.tensor(pose, dtype=torch.float32),
            "y_action": torch.tensor(int(row["exercise_id"]) - 1, dtype=torch.long),
            "y_correctness": torch.tensor(float(row["correctness"]), dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
            "subject_id": int(row["subject_id"]),
            "exercise_name": str(row["exercise_name"]),
            "prefix": prefix,
        }


def build_model(model_name):
    if model_name == "pa_mttcn":
        return MultiTaskTCN(input_dim=78, hidden_dim=128, num_actions=6, dropout=0.1)
    if model_name == "tcn_c":
        return CorrectnessOnlyTCN(input_dim=78, hidden_dim=128, dropout=0.1)
    if model_name == "lstm_c":
        return CorrectnessOnlyRNN(input_dim=78, hidden_dim=128, dropout=0.1, rnn_type="lstm")
    if model_name == "lstm_mt":
        return MultiTaskRNN(input_dim=78, hidden_dim=128, num_actions=6, dropout=0.1, rnn_type="lstm")
    if model_name == "mask_pa_mttcn":
        return MultiTaskTCN(input_dim=104, hidden_dim=128, num_actions=6, dropout=0.1)
    raise ValueError(f"Unknown model: {model_name}")


def forward_model(model, model_name, x):
    if model_name in MULTITASK_MODELS:
        return model(x)
    correctness_logits = model(x)
    return None, correctness_logits


def compute_loss(model_name, action_logits, correctness_logits, y_action, y_correctness):
    correctness_loss = nn.BCEWithLogitsLoss()(correctness_logits, y_correctness)
    if model_name in MULTITASK_MODELS:
        action_loss = nn.CrossEntropyLoss()(action_logits, y_action)
        return action_loss + correctness_loss, action_loss.detach(), correctness_loss.detach()
    return correctness_loss, torch.tensor(float("nan")), correctness_loss.detach()


def train_one_epoch(model, model_name, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_action_loss = 0.0
    total_correctness_loss = 0.0
    total_n = 0
    for batch in loader:
        x = batch["x"].to(device)
        y_action = batch["y_action"].to(device)
        y_correctness = batch["y_correctness"].to(device)
        optimizer.zero_grad()
        action_logits, correctness_logits = forward_model(model, model_name, x)
        loss, action_loss, correctness_loss = compute_loss(
            model_name, action_logits, correctness_logits, y_action, y_correctness
        )
        loss.backward()
        optimizer.step()
        n = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * n
        if not torch.isnan(action_loss):
            total_action_loss += float(action_loss.cpu()) * n
        total_correctness_loss += float(correctness_loss.cpu()) * n
        total_n += n
    return {
        "loss": safe_divide(total_loss, total_n),
        "action_loss": safe_divide(total_action_loss, total_n) if model_name in MULTITASK_MODELS else math.nan,
        "correctness_loss": safe_divide(total_correctness_loss, total_n),
    }


@torch.no_grad()
def evaluate_loss(model, model_name, loader, device):
    model.eval()
    total_loss = 0.0
    total_action_loss = 0.0
    total_correctness_loss = 0.0
    total_n = 0
    for batch in loader:
        x = batch["x"].to(device)
        y_action = batch["y_action"].to(device)
        y_correctness = batch["y_correctness"].to(device)
        action_logits, correctness_logits = forward_model(model, model_name, x)
        loss, action_loss, correctness_loss = compute_loss(
            model_name, action_logits, correctness_logits, y_action, y_correctness
        )
        n = int(x.shape[0])
        total_loss += float(loss.cpu()) * n
        if not torch.isnan(action_loss):
            total_action_loss += float(action_loss.cpu()) * n
        total_correctness_loss += float(correctness_loss.cpu()) * n
        total_n += n
    return {
        "loss": safe_divide(total_loss, total_n),
        "action_loss": safe_divide(total_action_loss, total_n) if model_name in MULTITASK_MODELS else math.nan,
        "correctness_loss": safe_divide(total_correctness_loss, total_n),
    }


@torch.no_grad()
def predict(model, model_name, loader, device, seed, checkpoint_epoch, split_name):
    model.eval()
    rows = []
    for batch in loader:
        x = batch["x"].to(device)
        action_logits, correctness_logits = forward_model(model, model_name, x)
        correctness_prob = torch.sigmoid(correctness_logits).cpu().numpy()
        correctness_pred = (correctness_prob >= 0.5).astype(int)
        if action_logits is not None:
            action_pred = action_logits.argmax(dim=1).cpu().numpy().astype(int)
        else:
            action_pred = np.full(len(correctness_pred), -1, dtype=int)

        for i in range(len(correctness_pred)):
            rows.append(
                {
                    "sample_id": batch["sample_id"][i],
                    "subject_id": int(batch["subject_id"][i]),
                    "exercise_name": batch["exercise_name"][i],
                    "prefix": float(batch["prefix"][i]),
                    "true_action": int(batch["y_action"][i]),
                    "true_correctness": int(float(batch["y_correctness"][i])),
                    "predicted_action": int(action_pred[i]),
                    "predicted_correctness": int(correctness_pred[i]),
                    "correctness_logit": float(correctness_logits.cpu().numpy()[i]),
                    "correctness_probability": float(correctness_prob[i]),
                    "seed": int(seed),
                    "model": model_name,
                    "checkpoint_epoch": int(checkpoint_epoch),
                    "split": split_name,
                }
            )
    return pd.DataFrame(rows)


def update_manifest_status(output_dir, model_name, seed, status, message=""):
    manifest_path = output_dir / "RUN_MANIFEST.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.setdefault("runs", {})
    runs[f"{model_name}_seed_{seed}"] = {
        "status": status,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    model_status = manifest.setdefault("run_status", {}).get(model_name, "pending")
    if status == "complete":
        manifest["run_status"][model_name] = "partially_complete"
    if model_status == "failed":
        manifest["run_status"][model_name] = "failed"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def train_model_seed(model_name, seed, args, device):
    set_seed(seed)
    output_dir = args.output_dir.resolve()
    split_dir = output_dir / "configs"
    mask_aware = model_name == "mask_pa_mttcn"

    train_dataset = CameraReadyDataset(split_dir / "train_metadata.csv", mask_aware=mask_aware)
    val_dataset = CameraReadyDataset(split_dir / "validation_metadata.csv", mask_aware=mask_aware)
    test_dataset = CameraReadyDataset(split_dir / "test_metadata.csv", mask_aware=mask_aware)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = output_dir / "checkpoints" / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"{model_name}_seed_{seed}_best_val_loss.pt"
    best_val_loss = float("inf")
    best_epoch = -1
    train_rows = []
    val_rows = []

    update_manifest_status(output_dir, model_name, seed, "running")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, model_name, train_loader, optimizer, device)
        val_metrics = evaluate_loss(model, model_name, val_loader, device)

        train_rows.append({"model": model_name, "seed": seed, "epoch": epoch, **train_metrics})
        val_rows.append({"model": model_name, "seed": seed, "epoch": epoch, **val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_name": model_name,
                    "seed": seed,
                    "epoch": epoch,
                    "best_validation_loss": best_val_loss,
                    "hyperparameters": {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "learning_rate": args.lr,
                        "weight_decay": args.weight_decay,
                        "hidden_dim": 128,
                        "dilations": [1, 2, 4, 8],
                        "lambda": 1.0,
                    },
                    "split_subjects": {
                        "train": [1, 2, 3, 4, 5, 6, 7],
                        "validation": [8],
                        "test": [9, 10],
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                best_ckpt,
            )

        print(
            f"{model_name} seed {seed} epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f}"
        )

    pd.DataFrame(train_rows).to_csv(
        output_dir / "logs" / f"{model_name}_seed_{seed}_training_log.csv",
        index=False,
    )
    pd.DataFrame(val_rows).to_csv(
        output_dir / "logs" / f"{model_name}_seed_{seed}_validation_log.csv",
        index=False,
    )

    if best_epoch < 0:
        raise RuntimeError(f"{model_name} seed {seed} produced no usable validation checkpoint.")
    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    pred_df = predict(model, model_name, test_loader, device, seed, best_epoch, "test")
    pred_path = output_dir / "predictions" / f"{model_name}_seed_{seed}_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    val_pred = predict(model, model_name, val_loader, device, seed, best_epoch, "validation")
    val_pred.to_csv(output_dir / "predictions" / f"{model_name}_seed_{seed}_validation_predictions.csv", index=False)

    metadata = {
        "model": model_name,
        "seed": seed,
        "checkpoint_path": best_ckpt.relative_to(output_dir).as_posix(),
        "checkpoint_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "test_prediction_csv": pred_path.relative_to(output_dir).as_posix(),
        "device": str(device),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "test_evaluated_during_checkpoint_selection": False,
    }
    (output_dir / "checkpoints" / model_name / f"{model_name}_seed_{seed}_best_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (output_dir / "logs" / f"{model_name}_seed_{seed}.complete").write_text(
        "complete\n",
        encoding="utf-8",
    )
    update_manifest_status(output_dir, model_name, seed, "complete")
    return metadata


def finalize_model_status(output_dir, model_name, seeds):
    manifest_path = output_dir / "RUN_MANIFEST.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", {})
    complete = all(runs.get(f"{model_name}_seed_{seed}", {}).get("status") == "complete" for seed in seeds)
    manifest.setdefault("run_status", {})[model_name] = "complete" if complete else "incomplete"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pa_mttcn", "tcn_c", "lstm_c", "lstm_mt", "mask_pa_mttcn"],
        choices=["pa_mttcn", "tcn_c", "lstm_c", "lstm_mt", "mask_pa_mttcn"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested CUDA, but CUDA is not available.")

    device = torch.device(args.device)
    for model_name in args.models:
        for seed in args.seeds:
            train_model_seed(model_name, seed, args, device)
        finalize_model_status(args.output_dir.resolve(), model_name, args.seeds)


if __name__ == "__main__":
    main()
