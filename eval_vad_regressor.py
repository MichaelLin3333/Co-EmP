from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import DatasetDict, load_dataset, load_from_disk
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import pearsonr, spearmanr


# ============================================================
# 1. Dataset loading
# ============================================================

def load_integrated_dataset(dataset_path: str) -> DatasetDict:
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # Case 1: folder contains hf_dataset/
    hf_dir = path / "hf_dataset"
    if hf_dir.exists():
        ds = load_from_disk(str(hf_dir))
        if not isinstance(ds, DatasetDict):
            ds = DatasetDict({"train": ds})
        return ds

    # Case 2: direct saved DatasetDict folder
    if (path / "dataset_dict.json").exists() or (path / "state.json").exists():
        ds = load_from_disk(str(path))
        if not isinstance(ds, DatasetDict):
            ds = DatasetDict({"train": ds})
        return ds

    # Case 3: JSONL split files
    data_files = {}
    for split in ["train", "validation", "test"]:
        fp = path / f"{split}.jsonl"
        if fp.exists():
            data_files[split] = str(fp)

    if data_files:
        return load_dataset("json", data_files=data_files)

    # Case 4: all.jsonl only
    all_file = path / "all.jsonl"
    if all_file.exists():
        return load_dataset("json", data_files={"train": str(all_file)})

    raise ValueError(
        f"Could not recognize dataset format in {dataset_path}. "
        "Expected hf_dataset/, train.jsonl, validation.jsonl, test.jsonl, or all.jsonl."
    )


def normalize_labels(example: Dict[str, Any]) -> Dict[str, Any]:
    if "labels" in example and example["labels"] is not None:
        labels = example["labels"]

        if isinstance(labels, str):
            labels = json.loads(labels)

        example["labels"] = [float(labels[0]), float(labels[1]), float(labels[2])]
        return example

    if all(k in example and example[k] is not None for k in ["valence", "arousal", "dominance"]):
        example["labels"] = [
            float(example["valence"]),
            float(example["arousal"]),
            float(example["dominance"]),
        ]
        return example

    raise ValueError(f"Missing VAD labels: {example}")


def clamp_labels(example: Dict[str, Any]) -> Dict[str, Any]:
    example["labels"] = [max(0.0, min(1.0, float(x))) for x in example["labels"]]
    return example


# ============================================================
# 2. Collator
# ============================================================

class EvalCollator:
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


# ============================================================
# 3. Metrics
# ============================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    result = {}

    mse = mean_squared_error(y_true, y_pred)
    result["mse"] = float(mse)
    result["rmse"] = float(math.sqrt(mse))
    result["mae"] = float(mean_absolute_error(y_true, y_pred))

    names = ["valence", "arousal", "dominance"]

    for i, name in enumerate(names):
        mse_i = mean_squared_error(y_true[:, i], y_pred[:, i])
        result[f"{name}_mse"] = float(mse_i)
        result[f"{name}_rmse"] = float(math.sqrt(mse_i))
        result[f"{name}_mae"] = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))

        try:
            result[f"{name}_r2"] = float(r2_score(y_true[:, i], y_pred[:, i]))
        except Exception:
            result[f"{name}_r2"] = float("nan")
        try:
            result[f"{name}_pearson"] = float(pearsonr(y_true[:, i], y_pred[:, i])[0])
        except Exception:
            result[f"{name}_pearson"] = float("nan")

        try:
            result[f"{name}_spearman"] = float(spearmanr(y_true[:, i], y_pred[:, i])[0])
        except Exception:
            result[f"{name}_spearman"] = float("nan")

    try:
        result["r2_mean"] = float(
            np.mean(
                [
                    result["valence_r2"],
                    result["arousal_r2"],
                    result["dominance_r2"],
                ]
            )
        )
    except Exception:
        result["r2_mean"] = float("nan")

    return result


def evaluate_groups(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_cols: List[str],
) -> pd.DataFrame:
    rows = []

    grouped = df.groupby(group_cols, dropna=False)

    for group_key, indices in grouped.groups.items():
        idx = list(indices)

        if len(idx) < 2:
            continue

        group_true = y_true[idx]
        group_pred = y_pred[idx]

        metrics = regression_metrics(group_true, group_pred)

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = {
            col: value
            for col, value in zip(group_cols, group_key)
        }

        row["n"] = len(idx)
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["rmse", "mae"], ascending=True)


# ============================================================
# 4. Prediction
# ============================================================

@torch.no_grad()
def predict_split(
    model,
    tokenizer,
    split_ds,
    device: str,
    batch_size: int,
    max_length: int,
    activation: str,
):
    model.eval()

    collator = EvalCollator(tokenizer, max_length=max_length)

    dataloader = DataLoader(
        split_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    all_preds = []
    all_labels = []

    for batch in dataloader:
        labels = batch.pop("labels").numpy()

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        outputs = model(**batch)
        logits = outputs.logits.float()

        if activation == "sigmoid":
            preds = torch.sigmoid(logits)
        elif activation == "clip":
            preds = torch.clamp(logits, 0.0, 1.0)
        elif activation == "none":
            preds = logits
        else:
            raise ValueError(f"Unknown activation: {activation}")

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels)

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    if activation == "none":
        # Still clamp for report stability unless you explicitly want raw outputs.
        y_pred = np.clip(y_pred, 0.0, 1.0)

    return y_true, y_pred


# ============================================================
# 5. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="integrated_vad_utterance_dataset")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="vad_source_eval")

    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument(
        "--activation",
        type=str,
        default="none",
        choices=["sigmoid", "clip", "none"],
        help=(
            "Use sigmoid if your training loss applied sigmoid to logits. "
            "Use clip if your training loss used raw logits but clipped only for metrics."
        ),
    )

    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading dataset from: {args.dataset_path}")
    ds = load_integrated_dataset(args.dataset_path)

    if args.split not in ds:
        raise ValueError(f"Split {args.split} not found. Available splits: {list(ds.keys())}")

    ds = ds.map(normalize_labels)
    ds = ds.map(clamp_labels)

    raw_split = ds[args.split]
    raw_df = pd.DataFrame(raw_split)

    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)

    print(f"Running prediction on split: {args.split}")
    y_true, y_pred = predict_split(
        model=model,
        tokenizer=tokenizer,
        split_ds=raw_split,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        activation=args.activation,
    )

    overall = regression_metrics(y_true, y_pred)
    overall["split"] = args.split
    overall["n"] = len(raw_split)

    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(out_dir / f"{args.split}_overall.csv", index=False)

    print("\nOverall metrics:")
    print(overall_df.to_string(index=False))

    groupings = [
        ["source"],
        ["label_source"],
        ["source", "label_source"],
    ]

    for group_cols in groupings:
        report = evaluate_groups(
            df=raw_df,
            y_true=y_true,
            y_pred=y_pred,
            group_cols=group_cols,
        )

        name = "_by_" + "_and_".join(group_cols)
        save_path = out_dir / f"{args.split}{name}.csv"
        report.to_csv(save_path, index=False)

        print(f"\nMetrics grouped by {group_cols}:")
        print(report.to_string(index=False))
        print(f"Saved: {save_path}")

    # Save predictions for error analysis.
    pred_df = raw_df.copy()
    pred_df["pred_valence"] = y_pred[:, 0]
    pred_df["pred_arousal"] = y_pred[:, 1]
    pred_df["pred_dominance"] = y_pred[:, 2]
    pred_df["abs_err_valence"] = np.abs(y_pred[:, 0] - y_true[:, 0])
    pred_df["abs_err_arousal"] = np.abs(y_pred[:, 1] - y_true[:, 1])
    pred_df["abs_err_dominance"] = np.abs(y_pred[:, 2] - y_true[:, 2])
    pred_df["abs_err_mean"] = pred_df[
        ["abs_err_valence", "abs_err_arousal", "abs_err_dominance"]
    ].mean(axis=1)

    pred_path = out_dir / f"{args.split}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved predictions: {pred_path}")


if __name__ == "__main__":
    main()