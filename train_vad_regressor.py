from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset, load_from_disk
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ============================================================
# 1. Weighted Trainer
# ============================================================

class WeightedMSETrainer(Trainer):
    """
    Custom Trainer for VAD regression.

    Computes weighted MSE across 3 dimensions:
    valence, arousal, dominance.

    This is useful because the integrated dataset mixes:
    - gold VAD labels from EmoBank
    - weak converted labels from DailyDialog / EmpatheticDialogues
    - synthetic VAD labels from custom AI-generated dialogues
    """

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        **kwargs,
    ):
        loss_weight = inputs.pop("loss_weight", None)
        labels = inputs.get("labels")

        labels = labels.float()

        if not torch.isfinite(labels).all():
            print("Bad labels:", labels)
            raise ValueError("Non-finite labels found.")

        if loss_weight is not None:
            loss_weight = loss_weight.float()
            if not torch.isfinite(loss_weight).all():
                print("Bad loss weights:", loss_weight)
                raise ValueError("Non-finite loss weights found.")

        outputs = model(**inputs)
        logits = outputs.logits

        labels = labels.float()

        if not torch.isfinite(labels).all():
            print("Bad labels:", labels)
            raise ValueError("Non-finite labels found.")

        if loss_weight is not None:
            loss_weight = loss_weight.float()
            if not torch.isfinite(loss_weight).all():
                print("Bad loss weights:", loss_weight)
                raise ValueError("Non-finite loss weights found.")
            
        

        # Per-example MSE averaged across V/A/D.
        per_example_loss = F.mse_loss(
            logits.float(),
            labels.float(),
            reduction="none",
        ).mean(dim=1)

        if not torch.isfinite(per_example_loss).all():
            print("Bad per-example loss:", per_example_loss)
            print("Logits:", logits)
            print("Labels:", labels)
            raise ValueError("Non-finite per-example loss found.")

        if loss_weight is not None:
            #print("loss_weight batch:", loss_weight[:8])
            loss_weight = loss_weight.to(per_example_loss.device).float()
            loss = (per_example_loss * loss_weight).sum() / loss_weight.sum().clamp_min(1e-8)
        else:
            loss = per_example_loss.mean()
            #print("No loss_weight found!")

        if not torch.isfinite(loss):
            print("Bad final loss:", loss)
            print("Per-example loss:", per_example_loss)
            print("Loss weight:", loss_weight)
            raise ValueError("Non-finite final loss found.")

        return (loss, outputs) if return_outputs else loss


# ============================================================
# 2. Dataset loading
# ============================================================

def load_integrated_dataset(dataset_path: str) -> DatasetDict:
    """
    Loads either:
    1. a Hugging Face dataset directory saved by DatasetDict.save_to_disk()
    2. a directory containing train.jsonl / validation.jsonl / test.jsonl
    3. a single all.jsonl file
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # Case 1: Saved HF dataset directory.
    if (path / "dataset_dict.json").exists() or (path / "state.json").exists():
        ds = load_from_disk(str(path))
        if not isinstance(ds, DatasetDict):
            ds = DatasetDict({"train": ds})
        return ds

    # Case 2: Previous integrated dataset output folder.
    hf_dir = path / "hf_dataset"
    if hf_dir.exists():
        ds = load_from_disk(str(hf_dir))
        if not isinstance(ds, DatasetDict):
            ds = DatasetDict({"train": ds})
        return ds

    # Case 3: JSONL split files.
    data_files = {}

    for split in ["train", "validation", "test"]:
        file_path = path / f"{split}.jsonl"
        if file_path.exists():
            data_files[split] = str(file_path)

    if data_files:
        return load_dataset("json", data_files=data_files)

    # Case 4: all.jsonl only.
    all_file = path / "all.jsonl"
    if all_file.exists():
        ds = load_dataset("json", data_files={"train": str(all_file)})
        return ds

    raise ValueError(
        f"Could not recognize dataset format in {dataset_path}. "
        "Expected hf_dataset/, train.jsonl, validation.jsonl, test.jsonl, or all.jsonl."
    )


def ensure_splits(ds: DatasetDict, seed: int) -> DatasetDict:
    """
    Ensures train/validation/test splits exist.

    If only train exists, creates:
    - train: 80%
    - validation: 10%
    - test: 10%
    """

    if "train" not in ds:
        # Use the first available split as train.
        first_split = list(ds.keys())[0]
        ds = DatasetDict({"train": ds[first_split]})

    if "validation" in ds and "test" in ds:
        return ds

    if "validation" not in ds and "test" not in ds:
        temp = ds["train"].train_test_split(test_size=0.2, seed=seed)
        test_valid = temp["test"].train_test_split(test_size=0.5, seed=seed)

        return DatasetDict(
            {
                "train": temp["train"],
                "validation": test_valid["train"],
                "test": test_valid["test"],
            }
        )

    if "validation" not in ds and "test" in ds:
        temp = ds["train"].train_test_split(test_size=0.1, seed=seed)
        return DatasetDict(
            {
                "train": temp["train"],
                "validation": temp["test"],
                "test": ds["test"],
            }
        )

    if "validation" in ds and "test" not in ds:
        temp = ds["train"].train_test_split(test_size=0.1, seed=seed)
        return DatasetDict(
            {
                "train": temp["train"],
                "validation": ds["validation"],
                "test": temp["test"],
            }
        )

    return ds


# ============================================================
# 3. Label handling
# ============================================================

def get_label_weight(example: Dict[str, Any]) -> float:
    """
    Loss weights by label reliability.

    You can tune these.

    Suggested:
    - EmoBank gold VAD: high weight
    - custom synthetic VAD: medium/high, depending on quality
    - categorical converted labels: lower weight
    """

    label_source = str(example.get("label_source", "")).lower()
    source = str(example.get("source", "")).lower()

    if "gold_vad" in label_source or "emobank" in source:
        return 1.0

    if "synthetic" in label_source or "custom" in source:
        return 0.4

    if "categorical_to_vad" in label_source:
        return 0.15

    return 0.5


def normalize_example_labels(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures every example has:
    labels = [valence, arousal, dominance]
    """

    if "labels" in example and example["labels"] is not None:
        labels = example["labels"]

        # Sometimes JSON loading may return labels as a string.
        if isinstance(labels, str):
            labels = json.loads(labels)

        example["labels"] = [float(labels[0]), float(labels[1]), float(labels[2])]
        return example

    required = ["valence", "arousal", "dominance"]

    if all(k in example and example[k] is not None for k in required):
        example["labels"] = [
            float(example["valence"]),
            float(example["arousal"]),
            float(example["dominance"]),
        ]
        return example

    raise ValueError(f"Example missing VAD labels: {example}")


def clamp_labels(example: Dict[str, Any]) -> Dict[str, Any]:
    labels = example["labels"]
    example["labels"] = [max(0.0, min(1.0, float(x))) for x in labels]
    return example


# ============================================================
# 4. Metrics
# ============================================================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    # Optional clamp because labels are [0, 1].
    predictions = np.clip(predictions, 0.0, 1.0)

    mse_all = mean_squared_error(labels, predictions)
    rmse_all = math.sqrt(mse_all)
    mae_all = mean_absolute_error(labels, predictions)

    metrics = {
        "mse": mse_all,
        "rmse": rmse_all,
        "mae": mae_all,
    }

    names = ["valence", "arousal", "dominance"]

    for i, name in enumerate(names):
        y_true = labels[:, i]
        y_pred = predictions[:, i]

        metrics[f"{name}_mse"] = mean_squared_error(y_true, y_pred)
        metrics[f"{name}_rmse"] = math.sqrt(metrics[f"{name}_mse"])
        metrics[f"{name}_mae"] = mean_absolute_error(y_true, y_pred)

        try:
            metrics[f"{name}_r2"] = r2_score(y_true, y_pred)
        except Exception:
            metrics[f"{name}_r2"] = float("nan")

    return metrics


class VADDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        loss_weights = [f.pop("loss_weight") for f in features]
        labels = [f.pop("labels") for f in features]

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(labels, dtype=torch.float)
        batch["loss_weight"] = torch.tensor(loss_weights, dtype=torch.float)

        return batch

# ============================================================
# 5. Main training function
# ============================================================

def train(args):
    set_seed(args.seed)

    ds = load_integrated_dataset(args.dataset_path)
    ds = ensure_splits(ds, seed=args.seed)

    print(ds)

    # Normalize labels.
    ds = ds.map(normalize_example_labels)
    ds = ds.map(clamp_labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True,)

    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

        tokenized["labels"] = [
            [float(x[0]), float(x[1]), float(x[2])]
            for x in batch["labels"]
        ]

        # Compute per-example reliability weights.
        weights = []
        n = len(batch["text"])

        for i in range(n):
            ex = {k: batch[k][i] for k in batch.keys()}
            weights.append(float(get_label_weight(ex)))

        tokenized["loss_weight"] = weights

        return tokenized

    # Remove all non-model columns after tokenization.
    original_columns = ds["train"].column_names

    tokenized_ds = ds.map(
        tokenize_batch,
        batched=True,
        remove_columns=original_columns,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        problem_type="regression",
        local_files_only=True,
    )

    model = model.float()

    print("Model dtype:", next(model.parameters()).dtype)

    bad_params = []
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            bad_params.append(name)

    print("Non-finite parameters:", bad_params[:10])

    # Useful label names for saved config.
    model.config.id2label = {
        0: "valence",
        1: "arousal",
        2: "dominance",
    }
    model.config.label2id = {
        "valence": 0,
        "arousal": 1,
        "dominance": 2,
    }

    data_collator = VADDataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,

        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,

        load_best_model_at_end=True,
        metric_for_best_model="eval_rmse",
        greater_is_better=False,

        save_total_limit=2,

        report_to=args.report_to,
        seed=args.seed,

        remove_unused_columns=False
    )

    trainer = WeightedMSETrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    print("\nEvaluating on validation set...")
    val_metrics = trainer.evaluate(tokenized_ds["validation"])
    print(val_metrics)

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(tokenized_ds["test"], metric_key_prefix="test")
    print(test_metrics)

    # Save final model.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_path = Path(args.output_dir) / "final_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "validation": val_metrics,
                "test": test_metrics,
            },
            f,
            indent=2,
        )

    print(f"\nSaved model to: {args.output_dir}")
    print(f"Saved metrics to: {metrics_path}")


# ============================================================
# 6. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="integrated_vad_utterance_dataset",
        help="Path to integrated dataset folder.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        help=(
            "Base encoder model. Good options: "
            "roberta-base, distilroberta-base, microsoft/deberta-v3-base"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="vad_roberta_regressor",
    )

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--logging_steps", type=int, default=50)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help='Use "tensorboard" if you installed tensorboard.',
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)