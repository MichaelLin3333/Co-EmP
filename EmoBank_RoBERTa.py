# train_roberta_emobank_vad.py
# Two-stage training on EmoBank:
#   Stage 1: continued pretraining (MLM / TAPT-style) on EmoBank text
#   Stage 2: supervised fine-tuning for VAD regression
#
# Usage:
#   python train_roberta_emobank_vad.py --emobank_csv /path/to/EmoBank/corpus/emobank.csv
#
# If --emobank_csv is omitted, the script falls back to the raw GitHub file.

import os
import argparse
import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from scipy.stats import pearsonr

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

DEFAULT_EMOBANK_RAW_URL = (
    "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emobank_csv", type=str, default="EmoBank/corpus/emobank.csv",
                        help="Local path to EmoBank/corpus/emobank.csv. If omitted, uses GitHub raw URL.")
    parser.add_argument("--base_model", type=str, default="FacebookAI/roberta-base")
    parser.add_argument("--output_root", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=128)

    # Stage 1: MLM continued pretraining
    parser.add_argument("--mlm_epochs", type=int, default=5)
    parser.add_argument("--mlm_lr", type=float, default=5e-5)
    parser.add_argument("--mlm_batch_size", type=int, default=16)
    parser.add_argument("--mlm_weight_decay", type=float, default=0.01)
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    # Stage 2: supervised VAD regression
    parser.add_argument("--sft_epochs", type=int, default=10)
    parser.add_argument("--sft_lr", type=float, default=2e-5)
    parser.add_argument("--sft_batch_size", type=int, default=16)
    parser.add_argument("--sft_eval_batch_size", type=int, default=32)
    parser.add_argument("--sft_weight_decay", type=float, default=0.01)

    return parser.parse_args()


def load_emobank_dataframe(emobank_csv: str | None) -> pd.DataFrame:
    source = emobank_csv if emobank_csv else DEFAULT_EMOBANK_RAW_URL
    df = pd.read_csv(source)

    required_cols = {"split", "V", "A", "D", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Clean up
    df = df[["split", "V", "A", "D", "text"]].copy()
    df["text"] = df["text"].astype(str).fillna("")
    for c in ["V", "A", "D"]:
        df[c] = df[c].astype(np.float32)

    valid_splits = {"train", "dev", "test"}
    found_splits = set(df["split"].unique())
    if not valid_splits.issubset(found_splits):
        raise ValueError(f"Expected splits {valid_splits}, found {found_splits}")

    return df


def to_hf_dataset(df_split: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df_split.reset_index(drop=True), preserve_index=False)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(pearsonr(x, y)[0])


def compute_vad_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.asarray(preds, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    v_r = safe_pearson(preds[:, 0], labels[:, 0])
    a_r = safe_pearson(preds[:, 1], labels[:, 1])
    d_r = safe_pearson(preds[:, 2], labels[:, 2])
    mean_pearson = float(np.mean([v_r, a_r, d_r]))

    mae = float(np.mean(np.abs(preds - labels)))
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))

    return {
        "v_pearson": v_r,
        "a_pearson": a_r,
        "d_pearson": d_r,
        "mean_pearson": mean_pearson,
        "mae": mae,
        "rmse": rmse,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_root, exist_ok=True)
    tapt_dir = os.path.join(args.output_root, "roberta-emobank-tapt")
    vad_dir = os.path.join(args.output_root, "roberta-emobank-vad")

    print("Loading EmoBank...")
    df = load_emobank_dataframe(args.emobank_csv)

    train_df = df[df["split"] == "train"].copy()
    dev_df = df[df["split"] == "dev"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    # ------------------------------------------------------------
    # Stage 1: continued pretraining / TAPT-style MLM
    # ------------------------------------------------------------
    # Strict setup: only use TRAIN text here to avoid leaking dev/test sentences.
    mlm_train_ds = Dataset.from_dict({"text": train_df["text"].tolist()})

    def tokenize_mlm(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    mlm_train_tok = mlm_train_ds.map(
        tokenize_mlm,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing MLM corpus",
    )

    mlm_model = AutoModelForMaskedLM.from_pretrained(args.base_model)

    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    mlm_args = TrainingArguments(
        output_dir=tapt_dir,
        per_device_train_batch_size=args.mlm_batch_size,
        learning_rate=args.mlm_lr,
        weight_decay=args.mlm_weight_decay,
        num_train_epochs=args.mlm_epochs,
        warmup_ratio=0.1,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=25,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    mlm_trainer = Trainer(
        model=mlm_model,
        args=mlm_args,
        train_dataset=mlm_train_tok,
        data_collator=mlm_collator,
    )

    print("\n=== Stage 1: continued pretraining (MLM) ===")
    mlm_trainer.train()
    mlm_trainer.save_model(tapt_dir)
    tokenizer.save_pretrained(tapt_dir)

    # ------------------------------------------------------------
    # Stage 2: supervised fine-tuning for VAD regression
    # ------------------------------------------------------------
    train_hf = to_hf_dataset(train_df[["text", "V", "A", "D"]])
    dev_hf = to_hf_dataset(dev_df[["text", "V", "A", "D"]])
    test_hf = to_hf_dataset(test_df[["text", "V", "A", "D"]])

    vad_ds = DatasetDict({
        "train": train_hf,
        "dev": dev_hf,
        "test": test_hf,
    })

    reg_tokenizer = AutoTokenizer.from_pretrained(tapt_dir, use_fast=True)

    def tokenize_regression(batch):
        enc = reg_tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        labels = np.stack([batch["V"], batch["A"], batch["D"]], axis=1).astype(np.float32)
        enc["labels"] = labels.tolist()
        return enc

    vad_tok = vad_ds.map(
        tokenize_regression,
        batched=True,
        remove_columns=["text", "V", "A", "D"],
        desc="Tokenizing VAD data",
    )

    # Important: explicitly force regression with 3 outputs.
    config = AutoConfig.from_pretrained(
        tapt_dir,
        num_labels=3,
        problem_type="regression",
        id2label={0: "V", 1: "A", 2: "D"},
        label2id={"V": 0, "A": 1, "D": 2},
    )

    vad_model = AutoModelForSequenceClassification.from_pretrained(
        tapt_dir,
        config=config,
    )

    sft_args = TrainingArguments(
        output_dir=vad_dir,
        per_device_train_batch_size=args.sft_batch_size,
        per_device_eval_batch_size=args.sft_eval_batch_size,
        learning_rate=args.sft_lr,
        weight_decay=args.sft_weight_decay,
        num_train_epochs=args.sft_epochs,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mean_pearson",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=25,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    vad_trainer = Trainer(
        model=vad_model,
        args=sft_args,
        train_dataset=vad_tok["train"],
        eval_dataset=vad_tok["dev"],
        data_collator=default_data_collator,
        compute_metrics=compute_vad_metrics,
    )

    print("\n=== Stage 2: supervised fine-tuning (VAD regression) ===")
    vad_trainer.train()

    print("\n=== Dev evaluation (best checkpoint) ===")
    dev_metrics = vad_trainer.evaluate(eval_dataset=vad_tok["dev"])
    for k, v in dev_metrics.items():
        print(f"{k}: {v}")

    print("\n=== Test evaluation (best checkpoint) ===")
    test_metrics = vad_trainer.evaluate(eval_dataset=vad_tok["test"])
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    vad_trainer.save_model(vad_dir)
    reg_tokenizer.save_pretrained(vad_dir)

    print(f"\nSaved TAPT model to: {tapt_dir}")
    print(f"Saved VAD regression model to: {vad_dir}")


if __name__ == "__main__":
    main()