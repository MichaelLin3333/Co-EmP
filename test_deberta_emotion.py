# test_deberta_emotion.py
"""
Evaluate a DeBERTa checkpoint on:
- GoEmotions         (multi-label)
- DailyDialog        (single-label)
- EmpatheticDialogues (single-label; using `context` as label)

Examples:
  python test_deberta_emotion.py \
      --model_name_or_path /path/to/goemotions_deberta \
      --dataset goemotions \
      --split test

  python test_deberta_emotion.py \
      --model_name_or_path /path/to/dailydialog_deberta \
      --dataset dailydialog \
      --split test \
      --history_turns 2

  python test_deberta_emotion.py \
      --model_name_or_path /path/to/empathetic_deberta \
      --dataset empatheticdialogues \
      --split test \
      --ed_text_mode prompt_utterance
"""

import argparse
import json
import math
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DAILYDIALOG_LABELS = [
    "no emotion",
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
]


def normalize_label(x: str) -> str:
    return x.lower().strip().replace("_", " ").replace("-", " ")


def get_model_label_names(model) -> List[str] | None:
    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        return None

    out = []
    for i in range(len(id2label)):
        if i in id2label:
            out.append(str(id2label[i]))
        elif str(i) in id2label:
            out.append(str(id2label[str(i)]))
        else:
            return None
    return out


def maybe_align_dataset_labels_to_model(
    y: np.ndarray, dataset_label_names: List[str], model
) -> Tuple[np.ndarray, List[str]]:
    """
    Reorder or remap dataset labels to model.config.id2label order when possible.
    This is very useful if your fine-tuned checkpoint stores labels in a different order.
    """
    model_label_names = get_model_label_names(model)
    if model_label_names is None:
        print("[Info] model.config.id2label not found; using dataset label order as-is.")
        return y, dataset_label_names

    if len(model_label_names) != len(dataset_label_names):
        print(
            f"[Warning] Label count mismatch: model has {len(model_label_names)}, "
            f"dataset has {len(dataset_label_names)}. No alignment applied."
        )
        return y, dataset_label_names

    dataset_norm_to_idx = {normalize_label(n): i for i, n in enumerate(dataset_label_names)}

    if not all(normalize_label(n) in dataset_norm_to_idx for n in model_label_names):
        print(
            "[Warning] Model label names do not cleanly match dataset label names. "
            "No alignment applied."
        )
        return y, dataset_label_names

    # Multi-label: reorder columns
    if y.ndim == 2:
        reorder = [dataset_norm_to_idx[normalize_label(n)] for n in model_label_names]
        y = y[:, reorder]
        print("[Info] Reordered multi-label ground truth to match model.config.id2label.")
        return y, model_label_names

    # Single-label: remap class ids
    dataset_idx_to_model_idx = {
        dataset_norm_to_idx[normalize_label(model_name)]: i
        for i, model_name in enumerate(model_label_names)
    }
    y = np.array([dataset_idx_to_model_idx[int(label)] for label in y], dtype=np.int64)
    print("[Info] Remapped single-label targets to match model.config.id2label.")
    return y, model_label_names


def load_goemotions(split: str) -> Tuple[List[str], np.ndarray, List[str], str]:
    """
    Uses SetFit/go_emotions, which is the simplified classification-ready port.
    Label columns are inferred from all columns except known metadata.
    """
    ds = load_dataset("SetFit/go_emotions")

    if split not in ds:
        raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")

    split_ds = ds[split]
    exclude = {"text", "id"}
    label_cols = [c for c in split_ds.column_names if c not in exclude]

    texts = split_ds["text"]
    labels = np.stack([np.array(split_ds[c], dtype=np.float32) for c in label_cols], axis=1)
    return texts, labels, label_cols, "multi_label"


def load_dailydialog(split: str, history_turns: int = 0) -> Tuple[List[str], np.ndarray, List[str], str]:
    """
    Flattens each dialogue into utterance-level samples.
    Optionally prepends the previous N utterances as context.
    """
    ds = load_dataset("Akhil391/daily_dialog")

    if split not in ds:
        raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")

    texts = []
    labels = []

    for ex in ds[split]:
        dialog = ex["dialog"]
        emotions = ex["emotion"]

        for i, (utt, emo) in enumerate(zip(dialog, emotions)):
            if history_turns > 0:
                start = max(0, i - history_turns)
                history = dialog[start:i]
                if history:
                    text = " [SEP] ".join(history) + " [UTT] " + utt
                else:
                    text = utt
            else:
                text = utt

            texts.append(text)
            labels.append(int(emo))

    return texts, np.array(labels, dtype=np.int64), DAILYDIALOG_LABELS, "single_label"


def load_empatheticdialogues(
    split: str,
    text_mode: str = "prompt"
) -> Tuple[List[str], np.ndarray, List[str], str]:
    """
    Uses `context` as the label.
    text_mode:
      - prompt
      - utterance
      - prompt_utterance
    """
    ds = load_dataset("facebook/empathetic_dialogues")

    if split not in ds:
        raise ValueError(f"Split '{split}' not found. Available: {list(ds.keys())}")

    # Build label vocabulary from train split for stable mapping
    train_contexts = ds["train"]["context"]
    label_names = sorted(set(train_contexts))
    label2id = {name: i for i, name in enumerate(label_names)}

    texts = []
    labels = []

    for ex in ds[split]:
        prompt = ex["prompt"]
        utterance = ex["utterance"]
        context = ex["context"]

        if text_mode == "prompt":
            text = prompt
        elif text_mode == "utterance":
            text = utterance
        elif text_mode == "prompt_utterance":
            text = f"{prompt} [SEP] {utterance}"
        else:
            raise ValueError(f"Unknown text_mode: {text_mode}")

        texts.append(text)
        labels.append(label2id[context])

    return texts, np.array(labels, dtype=np.int64), label_names, "single_label"


@torch.no_grad()
def run_eval(
    model,
    tokenizer,
    texts: List[str],
    y_true: np.ndarray,
    task_type: str,
    batch_size: int,
    max_length: int,
    threshold: float,
    device: torch.device,
):
    model.eval()
    model.to(device)

    all_logits = []
    total_loss = 0.0
    total_examples = 0

    for start in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
        batch_texts = texts[start:start + batch_size]
        batch_y = y_true[start:start + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(**enc).logits

        if task_type == "multi_label":
            labels_t = torch.tensor(batch_y, dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(logits, labels_t)
        else:
            labels_t = torch.tensor(batch_y, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels_t)

        total_loss += loss.item() * len(batch_texts)
        total_examples += len(batch_texts)
        all_logits.append(logits.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    avg_loss = total_loss / max(total_examples, 1)

    if task_type == "multi_label":
        probs = 1.0 / (1.0 + np.exp(-logits))
        y_pred = (probs >= threshold).astype(np.int32)

        metrics = {
            "eval_loss": avg_loss,
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
            "exact_match": float((y_true == y_pred).all(axis=1).mean()),
        }
    else:
        y_pred = logits.argmax(axis=-1)

        metrics = {
            "eval_loss": avg_loss,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    return metrics, y_pred, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["goemotions", "dailydialog", "empatheticdialogues"],
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--history_turns", type=int, default=0)
    parser.add_argument(
        "--ed_text_mode",
        type=str,
        default="prompt",
        choices=["prompt", "utterance", "prompt_utterance"],
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_predictions", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)

    if args.dataset == "goemotions":
        texts, y_true, label_names, task_type = load_goemotions(args.split)
    elif args.dataset == "dailydialog":
        texts, y_true, label_names, task_type = load_dailydialog(
            args.split,
            history_turns=args.history_turns,
        )
    elif args.dataset == "empatheticdialogues":
        texts, y_true, label_names, task_type = load_empatheticdialogues(
            args.split,
            text_mode=args.ed_text_mode,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.max_samples is not None:
        texts = texts[:args.max_samples]
        y_true = y_true[:args.max_samples]

    print(f"[Info] Dataset: {args.dataset}")
    print(f"[Info] Split: {args.split}")
    print(f"[Info] Number of examples: {len(texts)}")
    print(f"[Info] Task type: {task_type}")
    print(f"[Info] Number of labels: {len(label_names)}")

    # Try to align dataset labels to model label order
    y_true, label_names = maybe_align_dataset_labels_to_model(y_true, label_names, model)

    # Basic sanity checks
    model_num_labels = model.config.num_labels
    if task_type == "multi_label":
        if y_true.shape[1] != model_num_labels:
            raise ValueError(
                f"Model num_labels={model_num_labels}, but dataset has {y_true.shape[1]} labels. "
                "Your checkpoint head does not match this dataset."
            )
    else:
        if len(label_names) != model_num_labels:
            raise ValueError(
                f"Model num_labels={model_num_labels}, but dataset has {len(label_names)} labels. "
                "Your checkpoint head does not match this dataset."
            )

    metrics, y_pred, logits = run_eval(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        y_true=y_true,
        task_type=task_type,
        batch_size=args.batch_size,
        max_length=args.max_length,
        threshold=args.threshold,
        device=device,
    )

    print("\n===== Metrics =====")
    print(json.dumps(metrics, indent=2))

    print("\n===== Classification Report =====")
    if task_type == "multi_label":
        report = classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
    else:
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(label_names))),
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
    print(report)

    if args.save_predictions:
        os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)
        rows = []

        if task_type == "multi_label":
            probs = 1.0 / (1.0 + np.exp(-logits))
            for text, gold, pred, prob in zip(texts, y_true, y_pred, probs):
                gold_labels = [label_names[i] for i, v in enumerate(gold) if int(v) == 1]
                pred_labels = [label_names[i] for i, v in enumerate(pred) if int(v) == 1]
                top_scores = sorted(
                    [(label_names[i], float(prob[i])) for i in range(len(label_names))],
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
                rows.append(
                    {
                        "text": text,
                        "gold_labels": gold_labels,
                        "pred_labels": pred_labels,
                        "top5_scores": top_scores,
                    }
                )
        else:
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            for text, gold, pred, prob in zip(texts, y_true, y_pred, probs):
                rows.append(
                    {
                        "text": text,
                        "gold_label": label_names[int(gold)],
                        "pred_label": label_names[int(pred)],
                        "confidence": float(prob[int(pred)]),
                    }
                )

        with open(args.save_predictions, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        print(f"\n[Info] Saved predictions to: {args.save_predictions}")


if __name__ == "__main__":
    main()