#!/usr/bin/env python3
"""
Train a custom JSONL-only DeBERTa-v3 appraisal encoder.

Expected JSONL format: one dialogue sample per line.
Each dialogue turn should contain:
    turn["listener_appraisal"] with all APPRAISAL_DIMS labels in [0, 1].

The model predicts the listener/target speaker's appraisal of the current utterance.
This means each training example is target-relative:
    current utterance by A -> listener_appraisal target B
    current utterance by B -> listener_appraisal target A

Install:
    pip install torch transformers scikit-learn numpy pandas sentencepiece protobuf tqdm

Example:
    python train_appraisal_encoder.py \
        --custom_jsonl EmoDynamic/appraisal_dataset.jsonl \
        --output_dir outputs/appraisal_deberta_v3 \
        --model_name microsoft/deberta-v3-base \
        --epochs 5 \
        --batch_size 8 \
        --eval_batch_size 16 \
        --max_length 512 \
        --fp16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
    set_seed,
)


APPRAISAL_DIMS = [
    "personal_relevance",
    "goal_conduciveness",
    "unexpectedness",
    "agency_self",
    "agency_other",
    "controllability",
    "norm_violation",
    "relationship_impact",
]

VAD_DIMS = ["v", "a", "d"]


@dataclass
class TurnExample:
    text: str
    labels: List[float]
    dialogue_id: int
    line_no: int
    turn_index: int
    speaker: str
    target: str
    raw_utterance: str


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def as_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def format_vad(vad: Optional[Dict[str, Any]]) -> str:
    if not isinstance(vad, dict):
        return ""
    parts = []
    for dim in VAD_DIMS:
        v = as_float_or_none(vad.get(dim))
        if v is not None:
            parts.append(f"{dim}={v:.2f}")
    return ", ".join(parts)


def flatten_persona(persona: Optional[Dict[str, Any]], speaker: str) -> str:
    if not isinstance(persona, dict):
        return f"Speaker {speaker}: no persona provided"

    role = persona.get("role", "")
    traits = persona.get("traits", {}) if isinstance(persona.get("traits"), dict) else {}
    big_five = traits.get("big_five", {}) if isinstance(traits.get("big_five"), dict) else {}
    style = traits.get("style", persona.get("style", ""))
    background = traits.get("background", persona.get("background", ""))
    initial_vad = persona.get("initial_vad", {}) if isinstance(persona.get("initial_vad"), dict) else {}

    pieces = [f"Speaker {speaker}"]
    if role:
        pieces.append(f"role={role}")
    if big_five:
        bf_text = ", ".join(
            f"{k}={float(v):.2f}"
            for k, v in big_five.items()
            if as_float_or_none(v) is not None
        )
        if bf_text:
            pieces.append(f"BigFive({bf_text})")
    if style:
        pieces.append(f"style={style}")
    if background:
        pieces.append(f"background={background}")
    vad_text = format_vad(initial_vad)
    if vad_text:
        pieces.append(f"initial_VAD({vad_text})")

    return "; ".join(pieces)


def build_turn_input(
    sample: Dict[str, Any],
    dialogue_id: int,
    turn_index: int,
    history_turns: int,
    include_persona: bool,
    include_vad: bool,
) -> str:
    scenario = sample.get("scenario", {}) if isinstance(sample.get("scenario"), dict) else {}
    personas = sample.get("personas", {}) if isinstance(sample.get("personas"), dict) else {}
    dialogue = sample.get("dialogue", []) if isinstance(sample.get("dialogue"), list) else []
    turn = dialogue[turn_index]

    speaker = str(turn.get("speaker", "?")).strip()
    utterance = str(turn.get("text", "")).strip()
    appraisal = turn.get("listener_appraisal", {}) if isinstance(turn.get("listener_appraisal"), dict) else {}
    target = str(appraisal.get("target", "?")).strip()

    lines: List[str] = []

    lines.append("Task: Predict the listener's cognitive appraisal of the current utterance.")
    lines.append("Output dimensions are continuous values from 0 to 1.")
    lines.append(f"Dialogue ID: {dialogue_id}")

    title = str(scenario.get("title", "")).strip()
    desc = str(scenario.get("description", "")).strip()
    if title or desc:
        lines.append("\n[SCENARIO CONTEXT]")
        lines.append("This section gives pivotal background information needed to interpret the dialogue.")
        if title:
            lines.append(f"Title: {title}")
        if desc:
            lines.append(f"Description: {desc}")

    if include_persona:
        lines.append("\n[PERSONAS]")
        if speaker in personas:
            lines.append(flatten_persona(personas.get(speaker), speaker))
        else:
            lines.append(f"Speaker {speaker}: persona not provided")
        if target in personas and target != speaker:
            lines.append(flatten_persona(personas.get(target), target))
        elif target != speaker:
            lines.append(f"Speaker {target}: persona not provided")

    start = max(0, turn_index - history_turns)
    history = dialogue[start:turn_index]
    if history:
        lines.append("\n[DIALOGUE HISTORY]")
        for h in history:
            h_speaker = str(h.get("speaker", "?")).strip()
            h_text = str(h.get("text", "")).strip()
            if not h_text:
                continue
            if include_vad:
                vad_text = format_vad(h.get("vad"))
                if vad_text:
                    lines.append(f"{h_speaker}: {h_text} [speaker_VAD: {vad_text}]")
                else:
                    lines.append(f"{h_speaker}: {h_text}")
            else:
                lines.append(f"{h_speaker}: {h_text}")

    lines.append("\n[CURRENT UTTERANCE]")
    lines.append(f"Speaker: {speaker}")
    lines.append(f"Listener / appraisal target: {target}")
    lines.append(f"Text: {utterance}")
    if include_vad:
        vad_text = format_vad(turn.get("vad"))
        if vad_text:
            lines.append(f"Speaker VAD: {vad_text}")

    lines.append("\n[APPRAISAL TARGET]")
    lines.append(f"Predict how Speaker {target} appraises Speaker {speaker}'s current utterance.")

    return "\n".join(lines)


def load_jsonl_dialogues(path: str | Path) -> List[Tuple[int, Dict[str, Any]]]:
    path = Path(path)
    samples: List[Tuple[int, Dict[str, Any]]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON error at line {line_no}: {e}")
                continue

            if not isinstance(sample.get("dialogue"), list) or not sample["dialogue"]:
                print(f"[WARN] Skipping line {line_no}: missing/non-empty dialogue list")
                continue

            samples.append((line_no, sample))

    return samples


def extract_turn_examples(
    dialogue_items: Sequence[Tuple[int, Dict[str, Any]]],
    history_turns: int,
    include_persona: bool,
    include_vad: bool,
    strict_labels: bool,
) -> List[TurnExample]:
    examples: List[TurnExample] = []
    skipped_missing = 0
    skipped_bad = 0

    for dialogue_id, (line_no, sample) in enumerate(dialogue_items):
        dialogue = sample.get("dialogue", [])
        for turn_index, turn in enumerate(dialogue):
            if not isinstance(turn, dict):
                skipped_bad += 1
                continue

            app = turn.get("listener_appraisal")
            if not isinstance(app, dict):
                skipped_missing += 1
                continue

            labels: List[float] = []
            missing_dims = []
            bad_dims = []
            for dim in APPRAISAL_DIMS:
                v = as_float_or_none(app.get(dim))
                if v is None:
                    missing_dims.append(dim)
                    labels.append(0.0)
                else:
                    if v < 0.0 or v > 1.0:
                        bad_dims.append((dim, v))
                    labels.append(clamp01(v))

            if missing_dims and strict_labels:
                skipped_missing += 1
                print(f"[WARN] Line {line_no}, turn {turn_index}: missing labels {missing_dims}; skipped")
                continue
            if bad_dims:
                print(f"[WARN] Line {line_no}, turn {turn_index}: labels outside [0,1] were clamped: {bad_dims}")

            text = build_turn_input(
                sample=sample,
                dialogue_id=dialogue_id,
                turn_index=turn_index,
                history_turns=history_turns,
                include_persona=include_persona,
                include_vad=include_vad,
            )

            speaker = str(turn.get("speaker", "?")).strip()
            target = str(app.get("target", "?")).strip()
            raw_utterance = str(turn.get("text", "")).strip()

            examples.append(
                TurnExample(
                    text=text,
                    labels=labels,
                    dialogue_id=dialogue_id,
                    line_no=line_no,
                    turn_index=turn_index,
                    speaker=speaker,
                    target=target,
                    raw_utterance=raw_utterance,
                )
            )

    print(f"[INFO] Built {len(examples)} turn-level examples")
    if skipped_missing:
        print(f"[INFO] Skipped {skipped_missing} turns with missing appraisal labels")
    if skipped_bad:
        print(f"[INFO] Skipped {skipped_bad} malformed turns")

    return examples


def split_dialogues(
    samples: List[Tuple[int, Dict[str, Any]]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Tuple[int, Dict[str, Any]]], List[Tuple[int, Dict[str, Any]]]]:
    rng = random.Random(seed)
    samples = samples[:]
    rng.shuffle(samples)

    n = len(samples)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    test = samples[:n_test]
    val = samples[n_test:n_test + n_val]
    train = samples[n_test + n_val:]

    return train, val, test


class AppraisalDataset(Dataset):
    def __init__(self, examples: Sequence[TurnExample], tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc["labels"] = torch.tensor(ex.labels, dtype=torch.float32)
        return enc


def make_collate_fn(tokenizer: PreTrainedTokenizerBase):
    base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = torch.stack([x.pop("labels") for x in batch])
        out = base_collator(batch)
        out["labels"] = labels
        return out

    return collate


def optimizer_grouped_parameters(model: torch.nn.Module, weight_decay: float):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    return [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


def get_amp_context(device: torch.device, fp16: bool, bf16: bool):
    enabled = device.type == "cuda" and (fp16 or bf16)
    dtype = torch.float16 if fp16 else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def predict_from_logits(logits: torch.Tensor, sigmoid_outputs: bool) -> torch.Tensor:
    if sigmoid_outputs:
        return torch.sigmoid(logits)
    return logits


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    optimizer.zero_grad(set_to_none=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.fp16))
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)

    for step, batch in enumerate(progress, start=1):
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        with get_amp_context(device, args.fp16, args.bf16):
            outputs = model(**batch)
            preds = predict_from_logits(outputs.logits, args.sigmoid_outputs)
            loss = F.mse_loss(preds, labels)
            loss = loss / args.gradient_accumulation_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = step % args.gradient_accumulation_steps == 0 or step == len(loader)
        if should_step:
            if args.grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        total_loss += loss.item() * args.gradient_accumulation_steps * batch_size
        total_count += batch_size
        progress.set_postfix(loss=total_loss / max(1, total_count))

    return {"train_mse_loss": total_loss / max(1, total_count)}


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    y_pred = np.clip(y_pred, 0.0, 1.0)

    metrics[f"{prefix}mse"] = float(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))
    metrics[f"{prefix}rmse"] = float(math.sqrt(metrics[f"{prefix}mse"]))
    metrics[f"{prefix}mae"] = float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)))

    for j, dim in enumerate(APPRAISAL_DIMS):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        metrics[f"{prefix}{dim}_mse"] = float(mean_squared_error(yt, yp))
        metrics[f"{prefix}{dim}_rmse"] = float(math.sqrt(metrics[f"{prefix}{dim}_mse"]))
        metrics[f"{prefix}{dim}_mae"] = float(mean_absolute_error(yt, yp))
        if len(yt) >= 2 and np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            metrics[f"{prefix}{dim}_r"] = float(np.corrcoef(yt, yp)[0, 1])
        else:
            metrics[f"{prefix}{dim}_r"] = float("nan")
        try:
            metrics[f"{prefix}{dim}_r2"] = float(r2_score(yt, yp))
        except Exception:
            metrics[f"{prefix}{dim}_r2"] = float("nan")

    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    examples: Sequence[TurnExample],
    device: torch.device,
    args: argparse.Namespace,
    split_name: str,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, float]:
    model.eval()
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(loader, desc=f"eval {split_name}", leave=False):
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        with get_amp_context(device, args.fp16, args.bf16):
            outputs = model(**batch)
            preds = predict_from_logits(outputs.logits, args.sigmoid_outputs)
            loss = F.mse_loss(preds, labels)

        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)
        all_preds.append(preds.detach().float().cpu().numpy())
        all_labels.append(labels.detach().float().cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    metrics = compute_regression_metrics(y_true, y_pred)
    metrics[f"{split_name}_loss"] = total_loss / max(1, total_count)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pred_path = output_dir / f"{split_name}_predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for ex, yt, yp in zip(examples, y_true, np.clip(y_pred, 0.0, 1.0)):
                row = {
                    "line_no": ex.line_no,
                    "dialogue_id": ex.dialogue_id,
                    "turn_index": ex.turn_index,
                    "speaker": ex.speaker,
                    "target": ex.target,
                    "utterance": ex.raw_utterance,
                    "gold": {dim: float(yt[i]) for i, dim in enumerate(APPRAISAL_DIMS)},
                    "pred": {dim: float(yp[i]) for i, dim in enumerate(APPRAISAL_DIMS)},
                    "abs_error": {dim: float(abs(yp[i] - yt[i])) for i, dim in enumerate(APPRAISAL_DIMS)},
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Target-wise and speaker-wise diagnostics.
        diag_rows = []
        for group_name, values in [("target", [e.target for e in examples]), ("speaker", [e.speaker for e in examples])]:
            for value in sorted(set(values)):
                idx = np.array([v == value for v in values])
                if idx.sum() == 0:
                    continue
                m = compute_regression_metrics(y_true[idx], y_pred[idx], prefix="")
                diag_rows.append({"group": group_name, "value": value, "n": int(idx.sum()), **m})
        pd.DataFrame(diag_rows).to_csv(output_dir / f"{split_name}_group_metrics.csv", index=False)

    return metrics


def save_model(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
    args: argparse.Namespace,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "appraisal_dims": APPRAISAL_DIMS,
        "model_name": args.model_name,
        "sigmoid_outputs": args.sigmoid_outputs,
        "max_length": args.max_length,
        "history_turns": args.history_turns,
        "include_persona": not args.no_persona,
        "include_vad": not args.no_vad,
        "metrics": metrics or {},
    }
    with (output_dir / "appraisal_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with (output_dir / "appraisal_dims.json").open("w", encoding="utf-8") as f:
        json.dump(APPRAISAL_DIMS, f, indent=2, ensure_ascii=False)


def json_safe(obj: Any) -> Any:
    """Convert numpy/pandas scalar types into normal Python JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def save_manifest(
    args: argparse.Namespace,
    train_examples: Sequence[TurnExample],
    val_examples: Sequence[TurnExample],
    test_examples: Sequence[TurnExample],
) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def value_counts_as_plain_dict(values: Sequence[str]) -> Dict[str, int]:
        if not values:
            return {}
        counts = pd.Series(list(values)).value_counts()
        return {str(k): int(v) for k, v in counts.items()}

    def summarize(examples: Sequence[TurnExample]) -> Dict[str, Any]:
        return {
            "turn_examples": int(len(examples)),
            "unique_dialogues": int(len(set(e.dialogue_id for e in examples))),
            "speaker_counts": value_counts_as_plain_dict([e.speaker for e in examples]),
            "target_counts": value_counts_as_plain_dict([e.target for e in examples]),
        }

    manifest = {
        "appraisal_dims": APPRAISAL_DIMS,
        "args": json_safe(vars(args)),
        "train": summarize(train_examples),
        "val": summarize(val_examples),
        "test": summarize(test_examples),
    }
    with (out / "training_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(manifest), f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--custom_jsonl", type=str, required=True, help="Path to your custom appraisal-dialogue JSONL file.")
    p.add_argument("--output_dir", type=str, default="outputs/appraisal_deberta_v3")
    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--local_files_only", action="store_true", help="Use only locally cached model/tokenizer files.")

    p.add_argument("--history_turns", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--no_persona", action="store_true")
    p.add_argument("--no_vad", action="store_true")
    p.add_argument("--non_strict_labels", action="store_true", help="Do not skip turns with missing appraisal dimensions; missing values become 0.")

    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--test_ratio", type=float, default=0.00)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--sigmoid_outputs", action="store_true", default=True, help="Apply sigmoid to model logits before MSE loss. Recommended for [0,1] labels.")
    p.add_argument("--no_sigmoid_outputs", dest="sigmoid_outputs", action="store_false")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and (args.fp16 or args.bf16):
        print("[WARN] fp16/bf16 requested but CUDA is unavailable. Mixed precision will be disabled.")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading JSONL: {args.custom_jsonl}")

    dialogues = load_jsonl_dialogues(args.custom_jsonl)
    if not dialogues:
        raise RuntimeError("No valid dialogue samples found.")

    train_dialogues, val_dialogues, test_dialogues = split_dialogues(
        dialogues,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(
        f"[INFO] Dialogue split: train={len(train_dialogues)}, "
        f"val={len(val_dialogues)}, test={len(test_dialogues)}"
    )

    common_extract_kwargs = dict(
        history_turns=args.history_turns,
        include_persona=not args.no_persona,
        include_vad=not args.no_vad,
        strict_labels=not args.non_strict_labels,
    )
    train_examples = extract_turn_examples(train_dialogues, **common_extract_kwargs)
    val_examples = extract_turn_examples(val_dialogues, **common_extract_kwargs)
    test_examples = extract_turn_examples(test_dialogues, **common_extract_kwargs)

    if not train_examples:
        raise RuntimeError("No train examples created. Check listener_appraisal labels and JSONL format.")
    if not val_examples:
        print("[WARN] No validation examples. Increase dataset size or val_ratio.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, local_files_only=args.local_files_only)

    id2label = {i: dim for i, dim in enumerate(APPRAISAL_DIMS)}
    label2id = {dim: i for i, dim in enumerate(APPRAISAL_DIMS)}
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(APPRAISAL_DIMS),
        problem_type="regression",
        id2label=id2label,
        label2id=label2id,
        local_files_only=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=args.local_files_only,
        use_safetensors=False,  # prevents Transformers from starting a background Hub safetensors-conversion request
    )
    model.to(device)

    train_ds = AppraisalDataset(train_examples, tokenizer, args.max_length)
    val_ds = AppraisalDataset(val_examples, tokenizer, args.max_length) if val_examples else None
    test_ds = AppraisalDataset(test_examples, tokenizer, args.max_length) if test_examples else None

    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    ) if val_ds is not None else None
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    ) if test_ds is not None else None

    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = AdamW(optimizer_grouped_parameters(model, args.weight_decay), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"[INFO] Train turn examples: {len(train_examples)}")
    print(f"[INFO] Val turn examples:   {len(val_examples)}")
    print(f"[INFO] Test turn examples:  {len(test_examples)}")
    print(f"[INFO] Total optimizer steps: {total_steps}, warmup steps: {warmup_steps}")

    save_manifest(args, train_examples, val_examples, test_examples)

    best_val_mae = float("inf")
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            args=args,
        )

        print(f"\n[Epoch {epoch}] train_mse_loss={train_metrics['train_mse_loss']:.6f}")

        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                examples=val_examples,
                device=device,
                args=args,
                split_name="val",
                output_dir=args.output_dir,
            )
            print(
                f"[Epoch {epoch}] val_mae={val_metrics['mae']:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} val_mse={val_metrics['mse']:.6f}"
            )

            metrics_path = Path(args.output_dir) / f"val_metrics_epoch_{epoch}.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2, ensure_ascii=False)

            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                best_metrics = val_metrics
                save_model(model, tokenizer, Path(args.output_dir) / "best", args, best_metrics)
                print(f"[INFO] Saved new best checkpoint to {Path(args.output_dir) / 'best'}")
        else:
            save_model(model, tokenizer, Path(args.output_dir) / "last", args, train_metrics)

    save_model(model, tokenizer, Path(args.output_dir) / "last", args, best_metrics)
    print(f"[INFO] Saved last checkpoint to {Path(args.output_dir) / 'last'}")

    if test_loader is not None:
        print("[INFO] Evaluating final model on test set")
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            examples=test_examples,
            device=device,
            args=args,
            split_name="test",
            output_dir=args.output_dir,
        )
        with (Path(args.output_dir) / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2, ensure_ascii=False)
        print(
            f"[TEST] mae={test_metrics['mae']:.6f} "
            f"rmse={test_metrics['rmse']:.6f} mse={test_metrics['mse']:.6f}"
        )

    print("[DONE]")
    print(f"Best checkpoint: {Path(args.output_dir) / 'best' if val_examples else Path(args.output_dir) / 'last'}")


if __name__ == "__main__":
    main()
