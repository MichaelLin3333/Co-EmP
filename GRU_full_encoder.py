import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


BIG5_KEYS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

VAD_KEYS = ["v", "a", "d"]


# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_vad_values(values: Sequence[float], mode: str) -> List[float]:
    """
    Normalizes VAD values according to the selected mode.

    Supported modes:
        none:
            Keep values exactly as they appear in the JSONL.

        neg1_1_to_0_1:
            Convert VAD from [-1, 1] to [0, 1].
            x_norm = (x + 1) / 2

        one5_to_0_1:
            Convert EmoBank-style VAD from [1, 5] to [0, 1].
            x_norm = (x - 1) / 4
    """
    xs = [float(x) for x in values]

    if mode == "none":
        return xs
    if mode == "neg1_1_to_0_1":
        return [(x + 1.0) / 2.0 for x in xs]
    if mode == "one5_to_0_1":
        return [(x - 1.0) / 4.0 for x in xs]

    raise ValueError(f"Unknown VAD normalization mode: {mode}")


def parse_vad(vad_obj: Dict[str, Any], vad_norm: str = "none") -> List[float]:
    values = [float(vad_obj[k]) for k in VAD_KEYS]
    return normalize_vad_values(values, vad_norm)


def parse_big5(persona_obj: Dict[str, Any]) -> List[float]:
    big5 = persona_obj["traits"]["big_five"]
    return [float(big5[k]) for k in BIG5_KEYS]


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_id}: {e}") from e
    return records


def split_dialogues(
    dialogues: Sequence[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    indices = list(range(len(dialogues)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = int(round(len(indices) * val_ratio))
    val_ids = set(indices[:n_val])

    train_dialogues = [d for i, d in enumerate(dialogues) if i not in val_ids]
    val_dialogues = [d for i, d in enumerate(dialogues) if i in val_ids]

    return train_dialogues, val_dialogues


# ============================================================
# Build speaker-specific GRU sequences from dialogue JSON
# ============================================================


def build_speaker_sequences(
    dialogues: Sequence[Dict[str, Any]],
    include_first_turn: bool = True,
    min_steps: int = 1,
    chunk_len: Optional[int] = None,
    vad_norm: str = "none",
) -> List[Dict[str, Any]]:
    """
    Converts dialogue-level JSON records into speaker-specific transition sequences.

    For an ABAB dialogue:
        A1, B1, A2, B2, A3

    A's sequence becomes:
        step 1: self_prev=A_initial or A1, other=B_initial or B1, target=A1 or A2
        step 2: self_prev=A1, other=B1, target=A2
        step 3: self_prev=A2, other=B2, target=A3

    With include_first_turn=True:
        The first utterance of each speaker is trained using that speaker's
        persona initial_vad as self_prev and the latest known other VAD.

    With include_first_turn=False:
        The first utterance of each speaker is used only to update last_vad,
        not as a supervised transition.

    chunk_len:
        None = preserve full speaker-specific sequences.
        1 = one-turn cropped transition training.
        K = truncated sequence chunks of length K.
    """

    all_sequences: List[Dict[str, Any]] = []

    for dialogue_id, dialog in enumerate(dialogues):
        personas = dialog.get("personas", {})
        utterances = dialog.get("dialogue", [])

        if not personas or not utterances:
            continue

        speakers = list(personas.keys())
        if len(speakers) < 2:
            continue

        personality: Dict[str, List[float]] = {}
        initial_vad: Dict[str, List[float]] = {}

        for spk in speakers:
            try:
                personality[spk] = parse_big5(personas[spk])
                initial_vad[spk] = parse_vad(personas[spk]["initial_vad"], vad_norm=vad_norm)
            except KeyError as e:
                raise KeyError(
                    f"Missing persona field for speaker {spk} in dialogue {dialogue_id}: {e} {dialog}"
                ) from e

        last_vad: Dict[str, List[float]] = {
            spk: list(initial_vad[spk]) for spk in speakers
        }
        has_spoken: Dict[str, bool] = {spk: False for spk in speakers}

        steps_by_speaker: Dict[str, List[Dict[str, List[float]]]] = {
            spk: [] for spk in speakers
        }

        for utt_id, utt in enumerate(utterances):
            spk = utt.get("speaker")
            if spk not in speakers:
                continue

            target_vad = parse_vad(utt["vad"], vad_norm=vad_norm)

            # For two-speaker data, this selects the other speaker.
            # If you later extend to multi-party dialogue, replace this with
            # an attention/pooling representation over all other speakers.
            other_candidates = [s for s in speakers if s != spk]
            other_spk = other_candidates[0]

            self_prev = list(last_vad[spk])
            other_now = list(last_vad[other_spk])

            use_as_training_step = include_first_turn or has_spoken[spk]

            if use_as_training_step:
                steps_by_speaker[spk].append(
                    {
                        "v_self_prev": self_prev,
                        "v_other": other_now,
                        "v_target": target_vad,
                    }
                )

            last_vad[spk] = target_vad
            has_spoken[spk] = True

        title = dialog.get("scenario", {}).get("title", f"dialogue_{dialogue_id}")

        for spk in speakers:
            steps = steps_by_speaker[spk]
            if len(steps) < min_steps:
                continue

            base = {
                "dialogue_id": dialogue_id,
                "scenario_title": title,
                "speaker": spk,
                "personality": personality[spk],
            }

            if chunk_len is None:
                seq = dict(base)
                seq["steps"] = steps
                all_sequences.append(seq)
            else:
                for start in range(0, len(steps), chunk_len):
                    chunk = steps[start : start + chunk_len]
                    if len(chunk) < min_steps:
                        continue
                    seq = dict(base)
                    seq["chunk_start"] = start
                    seq["steps"] = chunk
                    all_sequences.append(seq)

    return all_sequences


# ============================================================
# Dataset and collator
# ============================================================


class EmotionSequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[Dict[str, Any]]):
        self.sequences = list(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq = self.sequences[idx]
        steps = seq["steps"]

        return {
            "personality": torch.tensor(seq["personality"], dtype=torch.float32),
            "v_self_prev": torch.tensor(
                [s["v_self_prev"] for s in steps], dtype=torch.float32
            ),
            "v_other": torch.tensor(
                [s["v_other"] for s in steps], dtype=torch.float32
            ),
            "v_target": torch.tensor(
                [s["v_target"] for s in steps], dtype=torch.float32
            ),
            "length": len(steps),
            "meta": {
                "dialogue_id": seq.get("dialogue_id"),
                "speaker": seq.get("speaker"),
                "scenario_title": seq.get("scenario_title"),
            },
        }


def collate_emotion_sequences(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)

    personality_dim = batch[0]["personality"].shape[-1]
    vad_dim = batch[0]["v_target"].shape[-1]

    personality = torch.zeros(batch_size, personality_dim, dtype=torch.float32)
    v_self_prev = torch.zeros(batch_size, max_len, vad_dim, dtype=torch.float32)
    v_other = torch.zeros(batch_size, max_len, vad_dim, dtype=torch.float32)
    v_target = torch.zeros(batch_size, max_len, vad_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    metas = []

    for i, item in enumerate(batch):
        L = item["length"]
        personality[i] = item["personality"]
        v_self_prev[i, :L] = item["v_self_prev"]
        v_other[i, :L] = item["v_other"]
        v_target[i, :L] = item["v_target"]
        mask[i, :L] = 1.0
        lengths[i] = L
        metas.append(item["meta"])

    return {
        "personality": personality,
        "v_self_prev": v_self_prev,
        "v_other": v_other,
        "v_target": v_target,
        "mask": mask,
        "lengths": lengths,
        "meta": metas,
    }


# ============================================================
# Model
# ============================================================


class EmotionStateGRU(nn.Module):
    """
    Full-sequence GRU for personality-conditioned emotion dynamics.

    Notation:
        v_A(t): observable VAD vector, shape [3]
        z_A(t): latent cumulative emotional state, shape [z_dim]

    For each speaker-specific transition:
        x_t = [v_A(t-1), v_B(t), v_B(t)-v_A(t-1), personality_A]
        z_A(t) = GRU(x_t, z_A(t-1))
        v_A_hat(t) = decoder(z_A(t))

    Important:
        The GRU output z_seq has shape [B, T, z_dim].
        The decoded VAD output v_pred has shape [B, T, 3].

    If you normalize VAD to [0, 1], use output_range="0_1".
    If you keep VAD in [-1, 1], use output_range="neg1_1".
    """

    def __init__(
        self,
        vad_dim: int = 3,
        personality_dim: int = 5,
        z_dim: int = 64,
        input_hidden_dim: Optional[int] = None,
        decoder_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        output_range: str = "0_1",
    ):
        super().__init__()

        self.vad_dim = vad_dim
        self.personality_dim = personality_dim
        self.z_dim = z_dim
        self.output_range = output_range

        input_hidden_dim = input_hidden_dim or z_dim
        decoder_hidden_dim = decoder_hidden_dim or z_dim

        raw_input_dim = vad_dim + vad_dim + vad_dim + personality_dim
        #               self_prev + other_now + diff + personality

        input_layers: List[nn.Module] = [nn.Linear(raw_input_dim, input_hidden_dim)]
        if use_layernorm:
            input_layers.append(nn.LayerNorm(input_hidden_dim))
        input_layers += [nn.Tanh(), nn.Dropout(dropout), nn.Linear(input_hidden_dim, z_dim), nn.Tanh()]
        self.input_proj = nn.Sequential(*input_layers)

        self.init_state = nn.Sequential(
            nn.Linear(vad_dim + personality_dim, z_dim),
            nn.Tanh(),
        )

        self.gru = nn.GRU(
            input_size=z_dim,
            hidden_size=z_dim,
            batch_first=True,
        )

        if output_range == "0_1":
            output_activation: nn.Module = nn.Sigmoid()
        elif output_range == "neg1_1":
            output_activation = nn.Tanh()
        elif output_range == "raw":
            output_activation = nn.Identity()
        else:
            raise ValueError(
                "output_range must be one of: '0_1', 'neg1_1', 'raw'"
            )

        self.vad_decoder = nn.Sequential(
            nn.Linear(z_dim, decoder_hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, vad_dim),
            output_activation,
        )

    def make_h0(self, v_initial: torch.Tensor, personality: torch.Tensor) -> torch.Tensor:
        """
        v_initial:   [B, 3]
        personality: [B, 5]

        returns:
            h0: [1, B, z_dim]
        """
        h0 = self.init_state(torch.cat([v_initial, personality], dim=-1))
        return h0.unsqueeze(0)

    def build_input(
        self,
        v_self_prev: torch.Tensor,
        v_other: torch.Tensor,
        personality: torch.Tensor,
    ) -> torch.Tensor:
        """
        v_self_prev: [B, T, 3]
        v_other:     [B, T, 3]
        personality: [B, 5]

        returns:
            x: [B, T, z_dim]
        """
        B, T, _ = v_self_prev.shape
        personality_seq = personality.unsqueeze(1).expand(B, T, -1)
        diff = v_other - v_self_prev

        x_raw = torch.cat(
            [v_self_prev, v_other, diff, personality_seq],
            dim=-1,
        )
        return self.input_proj(x_raw)

    def forward(
        self,
        v_self_prev: torch.Tensor,
        v_other: torch.Tensor,
        personality: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full-sequence training/inference.

        v_self_prev: [B, T, 3]
        v_other:     [B, T, 3]
        personality: [B, 5]
        h0:          [1, B, z_dim], optional
        """
        if h0 is None:
            h0 = self.make_h0(v_self_prev[:, 0], personality)

        x = self.build_input(v_self_prev, v_other, personality)
        z_seq, h_final = self.gru(x, h0)
        v_pred = self.vad_decoder(z_seq)

        return {
            "v_pred": v_pred,
            "z_seq": z_seq,
            "h_final": h_final,
        }

    @torch.no_grad()
    def step(
        self,
        v_self_prev: torch.Tensor,
        v_other: torch.Tensor,
        personality: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        One-step update for live simulation/inference.

        v_self_prev: [B, 3]
        v_other:     [B, 3]
        personality: [B, 5]
        h_prev:      [1, B, z_dim], optional

        returns:
            v_pred:  [B, 3]
            z:       [B, z_dim]
            h_final: [1, B, z_dim]
        """
        out = self.forward(
            v_self_prev=v_self_prev.unsqueeze(1),
            v_other=v_other.unsqueeze(1),
            personality=personality,
            h0=h_prev,
        )
        return {
            "v_pred": out["v_pred"][:, 0],
            "z": out["z_seq"][:, 0],
            "h_final": out["h_final"],
        }


# ============================================================
# Losses and metrics
# ============================================================


def masked_mse(v_pred: torch.Tensor, v_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # [B, T, 3] -> [B, T]
    err = ((v_pred - v_target) ** 2).mean(dim=-1)
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mae(v_pred: torch.Tensor, v_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = torch.abs(v_pred - v_target).mean(dim=-1)
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


def masked_rmse(v_pred: torch.Tensor, v_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(masked_mse(v_pred, v_target, mask).clamp_min(1e-12))


def flat_dim_metrics(
    pred_flat: torch.Tensor,
    target_flat: torch.Tensor,
) -> Dict[str, float]:
    """
    Dimension-wise MSE/MAE/RMSE/R2 for flattened valid predictions.

    pred_flat:   [N, 3]
    target_flat: [N, 3]

    This avoids concatenating padded [B, T, 3] tensors from different batches,
    because different batches can have different max sequence lengths T.
    """
    metrics: Dict[str, float] = {}
    names = ["valence", "arousal", "dominance"]

    if pred_flat.numel() == 0:
        for name in names:
            metrics[f"{name}_mse"] = float("nan")
            metrics[f"{name}_mae"] = float("nan")
            metrics[f"{name}_rmse"] = float("nan")
            metrics[f"{name}_r2"] = float("nan")
        return metrics

    for i, name in enumerate(names):
        p = pred_flat[:, i]
        y = target_flat[:, i]
        mse = torch.mean((p - y) ** 2)
        mae = torch.mean(torch.abs(p - y))
        rmse = torch.sqrt(mse.clamp_min(1e-12))
        sse = torch.sum((p - y) ** 2)
        sst = torch.sum((y - torch.mean(y)) ** 2)
        r2 = 1.0 - sse / sst.clamp_min(1e-12)

        metrics[f"{name}_mse"] = float(mse.detach().cpu())
        metrics[f"{name}_mae"] = float(mae.detach().cpu())
        metrics[f"{name}_rmse"] = float(rmse.detach().cpu())
        metrics[f"{name}_r2"] = float(r2.detach().cpu())

    return metrics


# ============================================================
# Train / evaluate
# ============================================================


def run_epoch(
    model: EmotionStateGRU,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    n_batches = 0

    # Store only valid unpadded positions. Each element is [N_valid_in_batch, 3].
    # Do NOT store full [B, T, 3] tensors, because T can differ between batches.
    all_preds_flat = []
    all_targets_flat = []

    for batch in loader:
        personality = batch["personality"].to(device)
        v_self_prev = batch["v_self_prev"].to(device)
        v_other = batch["v_other"].to(device)
        v_target = batch["v_target"].to(device)
        mask = batch["mask"].to(device)

        with torch.set_grad_enabled(is_train):
            out = model(
                v_self_prev=v_self_prev,
                v_other=v_other,
                personality=personality,
            )
            v_pred = out["v_pred"]
            loss = masked_mse(v_pred, v_target, mask)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        with torch.no_grad():
            total_loss += float(loss.detach().cpu())
            total_mae += float(masked_mae(v_pred, v_target, mask).detach().cpu())
            total_rmse += float(masked_rmse(v_pred, v_target, mask).detach().cpu())
            n_batches += 1

            valid = mask.detach().bool()
            all_preds_flat.append(v_pred.detach().cpu()[valid.cpu()])
            all_targets_flat.append(v_target.detach().cpu()[valid.cpu()])

    if n_batches == 0:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan")}

    pred_flat = torch.cat(all_preds_flat, dim=0)
    target_flat = torch.cat(all_targets_flat, dim=0)

    metrics = {
        "mse": total_loss / n_batches,
        "mae": total_mae / n_batches,
        "rmse": total_rmse / n_batches,
    }
    metrics.update(flat_dim_metrics(pred_flat, target_flat))
    return metrics


def save_checkpoint(
    path: str | Path,
    model: EmotionStateGRU,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_mse: float,
    config: Dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_mse": best_val_mse,
            "config": config,
            "big5_keys": BIG5_KEYS,
            "vad_keys": VAD_KEYS,
        },
        path,
    )


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    keys = ["mse", "rmse", "mae", "valence_r2", "arousal_r2", "dominance_r2"]
    parts = []
    for k in keys:
        if k in metrics:
            parts.append(f"{prefix}_{k}={metrics[k]:.5f}")
    return " | ".join(parts)


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="emotion_gru_runs/run1")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--vad_norm",
        type=str,
        default="neg1_1_to_0_1",
        choices=["none", "neg1_1_to_0_1", "one5_to_0_1"],
        help=(
            "VAD normalization. For your current custom JSONL example, use the default: "
            "neg1_1_to_0_1, because your values appear to be in [-1, 1]."
        ),
    )

    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--include_first_turn",
        action="store_true",
        help="Train first utterance of each speaker using persona initial_vad as previous state.",
    )
    parser.add_argument(
        "--exclude_first_turn",
        action="store_true",
        help="Skip first utterance of each speaker. Overrides --include_first_turn.",
    )
    parser.add_argument(
        "--chunk_len",
        type=int,
        default=0,
        help="0 = full speaker sequences. 1 = one-turn pairs. K = chunks of K transitions.",
    )
    parser.add_argument("--min_steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_first_turn = True
    if args.exclude_first_turn:
        include_first_turn = False
    elif args.include_first_turn:
        include_first_turn = True

    chunk_len = None if args.chunk_len == 0 else args.chunk_len

    raw_dialogues = load_jsonl(args.jsonl_path)
    train_dialogues, val_dialogues = split_dialogues(
        raw_dialogues,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_sequences = build_speaker_sequences(
        train_dialogues,
        include_first_turn=include_first_turn,
        min_steps=args.min_steps,
        chunk_len=chunk_len,
        vad_norm=args.vad_norm,
    )
    val_sequences = build_speaker_sequences(
        val_dialogues,
        include_first_turn=include_first_turn,
        min_steps=args.min_steps,
        chunk_len=chunk_len,
        vad_norm=args.vad_norm,
    )

    if len(train_sequences) == 0:
        raise RuntimeError("No training sequences were built. Check your JSONL format.")
    if len(val_sequences) == 0:
        print("Warning: no validation sequences were built. Validation will be skipped.")

    train_dataset = EmotionSequenceDataset(train_sequences)
    val_dataset = EmotionSequenceDataset(val_sequences)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_emotion_sequences,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_emotion_sequences,
    )

    output_range = "0_1" if args.vad_norm in {"neg1_1_to_0_1", "one5_to_0_1"} else "raw"

    model = EmotionStateGRU(
        vad_dim=3,
        personality_dim=5,
        z_dim=args.z_dim,
        dropout=args.dropout,
        output_range=output_range,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config = vars(args).copy()
    config.update(
        {
            "n_raw_dialogues": len(raw_dialogues),
            "n_train_dialogues": len(train_dialogues),
            "n_val_dialogues": len(val_dialogues),
            "n_train_sequences": len(train_sequences),
            "n_val_sequences": len(val_sequences),
            "device": str(device),
            "big5_order": BIG5_KEYS,
            "vad_order": VAD_KEYS,
            "vad_norm": args.vad_norm,
            "model_output_range": output_range,
        }
    )

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("Loaded data:")
    print(f"  raw dialogues:     {len(raw_dialogues)}")
    print(f"  train dialogues:   {len(train_dialogues)}")
    print(f"  val dialogues:     {len(val_dialogues)}")
    print(f"  train sequences:   {len(train_sequences)}")
    print(f"  val sequences:     {len(val_sequences)}")
    print(f"  chunk_len:         {chunk_len}")
    print(f"  include_first_turn:{include_first_turn}")
    print(f"  device:            {device}")

    best_val_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        if len(val_sequences) > 0:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    device=device,
                    grad_clip=args.grad_clip,
                )
            val_mse = val_metrics["mse"]
        else:
            val_metrics = {}
            val_mse = train_metrics["mse"]

        print(
            f"Epoch {epoch:03d} | "
            f"{format_metrics('train', train_metrics)} | "
            f"{format_metrics('val', val_metrics)}"
        )

        save_checkpoint(
            output_dir / "last_model.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_mse=best_val_mse,
            config=config,
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            save_checkpoint(
                output_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_mse=best_val_mse,
                config=config,
            )
            print(f"  saved new best: val_mse={best_val_mse:.6f}")

    print("Training complete.")
    print(f"Best validation MSE: {best_val_mse:.6f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
