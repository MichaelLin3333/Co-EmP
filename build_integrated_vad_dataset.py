from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


# ============================================================
# 1. VAD prototype mappings
# Scale: [0, 1]
# These are weak labels, not gold labels.
# EmoBank remains the gold VAD source.
# ============================================================

DAILYDIALOG_ID2EMOTION = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise",
}

BASIC_EMOTION_TO_VAD = {
    "neutral":   (0.50, 0.20, 0.50),
    "anger":     (0.20, 0.80, 0.65),
    "disgust":   (0.15, 0.60, 0.45),
    "fear":      (0.10, 0.85, 0.20),
    "happiness": (0.85, 0.65, 0.70),
    "sadness":   (0.15, 0.25, 0.25),
    "surprise":  (0.55, 0.85, 0.45),
}


# EmpatheticDialogues has around 32 emotion-context labels.
# These are approximate VAD prototypes for weak supervision.
EMPATHETIC_TO_VAD = {
    "surprised":     (0.55, 0.85, 0.45),
    "excited":       (0.85, 0.85, 0.70),
    "angry":         (0.20, 0.80, 0.65),
    "proud":         (0.80, 0.60, 0.80),
    "sad":           (0.15, 0.25, 0.25),
    "annoyed":       (0.25, 0.65, 0.55),
    "grateful":      (0.85, 0.45, 0.65),
    "lonely":        (0.20, 0.30, 0.20),
    "afraid":        (0.10, 0.85, 0.20),
    "terrified":     (0.05, 0.95, 0.10),
    "guilty":        (0.20, 0.45, 0.25),
    "impressed":     (0.75, 0.60, 0.55),
    "disgusted":     (0.15, 0.60, 0.45),
    "hopeful":       (0.75, 0.55, 0.60),
    "confident":     (0.80, 0.55, 0.85),
    "furious":       (0.10, 0.90, 0.75),
    "anxious":       (0.20, 0.80, 0.25),
    "anticipating":  (0.60, 0.70, 0.50),
    "joyful":        (0.90, 0.70, 0.70),
    "nostalgic":     (0.60, 0.35, 0.45),
    "disappointed":  (0.20, 0.35, 0.30),
    "prepared":      (0.65, 0.45, 0.75),
    "jealous":       (0.25, 0.65, 0.40),
    "content":       (0.80, 0.30, 0.65),
    "devastated":    (0.05, 0.45, 0.10),
    "embarrassed":   (0.25, 0.65, 0.20),
    "caring":        (0.80, 0.40, 0.60),
    "sentimental":   (0.65, 0.40, 0.45),
    "trusting":      (0.75, 0.35, 0.60),
    "ashamed":       (0.15, 0.45, 0.15),
    "apprehensive":  (0.30, 0.65, 0.25),
    "faithful":      (0.75, 0.35, 0.65),
}


# ============================================================
# 2. Utility functions
# ============================================================

def clean_text(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None

    text = text.replace("_comma_", ",")
    text = re.sub(r"\s+", " ", text).strip()

    # Remove simple speaker prefixes if your synthetic data has them.
    text = re.sub(r"^(speaker\s*)?[AB]\s*:\s*", "", text, flags=re.IGNORECASE)

    if not text:
        return None

    return text


def normalize_label(label: Any) -> str:
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_vad_triplet(
    v: Any,
    a: Any,
    d: Any,
    scale: str = "auto",
) -> Tuple[float, float, float]:
    """
    Returns VAD in [0, 1].

    Supported scales:
    - "0_1": already [0, 1]
    - "minus1_1": [-1, 1]
    - "1_5": EmoBank-style scale
    - "1_9": ANEW/NRC-style scale
    - "auto": infer from value range
    """
    vals = [float(v), float(a), float(d)]

    if scale == "auto":
        mn, mx = min(vals), max(vals)
        if mn < 0:
            scale = "minus1_1"
        elif mx > 1 and mx <= 5:
            scale = "1_5"
        elif mx > 5 and mx <= 9:
            scale = "1_9"
        else:
            scale = "0_1"

    if scale == "0_1":
        vals = vals
    elif scale == "minus1_1":
        vals = [(x + 1.0) / 2.0 for x in vals]
    elif scale == "1_5":
        vals = [(x - 1.0) / 4.0 for x in vals]
    elif scale == "1_9":
        vals = [(x - 1.0) / 8.0 for x in vals]
    else:
        raise ValueError(f"Unsupported VAD scale: {scale}")

    return tuple(clamp01(x) for x in vals)


def make_row(
    text: str,
    vad: Tuple[float, float, float],
    source: str,
    split: str,
    label_source: str,
    original_label: Optional[Any] = None,
) -> Dict[str, Any]:
    v, a, d = vad

    return {
        "text": text,
        "labels": [float(v), float(a), float(d)],
        "valence": float(v),
        "arousal": float(a),
        "dominance": float(d),
        "source": source,
        "split": standardize_split(split),
        "label_source": label_source,
        "original_label": None if original_label is None else str(original_label),
    }


def standardize_split(split: Any) -> str:
    s = str(split).lower().strip()

    if s in {"valid", "validation", "dev"}:
        return "validation"
    if s in {"test", "testing"}:
        return "test"
    return "train"


def first_existing(example: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return None


def try_load_dataset(specs: List[Tuple[Any, ...]]):
    errors = []

    for spec in specs:
        try:
            return load_dataset(*spec)
        except Exception as e:
            errors.append(f"{spec}: {repr(e)}")

    raise RuntimeError("Could not load dataset. Errors:\n" + "\n".join(errors))


# ============================================================
# 3. Load DailyDialog
# ============================================================

def load_dailydialog_as_vad() -> pd.DataFrame:
    ds = try_load_dataset([
        ("roskoN/dailydialog", "full"),
        ("roskoN/dailydialog",),
        ("daily_dialog",),
    ])

    rows = []

    for split_name, split_ds in ds.items():
        for example in split_ds:
            utterances = first_existing(example, ["utterances", "dialog", "dialogue"])
            emotions = first_existing(example, ["emotions", "emotion"])

            if utterances is None or emotions is None:
                continue

            for utterance, emotion_id in zip(utterances, emotions):
                text = clean_text(utterance)
                if text is None:
                    continue

                try:
                    emotion_id = int(emotion_id)
                except Exception:
                    continue

                emotion_name = DAILYDIALOG_ID2EMOTION.get(emotion_id, "neutral")
                vad = BASIC_EMOTION_TO_VAD[emotion_name]

                rows.append(
                    make_row(
                        text=text,
                        vad=vad,
                        source="DailyDialog",
                        split=split_name,
                        label_source="categorical_to_vad_prototype",
                        original_label=emotion_name,
                    )
                )

    return pd.DataFrame(rows)


# ============================================================
# 4. Load EmpatheticDialogues
# ============================================================

def load_empathetic_as_vad() -> pd.DataFrame:
    ds = try_load_dataset([
        ("C:\\Users\\Michael Lin\\.cache\\huggingface\\datasets\\Estwld___empathetic_dialogues_llm\\default\\0.0.0\\e85c6e00b972d7340b9f61165b48e207425cd6e0"),
        ("Estwld/empathetic_dialogues_llm",),
        ("facebook/empathetic_dialogues",),
        ("testingtest111/empathetic_dialogues",),
    ])

    rows = []
    unknown_labels = set()

    for split_name, split_ds in ds.items():
        for example in split_ds:
            conv = example.get("conversations")

            for line in conv:
                if line.get("role") != "user":
                    continue
                text = clean_text(line.get("content"))
                if text is None:
                    continue

                emotion_label = normalize_label(example.get("emotion"))

                if emotion_label not in EMPATHETIC_TO_VAD:
                    unknown_labels.add(emotion_label)
                    continue

                vad = EMPATHETIC_TO_VAD[emotion_label]

                rows.append(
                    make_row(
                        text=text,
                        vad=vad,
                        source="EmpatheticDialogues",
                        split=split_name,
                        label_source="categorical_to_vad_prototype",
                        original_label=emotion_label,
                    )
                )

    if unknown_labels:
        print(f"[Warning] Skipped unknown EmpatheticDialogues labels: {sorted(unknown_labels)}")

    return pd.DataFrame(rows)


# ============================================================
# 5. Load EmoBank
# ============================================================

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def load_emobank_as_vad(emobank_path: Optional[str]) -> pd.DataFrame:
    """
    EmoBank usually has columns like:
    id, split, V, A, D, text

    The V/A/D values are commonly on a 1-5 scale, so this function
    normalizes them into [0, 1].
    """
    if emobank_path is None:
        emobank_path = (
            "EmoBank\corpus\emobank.csv"
        )

    df = pd.read_csv(emobank_path)

    text_col = find_column(df, ["text", "sentence", "utterance"])
    v_col = find_column(df, ["V", "valence", "val"])
    a_col = find_column(df, ["A", "arousal", "aro"])
    d_col = find_column(df, ["D", "dominance", "dom"])
    split_col = find_column(df, ["split", "partition"])

    required = {
        "text": text_col,
        "valence": v_col,
        "arousal": a_col,
        "dominance": d_col,
    }

    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(
            f"Missing required EmoBank columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    rows = []

    for _, row in df.iterrows():
        text = clean_text(row[text_col])
        if text is None:
            continue

        split = row[split_col] if split_col is not None else "train"

        vad = normalize_vad_triplet(
            row[v_col],
            row[a_col],
            row[d_col],
            scale="auto",
        )

        rows.append(
            make_row(
                text=text,
                vad=vad,
                source="EmoBank",
                split=split,
                label_source="gold_vad",
                original_label=None,
            )
        )

    return pd.DataFrame(rows)


# ============================================================
# 6. Load your custom AI-generated dialogue
# ============================================================


def load_custom_as_vad(custom_path: Optional[str]) -> pd.DataFrame:
    """
    Loads your custom AI-generated dialogue dataset.

    Expected JSONL format:

    {
      "scenario": {
        "title": "...",
        "description": "..."
      },
      "personas": {
        "A": {...},
        "B": {...}
      },
      "dialogue": [
        {
          "speaker": "A",
          "text": "...",
          "vad": {"v": 0.3, "a": 0.2, "d": 0.6}
        },
        {
          "speaker": "B",
          "text": "...",
          "vad": {"v": 0.4, "a": 0.6, "d": 0.5}
        }
      ]
    }

    Output:
    utterance-level VAD rows:
    {
      "text": utterance,
      "labels": [v, a, d],
      "valence": v,
      "arousal": a,
      "dominance": d,
      ...
    }
    """

    print("loading custom AI-generated dialogue dataset...")
    if custom_path is None:
        custom_path = "EmoDynamic/EmoDynamic.jsonl"

    path = Path(custom_path)
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON decode failed at line {line_idx + 1}: {e}")
                continue

            scenario = sample.get("scenario", {})
            scenario_title = scenario.get("title")
            scenario_description = scenario.get("description")

            dialogue = sample.get("dialogue", [])

            if not isinstance(dialogue, list):
                print(f"[Warning] Invalid dialogue field at line {line_idx + 1}")
                continue

            for turn_idx, turn in enumerate(dialogue):
                if not isinstance(turn, dict):
                    continue

                text = clean_text(turn.get("text"))
                if text is None:
                    continue

                speaker = turn.get("speaker")
                vad = turn.get("vad")

                if not isinstance(vad, dict):
                    print(
                        f"[Warning] Missing/invalid VAD at line {line_idx + 1}, "
                        f"turn {turn_idx + 1}"
                    )
                    continue

                try:
                    v = vad.get("v", vad.get("valence"))
                    a = vad.get("a", vad.get("arousal"))
                    d = vad.get("d", vad.get("dominance"))

                    if v is None or a is None or d is None:
                        raise ValueError("VAD must contain v/a/d or valence/arousal/dominance")

                    vad_tuple = normalize_vad_triplet(v, a, d, scale="auto")

                except Exception as e:
                    print(
                        f"[Warning] Failed parsing VAD at line {line_idx + 1}, "
                        f"turn {turn_idx + 1}: {e}"
                    )
                    continue

                row = make_row(
                    text=text,
                    vad=vad_tuple,
                    source="custom_ai_dialogue",
                    split="train",
                    label_source="synthetic_vad",
                    original_label=None,
                )

                # Optional metadata for later analysis.
                # These columns are not used as model input.
                row["speaker"] = speaker
                row["scenario_title"] = scenario_title
                row["scenario_description"] = scenario_description
                row["line_idx"] = line_idx
                row["turn_idx"] = turn_idx

                rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 7. Build integrated dataset
# ============================================================

def build_dataset(
    out_dir: str,
    emobank_path: Optional[str],
    custom_path: Optional[str],
    include_dailydialog: bool,
    include_empathetic: bool,
    include_emobank: bool,
    dedupe_text: bool,
) -> None:
    frames = []

    if include_dailydialog:
        print("Loading DailyDialog...")
        frames.append(load_dailydialog_as_vad())

    if include_empathetic:
        print("Loading EmpatheticDialogues...")
        frames.append(load_empathetic_as_vad())

    if include_emobank:
        print("Loading EmoBank...")
        frames.append(load_emobank_as_vad(emobank_path))

    if custom_path is not None:
        print("Loading custom AI-generated dialogue...")
        frames.append(load_custom_as_vad(custom_path))

    frames = [df for df in frames if df is not None and len(df) > 0]

    if not frames:
        raise RuntimeError("No data loaded.")

    df = pd.concat(frames, ignore_index=True)

    df = df.dropna(subset=["text", "valence", "arousal", "dominance"])
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df.dropna(subset=["text"])

    if dedupe_text:
        # Keep EmoBank first because it has true VAD.
        priority = {
            "gold_vad": 0,
            "synthetic_or_user_generated_vad": 1,
            "categorical_to_vad_prototype": 2,
        }
        df["_priority"] = df["label_source"].map(priority).fillna(99)
        df = df.sort_values("_priority")
        df = df.drop_duplicates(subset=["text"], keep="first")
        df = df.drop(columns=["_priority"])

    # Rebuild labels after any normalization/deduplication.
    df["labels"] = df[["valence", "arousal", "dominance"]].apply(
        lambda r: [float(r["valence"]), float(r["arousal"]), float(r["dominance"])],
        axis=1,
    )

    # Keep only model-relevant + audit columns.
    base_columns = [
        "text",
        "labels",
        "valence",
        "arousal",
        "dominance",
        "split",
        "source",
        "label_source",
        "original_label",
    ]

    optional_columns = [
        "speaker",
        "scenario_title",
        "scenario_description",
        "line_idx",
        "turn_idx",
    ]

    existing_columns = [
        col for col in base_columns + optional_columns
        if col in df.columns
    ]

    df = df[existing_columns]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save full table.
    df.to_json(out / "all.jsonl", orient="records", lines=True, force_ascii=False)
    df.to_csv(out / "all.csv", index=False)

    # Save split files.
    dataset_dict = {}

    for split_name, split_df in df.groupby("split"):
        split_df = split_df.reset_index(drop=True)
        split_df.to_json(
            out / f"{split_name}.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        dataset_dict[split_name] = Dataset.from_pandas(split_df, preserve_index=False)

    hf_dataset = DatasetDict(dataset_dict)
    hf_dataset.save_to_disk(out / "hf_dataset")

    counts = (
        df.groupby(["source", "split", "label_source"])
        .size()
        .reset_index(name="n")
        .sort_values(["source", "split", "label_source"])
    )
    counts.to_csv(out / "counts.csv", index=False)

    print("\nDone.")
    print(f"Saved to: {out.resolve()}")
    print("\nCounts:")
    print(counts.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out_dir",
        type=str,
        default="integrated_vad_utterance_dataset",
    )
    parser.add_argument(
        "--emobank_path",
        type=str,
        default=None,
        help="Optional local path to emobank.csv. If omitted, tries GitHub raw CSV.",
    )
    parser.add_argument(
        "--custom_path",
        type=str,
        default=None,
        help="Optional path to your custom JSON or JSONL file.",
    )

    parser.add_argument("--no_dailydialog", action="store_true")
    parser.add_argument("--no_empathetic", action="store_true")
    parser.add_argument("--no_emobank", action="store_true")
    parser.add_argument("--dedupe_text", action="store_true")

    args = parser.parse_args()

    build_dataset(
        out_dir=args.out_dir,
        emobank_path=args.emobank_path,
        custom_path=args.custom_path,
        include_dailydialog=not args.no_dailydialog,
        include_empathetic=not args.no_empathetic,
        include_emobank=not args.no_emobank,
        dedupe_text=args.dedupe_text,
    )


if __name__ == "__main__":
    main()