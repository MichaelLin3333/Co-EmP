"""
affective_qwen_gradio_inference.py

Emotion-dynamic chatbot inference script:

Speaker A is played by the human user.
For each user utterance:
    1. VAD regressor predicts Speaker A's current utterance VAD.
    2. GRU emotion-dynamics model predicts Speaker B's target VAD.
    3. Qwen generates Speaker B's response conditioned on dialogue context + predicted B VAD.
    4. VAD regressor scores Speaker B's generated response so the next turn has an updated B state.

GRU input format used by this script:
    [v_self_prev, v_other, v_diff, personality_seq]

For predicting Speaker B:
    v_self_prev    = previous/last known Speaker B VAD
    v_other        = current Speaker A VAD predicted from the user's utterance
    v_diff         = v_other - v_self_prev
    personality_seq = Speaker B personality vector

So with Big Five personality, the default GRU input dimension is:
    3 + 3 + 3 + 5 = 14

Expected usage example:

python affective_qwen_gradio_inference.py \
  --vad_model_path ./checkpoints/vad_roberta \
  --gru_checkpoint ./checkpoints/gru_dynamics.pt \
  --gru_model_class GRU_full_encoder:EmotionDynamicsGRU \
  --gru_model_kwargs '{"input_dim":14,"hidden_dim":128,"num_layers":1,"output_dim":3}' \
  --gru_forward_mode concat_seq \
  --qwen_model_path ./models/Qwen3.5-4B-Instruct \
  --vad_min 0 --vad_max 1

Important integration point:
    Edit GRUDynamicsAdapter.build_features() only if your training code defined v_diff differently.
    The current definition is:
        v_diff = v_other - v_self_prev

Personality format in the UI:
    - Raw text is always passed into Qwen as role/persona description.
    - For GRU numeric input, personality is parsed from either:
        1. JSON dict with Big Five keys:
           {"openness":0.7,"conscientiousness":0.4,"extraversion":0.2,"agreeableness":0.5,"neuroticism":0.8}
        2. JSON/list or comma-separated 5 floats:
           0.7,0.4,0.2,0.5,0.8
    - If no numeric personality can be parsed, zeros are used for the GRU personality vector,
      but the raw personality text still conditions Qwen.


Few-shot style guide:
- VAD ≈ (0.15, 0.20, 0.20): "I... I don't know what you want me to say. I'm tired, and I can't keep fighting about this."
- VAD ≈ (0.15, 0.85, 0.80): "No. Don't twist this around on me. You knew exactly what would happen, and you did it anyway."
- VAD ≈ (0.75, 0.25, 0.35): "It's okay. I'm not angry. I just need a moment to understand what you're asking from me."
- VAD ≈ (0.80, 0.75, 0.70): "Good, then let's fix it now. We still have time, and I'm not giving up on this."
- VAD ≈ (0.45, 0.80, 0.25): "Wait, wait—slow down. I can't tell if you're blaming me or asking for help."
- VAD ≈ (0.35, 0.35, 0.85): "Listen carefully. We are not making this worse by pretending nothing happened."
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

try:
    import gradio as gr
except ImportError as exc:
    raise RuntimeError(
        "Gradio is required for this script. Install it with: pip install gradio"
    ) from exc


VAD_KEYS = ["valence", "arousal", "dominance"]
BIG5_KEYS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

MODE_BASELINE = "1) Pure baseline mode"
MODE_EMOTION = "2) Emotion-supported mode"
MODE_ARENA = "3) Arena comparison"
MODE_TWO_AI = "4) Two-AI mode"
APP_MODES = [MODE_BASELINE, MODE_EMOTION, MODE_ARENA, MODE_TWO_AI]


# -----------------------------
# Basic parsing / formatting
# -----------------------------


def parse_json_maybe(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")
    return json.loads(text)


def parse_float_sequence(text: str, expected_len: int, name: str) -> List[float]:
    """Parse JSON list, JSON dict, or comma/space-separated floats."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError(f"{name} is empty.")

    # JSON list or dict
    try:
        obj = parse_json_maybe(raw)
        if isinstance(obj, dict):
            if name.lower().startswith("vad"):
                keys = VAD_KEYS
            else:
                keys = BIG5_KEYS
            missing = [k for k in keys if k not in obj]
            if missing:
                raise ValueError(f"{name} JSON dict is missing keys: {missing}")
            values = [float(obj[k]) for k in keys]
        elif isinstance(obj, list):
            values = [float(x) for x in obj]
        else:
            raise ValueError(f"{name} JSON must be a list or dict.")
    except json.JSONDecodeError:
        pieces = re.split(r"[,\s]+", raw)
        values = [float(p) for p in pieces if p != ""]

    if len(values) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} floats, got {len(values)}: {values}")
    return values


def parse_personality(raw_text: str, personality_dim: int = 5) -> Tuple[str, List[float], str]:
    """
    Returns:
        raw_personality_text: str passed to the LLM prompt.
        personality_vector: numeric vector passed to GRU.
        warning: warning string if fallback was used.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return "No explicit personality description provided.", [0.0] * personality_dim, (
            "No personality was provided; using zeros for GRU personality vector."
        )

    try:
        vec = parse_float_sequence(raw_text, personality_dim, "personality")
        return raw_text, vec, ""
    except Exception:
        # Try to find a numeric vector inside mixed text, e.g. "Big5: 0.1,0.2,0.3,0.4,0.5"
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", raw_text)
        if len(nums) == personality_dim:
            try:
                return raw_text, [float(x) for x in nums], ""
            except Exception:
                pass

    return raw_text, [0.0] * personality_dim, (
        "Could not parse numeric Big Five/personality vector; using zeros for GRU personality vector. "
        "The raw text is still passed to Qwen."
    )


def clamp_vector(values: Sequence[float], min_v: float, max_v: float) -> List[float]:
    return [max(min(float(x), max_v), min_v) for x in values]


def format_vad(values: Sequence[float]) -> str:
    return ", ".join(f"{k}={float(v):+.3f}" for k, v in zip(VAD_KEYS, values))


def tensor_from(values: Sequence[float], device: torch.device, shape: str = "batch") -> torch.Tensor:
    t = torch.tensor(values, dtype=torch.float32, device=device)
    if shape == "batch":
        return t.unsqueeze(0)
    if shape == "seq":
        return t.unsqueeze(0).unsqueeze(0)
    return t


def dynamic_import(class_path: str):
    """Import class from 'module.submodule:ClassName'."""
    if ":" not in class_path:
        raise ValueError("Class path must look like 'module.submodule:ClassName'.")
    module_name, class_name = class_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def resolve_torch_dtype(name: str):
    if name == "auto":
        return "auto"
    if not hasattr(torch, name):
        raise ValueError(f"Unknown torch dtype: {name}. Examples: float16, bfloat16, float32, auto")
    return getattr(torch, name)


def strip_thinking_and_labels(text: str, strip_thinking: bool = True) -> str:
    text = text or ""
    if strip_thinking:
        text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
        # Some models emit an opening think tag without closing it.
        text = re.sub(r"(?is)^<think>.*", "", text).strip() if text.strip().startswith("<think>") else text

    # Remove common role labels at the beginning only. This must work for both
    # Speaker A and Speaker B because the two-AI mode asks Qwen to play either role.
    text = re.sub(
        r"^\s*(Speaker\s*A|Speaker\s*B|A|B|Assistant|User)\s*[:：]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # If the model starts role-playing both sides, keep only the first generated segment.
    for marker in ["\nSpeaker A:", "\nA:", "\nSpeaker B:", "\nB:", "\nUser:", "\nAssistant:"]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text.strip()


# -----------------------------
# Exact GRU model class from training
# -----------------------------


class EmotionStateGRU(nn.Module):
    """
    Full-sequence GRU for personality-conditioned emotion dynamics.

    Transition input:
        x_t = [v_self_prev, v_other, v_other - v_self_prev, personality_self]

    For live inference, use step():
        v_pred, z, h_final = model.step(v_self_prev, v_other, personality, h_prev)
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
            raise ValueError("output_range must be one of: '0_1', 'neg1_1', 'raw'")

        self.vad_decoder = nn.Sequential(
            nn.Linear(z_dim, decoder_hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, vad_dim),
            output_activation,
        )

    def make_h0(self, v_initial: torch.Tensor, personality: torch.Tensor) -> torch.Tensor:
        h0 = self.init_state(torch.cat([v_initial, personality], dim=-1))
        return h0.unsqueeze(0)

    def build_input(
        self,
        v_self_prev: torch.Tensor,
        v_other: torch.Tensor,
        personality: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = v_self_prev.shape
        personality_seq = personality.unsqueeze(1).expand(B, T, -1)
        diff = v_other - v_self_prev
        x_raw = torch.cat([v_self_prev, v_other, diff, personality_seq], dim=-1)
        return self.input_proj(x_raw)

    def forward(
        self,
        v_self_prev: torch.Tensor,
        v_other: torch.Tensor,
        personality: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if h0 is None:
            h0 = self.make_h0(v_self_prev[:, 0], personality)

        x = self.build_input(v_self_prev, v_other, personality)
        z_seq, h_final = self.gru(x, h0)
        v_pred = self.vad_decoder(z_seq)

        return {"v_pred": v_pred, "z_seq": z_seq, "h_final": h_final}

    @torch.no_grad()
    def step(
        self,
        v_self_prev: torch.Tensor,
        v_other: torch.Tensor,
        personality: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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


# -----------------------------
# Runtime state
# -----------------------------


@dataclass
class SpeakerRuntime:
    name: str
    personality_raw: str
    personality_vec: List[float]
    last_vad: List[float]
    private_context: str = ""


@dataclass
class DialogueTurn:
    a_text: str
    # For human-A modes, a_vad is the VAD regressor score of the human utterance.
    # For two-AI mode, a_vad is the GRU target used to generate Speaker A.
    a_vad: List[float]
    b_target_vad: List[float]
    b_text: str
    b_realized_vad: List[float]
    b_baseline_text: str = ""
    b_baseline_realized_vad: Optional[List[float]] = None
    a_target_vad: Optional[List[float]] = None
    a_realized_vad: Optional[List[float]] = None
    mode: str = ""


@dataclass
class ConversationState:
    speaker_a: SpeakerRuntime
    speaker_b: SpeakerRuntime
    scenario: str = ""
    turns: List[DialogueTurn] = field(default_factory=list)
    h_a: Optional[torch.Tensor] = None
    h_b: Optional[torch.Tensor] = None
    initialized: bool = True


# -----------------------------
# VAD regressor adapter
# -----------------------------


class VADRegressorAdapter:
    """
    Default assumption:
        Your VAD model was saved with Hugging Face Trainer / save_pretrained(), and has
        AutoModelForSequenceClassification output logits of shape [batch, 3].

    If your regressor is custom, replace load_model() and predict() with your own code.
    """

    def __init__(self, args: argparse.Namespace):
        self.model_path = args.vad_model_path
        self.tokenizer_path = args.vad_tokenizer_path or args.vad_model_path
        self.max_length = args.vad_max_length
        self.vad_min = args.vad_min
        self.vad_max = args.vad_max
        self.output_activation = args.vad_output_activation
        self.device = torch.device(args.vad_device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _activate(self, pred: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "none":
            return pred
        if self.output_activation == "sigmoid":
            return torch.sigmoid(pred)
        if self.output_activation == "tanh":
            return torch.tanh(pred)
        if self.output_activation == "tanh_0_1":
            return (torch.tanh(pred) + 1.0) / 2.0
        raise ValueError(f"Unknown VAD output activation: {self.output_activation}")

    @torch.inference_mode()
    def predict(self, text: str) -> List[float]:
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        out = self.model(**tokens)
        pred = out.logits.squeeze(0).float()
        pred = self._activate(pred)
        values = pred.detach().cpu().tolist()
        if isinstance(values, float):
            values = [values]
        if len(values) != 3:
            raise RuntimeError(f"VAD regressor must output 3 values, got {len(values)}: {values}")
        return clamp_vector(values, self.vad_min, self.vad_max)


# -----------------------------
# GRU dynamics adapter
# -----------------------------


class GRUDynamicsAdapter:
    """
    Adapter for your trained GRU emotion-dynamics model.

    This version matches your trained GRU feature format:
        [v_self_prev, v_other, v_diff, personality_seq]

    Canonical conceptual call for Speaker B prediction:
        v_self_prev    = previous/last known VAD of Speaker B
        v_other        = current predicted VAD of Speaker A utterance
        v_diff         = v_other - v_self_prev
        personality_seq = Speaker B personality vector, repeated conceptually across the sequence

    With Big Five personality, one timestep has 14 features:
        3 + 3 + 3 + 5 = 14

    If your training script used the opposite difference direction, edit build_features().
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.checkpoint_path = args.gru_checkpoint
        self.device = torch.device(args.gru_device)
        self.vad_min = args.vad_min
        self.vad_max = args.vad_max
        self.output_activation = args.gru_output_activation
        self.forward_mode = args.gru_forward_mode

        self.model = self.load_model(args)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, args: argparse.Namespace) -> nn.Module:
        if not os.path.exists(args.gru_checkpoint):
            raise FileNotFoundError(f"GRU checkpoint not found: {args.gru_checkpoint}")

        if args.gru_torchscript:
            return torch.jit.load(args.gru_checkpoint, map_location=self.device)

        # If --gru_model_class is omitted, use the exact EmotionStateGRU class embedded
        # in this inference file. This matches the GRU class you used for training.
        if args.gru_model_class:
            ModelClass = dynamic_import(args.gru_model_class)
        else:
            ModelClass = EmotionStateGRU

        model_kwargs = json.loads(args.gru_model_kwargs) if args.gru_model_kwargs else {}
        model = ModelClass(**model_kwargs)

        ckpt = torch.load(args.gru_checkpoint, map_location="cpu")
        if isinstance(ckpt, dict):
            state_dict = (
                ckpt.get("model_state_dict")
                or ckpt.get("state_dict")
                or ckpt.get("model")
                or ckpt
            )
        else:
            state_dict = ckpt

        # Remove common DataParallel prefix.
        if isinstance(state_dict, dict):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        load_info = model.load_state_dict(state_dict, strict=args.gru_strict_load)
        if not args.gru_strict_load:
            print(f"[GRU load_state_dict] missing={load_info.missing_keys}")
            print(f"[GRU load_state_dict] unexpected={load_info.unexpected_keys}")
        return model

    def build_features(
        self,
        prev_self_vad: Sequence[float],
        prev_other_vad: Sequence[float],
        self_personality_vec: Sequence[float],
        other_personality_vec: Sequence[float],
    ) -> List[float]:
        """
        Build one GRU timestep using the feature order from your training setup:

            [v_self_prev, v_other, v_diff, personality_seq]

        Here:
            v_self_prev = previous VAD of the speaker being predicted
            v_other     = current VAD of the other speaker
            v_diff      = v_other - v_self_prev
            personality_seq = personality vector of the speaker being predicted

        For Speaker B prediction:
            v_self_prev = previous B VAD
            v_other     = current A VAD
            personality = B personality

        Note:
            other_personality_vec is intentionally unused because the trained GRU input
            only uses the predicted speaker's personality_seq.
        """
        v_self_prev = list(map(float, prev_self_vad))
        v_other = list(map(float, prev_other_vad))
        v_diff = [other - self_prev for other, self_prev in zip(v_other, v_self_prev)]
        personality_seq = list(map(float, self_personality_vec))

        return v_self_prev + v_other + v_diff + personality_seq

    def _activate(self, pred: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "none":
            return pred
        if self.output_activation == "sigmoid":
            return torch.sigmoid(pred)
        if self.output_activation == "tanh":
            return torch.tanh(pred)
        if self.output_activation == "tanh_0_1":
            return (torch.tanh(pred) + 1.0) / 2.0
        raise ValueError(f"Unknown GRU output activation: {self.output_activation}")

    def _extract_vad_output(self, out: Any) -> torch.Tensor:
        if isinstance(out, dict):
            for key in ["v_pred", "vad", "pred_vad", "prediction", "pred", "logits", "output"]:
                if key in out:
                    out = out[key]
                    break
            else:
                raise RuntimeError(f"Could not find VAD output in GRU dict keys: {list(out.keys())}")
        elif isinstance(out, (tuple, list)):
            out = out[0]

        if not torch.is_tensor(out):
            out = torch.tensor(out, dtype=torch.float32, device=self.device)

        # Accept [B, T, 3], [B, 3], or [3].
        if out.ndim == 3:
            out = out[:, -1, :]
        if out.ndim == 2:
            out = out[0]
        if out.ndim != 1:
            raise RuntimeError(f"Unexpected GRU output shape: {tuple(out.shape)}")
        if out.shape[0] < 3:
            raise RuntimeError(f"GRU output must contain at least 3 values, got shape {tuple(out.shape)}")
        return out[:3].float()

    @torch.inference_mode()
    def step_responder_vad(
        self,
        prev_self_vad: Sequence[float],
        other_vad: Sequence[float],
        self_personality_vec: Sequence[float],
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[List[float], torch.Tensor]:
        """
        Live one-step transition using the trained model's native API:

            out = model.step(v_self_prev, v_other, personality, h_prev)

        Returns:
            predicted_vad: List[float]
            h_final: torch.Tensor with shape [1, B, z_dim]
        """
        v_self_prev = tensor_from(prev_self_vad, self.device, "batch")
        v_other = tensor_from(other_vad, self.device, "batch")
        personality = tensor_from(self_personality_vec, self.device, "batch")
        if h_prev is not None:
            h_prev = h_prev.to(self.device)

        if hasattr(self.model, "step"):
            out = self.model.step(
                v_self_prev=v_self_prev,
                v_other=v_other,
                personality=personality,
                h_prev=h_prev,
            )
            pred = self._extract_vad_output(out)
            h_final = out.get("h_final", h_prev) if isinstance(out, dict) else h_prev
        else:
            # Fallback for a model without step(), using its forward() signature.
            out = self.model(
                v_self_prev=v_self_prev.unsqueeze(1),
                v_other=v_other.unsqueeze(1),
                personality=personality,
                h0=h_prev,
            )
            pred = self._extract_vad_output(out)
            h_final = out.get("h_final", h_prev) if isinstance(out, dict) else h_prev

        pred = self._activate(pred)
        return clamp_vector(pred.detach().cpu().tolist(), self.vad_min, self.vad_max), h_final


# -----------------------------
# Qwen generator adapter
# -----------------------------


class QwenGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_path = args.qwen_model_path
        self.max_input_tokens = args.qwen_max_input_tokens
        self.strip_thinking = args.strip_thinking
        self.enable_thinking = args.enable_thinking

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        dtype = resolve_torch_dtype(args.qwen_dtype)

        if args.qwen_device_map == "none":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            self.model.to(torch.device(args.qwen_device))
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map=args.qwen_device_map,
                trust_remote_code=True,
            )

        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _messages_to_plain_prompt(self, messages: List[Dict[str, str]]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lines.append(f"System instruction:\n{content}")
            elif role == "user":
                lines.append(f"Speaker A:\n{content}")
            else:
                lines.append(f"Speaker B:\n{content}")
        lines.append("Speaker B:")
        return "\n\n".join(lines)

    def _tokenize_messages(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        # Prefer the model's native chat template when available.
        input_ids = None
        if getattr(self.tokenizer, "chat_template", None):
            try:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

            # Different tokenizer versions may return a Tensor, BatchEncoding, or dict.
            if torch.is_tensor(encoded):
                input_ids = encoded
            elif isinstance(encoded, dict):
                input_ids = encoded.get("input_ids")
            elif hasattr(encoded, "input_ids"):
                input_ids = encoded.input_ids

        if input_ids is None:
            prompt = self._messages_to_plain_prompt(messages)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        # Ensure shape [batch, seq_len]. Some tokenizers may return [seq_len].
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        # Left truncate to keep newest context if needed.
        if input_ids.shape[-1] > self.max_input_tokens:
            input_ids = input_ids[:, -self.max_input_tokens :]
        return input_ids.to(self._model_device())

    @torch.inference_mode()
    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int, temperature: float, top_p: float) -> str:
        input_ids = self._tokenize_messages(messages)
        out_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            repetition_penalty=self.args.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        new_ids = out_ids[0, input_ids.shape[-1] :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        text = strip_thinking_and_labels(text, strip_thinking=self.strip_thinking)
        return text or "..."


# -----------------------------
# Prompt construction
# -----------------------------


def describe_vad_for_prompt(vad: Sequence[float]) -> str:
    """
    Convert a numeric VAD vector into compact natural-language acting direction.
    Assumes VAD is normalized to [0, 1]. If your model uses [-1, 1], the labels
    are still directionally useful but you may want to adjust the thresholds.
    """
    v, a, d = [float(x) for x in vad]

    def band(x: float, low: str, mid: str, high: str) -> str:
        if x < 0.35:
            return low
        if x > 0.65:
            return high
        return mid

    valence = band(v, "negative / displeased / hurt", "mixed or emotionally controlled", "positive / warm / pleased")
    arousal = band(a, "low-energy / subdued / restrained", "moderate intensity", "high-energy / tense / urgent")
    dominance = band(d, "low-control / uncertain / yielding", "balanced control", "high-control / assertive / commanding")

    return (
        f"Valence {v:.2f}: {valence}; "
        f"Arousal {a:.2f}: {arousal}; "
        f"Dominance {d:.2f}: {dominance}."
    )


def speaker_label(speaker_id: str) -> str:
    speaker_id = speaker_id.upper().strip()
    if speaker_id not in {"A", "B"}:
        raise ValueError(f"speaker_id must be 'A' or 'B', got {speaker_id!r}")
    return f"Speaker {speaker_id}"


def get_runtime_for_speaker(state: ConversationState, speaker_id: str) -> SpeakerRuntime:
    return state.speaker_a if speaker_id.upper() == "A" else state.speaker_b


def get_other_speaker_id(speaker_id: str) -> str:
    return "B" if speaker_id.upper() == "A" else "A"


def build_transcript(state: ConversationState, max_context_turns: int, use_baseline_history: bool = False) -> str:
    recent_turns = state.turns[-max_context_turns:] if max_context_turns > 0 else []
    if not recent_turns:
        return "No previous dialogue turns."

    start_index = max(1, len(state.turns) - len(recent_turns) + 1)
    lines: List[str] = []
    for i, turn in enumerate(recent_turns, start=start_index):
        b_text = turn.b_baseline_text if use_baseline_history and turn.b_baseline_text else turn.b_text
        lines.append(f"Turn {i}")
        lines.append(f"Speaker A: {turn.a_text}")
        lines.append(f"Speaker B: {b_text}")
    return "\n".join(lines)


def build_speaker_messages(
    state: ConversationState,
    speaker_id: str,
    current_other_text: Optional[str],
    target_vad: Optional[Sequence[float]],
    max_context_turns: int,
    use_vad_hint: bool,
    use_baseline_history: bool = False,
    director_note: str = "",
) -> List[Dict[str, str]]:
    """
    Generic prompt builder used by all four modes.

    Privacy rule:
        Only the responding speaker's private context is inserted. The other
        speaker's private context is never inserted into this prompt.
    """
    speaker_id = speaker_id.upper().strip()
    other_id = get_other_speaker_id(speaker_id)
    speaker = get_runtime_for_speaker(state, speaker_id)
    other = get_runtime_for_speaker(state, other_id)
    speaker_name = speaker_label(speaker_id)
    other_name = speaker_label(other_id)

    scenario_text = state.scenario.strip() or "No additional scenario was provided."
    transcript = build_transcript(state, max_context_turns=max_context_turns, use_baseline_history=use_baseline_history)
    private_context = speaker.private_context.strip() or "None"
    director_note = (director_note or "").strip()

    affect_block = ""
    if use_vad_hint:
        if target_vad is None:
            raise ValueError("target_vad is required when use_vad_hint=True")
        vad_description = describe_vad_for_prompt(target_vad)
        affect_block = f"""
Current target affect for {speaker_name}:
{format_vad(target_vad)}
{vad_description}

How to interpret VAD [0-1 scale] for acting direction:
- Valence controls emotional pleasantness.
  - Low valence: hurt, angry, disappointed, bitter, afraid, sad, resentful, distrustful.
  - Mid valence: neutral, conflicted, controlled, ambivalent, cautious.
  - High valence: warm, relieved, amused, caring, hopeful, affectionate.
- Arousal controls emotional intensity and energy.
  - Low arousal: quiet, tired, slow, restrained, flat, numb, resigned.
  - Mid arousal: conversational, steady, attentive.
  - High arousal: urgent, tense, excited, defensive, panicked, explosive.
- Dominance controls perceived control and social force.
  - Low dominance: hesitant, apologetic, uncertain, submissive, pleading, avoidant.
  - Mid dominance: balanced, cooperative, explanatory.
  - High dominance: firm, commanding, confrontational, decisive, protective, controlling.

Your words should strictly abide by the target VAD above. Think carefully how {format_vad(target_vad)} guide your tone, attitude, and emotional expression.
""".strip()
    else:
        affect_block = (
            "Use the scenario, personalities, private context, and dialogue history."
        )

    if current_other_text and current_other_text.strip():
        current_turn_instruction = f"{other_name} now says:\n{current_other_text.strip()}"
    else:
        current_turn_instruction = (
            "This is the beginning of the scene or the next autonomous step. "
            f"Continue naturally as {speaker_name}."
        )

    director_block = f"\nDirector / experimenter note for this next generation:\n{director_note}\n" if director_note else ""

    system = f"""
You are {speaker_name} in a two-person dialogue simulation.
Write only {speaker_name}'s next spoken reply.

Rules:
- Output only {speaker_name}'s spoken reply.
- Do not output analysis, labels, bullet points, JSON, stage directions, or multiple candidate replies.
- Do not write the other speaker's next line.
- Do not say words like "valence", "arousal", "dominance", "VAD", "emotion score", or any numeric affect value.
- Stay consistent with {speaker_name}'s personality, private context, and the public scenario.
- You may use {speaker_name}'s private context internally, but do not reveal hidden information unless the character would naturally choose to say it.
- Avoid long speeches. Try emulate a realistic conversation turn length.

Scenario/shared context visible to both speakers:
{scenario_text}

{speaker_name}  personality:
{speaker.personality_raw}

{speaker_name} personal context:
{private_context}

Affect/generation condition:
{affect_block}
""".strip()

    user = f"""
Recent dialogue transcript:
{transcript}

{current_turn_instruction}
{director_block}
Respond now as {speaker_name}. Output only the spoken reply.
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_generation_messages(
    state: ConversationState,
    current_a_text: str,
    b_target_vad: Sequence[float],
    max_context_turns: int,
) -> List[Dict[str, str]]:
    return build_speaker_messages(
        state=state,
        speaker_id="B",
        current_other_text=current_a_text,
        target_vad=b_target_vad,
        max_context_turns=max_context_turns,
        use_vad_hint=True,
    )


def build_baseline_messages(
    state: ConversationState,
    current_a_text: str,
    max_context_turns: int,
) -> List[Dict[str, str]]:
    return build_speaker_messages(
        state=state,
        speaker_id="B",
        current_other_text=current_a_text,
        target_vad=None,
        max_context_turns=max_context_turns,
        use_vad_hint=False,
        use_baseline_history=True,
    )


# -----------------------------
# Gradio callbacks
# -----------------------------


def make_state(
    personality_a: str,
    initial_vad_a: str,
    personality_b: str,
    initial_vad_b: str,
    scenario: str,
    speaker_a_private_context: str,
    speaker_b_private_context: str,
    personality_dim: int,
    vad_min: float,
    vad_max: float,
) -> Tuple[ConversationState, Dict[str, Any], str]:
    a_raw, a_vec, a_warn = parse_personality(personality_a, personality_dim=personality_dim)
    b_raw, b_vec, b_warn = parse_personality(personality_b, personality_dim=personality_dim)
    a_vad = clamp_vector(parse_float_sequence(initial_vad_a, 3, "VAD A"), vad_min, vad_max)
    b_vad = clamp_vector(parse_float_sequence(initial_vad_b, 3, "VAD B"), vad_min, vad_max)

    state = ConversationState(
        speaker_a=SpeakerRuntime("A", a_raw, a_vec, a_vad, speaker_a_private_context or ""),
        speaker_b=SpeakerRuntime("B", b_raw, b_vec, b_vad, speaker_b_private_context or ""),
        scenario=scenario,
    )

    debug = {
        "speaker_a_initial_vad": dict(zip(VAD_KEYS, a_vad)),
        "speaker_b_initial_vad": dict(zip(VAD_KEYS, b_vad)),
        "speaker_a_personality_vec": dict(zip(BIG5_KEYS[:personality_dim], a_vec)),
        "speaker_b_personality_vec": dict(zip(BIG5_KEYS[:personality_dim], b_vec)),
        "speaker_a_private_context_status": "stored and only sent when Qwen is generating Speaker A in two-AI mode",
        "speaker_b_private_context_status": "stored and only sent when Qwen is generating Speaker B",
        "warnings": [w for w in [a_warn, b_warn] if w],
    }
    status = "Session initialized. Choose a mode and continue."
    if a_warn or b_warn:
        status += "\n\nWarning: " + " ".join([w for w in [a_warn, b_warn] if w])
    return state, debug, status


def initialize_session(
    personality_a: str,
    initial_vad_a: str,
    personality_b: str,
    initial_vad_b: str,
    scenario: str,
    speaker_a_private_context: str,
    speaker_b_private_context: str,
):
    try:
        state, debug, status = make_state(
            personality_a=personality_a,
            initial_vad_a=initial_vad_a,
            personality_b=personality_b,
            initial_vad_b=initial_vad_b,
            scenario=scenario,
            speaker_a_private_context=speaker_a_private_context,
            speaker_b_private_context=speaker_b_private_context,
            personality_dim=APP_ARGS.personality_dim,
            vad_min=APP_ARGS.vad_min,
            vad_max=APP_ARGS.vad_max,
        )
        return state, [], [], debug, status
    except Exception as exc:
        return None, [], [], {"error": str(exc)}, f"Initialization error: {exc}"


def normalize_gradio_history(chat_history: Any) -> List[Dict[str, str]]:
    """
    Current Gradio in this environment expects messages format:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    Some versions do not accept gr.Chatbot(type="messages") even though they expect
    message dictionaries at runtime. Therefore the UI omits type=..., but callbacks
    still return messages-format history. This helper also converts older tuple-pair
    histories if they appear.
    """
    if not chat_history:
        return []

    # Already messages format.
    if isinstance(chat_history, list) and chat_history and isinstance(chat_history[0], dict):
        clean: List[Dict[str, str]] = []
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in {"user", "assistant", "system"}:
                clean.append({"role": role, "content": str(content)})
        return clean

    # Convert old tuple/list-pair format.
    if isinstance(chat_history, list) and chat_history and isinstance(chat_history[0], (tuple, list)):
        messages: List[Dict[str, str]] = []
        for pair in chat_history:
            if len(pair) >= 1 and pair[0] not in [None, ""]:
                messages.append({"role": "user", "content": str(pair[0])})
            if len(pair) >= 2 and pair[1] not in [None, ""]:
                messages.append({"role": "assistant", "content": str(pair[1])})
        return messages

    return []


def ensure_state(
    state: Optional[ConversationState],
    personality_a: str,
    initial_vad_a: str,
    personality_b: str,
    initial_vad_b: str,
    scenario: str,
    speaker_a_private_context: str,
    speaker_b_private_context: str,
) -> ConversationState:
    if state is None or not getattr(state, "initialized", False):
        state, _, _ = make_state(
            personality_a=personality_a,
            initial_vad_a=initial_vad_a,
            personality_b=personality_b,
            initial_vad_b=initial_vad_b,
            scenario=scenario,
            speaker_a_private_context=speaker_a_private_context,
            speaker_b_private_context=speaker_b_private_context,
            personality_dim=APP_ARGS.personality_dim,
            vad_min=APP_ARGS.vad_min,
            vad_max=APP_ARGS.vad_max,
        )
    return state


def display_with_metadata(text: str, metadata_lines: List[str], show_metadata: bool) -> str:
    if not show_metadata:
        return text
    return text + "\n\n" + "\n".join(["---"] + metadata_lines)


def append_pair(history: List[Dict[str, str]], speaker_a_text: str, speaker_b_text: str, label_a: bool = False, label_b: bool = False) -> None:
    a_content = f"Speaker A: {speaker_a_text}" if label_a else speaker_a_text
    b_content = f"Speaker B: {speaker_b_text}" if label_b else speaker_b_text
    history.append({"role": "user", "content": a_content})
    history.append({"role": "assistant", "content": b_content})


def run_baseline_turn(
    user_message: str,
    main_history: List[Dict[str, str]],
    comparison_history: List[Dict[str, str]],
    state: ConversationState,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_context_turns: int,
    show_metadata: bool,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any], str]:
    if not user_message:
        return main_history, comparison_history, {"warning": "Empty Speaker A message ignored."}, "Empty Speaker A message ignored."

    baseline_messages = build_baseline_messages(
        state=state,
        current_a_text=user_message,
        max_context_turns=max_context_turns,
    )
    b_baseline_response = QWEN.generate(
        messages=baseline_messages,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )

    # Diagnostic/state-continuity scoring only. This does not condition the baseline prompt.
    a_current_vad = VAD.predict(user_message)
    b_baseline_realized_vad = VAD.predict(b_baseline_response)
    state.speaker_a.last_vad = a_current_vad
    state.speaker_b.last_vad = b_baseline_realized_vad
    state.turns.append(
        DialogueTurn(
            a_text=user_message,
            a_vad=a_current_vad,
            b_target_vad=b_baseline_realized_vad,
            b_text=b_baseline_response,
            b_realized_vad=b_baseline_realized_vad,
            b_baseline_text=b_baseline_response,
            b_baseline_realized_vad=b_baseline_realized_vad,
            mode=MODE_BASELINE,
        )
    )

    display = display_with_metadata(
        b_baseline_response,
        [
            f"A utterance VAD diagnostic: {format_vad(a_current_vad)}",
            f"B realized VAD diagnostic: {format_vad(b_baseline_realized_vad)}",
            "Prompt condition: pure baseline; no explicit VAD hint and no GRU target used for generation.",
        ],
        show_metadata,
    )
    append_pair(main_history, user_message, display)

    debug = {
        "mode": MODE_BASELINE,
        "turn_index": len(state.turns),
        "speaker_a_vad_diagnostic": dict(zip(VAD_KEYS, a_current_vad)),
        "speaker_b_realized_vad_diagnostic": dict(zip(VAD_KEYS, b_baseline_realized_vad)),
        "generation_condition": "baseline prompt only; VAD/GRU not used to condition generation",
    }
    status = f"Baseline turn {len(state.turns)} complete. Diagnostic B state: {format_vad(state.speaker_b.last_vad)}"
    return main_history, comparison_history, debug, status


def run_emotion_supported_turn(
    user_message: str,
    main_history: List[Dict[str, str]],
    comparison_history: List[Dict[str, str]],
    state: ConversationState,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_context_turns: int,
    show_metadata: bool,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any], str]:
    if not user_message:
        return main_history, comparison_history, {"warning": "Empty Speaker A message ignored."}, "Empty Speaker A message ignored."

    # 1. Regress current Speaker A VAD from the user's utterance.
    a_current_vad = VAD.predict(user_message)

    # 2. Predict Speaker B's target VAD using B's recurrent state.
    b_target_vad, h_b_next = GRU_DYNAMICS.step_responder_vad(
        prev_self_vad=state.speaker_b.last_vad,
        other_vad=a_current_vad,
        self_personality_vec=state.speaker_b.personality_vec,
        h_prev=state.h_b,
    )

    # 3. Generate Speaker B response conditioned on target B VAD.
    messages = build_generation_messages(
        state=state,
        current_a_text=user_message,
        b_target_vad=b_target_vad,
        max_context_turns=max_context_turns,
    )
    b_response = QWEN.generate(
        messages=messages,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )

    # 4. Score generated B text for diagnostics only.
    b_realized_vad = VAD.predict(b_response)

    # 5. Keep Speaker A's recurrent latent state coherent, using B's GRU target.
    _, h_a_next = GRU_DYNAMICS.step_responder_vad(
        prev_self_vad=state.speaker_a.last_vad,
        other_vad=b_target_vad,
        self_personality_vec=state.speaker_a.personality_vec,
        h_prev=state.h_a,
    )

    # State update rule: A = regressed user utterance, B = GRU target, not realized VAD.
    state.speaker_a.last_vad = a_current_vad
    state.speaker_b.last_vad = b_target_vad
    state.h_a = h_a_next.detach() if h_a_next is not None else None
    state.h_b = h_b_next.detach() if h_b_next is not None else None
    state.turns.append(
        DialogueTurn(
            a_text=user_message,
            a_vad=a_current_vad,
            b_target_vad=b_target_vad,
            b_text=b_response,
            b_realized_vad=b_realized_vad,
            mode=MODE_EMOTION,
        )
    )

    display = display_with_metadata(
        b_response,
        [
            f"A utterance VAD: {format_vad(a_current_vad)}",
            f"B target VAD from GRU: {format_vad(b_target_vad)}",
            f"B realized VAD from regressor: {format_vad(b_realized_vad)}",
            "State update: B uses GRU target, not realized VAD.",
        ],
        show_metadata,
    )
    append_pair(main_history, user_message, display)

    debug = {
        "mode": MODE_EMOTION,
        "turn_index": len(state.turns),
        "speaker_a_predicted_vad": dict(zip(VAD_KEYS, a_current_vad)),
        "speaker_b_target_vad_from_gru": dict(zip(VAD_KEYS, b_target_vad)),
        "speaker_b_realized_vad_from_regressor": dict(zip(VAD_KEYS, b_realized_vad)),
        "speaker_a_personality_vec": dict(zip(BIG5_KEYS[: APP_ARGS.personality_dim], state.speaker_a.personality_vec)),
        "speaker_b_personality_vec": dict(zip(BIG5_KEYS[: APP_ARGS.personality_dim], state.speaker_b.personality_vec)),
    }
    status = f"Emotion-supported turn {len(state.turns)} complete. Current B state follows GRU target: {format_vad(state.speaker_b.last_vad)}"
    return main_history, comparison_history, debug, status


def run_arena_turn(
    user_message: str,
    vad_history: List[Dict[str, str]],
    baseline_history: List[Dict[str, str]],
    state: ConversationState,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_context_turns: int,
    show_metadata: bool,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any], str]:
    if not user_message:
        return vad_history, baseline_history, {"warning": "Empty Speaker A message ignored."}, "Empty Speaker A message ignored."

    a_current_vad = VAD.predict(user_message)
    b_target_vad, h_b_next = GRU_DYNAMICS.step_responder_vad(
        prev_self_vad=state.speaker_b.last_vad,
        other_vad=a_current_vad,
        self_personality_vec=state.speaker_b.personality_vec,
        h_prev=state.h_b,
    )

    messages = build_generation_messages(
        state=state,
        current_a_text=user_message,
        b_target_vad=b_target_vad,
        max_context_turns=max_context_turns,
    )
    b_response = QWEN.generate(
        messages=messages,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )

    baseline_messages = build_baseline_messages(
        state=state,
        current_a_text=user_message,
        max_context_turns=max_context_turns,
    )
    b_baseline_response = QWEN.generate(
        messages=baseline_messages,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )

    b_realized_vad = VAD.predict(b_response)
    b_baseline_realized_vad = VAD.predict(b_baseline_response)

    _, h_a_next = GRU_DYNAMICS.step_responder_vad(
        prev_self_vad=state.speaker_a.last_vad,
        other_vad=b_target_vad,
        self_personality_vec=state.speaker_a.personality_vec,
        h_prev=state.h_a,
    )

    state.speaker_a.last_vad = a_current_vad
    state.speaker_b.last_vad = b_target_vad
    state.h_a = h_a_next.detach() if h_a_next is not None else None
    state.h_b = h_b_next.detach() if h_b_next is not None else None
    state.turns.append(
        DialogueTurn(
            a_text=user_message,
            a_vad=a_current_vad,
            b_target_vad=b_target_vad,
            b_text=b_response,
            b_realized_vad=b_realized_vad,
            b_baseline_text=b_baseline_response,
            b_baseline_realized_vad=b_baseline_realized_vad,
            mode=MODE_ARENA,
        )
    )

    vad_display_response = display_with_metadata(
        b_response,
        [
            f"A utterance VAD: {format_vad(a_current_vad)}",
            f"B target VAD from GRU: {format_vad(b_target_vad)}",
            f"B realized VAD from regressor: {format_vad(b_realized_vad)}",
            "State update: B uses GRU target, not realized VAD.",
        ],
        show_metadata,
    )

    baseline_display_response = display_with_metadata(
        b_baseline_response,
        [
            f"A utterance VAD: {format_vad(a_current_vad)}",
            f"Baseline B realized VAD from regressor: {format_vad(b_baseline_realized_vad)}",
            "Prompt condition: no explicit VAD hint.",
        ],
        show_metadata,
    )

    append_pair(vad_history, user_message, vad_display_response)
    append_pair(baseline_history, user_message, baseline_display_response)

    debug = {
        "mode": MODE_ARENA,
        "turn_index": len(state.turns),
        "speaker_a_predicted_vad": dict(zip(VAD_KEYS, a_current_vad)),
        "speaker_b_target_vad_from_gru": dict(zip(VAD_KEYS, b_target_vad)),
        "speaker_b_realized_vad_from_regressor": dict(zip(VAD_KEYS, b_realized_vad)),
        "baseline_b_realized_vad_from_regressor": dict(zip(VAD_KEYS, b_baseline_realized_vad)),
        "speaker_a_personality_vec": dict(zip(BIG5_KEYS[: APP_ARGS.personality_dim], state.speaker_a.personality_vec)),
        "speaker_b_personality_vec": dict(zip(BIG5_KEYS[: APP_ARGS.personality_dim], state.speaker_b.personality_vec)),
    }
    status = f"Arena turn {len(state.turns)} complete. Current B state follows GRU target: {format_vad(state.speaker_b.last_vad)}"
    return vad_history, baseline_history, debug, status


def run_two_ai_turn(
    director_note: str,
    main_history: List[Dict[str, str]],
    comparison_history: List[Dict[str, str]],
    state: ConversationState,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_context_turns: int,
    show_metadata: bool,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any], str]:
    """
    Two-AI mode advances one full A->B exchange.

    Variable discipline:
        - a_target_vad is produced by Speaker A's GRU state using Speaker B's current state.
        - b_target_vad is produced by Speaker B's GRU state using a_target_vad.
        - a_realized_vad and b_realized_vad are diagnostics only.
        - Speaker A generation receives only Speaker A private context.
        - Speaker B generation receives only Speaker B private context.
    """
    director_note = (director_note or "").strip()

    # 1. Predict Speaker A's next target affect from A's recurrent state and B's current GRU state.
    a_target_vad, h_a_next = GRU_DYNAMICS.step_responder_vad(
        prev_self_vad=state.speaker_a.last_vad,
        other_vad=state.speaker_b.last_vad,
        self_personality_vec=state.speaker_a.personality_vec,
        h_prev=state.h_a,
    )

    if state.turns:
        current_b_text = state.turns[-1].b_text
    else:
        current_b_text = ""

    # 2. Generate Speaker A. A sees A's own private context only.
    a_messages = build_speaker_messages(
        state=state,
        speaker_id="A",
        current_other_text=current_b_text,
        target_vad=a_target_vad,
        max_context_turns=max_context_turns,
        use_vad_hint=True,
        director_note=director_note,
    )
    a_response = QWEN.generate(
        messages=a_messages,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )
    a_realized_vad = VAD.predict(a_response)

    # 3. Predict Speaker B's target affect from B's recurrent state and A's GRU target.
    #    We intentionally use a_target_vad, not a_realized_vad, to keep recurrence fully GRU-driven.
    b_target_vad, h_b_next = GRU_DYNAMICS.step_responder_vad(
        prev_self_vad=state.speaker_b.last_vad,
        other_vad=a_target_vad,
        self_personality_vec=state.speaker_b.personality_vec,
        h_prev=state.h_b,
    )

    # 4. Generate Speaker B. B sees B's own private context only.
    b_messages = build_speaker_messages(
        state=state,
        speaker_id="B",
        current_other_text=a_response,
        target_vad=b_target_vad,
        max_context_turns=max_context_turns,
        use_vad_hint=True,
        director_note=director_note,
    )
    b_response = QWEN.generate(
        messages=b_messages,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )
    b_realized_vad = VAD.predict(b_response)

    # 5. Authoritative recurrent states are the GRU targets, not realized diagnostics.
    state.speaker_a.last_vad = a_target_vad
    state.speaker_b.last_vad = b_target_vad
    state.h_a = h_a_next.detach() if h_a_next is not None else None
    state.h_b = h_b_next.detach() if h_b_next is not None else None
    state.turns.append(
        DialogueTurn(
            a_text=a_response,
            a_vad=a_target_vad,
            b_target_vad=b_target_vad,
            b_text=b_response,
            b_realized_vad=b_realized_vad,
            a_target_vad=a_target_vad,
            a_realized_vad=a_realized_vad,
            mode=MODE_TWO_AI,
        )
    )

    a_display = display_with_metadata(
        a_response,
        [
            f"A target VAD from A-GRU: {format_vad(a_target_vad)}",
            f"A realized VAD from regressor: {format_vad(a_realized_vad)}",
            "A private context used: yes; B private context used for A: no.",
        ],
        show_metadata,
    )
    b_display = display_with_metadata(
        b_response,
        [
            f"B target VAD from B-GRU: {format_vad(b_target_vad)}",
            f"B realized VAD from regressor: {format_vad(b_realized_vad)}",
            "B private context used: yes; A private context used for B: no.",
            "State update: both speakers use GRU targets, not realized VAD.",
        ],
        show_metadata,
    )

    append_pair(main_history, a_display, b_display, label_a=True, label_b=True)

    debug = {
        "mode": MODE_TWO_AI,
        "turn_index": len(state.turns),
        "speaker_a_target_vad_from_a_gru": dict(zip(VAD_KEYS, a_target_vad)),
        "speaker_a_realized_vad_from_regressor": dict(zip(VAD_KEYS, a_realized_vad)),
        "speaker_b_target_vad_from_b_gru": dict(zip(VAD_KEYS, b_target_vad)),
        "speaker_b_realized_vad_from_regressor": dict(zip(VAD_KEYS, b_realized_vad)),
        "state_update_rule": "A.last_vad=a_target_vad, B.last_vad=b_target_vad; realized VAD is diagnostic only",
        "privacy_rule": {
            "speaker_a_generation_received_private_context": "A only",
            "speaker_b_generation_received_private_context": "B only",
        },
        "director_note": director_note,
    }
    status = (
        f"Two-AI turn {len(state.turns)} complete. "
        f"A state: {format_vad(state.speaker_a.last_vad)} | "
        f"B state: {format_vad(state.speaker_b.last_vad)}"
    )
    return main_history, comparison_history, debug, status


def run_one_turn(
    app_mode: str,
    user_message: str,
    main_chat_history: Any,
    comparison_chat_history: Any,
    state: Optional[ConversationState],
    personality_a: str,
    initial_vad_a: str,
    personality_b: str,
    initial_vad_b: str,
    scenario: str,
    speaker_a_private_context: str,
    speaker_b_private_context: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_context_turns: int,
    show_metadata: bool,
):
    user_message = (user_message or "").strip()
    app_mode = app_mode or MODE_ARENA
    main_chat_history = normalize_gradio_history(main_chat_history)
    comparison_chat_history = normalize_gradio_history(comparison_chat_history)

    try:
        state = ensure_state(
            state=state,
            personality_a=personality_a,
            initial_vad_a=initial_vad_a,
            personality_b=personality_b,
            initial_vad_b=initial_vad_b,
            scenario=scenario,
            speaker_a_private_context=speaker_a_private_context,
            speaker_b_private_context=speaker_b_private_context,
        )

        if app_mode == MODE_BASELINE:
            main_chat_history, comparison_chat_history, debug, status = run_baseline_turn(
                user_message=user_message,
                main_history=main_chat_history,
                comparison_history=comparison_chat_history,
                state=state,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                max_context_turns=max_context_turns,
                show_metadata=show_metadata,
            )
        elif app_mode == MODE_EMOTION:
            main_chat_history, comparison_chat_history, debug, status = run_emotion_supported_turn(
                user_message=user_message,
                main_history=main_chat_history,
                comparison_history=comparison_chat_history,
                state=state,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                max_context_turns=max_context_turns,
                show_metadata=show_metadata,
            )
        elif app_mode == MODE_ARENA:
            main_chat_history, comparison_chat_history, debug, status = run_arena_turn(
                user_message=user_message,
                vad_history=main_chat_history,
                baseline_history=comparison_chat_history,
                state=state,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                max_context_turns=max_context_turns,
                show_metadata=show_metadata,
            )
        elif app_mode == MODE_TWO_AI:
            main_chat_history, comparison_chat_history, debug, status = run_two_ai_turn(
                director_note=user_message,
                main_history=main_chat_history,
                comparison_history=comparison_chat_history,
                state=state,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                max_context_turns=max_context_turns,
                show_metadata=show_metadata,
            )
        else:
            debug = {"error": f"Unknown mode: {app_mode}"}
            status = f"Unknown mode: {app_mode}"

        return "", main_chat_history, comparison_chat_history, state, debug, status

    except Exception as exc:
        tb = traceback.format_exc()
        err = f"Runtime error: {type(exc).__name__}: {exc}\n\nTraceback:\n{tb}"
        debug = {"error": str(exc), "exception_type": type(exc).__name__, "traceback": tb}
        if app_mode == MODE_TWO_AI:
            main_chat_history.append({"role": "assistant", "content": err})
        else:
            main_chat_history.append({"role": "user", "content": user_message})
            main_chat_history.append({"role": "assistant", "content": err})
            if app_mode == MODE_ARENA:
                comparison_chat_history.append({"role": "user", "content": user_message})
                comparison_chat_history.append({"role": "assistant", "content": err})
        return "", main_chat_history, comparison_chat_history, state, debug, err


def mode_help_text(app_mode: str) -> str:
    if app_mode == MODE_BASELINE:
        return (
            "**Pure baseline mode.** Human plays Speaker A; Qwen plays Speaker B with only scenario/personality/history. "
            "No explicit VAD hint or GRU target is used for generation."
        )
    if app_mode == MODE_EMOTION:
        return (
            "**Emotion-supported mode.** Human plays Speaker A; Speaker A utterance is scored by the VAD regressor, "
            "the GRU predicts Speaker B's target VAD, and Qwen answers as Speaker B using that target."
        )
    if app_mode == MODE_ARENA:
        return (
            "**Arena comparison.** The left panel is GRU/VAD-conditioned. The right panel is the baseline with the same public setup/history, "
            "but no explicit VAD hint."
        )
    if app_mode == MODE_TWO_AI:
        return (
            "**Two-AI mode.** Qwen generates both Speaker A and Speaker B. Each side has its own GRU hidden state, last VAD, personality vector, "
            "and private context. The text box becomes an optional director note, not Speaker A dialogue."
        )
    return "Choose a mode."


def update_mode_ui(app_mode: str):
    help_md = mode_help_text(app_mode)
    if app_mode == MODE_ARENA:
        return (
            help_md,
            gr.update(label="Model A — GRU/VAD-conditioned Speaker B", visible=True),
            gr.update(label="Model B — Baseline, no VAD hint", visible=True),
            gr.update(label="Speaker A input", placeholder="Type Speaker A's next utterance and press Enter..."),
            gr.update(value="Send as Speaker A"),
        )
    if app_mode == MODE_TWO_AI:
        return (
            help_md,
            gr.update(label="Two-AI transcript: Speaker A ↔ Speaker B", visible=True),
            gr.update(label="Comparison panel", visible=False),
            gr.update(label="Optional director note", placeholder="Optional: steer the next autonomous exchange. Leave blank to just continue."),
            gr.update(value="Advance one AI↔AI turn"),
        )
    if app_mode == MODE_BASELINE:
        return (
            help_md,
            gr.update(label="Pure baseline transcript", visible=True),
            gr.update(label="Comparison panel", visible=False),
            gr.update(label="Speaker A input", placeholder="Type Speaker A's next utterance and press Enter..."),
            gr.update(value="Send as Speaker A"),
        )
    return (
        help_md,
        gr.update(label="Emotion-supported transcript", visible=True),
        gr.update(label="Comparison panel", visible=False),
        gr.update(label="Speaker A input", placeholder="Type Speaker A's next utterance and press Enter..."),
        gr.update(value="Send as Speaker A"),
    )


# -----------------------------
# UI
# -----------------------------


def build_ui(args: argparse.Namespace):
    midpoint = (args.vad_min + args.vad_max) / 2.0
    default_vad = f"{midpoint:.3f}, {midpoint:.3f}, {midpoint:.3f}"

    with gr.Blocks(title="Emotion-Dynamic Qwen Chatbot") as demo:
        gr.Markdown(
            "# Emotion-Dynamic Qwen Chatbot\n"
            "Four modes: pure baseline, emotion-supported, arena comparison, and two-AI autonomous dialogue."
        )

        state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Session setup")
                app_mode = gr.Radio(
                    choices=APP_MODES,
                    value=MODE_ARENA,
                    label="Page mode",
                )
                mode_help = gr.Markdown(mode_help_text(MODE_ARENA))
                scenario = gr.Textbox(
                    label="Shared scenario / public dialogue context",
                    lines=5,
                    placeholder="Information both speakers can know. Example: A tense family conversation about unpaid school tuition...",
                )
                speaker_a_private_context = gr.Textbox(
                    label="Speaker A private context — visible only to Speaker A",
                    lines=4,
                    placeholder=(
                        "Private information for Speaker A. In two-AI mode, this is sent only when Qwen generates Speaker A. "
                        "It is never sent to Speaker B."
                    ),
                )
                speaker_b_private_context = gr.Textbox(
                    label="Speaker B private context — visible only to Speaker B",
                    lines=4,
                    placeholder=(
                        "Private information Speaker B knows but Speaker A should not know. "
                        "This is sent only when Qwen generates Speaker B."
                    ),
                )
                personality_a = gr.Textbox(
                    label="Speaker A personality / Big Five vector",
                    lines=5,
                    placeholder=(
                        "Raw personality text, or numeric Big Five vector.\n"
                        "Example: {\"openness\":0.6,\"conscientiousness\":0.4,\"extraversion\":0.3,\"agreeableness\":0.5,\"neuroticism\":0.8}"
                    ),
                )
                initial_vad_a = gr.Textbox(
                    label="Speaker A initial VAD",
                    value=default_vad,
                    placeholder="valence, arousal, dominance",
                )
                personality_b = gr.Textbox(
                    label="Speaker B personality / Big Five vector",
                    lines=5,
                    placeholder="Raw personality text, JSON Big Five dict, or 5 comma-separated floats.",
                )
                initial_vad_b = gr.Textbox(
                    label="Speaker B initial VAD",
                    value=default_vad,
                    placeholder="valence, arousal, dominance",
                )

                with gr.Accordion("Generation settings", open=False):
                    max_new_tokens = gr.Slider(16, 512, value=args.max_new_tokens, step=1, label="Max new tokens")
                    temperature = gr.Slider(0.0, 1.5, value=args.temperature, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=args.top_p, step=0.05, label="Top-p")
                    max_context_turns = gr.Slider(0, 20, value=args.max_context_turns, step=1, label="Context turns sent to Qwen")
                    show_metadata = gr.Checkbox(value=True, label="Show VAD metadata in chat")

                start_btn = gr.Button("Start / Reset session", variant="primary")
                status = gr.Markdown("Not initialized yet.")
                debug = gr.JSON(label="Latest affect state / debug")

            with gr.Column(scale=2):
                gr.Markdown("## Dialogue workspace")
                with gr.Row():
                    with gr.Column():
                        main_chatbot = gr.Chatbot(label="Model A — GRU/VAD-conditioned Speaker B", height=650)
                    with gr.Column():
                        comparison_chatbot = gr.Chatbot(label="Model B — Baseline, no VAD hint", height=650)

                user_box = gr.Textbox(
                    label="Speaker A input",
                    placeholder="Type Speaker A's next utterance and press Enter...",
                    lines=3,
                )
                send_btn = gr.Button("Send as Speaker A")

        app_mode.change(
            fn=update_mode_ui,
            inputs=[app_mode],
            outputs=[mode_help, main_chatbot, comparison_chatbot, user_box, send_btn],
        )

        start_btn.click(
            fn=initialize_session,
            inputs=[personality_a, initial_vad_a, personality_b, initial_vad_b, scenario, speaker_a_private_context, speaker_b_private_context],
            outputs=[state, main_chatbot, comparison_chatbot, debug, status],
        )

        send_inputs = [
            app_mode,
            user_box,
            main_chatbot,
            comparison_chatbot,
            state,
            personality_a,
            initial_vad_a,
            personality_b,
            initial_vad_b,
            scenario,
            speaker_a_private_context,
            speaker_b_private_context,
            max_new_tokens,
            temperature,
            top_p,
            max_context_turns,
            show_metadata,
        ]
        send_outputs = [user_box, main_chatbot, comparison_chatbot, state, debug, status]

        user_box.submit(fn=run_one_turn, inputs=send_inputs, outputs=send_outputs)
        send_btn.click(fn=run_one_turn, inputs=send_inputs, outputs=send_outputs)

    return demo


# -----------------------------
# CLI args and startup
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # VAD regressor
    parser.add_argument("--vad_model_path", type=str, required=True, help="HF save_pretrained directory for VAD regressor")
    parser.add_argument("--vad_tokenizer_path", type=str, default=None, help="Optional tokenizer path; defaults to vad_model_path")
    parser.add_argument("--vad_max_length", type=int, default=256)
    parser.add_argument("--vad_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vad_output_activation", choices=["none", "sigmoid", "tanh", "tanh_0_1"], default="none")
    parser.add_argument("--vad_min", type=float, default=-1.0, help="Minimum VAD value after clamping")
    parser.add_argument("--vad_max", type=float, default=1.0, help="Maximum VAD value after clamping")

    # GRU dynamics model
    parser.add_argument("--gru_checkpoint", type=str, required=True)
    parser.add_argument("--gru_model_class", type=str, default=None, help="Example: GRU_full_encoder:EmotionDynamicsGRU")
    parser.add_argument("--gru_model_kwargs", type=str, default="{}", help="JSON kwargs for constructing GRU model")
    parser.add_argument("--gru_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--gru_forward_mode",
        choices=["native_step", "concat_seq", "concat_flat", "kwargs"],
        default="native_step",
        help="native_step uses EmotionStateGRU.step(); old concat modes are kept only for compatibility.",
    )
    parser.add_argument("--gru_output_activation", choices=["none", "sigmoid", "tanh", "tanh_0_1"], default="none")
    parser.add_argument("--gru_torchscript", action="store_true")
    parser.add_argument("--gru_strict_load", action="store_true")
    parser.add_argument("--personality_dim", type=int, default=5)

    # Qwen model
    parser.add_argument("--qwen_model_path", type=str, required=True, help="Local path or HF id for Qwen model")
    parser.add_argument("--qwen_dtype", type=str, default="auto", help="auto, float16, bfloat16, float32")
    parser.add_argument("--qwen_device_map", type=str, default="auto", help="auto, cuda, cpu, balanced, or none")
    parser.add_argument("--qwen_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Used only when --qwen_device_map none")
    parser.add_argument("--qwen_max_input_tokens", type=int, default=4096)
    parser.add_argument("--enable_thinking", action="store_true", help="Pass enable_thinking=True to supported Qwen chat templates")
    parser.add_argument("--strip_thinking", action="store_true", default=True, help="Strip <think>...</think> from visible response")
    parser.add_argument("--no_strip_thinking", dest="strip_thinking", action="store_false")

    # Generation/UI defaults
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_context_turns", type=int, default=8)
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")

    return parser.parse_args()


APP_ARGS: argparse.Namespace
VAD: VADRegressorAdapter
GRU_DYNAMICS: GRUDynamicsAdapter
QWEN: QwenGenerator


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    APP_ARGS = parse_args()

    print("Loading VAD regressor...")
    VAD = VADRegressorAdapter(APP_ARGS)

    print("Loading GRU dynamics model...")
    GRU_DYNAMICS = GRUDynamicsAdapter(APP_ARGS)

    print("Loading Qwen model...")
    QWEN = QwenGenerator(APP_ARGS)

    print("Launching Gradio UI...")
    demo = build_ui(APP_ARGS)
    demo.launch(
        server_name=APP_ARGS.server_name,
        server_port=APP_ARGS.server_port,
        share=APP_ARGS.share,
    )
