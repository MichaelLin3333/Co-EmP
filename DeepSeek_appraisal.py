import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# ============================================================
# 1. Appraisal dimensions
# ============================================================

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


# ============================================================
# 2. Prompt
# ============================================================

SYSTEM_PROMPT = """
You are an expert affective-computing annotator.

Your task is to label one listener-relative appraisal vector for each dialogue turn.

For each turn:
- The speaker is the person who says the current utterance.
- The listener is the other person.
- The appraisal vector describes how the listener interprets/appraises the speaker's utterance in context.

Do not label the speaker's self-appraisal.
Do not label the emotion expressed by the utterance.
Do not merely copy VAD.

The appraisal vector should explain the semantic-cognitive reason why the listener's emotion may change after hearing the utterance.

All values must be floats from 0.0 to 1.0.

Dimensions:

1. personal_relevance:
How personally important this utterance is to the listener.
0.0 = irrelevant; 1.0 = extremely personally important.

2. goal_conduciveness:
Whether this utterance helps or harms the listener's goals, needs, identity, safety, self-image, or relationship.
0.0 = strongly harmful/obstructive; 1.0 = strongly helpful/supportive.

3. unexpectedness:
How surprising, shocking, or unforeseen this utterance is to the listener.
Usually for new information, betrayal, accusation, threat, confession, or sudden reversal.
0.0 = completely expected; 1.0 = extremely unexpected.

4. agency_self:
How much the listener sees themself as responsible for the situation.
0.0 = no self-responsibility; 1.0 = mostly self-caused.

5. agency_other:
How much the listener sees the speaker as responsible for the situation.
o not set agency_other high merely because the speaker said the utterance. Set agency_other high only when the listener would attribute meaningful responsibility, blame, or credit to the speaker for the situation.
0.0 = speaker not responsible; 1.0 = speaker highly responsible.

6. controllability:
How much the listener feels they can influence or control the situation after hearing this.
0.0 = uncontrollable; 1.0 = highly controllable.

7. norm_violation:
How much the utterance violates moral, social, personal, or relationship norms from the listener's perspective.
0.0 = no violation; 1.0 = severe violation.

8. relationship_impact:
How strongly this utterance affects the relationship between speaker and listener.
0.0 = no relationship impact; 1.0 = very strong relationship impact.

Important:
- Use the scenario, personas, Big Five traits, speaker backgrounds, dialogue history, and current utterance.
- Values should be nuanced continuous values.
- Avoid using only 0.0, 0.5, and 1.0.
- Return valid JSON only.
"""


def build_user_prompt(sample: Dict[str, Any]) -> str:
    """
    Builds the user prompt for one JSONL sample.
    The model labels all turns in the dialogue at once.
    """

    compact_sample = {
        "scenario": sample.get("scenario", {}),
        "personas": sample.get("personas", {}),
        "dialogue": sample.get("dialogue", []),
    }

    expected_schema = {
        "turn_appraisals": [
            {
                "turn_index": 0,
                "speaker": "A",
                "listener": "B",
                "listener_appraisal": {
                    dim: 0.5 for dim in APPRAISAL_DIMS
                },
            }
        ]
    }

    prompt = f"""
Generate listener-relative appraisal labels for the following dialogue.

Output must be valid JSON.

The output must follow this exact structure:

{json.dumps(expected_schema, indent=2)}

Rules:
- Include exactly one item in "turn_appraisals" for every dialogue turn.
- turn_index must match the index in the dialogue list, starting from 0.
- speaker must match the original turn speaker.
- listener must be the other speaker.
- If speaker is "A", listener must be "B".
- If speaker is "B", listener must be "A".
- Every appraisal dimension must be present.
- Every appraisal value must be a float from 0.0 to 1.0.
- Do not include explanations.
- Do not include markdown.
- Do not include any text outside the JSON object.

Dialogue sample:

{json.dumps(compact_sample, ensure_ascii=False, indent=2)}
""".strip()

    return prompt


# ============================================================
# 3. JSON parsing and validation
# ============================================================

def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Parses JSON from the model response.
    Handles accidental markdown fences.
    """

    if text is None:
        raise ValueError("Model response content is None.")

    text = text.strip()

    # Remove markdown JSON fences if present.
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    # Try direct JSON parsing.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON object found in model response.")

    candidate = text[start:end + 1]
    return json.loads(candidate)


def is_float_0_1(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def validate_appraisal_vector(vec: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(vec, dict):
        return False, "Appraisal vector is not a dictionary."

    for dim in APPRAISAL_DIMS:
        if dim not in vec:
            return False, f"Missing dimension: {dim}"

        if not is_float_0_1(vec[dim]):
            return False, f"Invalid value for {dim}: {vec[dim]}"

    return True, ""


def validate_response(parsed: Dict[str, Any], sample: Dict[str, Any]) -> Tuple[bool, str]:
    dialogue = sample.get("dialogue", [])
    appraisals = parsed.get("turn_appraisals")

    if not isinstance(dialogue, list):
        return False, "Input sample dialogue is not a list."

    if not isinstance(appraisals, list):
        return False, "Response missing valid 'turn_appraisals' list."

    if len(appraisals) != len(dialogue):
        return False, f"Expected {len(dialogue)} appraisals, got {len(appraisals)}."

    for i, item in enumerate(appraisals):
        if not isinstance(item, dict):
            return False, f"Turn appraisal {i} is not a dictionary."

        expected_speaker = dialogue[i].get("speaker")

        if expected_speaker not in ["A", "B"]:
            return False, f"Invalid original speaker at turn {i}: {expected_speaker}"

        expected_listener = "B" if expected_speaker == "A" else "A"

        if item.get("turn_index") != i:
            return False, f"Invalid turn_index at item {i}: {item.get('turn_index')}"

        if item.get("speaker") != expected_speaker:
            return False, (
                f"Invalid speaker at turn {i}: "
                f"expected {expected_speaker}, got {item.get('speaker')}"
            )

        if item.get("listener") != expected_listener:
            return False, (
                f"Invalid listener at turn {i}: "
                f"expected {expected_listener}, got {item.get('listener')}"
            )

        if "listener_appraisal" not in item:
            return False, f"Missing listener_appraisal at turn {i}"

        ok, msg = validate_appraisal_vector(item["listener_appraisal"])
        if not ok:
            return False, f"Invalid listener_appraisal at turn {i}: {msg}"

    return True, ""


# ============================================================
# 4. Attach labels to original sample
# ============================================================

def attach_appraisals(sample: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds listener_appraisal directly into each dialogue turn.

    Output turn format:

    {
      "speaker": "A",
      "text": "...",
      "vad": {...},
      "listener_appraisal": {
        "target": "B",
        "personal_relevance": ...,
        ...
      }
    }
    """

    output = dict(sample)
    output["dialogue"] = [dict(turn) for turn in sample["dialogue"]]

    for item in parsed["turn_appraisals"]:
        i = item["turn_index"]
        listener = item["listener"]
        appraisal = item["listener_appraisal"]

        output["dialogue"][i]["listener_appraisal"] = {
            "target": listener,
            **{
                dim: float(appraisal[dim])
                for dim in APPRAISAL_DIMS
            },
        }

    return output


# ============================================================
# 5. DeepSeek API call
# ============================================================

def call_deepseek(
    client: OpenAI,
    sample: Dict[str, Any],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    max_retries: int = 3,
    sleep_seconds: float = 2.0,
    use_thinking: bool = False,
    use_json_mode: bool = True,
) -> Dict[str, Any]:
    """
    Calls DeepSeek and returns validated parsed JSON.
    """

    user_prompt = build_user_prompt(sample)

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            if use_thinking:
                kwargs["reasoning_effort"] = "high"
                kwargs["extra_body"] = {
                    "thinking": {
                        "type": "enabled"
                    }
                }

            response = client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content
            parsed = extract_json_object(content)

            ok, msg = validate_response(parsed, sample)
            if not ok:
                raise ValueError(f"Validation failed: {msg}")

            return parsed

        except Exception as e:
            last_error = e
            print(f"[Retry {attempt}/{max_retries}] {repr(e)}")

            if attempt < max_retries:
                time.sleep(sleep_seconds * attempt)

    raise RuntimeError(
        f"DeepSeek call failed after {max_retries} retries: {repr(last_error)}"
    )


# ============================================================
# 6. Resume logic
# ============================================================

def count_existing_lines(path: Path) -> int:
    """
    Counts non-empty output lines for resume mode.
    If the output already has N lines, the script skips the first N input lines.
    """

    if not path.exists():
        return 0

    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1

    return count


# ============================================================
# 7. Main JSONL processing
# ============================================================

def process_jsonl(
    input_path: Path,
    output_path: Path,
    failed_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    resume: bool = True,
    max_samples: Optional[int] = None,
    use_thinking: bool = False,
    use_json_mode: bool = True,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> None:
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    skip_count = count_existing_lines(output_path) if resume else 0

    print("========== Synthetic Listener Appraisal Labeling ==========")
    print(f"Input file:      {input_path}")
    print(f"Output file:     {output_path}")
    print(f"Failed file:     {failed_path}")
    print(f"Model:           {model}")
    print(f"Base URL:        {base_url}")
    print(f"Resume:          {resume}")
    print(f"Skipping lines:  {skip_count}")
    print(f"Max samples:     {max_samples}")
    print(f"Thinking mode:   {use_thinking}")
    print(f"JSON mode:       {use_json_mode}")
    print("==========================================================")
    print()

    processed = 0
    written = 0
    failed = 0

    output_mode = "a" if resume else "w"
    failed_mode = "a" if resume else "w"

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open(output_mode, encoding="utf-8") as fout, \
         failed_path.open(failed_mode, encoding="utf-8") as ffailed:

        for line_idx, line in enumerate(fin):
            if not line.strip():
                continue

            if line_idx < skip_count:
                continue

            if max_samples is not None and processed >= max_samples:
                break

            processed += 1

            try:
                sample = json.loads(line)

                if "dialogue" not in sample or not isinstance(sample["dialogue"], list):
                    raise ValueError("Sample missing valid dialogue list.")

                parsed = call_deepseek(
                    client=client,
                    sample=sample,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    use_thinking=use_thinking,
                    use_json_mode=use_json_mode,
                )

                labeled_sample = attach_appraisals(sample, parsed)

                fout.write(json.dumps(labeled_sample, ensure_ascii=False) + "\n")
                fout.flush()

                written += 1
                print(f"[OK] input_line={line_idx} written={written}")

            except Exception as e:
                failed += 1
                print(f"[FAILED] input_line={line_idx} error={repr(e)}")

                fail_record = {
                    "line_idx": line_idx,
                    "error": repr(e),
                    "raw_line": line.strip(),
                }

                ffailed.write(json.dumps(fail_record, ensure_ascii=False) + "\n")
                ffailed.flush()

    print()
    print("========== Done ==========")
    print(f"New samples processed: {processed}")
    print(f"Successfully written:  {written}")
    print(f"Failed:                {failed}")


# ============================================================
# 8. CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate listener-relative appraisal vectors for dialogue JSONL data."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file path.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path with listener_appraisal attached.",
    )

    parser.add_argument(
        "--failed",
        default="failed_listener_appraisal.jsonl",
        help="JSONL file for failed samples.",
    )

    parser.add_argument(
        "--api-key",
        default='sk-eaa79f0adff64bc7a086de4bf62cc9bd',
        help="DeepSeek API key. Prefer using DEEPSEEK_API_KEY environment variable.",
    )

    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="DeepSeek OpenAI-compatible base URL.",
    )

    parser.add_argument(
        "--model",
        default="deepseek-v4-flash",
        help="Model name.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of new samples to process.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode and overwrite output/failed files.",
    )

    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable DeepSeek thinking mode with reasoning_effort='high'.",
    )

    parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="Disable response_format={'type': 'json_object'}.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum output tokens.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum API retries per sample.",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError(
            "No API key found. Set DEEPSEEK_API_KEY or pass --api-key."
        )

    process_jsonl(
        input_path=Path(args.input),
        output_path=Path(args.output),
        failed_path=Path(args.failed),
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        resume=not args.no_resume,
        max_samples=args.max_samples,
        use_thinking=args.thinking,
        use_json_mode=not args.no_json_mode,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()