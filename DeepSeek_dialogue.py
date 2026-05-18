import json
from openai import OpenAI

client = OpenAI(
    api_key='sk-eaa79f0adff64bc7a086de4bf62cc9bd',
    base_url="https://api.deepseek.com")

def generate_dialogue(sample):
    scenario = sample["scenario"]["description"]
    personas = sample["personas"]

    prompt = f"""
    You are an expert in affective psychology.
    You are tasked with simulating realistic emotional dialogues given context. Each scenario involves two personas (A and B) interacting in a specific context. The personas have distinct roles, personality traits (based on the Big Five), communication styles, and backgrounds. Each persona also has an initial emotional state represented by Valence-Arousal-Dominance (VAD) values.

    Scenario:
    {sample}

    Task:
    Generate a dialogue (A and B alternating).
    
    Each turn must include:
    - speaker ("A" or "B")
    - text
    - VAD in [-1,1]

    Example OUTPUT format:
    [
    {{
    "speaker": "A",
    "text": "I can't believe this is happening today… the florist just called, half the arrangements are wrong.",
    "vad": {{"v": -0.75, "a": 0.85, "d": -0.55}}
    }},
    {{
    "speaker": "B",
    "text": "Hey, take a breath. We still have time, and I’ve already contacted a backup vendor.",
    "vad": {{"v": 0.35, "a": 0.25, "d": 0.80}}
    }},
    {{
    "speaker": "A",
    "text": "But everything has to be perfect… my mom is already upset about the seating chart.",
    "vad": {{"v": -0.70, "a": 0.90, "d": -0.60}}
    }},
    {{
    "speaker": "B",
    "text": "I understand, but perfection isn't what people will remember—it's how you feel today.",
    "vad": {{"v": 0.40, "a": 0.30, "d": 0.75}}  
    }}
    ]

    Rules:
    - VAD must evolve based on dialogue content, persona traits, and previous VAD.
    - realistic emotional progression is required (e.g., a highly neurotic persona may experience sharper drops in valence).
    - Initial_vad MUST match the first turn of the dialogue for each persona.
    - Emotion must match dialogue content
    - Avoid random jumps unless justified
    - No explanations

    Output:
    JSON array only
    """

    response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": prompt}
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
    )

    return response

import re

def extract_json_array(content):
    if content is None:
        return None

    matches = re.findall(r'\[[\s\S]*?\]', content)

    for m in matches:
        try:
            return json.loads(m)
        except:
            continue

    return None

def validate_dialogue(dialogue):
    if not isinstance(dialogue, list):
        return False

    for turn in dialogue:
        if "speaker" not in turn or "text" not in turn or "vad" not in turn:
            return False

        vad = turn["vad"]
        if not all(k in vad for k in ["v", "a", "d"]):
            return False

        # check range
        if any(abs(vad[k]) > 1 for k in vad):
            return False

    return True


def expand_with_dialogue(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        sample = json.loads(line)

        response = generate_dialogue(sample)

        content = response.choices[0].message.content

        dialogue = extract_json_array(content)

        if dialogue is None:
            print(f"Failed parsing at {idx}")
            continue

        if not validate_dialogue(dialogue):
            print(dialogue)
            print(f"Invalid VAD structure at {idx}")
            continue

        output = {
            "scenario": sample["scenario"],
            "personas": sample["personas"],
            "dialogue": dialogue
        }

        with open(output_path, "a", encoding="utf-8") as out:
            out.write(json.dumps(output, ensure_ascii=False) + "\n")

from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def process_line(line):
    
    time.sleep(0.2)
    sample = json.loads(line)
    response = generate_dialogue(sample)
    content = response.choices[0].message.content
    dialogue = extract_json_array(content)

    if dialogue is None:
        print(f"Failed parsing at sample {sample['scenario']['description'][:30]}...")
        return None

    if not validate_dialogue(dialogue):
        print(dialogue)
        print(f"Invalid VAD structure at sample {sample['scenario']['description'][:30]}...")
        return None

    output = {
        "scenario": sample["scenario"],
        "personas": sample["personas"],
        "dialogue": dialogue
    }
    return output




def run_parallel(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = lines[2210:]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_line, line) for line in lines]
        print("hi")

        with open(output_path, "a", encoding="utf-8") as out:
            for i, future in enumerate(as_completed(futures)):
                result = future.result()

                if result:
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    print(f"Written {i}")


if __name__ == "__main__":
    run_parallel("EmoDynamic/expanded.jsonl", "EmoDynamic/EmoDynamic.jsonl")