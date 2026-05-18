# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

import json
import re

def extract_json(content):
  match = re.search(r'\{.*\}|\[.*\]', content, re.DOTALL)
  if not match:
    return None
  try:
    return json.loads(match.group())
  except:
    return None

def save_sample(response, path="EmoDynamic/dataset4.jsonl"):
  content = response.choices[0].message.content
  data = extract_json(content)

  if data is None:
    print("Invalid JSON, skipping...")
    return

  if not isinstance(data, list):
    data = [data]

  with open(path, "a", encoding="utf-8") as f:
    for item in data:
      f.write(json.dumps(item, ensure_ascii=False) + "\n")

import json
import re

def get_latest_id(path="EmoDynamic/dataset4.jsonl"):
  try:
    with open(path, "r", encoding="utf-8") as f:
      lines = f.readlines()
  except FileNotFoundError:
    return 1  # start from S001

  # iterate backwards for efficiency
  for line in reversed(lines):
    line = line.strip()
    if not line:
      continue
    try:
      obj = json.loads(line)
      id_str = obj.get("id", "")
      match = re.match(r"S(\d+)", id_str)
      if match:
        return int(match.group(1))
    except:
      continue

  return 1

client = OpenAI(
    api_key='sk-eaa79f0adff64bc7a086de4bf62cc9bd',
    base_url="https://api.deepseek.com")

system_prompt = """
You are generating realistic two-person emotional interaction scenarios for a machine learning dataset.

Each scenario must define:
- a concrete situation,
- two personas A and B,
- their Big Five traits,
- their communication style,
- their psychologically relevant background,
- their initial Valence-Arousal-Dominance state.

Return valid JSON only.

Each scenario must follow this exact structure:

[
  {
    "id": "S001",
    "scenario": {
      "title": "Short Scenario Title",
      "description": "One concrete sentence describing the starting situation."
    },
    "personas": {
      "A": {
        "role": "specific role of A",
        "traits": {
          "big_five": {
            "openness": 0.0,
            "conscientiousness": 0.0,
            "extraversion": 0.0,
            "agreeableness": 0.0,
            "neuroticism": 0.0
          },
          "style": "brief communication style",
          "background": "brief psychologically relevant background"
        },
        "initial_vad": {"v": 0.0, "a": 0.0, "d": 0.0}
      },
      "B": {
        "role": "specific role of B",
        "traits": {
          "big_five": {
            "openness": 0.0,
            "conscientiousness": 0.0,
            "extraversion": 0.0,
            "agreeableness": 0.0,
            "neuroticism": 0.0
          },
          "style": "brief communication style",
          "background": "brief psychologically relevant background"
        },
        "initial_vad": {"v": 0.0, "a": 0.0, "d": 0.0}
      }
    }
  }
]

Value rules:
- Big Five values must be floats from 0.0 to 1.0.
- VAD values must be floats from -1.0 to 1.0.
- Valence: -1.0 very negative, 0.0 neutral, 1.0 very positive.
- Arousal: -1.0 very calm/passive, 0.0 neutral, 1.0 highly activated.
- Dominance: -1.0 powerless/submissive, 0.0 neutral, 1.0 powerful/in-control.

Quality rules:
- Do not include dialogue.
- Do not include explanations.
- Do not include markdown.
- Descriptions must be one sentence.
- Scenarios should support varied emotional movement during later dialogue.
- Avoid making every scenario dramatic, conflict-centered, or highly personally relevant.
- Include realistic subtle situations, not only obvious confrontations or confessions.
- Return only the JSON array.
"""



latest_id = get_latest_id()
start_id = latest_id + 1

while start_id <= 1000:

  latest_id = get_latest_id()
  start_id = latest_id + 1

  user_prompt = f"""
  Generate 10 scenarios.

  Start with id as S{start_id:03d}.

  This batch should deliberately expand emotional diversity.

  Required batch mix:
  - 2 low-personal-relevance scenarios
  - 2 high-personal-relevance scenarios
  - 2 asymmetric-relevance scenarios where one person cares much more than the other
  - 2 subtle or ambiguous scenarios where the emotional shift is not obvious
  - 2 positive or repairing scenarios
  - 2 negative or deteriorating scenarios
  - 2 scenarios involving dominance reversal, where the initially less powerful person gains control

  Include varied stakes:
  - low-stakes everyday situations
  - medium-stakes social/professional situations
  - high-stakes personal situations

  Avoid these overused scenario types:
  - roommate disputes
  - neighbor noise complaints
  - inheritance disputes
  - job promotions
  - failed exams
  - divorce announcements
  - medical diagnoses
  - ex-partner café meetings
  - secret confessions between friends
  - generic project failures
  - wrong coffee orders
  - dropped groceries

  Make the scenarios unusual but realistic.

  Return only the JSON array.
  """

  print(f"S{start_id:03d}")

  response = client.chat.completions.create(
      model="deepseek-v4-flash",
      messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt},
      ],
      stream=False,
      reasoning_effort="high",
      extra_body={"thinking": {"type": "enabled"}}
  )

  print(response.choices[0].message.content)

  save_sample(response) 