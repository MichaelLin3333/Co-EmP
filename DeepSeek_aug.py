import re
import json
from openai import OpenAI

# number of persona variants per scenario
N_VARIANTS = 3




def generate_personas(description, n_variants=3):
    system_prompt = """
    You are an expert in affective psychology.
    You are generating personality variations for a fixed scenario.

    Each persona must include:
    - role
    - Big Five traits (0-1)
    - style
    - background
    - initial vad in [-1,1]

    EXAMPLE INPUT:
    Scenario: A student returns home after failing an important exam and must explain it to a strict parent.

    Generate N_VARIANTS different persona variations for the given scenario.


    EXAMPLE JSON OUTPUT:
    [
    "personas": {
      "A": {
        "role": "parent",
        "traits": {
          "big_five": {
            "openness": 0.3,
            "conscientiousness": 0.9,
            "extraversion": 0.4,
            "agreeableness": 0.2,
            "neuroticism": 0.7
          },
          "style": "critical, controlling",
          "background": "values discipline and achievement"
        },
        "initial_vad": {"v": -0.2, "a": 0.6, "d": 0.7}
      },
      "B": {
        "role": "student",
        "traits": {
          "big_five": {
            "openness": 0.5,
            "conscientiousness": 0.4,
            "extraversion": 0.3,
            "agreeableness": 0.6,
            "neuroticism": 0.8
          },
          "style": "defensive, anxious",
          "background": "struggles with academic pressure"
        },
        "initial_vad": {"v": -0.6, "a": 0.7, "d": -0.4}
      }
    }
    ...(Give n persona variations)
    ]
    

    IMPORTANT:
    - DO NOT repeat the scenario
    - USE the exact same format and keys as the example output
    - ONLY output a JSON array
    - No explanations
    """
    client = OpenAI(
        api_key='sk-eaa79f0adff64bc7a086de4bf62cc9bd',
        base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""Scenario: {description}
         Generate {n_variants} persona variations for the given scenario."""},
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
    )

    return response.choices[0].message.content

def extract_json_array(content):
    match = re.search(r'\[.*\]', content, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except:
        return None

def expand_dataset(input_path="dataset.jsonl", output_path="expanded.jsonl"):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0

    for line in lines:
        
        base = json.loads(line)
        scenario = base["scenario"]

        if len(base["id"]) <= 4:
            continue

        if base["id"] < "S1010":
            continue
        
        print(base['id'])
        content = generate_personas(scenario["description"], N_VARIANTS)
        #print(base['id'] + ": " + content)
        persona_list = extract_json_array(content)


        if persona_list is None:
            print("Skipping invalid response")
            print(base['id'] + ": " + content)
            continue

        try:
            persona_list.append({"personas": base["personas"]})  # include original as well

        except Exception as e:
            print(f"Error occurred while processing {base['id']}: {e}")
            print(base)
            break

        #print(persona_list)
        #print(persona_list[0]['personas'])
        #print(len(persona_list))

        # merge locally (NO token waste)
        with open(output_path, "a", encoding="utf-8") as out:
            for p in persona_list:
                new_sample = {
                    "scenario": scenario,
                    "personas": p["personas"]
                }
                out.write(json.dumps(new_sample, ensure_ascii=False) + "\n")

        i += 1
        if i % 10 == 0:
            print(f"Processed {i}/{len(lines)} samples")


if __name__ == "__main__":
    expand_dataset("EmoDynamic/dataset4.jsonl", "EmoDynamic/expanded.jsonl")