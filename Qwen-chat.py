import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread


MODEL_PATH = r"C:\\Users\\Michael Lin\\projects\\IOAI2025\\qwen3-8b-local"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
SYSTEM_PROMPT = "You are a helpful assistant."

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

model.eval()

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

print("Streaming chat ready.")
print("Type 'exit' or 'quit' to stop.")
print("Type 'clear' to reset conversation.\n")


def stream_reply(messages_list):
    text = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    full_response = ""
    print("Qwen3: ", end="", flush=True)
    for new_text in streamer:
        print(new_text, end="", flush=True)
        full_response += new_text
    print("\n")

    return full_response.strip()


while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        print("Exiting chat.")
        break

    if user_input.lower() == "clear":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        print("Conversation cleared.\n")
        continue

    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    try:
        assistant_reply = stream_reply(messages)
        messages.append({"role": "assistant", "content": assistant_reply})
    except Exception as e:
        print(f"Generation error: {e}\n")