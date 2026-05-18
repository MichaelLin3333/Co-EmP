import re
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG
# =========================
MODEL_PATH = "C://Users//Michael Lin//projects//Models//models--Qwen--Qwen3.5-4B"   # change this
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
FORCE_CPU = False

MAX_NEW_TOKENS_DEFAULT = 512
TEMPERATURE_DEFAULT = 0.6   # Qwen recommends lower temp for thinking mode
TOP_P_DEFAULT = 0.95


# =========================
# LOAD MODEL
# =========================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

print("Loading model...")
if FORCE_CPU:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to("cpu")
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Model loaded.")


# =========================
# HELPERS
# =========================
def get_model_device():
    if FORCE_CPU:
        return torch.device("cpu")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def parse_think_and_answer(raw_text: str):
    """
    Parse:
      <think> ... </think> final answer
    If no think block exists, reasoning will be empty.
    """
    if not raw_text:
        return "", ""

    match = re.search(r"<think>\s*(.*?)\s*</think>\s*(.*)", raw_text, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        answer = match.group(2).strip()
        return reasoning, answer

    return "", raw_text.strip()


def build_messages(chat_history, user_message, system_prompt, force_think):
    """
    chat_history is stored as OpenAI-style messages:
    [{"role": "user"/"assistant", "content": "..."}]
    """
    messages = []

    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    messages.extend(chat_history)

    final_user_message = user_message.strip()

    # Optional convenience: append /think unless the user already specified a mode.
    if force_think and "/think" not in final_user_message and "/no_think" not in final_user_message:
        final_user_message += " /think"

    messages.append({"role": "user", "content": final_user_message})
    return messages


def generate_reply(chat_history, user_message, system_prompt, temperature, top_p, max_new_tokens, force_think):
    if not user_message.strip():
        return chat_history, chat_history, "", "", ""

    messages = build_messages(
        chat_history=chat_history,
        user_message=user_message,
        system_prompt=system_prompt,
        force_think=force_think
    )

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False   # explicitly retain thinking mode
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = get_model_device()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=(float(temperature) > 0),
            temperature=float(temperature),
            top_p=float(top_p),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    reasoning, final_answer = parse_think_and_answer(raw_output)

    # Keep the FULL raw output in chat so the <think> block is retained
    updated_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": raw_output}
    ]

    return updated_history, updated_history, reasoning, final_answer, ""


def clear_all():
    return [], [], "", "", ""


# =========================
# UI
# =========================
with gr.Blocks(title="Local Qwen3 Thinking Chat") as demo:
    gr.Markdown("## Local Qwen3 Chat with Visible Thinking")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=520
            )

            user_input = gr.Textbox(
                label="Your message",
                placeholder="Type a message and press Send..."
            )

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=2):
            system_prompt = gr.Textbox(
                label="System prompt",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=4
            )

            force_think = gr.Checkbox(
                label="Force /think on each turn unless user specifies otherwise",
                value=True
            )

            temperature = gr.Slider(
                label="Temperature",
                minimum=0.0,
                maximum=1.5,
                value=TEMPERATURE_DEFAULT,
                step=0.05
            )

            top_p = gr.Slider(
                label="Top-p",
                minimum=0.1,
                maximum=1.0,
                value=TOP_P_DEFAULT,
                step=0.05
            )

            max_new_tokens = gr.Slider(
                label="Max new tokens",
                minimum=32,
                maximum=2048,
                value=MAX_NEW_TOKENS_DEFAULT,
                step=32
            )

            reasoning_box = gr.Textbox(
                label="Reasoning extracted from <think>",
                lines=12
            )

            answer_box = gr.Textbox(
                label="Final answer extracted after </think>",
                lines=8
            )

    history_state = gr.State([])

    send_btn.click(
        fn=generate_reply,
        inputs=[
            history_state,
            user_input,
            system_prompt,
            temperature,
            top_p,
            max_new_tokens,
            force_think
        ],
        outputs=[
            chatbot,
            history_state,
            reasoning_box,
            answer_box,
            user_input
        ]
    )

    user_input.submit(
        fn=generate_reply,
        inputs=[
            history_state,
            user_input,
            system_prompt,
            temperature,
            top_p,
            max_new_tokens,
            force_think
        ],
        outputs=[
            chatbot,
            history_state,
            reasoning_box,
            answer_box,
            user_input
        ]
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[
            chatbot,
            history_state,
            reasoning_box,
            answer_box,
            user_input
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)

"""

you are now a middle-aged man in a struggling family. Yoare a person of high self-esteem to a point egotistic, yet you fail your career. You are now talking with your child, John.
You should output only the responses of the role you areu  playing. Your language style should mimic that of the person you are playing. Colloquial language is encouraged as it assimilates conversation.

NO thinking. NO analysis. NO signposting

John: School sent the check. I need more money for tuition.
you:

"""