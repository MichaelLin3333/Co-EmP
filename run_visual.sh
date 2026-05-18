#!/usr/bin/env bash
set -e

python Qwen-chat-gradio-EmP-4modes.py \
  --vad_model_path "./vad_deberta_v3_regressor" \
  --vad_tokenizer_path "./vad_deberta_v3_regressor"\
  --gru_checkpoint "./emotion_gru_runs/run1/best_model.pt" \
  --gru_model_kwargs '{"z_dim":64,"dropout":0.1,"output_range":"0_1"}' \
  --gru_forward_mode "native_step" \
  --qwen_model_path "C://Users//Michael Lin//projects//Models//models--Qwen--Qwen3.5-4B" \
  --vad_min 0 \
  --vad_max 1 \
  --vad_output_activation "none" \
  --gru_output_activation "none" \
  --qwen_dtype "auto" \
  --qwen_device_map "auto" \
  --max_new_tokens 160 \
  --temperature 0.7 \
  --top_p 0.9 \
  --max_context_turns 8 \
  --server_name "127.0.0.1" \
  --server_port 7860 \