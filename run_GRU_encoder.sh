python GRU_full_encoder.py \
  --jsonl_path EmoDynamic/EmoDynamic.jsonl \
  --output_dir emotion_gru_runs/run1 \
  --epochs 40 \
  --batch_size 16 \
  --lr 2e-4 \
  --z_dim 64