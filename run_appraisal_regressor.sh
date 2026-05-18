python train_appraisal_regressor.py \
  --custom_jsonl EmoDynamic/EmoDynamic_with_appraisal.jsonl \
  --output_dir outputs/appraisal_deberta_v3 \
  --model_name microsoft/deberta-v3-base \
  --epochs 4 \
  --batch_size 8 \
  --eval_batch_size 16 \
  --max_length 512 \
  --local_files_only \
  --bf16