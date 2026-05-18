#python test_deberta_emotion.py \
#      --model_name_or_path /microsoft/deberta-v3-base \
#      --dataset goemotions \
#      --split test

#python build_integrated_vad_dataset.py --custom_path EmoDynamic/EmoDynamic.jsonl

#python train_vad_regressor.py \
#  --seed 42 \
#  --dataset_path integrated_vad_utterance_dataset \
#  --model_name microsoft/deberta-v3-base \
#  --output_dir vad_deberta_v3_regressor \
#  --epochs 4 \
#  --train_batch_size 8 \
#  --eval_batch_size 16 \
#  --learning_rate 1e-5 \
#  --report_to tensorboard \

python eval_vad_regressor.py \
  --dataset_path integrated_vad_utterance_dataset \
  --model_path vad_deberta_v3_regressor \
  --split test \
  --activation clip \
  --output_dir vad_source_eval_test

#python train_vad_regressor.py \
#  --seed 42 \
#  --dataset_path emobank_only_vad_dataset \
#  --model_name microsoft/deberta-v3-base \
#  --output_dir vad_deberta_v3_emobank_only \
#  --epochs 3 \
#  --train_batch_size 8 \
#  --eval_batch_size 16 \
#  --learning_rate 1e-5 \
#  --report_to tensorboard \

#python eval_vad_regressor.py \
#  --dataset_path emobank_only_vad_dataset \
#  --model_path vad_deberta_v3_emobank_only \
#  --split test \
#  --activation sigmoid \
#  --output_dir eval_emobank_only