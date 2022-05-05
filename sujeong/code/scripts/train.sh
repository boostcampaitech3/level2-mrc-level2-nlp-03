# 학습 예시 (train_dataset 사용)
python train.py \
--output_dir ./models/train_dataset \
--dataset_name train_dataset/ \
--use_wandb True \
--do_train True \
--do_eval True \
--exp_name add_tokens \
--model_name_or_path klue/bert-base \
--overwrite_cache True \
--add_tokens False