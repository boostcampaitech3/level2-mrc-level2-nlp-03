# 학습 예시 (train_dataset 사용)
python train.py \
--output_dir ./models/train_dataset \
--dataset_name train_dataset/ \
--use_wandb True \
--do_train True \
--do_eval True \
--exp_name clean_roberta \
--model_name_or_path klue/roberta-large \
--overwrite_cache True \
--overwrite_output_dir True \
--add_tokens False