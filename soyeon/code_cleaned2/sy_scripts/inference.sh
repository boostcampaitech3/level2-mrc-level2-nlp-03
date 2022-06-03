python inference.py \
--output_dir ./outputs/test_dataset/ \
--dataset_name test_dataset/ \
--model_name_or_path /opt/ml/level2-mrc-level2-nlp-03/soyeon/code_cleaned2/models/train_dataset/checkpoint-1000 \
--use_wandb False \
--exp_name clean_exp_infrence \
--do_predict True \
--do_eval False \
--overwrite_output_dir True \
--add_tokens False \
--dpr False \
--dpr_negative False \
--bm25 False \
--top_k_retrieval 20