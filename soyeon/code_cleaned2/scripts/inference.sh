# ODQA 실행 (test_dataset 사용)
# wandb 가 로그인 되어있다면 자동으로 결과가 wandb 에 저장됩니다. 아니면 단순히 출력됩니다
python inference.py \
--output_dir ./outputs/test_dataset/ \
--dataset_name test_dataset/ \
--model_name_or_path /opt/ml/level2-mrc-level2-nlp-03/soyeon/code_cleaned2/models/train_dataset/checkpoint-2700 \
--use_wandb False \
--exp_name clean_exp_infrence \
--do_predict True \
--do_eval False \
--overwrite_output_dir True \
--add_tokens False \
--dpr False \
--bm25 True \
--top_k_retrieval 35


