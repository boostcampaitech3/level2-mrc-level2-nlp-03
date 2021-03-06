#python inference.py \
#--output_dir ./outputs/valid_dataset/ \
#--dataset_name train_dataset/ \
#--model_name_or_path klue/roberta-large \
#--use_wandb True \
#--exp_name inference_validate \
#--do_predict False \
#--do_eval True \
#--overwrite_output_dir True \
#--add_tokens False \
#--dpr False \
#--dpr_negative False \
#--bm25 False

# 아래는 sparse matrix 확인 scripts
#python inference.py \
#--output_dir ./outputs/valid_dataset/ \
#--dataset_name train_dataset/ \
#--model_name_or_path /opt/ml/level2-mrc-level2-nlp-03/soyeon/old_baseline/model_dir/checkpoint-2500 \ # 2500일때 single로  EM이 얼마였는지 까먹
#--use_wandb True \
#--exp_name inference_validate \
#--do_predict False \
#--do_eval True \
#--overwrite_output_dir True \
#--add_tokens False \
#--dpr False \
#--dpr_negative False \
#--bm25 False \
#--top_k_retrieval 30

python inference.py \
--output_dir ./outputs/valid_dataset/ \
--dataset_name train_dataset/ \
--model_name_or_path /opt/ml/level2-mrc-level2-nlp-03/soyeon/code_cleaned2/models/train_dataset/checkpoint-1000 \
--use_wandb True \
--exp_name inference_validate \
--do_predict False \
--do_eval True \
--overwrite_output_dir True \
--add_tokens False \
--dpr False \
--dpr_negative False \
--bm25 False \
--top_k_retrieval 10

python inference.py \
--output_dir ./outputs/valid_dataset/ \
--dataset_name train_dataset/ \
--model_name_or_path /opt/ml/level2-mrc-level2-nlp-03/soyeon/code_cleaned2/models/train_dataset/checkpoint-1000 \
--use_wandb True \
--exp_name inference_validate \
--do_predict False \
--do_eval True \
--overwrite_output_dir True \
--add_tokens False \
--dpr False \
--dpr_negative False \
--bm25 False \
--top_k_retrieval 20

python inference.py \
--output_dir ./outputs/valid_dataset/ \
--dataset_name train_dataset/ \
--model_name_or_path /opt/ml/level2-mrc-level2-nlp-03/soyeon/code_cleaned2/models/train_dataset/checkpoint-1000 \
--use_wandb True \
--exp_name inference_validate \
--do_predict False \
--do_eval True \
--overwrite_output_dir True \
--add_tokens False \
--dpr False \
--dpr_negative False \
--bm25 False \
--top_k_retrieval 30
