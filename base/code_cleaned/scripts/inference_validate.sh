# ODQA 실행 (test_dataset 사용)
# wandb 가 로그인 되어있다면 자동으로 결과가 wandb 에 저장됩니다. 아니면 단순히 출력됩니다

# 이 코드는 test dataset이 아닌 train dataset에 있는 validation 데이터를 이용해 
# retrieval와 inference 성능을 평가합니다. 

# Sparse Retrieval의 경우 성능이 바뀌지 않겠지만, Dense Retrieval의 경우 학습에 따라 성능이 바뀔 수 있으니 \
# 이 코드를 먼저 돌려보시고 inference.sh를 실행하는 것을 권장합니다.
# Sparse Retrieval의 경우에도 이 코드를 돌리면 소연님이 구현하신대로 wandb 에서 결과를 볼 수 있습니다.

# ***주의*** : 해당 코드로 저장되는 predictions.json은 test dataset의 결과물이 아니기 때문에 
#             리더보드에 제출하면 안됩니다..!(이 파일은 outputs/valid_dataset/에 저장됩니다. )

python inference_validate.py \
--output_dir ./outputs/valid_dataset/ \
--dataset_name train_dataset/ \
--model_name_or_path ./models/train_dataset/ \
--use_wandb True \
--exp_name inference_validate \
--do_predict False \
--do_eval True \
--overwrite_output_dir True \
--add_tokens False