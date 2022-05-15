import logging
import os
import sys
import wandb
from pprint import pprint
from typing import NoReturn

from arguments import SettingArguments, DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk, load_metric

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

import utils_qa
import timeit
import os

from custom.base_callback import customBaseWandbCallback
logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (SettingArguments, ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    setting_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.eval_steps= 50#200
    training_args.evaluation_strategy = 'steps'
    training_args.logging_steps = 50#200
    training_args.save_steps = 500#800
    training_args.report_to = ['wandb']

    print(setting_args)
    print(model_args.model_name_or_path)
    # training_args.num_train_epoch = 5
    # training_args.lr_scheduler = 5
    training_args.warmup_steps = 180
    training_args.lr_scheduler_type = 'linear'#"cosine_with_restarts" # ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    # training_args.per_device_train_batch_size = 4
    # print(training_args.per_device_train_batch_size)

    dataset_full_path = os.path.join(data_args.dataset_path, data_args.dataset_name)

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {dataset_full_path}")

    # wandb 설절
    if setting_args.use_wandb:
        if setting_args.exp_name:
            exp_full_name = f'{setting_args.exp_name}_{model_args.model_name_or_path}_{dataset_full_path}_{training_args.learning_rate}'#_{training_args.optim}'
        else:
            exp_full_name = f'{model_args.model_name_or_path}_{dataset_full_path}_{training_args.learning_rate}'#_{training_args.optim}'
        wandb.login()
        # project : 우리 그룹내에서 본인이 만든 프로젝트 이름
        # name : 저장되는 실험 이름
        # entity : 우리 그룹/팀 이름

        wandb.init(project='kimcando',
                    name=exp_full_name,
                    entity='mrc-competition')  # nlp-03
        wandb.config.update(training_args)
        print('#######################')
        print(f'Experiments name: {exp_full_name}')
        print('#######################')
    else:
        exp_full_name = ''
        print('@@@@@@@@Notice@@@@@@@@@@')
        print('YOU ARE NOT LOGGING RESULTS NOW')
        print('@@@@@@@@$$$$$$@@@@@@@@@@')


    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(dataset_full_path)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )
    
    # add 'question' and 'context' tokens
    if data_args.add_tokens:
        special_tokens_dict = {'additional_special_tokens': ['question','context']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        #model.resize_token_embeddings(len(tokenizer))

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    start_time = timeit.default_timer() # 시작 시간 체크

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    #question_column_name = "question" if "question" in column_names else column_names[0]
    #context_column_name = "context" if "context" in column_names else column_names[1]
    #answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = utils_qa.check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Train preprocessing / 전처리를 진행합니다. # 함수 자리 

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        
        # dataset에서 train feature를 생성합니다.
        train_dataset, train_len_info = utils_qa.preprocess_dataset_with_answers(
            dataset=train_dataset, 
            tokenizer=tokenizer, 
            data_args=data_args, 
            column_names=column_names, 
            pad_on_right=pad_on_right, 
            max_seq_length=max_seq_length, 
            is_train=training_args.do_train)

    # Validation preprocessing + Training preprocessing # 함수 자리 
    # Validation preprocessing (but not giving labels) # 함수 자리 

    if training_args.do_eval:
        eval_dataset = datasets["validation"]

        # # Validation Feature 생성
        # eval_dataset = preprocess_dataset_with_no_answers(
        #    dataset=eval_dataset, 
        #    tokenizer=tokenizer, 
        #    data_args=data_args,
        #    column_names=column_names,
        #    pad_on_right=pad_on_right,
        #    max_seq_length=max_seq_length
        #    )

        # Validation Feature 생성 with answers
        eval_dataset,eval_len_info = utils_qa.preprocess_dataset_with_answers(
            dataset=eval_dataset, 
            tokenizer=tokenizer, 
            data_args=data_args, 
            column_names=column_names, 
            pad_on_right=pad_on_right,
            max_seq_length=max_seq_length, 
            is_train=not training_args.do_eval)
    
    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:# 함수 자리 

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Trainer 초기화
    trainer = utils_qa.initiate_trainer(
        model=model,
        training_args=training_args,
        data_args=data_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[customBaseWandbCallback],
        answer_column_name="answers" if "answers" in column_names else column_names[2],
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    terminate_time = timeit.default_timer() # 종료 시간 체크  
 
    print(f"{(terminate_time - start_time)//60}분 {(terminate_time - start_time)%60:.3f}초 걸렸습니다.")

if __name__ == "__main__":
    main()