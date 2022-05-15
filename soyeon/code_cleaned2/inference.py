"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.
대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""

import logging
import sys
from typing import Callable, Dict, List, NoReturn, Tuple

from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd

import numpy as np
from arguments import DataTrainingArguments, ModelArguments, SettingArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from retrieval import SparseRetrieval

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

from retrieval import SparseRetrieval
from dense_retriver import DenseRetrieval

import timeit
import time
import os

#from trainer_qa import  QuestionAnsweringBaseTrainer
from custom.base_callback import customBaseWandbCallback

import wandb

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (SettingArguments, ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    setting_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True

    dataset_full_path = os.path.join(data_args.dataset_path, data_args.dataset_name)

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {dataset_full_path}")

    # wandb 설정
    if setting_args.use_wandb:
        if setting_args.exp_name:
            exp_full_name = f'{setting_args.exp_name}_{model_args.model_name_or_path}_{dataset_full_path}_{training_args.learning_rate}'  # _{training_args.optim}'
        else:
            exp_full_name = f'{model_args.model_name_or_path}_{dataset_full_path}_{training_args.learning_rate}'  # _{training_args.optim}'
        wandb.login()
        # project : 우리 그룹내에서 본인이 만든 프로젝트 이름
        # name : 저장되는 실험 이름
        # entity : 우리 그룹/팀 이름

        wandb.init(project='wecando',
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
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
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
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
    breakpoint()
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    
    # add 'question' and 'context' tokens
    if data_args.add_tokens:
        special_tokens_dict = {'additional_special_tokens': ['question','context']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        #model.resize_token_embeddings(len(tokenizer))

    # True일 경우 : run passage retrieval

    if not data_args.dpr:
        p_encoder, q_encoder= None, None

    else:
        p_encoder = p_path
        q_encoder = q_path

    if data_args.eval_retrieval:
        # doc score들 일단 뽑아만 둘려고 df 따로 받아서 run_mrc에 넣어줌
        breakpoint()
        datasets, df = run_sparse_retrieval( tokenizer, datasets, training_args, data_args, p_encoder= p_encoder, q_encoder = q_encoder )

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args,
                datasets, tokenizer, model,
                df)


def run_sparse_retrieval(
        tokenize_fn,
        datasets: DatasetDict,
        training_args: TrainingArguments,
        data_args: DataTrainingArguments,
        data_path: str = "/opt/ml/input/data",
        context_path: str = "wikipedia_documents.json",
        p_encoder = None,
        q_encoder = None
) :
    
    # Query에 맞는 Passage들을 Retrieval 합니다.

    if not data_args.dpr_negative:
        # dpr과 dpr_negative의 차이?
        retriever = SparseRetrieval(
            tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path,
            is_bm25=data_args.bm25,
            p_encoder= p_encoder, q_encoder=q_encoder,
            use_wiki_preprocessing=data_args.use_wiki_preprocessing
        )

        retriever.get_sparse_embedding()
    else:
        retriever = DenseRetrieval(tokenize_fn=tokenize_fn, data_path = data_path, 
                                context_path = context_path, dataset_path=data_path+"/train_dataset", 
                                tokenizer=tokenizer, train_data=datasets["validation"], 
                                num_neg=12, is_bm25=data_args.bm25)

        model_checkpoint = "klue/bert-base"
        retriever.load_model(model_checkpoint, "./outputs/dpr/p_encoder_14.pt", "./outputs/dpr/q_encoder_14.pt")
        retriever.get_dense_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        # if bm25, parallel is faster. ELSE, numpy in TFIDF outperforms the parallel. :/
        if data_args.bm25:
            start = time.time()
            print("Calculating BM25 similarity...")
            if data_args.dpr : # dpr + bm25 
                df = retriever.retrieve_dpr(
                    datasets["validation"], topk=data_args.top_k_retrieval
                )
            else:
                df = retriever.retrieve(
                    datasets["validation"], topk=data_args.top_k_retrieval
                )
            end = time.time()
            print("Done! similarity processing time :%d secs "%(int(end - start)))
        else:
            df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # retrieval return에서 doc score도 return하도록 함!
    # 만약 top-k의 성능을 보고 싶으면 여기서 멈춰도 될듯?

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),

                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),

                "document_id": Value(dtype="int64", id=None),  # 추가 했습니다
                "title": Value(dtype="string", id=None),  # 추가했습니다

                # "context_list":Sequence(feature={"single_conteext":Value(dtype='string',id=None)}),
                # "doc_scores": Sequence(feature={"single_score": Value(dtype='float', id=None)})
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets, df


def run_mrc(
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        model_args: ModelArguments,
        datasets: DatasetDict,
        tokenizer,
        model,
        df
) -> NoReturn:
    start_time = timeit.default_timer()  # 시작 시간 체크

    print(datasets)
    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names
    print("column_names:", column_names)

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

    # Validation preprocessing + Training preprocessing # 함수 자리

    # Validation preprocessing / 전처리를 진행합니다. # 함수 자리 

    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    if training_args.do_predict:
        eval_dataset, eval_len_info = utils_qa.preprocess_dataset_with_no_answers(
            dataset=eval_dataset, 
            tokenizer=tokenizer, 
            data_args=data_args,
            column_names=column_names,
            pad_on_right=pad_on_right,
            max_seq_length=max_seq_length,
            data_df=df
            )
    elif training_args.do_eval:
        eval_dataset, eval_len_info = utils_qa.preprocess_dataset_with_answers(
            dataset=eval_dataset, 
            tokenizer=tokenizer, 
            data_args=data_args, 
            column_names=column_names, 
            pad_on_right=pad_on_right, 
            max_seq_length=max_seq_length,
            data_df=df,
            is_train = not training_args.do_eval)

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:# 함수 자리

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    
    # Trainer 초기화
    trainer = utils_qa.initiate_trainer(
        model=model,
        training_args=training_args,
        data_args=data_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[customBaseWandbCallback],
        answer_column_name="answers" if "answers" in column_names else column_names[2],
        eval_len_info = eval_len_info,
        data_df = df
    )

    logger.info("*** Evaluate ***")
    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        print('#####NOW ON PREDICTION')
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if training_args.do_eval:
        print('#####NOW ON VALIDATION')
        metrics = trainer.evaluate(df=df)
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    terminate_time = timeit.default_timer()  # 종료 시간 체크

    print(f"{(terminate_time - start_time) // 60}분 {(terminate_time - start_time) % 60:.3f}초 걸렸습니다.")


if __name__ == "__main__":
    main()
