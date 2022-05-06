import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from transformers import AutoTokenizer
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


# Retrieval로 수정한 함수
class DenseRetrieval:
    def __init__(
        self,
        args,
        datasets,
        tokenize_fn,
        num_neg,
        p_encoder, 
        q_encoder,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ):

        '''
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        '''
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        
        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        self.num_neg = num_neg
        self.ids = list(range(len(self.contexts)))
        self.args = args
        self.batch_size = args.per_device_train_batch_size
        self.dataset = datasets

        self.tokenizer = tokenize_fn
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        self.prepare_in_batch_negative(num_neg=num_neg)


    def prepare_in_batch_negative(self, dataset=None, num_neg=2, tokenizer=None):
        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.        
        corpus = np.array(list(set([example for example in self.contexts])))
        p_with_neg = []

        # for c in dataset['context']:
        for c in self.contexts:
            
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(dataset, padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(1, -1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(1, -1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(1, -1, num_neg+1, max_len)
        q_seqs['input_ids'] = q_seqs['input_ids'].view(1, -1, 1, max_len)
        q_seqs['attention_mask'] = q_seqs['attention_mask'].view(1, -1, 1, max_len)
        q_seqs['token_type_ids'] = q_seqs['token_type_ids'].view(1, -1, 1, max_len)

        print(p_seqs['input_ids'].size())
        print(q_seqs['input_ids'].size())

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)

        valid_seqs = tokenizer(self.contexts, padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)


    def train(self, args=None):

        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    p_encoder.train()
                    q_encoder.train()
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }
            
                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

        return self.p_encoder, self.q_encoder      


    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):

        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer(query, padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_emb = q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)

            p_embs = []
            for batch in self.passage_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        p_embs = torch.stack(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        doc_indices = torch.argsort(dot_prod_scores, dim=1, descending=True)[:, :k]
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        return dot_prod_scores, doc_indices
    


        


    # def get_dense_embedding(self) -> NoReturn:
        
    #     """
    #     Summary:
    #         BERT Encoder로 Passage Embedding을 만들고
    #         만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
    #     """
    #     corpus = list(set(self.contexts))
    #     model_checkpoint = "bert-base-multilingual-cased"
    #     tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #     training_datset = self.contexts
    #     # set number of neagative sample
    #     num_neg = 3

    #     corpus = np.array(corpus)
    #     p_with_neg = []

    #     for c in training_dataset['context']:
    #         while True:
    #             neg_idxs = np.random.randint(len(corpus), size=num_neg)

    #             if not c in corpus[neg_idxs]:
    #                 p_neg = corpus[neg_idxs]

    #                 p_with_neg.append(c)
    #                 p_with_neg.extend(p_neg)
    #                 break
        
    #     q_seqs = tokenizer(training_dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
    #     p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

    #     max_len = p_seqs['input_ids'].size(-1)
    #     p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
    #     p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
    #     p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

    #     train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
    #                     q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])
        
        
    #     # load pre-trained model on cuda (if available)
    #     p_encoder = BertEncoder.from_pretrained(model_checkpoint)
    #     q_encoder = BertEncoder.from_pretrained(model_checkpoint)

    #     if torch.cuda.is_available():
    #         p_encoder.cuda()
    #         q_encoder.cuda()
        
    #     def train(args, num_neg, dataset, p_model, q_model):
    
    #         # Dataloader
    #         train_sampler = RandomSampler(dataset)
    #         train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

    #         # Optimizer
    #         no_decay = ['bias', 'LayerNorm.weight']
    #         optimizer_grouped_parameters = [
    #                 {'params': [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    #                 {'params': [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    #                 {'params': [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    #                 {'params': [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #                 ]
    #         optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #         t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    #         scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    #         # Start training!
    #         global_step = 0
            
    #         p_model.zero_grad()
    #         q_model.zero_grad()
    #         torch.cuda.empty_cache()
            
    #         train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    #         for _ in train_iterator:
    #             epoch_iterator = tqdm(train_dataloader, desc="Iteration")

    #             for step, batch in enumerate(epoch_iterator):
    #             q_encoder.train()
    #             p_encoder.train()
                
    #             targets = torch.zeros(args.per_device_train_batch_size).long()
    #             if torch.cuda.is_available():
    #                 batch = tuple(t.cuda() for t in batch)
    #                 targets = targets.cuda()

    #             p_inputs = {'input_ids': batch[0].view(
    #                                             args.per_device_train_batch_size*(num_neg+1), -1),
    #                         'attention_mask': batch[1].view(
    #                                             args.per_device_train_batch_size*(num_neg+1), -1),
    #                         'token_type_ids': batch[2].view(
    #                                             args.per_device_train_batch_size*(num_neg+1), -1)
    #                         }
                
    #             q_inputs = {'input_ids': batch[3],
    #                         'attention_mask': batch[4],
    #                         'token_type_ids': batch[5]}
                
    #             p_outputs = p_model(**p_inputs)  #(batch_size*(num_neg+1), emb_dim)
    #             q_outputs = q_model(**q_inputs)  #(batch_size*, emb_dim)

    #             # Calculate similarity score & loss
    #             p_outputs = p_outputs.view(args.per_device_train_batch_size, -1, num_neg+1)
    #             q_outputs = q_outputs.view(args.per_device_train_batch_size, 1, -1)

    #             sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()  #(batch_size, num_neg+1)
    #             sim_scores = sim_scores.view(args.per_device_train_batch_size, -1)
    #             sim_scores = F.log_softmax(sim_scores, dim=1)

    #             loss = F.nll_loss(sim_scores, targets)
    #             print(loss)

    #             loss.backward()
    #             optimizer.step()
    #             scheduler.step()
    #             q_model.zero_grad()
    #             p_model.zero_grad()
    #             global_step += 1
                
    #             torch.cuda.empty_cache()


                
    #         return p_model, q_model

    #     args = TrainingArguments(
    #         output_dir="dense_retireval",
    #         evaluation_strategy="epoch",
    #         learning_rate=2e-5,
    #         per_device_train_batch_size=2,
    #         per_device_eval_batch_size=2,
    #         num_train_epochs=2,
    #         weight_decay=0.01
    #     )

    # p_encoder, q_encoder = train(args, num_neg, train_dataset, p_encoder, q_encoder)










        
        




if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", metavar="./data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="./data", type=str, help="")
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, help=""
    )
    parser.add_argument("--use_faiss", metavar=True, type=bool, help="")

    args = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False,)

    # 메모리가 부족한 경우 일부만 사용하세요 !
    num_sample = 1500
    sample_idx = np.random.choice(range(len(train_dataset)), num_sample)
    train_dataset = train_dataset[sample_idx]

    args = TrainingArguments(
        output_dir=os.path.join(args.data_path, "dense_retireval"),
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01
    )
    model_checkpoint = 'klue/roberta-large'

    # 혹시 위에서 사용한 encoder가 있다면 주석처리 후 진행해주세요 (CUDA ...)
    p_encoder = BertEncoder.from_pretrained(args.model_name_or_path).to(args.device)
    q_encoder = BertEncoder.from_pretrained(args.model_name_or_path).to(args.device)

    retriever = DenseRetrieval(args=args, dataset=train_dataset, num_neg=2, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder)

    

    if args.use_faiss:

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)
