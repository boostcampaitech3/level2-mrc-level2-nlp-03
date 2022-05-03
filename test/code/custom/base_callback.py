from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerState, TrainerCallback, TrainerControl, CallbackHandler

import wandb
import os
import pandas as pd
from datetime import datetime


def rewrite_logs(d):
    # 이번 대회는 Logs가 여러가지로 필요하겠습니다.
    # 전체 pipelie 확인을 위해 retrieval -> reader 를 한번에 돌려서 eval f1, EM 확인하는 것
    # retrieval, reader 각각의 성능을 파악하는 과정이 필요하기 때문입니다.

    # 따라서 기본적으로는 train, eval로 가되
    # 각 stage( retrieval or reader)에 따른 성능을 주요하게 보고 싶다면 아래와 같이 명명하면 되겠습니다.
    # write_mode in ['train', 'eval', 'ret_train_only', 'ret_eval_only', 'read_train', 'read_eval']

    # compute_metrics에서 return 해준 dictionary ( 기본 : EM, f1) 기준으로 처리
    # retrieval 경우 recall@K / precision@K 등의 metric이 필요하겠습니다
    new_d = {}

    for k, v in d.items():
        new_d[k] = v
    return new_d


def make_dirs(path):
    # args에 지정된 폴더가 존재하나 해당 폴더가 없을 경우 대비
    # model save
    os.makedirs(path, exist_ok=True)


class customBaseWandbCallback(WandbCallback):
    def __init__(self):
        " Training, Eval 때 confusion matrix는 Figure class라서 예외처리 추가"
        super(customBaseWandbCallback, self).__init__()

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return

        if not self._initialized:
            self.setup(args, state, model)

        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            # 이렇게 안하면 자꾸 wandb의 train/loss, learning rate이 train/global_step에 먹힌다;
            # if len(logs.keys()) == 3:
            #     self._wandb.log({**logs, "train/global_step": state.global_step})

            for metric, vals in logs.items():
                if isinstance(vals, float) or isinstance(vals, int):
                    self._wandb.log({metric:vals})
                else:
                    self._wandb.log({metric: [self._wandb.Image(vals)]})