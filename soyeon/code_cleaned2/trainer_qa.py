# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Question-Answering task와 관련된 'Trainer'의 subclass 코드 입니다.
"""
from typing import Optional, List, Dict, Callable, Union, Any, Tuple
import torch
import torch.nn as nn

from transformers import Trainer, is_datasets_available, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

# Huggingface의 Trainer를 상속받아 QuestionAnswering을 위한 Trainer를 생성합니다.
# Base Trainer로 변환
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # 일시적으로 metric computation를 불가능하게 한 상태이며, 해당 코드에서는 loop 내에서 metric 계산을 수행합니다.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions, self.args
            )
            metrics = self.compute_metrics(eval_preds)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: PyTorch/XLA에 대한 Logging debug metrics (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)

        # 일시적으로 metric computation를 불가능하게 한 상태이며, 해당 코드에서는 loop 내에서 metric 계산을 수행합니다.
        # evaluate 함수와 동일하게 구성되어있습니다
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"],
                columns=list(test_dataset.features.keys()),
            )

        predictions = self.post_process_function(
            test_examples, test_dataset, output.predictions, self.args
        )
        return predictions

# Huggingface의 Trainer를 상속받아 QuestionAnswering을 위한 Trainer를 생성합니다.
# baseline huggingface overriding


class QuestionAnsweringBaseTrainer(QuestionAnsweringTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.self_eval_cnt = 1
        # self.control.should_evaluate 업데이트가 되면서 eval_steps에 따라 돼야하는데 동작이 이상함. -> 내부에서 따로 eval_steps에 따라 동작하도록함
        self.self_train_log_cnt = kwargs['args'].logging_steps
        self.max_eval_cnt = kwargs['args'].eval_steps
        self.self_eval_check_mode = True if kwargs['args'].evaluation_strategy =='steps' else False
        # self.callback_handler.callbacks.pop(2) # 아래와 같은 상황에서 WandbCallback 제거
        # [<transformers.trainer_callback.DefaultFlowCallback object at 0x7f333f3a3370>, <transformers.integrations.TensorBoardCallback object at 0x7f333f3a33d0>, <transformers.integrations.WandbCallback object at 0x7f333f3a35b0>, <custom.base_callback.customBaseWandbCallback object at 0x7f333f3a3490>, <transformers.trainer_callback.ProgressCallback object at 0x7f333f3a3430>]
        self.do_log_topk = False

    # def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    #     """
    #     Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    #     passed as an argument.
    #     Args:
    #         num_training_steps (int): The number of training steps to do.
    #     """
    #     from transformers.optimization import get_scheduler
    #     if self.lr_scheduler is None:
    #         self.lr_scheduler = get_scheduler(
    #             self.args.lr_scheduler_type,
    #             optimizer=self.optimizer if optimizer is None else optimizer,
    #             num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
    #             num_training_steps=num_training_steps,
    #         )
    #     return self.lr_scheduler

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval=None):
        if self.control.should_log:
            # if is_torch_tpu_available():
            #     xm.mark_step()

            logs = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.do_log_topk = False
            self.log(logs, write_mode='train')
            self.self_eval_cnt+= self.self_train_log_cnt #(self.self_train_log_cnt//self.max_eval_cnt)

        metrics = None

        # 밑에 self.control_should_evaluate 이 eval_steps에 따라 작동이 안돼서..

        if self.self_eval_check_mode and self.max_eval_cnt <= self.self_eval_cnt:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self.self_eval_cnt = 1

        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def log(self, logs, eval_gt=None, eval_pred=None, write_mode='eval'):
        ############
        # 수정한 부분 : 여긴 아예 많이 고침
        ############

        # wandb에 저장하는 값들을 위해서 overiding
        # 이 부분은 evaluation에서
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        # 매번 reset인지는 확인 필요함
        # # https://github.com/huggingface/transformers/blob/daecae1f1ce02d2dab23742d24f7a66a7d20cb79/src/transformers/trainer_callback.py#L284

        if write_mode == 'eval':
            # train 이면 _maybe_log_save_evaluate의 self.log 에서 input, prediction을 굳이 넘기진 않기 때문(넘겨오게 할 수도 있지만 굳이?)
            self.state.log_history.append({'eval_gt': eval_gt})
            self.state.log_history.append({'eval_pred': eval_pred})
        else:
            self.state.log_history.append({'eval_gt': None})
            self.state.log_history.append({'eval_pred': None})

        # 여기는 PR curve 기록을 위한 용도임!
        if self.do_log_topk:
            self.state.log_history.append({'topk_info': [self.topk_pr_columns, self.topk_pr]})
        else:
            self.state.log_history.append({'topk_info':None})

        self.state.log_history.append({'write_mode': write_mode})

        # callback_hanlder.on_log가 돌아가면 trainer_callback.py에서 train/logs 값으로 on_log를 호출하는 것 같다..!
        # customWandb 를 넣어주는데 wandbCallback이 기본으로 들어가있음 -> 위에서 pop 시켜줌
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, df=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        ############
        # 수정한 부분
        ############
        self.do_log_topk= False
        if df is not None:
            self.do_log_topk = True
            self.topk_pr_columns = ["num_k", "include", "gt_doc", "retrieval_docs", "scores"]
            self.topk_pr = []
            for idx in range(len(df)):
                cur_data = df.iloc[idx]
                yes = 1 if cur_data.document_id in cur_data.context_id else 0

                self.topk_pr.append([len(df.iloc[idx].context_id),
                                     yes,
                                     df.iloc[idx].document_id,
                                     df.iloc[idx].context_id,
                                     df.iloc[idx].doc_scores])

        ############
        # 일시적으로 metric computation를 불가능하게 한 상태이며, 해당 코드에서는 loop 내에서 metric 계산을 수행합니다.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None

        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions, self.args
            )
            metrics = self.compute_metrics(eval_preds)
            ############
            # 수정한 부분
            ############
            metrics.update({'loss': output.metrics['eval_loss']})
            self.log(metrics, eval_gt=eval_examples, eval_pred = eval_preds, write_mode='eval')
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: PyTorch/XLA에 대한 Logging debug metrics (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        ############
        # 수정한 부분
        ############
        # self.control = self.callback_handler.on_evaluate(
        #     self.args, self.state, self.control, metrics
        # )
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)

        # 일시적으로 metric computation를 불가능하게 한 상태이며, 해당 코드에서는 loop 내에서 metric 계산을 수행합니다.
        # evaluate 함수와 동일하게 구성되어있습니다
        compute_metrics = self.compute_metrics
        self.compute_metrics = None

        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"],
                columns=list(test_dataset.features.keys()),
            )

        predictions = self.post_process_function(
            test_examples, test_dataset, output.predictions, self.args
        )
        return predictions


    # def prediction_step(
    #     self,
    #     model: nn.Module,
    #     inputs: Dict[str, Union[torch.Tensor, Any]],
    #     prediction_loss_only: bool,
    #     ignore_keys: Optional[List[str]] = None,
    # ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     """
    #     오버라이딩 필요부분
    #     """
    #     has_labels = all(inputs.get(k) is not None for k in self.label_names)
    #     inputs = self._prepare_inputs(inputs)
    #     if ignore_keys is None:
    #         if hasattr(self.model, "config"):
    #             ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
    #         else:
    #             ignore_keys = []
    #
    #     # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
    #     breakpoint()
    #     if has_labels:
    #         labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
    #         if len(labels) == 1:
    #             labels = labels[0]
    #     else:
    #         labels = None
    #
    #     with torch.no_grad():
    #         # sagemaker 부분 날림
    #         breakpoint()
    #         if has_labels:
    #             with self.autocast_smart_context_manager():
    #                 loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
    #             loss = loss.mean().detach()
    #
    #             if isinstance(outputs, dict):
    #                 logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
    #             else:
    #                 logits = outputs[1:]
    #         else:
    #             loss = None
    #             with self.autocast_smart_context_manager():
    #                 outputs = model(**inputs)
    #             if isinstance(outputs, dict):
    #                 logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
    #             else:
    #                 logits = outputs
    #             # TODO: this needs to be fixed and made cleaner later.
    #             if self.args.past_index >= 0:
    #                 self._past = outputs[self.args.past_index - 1]
    #
    #     if prediction_loss_only:
    #         return (loss, None, None)
    #
    #     logits = nested_detach(logits)
    #     if len(logits) == 1:
    #         logits = logits[0]
    #
    #     return (loss, logits, labels)

class QuestionAnsweringBaseTrainer2(QuestionAnsweringBaseTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoother = nn.MSELoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # outpusts['start_logits'], outpusts['end_logits']

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss