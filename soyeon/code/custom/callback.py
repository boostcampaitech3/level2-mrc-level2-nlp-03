from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerState,TrainerCallback,TrainerControl, CallbackHandler

import wandb
import os
import pandas as pd
from datetime import datetime

def rewrite_logs(write_mode, d):
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
        new_d[f"{write_mode}/" + k] = v
    return new_d

def make_dirs(path):
    # args에 지정된 폴더가 존재하나 해당 폴더가 없을 경우 대비
    # model save
    os.makedirs(path, exist_ok=True)


class customWandbCallback(WandbCallback):
    def __init__(self, auto_save=True, save_path= './test_json_results'):
        " Training, Eval 때 confusion matrix는 Figure class라서 예외처리 추가"
        super(customWandbCallback, self).__init__()
        self.auto_save = auto_save
        self.save_path = save_path
        self.make_dirs(self.save_path)

    def make_dirs(self, path):
        os.makedirs(path, exist_ok=True)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return

        if state.log_history[-1]['write_mode'] =='train':
            write_mode = 'train'
        else:
            write_mode = 'eval'
        # https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/integrations.py#L535
        if not self._initialized:
            self.setup(args, state, model)
        # breakpoint()
        if state.is_world_process_zero:

            # 모드에 따라 이름을 변경해주는 함수
            logs = rewrite_logs(write_mode, logs)

            for metric, vals in logs.items():
                if isinstance(vals, float) or isinstance(vals, int):
                    self._wandb.log({metric:vals})
                else:
                    self._wandb.log({metric: [self._wandb.Image(vals)]})

            # 테이블 추가 & auto_save에 따라서 서버에 csv 저장
            # Trainer에서 저장할 때 append로 했기 때문에 [1], [2]와 같은 indexing 필요. (안전하게 하려고..)

            # retrieval
            if write_mode =='eval':
                # train 때의 log랑 쌓여있음 -> 뒤에서 3번째꺼
                eval_gt = state.log_history[-4]['eval_gt']
                eval_pred = state.log_history[-3]['eval_pred']
                columns = ['gt_id','gt_title', 'gt_doc_id','gt_context', 'gt_len','gt_start_idx', 'question','gt_answer', 'prediction']
                data = []
                # https://docs.wandb.ai/guides/track/log/media

                # 마지막 eval일 때는 eval_pred[0]가 없는 것 같당!
                # need to fix @TODO
                if eval_gt is not None:
                    for idx in range(len(eval_pred[0])):
                        cur_gt_data = eval_gt[idx]

                        pred_data = eval_pred[0][idx] # 답이 1개인 것만 있다해서 answer는 1개만 받습니당
                        new_data = [cur_gt_data['id'], cur_gt_data['title'], cur_gt_data['document_id'],
                                    cur_gt_data['context'], len(cur_gt_data['context']),
                                    cur_gt_data['answers']['answer_start'], cur_gt_data['question'],
                                    cur_gt_data['answers']['text'][0] ,pred_data['prediction_text']]
                        data.append(new_data)
                    test_table = wandb.Table(data = data, columns=columns)

                    if self.auto_save:
                        cur_date = datetime.now()
                        df = pd.DataFrame(data, columns = columns)
                        df.to_csv(os.path.join(self.save_path, f'{cur_date.strftime("%d-%b-%Y (%H:%M:%S.%f)")}-results.json'))

                    self._wandb.log({'test_results': test_table})
                    # test_artifacts.add(test_table, 'test table')
                    # self._wandb.run.log_artifact(test_artifacts)
                    # durleh 마찬가지로 누적돼서..
                    if state.log_history[-2]['topk_info'] is not None:
                        topk_table = wandb.Table(data=state.log_history[3]['topk_info'][1], columns = state.log_history[3]['topk_info'][0])
                        self._wandb.log({'topk_results': topk_table})
                    # 저장해주기 위해서 dataset type으로 저장된 eval_gt, eval_pred 제거
                    # breakpoint()
                    state.log_history.pop(-3)
                    state.log_history.pop(-3)
            """ 예제
            
            (Pdb) eval_pred[1][0]
            {'id': 'mrc-0-003264', 'answers': {'answer_start': [284], 'text': ['한보철강']}}
            (Pdb) eval_pred[0][0]
            {'id': 'mrc-0-003264', 'prediction_text': '대'}
            (Pdb) eval_Gt[0]
            *** NameError: name 'eval_Gt' is not defined
            (Pdb) eval_gt[0]
            {'title': '전효숙', 'context': '순천여자고등학교 졸업, 1973년 이화여자대학교를 졸업하고 1975년 제17회 사법시험에 합격하여 판사로 임용되었고 대법원 재판연구관, 수원지법 부장판사, 사법연수원 교수, 특허법원 부장판사 등을액주주들을 대표해 한보철강 부실대출에 책임이 있는 이철수 전 제일은행장 등 임원 4명을 상대로 제기한 손해배상청구소송에서 서울지방법원 민사합의17부는 1998년 7월 24일에 "한보철강에 부실 대출하여 은행에 막대한 손해를 끼친월 신행정수도의건설을위한특별조치법 위헌 확인 소송에서 9인의 재판관 중 유일하게 각하 견해를 내었다. 소수의견에서 전효숙 재판관은 다수견해의 문제점을 지적하면서 관습헌법 법리를 부정하였다. 전효숙 재판관은 서울대학교 근를 밝힌 바 있다.', 'question': '처음으로 부실 경영인에 대한 보상 선고를 받은 회사는?', 'id': 'mrc-0-003264', 'answers': {'answer_start': [284], 'text': ['한보철강']}, 'document_id': 9027, '__index_level_0__': 2146}

            """

