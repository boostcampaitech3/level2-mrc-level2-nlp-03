from transformers import EvalPrediction
from typing import Callable, Dict, List, NoReturn, Tuple
import numpy as np

from trainer_qa import QuestionAnsweringTrainer, QuestionAnsweringBaseTrainer
from utils.utils_qa import postprocess_qa_predictions

# Post-processing:
def initiate_trainer(
        model,
        training_args,
        data_args,
        train_dataset,
        eval_dataset,
        eval_examples,
        tokenizer,
        data_collator,
        compute_metrics,
        callbacks,
        answer_column_name):

    def post_processing_function(examples,features, predictions,training_args) -> EvalPrediction:
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            print(eval_dataset)
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in eval_examples #datasets["validation"]
            ]

            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    trainer = QuestionAnsweringBaseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    return trainer
