from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SettingArguments:
    """
    Arguments pertaining to basic setting(basic_env, wandb)
    """
    
    device: str = field(
        default='cuda:0',
        metadata={
            "help": "device id"
        },
    )

    # for wandb
    use_wandb: bool = field(
        default=False,
        metadata={
            "help": "if you're ready to log in wandb"
        },
    )

    exp_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "your experiment name"
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    resume: bool = field(
        default=False, 
        metadata={"help": "resume checkout"}
    ) 

    dpr_q_encoder_path : Optional[str] = field(
        default="./outputs/dpr/best_p_enc_model.pt",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    dpr_p_encoder_path : Optional[str] = field(
        default="./outputs/dpr/best_p_enc_model.pt",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    loss_alpha : Optional[float] = field(
        default=0.5,
        metadata={
            "help": "loss weighting for start and end"
        },
    )

    head_drop_ratio: Optional[float] = field(
        default=0.7,
        metadata={
            "help": "loss weighting for start and end"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_path: Optional[str] = field(
        default="/opt/ml/input/data",
        metadata={"help": "The name of the dataset to use."},
    )

    dataset_name: Optional[str] = field(
        default="train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=10,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
    add_tokens : bool = field(
        default=False, metadata={"help": "Whether to add 'question'and 'context' tokens"}
    )

    use_wiki_preprocessing: bool = field (
        default=False,
        metadata={
            "help": "Preprocess wiki documents"
        },
    )
    use_augment: int = field(
        default=0, metadata={"help": "0->none, 1->why, 2-> how, 3->all"}
    )
    dpr: bool = field(
        default=True,
        metadata={"help": "Run dpr+bm25 for default retrival"}
    )
    dpr_negative: bool = field(
        default=True,
        metadata={"help": "Run DenseRetrieval.py with negative sample"}
    )
    bm25: bool = field(
        default=True,
        metadata={"help": "Run bm25 for default retrival"}
    )




