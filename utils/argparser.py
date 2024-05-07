from dataclasses import dataclass, field
from typing import List
from argparse import ArgumentParser
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)


@dataclass
class MyArguments:
    task: str = field()
    api: str = field(default=None)
    note: str = field(default="debug")    
    my_seed: int = field(default=42)


@dataclass
class LLMArguments:
    hf_ckpt: str = field(default=None)
    together_ckpt: str = field(default=None)
    is_chat_model: bool = field(default=True)
    model_parallel: bool = field(default=False)
    system_prompt: str = "You are a helpful language assistant."
    max_new_tokens: int = field(default=512)
    temperature: float = field(default=0.2)
    do_sample: bool = field(default=True)
    top_k: int = field(default=60)
    top_p: float = field(default=0.9)
    num_beams: int = field(default=1)
    repetition_penalty: float = field(default=1.8)
    stop_tokens: List[str] = field(default_factory=lambda : ["</s>", "[/INST]"])


@dataclass
class RICLMArguments:
    k_for_ric: int = field(default=1)
    max_retrieval_seq_length: int = field(default=256)
    ric_stride: int = field(default=128)
    index_name: str = field(default='bm25')
    

@dataclass
class kNNLMArguments:
    knn_train_file: str = field(default=None)
    eval_subset: str = field(default='train')
    dstore_size: int = field(default=None) 
    k_for_knn: int = field(default=1024)
    keep_linebreaks: bool = field(default=True)
    block_size: int = field(default=None)
    knn_stride: int = field(default=512)
    max_eval_samples: int = field(default=None)
    lmbda: float = field(default=0.25)
    knn_temp: float = field(default=1.0)
    probe: int = field(default=32)
    
    
@dataclass
class DataArguments:
    raw_data_dir: str = field(default=None)
    io_input_path: str = field(default=None)
    io_output_root: str = field(default=None)
    eval_input_dir: str = field(default=None)
    eval_output_dir: str = field(default=None)
    datastore_root: str = field(default=None)
    
    
def get_args():
    parser = HfArgumentParser(
        (MyArguments, LLMArguments, RICLMArguments, kNNLMArguments, TrainingArguments, DataArguments)
    )
    (
        my_args,
        llm_args,
        ric_args,
        knn_args,
        training_args,
        data_args,
    ) = parser.parse_args_into_dataclasses()
    
    return my_args, llm_args, ric_args, knn_args, training_args, data_args


if __name__ == '__main__':
    get_args()
