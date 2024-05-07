import json
import re
import os
import random
import numpy as np
import torch
from datetime import datetime
from typing import Tuple, List


def fix_seeds(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_root, log_header, dataset_name, model_ckpt, note):
    model_name = model_ckpt.split("/")[-1]
    assert "/" not in log_header
    assert "/" not in dataset_name
    assert "/" not in model_name
    assert "/" not in note
    log_dir = os.path.join(log_root, f'{log_header}_logs/[{note}] {dataset_name}_{model_name}/{datetime.now().strftime("%Y-%m%d-%H%M")}')
    os.makedirs(log_dir, exist_ok=True)   
    return log_dir


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        js = json.load(f)
    return js


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()
    return txt


def read_raw_data_dir(raw_data_dir, recursive=True) -> List[str]:
    """only read txt files"""
    data = []
    if recursive:
        for root, dirs, files in os.walk(raw_data_dir):
            for f in files:
                if "txt" not in f:
                    continue
                full_path = os.path.join(root, f)
                d = read_txt(full_path)
                data.append(d)
    else:
        raise NotImplementedError
    
    return data