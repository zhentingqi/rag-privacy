from modules.LM import LM   
from modules.Index import BM25Index
from modules.knnlm_backbone import KNNWrapper, KNNSaver, DIST, KEY_TYPE

import os
import json
import math
from typing import List, Dict
from transformers import (
    Trainer,
    default_data_collator,
)
from datasets import load_dataset
from tqdm import tqdm
import wandb
wandb.init(mode="disabled") 
import logging
logger = logging.getLogger(__name__)

padding_index = -100


class RALM(object):
    def __init__(self, lm: LM) -> None:
        self.lm = lm

    def generate(self, query: str, compute_generation_scores=False, compute_input_loss=False):
        raise NotImplementedError
    
    def finish(self):
        raise NotImplementedError


class RICLM(RALM):
    def __init__(self, ric_args, data_args, lm: LM) -> None:
        super().__init__(lm)
        
        self.k = ric_args.k_for_ric
        
        assert data_args.raw_data_dir is not None
        data_src_name = data_args.raw_data_dir.split("/")[-1]
        datastore_path = os.path.join(data_args.datastore_root, f"RIC_LM+{data_src_name}+{lm.model_name}+{ric_args.max_retrieval_seq_length}+{ric_args.ric_stride}")
        
        #! index could be: BM25Index, VectorStoreIndex (from llama_index) 
        if ric_args.index_name == 'bm25':
            self.index = BM25Index(
                tokenizer=self.lm.tokenizer,
                max_retrieval_seq_length=ric_args.max_retrieval_seq_length,
                stride=ric_args.ric_stride,
                raw_data_dir=data_args.raw_data_dir,
                datastore_dir=datastore_path,
            )
        else:
            raise NotImplementedError
        
    def generate(self, query: str, compute_generation_scores=False, compute_input_loss=False):
        def concat_docs(docs: List[str]):
            docs_str = "\n\n".join(docs)
            return docs_str
        
        docs: List[str] = self.index.find_most_relevant_k_documents(query=query, k=self.k)
        docs_str = concat_docs(docs)
        
        lm_input = docs_str + "\n\n" + query
        output_dict = self.lm.generate(lm_input, compute_generation_scores, compute_input_loss)
        if compute_input_loss:
            docs_tokens = self.lm.tokenizer(docs_str + "\n\n", return_tensors="pt")["input_ids"]
            output_dict["token_loss_list"] = output_dict["token_loss_list"][docs_tokens.shape[1]:]
            output_dict["total_input_loss"] = sum(output_dict["token_loss_list"]) / len(output_dict["token_loss_list"])
            output_dict["token_ppl_list"] = output_dict["token_ppl_list"][docs_tokens.shape[1]:]
            output_dict["total_input_ppl"] = math.exp(output_dict["total_input_loss"])
        
        output_dict["retrieved_docs"] = docs
        output_dict["retrieved_docs_str"] = docs_str
        
        return output_dict
    
    def finish(self):
        return 


class kNNLM(RALM):
    def __init__(self, knn_args, training_args, data_args, lm: LM) -> None:
        super().__init__(lm)
        
        assert knn_args.knn_train_file is not None
        assert knn_args.eval_subset is not None
        assert data_args.raw_data_dir is not None
        
        data_src_name = data_args.raw_data_dir.split("/")[-1]
        datastore_path = os.path.join(data_args.datastore_root, f"kNN_LM+{data_src_name}+{lm.model_name}")
        
        if not os.path.exists(datastore_path):            
            assert knn_args.knn_train_file.split(".")[-1] == "txt"
            data_files = {"train": knn_args.knn_train_file}
            dataset_kwargs = {"keep_linebreaks": knn_args.keep_linebreaks}
            extension = "text"
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                **dataset_kwargs,
            )
            
            #! ======================================================
            #! Preprocessing 1: tokenize
            #! ======================================================
            # Preprocessing the datasets. First we tokenize all the texts.
            print("==> Preprocessing: tokenize...")
            column_names = raw_datasets["train"].column_names
            text_column_name = "text" if "text" in column_names else column_names[0]

            def tokenize_function(examples):
                return self.lm.tokenizer(examples[text_column_name])

            with training_args.main_process_first(desc="dataset map tokenization"):
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                    load_from_cache_file=True,
                    desc="Running tokenizer on dataset",
                )
            
            #! ======================================================
            #! Preprocessing 2: concatenate
            #! ======================================================
            print("==> Preprocessing: concatenate...")
            if knn_args.block_size is None:
                block_size = self.lm.tokenizer.model_max_length
                if block_size > 1024:
                    logger.warning(
                        f"The tokenizer picked seems to have a very large `model_max_length` ({self.lm.tokenizer.model_max_length}). "
                        "Picking 1024 instead. You can change that default value by passing --block_size xxx."
                    )
                    block_size = 1024
            else:
                if knn_args.block_size > self.lm.tokenizer.model_max_length:
                    logger.warning(
                        f"The block_size passed ({knn_args.block_size}) is larger than the maximum length for the model"
                        f"({self.lm.tokenizer.model_max_length}). Using block_size={self.lm.tokenizer.model_max_length}."
                    )
                block_size = min(knn_args.block_size, self.lm.tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.

            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                if total_length >= block_size:
                    total_length = (total_length // block_size) * block_size

                input_ids = []
                attention_mask = []
                labels = []
                # We implement a sliding window, so all tokens have a non-zero context in their prediction.
                # We then mask the duplicate tokens' labels, to not count any token twice in the loss.
                for i in tqdm(range(0, total_length, knn_args.knn_stride), total=total_length):
                    begin_loc = max(i + knn_args.knn_stride - block_size, 0)
                    end_loc = min(i + knn_args.knn_stride, total_length)
                    trg_len = end_loc - i
                    cur_input_ids = concatenated_examples["input_ids"][begin_loc:end_loc]
                    cur_labels = list(cur_input_ids)
                    cur_labels[:-trg_len] = [padding_index] * (len(cur_labels) - trg_len)

                    if len(cur_input_ids) < block_size:
                        padding_size = block_size - len(cur_input_ids)
                        pad_token_id = (
                            self.lm.tokenizer.pad_token_id
                            if self.lm.tokenizer.pad_token_id is not None
                            else self.lm.tokenizer.eos_token_id
                        )
                        cur_input_ids += [pad_token_id] * padding_size
                        cur_labels += [padding_index] * padding_size

                    input_ids.append(cur_input_ids)
                    attention_mask.append([1] * len(cur_labels))
                    labels.append(cur_labels)

                result = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }
                return result

            with training_args.main_process_first(desc="grouping texts together"):
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    load_from_cache_file=True,
                    desc=f"Grouping texts in chunks of {block_size}",
                )

            for split, data in lm_datasets.items():
                total_eval_tokens = 0
                for chunk in data["labels"]:
                    total_eval_tokens += len([x for x in chunk[1:] if x != padding_index])
                logger.info(f"[{split}] Total eval tokens: {total_eval_tokens}")
                if knn_args.dstore_size is None and split == "train":
                    knn_args.dstore_size = total_eval_tokens  
            
            #! ======================================================
            #! Use Trainer to process the whole dataset and get the datastore
            #! ======================================================
            print("==> Trainer evaluating...")
            eval_dataset = lm_datasets["train"]
            
            trainer = Trainer(
                model=self.lm.model,
                args=training_args,
                eval_dataset=eval_dataset,
                tokenizer=self.lm.tokenizer,
                # Data collator will default to DataCollatorWithPadding, so we change it.
                data_collator=default_data_collator,
            )
            
            self.knn_wrapper = KNNSaver(
                dstore_size=knn_args.dstore_size, 
                dstore_dir=datastore_path, 
                dimension=self.lm.model.config.hidden_size,
                knn_keytype=KEY_TYPE.last_ffn_input,
            )
            self.knn_wrapper.break_into(self.lm.model)
            
            metrics = trainer.evaluate()
            
            max_eval_samples = (
                knn_args.max_eval_samples
                if knn_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity

            knn_metrics = self.knn_wrapper.get_metrics()
            metrics.update(knn_metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            
            #! ======================================================
            #! Build index
            #! ======================================================
            print("==> Building index...")
            self.knn_wrapper.build_index()
            
            #! ======================================================
            #! Save
            #! ======================================================
            extra_info = {"dstore_size": knn_args.dstore_size}
            with open(os.path.join(datastore_path, "extra_info.json"), "w") as f:
                json.dump(extra_info, f)
        else:
            with open(os.path.join(datastore_path, "extra_info.json"), "r") as f:
                extra_info = json.load(f)
                knn_args.dstore_size = extra_info["dstore_size"]

            self.knn_wrapper = KNNWrapper(
                dstore_size=knn_args.dstore_size, 
                dstore_dir=datastore_path,
                dimension=self.lm.model.config.hidden_size,
                knn_sim_func=DIST.l2,
                knn_keytype=KEY_TYPE.last_ffn_input,
                no_load_keys=True, 
                move_dstore_to_mem=True, 
                knn_gpu=True,
                recompute_dists=False,
                k=knn_args.k_for_knn, 
                lmbda=knn_args.lmbda, 
                knn_temp=knn_args.knn_temp, 
                probe=knn_args.probe
            )
            self.knn_wrapper.break_into(self.lm.model)
    
    def generate(self, query: str, compute_generation_scores=False, compute_input_loss=False):
        lm_input = query
        output_dict = self.lm.generate(lm_input, compute_generation_scores, compute_input_loss)
        return output_dict
    
    def finish(self):
        self.knn_wrapper.break_out()
        return