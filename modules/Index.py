import sys
sys.path.append(".")

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'False'

import json
import subprocess
import numpy as np
import torch 
from tqdm import tqdm
from typing import List, Dict, Union
from pyserini.search.lucene import LuceneSearcher

from utils.helpers import read_raw_data_dir


class Index(object):
    def __init__(self, raw_data_dir, datastore_dir) -> None:
        assert os.path.exists(raw_data_dir)
        self.raw_data_dir = raw_data_dir
        self.datastore_dir = datastore_dir
    
    def find_most_relevant_k_documents(query: str, k: int):
        raise NotImplementedError


class BM25Index(Index):
    def __init__(self, tokenizer, max_retrieval_seq_length: int, stride: int,
                 raw_data_dir, datastore_dir, recursive=True) -> None:
        super().__init__(raw_data_dir, datastore_dir)
        
        self.tokenizer = tokenizer
        self.max_retrieval_seq_length = max_retrieval_seq_length
        self.stride = stride
        
        if (not os.path.exists(datastore_dir)) or (len(os.listdir(datastore_dir)) == 0):
            os.makedirs(datastore_dir, exist_ok=True)
            
            #! step 1: tokenize raw data 
            print("==> Reading and tokenizing raw data...")
            data = read_raw_data_dir(raw_data_dir=raw_data_dir, recursive=recursive)
            # todo: process very long text?
            all_text = " ".join(data)
            
            all_words = all_text.split()
            step_size = 1024
            chunks_to_tokenize = [all_words[i:i + step_size] for i in range(0, len(all_words), step_size)]
            chunks_to_tokenize = [" ".join(chunk) for chunk in chunks_to_tokenize]
            
            final_tokens = []
            for chunk in tqdm(chunks_to_tokenize):
                tokenizer.parallelism = 8
                tokenized_data = tokenizer(chunk)['input_ids']
                final_tokens.extend(tokenized_data)
            final_tokens = np.array(final_tokens)
            print(f"==> Number of tokens: {len(final_tokens)}.")
            
            #! step 2: split tokenized data into chunks
            print("==> Making chunks...")
            tokens_as_chunks = self._get_token_chunks(
                final_tokens, 
                pad_token=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
            print(f"==> {len(tokens_as_chunks)} chunks in total.")

            self.tokens_dir = os.path.join(datastore_dir, "tokens")
            os.makedirs(self.tokens_dir, exist_ok=True)
            with open(os.path.join(self.tokens_dir, "data.jsonl"), "w") as f:
                for chunk_id, token_chunk in enumerate(tokens_as_chunks):
                    assert len(token_chunk) <= max_retrieval_seq_length
                    text = tokenizer.decode(token_chunk)
                    f.write(json.dumps({
                        "id": str(chunk_id),
                        "contents": text,
                        "input_ids": token_chunk.tolist()
                    })+"\n")
        
            #! step 3: build index on the datastore
            print("==> Start building index for %s at %s" % (self.tokens_dir, datastore_dir))
            command = """python -m pyserini.index.lucene \
            --collection JsonCollection \
            --input '%s' \
            --index '%s' \
            --generator DefaultLuceneDocumentGenerator \
            --storeRaw --threads 1""" % (self.tokens_dir, datastore_dir)
            ret_code = subprocess.run([command],
                                      shell=True,
                                      # stdout=subprocess.DEVNULL,
                                      # stderr=subprocess.STDOUT
                                      )
            if ret_code.returncode != 0:
                print("Failed to build the index")
                exit()
            else:
                print("Successfully built the index")
        else:
            print("==> Datastore exists at: ", datastore_dir)
        
        self.searcher = LuceneSearcher(datastore_dir)
    
    def _get_token_chunks(self, tokens: np.ndarray, pad_token: int) -> np.ndarray:
        assert tokens.ndim == 1, "Tokens should be flattened first!"
        num_tokens = len(tokens)
        tokens_as_chunks = []
        
        for begin_loc in range(0, num_tokens, self.stride):
            end_loc = min(begin_loc + self.max_retrieval_seq_length, num_tokens)
            token_chunk = tokens[begin_loc:end_loc].copy()
        
            if end_loc == num_tokens and len(token_chunk) < self.max_retrieval_seq_length:
                pads = np.array([pad_token for _ in range(self.max_retrieval_seq_length - len(token_chunk))])
                token_chunk = np.concatenate([token_chunk, pads])
        
            assert len(token_chunk) == self.max_retrieval_seq_length
            
            tokens_as_chunks.append(token_chunk)
        
        tokens_as_chunks = np.stack(tokens_as_chunks)
        return tokens_as_chunks

    def find_most_relevant_k_documents(self, query: str, k: int) -> List[str]:
        hits = self.searcher.search(query, k=k)
        docs = []
        for hit in hits:
            docid = hit.docid
            raw = self.searcher.doc(docid).raw()
            input_ids = json.loads(raw)["input_ids"]
            doc_str = self.tokenizer.decode(input_ids)
            docs.append(doc_str)
        return docs
    
    
if __name__ == '__main__':
    pass