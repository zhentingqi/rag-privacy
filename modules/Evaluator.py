import os
import json
import nltk
from evaluate import load
import numpy as np
import scipy
from typing import Dict, List, Union
from collections import defaultdict, Counter
import statistics
rouge = load("rouge")
sacrebleu = load("sacrebleu")
bertscore = load("bertscore")


class Evaluator:
    def __init__(self, predictions_str: List[str], references_str: List[str]) -> None:
        self.metric_rouge = rouge
        self.metric_bleu = sacrebleu
        self.metric_bertscore = bertscore
        self.predictions_str = predictions_str
        self.references_str = references_str

    def compute_metrics(self):
        return self._text_comparison_metrics()

    def _text_comparison_metrics(self) -> Dict[str, float]:
        # modify code from: https://github.com/jxmorris12/vec2text/blob/master/vec2text/trainers/base.py
        def mean(L: Union[List[int], List[float]]) -> float:
            assert len(L) > 0
            return sum(L) / len(L)

        def sem(L: List[float]) -> float:
            assert len(L) > 0
            result = scipy.stats.sem(np.array(L))
            if isinstance(result, np.ndarray):
                return result.mean().item()
            return result

        def count_overlapping_ngrams(s1: str, s2: str, n: int) -> int:
            ngrams_1 = nltk.ngrams(s1, n)
            ngrams_2 = nltk.ngrams(s2, n)
            ngram_counts_1 = Counter(ngrams_1)
            ngram_counts_2 = Counter(ngrams_2)
            total = 0
            for ngram, count in ngram_counts_1.items():
                total += min(count, ngram_counts_2[ngram])
            return total

        assert len(self.predictions_str) == len(self.references_str)
        num_preds = len(self.predictions_str)
        if not num_preds:
            return {}

        ###########################################################
        # Compute token, precision, recall, and ngram-level metrics.
        precision_sum = 0.0
        recall_sum = 0.0
        num_overlapping_words = []
        num_overlapping_bigrams = []
        num_overlapping_trigrams = []
        num_true_words = []
        num_pred_words = []
        f1s = []
        for i in range(num_preds):  # for each prediction
            true_words = nltk.tokenize.word_tokenize(self.references_str[i])
            pred_words = nltk.tokenize.word_tokenize(self.predictions_str[i])
            num_true_words.append(len(true_words))
            num_pred_words.append(len(pred_words))
            true_words_set = set(true_words)
            pred_words_set = set(pred_words)

            TP = len(true_words_set & pred_words_set)
            FP = len(true_words_set) - len(true_words_set & pred_words_set)
            FN = len(pred_words_set) - len(true_words_set & pred_words_set)

            precision = (TP) / (TP + FP + 1e-20)
            recall = (TP) / (TP + FN + 1e-20)
            try:
                f1 = (2 * precision * recall) / (precision + recall + 1e-20)
            except ZeroDivisionError:
                f1 = 0.0

            precision_sum += precision
            recall_sum += recall
            f1s.append(f1)

            ############################################################
            num_overlapping_words.append(
                count_overlapping_ngrams(true_words, pred_words, 1)
            )
            num_overlapping_bigrams.append(
                count_overlapping_ngrams(true_words, pred_words, 2)
            )
            num_overlapping_trigrams.append(
                count_overlapping_ngrams(true_words, pred_words, 3)
            )

        set_token_metrics = {
            "token_set_precision": (precision_sum / num_preds),
            "token_set_recall": (recall_sum / num_preds),
            "token_set_f1": mean(f1s),
            "token_set_f1_sem": sem(f1s),
            "n_ngrams_match_1": mean(num_overlapping_words),
            "n_ngrams_match_2": mean(num_overlapping_bigrams),
            "n_ngrams_match_3": mean(num_overlapping_trigrams),
            "num_true_words": mean(num_true_words),
            "num_pred_words": mean(num_pred_words),
        }
        ############################################################
        bleu_results = np.array(
            [
                self.metric_bleu.compute(predictions=[p], references=[r])["score"]
                for p, r in zip(self.predictions_str, self.references_str)
            ]
        )
        rouge_results = self.metric_rouge.compute(
            predictions=self.predictions_str, references=self.references_str, use_aggregator=False
        )
        bertscore_results = self.metric_bertscore.compute(
            predictions=self.predictions_str, references=self.references_str, lang="en"
        )
        exact_matches = np.array(self.predictions_str) == np.array(self.references_str)
        gen_metrics = {
            "bleu_score": mean(bleu_results),
            "bleu_score_sem": sem(bleu_results),
            "rougeL_score": mean(rouge_results["rougeL"]),  
            "rougeL_score_sem": sem(rouge_results["rougeL"]),  
            "bert_score": mean(bertscore_results["f1"]),
            "bert_score_sem": sem(bertscore_results["f1"]),
            "exact_match": mean(exact_matches),
            "exact_match_sem": sem(exact_matches),
        }

        all_metrics = {**set_token_metrics, **gen_metrics}
        return all_metrics
