import numpy as np
from .calculator import (
    MeanReciprocalRank,
    PrecisionAtK,
    MeanAveragePrecision,
    EnrichCorrectedMeanReciprocalRank,
    EnrichOriginalMeanReciprocalRank,
    RecallAtK,
)
from typing import List
from collections import Counter


def unique_words_by_frequency(words):
    # Count the frequency of each word in the list
    frequency = Counter(words)
    # Sort the words first by frequency, then by the order they appear in the original list
    sorted_words = sorted(set(words), key=lambda x: (-frequency[x], words.index(x)))
    return sorted_words


class Metric:
    def __init__(self, golds: List[str], preds: List[List[str]], decoding="mean"):
        self.golds = golds
        self.preds = preds
        self.decoding = decoding

    @staticmethod
    def get_hypernyms(line):
        clean_line = line.strip().replace("\n", ",").replace("-", " ").split(",")

        res = []
        for hyp in clean_line:
            if not hyp in ("", " ", ", ", ","):
                res.append(hyp.lower().strip())

        return res

    def get_metrics(self, scores=None, limit=15, return_raw=False):
        if not scores:
            scores = self.default_metrics()

        all_scores = {str(score): [] for score in scores}

        if self.decoding == "concat":
            for goldline, pred_options in zip(self.golds, self.preds):
                # one_line_metrics = {str(score): [] for score in scores}
                cur_portion = []
                for predline in pred_options:
                    cur_portion.extend(self.get_hypernyms(predline))
                cur_pred = unique_words_by_frequency(cur_portion)
                sorted_predline = ",".join(cur_pred)

                one_option_metrics = self.get_one_prediction(
                    goldline, sorted_predline, scores, limit
                )

                # for score in scores:
                #     one_line_metrics[str(score)].append(one_option_metrics[str(score)])

                for key in all_scores:
                    # max_line_value = np.mean(one_line_metrics[key])  # mean to max
                    all_scores[key].append(one_option_metrics[key])

        elif self.decoding == "mean":
            for goldline, pred_options in zip(self.golds, self.preds):
                one_line_metrics = {str(score): [] for score in scores}

                for predline in pred_options:
                    one_option_metrics = self.get_one_prediction(
                        goldline, predline, scores, limit
                    )

                    for score in scores:
                        one_line_metrics[str(score)].append(
                            one_option_metrics[str(score)]
                        )

                for key in all_scores:
                    max_line_value = np.mean(one_line_metrics[key])  # mean to max
                    all_scores[key].append(max_line_value)

        res = {}
        for key in all_scores:
            mean_value = np.mean(all_scores[key])
            res[key] = mean_value

        return all_scores if return_raw else res

    def default_metrics(self):
        scores = [
            MeanReciprocalRank(),
            MeanAveragePrecision(),
            PrecisionAtK(1),
            PrecisionAtK(3),
            PrecisionAtK(5),
            PrecisionAtK(15),
            EnrichCorrectedMeanReciprocalRank(),
            EnrichOriginalMeanReciprocalRank(),
            RecallAtK(1),
            RecallAtK(5),
            RecallAtK(10),
        ]
        return scores

    def get_one_prediction(self, goldline, predline, scores, limit):
        gold_hyps = self.get_hypernyms(goldline)
        pred_hyps = self.get_hypernyms(predline)
        gold_hyps_n = len(gold_hyps)
        r = [0 for i in range(limit)]

        for j in range(min(len(pred_hyps), limit)):
            pred_hyp = pred_hyps[j]
            if pred_hyp in gold_hyps:
                r[j] = 1

        res = {}
        for score in scores:
            if "Enrich" in str(score):
                res[str(score)] = score(pred_hyps, gold_hyps, r)
            else:
                res[str(score)] = score(r, gold_hyps_n)

        return res
