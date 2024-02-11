import numpy as np


class MeanReciprocalRank:
    """
     Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """

    def __init__(self):
        pass

    def __call__(self, r, *args):
        r = np.asarray(r).nonzero()[0]
        return 1.0 / (r[0] + 1) if r.size else 0.0

    def __str__(self):
        return "MRR"


class EnrichCorrectedMeanReciprocalRank:
    """
     Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """

    def __init__(self):
        pass

    def __call__(self, pred_hyps, gold_hyps, r, *args):
        mean_mrr = 0
        for gold in gold_hyps:
            if gold in pred_hyps:
                rank = pred_hyps.index(gold)
                lefter_positive = sum(r[:rank])
                mean_mrr += 1 / (rank + 1 - lefter_positive)

        return mean_mrr / len(gold_hyps)

    def __str__(self):
        return "EnrichCorrectedMRR"


class EnrichOriginalMeanReciprocalRank:
    """
     Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """

    def __init__(self):
        pass

    def __call__(self, pred_hyps, gold_hyps, r, *args):
        mean_mrr = 0
        for gold in gold_hyps:
            if gold in pred_hyps:
                rank = pred_hyps.index(gold)
                mean_mrr += 1 / (rank + 1)

        return mean_mrr / len(gold_hyps)

    def __str__(self):
        return "EnrichOriginalMRR"


class RecallAtK:
    """
    Score is recall @ k
    Relevance is binary (nonzero is relevant).
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Recall @ k
    Raises:
        ValueError: len(r) must be >= k
    """

    def __init__(self, k):
        self.k = k
        assert k >= 1

    def __call__(self, r, n):
        r = np.asarray(r)[: self.k] != 0
        if r.size != self.k:
            raise ValueError("Relevance score length < k")
        return np.sum(r) / n
        # Modified from the first version. Now the gold elements are taken into account

    def __str__(self):
        return "R@" + str(self.k)


class PrecisionAtK:
    """
    Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """

    def __init__(self, k):
        self.k = k
        assert k >= 1

    def __call__(self, r, n):
        r = np.asarray(r)[: self.k] != 0
        if r.size != self.k:
            raise ValueError("Relevance score length < k")
        return (np.mean(r) * self.k) / min(self.k, n)
        # Modified from the first version. Now the gold elements are taken into account

    def __str__(self):
        return "P@" + str(self.k)


class MeanAveragePrecision:
    """
    Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """

    def __init__(self):
        pass

    def __call__(self, r, n):
        r = np.asarray(r) != 0
        out = [PrecisionAtK(k + 1)(r, n) for k in range(r.size)]
        # Modified from the first version (removed "if r[k]"). All elements (zero and nonzero) are taken into account
        if not out:
            return 0.0
        return np.mean(out)

    def __str__(self):
        return "MAP"
