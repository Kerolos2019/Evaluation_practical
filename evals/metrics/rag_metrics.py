from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class MetricResult:
    name: str
    score: float
    note: str


class RagMetrics:
    """Retrieval and RAG-style metrics for context relevance and ranking quality."""

    @staticmethod
    def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> MetricResult:
        """
        Purpose: Check how many of the top-K retrieved items are actually relevant.
        What it does: Takes the first K retrieved items and counts how many of them appear in the relevant-items list.
        Returns: A MetricResult whose score is the precision ratio (relevant hits divided by K), higher is better.
        """
        # Keep only the first K retrieved items.
        # input: retrieved = ["doc1", "doc2", "doc3", "doc4"], k = 3
        # output: retrieved_k = ["doc1", "doc2", "doc3"]
        retrieved_k = list(retrieved[:k])
        # Count how many of those items are relevant.
        # input: retrieved_k = ["doc1", "doc2", "doc3"], relevant = ["doc2", "doc4", "doc5"]
        # output: hits = 1
        hits = sum(1 for item in retrieved_k if item in relevant)
        # Compute the precision score.
        # input: hits = 1, k = 3
        # output: score = 0.333
        score = hits / k if k else 0.0
        # Return the result as a MetricResult object.
        # input: score = 0.333
        # output: MetricResult(name="Precision@K", score=0.333, note="Higher is better.")
        return MetricResult(name="Precision@K", score=round(score, 3), note="Higher is better.")

    @staticmethod
    def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> MetricResult:
        """
        Purpose: Check how many of the relevant items were successfully retrieved within the top K.
        What it does: Takes the first K retrieved items and counts how many distinct relevant items were found among them.
        Returns: A MetricResult whose score is the recall ratio (relevant items found divided by total relevant items), higher is better.
        """
        # Keep only the first K retrieved items.
        # input: retrieved = ["doc1", "doc2", "doc3", "doc4"], k = 3
        # output: retrieved_k = ["doc1", "doc2", "doc3"]
        retrieved_k = list(retrieved[:k])
        # Count how many relevant items were found at least once (not how many times).
        # input: relevant = ["doc2", "doc4", "doc5"], retrieved_k = ["doc1", "doc2", "doc3"]
        # output: hits = 1
        hits = sum(1 for item in relevant if item in retrieved_k)
        # Compute the recall score.
        # input: hits = 1, len(relevant) = 3
        # output: score = 0.333
        score = hits / len(relevant) if relevant else 0.0
        # Return the result as a MetricResult object.
        # input: score = 0.333
        # output: MetricResult(name="Recall@K", score=0.333, note="Higher is better.")
        return MetricResult(name="Recall@K", score=round(score, 3), note="Higher is better.")

    @staticmethod
    def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> MetricResult:
        """
        Purpose: Check whether the first relevant result appears early in the ranked results (Mean Reciprocal Rank).
        What it does: Scans the retrieved items in order and, as soon as a relevant one is found, takes the reciprocal of its position.
        Returns: A MetricResult whose score is 1 divided by the rank of the first relevant hit (0.0 if none was found), higher is better.
        """
        # Check each retrieved item from the top of the list.
        # input: retrieved = ["doc1", "doc2", "doc3", "doc4"], relevant = ["doc2", "doc4", "doc5"]
        # output: first relevant hit found at index = 2 ("doc2")
        for index, item in enumerate(retrieved, start=1):
            # Stop as soon as a relevant item is found.
            # input: item = "doc2", index = 2
            # output: True (doc2 is in relevant)
            if item in relevant:
                # Return a reciprocal-rank score for that position.
                # input: index = 2
                # output: MetricResult(name="MRR", score=0.5, note="Higher is better.")
                return MetricResult(name="MRR", score=round(1.0 / index, 3), note="Higher is better.")
        # Return zero if no relevant result was found.
        # input: retrieved = ["doc1", "doc3"], relevant = ["doc2", "doc4", "doc5"]
        # output: MetricResult(name="MRR", score=0.0, note="No relevant result was found.")
        return MetricResult(name="MRR", score=0.0, note="No relevant result was found.")

    @staticmethod
    def ndcg(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> MetricResult:
        """
        Purpose: Check whether relevant items appear high in the ranking (Normalized Discounted Cumulative Gain).
        What it does: Computes the discounted cumulative gain of the top-K retrieved items (relevant items ranked higher score more), then divides by the gain of an ideal ranking to normalize it.
        Returns: A MetricResult whose score is the normalized ranking quality (0.0-1.0), higher is better.
        """
        # Start the discounted cumulative gain at zero.
        # input: (none)
        # output: dcg = 0.0
        dcg = 0.0
        # Add value for each relevant item based on its rank.
        # input: retrieved[:k] = ["doc1", "doc2", "doc3"], relevant = ["doc2", "doc4", "doc5"]
        # output: dcg = 0.5 (only "doc2" at rank 2 is relevant: 1/2)
        for index, item in enumerate(retrieved[:k], start=1):
            # Increase the score when the item is relevant.
            # input: item = "doc2", index = 2
            # output: dcg += 0.5
            if item in relevant:
                dcg += 1.0 / index

        # Start the ideal ranking score at zero.
        # input: (none)
        # output: ideal = 0.0
        ideal = 0.0
        # Build the best possible ranking for comparison.
        # input: k = 3, len(relevant) = 3
        # output: ideal = 1.833 (1/1 + 1/2 + 1/3)
        for index in range(1, min(k, len(relevant)) + 1):
            ideal += 1.0 / index

        # Normalize the score by the ideal ranking.
        # input: dcg = 0.5, ideal = 1.833
        # output: score = 0.273
        score = dcg / ideal if ideal else 0.0
        # Return the result as a MetricResult object.
        # input: score = 0.273
        # output: MetricResult(name="NDCG", score=0.273, note="Higher is better.")
        return MetricResult(name="NDCG", score=round(score, 3), note="Higher is better.")

    @staticmethod
    def context_precision(retrieved: Sequence[str], relevant: Sequence[str]) -> MetricResult:
        """
        Purpose: Check whether the retrieved context is mostly relevant (signal-to-noise of retrieval).
        What it does: Counts how many of the retrieved items are relevant, out of all retrieved items.
        Returns: A MetricResult whose score is the precision ratio (relevant items divided by total retrieved items), higher is better.
        """
        # Return zero if nothing was retrieved.
        # input: retrieved = []
        # output: MetricResult(name="Context Precision", score=0.0, note="No context was retrieved.")
        if not retrieved:
            return MetricResult(name="Context Precision", score=0.0, note="No context was retrieved.")
        # Count how many retrieved items are relevant.
        # input: retrieved = ["doc1", "doc2", "doc3", "doc4"], relevant = ["doc2", "doc4", "doc5"]
        # output: hits = 2
        hits = sum(1 for item in retrieved if item in relevant)
        # Compute the ratio of relevant items.
        # input: hits = 2, len(retrieved) = 4
        # output: score = 0.5
        score = hits / len(retrieved)
        # Return the result as a MetricResult object.
        # input: score = 0.5
        # output: MetricResult(name="Context Precision", score=0.5, note="Higher is better.")
        return MetricResult(name="Context Precision", score=round(score, 3), note="Higher is better.")

    @staticmethod
    def context_recall(retrieved: Sequence[str], relevant: Sequence[str]) -> MetricResult:
        """
        Purpose: Check whether the retrieved context covers all the evidence needed to answer the question.
        What it does: Counts how many distinct relevant items were found among the retrieved items, out of all relevant items.
        Returns: A MetricResult whose score is the recall ratio (relevant items covered divided by total relevant items), higher is better.
        """
        # Return zero if no relevant evidence is defined.
        # input: relevant = []
        # output: MetricResult(name="Context Recall", score=0.0, note="No relevant evidence was defined.")
        if not relevant:
            return MetricResult(name="Context Recall", score=0.0, note="No relevant evidence was defined.")
        # Count how many relevant items were found at least once (not how many times).
        # input: relevant = ["doc2", "doc4", "doc5"], retrieved = ["doc1", "doc2", "doc3", "doc4"]
        # output: hits = 2
        hits = sum(1 for item in relevant if item in retrieved)
        # Compute the ratio of relevant evidence covered.
        # input: hits = 2, len(relevant) = 3
        # output: score = 0.667
        score = hits / len(relevant)
        # Return the result as a MetricResult object.
        # input: score = 0.667
        # output: MetricResult(name="Context Recall", score=0.667, note="Higher is better.")
        return MetricResult(name="Context Recall", score=round(score, 3), note="Higher is better.")

    @staticmethod
    def context_relevance(retrieved: Sequence[str], question: str) -> MetricResult:
        """
        Purpose: Check whether the retrieved context is topically related to the question that was asked.
        What it does: Compares the question's words against each retrieved item's words and averages the word-overlap ratio across all retrieved items.
        Returns: A MetricResult whose score is the average overlap ratio between the question and the retrieved context, higher is better.
        """
        # Return zero if no context was retrieved.
        # input: retrieved = []
        # output: MetricResult(name="Context Relevance", score=0.0, note="No context was retrieved.")
        if not retrieved:
            return MetricResult(name="Context Relevance", score=0.0, note="No context was retrieved.")
        # Turn the question into a set of words.
        # input: question = "What is the capital of France"
        # output: q_words = {'what', 'is', 'the', 'capital', 'of', 'france'}
        q_words = set(question.lower().split())
        # Start the score at zero.
        # input: (none)
        # output: score = 0.0
        score = 0.0
        # Compare the question words with each retrieved item.
        # input: retrieved = ["Paris is the capital of France", "Berlin is in Germany"]
        # output: score accumulated across both items = 0.833
        for item in retrieved:
            # Turn the retrieved item into a set of words.
            # input: item = "Paris is the capital of France"
            # output: item_words = {'paris', 'is', 'the', 'capital', 'of', 'france'}
            item_words = set(item.lower().split())
            # Add the overlap ratio for this item.
            # input: q_words = {'what', 'is', 'the', 'capital', 'of', 'france'}, item_words = {'paris', 'is', 'the', 'capital', 'of', 'france'}
            # output: score += 0.667 (4 shared words / 6 question words)
            score += len(q_words & item_words) / max(len(q_words), 1)
        # Average the overlap score across all retrieved items.
        # input: score = 0.833, len(retrieved) = 2
        # output: score = 0.417
        score = score / len(retrieved)
        # Return the result as a MetricResult object.
        # input: score = 0.417
        # output: MetricResult(name="Context Relevance", score=0.417, note="Higher is better.")
        return MetricResult(name="Context Relevance", score=round(score, 3), note="Higher is better.")
