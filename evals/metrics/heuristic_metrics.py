from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from src import config


@dataclass
class MetricResult:
    name: str
    score: float
    note: str


@lru_cache(maxsize=1)
def _get_openai_client():
    """Create the OpenAI client once and reuse it (same singleton pattern as get_collection())."""
    from openai import OpenAI

    return OpenAI(api_key=config.OPENAI_API_KEY)


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, dropping punctuation.

    input:  "The cat sat on the mat."
    output: ['the', 'cat', 'sat', 'on', 'the', 'mat']
    """
    return re.findall(r"\w+", text.lower())


def _embed_tokens(tokens: list[str]) -> dict[str, np.ndarray]:
    """Embed each unique token with the project's OpenAI embedding model in ONE API call.

    input:  ['the', 'cat', 'sat', 'on', 'the', 'mat', 'a', 'is', 'sitting']
    output: {'the': array([...1536 floats...]), 'cat': array([...]), ...}   (unit-length vectors)
    """
    unique = sorted(set(tokens))
    response = _get_openai_client().embeddings.create(model=config.OPENAI_EMBEDDING_MODEL, input=unique)
    vectors = {}
    for token, item in zip(unique, response.data):
        vector = np.array(item.embedding, dtype=np.float32)
        vectors[token] = vector / (np.linalg.norm(vector) or 1.0)
    return vectors


class HeuristicMetrics:
    """Heuristic metrics for lightweight, non-LLM checks such as overlap and length-based scoring."""

    # candidate = model output, reference is the actual_output or ground truth
    @staticmethod
    def bleu_score(candidate: str, reference: str) -> MetricResult:
        """
        Purpose: Approximate the BLEU metric by measuring word overlap between the answer and the reference.
        What it does: Splits both texts into lowercase words and counts how many candidate words also appear in the reference.
        Returns: A MetricResult whose score is the overlap ratio (matched candidate words divided by total candidate words), higher is better.
        """
        # Split the candidate answer into words.
        # input: candidate = "The cat sat on the mat"
        # output: candidate_words = ['the', 'cat', 'sat', 'on', 'the', 'mat']
        candidate_words = candidate.lower().split()
        # Split the reference answer into words.
        # input: reference = "A cat is sitting on the mat"
        # output: reference_words = ['a', 'cat', 'is', 'sitting', 'on', 'the', 'mat']
        reference_words = reference.lower().split()
        # Count how many candidate words appear in the reference.
        # input: candidate_words = ['the', 'cat', 'sat', 'on', 'the', 'mat'], reference_words = ['a', 'cat', 'is', 'sitting', 'on', 'the', 'mat']
        # output: overlap = 5
        overlap = sum(1 for word in candidate_words if word in reference_words)
        # Compute the overlap ratio.
        # input: overlap = 5, candidate_words length = 6
        # output: score = 0.833
        score = overlap / max(len(candidate_words), 1)
        # Return the result as a MetricResult object.
        # input: score = 0.833
        # output: MetricResult(name="BLEU", score=0.833, note="Higher is better.")
        return MetricResult(name="BLEU", score=round(score, 3), note="Higher is better.")

    @staticmethod
    def rouge_score(candidate: str, reference: str) -> MetricResult:
        """
        Purpose: Approximate the ROUGE metric by measuring how much of the reference is covered by the answer.
        What it does: Turns both texts into sets of unique lowercase words and computes the overlap relative to the reference.
        Returns: A MetricResult whose score is the overlap ratio (shared words divided by reference words), higher is better.
        """
        # Turn the candidate answer into a set of words.
        # input: candidate = "The cat sat on the mat"
        # output: candidate_words = {'the', 'cat', 'sat', 'on', 'mat'}
        candidate_words = set(candidate.lower().split())
        # Turn the reference answer into a set of words.
        # input: reference = "A cat is sitting on the mat"
        # output: reference_words = {'a', 'cat', 'is', 'sitting', 'on', 'the', 'mat'}
        reference_words = set(reference.lower().split())
        # Compute the overlap between the two sets.
        # input: candidate_words = {'the', 'cat', 'sat', 'on', 'mat'}, reference_words = {'a', 'cat', 'is', 'sitting', 'on', 'the', 'mat'}
        # output: score = 0.571 (4 shared words / 7 reference words)
        score = len(candidate_words & reference_words) / max(len(reference_words), 1)
        # Return the result as a MetricResult object.
        # input: score = 0.571
        # output: MetricResult(name="ROUGE", score=0.571, note="Higher is better.")
        return MetricResult(name="ROUGE", score=round(score, 3), note="Higher is better.")

    @staticmethod
    def meteor_score(candidate: str, reference: str) -> MetricResult:
        """
        Purpose: Approximate the METEOR metric with a slightly more flexible word-overlap check.
        What it does: Turns both texts into sets of unique lowercase words and computes the overlap relative to the reference.
        Returns: A MetricResult whose score is the overlap ratio (shared words divided by reference words), higher is better.
        """
        # Turn the candidate answer into a set of words.
        # input: candidate = "The cat sat on the mat"
        # output: candidate_words = {'the', 'cat', 'sat', 'on', 'mat'}
        candidate_words = set(candidate.lower().split())
        # Turn the reference answer into a set of words.
        # input: reference = "A cat is sitting on the mat"
        # output: reference_words = {'a', 'cat', 'is', 'sitting', 'on', 'the', 'mat'}
        reference_words = set(reference.lower().split())
        # Compute the overlap between the two sets.
        # input: candidate_words = {'the', 'cat', 'sat', 'on', 'mat'}, reference_words = {'a', 'cat', 'is', 'sitting', 'on', 'the', 'mat'}
        # output: score = 0.571 (4 shared words / 7 reference words)
        score = len(candidate_words & reference_words) / max(len(reference_words), 1)
        # Return the result as a MetricResult object.
        # input: score = 0.571
        # output: MetricResult(name="METEOR", score=0.571, note="Higher is better.")
        return MetricResult(name="METEOR", score=round(score, 3), note="Higher is better.")

    @staticmethod
    def bertscore_score(candidate: str, reference: str) -> MetricResult:
        """
        Purpose: Measure semantic similarity between the answer and the reference, BERTScore-style,
                 using the project's OpenAI embedding model (config.OPENAI_EMBEDDING_MODEL).
        What it does: Embeds every word of both texts, builds a cosine-similarity grid between candidate
                      words and reference words, matches each word to its most similar word on the other
                      side (greedy matching), and combines the best matches into precision, recall and F1.
        Note: Original BERTScore uses contextual token embeddings from a local BERT model. The OpenAI API
              returns one vector per input string, so each word is embedded on its own (no sentence
              context) and there is no baseline rescaling. The matching and P/R/F1 math are the same.
        Returns: A MetricResult whose score is the F1 (0.0-1.0), higher is better. P and R are in the note.
        """
        # Split both texts into word tokens.
        # input: candidate = "The cat sat on the mat", reference = "A cat is sitting on the mat"
        # output: cand_tokens = ['the', 'cat', 'sat', 'on', 'the', 'mat']
        #         ref_tokens  = ['a', 'cat', 'is', 'sitting', 'on', 'the', 'mat']
        cand_tokens = _tokenize(candidate)
        ref_tokens = _tokenize(reference)
        # Guard against empty text: there are no tokens to compare.
        # input: cand_tokens = []
        # output: MetricResult(name="BERTScore", score=0.0, note="Empty candidate or reference.")
        if not cand_tokens or not ref_tokens:
            return MetricResult(name="BERTScore", score=0.0, note="Empty candidate or reference.")
        # Embed all unique words from both texts in a single OpenAI call.
        # input: cand_tokens + ref_tokens (9 unique words)
        # output: vectors = {'the': [...], 'cat': [...], 'sat': [...], ...}
        vectors = _embed_tokens(cand_tokens + ref_tokens)
        # Stack the vectors into matrices, one row per token (duplicates kept, as in BERTScore).
        # input: cand_tokens (6 words), ref_tokens (7 words)
        # output: cand_matrix shape = (6, 1536), ref_matrix shape = (7, 1536)
        cand_matrix = np.stack([vectors[t] for t in cand_tokens])
        ref_matrix = np.stack([vectors[t] for t in ref_tokens])
        # Cosine similarity of every candidate word with every reference word.
        # Vectors are unit length, so the dot product IS the cosine similarity.
        # input: cand_matrix (6, 1536), ref_matrix (7, 1536)
        # output: similarity shape = (6, 7); e.g. similarity['sat']['sitting'] is high, ['sat']['a'] is low
        similarity = cand_matrix @ ref_matrix.T
        # Precision: for each candidate word, its best match in the reference, averaged.
        # input: row maxima, e.g. [1.0, 1.0, 0.7, 1.0, 1.0, 1.0]   ('sat' best-matches 'sitting')
        # output: precision = mean of row maxima
        precision = float(similarity.max(axis=1).mean())
        # Recall: for each reference word, its best match in the candidate, averaged.
        # input: column maxima, e.g. [0.5, 1.0, 0.4, 0.7, 1.0, 1.0, 1.0]   ('a' and 'is' have no close match)
        # output: recall = mean of column maxima
        recall = float(similarity.max(axis=0).mean())
        # F1: harmonic mean of precision and recall, clamped to 0.0-1.0.
        # input: precision, recall
        # output: score = 2 * P * R / (P + R)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        score = min(max(f1, 0.0), 1.0)
        # Return the result as a MetricResult object.
        # input: score = f1, precision, recall
        # output: MetricResult(name="BERTScore", score=f1, note="F1 (OpenAI embeddings). P=..., R=... ")
        return MetricResult(
            name="BERTScore",
            score=round(score, 3),
            note=f"F1 (OpenAI {config.OPENAI_EMBEDDING_MODEL}). P={precision:.3f}, R={recall:.3f}. Higher is better.",
        )

    @staticmethod
    def length_penalty(text: str, min_words: int = 20) -> MetricResult:
        """
        Purpose: Check whether the response is long enough to be considered substantive.
        What it does: Counts the words in the response and compares the count against a minimum-word threshold.
        Returns: A MetricResult whose score is 1.0 if the response meets the minimum word count, otherwise 0.0.
        """
        # Count the words in the response.
        # input: text = "one two three four five"
        # output: word_count = 5
        word_count = len(text.split())
        # Give a full score when the answer is long enough.
        # input: word_count = 5, min_words = 20
        # output: score = 0.0
        score = 1.0 if word_count >= min_words else 0.0
        # Return the result as a MetricResult object.
        # input: score = 0.0
        # output: MetricResult(name="Length Penalty", score=0.0, note="Higher is better.")
        return MetricResult(name="Length Penalty", score=round(score, 3), note="Higher is better.")