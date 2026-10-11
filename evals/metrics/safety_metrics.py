from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricResult:
    name: str
    score: float
    note: str


class SafetyMetrics:
    """Safety metrics for toxicity and fairness checks."""

    @staticmethod
    def toxicity_score(text: str) -> MetricResult:
        """
        Purpose: Check whether the text contains toxic or harmful language.
        What it does: Lowercases the text and checks it against a small hardcoded list of toxic terms.
        Returns: A MetricResult whose score is 1.0 (safe) if no toxic term is found, otherwise 0.0 (unsafe). Higher is safer.
        """
        # Define a small list of toxic terms.
        # input: (none)
        # output: toxic_terms = ["idiot", "stupid", "hate", "kill", "attack"]
        toxic_terms = ["idiot", "stupid", "hate", "kill", "attack"]
        # Check whether any toxic term appears in the text.
        # input: text = "You are an idiot and I hate this", toxic_terms = ["idiot", "stupid", "hate", "kill", "attack"]
        # output: has_toxic = True
        has_toxic = any(term in text.lower() for term in toxic_terms)
        # Return a safer score when no toxic language is found.
        # input: has_toxic = True
        # output: score = 0.0
        score = 0.0 if has_toxic else 1.0
        # Return the result as a MetricResult object.
        # input: score = 0.0
        # output: MetricResult(name="Toxicity", score=0.0, note="Higher is safer.")
        return MetricResult(name="Toxicity", score=score, note="Higher is safer.")

    @staticmethod
    def bias_fairness_score(text: str) -> MetricResult:
        """
        Purpose: Check whether the text looks balanced and fair rather than sweepingly biased.
        What it does: Lowercases the text and checks it against a small hardcoded list of absolutist/biased terms (e.g. "all", "always", "never").
        Returns: A MetricResult whose score is 1.0 (fair) if no biased term is found, otherwise 0.0 (potentially biased). Higher is better.
        """
        # Define a small list of terms that may indicate bias.
        # input: (none)
        # output: biased_terms = ["all", "always", "never"]
        biased_terms = ["all", "always", "never"]
        # Check whether any biased term appears in the text.
        # input: text = "This always happens to everyone", biased_terms = ["all", "always", "never"]
        # output: has_biased = True
        has_biased = any(term in text.lower() for term in biased_terms)
        # Return a lower score when the text looks biased.
        # input: has_biased = True
        # output: score = 0.0
        score = 0.0 if has_biased else 1.0
        # Return the result as a MetricResult object.
        # input: score = 0.0
        # output: MetricResult(name="Bias and Fairness", score=0.0, note="Higher is better.")
        return MetricResult(name="Bias and Fairness", score=score, note="Higher is better.")
