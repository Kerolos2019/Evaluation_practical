

from dataclasses import dataclass 

from typing import Sequence 

@dataclass 
class MetricResult:
    name: str 
    score: float 
    note: str 


class RuleBasedMetrics: 

    @staticmethod 
    def tool_call_accuracy(actual_tool: str | None, expected_tool: str | None) -> MetricResult:

        score = 1.0 if actual_tool and expected_tool and actual_tool == expected_tool else 0.0 
        return MetricResult(name="Tool call accuracy", score=score, note="Higher is better..")

    @staticmethod 
    def keyword_coverage(answer: str, expected_keywords: Sequence[str]):

        answer_lower = answer.lower()

        hits = sum( 1 for keyword in expected_keywords if keyword.lower() in answer_lower )

        # compute the coverage ration 

        score = hits / len(expected_keywords) if expected_keywords else 0.0 

        return MetricResult(name="keyword coverage", score=score, note="Higher is better")

    # check whether a structured output payload contains all the fields it is expected to have 
    @staticmethod 
    def required_fields_present(payload: dict[str, object], required_fields: Sequence[str]) -> MetricResult:

        present = sum(1 for field in required_fields if field in payload)

        score = present / len(required_fields) if required_fields else 0.0 

        return MetricResult(name="Required fields present", score=score, note="Higher is better")

    

