from __future__ import annotations

import json
import sys
from pathlib import Path

from datetime import datetime 

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from deepeval.test_case import LLMTestCase
from src.agents import ask_with_tools
from src.pipeline import run_ingestion
from src.vectorstore import get_collection

from langfuse import get_client, propagate_attributes 
 
from evals.metrics.rule_based_metrics import RuleBasedMetrics
from evals.metrics.heuristic_metrics import HeuristicMetrics
from evals.metrics.rag_metrics import RagMetrics
from evals.metrics.safety_metrics import SafetyMetrics
from evals.metrics.llm_as_juge_metrics import EvalMetrics

GOLDEN_DATASET_PATH = ROOT_DIR / "evals" / "data" / "golden_dataset.json"
RESULTS_DIR = ROOT_DIR / "evals" / "results"

RETRIEVAL_TOP_K = 4

PASS_THRESHOLD = 0.8

MIN_PASS_RATE = 0.8

client = get_client() 

def load_golden_dataset() -> list[dict]:
    with GOLDEN_DATASET_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def normalize_source(source: str) -> str:
    normalized = source.replace("\\", "/")
    if normalized.startswith("datasets/"):
        normalized = normalized[len("datasets/"):]
    return normalized


def retrieve_context(query: str) -> tuple[list[str], list[str]]:


# Input:   "Who is the sales rep for the Snowflake Platinum account?"

# query()  → 4 nearest chunks
# Output:
#   texts   = ["Snowflake Platinum account — rep: …",
#              "CRM notes: Snowflake renewal …",
#              "Platinum tier includes …",
#              "Sales team directory …"]
#   sources = ["accounts/snowflake.md",
#              "accounts/crm_notes.md",
#              "pricing/tiers.md",
#              "team/directory.md"]
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=RETRIEVAL_TOP_K)
    texts = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    sources = [normalize_source(meta.get("source", "")) for meta in metadatas]
    return texts, sources


def score_case(case: dict) -> dict:
    query = case["query"]
    expected_answer = case["expected_answer"]
    expected_keywords = case["expected_keywords"]
    expected_tool = case.get("expected_tool")
    expected_sources = [normalize_source(s) for s in case["expected_sources"]]

    with client.start_as_current_observation(
        name=f"eval-case: {case['id']}", as_type="span"
    ) as trace:

        actual_answer, actual_tools = ask_with_tools(query, online_eval=False)
        actual_tool = actual_tools[0] if actual_tools else None

        retrieved_texts, retrieved_sources = retrieve_context(query)

        keyword_result = RuleBasedMetrics.keyword_coverage(actual_answer, expected_keywords)
        tool_result = RuleBasedMetrics.tool_call_accuracy(actual_tool, expected_tool)

        rouge_result = HeuristicMetrics.rouge_score(actual_answer, expected_answer)
        length_result = HeuristicMetrics.length_penalty(actual_answer)

        context_precision_result = RagMetrics.context_precision(retrieved_sources, expected_sources)
        context_recall_result = RagMetrics.context_recall(retrieved_sources, expected_sources)

        toxicity_result = SafetyMetrics.toxicity_score(actual_answer)

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_answer,
            expected_output=expected_answer,
            retrieval_context=retrieved_texts,
        )
        correctness_metric = EvalMetrics.correctness_metric()
        correctness_metric.measure(test_case)
        relevance_metric = EvalMetrics.relevance_metric()
        relevance_metric.measure(test_case)

        scores = {
            "keyword_coverage": keyword_result.score,
            "tool_call_accuracy": tool_result.score,
            "rouge": rouge_result.score,
            "length_penalty": length_result.score,
            "context_precision": context_precision_result.score,
            "context_recall": context_recall_result.score,
            "toxicity": toxicity_result.score,
            "correctness": correctness_metric.score,
            "relevance": relevance_metric.score,
        }

        passed = scores["correctness"] >= PASS_THRESHOLD and scores["tool_call_accuracy"] == 1.0

        trace.update(input=query, output=actual_answer)

        for metric_name, metric_value in scores.items():
            client.score_current_trace(name=metric_name, value=metric_value, data_type="NUMERIC")
        client.score_current_trace(name="passed", value=passed, data_type="BOOLEAN")

    return {
            "id": case["id"],
            "category": case["category"],
            "query": query,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "actual_tool": actual_tool,
            "retrieved_sources": retrieved_sources,
            "scores": scores,
            "passed": passed,
        }


def summarize(results: list[dict]) -> dict:
    metric_names = list(results[0]["scores"].keys())
    averages = {
        name: round(sum(r["scores"][name] for r in results) / len(results), 3)
        for name in metric_names
    }
    passed_count = sum(1 for r in results if r["passed"])
    return {
        "total_cases": len(results),
        "passed_cases": passed_count,
        "pass_rate": round(passed_count / len(results), 3),
        "average_scores": averages,
    }


def run() -> None:
    run_ingestion()

    cases = load_golden_dataset()

    session_id = f"eval-run-{datetime.now():%Y%m%d-%H%M%S}"

    results = []

    with propagate_attributes(session_id=session_id, tags=["eval-run"]):
        for case in cases:
            print(f"Running case: {case['id']}")
            result = score_case(case)
            results.append(result)

        summary = summarize(results)

        with client.start_as_current_observation(
            name="eval-summary", as_type="span"
        ) as summary_trace:
            summary_trace.update(input=f"{len(results)} cases", output=summary)
            for metric_name, avg_value in summary["average_scores"].items():
                 client.score_current_trace(name=f"avg_{metric_name}", value=avg_value, data_type="NUMERIC")
            client.score_current_trace(name="pass_rate", value=summary["pass_rate"], data_type="NUMERIC")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "latest.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": results}, f, indent=2)

    print(f"\nSaved {len(results)} results to {output_path}")
    print(f"Pass rate: {summary['pass_rate']:.0%} ({summary['passed_cases']}/{summary['total_cases']})")
    print(f"Average scores: {summary['average_scores']}")

    client.flush()

    # for CI
    if summary["pass_rate"] < MIN_PASS_RATE:
        print(
            f"FAILED: pass rate {summary['pass_rate']:.0%} is below the "
            f"required {MIN_PASS_RATE:.0%} threshold."
        )
        sys.exit(1)


if __name__ == "__main__":
    run()
