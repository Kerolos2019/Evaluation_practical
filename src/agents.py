"""Agentic workflow: an LLM agent that can call tools based on the user's question."""

import threading
from functools import lru_cache

from deepeval.test_case import LLMTestCase
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from evals.metrics.llm_as_juge_metrics import EvalMetrics
from evals.metrics.safety_metrics import SafetyMetrics
from src import config
from src.tools import ALL_TOOLS


SYSTEM_PROMPT = """You are a helpful assistant for a company's internal chatbot.
You have access to tools:
- search_knowledge_base: use it to answer questions about company data
  (CRM accounts, opportunities, sales reps, projects, revenue, proposals).
- get_weather: use it for weather related questions.

Only call a tool when it is relevant to the user's question. Answer normally
for general questions that don't require a tool. Always ground answers about
company data in the knowledge base tool's results, and mention when you could
not find relevant information.
"""

client = get_client()
handler = CallbackHandler()


@lru_cache(maxsize=1)
def get_agent():

    model = ChatOpenAI(model=config.OPENAI_CHAT_MODEL, api_key=config.OPENAI_API_KEY)

    return create_agent(model, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)


def _score_online(trace_id: str, question: str, answer: str) -> None:
    """Reference-free metrics for live traffic, where there's no golden answer to
    compare against (unlike evals/run_eval.py). Runs off the request thread so the
    LLM-judge call doesn't add latency to the user-facing response.
    """

    toxicity_result = SafetyMetrics.toxicity_score(answer)
    bias_result = SafetyMetrics.bias_fairness_score(answer)

    client.create_score(trace_id=trace_id, name="toxicity", value=toxicity_result.score, data_type="NUMERIC")
    client.create_score(trace_id=trace_id, name="bias_fairness", value=bias_result.score, data_type="NUMERIC")

    try:
        relevancy_metric = EvalMetrics.answer_relevancy_metric()
        relevancy_metric.measure(LLMTestCase(input=question, actual_output=answer))
        client.create_score(
            trace_id=trace_id, name="answer_relevancy", value=relevancy_metric.score, data_type="NUMERIC"
        )
    except Exception as exc:
        client.create_score(trace_id=trace_id, name="answer_relevancy_error", value=str(exc), data_type="TEXT")

    client.flush()


def ask_with_tools(
    question: str,
    chat_history: list[dict] | None = None,
    session_id: str | None = None,
    online_eval: bool = True,
) -> tuple[str, list[str]]:
    """Ask the agent a question and also report which tool(s) it called, in call order.

    `ask()` below is the plain version the chat UI uses when it doesn't care
    which tools ran. Pass `session_id` to group every question in a conversation
    under one Langfuse session (see src/app.py). `online_eval` is turned off by
    evals/run_eval.py, which already scores each case against the golden dataset
    and doesn't need the reference-free online metrics on top.
    """

    agent = get_agent()

    messages = list(chat_history or [])

    messages.append({"role": "user", "content": question})

    with propagate_attributes(session_id=session_id, tags=["chatbot"]):
        with client.start_as_current_observation(
            name=f"question: {question}", as_type="span"
        ) as trace:
            result = agent.invoke({"messages": messages}, config={"callbacks": [handler]})

            answer = result["messages"][-1].content

            tool_names = [
                message.name for message in result["messages"] if isinstance(message, ToolMessage)
            ]

            trace.update(input=question, output=answer)
            trace_id = client.get_current_trace_id()

    if online_eval:
        threading.Thread(target=_score_online, args=(trace_id, question, answer), daemon=True).start()

    return answer, tool_names


def ask(question: str, chat_history: list[dict] | None = None, session_id: str | None = None) -> str:

    answer, _tool_names = ask_with_tools(question, chat_history, session_id=session_id)
    return answer
