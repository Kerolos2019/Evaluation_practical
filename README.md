# Knowledge Base Chatbot + Evaluation

An internal chatbot that answers questions about company data (CRM exports, projects, revenue and proposal PDFs). It is backed by a retrieval-augmented GPT-5 agent, traced with [Langfuse](https://langfuse.com), scored with [DeepEval](https://deepeval.com), and gated in CI.

```
datasets/ ──► ingestion ──► ChromaDB
                               ▲
Streamlit UI ──► GPT-5 agent ──► tools (search_knowledge_base, get_weather)
                     ▲     │
CI ──► eval runner ──┘     └──► Langfuse (traces, sessions, scores)
```

The eval runner calls the same `ask_with_tools` function as the chat UI, so CI measures the same code path that users hit.

## Project structure

```
app.py                     # entry point: ingest, then launch the Streamlit UI
basic_evals.py             # minimal DeepEval example (one question, one metric)
docker-compose.yml         # self-hosted Langfuse, fetched from the Langfuse repo (see Setup)
datasets/                  # source data: 6 CSV exports + 5 proposal PDFs
src/
  config.py                # models, chunk sizes, paths
  loaders.py               # CSV / PDF / TXT / MD -> text
  chunking.py              # fixed-size chunks (1000 chars, 200 overlap)
  vectorstore.py           # persistent ChromaDB collection with OpenAI embeddings
  pipeline.py              # ingestion (skipped if the collection is already populated)
  tools.py                 # search_knowledge_base, get_weather (dummy)
  agents.py                # LangChain agent, Langfuse tracing, online scoring
  app.py, design.py        # Streamlit chat UI with Tailwind styling
evals/
  run_eval.py              # offline evaluation + CI gate
  data/golden_dataset.json # 18 golden test cases
  metrics/                 # rule-based, heuristic, RAG, safety, LLM-as-judge metrics
  results/latest.json      # output of the last eval run
docs/deepeval_notes.md     # DeepEval learning notes
.github/workflows/ci.yml   # runs the evaluation on every push and PR
```

## Setup

Requirements: Python 3.12, Docker, an OpenAI API key.

```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```
OPENAI_API_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_BASE_URL=http://localhost:3001
```

### Start Langfuse

The compose file is not shipped with this repo — get it from the Langfuse repository:

```bash
curl -O https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml
```

Langfuse serves its web container on port 3000 by default. This project expects 3001 (`LANGFUSE_BASE_URL`), so either remap the `langfuse-web` port to `3001:3000` in the compose file or set `LANGFUSE_BASE_URL=http://localhost:3000` in `.env`.

Then start it:

```bash
docker compose up -d
```

Open http://localhost:3001, create a project, and copy its public and secret keys into `.env`.

## Usage

### Chat UI

```bash
python app.py
```

On first run this ingests `datasets/` into ChromaDB (`.chroma/`), then launches Streamlit. Every question is traced in Langfuse under a `chatbot-<uuid>` session. Each answer is also scored in the background for `toxicity`, `bias_fairness` and `answer_relevancy`.

> Ingestion runs only when the collection is empty. After changing files in `datasets/`, delete `.chroma/` to re-ingest.

### Evaluation

```bash
python evals/run_eval.py
```

For each of the 18 golden cases, the runner asks the agent, retrieves the top-4 sources, and computes 9 metrics:

| Family       | Metrics                                   |
|--------------|-------------------------------------------|
| Rule-based   | `keyword_coverage`, `tool_call_accuracy`  |
| Heuristic    | `rouge`, `length_penalty`                 |
| Retrieval    | `context_precision`, `context_recall`     |
| Safety       | `toxicity`                                |
| LLM-as-judge | `correctness`, `relevance` (GEval)        |

- A case **passes** when `correctness >= 0.8` and the first tool called matches `expected_tool`.
- The run **fails** (exit code 1) when fewer than 80% of cases pass.
- Results are written to `evals/results/latest.json` and logged to Langfuse under an `eval-run-<timestamp>` session, with an `eval-summary` trace for the averages.

Latest run: **16/18 cases passed (88.9%)**.

## CI

`.github/workflows/ci.yml` runs `python evals/run_eval.py` on every push and pull request. It needs an `OPENAI_API_KEY` repository secret. Langfuse tracing is turned off in CI (`LANGFUSE_TRACING_ENABLED=false`), but the metrics and the pass/fail gate still run.

## Known issues

- The `accounts_north_america` golden case uses the key `expected_tools` instead of `expected_tool`, so its tool accuracy is always 0. Its expected answer also misspells "HealthFirst" as "HeathFirst".
- The safety metrics match substrings, so "skill" trips toxicity ("kill") and "small" trips bias ("all").
- `get_weather` returns fixed dummy data.
