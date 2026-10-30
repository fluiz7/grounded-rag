# grounded-rag

> Multi-agent RAG that **refuses to hallucinate**: a LangGraph pipeline where a Critic agent verifies every answer against the retrieved sources — and retries the search when the evidence is missing. 100% local (Ollama + Chroma), no API keys.

[![CI](https://github.com/fluiz7/grounded-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/fluiz7/grounded-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-000000?style=flat&logo=ollama&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Plain RAG chains answer even when the retrieved context doesn't support the answer — that's where hallucinated citations come from. **grounded-rag** adds an agentic verification loop on top of retrieval:

```mermaid
flowchart LR
    Q([question]) --> R[🔎 Retriever]
    R --> A[🧠 Answerer]
    A --> C[✅ Critic]
    C -- grounded --> OUT([answer + citations])
    C -- "not grounded → rewrites query" --> R
```

- **Retriever** — semantic search over your PDFs (Chroma, embedded).
- **Answerer** — answers *only* from the retrieved context, with inline `[n]` citations.
- **Critic** — a second LLM pass that audits the answer: if any claim is unsupported, it **rewrites the search query** and sends the pipeline back to retrieval (bounded by a retry budget). If no answer is ever approved, **none is returned**.

The whole loop is a compiled [LangGraph](https://github.com/langchain-ai/langgraph) state machine — see [`src/graph.py`](src/graph.py).

## Quickstart

**Prereqs:** Python 3.11+, [Ollama](https://ollama.com) installed and running.

```bash
# 1. models (one-time, ~5 GB total)
ollama pull llama3.1
ollama pull nomic-embed-text

# 2. install (in a virtualenv)
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# 3. index your PDFs (a file or a whole folder)
python -m src.cli ingest ./papers

# 4. ask — answers come with citations
python -m src.cli ask "What factors drive latency in edge computing under mobility?"
```

> ⚠️ Every `python` above must be the **virtualenv's** Python. If you skipped `activate`, prefix commands with `.venv\Scripts\python.exe` (Windows) / `.venv/bin/python` — a bare global `python` will fail with `ModuleNotFoundError: langgraph`.

Example output:

```
=== ANSWER ==================================================
Latency is primarily driven by signal quality rather than raw distance [1][3].
Weak signal conditions trigger retransmissions, producing heavy latency
tails of several seconds [1]. Linear fits of latency vs. distance show very
low R², confirming distance alone is a weak predictor [2].

=== SOURCES =================================================
  [1] wgrs2024.pdf — page 9
  [2] vehiclouds2024.pdf — page 6
  [3] wgrs2024.pdf — page 12
```

Add `--verbose` to see the critic's verdict and how many retrieval rounds were used.

## How the Critic loop works

1. The Answerer must cite every claim with `[n]` markers, or reply with the fixed refusal sentence.
2. **Deterministic citation check** (no LLM call): an answer with no citations, or citing a source number that was never retrieved (`[7]` with 5 sources), is rejected immediately.
3. The Critic re-reads *question + context + answer* and returns a **JSON verdict** (`{"grounded", "unsupported_claims", "query"}`, decoded in Ollama's JSON mode). The parser is **fail-closed**: an empty reply, prose without a verdict, `yes, partially`, or an approval that still lists unsupported claims all count as *not grounded*.
4. When the verdict is *no*, the graph routes back to the Retriever with the Critic's rewritten query — up to `RAG_MAX_RETRIES` times (default 2). **Evidence accumulates across rounds** (deduplicated), so support split between two searches can still ground one answer.
5. **An answer the Critic never approved is never returned as the answer.** The pipeline replies with the refusal and keeps the rejected text as an inspectable `draft` (`--verbose`).

A refusal asserts nothing, so it skips the Critic and ends the run (exit code 4). If the Critic model returns no output at all, the run stops with an explicit *critic failure* instead of disguising it as missing evidence.

This is a minimal, readable implementation of the *corrective RAG* idea (generate → verify → refine retrieval) — see [`src/graph.py`](src/graph.py).

### Exit codes

| Code | Meaning |
|---|---|
| `0` | answer verified by the Critic |
| `3` | an answer was drafted but could not be verified (rejected, or critic failure) |
| `4` | the documents don't contain the answer |

## Configuration

Everything is tunable via environment variables (see [`src/config.py`](src/config.py)):

| Variable | Default | Meaning |
|---|---|---|
| `RAG_CHAT_MODEL` | `llama3.1` | Ollama chat model (the Answerer) |
| `RAG_CRITIC_MODEL` | = `RAG_CHAT_MODEL` | model for the Critic — use a different one to avoid a model grading its own output |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `RAG_TOP_K` | `5` | chunks retrieved per round |
| `RAG_MAX_RETRIES` | `2` | extra retrieve→answer→critic cycles |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1000` / `150` | splitter settings |

## Project layout

```
src/
├── config.py   # env-driven settings
├── llm.py      # Ollama + Chroma factories (single place to swap backends)
├── ingest.py   # PDF -> chunks -> embeddings -> Chroma (idempotent: re-ingesting never duplicates)
├── graph.py    # the LangGraph state machine (Retriever / Answerer / Critic)
└── cli.py      # ingest & ask commands
```

## Testing

The agent pipeline is fully unit-tested **without any LLM server**: a scripted fake chat model and an in-memory vector store drive the LangGraph state machine. [`tests/test_adversarial.py`](tests/test_adversarial.py) pins every failure found in an adversarial review — inverted verdicts, markdown-formatted verdicts, phantom citations, discarded evidence, unverified answers leaking out, duplicated chunks, empty critic output — so none of them can come back.

```bash
pip install -r requirements-dev.txt
ruff check src tests   # lint
pytest                 # 40 tests, ~3s, no Ollama needed
```

CI runs the same lint + tests on Python 3.11 and 3.12 for every push (see badge above).

## Why local?

Running on Ollama means anyone can clone and reproduce the demo without API keys or cost — and the architecture is backend-agnostic: swapping `llm.py` to a hosted model (Claude, GPT, Bedrock) is a one-file change.

## License

[MIT](LICENSE) — © Luiz Felipe Cantanhede Cristino.
