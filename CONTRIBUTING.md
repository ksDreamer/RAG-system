# Contributing

Use Python 3.11–3.13 and `uv sync --locked`. Keep ingestion, retrieval and answer
generation independent of HTTP and the UI. Prefer an explicit function or protocol
over an orchestration framework. A new dependency needs a concrete use case.

Before proposing a change:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/rag_system
uv run pytest --cov=rag_system --cov-fail-under=85
uv run rag-system evaluate eval/questions.jsonl
uv build
```

For UI changes, also run the browser smoke test in the README. New indexing logic
must test replacement, deletion, partial failure and cache invalidation. Provider
tests must run offline using fixtures, with no credentials or paid calls. Keep
benchmark fixtures, general model evaluations and visual demos clearly labeled.
Do not commit private documents, API keys, local databases or model weights.

Document a migration whenever index formats, chunking or model identities change.
Use versioned model directories or identifiers: replacing model weights beneath an
unchanged identifier cannot be detected automatically.
