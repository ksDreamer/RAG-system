# Modernization audit — September 2026

Baseline: `08ac3f4741a5021632583c9fee3c2bca6c4499ef`, full clone with all available
history (12 commits, 2024 prototype). Both source modules and their history were
reviewed before replacement. The original project had no dependency manifest,
automated tests, package, or CI workflow.

## Problems addressed

| Original issue | Resulting behavior |
|---|---|
| File timestamp checks and a separate positional `.npy` cache | Transactional SQLite records; content and chunker fingerprints; coordinated deletion |
| Small FAISS results could include `-1` indices and repeat an unrelated chunk | Bounded candidate lists, stable chunk identities and duplicate suppression |
| A new encoder was loaded for each question | Optional models load once per application process |
| Provider-specific SDK imports were mandatory | Small provider protocols; no model or API dependency for the default experience |
| Prompt explicitly required an answer even without evidence | Empty-evidence abstention; structured claims; source/quote validation that withholds invalid drafts |
| Unbounded HTTP calls and direct response indexing | Connect/read limits, response-size limit, validated JSON contracts and redacted errors |
| Uploaded names became filesystem paths | Validated filenames; extracted text in SQLite; no original-file path construction |
| No automated engineering checks | Offline regressions, browser workflow test, evaluation fixture, package and CI checks |
| Screenshots described the old interface | Screenshots captured from the running new app with the bundled fictional collection |

## Validation performed locally

- Python 3.11: 49 pytest cases passed, with coverage above the 85% gate.
- Python 3.12 and 3.13: the same 49 cases passed in separate environments.
  The suite includes request-body bounds and an actual two-page PDF fixture.
- Ruff lint and formatting, mypy, and actionlint passed.
- Chromium smoke test passed: example ingestion, questions, citations, cache, abstention,
  upload, text-only rendering, deletion and mobile layout. Screenshots are real captures.
- Source distribution and wheel built; a wheel installed in a separate environment
  successfully ingested the bundled collection, answered a question, and loaded the
  app outside the checkout.
- The ten-case deterministic evaluation ran in a temporary index; its exact results
  and corpus hashes are in [evaluation.json](evaluation.json).
- Real local Sentence Transformers embedding and cross-encoder inference passed a
  two-document integration check. Model revisions and the result are recorded in
  [semantic-smoke.json](semantic-smoke.json). Reproduce it with
  `uv run --extra semantic python tools/smoke_semantic.py` (models download if absent).

Provider HTTP contracts were tested with mocked responses, including timeouts,
authentication/rate-limit/server failures, malformed JSON and oversized output.
No paid remote-model calls were made. The first GitHub Actions run exposed Windows'
legacy text encoding when reading the multilingual JSONL fixture. Evaluation input and
report files now explicitly use UTF-8; Git text files retain LF line endings across hosts.

## Remaining work and honest boundaries

The fixture's perfect retrieval/citation checks are not evidence of general answer
quality. Real-corpus retrieval evaluation, semantic entailment judgments and adversarial
prompt-injection evaluation remain necessary for a chosen model and workload. Scanned
and complex-layout PDF parsing, background ingestion, token streaming, and high-volume
vector search remain out of scope for the current lightweight application.

The resolved FastAPI/Starlette testing stack emits upstream deprecation warnings for
its httpx/AnyIO compatibility paths; tests pass and those warnings are retained.
The original project has no declared software license; this remains an author decision.

The modernization is now on `main`; follow the [CI matrix](https://github.com/MengyangGao/RAG-system/actions/workflows/ci.yml)
for remote platform results. No release or deployment was performed.
