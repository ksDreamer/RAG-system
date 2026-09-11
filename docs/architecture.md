# Architecture and tradeoffs

```text
Static HTML / CSS / JavaScript
           │ same-origin HTTP
       FastAPI (api.py)
           │
    KnowledgeBase (service.py)
      ┌────┴───────────────┐
 ingest / replace      query
      │                   │
 PDF / UTF-8         BM25 candidates
 page / section      + optional cosine candidates
 chunk + offsets       → RRF → optional cross-encoder
      │                   │
      └── SQLite WAL ──────┘
                          │
                AnswerProvider protocol
                 ├ extractive, offline
                 ├ Ollama, local HTTP
                 └ compatible Chat Completions API
                          │
                 structured claims + evidence
                          │
                 exact quote / source validation
```

## Index lifecycle

SQLite is the source of truth; there is no independent positional vector file.
Document identity derives from its filename. SHA-256 over bytes plus chunker version
detects unchanged uploads. A changed file is parsed and embedded before its old rows
are replaced inside one transaction. Failure rolls back the replacement. Renaming a
file creates a distinct document; remove the old name explicitly when appropriate.

Chunks preserve page, the most recent Markdown heading, and zero-based character
offsets in normalized page text (half-open ranges). They target 1,200 characters with
160 characters of overlap. PDF pages and Markdown sections remain separate. Tables, columns, scanned text,
and complex layouts are not reconstructed or OCRed.

An FTS5 index uses BM25 and Porter stemming for Latin words, with explicit CJK
character tokenization. Dense retrieval is optional. Model identity and chunk-content
hash key the embedding cache. RRF fuses up to 30 candidates from each channel, then an
optional cross-encoder orders the pool. Duplicate and heavily overlapping passages
are suppressed before selecting at most eight sources. Retrieval scores are ordering
signals, not calibrated confidence probabilities. The dense cosine cutoff is 0.35;
it is an engineering default, not a validated relevance threshold for arbitrary models.

## Answers and caches

Providers return structured claims and evidence. Every evidence entry must point to
a retrieved chunk and quote a nonempty exact substring. Any invalid entry withholds
the complete generated draft. An empty draft or empty retrieval result abstains.
Semantic entailment requires separate evaluation and human review. Offline mode
returns the most query-relevant paragraph from each of the top three passages as
verbatim excerpts rather than generated interpretations.

Cache keys include question, retrieval configuration, provider/model identity, prompt
version and the document fingerprints. Mutations clear stored answers. The cache is
bounded to 128 entries. The process serializes ingestion and answers with one lock;
SQLite transactions protect database integrity across CLI invocations. A query already
in progress may finish with its earlier snapshot. Fingerprinted keys prevent that
snapshot from becoming a current cached answer after an update.

## Deliberate scope

One process, one SQLite file, one static frontend. No LangChain, agent runtime,
Redis, vector service, user accounts, containers, or frontend package manager is
required. Dense search scans cached vectors in process and is intended for small
personal collections. Long indexing or generation requests serialize other work.
Background jobs, token streaming, OCR, parser isolation and large-corpus vector
indexes should be added only when a measured workload justifies them.

## References

- [SQLite FTS5](https://www.sqlite.org/fts5.html): full-text indexing and BM25.
- [Sentence Transformers retrieve and rerank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html): two-stage search.
- [Ollama chat API](https://docs.ollama.com/api/chat): local structured generation.
- [Chat Completions API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create): the compatibility adapter's wire format.

The core architecture is intentionally independent of a specific model vendor.
