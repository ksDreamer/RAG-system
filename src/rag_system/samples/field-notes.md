# Field notes: document workspaces

This is a fictional example collection for exploring RAG-system.

## Incremental indexing

Document updates use a SHA-256 content hash and a chunker version. Uploading unchanged bytes with the same filename skips indexing. Replacing a document removes its old chunks and invalidates cached answers in one SQLite transaction. Deleting a document removes its searchable chunks and unused embeddings.

## Evidence and citations

Each retrieved passage keeps its filename, page number, section and text offsets. An answer citation must refer to a retrieved passage and include an exact quote. An invalid reference withholds the entire generated answer. Quote validation verifies provenance, but cannot prove that a model's interpretation is correct.

## Operating limits

The local workspace accepts PDF, Markdown and UTF-8 text. The upload limit is 20 MiB, with at most 500 PDF pages and two million extracted characters. Scanned documents need OCR before upload. Run one server worker on localhost.
