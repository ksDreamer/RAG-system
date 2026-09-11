# Retrieval notebook

This fictional notebook describes the mechanisms implemented in the demo.

## Two-stage retrieval

BM25 retrieves keyword matches from SQLite FTS5. Optional local sentence embeddings add dense cosine-similarity candidates. Reciprocal rank fusion combines the two ordered lists. An optional cross-encoder reranks the candidates, followed by overlapping-passage suppression.

## Evaluation

Evaluate document recall at five, mean reciprocal rank at five, exact citation validity and abstention on unanswerable questions. The included evaluation set is a small deterministic fixture, not evidence of quality on real customer documents. Assess semantic support and tune similarity thresholds on a separate representative collection before relying on generated answers.

## Privacy

The default extractive mode works without network calls or model downloads. Choosing a remote answer provider sends the question and retrieved passages to that provider. The SQLite workspace stores extracted text and cached answers on local disk, without encryption. Deleting records is not secure erasure of backups or SQLite pages.
