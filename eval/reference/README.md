# Frozen engineering-document fixture

Three real InfoMatrix documents, copied without changes from
[infoMatrix at e1c745c6cbf3db66ead91a18b5921402cf428f21](https://github.com/MengyangGao/infoMatrix/tree/e1c745c6cbf3db66ead91a18b5921402cf428f21/docs), under the accompanying MIT license.
This snapshot stays frozen when the source project changes. It is a small retrieval
regression set, not a representative product benchmark. Labels were written manually;
there is no LLM judge. No private user documents are included.

The questions exercise overlapping storage/sync vocabulary, exact passage anchors,
selected-document scope, and out-of-corpus questions. Chinese questions against an
English corpus intentionally expose the default lexical retriever's language limit.
Run from the repository root:

```sh
uv run rag-system evaluate eval/reference/questions.jsonl --corpus eval/reference/corpus --output docs/reference-evaluation.json
```
