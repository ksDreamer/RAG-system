"""Transparent document-level retrieval and abstention checks; no LLM-as-judge dependency."""

import json
import statistics
import tempfile
from pathlib import Path

from rag_system.service import KnowledgeBase
from rag_system.store import digest


def evaluate_dataset(dataset: Path, corpus: Path) -> dict:
    cases = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("Evaluation dataset is empty.")
    with tempfile.TemporaryDirectory() as temporary:
        kb = KnowledgeBase(Path(temporary) / "eval.sqlite3")
        files = sorted(p for p in corpus.iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"})
        if not files:
            raise ValueError("Evaluation corpus is empty.")
        documents = {p.name: kb.ingest(p.name, p.read_bytes())["id"] for p in files}
        details, recalls, reciprocal_ranks, refusals, citation_validity = [], [], [], [], []
        passage_recalls = []
        for case in cases:
            expected = set(case["relevant_documents"])
            if not expected.issubset({p.name for p in files}):
                raise ValueError("Evaluation refers to an unknown document.")
            scope = case.get("search_documents")
            if scope is not None and not set(scope).issubset(documents):
                raise ValueError("Evaluation scope refers to an unknown document.")
            anchors = case.get("required_passages", [])
            for anchor in anchors:
                name, quote = anchor["document"], anchor["quote"]
                if name not in expected or not isinstance(quote, str) or not quote.strip():
                    raise ValueError("A passage anchor must name a relevant document and quote.")
                with kb.store.connect() as db:
                    texts = [
                        r[0]
                        for r in db.execute(
                            "SELECT text FROM chunks WHERE document_id=?", (documents[name],)
                        )
                    ]
                if not any(quote in value for value in texts):
                    raise ValueError("A passage anchor is absent from the indexed corpus.")
            result = kb.ask(
                case["question"],
                document_ids=None if scope is None else [documents[name] for name in scope],
            )
            passage_recall = (
                sum(
                    any(s.name == a["document"] and a["quote"] in s.text for s in result.sources)
                    for a in anchors
                )
                / len(anchors)
                if anchors
                else None
            )
            if passage_recall is not None:
                passage_recalls.append(passage_recall)
            names = list(dict.fromkeys(s.name for s in result.sources))
            if expected:
                recalls.append(len(expected.intersection(names)) / len(expected))
                reciprocal_ranks.append(
                    next((1 / (i + 1) for i, n in enumerate(names) if n in expected), 0)
                )
            else:
                refusals.append(result.status == "abstained")
            source_map = {s.id: s for s in result.sources}
            for claim in result.claims:
                for e in claim.evidence:
                    citation_validity.append(
                        e.source_id in source_map and e.quote in source_map[e.source_id].text
                    )
            details.append(
                {
                    "question": case["question"],
                    "retrieved": names,
                    "expected": sorted(expected),
                    "status": result.status,
                    "search_documents": scope,
                    "passage_anchor_recall_at_5": passage_recall,
                    "elapsed_ms": result.elapsed_ms,
                }
            )
        return {
            "scope": "Bundled deterministic fixture; not a general RAG benchmark or semantic faithfulness measurement.",
            "mode": "extractive / BM25",
            "cases": len(cases),
            "dataset_sha256": digest(dataset.read_bytes()),
            "corpus_sha256": {p.name: digest(p.read_bytes()) for p in files},
            "document_recall_at_5": statistics.mean(recalls) if recalls else None,
            "document_mrr_at_5": statistics.mean(reciprocal_ranks) if reciprocal_ranks else None,
            "passage_anchor_recall_at_5": statistics.mean(passage_recalls)
            if passage_recalls
            else None,
            "unanswerable_abstention_rate": statistics.mean(refusals) if refusals else None,
            "exact_citation_validity": statistics.mean(citation_validity)
            if citation_validity
            else None,
            "results": details,
        }
