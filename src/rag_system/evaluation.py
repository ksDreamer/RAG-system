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
        for p in files:
            kb.ingest(p.name, p.read_bytes())
        details, recalls, reciprocal_ranks, refusals, citation_validity = [], [], [], [], []
        for case in cases:
            expected = set(case["relevant_documents"])
            if not expected.issubset({p.name for p in files}):
                raise ValueError("Evaluation refers to an unknown document.")
            result = kb.ask(case["question"])
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
            "unanswerable_abstention_rate": statistics.mean(refusals) if refusals else None,
            "exact_citation_validity": statistics.mean(citation_validity)
            if citation_validity
            else None,
            "results": details,
        }
