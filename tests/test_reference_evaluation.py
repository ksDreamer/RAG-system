import json
from pathlib import Path

import pytest

from rag_system.evaluation import evaluate_dataset


def test_frozen_reference_passages_and_known_cross_language_miss():
    root = Path(__file__).resolve().parents[1] / "eval/reference"
    report = evaluate_dataset(root / "questions.jsonl", root / "corpus")
    assert report["cases"] == 18
    assert report["passage_anchor_recall_at_5"] == pytest.approx(14 / 15)
    assert report["unanswerable_abstention_rate"] == 1
    assert report["results"][-1]["status"] == "abstained"
    assert report["results"][-1]["expected"] == ["sync.md"]


@pytest.mark.parametrize(
    "change", ["unknown_scope", "wrong_document", "fabricated_quote", "empty_quote"]
)
def test_evaluation_rejects_invalid_passage_labels(tmp_path, change):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "source.md").write_text("Measured groundwater conditions.", encoding="utf-8")
    case = {
        "question": "groundwater",
        "relevant_documents": ["source.md"],
        "required_passages": [{"document": "source.md", "quote": "groundwater"}],
    }
    if change == "unknown_scope":
        case["search_documents"] = ["missing.md"]
    else:
        case["required_passages"][0].update(
            {
                "wrong_document": {"document": "missing.md"},
                "fabricated_quote": {"quote": "invented claim"},
                "empty_quote": {"quote": ""},
            }[change]
        )
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps(case), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_dataset(dataset, corpus)
