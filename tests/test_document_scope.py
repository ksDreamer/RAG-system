import pytest
from fastapi.testclient import TestClient

from rag_system.api import create_app
from rag_system.service import KnowledgeBase


@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(tmp_path / "index.sqlite3")


def test_scope_filters_before_candidate_limit_and_separates_cache(kb):
    selected = kb.ingest("selected.md", b"Retention period is sixty days. " + b"Context. " * 90)
    for number in range(35):
        kb.ingest(f"distractor-{number}.txt", f"Retention {number} days.".encode())
    assert "selected.md" not in {s.name for s in kb.ask("retention").sources}
    answer = kb.ask("retention", document_ids=[selected["id"]])
    assert not answer.cached
    assert {s.name for s in answer.sources} == {"selected.md"}
    assert kb.ask("retention", document_ids=[selected["id"], selected["id"]]).cached
    empty = kb.ask("retention", document_ids=[])
    assert empty.status == "abstained" and not empty.sources
    assert kb.ask("retention").sources  # an empty selection never poisons the whole-library cache
    kb.delete(selected["id"])
    with pytest.raises(ValueError, match="no longer available"):
        kb.ask("retention", document_ids=[selected["id"]])


def test_dense_retrieval_cannot_cross_document_scope(tmp_path):
    class Embedder:
        identity = "constant-test-vectors"

        def encode(self, texts):
            return [[1.0, 0.0] for _ in texts]

    kb = KnowledgeBase(tmp_path / "hybrid.sqlite3", embedder=Embedder())
    selected = kb.ingest("a.txt", b"Heat rises from underground springs.")
    kb.ingest("b.txt", b"Camera footage is stored for thirty days.")
    assert {s.name for s in kb.ask("unmatched", document_ids=[selected["id"]]).sources} == {"a.txt"}
    assert not kb.ask("unmatched", document_ids=[]).sources


def test_reader_paginates_and_focuses_exact_indexed_passage(kb):
    original = "# Field manual\n" + "Measured groundwater conditions. " * 1200
    doc = kb.ingest("manual.md", original.encode())
    first = kb.store.read_document(doc["id"], limit=3)
    assert first["total"] > 12
    second = kb.store.read_document(doc["id"], offset=3, limit=3)
    assert not {p["id"] for p in first["passages"]} & {p["id"] for p in second["passages"]}
    focus = second["passages"][1]["id"]
    focused = kb.store.read_document(doc["id"], focus=focus, limit=3)
    assert focused == {**second, "focus": focus}
    for passage in focused["passages"]:
        assert original[passage["start"] : passage["end"]] == passage["text"]
    assert not kb.store.read_document(doc["id"], offset=99999)["passages"]
    other = kb.ingest("other.txt", b"Unrelated source material.")
    with pytest.raises(KeyError, match="source passage changed"):
        kb.store.read_document(other["id"], focus=focus)
    kb.ingest("manual.md", b"This is the revised field manual.")
    with pytest.raises(KeyError, match="source passage changed"):
        kb.store.read_document(doc["id"], focus=focus)


def test_scope_and_reader_api_validation(kb):
    doc = kb.ingest("notes.txt", b"Local observations about groundwater.")
    with TestClient(create_app(kb)) as client:
        route = f"/api/documents/{doc['id']}/passages"
        assert client.get(route).json()["passages"][0]["text"].startswith("Local")
        for params in ({"offset": -1}, {"limit": 0}, {"limit": 51}):
            assert client.get(route, params=params).status_code == 422
        assert client.get(route, params={"focus": "stale"}).status_code == 404
        assert client.get("/api/documents/missing/passages").status_code == 404
        assert (
            client.post("/api/ask", json={"question": "groundwater", "document_ids": []}).json()[
                "sources"
            ]
            == []
        )
        assert (
            client.post(
                "/api/ask", json={"question": "groundwater", "document_ids": ["missing"]}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/ask", json={"question": "groundwater", "document_ids": [doc["id"]] * 101}
            ).status_code
            == 422
        )
