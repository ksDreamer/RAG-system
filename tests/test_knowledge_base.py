import json
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from rag_system.api import create_app
from rag_system.evaluation import evaluate_dataset
from rag_system.ingestion import chunk_pages, extract, tokens
from rag_system.models import Claim, Draft, Evidence
from rag_system.providers import ProviderError
from rag_system.retrieval import cosine, retrieve
from rag_system.service import KnowledgeBase


@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(tmp_path / "workspace.sqlite3")


def test_incremental_replace_delete_and_cache(kb):
    original = kb.ingest("manual.md", b"The reactor uses cobalt shielding.")
    assert original["status"] == "created"
    assert kb.ingest("manual.md", b"The reactor uses cobalt shielding.")["status"] == "unchanged"
    first = kb.ask("cobalt")
    assert first.status == "answered" and not first.cached
    assert kb.ask("cobalt").cached
    assert kb.ingest("manual.md", b"The reactor uses titanium shielding.")["status"] == "updated"
    assert kb.ask("cobalt").status == "abstained"
    assert not kb.ask("titanium").cached
    assert kb.delete(original["id"])
    assert kb.ask("titanium").status == "abstained"
    assert not kb.delete(original["id"])
    with kb.store.connect() as db:
        assert db.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 0


def test_failed_replacement_keeps_original(kb):
    kb.ingest("manual.txt", b"Retention policy is thirty days.")
    with pytest.raises(ValueError):
        kb.ingest("manual.txt", b"\xff\xff")
    assert kb.ask("retention").status == "answered"


def test_small_index_has_no_duplicate_negative_index_results(kb):
    kb.ingest("one.txt", b"One distinctive passage about geothermal energy.")
    answer = kb.ask("geothermal", 8)
    assert len(answer.sources) == 1
    assert len(answer.claims) == 1
    assert answer.claims[0].evidence[0].quote == answer.sources[0].text


@pytest.mark.parametrize(
    "name", ["../../secret.pdf", "/tmp/x.md", "folder\\x.txt", "bad\x00.txt", "code.py", ""]
)
def test_rejects_unsafe_or_unsupported_names(kb, name):
    with pytest.raises(ValueError):
        kb.ingest(name, b"hello world")


@pytest.mark.parametrize("body", [b"", b"  ", b"\xff"])
def test_rejects_empty_unreadable_text(kb, body):
    with pytest.raises(ValueError):
        kb.ingest("input.txt", body)


def test_pdf_blank_encrypted_and_malformed():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError, match="No readable text"):
        extract("scan.pdf", buffer.getvalue())
    writer.encrypt("secret")
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError, match="Encrypted"):
        extract("locked.pdf", buffer.getvalue())
    with pytest.raises(ValueError, match="Could not read"):
        extract("broken.pdf", b"not a PDF")


def test_chunk_offsets_overlap_and_page_boundary():
    text = "# Notes\n\n" + "A paragraph with a consistent reference. " * 100
    chunks = chunk_pages([(1, text), (2, "The second page is preserved.")])
    assert len(chunks) > 2
    for chunk in chunks:
        page_text = text if chunk.page == 1 else "The second page is preserved."
        assert page_text[chunk.start : chunk.end] == chunk.text
        assert len(chunk.text) <= 1200
    assert chunks[-1].page == 2 and chunks[-1].start == 0
    assert chunks[1].section == "Notes"
    with pytest.raises(ValueError):
        chunk_pages([(1, "abc")], size=10, overlap=10)


def test_cjk_and_query_syntax_are_safe(kb):
    kb.ingest("中文.txt", "删除文档时会同步移除索引。".encode())
    assert "删" in tokens("删除文档")
    assert kb.ask("删除文档").status == "answered"
    assert kb.ask('" OR * NOT (); --').status == "abstained"


class DraftProvider:
    identity = "test-draft"

    def __init__(self, kind):
        self.kind = kind

    def generate(self, query, sources):
        if self.kind == "empty":
            return Draft(claims=[])
        s = sources[0]
        return Draft(
            claims=[
                Claim(
                    text="A generated interpretation.",
                    evidence=[
                        Evidence(
                            source_id="invented" if self.kind == "id" else s.id,
                            quote="A fabricated quote" if self.kind == "quote" else s.text,
                        )
                    ],
                )
            ]
        )


@pytest.mark.parametrize("kind", ["id", "quote", "empty"])
def test_generated_answer_fails_closed(tmp_path, kind):
    kb = KnowledgeBase(tmp_path / "index.sqlite3", DraftProvider(kind))
    kb.ingest("guide.txt", b"Documents need valid citations and sources.")
    answer = kb.ask("citations")
    assert answer.status == "abstained"
    assert not answer.claims
    assert answer.sources


def test_valid_generated_evidence_is_returned(tmp_path):
    kb = KnowledgeBase(tmp_path / "index.sqlite3", DraftProvider("valid"))
    kb.ingest("guide.txt", b"Documents need valid citations and sources.")
    assert kb.ask("citations").status == "answered"


class FakeEmbedder:
    identity = "test-embedding-v1"

    def __init__(self):
        self.calls = []
        self.fail = False

    def encode(self, texts):
        self.calls.append(texts)
        if self.fail:
            return []
        return [[1.0, 0.0] if ("heat" in t or "thermal" in t) else [0.0, 1.0] for t in texts]


def test_hybrid_retrieval_and_embedding_cache(tmp_path):
    embedder = FakeEmbedder()
    kb = KnowledgeBase(tmp_path / "index.sqlite3", embedder=embedder)
    first = kb.ingest("thermal.txt", b"Geothermal heat comes from beneath the surface.")
    kb.ingest("thermal-copy.txt", b"Geothermal heat comes from beneath the surface.")
    assert len(embedder.calls) == 1
    kb.ingest("camera.txt", b"Camera video is recorded on an SD card.")
    answer = kb.ask("heat")
    assert answer.retrieval == "hybrid-rrf"
    assert answer.sources[0].name in ("thermal.txt", "thermal-copy.txt")
    assert len(answer.sources) == 1  # identical-text copies do not flood context
    kb.delete(first["id"])
    assert kb.ask("heat").sources[0].name == "thermal-copy.txt"
    embedder.fail = True
    with pytest.raises(ValueError, match="vectors"):
        kb.ingest("thermal-copy.txt", b"Heat replacement which failed embedding.")
    assert len(kb.store.documents()) == 2
    embedder.fail = False
    embedder.identity = "test-embedding-v2"
    kb.ask("thermal")
    assert any(
        "Geothermal heat comes from beneath the surface." in call for call in embedder.calls[3:]
    )


def test_embedding_validation_and_reranking(kb):
    assert cosine([1, 0], [1, 0]) == 1
    with pytest.raises(ValueError):
        cosine([1, 0], [1])
    with pytest.raises(ValueError):
        cosine([float("nan")], [1])
    kb.ingest("a.md", b"Document pipeline alpha supports ingestion.")
    kb.ingest("b.md", b"Document pipeline beta supports indexing.")

    class Reranker:
        identity = "test-reverse"

        def rank(self, query, sources):
            return sorted(sources, key=lambda s: s.name, reverse=True)

    assert retrieve(kb.store, "pipeline", reranker=Reranker())[0].name == "b.md"


@pytest.mark.parametrize(
    "question,top_k", [("", 5), ("   ", 5), ("a" * 2001, 5), ("valid", 0), ("valid", 9)]
)
def test_query_bounds(kb, question, top_k):
    with pytest.raises(ValueError):
        kb.ask(question, top_k)


def test_http_api_end_to_end_and_origins(kb):
    with TestClient(create_app(kb)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/status").json()["documents"] == 0
        assert client.post("/api/ask", json={"question": "energy"}).json()["status"] == "abstained"
        uploaded = client.post(
            "/api/documents", files={"file": ("notes.md", b"Solar energy powers the station.")}
        )
        assert uploaded.status_code == 200
        assert client.post("/api/ask", json={"question": "solar"}).json()["status"] == "answered"
        assert (
            client.post(
                "/api/documents", files={"file": ("../notes.md", b"bad upload")}
            ).status_code
            == 400
        )
        assert (
            client.post("/api/demo", headers={"Origin": "https://evil.example"}).status_code == 403
        )
        assert client.get("/api/documents", headers={"Origin": "null"}).status_code == 403
        assert client.get("/api/status", headers={"Host": "evil.example"}).status_code == 400
        assert client.delete("/api/documents/" + uploaded.json()["id"]).status_code == 200
        assert client.delete("/api/documents/missing").status_code == 404
        assert client.post("/api/ask", json={"question": " "}).status_code == 400
        assert client.post("/api/ask", json={"question": "x", "top_k": 50}).status_code == 422
        assert len(client.post("/api/demo").json()) == 3


def test_provider_error_is_not_cached_or_exposed(kb):
    class Broken:
        identity = "broken"

        def generate(self, query, sources):
            raise ProviderError("Provider request failed.")

    kb.provider = Broken()
    kb.ingest("notes.txt", b"Solar panels produce energy.")
    with TestClient(create_app(kb)) as client:
        response = client.post("/api/ask", json={"question": "solar"})
        assert response.status_code == 502
    with kb.store.connect() as db:
        assert db.execute("SELECT count(*) FROM cache").fetchone()[0] == 0


def test_fixture_evaluation_is_reproducible():
    from rag_system import evaluation

    root = Path(__file__).resolve().parents[1]
    result = evaluate_dataset(
        root / "eval/questions.jsonl", Path(evaluation.__file__).with_name("samples")
    )
    assert result["cases"] == 10
    assert result["document_recall_at_5"] == 1
    assert result["unanswerable_abstention_rate"] == 1
    assert result["exact_citation_validity"] == 1
    assert len(result["dataset_sha256"]) == 64
    assert json.loads(json.dumps(result)) == result


def test_markdown_sections_are_separate_and_excerpt_is_relevant(kb):
    text = "# Manual\n\nA guide to the observatory.\n\n## Storage\n\nMeteorite samples are stored in cabinet nine.\n\n## Access\n\nThe visitor entrance closes at five."
    kb.ingest("manual.md", text.encode())
    answer = kb.ask("meteorite cabinet")
    assert answer.claims[0].text == "Meteorite samples are stored in cabinet nine."
    assert answer.sources[0].section == "Storage"
    assert "visitor" not in answer.sources[0].text


def test_cache_fingerprint_survives_concurrent_cli_mutation(tmp_path):
    path = tmp_path / "shared.sqlite3"
    other = KnowledgeBase(path)

    class MutatingProvider:
        identity = "mutating"

        def generate(self, query, sources):
            other.ingest("record.txt", b"Reactor records now describe titanium.")
            return Draft(
                claims=[
                    Claim(
                        text=sources[0].text,
                        evidence=[Evidence(source_id=sources[0].id, quote=sources[0].text)],
                    )
                ]
            )

    kb = KnowledgeBase(path, MutatingProvider())
    kb.ingest("record.txt", b"Reactor records originally described cobalt.")
    assert kb.ask("cobalt").claims  # in-flight snapshot can finish
    assert kb.ask("cobalt").status == "abstained"  # must never reuse stale cached answer


def test_api_bounds_request_body_before_parsing(kb):
    with TestClient(create_app(kb)) as client:
        assert client.post("/api/ask", content=b"x" * 65537).status_code == 413
        assert client.post("/api/documents", content=iter([b"chunked"])).status_code == 411


def test_real_pdf_ingestion_keeps_page_provenance(kb):
    pdf = Path(__file__).with_name("fixtures") / "two-pages.pdf"
    kb.ingest(pdf.name, pdf.read_bytes())
    assert kb.ask("meteorite").sources[0].page == 1
    assert kb.ask("visitor entrance").sources[0].page == 2
