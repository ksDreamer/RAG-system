import json

import httpx
import pytest

from rag_system.models import Source
from rag_system.providers import HTTPProvider, ProviderError, configured_provider


@pytest.fixture
def sources():
    return [
        Source(
            id="s1",
            document_id="d1",
            name="example.txt",
            page=1,
            section="",
            start=0,
            end=30,
            text="Evidence requires a source.",
        )
    ]


def mock_client(monkeypatch, handler):
    real = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs)
    )


@pytest.mark.parametrize("kind", ["ollama", "compatible"])
def test_provider_contract(monkeypatch, sources, kind):
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"][0]["role"] == "system"
        assert "untrusted" in body["messages"][0]["content"]
        assert request.headers["Authorization"] == "Bearer test-key"
        content = json.dumps(
            {
                "claims": [
                    {
                        "text": "A source is required.",
                        "evidence": [{"source_id": "s1", "quote": sources[0].text}],
                    }
                ]
            }
        )
        if kind == "ollama":
            assert request.url.path == "/api/chat"
            assert body["format"]["type"] == "object"
            return httpx.Response(200, json={"message": {"content": content}})
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    mock_client(monkeypatch, handler)
    base = "http://localhost:11434" if kind == "ollama" else "https://provider.example/v1"
    assert HTTPProvider(kind, base, "test-model", "test-key").generate("sources?", sources).claims


@pytest.mark.parametrize(
    "status,body",
    [
        (401, {"error": "secret-token"}),
        (429, {}),
        (500, {}),
        (200, {}),
        (200, {"choices": []}),
        (200, {"choices": [{"message": {"content": "not JSON"}}]}),
    ],
)
def test_provider_errors_are_redacted(monkeypatch, sources, status, body):
    mock_client(monkeypatch, lambda request: httpx.Response(status, json=body))
    with pytest.raises(ProviderError) as error:
        HTTPProvider("compatible", "https://provider.example/v1", "model", "secret-token").generate(
            "question", sources
        )
    assert "secret-token" not in str(error.value)


def test_provider_timeout(monkeypatch, sources):
    def handler(request):
        raise httpx.ReadTimeout("secret server details", request=request)

    mock_client(monkeypatch, handler)
    with pytest.raises(ProviderError):
        HTTPProvider("ollama", "http://127.0.0.1:11434", "model").generate("question", sources)


def test_provider_response_size_limit(monkeypatch, sources):
    mock_client(monkeypatch, lambda request: httpx.Response(200, content=b"x" * 256_001))
    with pytest.raises(ProviderError, match="size limit"):
        HTTPProvider("ollama", "http://127.0.0.1:11434", "model").generate("question", sources)


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example",
        "ftp://localhost",
        "https://key:secret@example.com",
        "https://example.com?token=secret",
        "",
    ],
)
def test_provider_url_validation(url):
    with pytest.raises(ValueError):
        HTTPProvider("compatible", url, "model")


def test_default_provider_needs_no_key(monkeypatch):
    monkeypatch.delenv("RAG_PROVIDER", raising=False)
    assert configured_provider().identity == "extractive-v1"
    monkeypatch.setenv("RAG_PROVIDER", "unknown")
    with pytest.raises(ValueError):
        configured_provider()
