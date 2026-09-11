"""Optional real-model smoke check; downloads the explicitly named public models if missing."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from rag_system.providers import CrossEncoderReranker, SentenceEmbedder
from rag_system.service import KnowledgeBase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--output", type=Path, default=Path("artifacts/semantic-smoke.json"))
    args = parser.parse_args()
    embedding = SentenceEmbedder(args.embedding)
    reranker = CrossEncoderReranker(args.reranker)
    with TemporaryDirectory() as temporary:
        kb = KnowledgeBase(Path(temporary) / "index.sqlite3", embedder=embedding, reranker=reranker)
        kb.ingest(
            "thermal.md",
            b"Geothermal power plants use heat from beneath the surface of the Earth to generate electricity.",
        )
        kb.ingest("camera.md", b"Camera monitoring records video footage for a construction site.")
        answer = kb.ask("How can underground heat be converted to electric power?")
        assert answer.sources[0].name == "thermal.md"
        assert answer.status == "answered"
        result = {
            "scope": "Two-document real-model integration smoke check, not a quality benchmark.",
            "result": "passed",
            "embedding": embedding.identity,
            "reranker": reranker.identity,
            "retrieval": answer.retrieval,
            "top_source": answer.sources[0].name,
            "elapsed_ms": answer.elapsed_ms,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
