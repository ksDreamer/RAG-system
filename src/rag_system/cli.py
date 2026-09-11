"""A single entry point for the app, ingestion and reproducible evaluation."""

import argparse
import json
from pathlib import Path

from rag_system.service import configured_kb


def main():
    parser = argparse.ArgumentParser(description="Local documents, inspectable answers.")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the local web application")
    serve.add_argument("--port", type=int, default=8000)
    ingest = commands.add_parser("ingest", help="Incrementally import files")
    ingest.add_argument("files", type=Path, nargs="+")
    commands.add_parser("demo", help="Index bundled example documents")
    ask = commands.add_parser("ask", help="Ask a question and print evidence JSON")
    ask.add_argument("question")
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate a labeled JSONL set against a temporary index"
    )
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--corpus", type=Path, default=Path(__file__).with_name("samples"))
    evaluate.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "rag_system.api:create_app", factory=True, host="127.0.0.1", port=args.port, workers=1
        )
    elif args.command == "evaluate":
        from rag_system.evaluation import evaluate_dataset

        result = json.dumps(evaluate_dataset(args.dataset, args.corpus), indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result + "\n")
        print(result)
    else:
        kb = configured_kb()
        if args.command == "ask":
            print(kb.ask(args.question).model_dump_json(indent=2))
        else:
            files = (
                sorted(Path(__file__).with_name("samples").glob("*.md"))
                if args.command == "demo"
                else args.files
            )
            for path in files:
                print(json.dumps(kb.ingest(path.name, path.read_bytes())))
