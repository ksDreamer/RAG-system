"""Exercise the real local server/browser with a temporary workspace and capture screenshots."""

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/ui"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory() as workspace:
        env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_")}
        env.update(RAG_DATA_DIR=workspace, RAG_PROVIDER="extractive")
        with tempfile.TemporaryFile(mode="w+") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "rag_system.api:create_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                env=env,
                stdout=log,
                stderr=log,
            )
            try:
                for _ in range(100):
                    if server.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read())
                    try:
                        if httpx.get(url + "/api/status", timeout=1).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Local server did not become ready.")
                with sync_playwright() as p:
                    browser = p.chromium.launch()
                    page = browser.new_page(
                        viewport={"width": 1440, "height": 1060}, device_scale_factor=1
                    )
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url)
                    expect(
                        page.get_by_text("Your library is ready for its first document.")
                    ).to_be_visible()
                    page.get_by_role("button", name="Try the example collection").click()
                    expect(page.locator("#document-count")).to_have_text("3")
                    expect(page.locator("#demo-button")).to_be_enabled()
                    page.screenshot(path=str(args.output / "workspace.png"), full_page=True)
                    page.locator("#question").fill(
                        "How does incremental indexing detect unchanged documents?"
                    )
                    page.get_by_role("button", name="Ask workspace").click()
                    expect(page.locator("#answer-title")).to_have_text("From your documents")
                    expect(page.locator(".source-card").first).to_be_visible()
                    expect(page.locator("#ask-button")).to_be_enabled()
                    page.screenshot(path=str(args.output / "answer.png"), full_page=True)
                    page.get_by_role("button", name="Ask workspace").click()
                    expect(page.locator("#answer-meta")).to_contain_text("Cached")
                    page.get_by_role("button", name="Read in document").first.click()
                    expect(page.get_by_role("dialog")).to_be_visible()
                    expect(page.locator(".reader-passage.highlighted")).to_be_visible()
                    page.screenshot(path=str(args.output / "source-reader.png"), full_page=False)
                    page.keyboard.press("Escape")
                    expect(page.get_by_role("dialog")).not_to_be_visible()
                    for checkbox in page.locator(".document-select").all():
                        checkbox.uncheck()
                    expect(page.locator("#ask-button")).to_be_disabled()
                    expect(page.locator("#scope-label")).to_have_text("0 of 3 documents selected")
                    page.locator(".document-select").first.check()
                    with page.expect_request("**/api/ask") as scoped_request:
                        page.get_by_role("button", name="Ask workspace").click()
                    assert len(scoped_request.value.post_data_json["document_ids"]) == 1
                    expect(page.locator("#ask-button")).to_be_enabled()
                    page.locator("#all-documents").click()
                    expect(page.locator("#scope-label")).to_have_text("All 3 documents")
                    page.locator("#question").fill("What is the capital of Namibia?")
                    page.get_by_role("button", name="Ask workspace").click()
                    expect(page.locator("#answer-title")).to_have_text("More evidence needed")
                    expect(page.locator("#ask-button")).to_be_enabled()
                    page.locator("#file-input").set_input_files(
                        {
                            "name": "upload.txt",
                            "mimeType": "text/plain",
                            "buffer": b"The observatory stores meteorite samples. <script>window.injected=true</script>",
                        }
                    )
                    expect(page.locator("#document-count")).to_have_text("4")
                    page.locator("#question").fill("meteorite")
                    page.get_by_role("button", name="Ask workspace").click()
                    expect(page.locator("#answer-card")).to_contain_text("meteorite")
                    assert page.evaluate("window.injected === undefined")
                    expect(page.locator("#ask-button")).to_be_enabled()
                    page.get_by_role("button", name="Read upload.txt", exact=True).click()
                    expect(page.locator("#reader-passages")).to_contain_text(
                        "<script>window.injected=true</script>"
                    )
                    assert page.evaluate("window.injected === undefined")
                    page.set_viewport_size({"width": 390, "height": 844})
                    assert page.evaluate(
                        "document.querySelector('dialog').scrollWidth <= innerWidth"
                    )
                    page.locator("#reader-close").click()
                    page.get_by_role("button", name="Delete upload.txt").click()
                    expect(page.locator("#document-count")).to_have_text("3")
                    expect(page.locator("#answer-section")).not_to_be_visible()
                    page.set_viewport_size({"width": 390, "height": 844})
                    page.screenshot(path=str(args.output / "mobile.png"), full_page=True)
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                        "Mobile layout overflows"
                    )
                    assert not errors, errors
                    browser.close()
                print(
                    "PASS: load, demo, query, scoped retrieval, reader, citations, cache, abstention, upload, safe text, deletion, mobile layout."
                )
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()
