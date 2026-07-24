import socket
from collections.abc import Generator
from pathlib import Path

import arxiv
import requests

from inquiro.config import PAPERS_DIR
from inquiro.titles import load_titles, save_titles

# arxiv.Client's search request has no explicit timeout, so a slow or
# rate-limiting response from export.arxiv.org hangs the download step
# forever with no error. `requests` falls back to the socket default when no
# timeout is passed, so setting this makes that call fail instead of hang.
socket.setdefaulttimeout(30)


def download_papers(
    query: str = "structured pruning language models",
    max_results: int = 100,
    papers_dir: Path = PAPERS_DIR,
) -> int:
    downloaded = 0
    for _, _, short_id, title, status in iter_download_papers(query, max_results, papers_dir):
        icon = "✓" if status == "downloaded" else ("~" if status == "skipped" else "✗")
        print(f"[{icon}] {short_id} — {title[:70]}")
        if status == "downloaded":
            downloaded += 1
    return downloaded


def iter_download_papers(
    query: str,
    max_results: int = 200,
    papers_dir: Path = PAPERS_DIR,
) -> Generator[tuple[int, int, str, str, str], None, None]:
    """Yield (index, max_results, paper_id, title, status) for each arXiv paper.

    status is one of: "downloaded" | "skipped" | "failed"
    Existing PDFs are skipped so re-runs are safe.
    """
    papers_dir.mkdir(parents=True, exist_ok=True)
    titles = load_titles()
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = client.results(search)
    try:
        index = 0
        while True:
            try:
                paper = next(results)
            except StopIteration:
                break
            except arxiv.ArxivError:
                # arXiv's own client retries HTTP errors (e.g. 429 rate-limit)
                # a few times with no backoff, then raises. Treat that as "no
                # more results" so callers see zero papers instead of a crash.
                break
            index += 1

            short_id = paper.get_short_id()
            dest = papers_dir / f"{short_id}.pdf"
            # Record the title up front so it is captured even for skips/failures.
            titles[dest.name] = paper.title.strip()

            if dest.exists():
                yield index, max_results, short_id, paper.title, "skipped"
                continue

            try:
                resp = requests.get(paper.pdf_url, timeout=30)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                yield index, max_results, short_id, paper.title, "downloaded"
            except Exception:
                yield index, max_results, short_id, paper.title, "failed"
    finally:
        save_titles(titles)
