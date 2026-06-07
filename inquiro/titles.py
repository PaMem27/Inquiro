"""Persistent arXiv-id → paper-title mapping used to show friendly source names.

The corpus stores each chunk's ``source`` as the PDF filename (an arXiv id such
as ``1907.06051v2.pdf``). Titles are not in the PDFs reliably, so we keep a
small JSON sidecar next to the papers: ``{ "1907.06051v2.pdf": "Title…" }``.

It is populated for free as papers are downloaded, and can be backfilled for an
existing corpus via :func:`iter_backfill_titles`. Display always falls back to
the id, so a missing map never breaks the UI.
"""

import json
import re
from collections.abc import Generator
from pathlib import Path

import arxiv

from inquiro.config import PAPERS_DIR

TITLES_PATH = PAPERS_DIR / "titles.json"
_VERSION_RE = re.compile(r"v\d+$")


def _stem(source: str) -> str:
    return source[:-4] if source.endswith(".pdf") else source


def _base_id(stem: str) -> str:
    """Strip a trailing version suffix so 1907.06051v2 and v1 compare equal."""
    return _VERSION_RE.sub("", stem)


def load_titles(titles_path: Path = TITLES_PATH) -> dict[str, str]:
    if not titles_path.exists():
        return {}
    try:
        data = json.loads(titles_path.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_titles(titles: dict[str, str], titles_path: Path = TITLES_PATH) -> None:
    titles_path.parent.mkdir(parents=True, exist_ok=True)
    titles_path.write_text(json.dumps(titles, indent=2, ensure_ascii=False))


def resolve_source(source: str, titles: dict[str, str] | None = None) -> str:
    """Return the paper title for a stored ``source`` filename, else its id."""
    if titles is None:
        titles = load_titles()

    if source in titles:
        return titles[source]

    stem = _stem(source)
    if stem in titles:
        return titles[stem]

    base = _base_id(stem)
    for key, value in titles.items():
        if _base_id(_stem(key)) == base:
            return value

    return stem  # show the bare id rather than "….pdf"


def untitled_pdfs(papers_dir: Path = PAPERS_DIR) -> list[str]:
    """Filenames of locally present, arXiv-style PDFs that have no title yet."""
    titles = load_titles()
    missing: list[str] = []
    for path in sorted(papers_dir.glob("*.pdf")):
        if path.name in titles:
            continue
        if not path.stem[:1].isdigit():  # only arXiv ids are resolvable
            continue
        missing.append(path.name)
    return missing


def iter_backfill_titles(
    papers_dir: Path = PAPERS_DIR,
    batch_size: int = 50,
) -> Generator[tuple[int, int, str], None, None]:
    """Look up titles for local PDFs missing one via the arXiv API.

    Yields ``(done, total, label)`` after each paper so a progress bar can be
    driven, and persists the map after every batch so progress is durable.
    """
    titles = load_titles()
    missing = untitled_pdfs(papers_dir)
    total = len(missing)
    if total == 0:
        return

    client = arxiv.Client()
    done = 0
    for start in range(0, total, batch_size):
        batch = missing[start : start + batch_size]
        id_list = [_stem(name) for name in batch]

        found: dict[str, str] = {}
        try:
            for result in client.results(arxiv.Search(id_list=id_list)):
                found[_base_id(result.get_short_id())] = result.title.strip()
        except Exception:
            pass  # network hiccup: leave this batch unresolved, keep going

        for name in batch:
            title = found.get(_base_id(_stem(name)))
            if title:
                titles[name] = title
            done += 1
            yield done, total, (title or _stem(name))

        save_titles(titles)
