import shutil

import streamlit as st

from inquiro.config import CHROMA_DIR
from inquiro.corpus import (
    clear_chroma_cache,
    corpus_ready,
    iter_ingest_papers,
    iter_load_papers,
    mark_corpus_ready,
    read_corpus_info,
)
from inquiro.downloader import iter_download_papers
from inquiro.graph import build_graph
from inquiro.titles import iter_backfill_titles, load_titles, resolve_source, untitled_pdfs
from inquiro.utils import source_names

# Bounds and default for how many arXiv papers to fetch on first-run setup.
# The default is demo-friendly (a fresh clone indexes in a couple of minutes),
# but the user picks the actual count on the setup screen.
DEFAULT_PAPERS = 40
MIN_PAPERS = 5
MAX_PAPERS = 200

ROUTE_COLORS = {
    "definition": "#2e7d32",
    "method": "#1565c0",
    "comparison": "#ef6c00",
}


st.set_page_config(page_title="Inquiro", layout="centered")
st.title("🔍 Inquiro")
st.caption("An agentic RAG assistant for research papers")


@st.cache_resource
def load_graph():
    return build_graph()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    if corpus_ready():
        info = read_corpus_info()
        if info:
            topic, count = info
            if topic:
                st.markdown(f"**Topic:** {topic}")
            if count:
                st.markdown(f"**Papers indexed:** {count}")

        st.divider()

        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

        # Offer to resolve arXiv ids → titles when some are still missing.
        missing = untitled_pdfs()
        if missing:
            st.divider()
            st.caption(f"{len(missing)} sources still show as arXiv ids.")
            if st.button("🏷️ Fetch paper titles"):
                with st.status("Fetching titles from arXiv…", expanded=True) as status:
                    bar = st.progress(0.0)
                    info = st.empty()
                    for done, total, label in iter_backfill_titles():
                        bar.progress(done / total)
                        info.markdown(f"`{label[:70]}`")
                    status.update(label="✅ Titles updated", state="complete", expanded=False)
                st.rerun()

        st.divider()

        with st.expander("⚠️ Reset corpus"):
            st.warning(
                "Deletes the vector index and returns to setup. "
                "Papers already in `papers/` are kept on disk."
            )
            if st.button("Reset", type="secondary"):
                # Drop the cached graph (its open DB handle) and chromadb's
                # global client registry *before* deleting the files, so the
                # next build can't write through a stale handle.
                load_graph.clear()
                clear_chroma_cache()
                if CHROMA_DIR.exists():
                    shutil.rmtree(CHROMA_DIR)
                st.session_state.pop("messages", None)
                st.rerun()


# ── Setup flow ─────────────────────────────────────────────────────────────────
if not corpus_ready():
    st.subheader("📚 First-time Setup")
    st.markdown(
        "No paper corpus found. Enter a research topic and choose how many "
        "papers to pull from arXiv. Inquiro will download them, read them, and "
        "build a searchable vector index — then unlock the chat."
    )

    topic = st.text_input(
        "Research topic",
        placeholder="e.g. structured pruning transformer language models",
        help="Used as the arXiv search query.",
    )

    num_papers = st.slider(
        "Number of papers to fetch",
        min_value=MIN_PAPERS,
        max_value=MAX_PAPERS,
        value=DEFAULT_PAPERS,
        step=5,
        help="More papers means broader coverage but a longer first-run build.",
    )

    if st.button("🚀 Build Corpus", type="primary", disabled=not topic.strip()):
        topic = topic.strip()

        # ── Step 1: Download from arXiv ────────────────────────────────────
        with st.status("📥 Downloading papers from arXiv…", expanded=True) as dl_status:
            dl_bar = st.progress(0.0)
            dl_info = st.empty()
            downloaded = skipped = failed = 0

            for idx, total, paper_id, title, result in iter_download_papers(
                topic, max_results=num_papers
            ):
                if result == "downloaded":
                    downloaded += 1
                    icon = "✅"
                elif result == "skipped":
                    skipped += 1
                    icon = "⏭️"
                else:
                    failed += 1
                    icon = "❌"

                dl_bar.progress(min(idx / total, 1.0))
                short_title = title[:65] + "…" if len(title) > 65 else title
                dl_info.markdown(f"{icon} `{paper_id}` — {short_title}")

            dl_bar.progress(1.0)
            dl_status.update(
                label=f"✅ {downloaded} downloaded · {skipped} already had · {failed} failed",
                state="complete",
                expanded=False,
            )

        if downloaded + skipped == 0:
            st.error(
                "No papers were returned by arXiv. Try a different topic or check your connection."
            )
            st.stop()

        # ── Step 2: Read PDFs ──────────────────────────────────────────────
        with st.status("📖 Reading PDFs…", expanded=True) as read_status:
            read_bar = st.progress(0.0)
            read_info = st.empty()
            papers = []

            for idx, total, name, doc in iter_load_papers():
                if total > 0:
                    read_bar.progress(idx / total)
                read_info.markdown(f"`{name}`")
                if doc:
                    papers.append(doc)

            read_bar.progress(1.0)
            read_status.update(
                label=f"📖 {len(papers)} papers read",
                state="complete",
                expanded=False,
            )

        if not papers:
            st.error("Could not extract text from any PDF. Check that the papers/ directory is populated.")
            st.stop()

        # ── Step 3: Embed and index ────────────────────────────────────────
        with st.status("🔢 Building vector index…", expanded=True) as idx_status:
            idx_bar = st.progress(0.0)
            idx_info = st.empty()
            papers_indexed = 0

            for idx, total, source in iter_ingest_papers(papers):
                idx_bar.progress(idx / total)
                idx_info.markdown(f"Embedding `{source}`…")
                papers_indexed = total

            idx_bar.progress(1.0)

            # Record completion so future launches skip setup.
            mark_corpus_ready(topic, papers_indexed)

            idx_status.update(
                label=f"✅ {papers_indexed} papers indexed into ChromaDB",
                state="complete",
                expanded=False,
            )

        st.balloons()
        st.success("Corpus ready! Loading the chat interface…")
        st.rerun()

    st.stop()


# ── Chat UI ────────────────────────────────────────────────────────────────────
graph = load_graph()
TITLES = load_titles()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_route_badge(route: str) -> None:
    color = ROUTE_COLORS.get(route, "#555555")
    st.markdown(
        f'<span style="background-color: {color}; color: white; '
        f'padding: 2px 10px; border-radius: 6px; font-size: 0.8em; '
        f'font-weight: 600;">{route}</span>',
        unsafe_allow_html=True,
    )


def render_assistant_message(content: str, meta: dict | None) -> None:
    if meta and meta.get("route"):
        render_route_badge(meta["route"])

    st.markdown(content)

    if not meta:
        return

    sources = meta.get("sources") or []
    with st.expander("📄 Sources"):
        if sources:
            seen: set[str] = set()
            for source in sources:
                title = resolve_source(source, TITLES)
                if title in seen:
                    continue
                seen.add(title)
                st.markdown(f"- {title}")
        else:
            st.markdown("_No sources found._")

    with st.expander("✅ Evaluation"):
        eval_score = meta.get("eval_score", "FAIL")
        if eval_score == "PASS":
            st.success(eval_score)
        else:
            st.error(eval_score)
        if meta.get("eval_feedback"):
            st.markdown(meta["eval_feedback"])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message["content"], message.get("meta"))
        else:
            st.markdown(message["content"])


question = st.chat_input("Ask about your research corpus…")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "meta": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = graph.invoke({"question": question})

        answer = result.get("answer", "")
        meta = {
            "route": result.get("route"),
            "sources": source_names(result.get("docs", [])),
            "eval_score": result.get("eval_score", "FAIL"),
            "eval_feedback": result.get("eval_feedback"),
        }
        render_assistant_message(answer, meta)

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
