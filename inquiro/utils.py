from typing import Any

from langchain_core.documents import Document


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    text_parts.append(text)
            else:
                text_parts.append(str(block))

        return "\n".join(text_parts)

    return str(content)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def source_names(docs: list[Document]) -> list[str]:
    return [str(doc.metadata.get("source", "unknown")) for doc in docs]
