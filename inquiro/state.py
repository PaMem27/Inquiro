from typing import Literal, TypedDict

from langchain_core.documents import Document

Route = Literal["definition", "comparison", "method"]


class RAGState(TypedDict, total=False):
    question: str
    route: Route
    docs: list[Document]
    answer: str
    eval_score: str
    eval_feedback: str
