from functools import partial

from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from inquiro.evaluation import evaluate_answer
from inquiro.generation import generate_answer
from inquiro.retrieval import (
    retrieve_comparison,
    retrieve_definition,
    retrieve_method,
)
from inquiro.resources import build_llm, build_vectorstore
from inquiro.routing import choose_retriever, route_question
from inquiro.state import RAGState


def build_graph(llm: BaseChatModel | None = None, vectorstore: Chroma | None = None):
    llm = llm or build_llm()
    vectorstore = vectorstore or build_vectorstore()

    builder = StateGraph(RAGState)
    builder.add_node("route_question", partial(route_question, llm))
    builder.add_node("retrieve_definition", partial(retrieve_definition, vectorstore))
    builder.add_node("retrieve_comparison", partial(retrieve_comparison, vectorstore))
    builder.add_node("retrieve_method", partial(retrieve_method, vectorstore))
    builder.add_node("generate_answer", partial(generate_answer, llm))
    builder.add_node("evaluate_answer", partial(evaluate_answer, llm))

    builder.add_edge(START, "route_question")
    builder.add_conditional_edges(
        "route_question",
        choose_retriever,
        {
            "definition": "retrieve_definition",
            "comparison": "retrieve_comparison",
            "method": "retrieve_method",
        },
    )
    builder.add_edge("retrieve_definition", "generate_answer")
    builder.add_edge("retrieve_comparison", "generate_answer")
    builder.add_edge("retrieve_method", "generate_answer")
    builder.add_edge("generate_answer", "evaluate_answer")
    builder.add_edge("evaluate_answer", END)

    return builder.compile()


_graph: CompiledStateGraph | None = None


def get_graph() -> CompiledStateGraph:
    """Build the graph once per process and reuse the cached instance.

    Building the graph loads the HuggingFace embedding model from disk, which is
    slow, so callers should prefer this over ``build_graph`` for repeated use.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def answer_question(question: str) -> RAGState:
    graph = get_graph()
    return graph.invoke({"question": question})
