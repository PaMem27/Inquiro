from langchain_chroma import Chroma

from inquiro.state import RAGState

DEFINITION_RESULT_COUNT = 4
METHOD_RESULT_COUNT = 6
COMPARISON_RESULT_COUNT = 8


def retrieve_definition(vectorstore: Chroma, state: RAGState) -> RAGState:
    docs = vectorstore.similarity_search(
        state["question"],
        k=DEFINITION_RESULT_COUNT,
    )
    return {"docs": docs}


def retrieve_method(vectorstore: Chroma, state: RAGState) -> RAGState:
    query = state["question"] + " method algorithm mechanism implementation procedure"
    docs = vectorstore.similarity_search(query, k=METHOD_RESULT_COUNT)
    return {"docs": docs}


def retrieve_comparison(vectorstore: Chroma, state: RAGState) -> RAGState:
    query = state["question"] + " compare versus difference advantage limitation"
    docs = vectorstore.similarity_search(query, k=COMPARISON_RESULT_COUNT)
    return {"docs": docs}
