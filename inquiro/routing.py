from typing import cast

from langchain_cohere import ChatCohere

from inquiro.state import RAGState, Route
from inquiro.utils import content_to_text

VALID_ROUTES: tuple[Route, ...] = ("definition", "comparison", "method")


def normalize_route(raw_route: str) -> Route:
    route = raw_route.strip().lower()
    if route in VALID_ROUTES:
        return cast(Route, route)

    for candidate in VALID_ROUTES:
        if candidate in route:
            return candidate

    return "definition"


def route_question(llm: ChatCohere, state: RAGState) -> RAGState:
    prompt = f"""
Classify this research-paper question into exactly one category:

definition: asks what a concept means
comparison: compares two or more methods/papers
method: asks how an approach works

Question: {state["question"]}

Return only one label.
"""
    response = llm.invoke(prompt)
    route = normalize_route(content_to_text(response.content))
    return {"route": route}


def choose_retriever(state: RAGState) -> Route:
    return state["route"]
