import re

from langchain_cohere import ChatCohere

from inquiro.state import RAGState
from inquiro.utils import content_to_text, format_docs


def parse_score(text: str) -> str:
    """Return "PASS" only when the response clearly indicates a pass.

    A pass is recognised when the stripped response is exactly "PASS", or when a
    line reads "score: PASS" (case-insensitive). Anything else is a FAIL, which
    avoids false positives from phrases like "does not pass" or "passable".
    """
    stripped = text.strip()
    if stripped.upper() == "PASS":
        return "PASS"

    for line in text.splitlines():
        match = re.search(r"score:\s*(\w+)", line, re.IGNORECASE)
        if match and match.group(1).upper() == "PASS":
            return "PASS"

    return "FAIL"


def evaluate_answer(llm: ChatCohere, state: RAGState) -> RAGState:
    prompt = f"""
Evaluate the answer against the retrieved context.

Question:
{state["question"]}

Context:
{format_docs(state["docs"])}

Answer:
{state["answer"]}

Return:
score: PASS or FAIL
feedback: one short reason

Criteria:
- Is the answer grounded in the context?
- Does it avoid unsupported claims?
- Does it answer the question?
"""
    response = llm.invoke(prompt)
    text = content_to_text(response.content)
    score = parse_score(text)

    return {
        "eval_score": score,
        "eval_feedback": text,
    }
