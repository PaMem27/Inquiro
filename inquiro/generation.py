from langchain_core.language_models.chat_models import BaseChatModel

from inquiro.state import RAGState
from inquiro.utils import content_to_text, format_docs


def generate_answer(llm: BaseChatModel, state: RAGState) -> RAGState:
    prompt = f"""
You are Inquiro, a research assistant for academic papers.

Question type: {state["route"]}

Use only the context below.
If the answer is not supported, say you don't know.

For definition questions: explain clearly.
For comparison questions: contrast methods directly.
For method questions: answer step-by-step.

Context:
{format_docs(state["docs"])}

Question:
{state["question"]}

Answer:
"""
    response = llm.invoke(prompt)
    return {"answer": content_to_text(response.content)}
