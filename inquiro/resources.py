from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from inquiro.config import CHROMA_DIR, EMBEDDING_MODEL, LLM_MODEL


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore() -> Chroma:
    embeddings = build_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def build_llm() -> BaseChatModel:
    load_dotenv()
    return ChatGroq(model=LLM_MODEL)
