from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere
from langchain_huggingface import HuggingFaceEmbeddings

from inquiro.config import CHROMA_DIR, COHERE_MODEL, EMBEDDING_MODEL


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore() -> Chroma:
    embeddings = build_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def build_llm() -> ChatCohere:
    load_dotenv()
    return ChatCohere(model=COHERE_MODEL)
