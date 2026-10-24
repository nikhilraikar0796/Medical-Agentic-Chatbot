
from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def clients():
    if not settings.pinecone_api_key:
        raise ValueError(
            "PINECONE_API_KEY must be set in the environment variables."
        )

    pc = Pinecone(api_key=settings.pinecone_api_key)
    return pc


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def ensure_index():
    pc = clients()

    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    return pc.Index(settings.pinecone_index)


def embed(texts):
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)


def search_kb(query, top_k=5):
    index = ensure_index()
    vector = get_embeddings().embed_query(query)

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
    )

    return [
        {
            "page": match.metadata.get("page"),
            "text": match.metadata.get("text", ""),
            "score": round(match.score, 3),
            "source": "Gale Encyclopedia (2002)",
        }
        for match in result.matches
        if match.score >= 0.30
    ]
