import os
import chromadb
from chromadb.config import Settings
from embeddings import embed_texts, get_embedding_dimension

_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "reposage_code_chunks")
_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")

_client = None
_collection = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_collection():
    global _collection
    if _collection is None:
        client = _get_client()
        try:
            _collection = client.get_collection(_COLLECTION_NAME)
        except ValueError:
            _collection = client.create_collection(
                _COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
    return _collection


def ingest_chunks(
    chunks: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> int:
    collection = _get_collection()
    embeddings = embed_texts(chunks)
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
    )
    return len(chunks)


def query_similar(
    query: str,
    n_results: int = 5,
) -> list[dict]:
    collection = _get_collection()
    query_embedding = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    output = []
    if not results["ids"]:
        return output
    for i, doc_id in enumerate(results["ids"][0]):
        output.append({
            "id": doc_id,
            "score": float(results["distances"][0][i]) if results["distances"] else 0.0,
            "document": results["documents"][0][i] if results["documents"] else "",
            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
        })
    return output


def get_collection_stats() -> dict:
    collection = _get_collection()
    count = collection.count()
    return {
        "collection": _COLLECTION_NAME,
        "chunk_count": count,
        "embedding_dimension": get_embedding_dimension(),
    }
