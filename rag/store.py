import chromadb

_client = None
_collection = None

def get_collection(collection_name: str = "rag_docs"):
    if _collection is None:
        _client = chromadb.PersistentClient(".chroma")
        _collection = _client.get_or_create_collection(collection_name)
    
    return _collection

def upsert(chunks: list[dict], embeddings: list[list[float]]):
    col = get_collection()
    
    col.upsert(
        ids = [c["chunk_id"] for c in chunks],
        embeddings = embeddings,
        metadatas = [{"source": c["source"]} for c in chunks],
        documents = [c["text"] for c in chunks],
    )
    
def query(embedding: list[float], top_k: int = 5) -> list[dict]:
    col = get_collection()
    
    res = col.query(
        query_embeddings=[embedding],
        n_results = top_k
    )
    
    return [
        {"text": doc, "source": meta["source"], "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        )
    ]
    
    