from pathlib import Path

def load_and_chunk(doc_path: str, chunk_size: int = 512, overlap: int = 50) -> list[dict]:
    """Return list of {text, source, chunk_id} dicts."""

    text = Path(doc_path).read_text(encoding="utf-8")
    chunks = []
    words = text.split()
    
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunks.append({
            "text": " ".join(chunk_words),
            "source": doc_path,
            "chunk_id": f"{Path(doc_path).stem}_{i}",
        })

        i += chunk_size-overlap
    
    return chunks 
