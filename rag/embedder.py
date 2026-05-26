from sentence_transformers import SentenceTransformer

_model = None 

def get_model():
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    get_model()
    
    encoded = _model.encode(texts, show_progress_bar=True)
    
    return encoded