from sentence_transformers import SentenceTransformer

_model = None 

def get_model():
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    model = get_model()
    
    encoded = model.encode(texts, show_progress_bar=True).to_list()
    
    return encoded