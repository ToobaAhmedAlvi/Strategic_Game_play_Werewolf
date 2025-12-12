from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    """
    Uses a local MiniLM model to generate 384-d embeddings.
    No OpenAI API required.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)

    def embed(self, texts):
        """
        texts: list[str]
        returns: list[np.ndarray] shape (len(texts), 384)
        """
        return self.encoder.encode(texts, convert_to_numpy=True).tolist()
