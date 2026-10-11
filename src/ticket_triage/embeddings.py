"""
Embedding and Vector Store management module using sentence-transformers and FAISS.
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from .schemas import HistoricalTicket
from .config import EMBEDDING_MODEL_NAME, VECTOR_STORE_PATH


class VectorStoreManager:
    """
    Manages local vector embedding generation, FAISS index construction,
    and index persistence to disk for HistoricalTicket objects.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

    def _prepare_text(self, ticket: HistoricalTicket) -> str:
        """Combines subject and body into a single searchable text representation."""
        return f"Subject: {ticket.subject}\nBody: {ticket.body}"

    def build_index(self, tickets: List[HistoricalTicket]) -> None:
        """Generates embeddings for historical tickets and builds FAISS index."""
        texts = [self._prepare_text(t) for t in tickets]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        
        # Preserve resolution and category metadata for RAG context building[cite: 1]
        self.metadata = [
            {
                "ticket_id": t.ticket_id,
                "subject": t.subject,
                "body": t.body,
                "resolution": t.resolution,
                "category": t.category.value if t.category else None,
                "urgency": t.urgency.value if t.urgency else None
            }
            for t in tickets
        ]

    def search(self, query_text: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieves top_k most similar historical tickets given a query string.
        
        Returns:
            List of tuples: (ticket_metadata_dict, similarity_score)
        """
        if self.index is None:
            raise ValueError("Vector index is not initialized. Build or load index first.")

        query_vector = self.model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector.astype(np.float32), top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))
                
        return results

    def save(self, directory_path: str = str(VECTOR_STORE_PATH)) -> None:
        """Persists FAISS index and metadata to disk."""
        os.makedirs(directory_path, exist_ok=True)
        index_file = os.path.join(directory_path, "faiss.index")
        meta_file = os.path.join(directory_path, "metadata.pkl")

        faiss.write_index(self.index, index_file)
        with open(meta_file, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, directory_path: str = str(VECTOR_STORE_PATH)) -> bool:
        """Loads FAISS index and metadata from disk if available."""
        index_file = os.path.join(directory_path, "faiss.index")
        meta_file = os.path.join(directory_path, "metadata.pkl")

        if os.path.exists(index_file) and os.path.exists(meta_file):
            self.index = faiss.read_index(index_file)
            with open(meta_file, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        return False