"""
RAG Retrieval module for finding top-N similar historical tickets and resolutions.
"""

from typing import List, Dict, Any, Tuple
from .schemas import TicketInput
from .embeddings import VectorStoreManager
from .config import TOP_K_SIMILAR


class TicketRetriever:
    """
    Retrieves contextually similar historical resolved tickets using the local vector store.
    """

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def retrieve_similar_tickets(
        self, 
        ticket: TicketInput, 
        top_k: int = TOP_K_SIMILAR
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Queries vector store with incoming ticket text to retrieve top_k historical context matches.

        Returns:
            List of tuples: (metadata_dict, similarity_score)
        """
        query_text = f"Subject: {ticket.subject}\nBody: {ticket.body}"
        return self.vector_store.search(query_text, top_k=top_k)

    def format_retrieved_context(self, retrieved_matches: List[Tuple[Dict[str, Any], float]]) -> str:
        """
        Formats retrieved historical tickets into a clean string context block for LLM prompts.
        """
        if not retrieved_matches:
            return "No historical reference tickets found."

        formatted_blocks = []
        for rank, (meta, score) in enumerate(retrieved_matches, 1):
            block = (
                f"Reference Ticket #{rank} [ID: {meta['ticket_id']}] (Similarity Score: {score:.2f}):\n"
                f"- Category: {meta.get('category', 'N/A')}\n"
                f"- Urgency: {meta.get('urgency', 'N/A')}\n"
                f"- Subject: {meta['subject']}\n"
                f"- Resolution: {meta['resolution']}\n"
            )
            formatted_blocks.append(block)

        return "\n".join(formatted_blocks)