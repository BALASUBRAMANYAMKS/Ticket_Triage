"""
Batch Processing Engine.
Runs the complete RAG + Decision pipeline across ticket batches and exports structured outputs.
"""

import json
from typing import List, Dict, Any
from pathlib import Path

from .schemas import TicketInput, TriageOutput
from .data_loader import load_historical_tickets, load_day1_tickets, load_heldout_tickets
from .embeddings import VectorStoreManager
from .retrieval import TicketRetriever
from .decision import DecisionEngine
from .config import OUTPUTS_DIR, VECTOR_STORE_PATH


class BatchProcessor:
    """
    Executes batch processing for incoming support ticket datasets.
    """

    def __init__(self, rebuild_index: bool = False):
        self.vector_store = VectorStoreManager()
        
        # Load or build FAISS vector index from historical resolved tickets
        if not rebuild_index and self.vector_store.load(str(VECTOR_STORE_PATH)):
            print("[BatchProcessor] Loaded existing FAISS vector store from disk.")
        else:
            print("[BatchProcessor] Building new FAISS vector store from historical tickets...")
            historical_tickets = load_historical_tickets()
            self.vector_store.build_index(historical_tickets)
            self.vector_store.save(str(VECTOR_STORE_PATH))
            print(f"[BatchProcessor] Indexed {len(historical_tickets)} historical tickets successfully.")

        self.retriever = TicketRetriever(self.vector_store)
        self.decision_engine = DecisionEngine(self.retriever)

    def process_tickets(self, tickets: List[TicketInput]) -> List[TriageOutput]:
        """Runs decision engine sequentially across a list of input tickets."""
        results: List[TriageOutput] = []
        for idx, ticket in enumerate(tickets, 1):
            print(f"[{idx}/{len(tickets)}] Triaging Ticket ID: {ticket.ticket_id}...")
            output = self.decision_engine.process_ticket(ticket)
            results.append(output)
            
        return results

    def save_results(self, results: List[TriageOutput], output_path: Path = OUTPUTS_DIR / "triage_results.json") -> None:
        """Serializes output results to valid JSON[cite: 1]."""
        dict_results = [res.model_dump() for res in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dict_results, f, indent=2)
        print(f"[BatchProcessor] Saved batch results to: {output_path}")


def run_full_20_ticket_batch() -> List[TriageOutput]:
    """
    Convenience runner processing all 20 tickets (Day 1 + Held-out set)[cite: 1].
    """
    day1_tickets = load_day1_tickets()
    heldout_tickets = load_heldout_tickets()
    all_tickets = day1_tickets + heldout_tickets

    print(f"\n=== Starting Batch Run Over {len(all_tickets)} Tickets ===")
    processor = BatchProcessor(rebuild_index=False)
    results = processor.process_tickets(all_tickets)
    processor.save_results(results)
    
    return results


if __name__ == "__main__":
    run_full_20_ticket_batch()