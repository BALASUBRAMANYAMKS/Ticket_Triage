"""
Data loading module with strict schema verification and duplicate detection.
Parses raw JSON ticket datasets into validated Pydantic models.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Set
from .schemas import TicketInput, HistoricalTicket
from .config import (
    HISTORICAL_TICKETS_PATH,
    DAY1_TICKETS_PATH,
    HELDOUT_TICKETS_PATH
)


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Loads raw JSON data from a given path and validates that the root structure is a list.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Required dataset file not found at: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Invalid dataset format in {file_path.name}: Expected a JSON list of ticket objects.")

    return data


def _extract_ticket_id(item: Dict[str, Any], file_path: Path) -> str:
    """Extracts and validates ticket_id string, failing fast if missing or blank."""
    raw_id = item.get("ticket_id") or item.get("id")
    
    if raw_id is None or str(raw_id).strip() == "":
        raise ValueError(f"Malformed record in {file_path.name}: Ticket ID is missing or blank.")
    
    return str(raw_id).strip()


def load_tickets(file_path: Path) -> List[TicketInput]:
    """
    Loads incoming test tickets and validates uniqueness of ticket IDs.
    """
    raw_data = load_json_file(file_path)
    tickets: List[TicketInput] = []
    seen_ids: Set[str] = set()

    for idx, item in enumerate(raw_data):
        ticket_id = _extract_ticket_id(item, file_path)
        
        if ticket_id in seen_ids:
            raise ValueError(f"Duplicate ticket ID '{ticket_id}' detected in {file_path.name} at record index {idx}.")
        
        seen_ids.add(ticket_id)
        
        ticket = TicketInput(
            ticket_id=ticket_id,
            subject=str(item.get("subject", "")).strip(),
            body=str(item.get("body") or item.get("description") or "").strip()
        )
        tickets.append(ticket)
        
    return tickets


def load_historical_tickets(file_path: Path = HISTORICAL_TICKETS_PATH) -> List[HistoricalTicket]:
    """
    Loads 120 historical tickets, strictly preserving resolutions for RAG retrieval context[cite: 1].
    """
    raw_data = load_json_file(file_path)
    historical_tickets: List[HistoricalTicket] = []
    seen_ids: Set[str] = set()

    for idx, item in enumerate(raw_data):
        ticket_id = _extract_ticket_id(item, file_path)
        
        if ticket_id in seen_ids:
            raise ValueError(f"Duplicate ticket ID '{ticket_id}' detected in historical dataset at index {idx}.")
        
        resolution = item.get("resolution")
        if not resolution or str(resolution).strip() == "":
            raise ValueError(f"Historical ticket '{ticket_id}' is missing a required resolution.")

        seen_ids.add(ticket_id)

        historical_ticket = HistoricalTicket(
            ticket_id=ticket_id,
            subject=str(item.get("subject", "")).strip(),
            body=str(item.get("body") or item.get("description") or "").strip(),
            resolution=str(resolution).strip(),
            category=item.get("category"),
            urgency=item.get("urgency")
        )
        historical_tickets.append(historical_ticket)

    return historical_tickets


def load_day1_tickets() -> List[TicketInput]:
    """Loads Day 1 incoming test tickets."""
    return load_tickets(DAY1_TICKETS_PATH)


def load_heldout_tickets() -> List[TicketInput]:
    """Loads held-out incoming test tickets."""
    return load_tickets(HELDOUT_TICKETS_PATH)