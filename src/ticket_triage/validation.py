"""Schema validation and one-attempt response repair."""

import logging
from pathlib import Path
from typing import Callable, List, Optional

from pydantic import ValidationError

from .config import LOG_FILE_PATH, MAX_SCHEMA_RETRY_ATTEMPTS
from .schemas import TicketCategory, TriageAction, TriageOutput, UrgencyLevel


logger = logging.getLogger("ticket_triage.validation")
if not logger.handlers:
    Path(LOG_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


RepairCallback = Callable[[str], str]


def build_safe_fallback(
    ticket_id: str,
    reason: str,
    retrieved_ticket_ids: Optional[List[str]] = None,
) -> TriageOutput:
    """Create a valid result that sends the ticket to a human."""
    return TriageOutput(
        ticket_id=ticket_id,
        category=TicketCategory.TECHNICAL,
        urgency=UrgencyLevel.P1,
        action=TriageAction.ESCALATE,
        reasoning=" ".join(reason.split()[:50]),
        retrieved_ticket_ids=retrieved_ticket_ids or [],
        confidence=0.0,
    )


def _validate_response(
    raw_text: str,
    ticket_id: str,
    retrieved_ticket_ids: Optional[List[str]],
) -> TriageOutput:
    """Validate JSON, then restore application-owned IDs."""
    output = TriageOutput.model_validate_json(raw_text)
    output.ticket_id = ticket_id
    output.retrieved_ticket_ids = retrieved_ticket_ids or []
    return output


def _log_validation_error(ticket_id: str, exc: ValidationError) -> None:
    """Log field locations and messages without logging customer/model input."""
    errors = exc.errors(
        include_input=False,
        include_context=False,
        include_url=False,
    )
    concise_errors = [
        {"loc": error.get("loc"), "msg": error.get("msg"), "type": error.get("type")}
        for error in errors
    ]
    logger.warning(
        "Invalid response for ticket_id=%s; validation_errors=%s",
        ticket_id,
        concise_errors,
    )


def validate_and_repair_response(
    raw_response: str,
    fallback_ticket_id: str,
    repair_callback: Optional[RepairCallback] = None,
    retrieved_ticket_ids: Optional[List[str]] = None,
) -> TriageOutput:
    """
    Validate a model response. On failure, make one repair attempt,
    then return the safe escalation fallback.
    """
    try:
        return _validate_response(
            raw_response,
            fallback_ticket_id,
            retrieved_ticket_ids,
        )
    except ValidationError as exc:
        _log_validation_error(fallback_ticket_id, exc)
    except Exception as exc:
        logger.warning(
            "Could not parse response for ticket_id=%s (%s: %s); "
            "repair will be attempted.",
            fallback_ticket_id,
            type(exc).__name__,
            str(exc)[:300],
        )

    repair_attempts = max(0, MAX_SCHEMA_RETRY_ATTEMPTS)

    for attempt in range(repair_attempts):
        if repair_callback is None:
            logger.error(
                "No repair callback available for ticket_id=%s.",
                fallback_ticket_id,
            )
            break

        repair_prompt = f"""
Repair the following response into one valid JSON object with these fields:
ticket_id, category, urgency, action, reasoning,
retrieved_ticket_ids, confidence.

Preserve the intended decision where possible. Do not invent a new
classification. Use {fallback_ticket_id!r} as the ticket_id.
Return only a JSON object.

Invalid response:
{raw_response[:12000]}
"""

        try:
            repaired_text = repair_callback(repair_prompt)
            output = _validate_response(
                repaired_text,
                fallback_ticket_id,
                retrieved_ticket_ids,
            )
            logger.info(
                "Response repaired for ticket_id=%s on attempt %d.",
                fallback_ticket_id,
                attempt + 1,
            )
            return output

        except ValidationError as exc:
            _log_validation_error(fallback_ticket_id, exc)
        except Exception as exc:
            # Log the error, not the raw response or repair prompt.
            logger.warning(
                "Repair attempt %d failed for ticket_id=%s (%s: %s).",
                attempt + 1,
                fallback_ticket_id,
                type(exc).__name__,
                str(exc)[:300],
            )

    logger.error(
        "Repair failed; using escalation fallback for ticket_id=%s.",
        fallback_ticket_id,
    )
    return build_safe_fallback(
        ticket_id=fallback_ticket_id,
        reason="Invalid model response; human review required.",
        retrieved_ticket_ids=retrieved_ticket_ids,
    )