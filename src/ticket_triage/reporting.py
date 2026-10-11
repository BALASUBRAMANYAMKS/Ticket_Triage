"""Generate a Markdown report from batch results and evaluation metrics."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    CONFIDENCE_ABSTENTION_THRESHOLD,
    HISTORICAL_TICKETS_PATH,
    OUTPUTS_DIR,
)
from .data_loader import load_json_file
from .schemas import TriageOutput


def _enum_value(value: Any) -> str:
    """Return an enum's value or a readable string."""
    return str(getattr(value, "value", value))


def _markdown_cell(value: Any) -> str:
    """Flatten and escape text so it stays inside one Markdown table cell."""
    text = str(value or "")
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return text.replace("|", r"\|")


def _load_historical_ticket_lookup() -> Dict[str, Dict[str, Any]]:
    """Load historical records so the report can show retrieved resolutions."""
    records = load_json_file(HISTORICAL_TICKETS_PATH)

    return {
        str(record.get("ticket_id") or record.get("id")): record
        for record in records
        if record.get("ticket_id") or record.get("id")
    }


def generate_markdown_report(
    results: List[TriageOutput],
    eval_metrics: Optional[Dict[str, Any]] = None,
    output_path: Path = OUTPUTS_DIR / "triage_report.md",
) -> str:
    """Write a human-readable batch report and return its Markdown text."""

    historical_by_id = _load_historical_ticket_lookup()

    total_tickets = len(results)
    categories: Dict[str, int] = {}
    urgencies: Dict[str, int] = {}
    actions: Dict[str, int] = {}
    low_confidence_count = 0

    for result in results:
        category = _enum_value(result.category)
        urgency = _enum_value(result.urgency)
        action = _enum_value(result.action)

        categories[category] = categories.get(category, 0) + 1
        urgencies[urgency] = urgencies.get(urgency, 0) + 1
        actions[action] = actions.get(action, 0) + 1

        if result.confidence < CONFIDENCE_ABSTENTION_THRESHOLD:
            low_confidence_count += 1

    lines = [
        "# Support Ticket Triage Report",
        "",
        "**Dataset:** Synthetic tickets",
        f"**Total processed tickets:** {total_tickets}",
        "",
        "## Executive Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Auto-resolved | {actions.get('auto_resolve', 0)} |",
        f"| Escalated | {actions.get('escalate', 0)} |",
        f"| Clarification requested | {actions.get('request_clarification', 0)} |",
        (
            f"| Low confidence (< {CONFIDENCE_ABSTENTION_THRESHOLD:.2f}) "
            f"| {low_confidence_count} |"
        ),
        "",
        "## Category Distribution",
        "",
    ]

    if categories:
        for category in sorted(categories):
            lines.append(f"- **{_markdown_cell(category)}**: {categories[category]}")
    else:
        lines.append("- No tickets processed.")

    lines.extend(["", "## Urgency Distribution", ""])

    if urgencies:
        for urgency in sorted(urgencies):
            lines.append(f"- **{_markdown_cell(urgency)}**: {urgencies[urgency]}")
    else:
        lines.append("- No tickets processed.")

    if eval_metrics:
        if eval_metrics.get("status") == "complete" and "accuracies" in eval_metrics:
            accuracy = eval_metrics["accuracies"]
            lines.extend(
                [
                    "",
                    "## Held-Out Gold Label Accuracy",
                    "",
                    f"- **Held-out sample size:** {eval_metrics['heldout_sample_size']}",
                    f"- **Category accuracy:** {accuracy['category_accuracy'] * 100:.1f}%",
                    f"- **Urgency accuracy:** {accuracy['urgency_accuracy'] * 100:.1f}%",
                    f"- **Action accuracy:** {accuracy['action_accuracy'] * 100:.1f}%",
                    f"- **Overall exact match:** {accuracy['overall_exact_match'] * 100:.1f}%",
                    "",
                ]
            )

            confusion_matrix = eval_metrics.get("confusion_matrix", {})
            headers = list(confusion_matrix.keys())

            if headers:
                lines.extend(
                    [
                        "### Category Confusion Matrix",
                        "",
                        "| Ground truth / predicted | "
                        + " | ".join(_markdown_cell(item) for item in headers)
                        + " |",
                        "| " + " | ".join(["---"] * (len(headers) + 1)) + " |",
                    ]
                )

                for actual_category, row in confusion_matrix.items():
                    values = [
                        str(row.get(predicted_category, 0))
                        for predicted_category in headers
                    ]
                    lines.append(
                        f"| **{_markdown_cell(actual_category)}** | "
                        + " | ".join(values)
                        + " |"
                    )
        else:
            missing = eval_metrics.get("missing_ticket_ids", [])
            lines.extend(
                [
                    "",
                    "## Held-Out Evaluation",
                    "",
                    "**Status:** Incomplete; accuracy is not reported.",
                    f"- Matched held-out tickets: "
                    f"{eval_metrics.get('matched_heldout_sample_size', 0)} "
                    f"of {eval_metrics.get('expected_heldout_sample_size', 0)}",
                    f"- Missing ticket IDs: {', '.join(missing) if missing else 'None'}",
                    "",
                ]
            )

    lines.extend(
        [
            "",
            "## Itemized Triage Results",
            "",
            "| Ticket ID | Category | Urgency | Action | Confidence | Reasoning | Retrieved IDs |",
            "| --- | --- | --- | --- | ---: | --- | --- |",
        ]
    )

    for result in results:
        retrieved_ids = (
            ", ".join(result.retrieved_ticket_ids)
            if result.retrieved_ticket_ids
            else "None"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(result.ticket_id),
                    _markdown_cell(_enum_value(result.category)),
                    _markdown_cell(_enum_value(result.urgency)),
                    _markdown_cell(_enum_value(result.action)),
                    f"{result.confidence:.2f}",
                    _markdown_cell(result.reasoning),
                    _markdown_cell(retrieved_ids),
                ]
            )
            + " |"
        )

    # The IDs are already in retrieval rank order. Look up each historical
    # ticket to show the evidence and resolution behind the triage decision.
    lines.extend(
        [
            "",
            "## Retrieved Historical Evidence",
            "",
            "Matches are listed in retrieval rank order for each incoming ticket.",
            "",
        ]
    )

    for result in results:
        lines.append(f"### Incoming ticket {_markdown_cell(result.ticket_id)}")
        lines.append("")

        if not result.retrieved_ticket_ids:
            lines.append("- No historical tickets were retrieved.")
            lines.append("")
            continue

        for rank, retrieved_id in enumerate(result.retrieved_ticket_ids, start=1):
            historical = historical_by_id.get(str(retrieved_id))

            if historical is None:
                lines.append(
                    f"{rank}. **{_markdown_cell(retrieved_id)}** — "
                    "historical record not found."
                )
                continue

            category = historical.get("category", "N/A")
            urgency = historical.get("urgency", "N/A")
            resolution = historical.get("resolution", "Resolution unavailable.")

            lines.append(
                f"{rank}. **{_markdown_cell(retrieved_id)}** "
                f"(Category: {_markdown_cell(category)}; "
                f"Urgency: {_markdown_cell(urgency)})"
            )
            lines.append(
                f"   - **Resolution:** {_markdown_cell(resolution)}"
            )

        lines.append("")

    report_content = "\n".join(lines) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write(report_content)

    print(f"[Reporting] Report generated: {output_path}")
    return report_content