"""Evaluate held-out predictions against gold labels."""

import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from .config import HELDOUT_GOLD_LABELS_PATH, OUTPUTS_DIR
from .schemas import (
    TicketCategory,
    TriageAction,
    TriageOutput,
    UrgencyLevel,
)


class GoldLabel(BaseModel):
    """Validated gold label for one held-out ticket."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticket_id: str = Field(min_length=1)
    category: TicketCategory
    urgency: UrgencyLevel
    action: TriageAction


class Evaluator:
    """Score held-out predictions and build a category confusion matrix."""

    def __init__(self, gold_labels_path: Path = HELDOUT_GOLD_LABELS_PATH):
        self.gold_labels = self._load_and_validate_gold_labels(gold_labels_path)
        self.categories = [category.value for category in TicketCategory]

    @staticmethod
    def _load_and_validate_gold_labels(path: Path) -> Dict[str, GoldLabel]:
        if not path.exists():
            raise FileNotFoundError(f"Gold label file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            raw_labels = json.load(file)

        if not isinstance(raw_labels, list):
            raise ValueError("Gold labels must be a JSON list.")

        gold_map: Dict[str, GoldLabel] = {}

        for index, item in enumerate(raw_labels):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Gold label at index {index} must be a JSON object."
                )

            label = GoldLabel.model_validate(item)

            if label.ticket_id in gold_map:
                raise ValueError(
                    f"Duplicate gold label ID: {label.ticket_id}"
                )

            gold_map[label.ticket_id] = label

        if not gold_map:
            raise ValueError("The gold label file contains no labels.")

        return gold_map

    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUTS_DIR / "evaluation.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

    def evaluate_predictions(
        self,
        predictions: List[TriageOutput],
    ) -> Dict[str, Any]:
        """Evaluate predictions for every ticket represented in the gold file."""

        predictions_by_id: Dict[str, TriageOutput] = {}
        duplicate_ids = set()

        for prediction in predictions:
            ticket_id = prediction.ticket_id
            if ticket_id in predictions_by_id:
                duplicate_ids.add(ticket_id)
            else:
                predictions_by_id[ticket_id] = prediction

        heldout_ids = set(self.gold_labels)
        predicted_heldout_ids = heldout_ids.intersection(predictions_by_id)
        missing_ids = sorted(heldout_ids - predicted_heldout_ids)

        # The input batch may also contain the ten Day 1 tickets.
        non_heldout_count = sum(
            ticket_id not in heldout_ids for ticket_id in predictions_by_id
        )

        if duplicate_ids or missing_ids:
            metrics = {
                "status": "incomplete",
                "expected_heldout_sample_size": len(heldout_ids),
                "matched_heldout_sample_size": len(predicted_heldout_ids),
                "missing_ticket_ids": missing_ids,
                "duplicate_prediction_ids": sorted(duplicate_ids),
                "non_heldout_predictions_ignored": non_heldout_count,
            }
            self._save_metrics(metrics)
            return metrics

        confusion_matrix = {
            actual: {predicted: 0 for predicted in self.categories}
            for actual in self.categories
        }

        category_correct = 0
        urgency_correct = 0
        action_correct = 0
        exact_match_correct = 0

        for ticket_id, gold in self.gold_labels.items():
            prediction = predictions_by_id[ticket_id]

            predicted_category = prediction.category.value
            predicted_urgency = prediction.urgency.value
            predicted_action = prediction.action.value

            gold_category = gold.category.value
            gold_urgency = gold.urgency.value
            gold_action = gold.action.value

            category_correct += predicted_category == gold_category
            urgency_correct += predicted_urgency == gold_urgency
            action_correct += predicted_action == gold_action

            confusion_matrix[gold_category][predicted_category] += 1

            if (
                predicted_category == gold_category
                and predicted_urgency == gold_urgency
                and predicted_action == gold_action
            ):
                exact_match_correct += 1

        total = len(self.gold_labels)
        metrics = {
            "status": "complete",
            "heldout_sample_size": total,
            "non_heldout_predictions_ignored": non_heldout_count,
            "accuracies": {
                "category_accuracy": round(category_correct / total, 4),
                "urgency_accuracy": round(urgency_correct / total, 4),
                "action_accuracy": round(action_correct / total, 4),
                "overall_exact_match": round(exact_match_correct / total, 4),
            },
            "confusion_matrix": confusion_matrix,
        }

        self._save_metrics(metrics)
        return metrics