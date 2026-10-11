"""
Main entrypoint for the Ticket Triage System.
Orchestrates batch triage execution, held-out gold label evaluation, and report generation.
"""

import sys
from pathlib import Path

# Dynamically add 'src' directory to Python path for reliable module resolution
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
from ticket_triage.config import (
    HISTORICAL_TICKETS_PATH,
    DAY1_TICKETS_PATH,
    HELDOUT_TICKETS_PATH,
    HELDOUT_GOLD_LABELS_PATH,
    OUTPUTS_DIR
)
from ticket_triage.batch import BatchProcessor
from ticket_triage.data_loader import (
    load_day1_tickets, 
    load_heldout_tickets
)
from ticket_triage.evaluation import Evaluator
from ticket_triage.reporting import generate_markdown_report


def verify_datasets_exist() -> bool:
    """Verifies that all required JSON dataset files exist before execution."""
    required_paths = [
        HISTORICAL_TICKETS_PATH,
        DAY1_TICKETS_PATH,
        HELDOUT_TICKETS_PATH,
        HELDOUT_GOLD_LABELS_PATH
    ]
    missing = [path.name for path in required_paths if not path.exists()]
    
    if missing:
        print(f"[Error] Missing required data files in data/: {', '.join(missing)}")
        print("Please run 'python generate_mock_data.py' first to create test datasets.")
        return False
    return True


def run_pipeline(rebuild_index: bool = False) -> None:
    """Runs the complete end-to-end triage, evaluation, and reporting pipeline."""
    if not verify_datasets_exist():
        sys.exit(1)

    print("\n" + "="*60)
    print("      SUPPORT TICKET TRIAGE & RAG DECISION ENGINE      ")
    print("="*60 + "\n")

    # Step 1: Load Input Tickets
    print("[1/4] Loading Day 1 and Held-out ticket batches...")
    day1 = load_day1_tickets()
    heldout = load_heldout_tickets()
    all_tickets = day1 + heldout
    print(f"      Loaded {len(day1)} Day 1 tickets and {len(heldout)} Held-out tickets ({len(all_tickets)} total).")

    # Step 2: Initialize RAG Batch Processor
    print("\n[2/4] Initializing Vector Store and Decision Engine...")
    processor = BatchProcessor(rebuild_index=rebuild_index)

    # Step 3: Run Batch Execution
    print(f"\n[3/4] Processing {len(all_tickets)} tickets through LLM decision layer...")
    results = processor.process_tickets(all_tickets)
    processor.save_results(results)

    # Step 4: Evaluate against Held-out Gold Labels & Generate Report
    print("\n[4/4] Evaluating results against gold labels & generating summary report...")
    evaluator = Evaluator()
    eval_metrics = evaluator.evaluate_predictions(results)

    report_path = OUTPUTS_DIR / "triage_report.md"
    generate_markdown_report(results, eval_metrics, output_path=report_path)

    # Output Summary Table to Console
    print("\n" + "="*60)
    print("                    RUN COMPLETE                       ")
    print("="*60)
    if "accuracies" in eval_metrics:
        acc = eval_metrics["accuracies"]
        print(f" Category Accuracy : {acc.get('category_accuracy', 0) * 100:.1f}%")
        print(f" Urgency Accuracy  : {acc.get('urgency_accuracy', 0) * 100:.1f}%")
        print(f" Action Accuracy   : {acc.get('action_accuracy', 0) * 100:.1f}%")
        print(f" Exact Match Rate  : {acc.get('overall_exact_match', 0) * 100:.1f}%")
    print(f"\n Detailed JSON output : outputs/triage_results.json")
    print(f" Evaluation Metrics   : outputs/evaluation.json")
    print(f" Markdown Report      : outputs/triage_report.md")
    print(f" Validation Log       : logs/validation_failures.log")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Support Ticket Triage CLI Runner")
    parser.add_argument(
        "--rebuild-index", 
        action="store_true", 
        help="Force rebuild of FAISS vector store index from historical tickets"
    )
    args = parser.parse_args()

    run_pipeline(rebuild_index=args.rebuild_index)