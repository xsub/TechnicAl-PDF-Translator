from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from translator.schemas import JobConfig
from translator.workflow import finalize_with_operator_decisions, run_mvp_pipeline


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Translate a technical PDF with audited MVP workflow.")
    parser.add_argument("pdf", help="Path to source PDF")
    parser.add_argument("--output-dir", default="storage/output")
    parser.add_argument("--mode", choices=["standard", "high_assurance", "strict_regulatory"], default="standard")
    parser.add_argument("--translator", choices=["mock", "openai"], default="mock")
    parser.add_argument("--reviewer", choices=["mock", "anthropic", "openai"], default="mock")
    parser.add_argument("--auto-accept-unresolved", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Print detailed workflow and LLM logs to the console.")
    args = parser.parse_args()

    config = JobConfig(
        source_pdf_path=str(Path(args.pdf)),
        output_dir=args.output_dir,
        mode=args.mode,
        translator_provider=args.translator,
        reviewer_provider=args.reviewer,
        require_human_review=not args.auto_accept_unresolved,
        debug=args.debug,
    )
    state = run_mvp_pipeline(config)

    if state.get("status") == "needs_human_review" and args.auto_accept_unresolved:
        decisions = {segment_id: {"action": "accept"} for segment_id in state.get("unresolved_segments", [])}
        state = finalize_with_operator_decisions(state, decisions)

    print(f"Status: {state.get('status')}")
    if state.get("output_pdf_path"):
        print(f"PDF: {state['output_pdf_path']}")
    if state.get("report_path"):
        print(f"Raport: {state['report_path']}")
    if state.get("unresolved_segments"):
        print(f"Segmenty do decyzji operatora: {', '.join(state['unresolved_segments'])}")


if __name__ == "__main__":
    main()
