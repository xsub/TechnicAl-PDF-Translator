from __future__ import annotations

from translator.nodes import (
    apply_operator_decisions,
    extract_pdf,
    prepare_segments,
    render_pdf,
    resolve_findings,
    review_translation,
    revise_flagged_segments,
    translate_segments,
    validate_invariants,
    verify_output,
)
from translator.state import TranslationState


def build_graph():
    """Build the LangGraph workflow when langgraph is installed.

    The project can run without LangGraph in tests and demos via
    `translator.workflow.run_mvp_pipeline`, but this function keeps the MVP
    ready for checkpointed graph execution.
    """

    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Brakuje langgraph. Uruchom: python -m pip install -e .") from exc

    builder = StateGraph(TranslationState)
    builder.add_node("extract_pdf", extract_pdf)
    builder.add_node("prepare_segments", prepare_segments)
    builder.add_node("translate_segments", translate_segments)
    builder.add_node("validate_invariants", validate_invariants)
    builder.add_node("review_translation", review_translation)
    builder.add_node("resolve_findings", resolve_findings)
    builder.add_node("revise_flagged_segments", revise_flagged_segments)
    builder.add_node("human_review", _human_review_interrupt)
    builder.add_node("render_pdf", render_pdf)
    builder.add_node("verify_output", verify_output)

    builder.add_edge(START, "extract_pdf")
    builder.add_edge("extract_pdf", "prepare_segments")
    builder.add_edge("prepare_segments", "translate_segments")
    builder.add_edge("translate_segments", "validate_invariants")
    builder.add_edge("validate_invariants", "review_translation")
    builder.add_edge("review_translation", "resolve_findings")
    builder.add_conditional_edges(
        "resolve_findings",
        _route_after_resolve,
        {
            "revise_flagged_segments": "revise_flagged_segments",
            "human_review": "human_review",
            "render_pdf": "render_pdf",
        },
    )
    builder.add_edge("revise_flagged_segments", "validate_invariants")
    builder.add_edge("human_review", "render_pdf")
    builder.add_edge("render_pdf", "verify_output")
    builder.add_edge("verify_output", END)
    return builder.compile(checkpointer=InMemorySaver())


def _route_after_resolve(state: TranslationState) -> str:
    if state.get("revision_required_segments"):
        return "revise_flagged_segments"
    if state.get("unresolved_segments") and state["config"].require_human_review:
        return "human_review"
    return "render_pdf"


def _human_review_interrupt(state: TranslationState) -> TranslationState:
    if state.get("unresolved_segments") and not state.get("operator_decisions"):
        try:
            from langgraph.types import interrupt
        except ImportError as exc:
            raise RuntimeError("Brakuje langgraph.types.interrupt.") from exc

        resume_value = interrupt(
            {
                "kind": "operator_decisions_required",
                "unresolved_segments": state.get("unresolved_segments", []),
                "message": "Provide decisions keyed by segment_id: {action: accept|edit|keep_source, text?: string}.",
            }
        )
        if isinstance(resume_value, dict):
            state = {**state, "operator_decisions": resume_value}

    return apply_operator_decisions(state)
