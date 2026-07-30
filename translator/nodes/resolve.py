from __future__ import annotations

from translator.debug import log_debug
from translator.state import TranslationState


def resolve_findings(state: TranslationState) -> TranslationState:
    config = state["config"]
    attempts = state.get("revision_attempts", {})
    unresolved: set[str] = set()
    revision_required: set[str] = set()

    for issue in state.get("deterministic_issues", []):
        if issue.severity == "critical":
            unresolved.add(issue.segment_id)

    for segment_id, result in state.get("review_results", {}).items():
        blocking_findings = [finding for finding in result.findings if finding.severity in {"critical", "major"}]
        if not blocking_findings:
            continue

        can_auto_revise = (
            attempts.get(segment_id, 0) < config.max_revision_attempts
            and all(finding.proposed_translation for finding in blocking_findings)
        )
        if can_auto_revise:
            revision_required.add(segment_id)
        else:
            unresolved.add(segment_id)

    if config.mode == "strict_regulatory":
        for segment in state.get("segments", []):
            if segment.block_type == "table_cell" or "regulation" in segment.source_text.lower():
                unresolved.add(segment.segment_id)

    log_debug(
        "findings.resolve.done",
        deterministic_issues=len(state.get("deterministic_issues", [])),
        review_results=len(state.get("review_results", {})),
        unresolved_segments=len(unresolved),
        revision_required_segments=len(revision_required),
        mode=config.mode,
    )
    return {
        **state,
        "unresolved_segments": sorted(unresolved),
        "revision_required_segments": sorted(revision_required),
        "status": "findings_resolved",
    }
