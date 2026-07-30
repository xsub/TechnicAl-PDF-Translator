from __future__ import annotations

from translator.schemas import OperatorDecision
from translator.state import TranslationState


def apply_operator_decisions(state: TranslationState) -> TranslationState:
    raw_decisions = state.get("operator_decisions", {})
    decisions = {
        segment_id: decision if isinstance(decision, OperatorDecision) else OperatorDecision(segment_id=segment_id, **decision)
        for segment_id, decision in raw_decisions.items()
    }
    translations = dict(state.get("translations", {}))
    segments = {segment.segment_id: segment for segment in state.get("segments", [])}
    unresolved = set(state.get("unresolved_segments", []))

    for segment_id, decision in decisions.items():
        if segment_id not in translations or segment_id not in unresolved:
            continue

        decision_note = decision.note or ""
        if decision.action == "keep_source":
            translations[segment_id] = translations[segment_id].model_copy(
                update={
                    "translated_text": segments[segment_id].source_text,
                    "translator_notes": [
                        *translations[segment_id].translator_notes,
                        "Operator kept source text.",
                        *([decision_note] if decision_note else []),
                    ],
                    "confidence": "medium",
                }
            )
        elif decision.action == "edit" and decision.text is not None:
            translations[segment_id] = translations[segment_id].model_copy(
                update={
                    "translated_text": decision.text,
                    "translator_notes": [
                        *translations[segment_id].translator_notes,
                        "Operator edited translation.",
                        *([decision_note] if decision_note else []),
                    ],
                    "confidence": "high",
                }
            )
        elif decision.action == "accept":
            translations[segment_id] = translations[segment_id].model_copy(
                update={
                    "translator_notes": [
                        *translations[segment_id].translator_notes,
                        "Operator accepted translation.",
                        *([decision_note] if decision_note else []),
                    ],
                    "confidence": "high",
                }
            )

        unresolved.discard(segment_id)

    return {
        **state,
        "translations": translations,
        "unresolved_segments": sorted(unresolved),
        "status": "operator_decisions_applied",
    }
