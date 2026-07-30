from __future__ import annotations

from typing import Any


SAVEABLE_OPERATOR_ACTIONS = {"accept", "edit", "keep_source"}


def normalize_decision_text(text: object) -> str:
    return " ".join(str(text or "").split())


def build_selected_operator_decision(
    *,
    action: object,
    edited_text: object,
    original_text: object,
    note: object = None,
) -> dict[str, Any] | None:
    """Return an operator decision only when the user actually selected one.

    In the Streamlit review UI we render several segments at once. Rendering a
    segment must not mean accepting it. A segment becomes selected when:
    - the operator explicitly picks accept/edit/keep_source,
    - the approved text differs from the current translation,
    - or a phrase-refinement note exists.
    """

    action_text = str(action or "skip")
    text = str(edited_text or "")
    note_text = str(note or "").strip()
    text_changed = normalize_decision_text(text) != normalize_decision_text(original_text)

    if action_text not in SAVEABLE_OPERATOR_ACTIONS and not text_changed and not note_text:
        return None

    if text_changed and action_text != "keep_source":
        final_action = "edit"
    else:
        final_action = action_text if action_text in SAVEABLE_OPERATOR_ACTIONS else "edit"
    decision: dict[str, Any] = {
        "action": final_action,
        "text": text,
    }
    if note_text:
        decision["note"] = note_text
    return decision
