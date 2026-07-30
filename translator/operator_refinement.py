from __future__ import annotations

from typing import Iterable


def clean_operator_phrase(text: object, *, limit: int = 400) -> str:
    phrase = " ".join(str(text or "").split())
    if not phrase or len(phrase) > limit:
        return ""
    return phrase


def suggest_operator_replacement_text(current_translation: object) -> str:
    """Suggest the current translated text as the fragment to replace."""

    return clean_operator_phrase(current_translation, limit=600)


def suggest_operator_preferred_phrase(issues: Iterable[object], current_translation: object) -> str:
    """Suggest the better phrase/proposal for the operator refinement form.

    Review providers are not always perfectly consistent: the best replacement
    may arrive as `proposed_translation`, but some reviewers put it in
    `translation_evidence`. The UI should still keep the semantics of the form:
    the left field is the current text to replace, the right field is the better
    target-language phrase to try.
    """

    current = clean_operator_phrase(current_translation, limit=1000)
    current_normalized = current.lower()

    for issue in issues:
        proposal = clean_operator_phrase(getattr(issue, "proposed_translation", ""), limit=600)
        if proposal and proposal.lower() != current_normalized:
            return proposal

    for issue in issues:
        evidence = clean_operator_phrase(getattr(issue, "translation_evidence", ""), limit=600)
        evidence_normalized = evidence.lower()
        if (
            evidence
            and evidence_normalized != current_normalized
            and evidence_normalized not in current_normalized
            and current_normalized not in evidence_normalized
        ):
            return evidence

    return ""
