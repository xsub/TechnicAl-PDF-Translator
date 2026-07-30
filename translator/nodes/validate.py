from __future__ import annotations

from translator.debug import log_debug
from translator.domain.glossary import load_glossary
from translator.domain.protected import protect_segment, validate_segment_invariants
from translator.progress import ProgressCallback, emit_progress
from translator.state import TranslationState


def prepare_segments(
    state: TranslationState,
    progress_callback: ProgressCallback | None = None,
) -> TranslationState:
    source_segments = state.get("segments", [])
    total = len(source_segments)
    segments = []
    for index, segment in enumerate(source_segments, start=1):
        if index == 1 or index == total or index % 25 == 0:
            emit_progress(
                progress_callback,
                stage="prepare",
                message="Przygotowuję segmenty i chronione wartości",
                current=index,
                total=total,
                segment_id=segment.segment_id,
            )
        segments.append(protect_segment(segment))
    log_debug("segments.prepare.done", segments=len(segments))
    return {**state, "segments": segments, "status": "segments_prepared"}


def validate_invariants(
    state: TranslationState,
    progress_callback: ProgressCallback | None = None,
) -> TranslationState:
    config = state["config"]
    glossary = load_glossary(config.glossary_path)
    translations = state.get("translations", {})
    issues = []
    segments = state.get("segments", [])
    total = len(segments)

    for index, segment in enumerate(segments, start=1):
        translation = translations.get(segment.segment_id)
        if not translation:
            continue
        if index == 1 or index == total or index % 25 == 0:
            emit_progress(
                progress_callback,
                stage="validate",
                message="Sprawdzam liczby, jednostki i odnośniki",
                current=index,
                total=total,
                segment_id=segment.segment_id,
            )
        issues.extend(validate_segment_invariants(segment, translation))
        issues.extend(glossary.validate_translation(segment.segment_id, segment.source_text, translation.translated_text))

    log_debug(
        "segments.validate.done",
        segments=len(segments),
        translations=len(translations),
        issues=len(issues),
        critical_issues=sum(1 for issue in issues if issue.severity == "critical"),
    )
    return {**state, "deterministic_issues": issues, "status": "invariants_validated"}
