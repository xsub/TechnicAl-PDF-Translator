from __future__ import annotations

from pathlib import Path
from typing import Mapping

from translator.debug import DebugTimer, configure_debug_logging, log_debug
from translator.nodes import (
    apply_operator_decisions,
    extract_pdf,
    prepare_segments,
    render_pdf,
    resolve_findings,
    review_translation,
    revise_flagged_segments,
    translate_and_review_segments,
    validate_invariants,
    verify_output,
)
from translator.progress import CheckpointCallback, ProgressCallback, emit_progress
from translator.schemas import JobConfig, OperatorDecision
from translator.state import TranslationState
from translator.storage import JobStore
from translator.translation_cache import TranslationCache, build_translation_cache_scope
from translator.utils import new_job_id


def run_mvp_pipeline(
    config: JobConfig,
    progress_callback: ProgressCallback | None = None,
) -> TranslationState:
    state = _initial_state(config)
    configure_debug_logging(config.debug, job_id=state["job_id"], output_dir=config.output_dir)
    log_debug(
        "workflow.start",
        job_id=state["job_id"],
        source_pdf_path=config.source_pdf_path,
        output_dir=config.output_dir,
        mode=config.mode,
        translator_provider=config.translator_provider,
        reviewer_provider=config.reviewer_provider,
        require_human_review=config.require_human_review,
        max_revision_attempts=config.max_revision_attempts,
    )
    store = _store_for_config(config)
    _prepare_translation_cache(config, state)
    checkpoint_callback = _checkpoint_callback(store)

    emit_progress(progress_callback, stage="extract", message="Ekstrahuję tekst i tabele z PDF-a")
    with DebugTimer("node.extract_pdf", job_id=state["job_id"]):
        state = extract_pdf(state)
    store.save_state(state)
    emit_progress(
        progress_callback,
        stage="extract",
        message=f"Ekstrakcja zakończona: {len(state.get('segments', []))} segmentów",
    )

    with DebugTimer("node.prepare_segments", job_id=state["job_id"], segments=len(state.get("segments", []))):
        state = prepare_segments(state, progress_callback)
    store.save_state(state)

    with DebugTimer("node.translate_and_review_segments", job_id=state["job_id"], segments=len(state.get("segments", []))):
        state = translate_and_review_segments(state, progress_callback, checkpoint_callback)
    store.save_state(state)

    state = _validate_review_resolve_loop(
        state,
        store,
        progress_callback,
        checkpoint_callback,
        skip_initial_review_if_complete=True,
    )

    if state.get("unresolved_segments") and config.require_human_review:
        state = {**state, "status": "needs_human_review"}
        store.save_state(state)
        emit_progress(
            progress_callback,
            stage="human_review",
            message=f"Wymagana decyzja operatora: {len(state.get('unresolved_segments', []))} segmentów",
        )
        log_debug(
            "workflow.stop.needs_human_review",
            job_id=state["job_id"],
            unresolved_segments=len(state.get("unresolved_segments", [])),
        )
        return state

    if state.get("unresolved_segments") and not config.require_human_review:
        state = {**state, "status": "rendering_with_unresolved_warnings"}

    state = _render_verify_report(state, store, progress_callback)
    log_debug("workflow.done", job_id=state["job_id"], status=state.get("status"))
    return state


def resume_mvp_pipeline(
    state: TranslationState,
    progress_callback: ProgressCallback | None = None,
) -> TranslationState:
    config = state["config"]
    configure_debug_logging(config.debug, job_id=state["job_id"], output_dir=config.output_dir)
    log_debug(
        "workflow.resume.start",
        job_id=state["job_id"],
        status=state.get("status"),
        segments=len(state.get("segments", [])),
        translations=len(state.get("translations", {})),
        review_results=len(state.get("review_results", {})),
    )
    store = _store_for_config(config)
    _prepare_translation_cache(config, state)
    checkpoint_callback = _checkpoint_callback(store)

    if not state.get("segments"):
        emit_progress(progress_callback, stage="extract", message="Wznawiam: ekstrahuję PDF")
        with DebugTimer("node.extract_pdf", job_id=state["job_id"]):
            state = extract_pdf(state)
        store.save_state(state)

    if state.get("segments") and not any(segment.protected_tokens for segment in state.get("segments", [])):
        emit_progress(progress_callback, stage="prepare", message="Wznawiam: przygotowuję segmenty")
        with DebugTimer("node.prepare_segments", job_id=state["job_id"], segments=len(state.get("segments", []))):
            state = prepare_segments(state, progress_callback)
        store.save_state(state)

    if _needs_translate_review_pipeline(state):
        emit_progress(
            progress_callback,
            stage="pipeline",
            message=(
                "Wznawiam pipeline tłumaczenie→recenzja "
                f"({len(state.get('translations', {}))}/{len(state.get('segments', []))} tłumaczeń, "
                f"{len(state.get('review_results', {}))}/{len(state.get('segments', []))} recenzji)"
            ),
        )
        with DebugTimer("node.translate_and_review_segments.resume", job_id=state["job_id"], segments=len(state.get("segments", []))):
            state = translate_and_review_segments(state, progress_callback, checkpoint_callback)
        store.save_state(state)

    state = _validate_review_resolve_loop(
        state,
        store,
        progress_callback,
        checkpoint_callback,
        resume_existing_reviews=True,
        skip_initial_review_if_complete=True,
    )

    if state.get("unresolved_segments") and config.require_human_review:
        state = {**state, "status": "needs_human_review"}
        store.save_state(state)
        emit_progress(
            progress_callback,
            stage="human_review",
            message=f"Wymagana decyzja operatora: {len(state.get('unresolved_segments', []))} segmentów",
        )
        return state

    if state.get("unresolved_segments") and not config.require_human_review:
        state = {**state, "status": "rendering_with_unresolved_warnings"}

    state = _render_verify_report(state, store, progress_callback)
    log_debug("workflow.resume.done", job_id=state["job_id"], status=state.get("status"))
    return state


def finalize_with_operator_decisions(
    state: TranslationState,
    decisions: Mapping[str, dict | OperatorDecision],
    progress_callback: ProgressCallback | None = None,
) -> TranslationState:
    configure_debug_logging(state["config"].debug, job_id=state["job_id"], output_dir=state["config"].output_dir)
    log_debug(
        "workflow.finalize.start",
        job_id=state["job_id"],
        decisions=len(decisions),
    )
    parsed_decisions = {
        segment_id: decision if isinstance(decision, OperatorDecision) else OperatorDecision(segment_id=segment_id, **decision)
        for segment_id, decision in decisions.items()
    }
    store = _store_for_config(state["config"])
    state = {**state, "operator_decisions": parsed_decisions}
    state = apply_operator_decisions(state)
    store.save_state(state)

    if state.get("unresolved_segments"):
        state = {**state, "status": "needs_human_review"}
        store.save_state(state)
        log_debug(
            "workflow.finalize.needs_more_human_review",
            job_id=state["job_id"],
            unresolved_segments=len(state.get("unresolved_segments", [])),
        )
        return state

    return _render_verify_report(state, store, progress_callback)


def _initial_state(config: JobConfig) -> TranslationState:
    job_id = config.job_id or new_job_id()
    config = config.model_copy(update={"job_id": job_id})
    return {
        "job_id": job_id,
        "config": config,
        "source_pdf_path": config.source_pdf_path,
        "source_language": config.source_language,
        "target_language": config.target_language,
        "document_metadata": {},
        "segments": [],
        "translations": {},
        "deterministic_issues": [],
        "review_results": {},
        "unresolved_segments": [],
        "revision_required_segments": [],
        "revision_attempts": {},
        "translation_memory_hits": 0,
        "persistent_translation_cache_hits": 0,
        "translation_memory_misses": 0,
        "translation_cache_scope": build_translation_cache_scope(config),
        "llm_inflight": None,
        "operator_decisions": {},
        "output_pdf_path": None,
        "output_verification": None,
        "report_path": None,
        "status": "created",
    }


def _validate_review_resolve_loop(
    state: TranslationState,
    store: JobStore,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    *,
    resume_existing_reviews: bool = False,
    skip_initial_review_if_complete: bool = False,
) -> TranslationState:
    max_cycles = state["config"].max_revision_attempts + 1
    for cycle in range(1, max_cycles + 1):
        log_debug("workflow.review_cycle.start", job_id=state["job_id"], cycle=cycle, max_cycles=max_cycles)
        emit_progress(progress_callback, stage="validate", message=f"Waliduję tłumaczenia - cykl {cycle}/{max_cycles}")
        with DebugTimer("node.validate_invariants", job_id=state["job_id"], cycle=cycle):
            state = validate_invariants(state, progress_callback)
        store.save_state(state)

        if skip_initial_review_if_complete and cycle == 1 and _reviews_complete(state):
            emit_progress(
                progress_callback,
                stage="review",
                message=f"Recenzja już gotowa z pipeline - cykl {cycle}/{max_cycles}",
            )
        else:
            emit_progress(progress_callback, stage="review", message=f"Uruchamiam recenzję - cykl {cycle}/{max_cycles}")
            with DebugTimer("node.review_translation", job_id=state["job_id"], cycle=cycle):
                state = review_translation(
                    state,
                    progress_callback,
                    checkpoint_callback,
                    skip_existing=resume_existing_reviews and cycle == 1,
                )
            store.save_state(state)

        emit_progress(progress_callback, stage="resolve", message="Rozstrzygam problemy i routing")
        with DebugTimer("node.resolve_findings", job_id=state["job_id"], cycle=cycle):
            state = resolve_findings(state)
        store.save_state(state)
        log_debug(
            "workflow.review_cycle.resolved",
            job_id=state["job_id"],
            cycle=cycle,
            deterministic_issues=len(state.get("deterministic_issues", [])),
            review_results=len(state.get("review_results", {})),
            unresolved_segments=len(state.get("unresolved_segments", [])),
            revision_required_segments=len(state.get("revision_required_segments", [])),
        )

        if not state.get("revision_required_segments"):
            break

        with DebugTimer("node.revise_flagged_segments", job_id=state["job_id"], cycle=cycle):
            state = revise_flagged_segments(state, progress_callback, checkpoint_callback)
        store.save_state(state)

    return state


def _render_verify_report(
    state: TranslationState,
    store: JobStore,
    progress_callback: ProgressCallback | None = None,
) -> TranslationState:
    emit_progress(progress_callback, stage="render", message="Generuję PDF wynikowy")
    with DebugTimer("node.render_pdf", job_id=state["job_id"]):
        state = render_pdf(state)
    store.save_state(state)

    emit_progress(progress_callback, stage="verify", message="Weryfikuję PDF wynikowy")
    with DebugTimer("node.verify_output", job_id=state["job_id"]):
        state = verify_output(state)
    store.save_state(state)

    emit_progress(progress_callback, stage="report", message="Zapisuję raport audytowy")
    report_path = store.write_report(state)
    state = {**state, "report_path": str(report_path)}
    store.save_state(state)
    emit_progress(progress_callback, stage="done", message="Workflow zakończony")
    return state


def _needs_translate_review_pipeline(state: TranslationState) -> bool:
    segments_count = len(state.get("segments", []))
    if not segments_count:
        return False

    translations_count = len(state.get("translations", {}))
    review_results_count = len(state.get("review_results", {}))
    return translations_count < segments_count or review_results_count < translations_count


def _reviews_complete(state: TranslationState) -> bool:
    translations_count = len(state.get("translations", {}))
    if translations_count == 0:
        return False
    return len(state.get("review_results", {})) >= translations_count


def _store_for_config(config: JobConfig) -> JobStore:
    output_parent = Path(config.output_dir).parent
    return JobStore(output_parent / "jobs.db")


def _prepare_translation_cache(config: JobConfig, state: TranslationState) -> None:
    cache = TranslationCache.for_config(config)
    with DebugTimer("translation_cache.prepare", job_id=state["job_id"], db_path=str(cache.db_path)):
        cache.backfill_from_jobs()
        cache.seed_from_state(state)


def _checkpoint_callback(store: JobStore) -> CheckpointCallback:
    def checkpoint(partial_state: dict) -> None:
        store.save_state(partial_state)  # type: ignore[arg-type]

    return checkpoint
