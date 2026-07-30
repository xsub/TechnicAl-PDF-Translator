from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time

from translator.debug import log_debug, text_preview
from translator.domain.glossary import load_glossary
from translator.llm.clients import (
    build_translator,
    estimate_revision_request_tokens,
    estimate_translation_request_tokens,
)
from translator.progress import CheckpointCallback, ProgressCallback, emit_progress
from translator.schemas import DocumentSegment, ReviewFinding, TranslationResult
from translator.state import TranslationState
from translator.translation_cache import TranslationCache, copy_translation_from_cache, normalize_translation_source


TranslationMemory = dict[str, tuple[str, TranslationResult]]


@dataclass
class PendingTranslationGroup:
    segment: DocumentSegment
    duplicates: list[DocumentSegment]
    estimated_input_tokens: int = 0


def translate_segments(
    state: TranslationState,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> TranslationState:
    config = state["config"]
    glossary = load_glossary(config.glossary_path)
    translation_cache = TranslationCache.for_config(config)
    client = None
    translations = dict(state.get("translations", {}))
    segments = state.get("segments", [])
    total = len(segments)
    memory = _build_translation_memory(segments, translations)
    memory_hits = int(state.get("translation_memory_hits", 0))
    persistent_cache_hits = int(state.get("persistent_translation_cache_hits", 0))
    memory_misses = int(state.get("translation_memory_misses", 0))
    cache_scope = state.get("translation_cache_scope") or translation_cache.scope

    pending_by_key: dict[str, PendingTranslationGroup] = {}
    pending_groups: list[PendingTranslationGroup] = []

    for index, segment in enumerate(segments, start=1):
        if segment.segment_id in translations:
            emit_progress(
                progress_callback,
                stage="translate",
                message="Pomijam już przetłumaczony segment",
                current=index,
                total=total,
                segment_id=segment.segment_id,
            )
            continue
        started_at = time.perf_counter()
        memory_key = normalize_translation_source(segment.source_text)
        log_debug(
            "segment.translate.start",
            segment_id=segment.segment_id,
            index=index,
            total=total,
            page_number=segment.page_number,
            block_type=segment.block_type,
            source_chars=len(segment.source_text),
            protected_tokens=len(segment.protected_tokens),
            source_preview=text_preview(segment.source_text),
        )
        if memory_key and memory_key in memory:
            source_segment_id, cached_translation = memory[memory_key]
            translations[segment.segment_id] = copy_translation_from_cache(
                segment,
                cached_translation,
                source_label=f"segment {source_segment_id}",
            )
            translation = translations[segment.segment_id]
            memory_hits += 1
            partial_state = {
                **state,
                "translations": translations,
                "translation_memory_hits": memory_hits,
                "persistent_translation_cache_hits": persistent_cache_hits,
                "translation_memory_misses": memory_misses,
                "translation_cache_scope": cache_scope,
                "llm_inflight": None,
                "status": f"translating {index}/{total}",
            }
            if checkpoint_callback:
                checkpoint_callback(partial_state)
            log_debug(
                "segment.translate.memory_hit",
                segment_id=segment.segment_id,
                source_segment_id=source_segment_id,
                index=index,
                total=total,
                duration_s=round(time.perf_counter() - started_at, 3),
                translated_chars=len(translation.translated_text),
                translated_preview=text_preview(translation.translated_text),
            )
            emit_progress(
                progress_callback,
                stage="translate",
                message="Używam zapisanego tłumaczenia identycznego segmentu",
                current=index,
                total=total,
                segment_id=segment.segment_id,
            )
            continue

        cached_translation = translation_cache.lookup(segment)
        if cached_translation:
            translations[segment.segment_id] = cached_translation
            memory.setdefault(memory_key, (segment.segment_id, cached_translation))
            persistent_cache_hits += 1
            partial_state = {
                **state,
                "translations": translations,
                "translation_memory_hits": memory_hits,
                "persistent_translation_cache_hits": persistent_cache_hits,
                "translation_memory_misses": memory_misses,
                "translation_cache_scope": cache_scope,
                "llm_inflight": None,
                "status": f"translating {index}/{total}",
            }
            if checkpoint_callback:
                checkpoint_callback(partial_state)
            log_debug(
                "segment.translate.persistent_cache_hit",
                segment_id=segment.segment_id,
                index=index,
                total=total,
                duration_s=round(time.perf_counter() - started_at, 3),
                translated_chars=len(cached_translation.translated_text),
                translated_preview=text_preview(cached_translation.translated_text),
            )
            emit_progress(
                progress_callback,
                stage="translate",
                message="Używam trwałego cache tłumaczenia",
                current=index,
                total=total,
                segment_id=segment.segment_id,
            )
            continue

        group_key = memory_key or f"segment:{segment.segment_id}"
        if group_key in pending_by_key:
            pending_by_key[group_key].duplicates.append(segment)
            continue

        group = PendingTranslationGroup(segment=segment, duplicates=[])
        pending_by_key[group_key] = group
        pending_groups.append(group)

    if pending_groups:
        if client is None:
            client = build_translator(config)
        concurrency = _concurrency_limit(config.translation_concurrency, len(pending_groups))
        model_name = getattr(client, "model_name", config.translator_provider)
        log_debug(
            "segments.translate.parallel.start",
            pending_groups=len(pending_groups),
            concurrency=concurrency,
            provider=config.translator_provider,
            model=model_name,
        )

        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="translate") as executor:
            pending_iter = iter(pending_groups)
            active: dict[Future[TranslationResult], PendingTranslationGroup] = {}

            def submit_next() -> bool:
                try:
                    group = next(pending_iter)
                except StopIteration:
                    return False

                group.estimated_input_tokens = (
                    estimate_translation_request_tokens(group.segment, glossary, config)
                    if config.translator_provider != "mock"
                    else 0
                )
                log_debug(
                    "segment.translate.request_start",
                    segment_id=group.segment.segment_id,
                    total=total,
                    page_number=group.segment.page_number,
                    block_type=group.segment.block_type,
                    source_chars=len(group.segment.source_text),
                    protected_tokens=len(group.segment.protected_tokens),
                    duplicate_segments=len(group.duplicates),
                    source_preview=text_preview(group.segment.source_text),
                )
                future = executor.submit(client.translate, group.segment, glossary, config)
                active[future] = group
                return True

            for _ in range(concurrency):
                if not submit_next():
                    break

            if checkpoint_callback:
                checkpoint_callback(
                    _partial_translation_state(
                        state,
                        translations,
                        memory_hits,
                        persistent_cache_hits,
                        memory_misses,
                        cache_scope,
                        _inflight_state(
                            active.values(),
                            operation="translate",
                            provider=config.translator_provider,
                            model=model_name,
                        ),
                        total,
                    )
                )
            emit_progress(
                progress_callback,
                stage="translate",
                message="Tłumaczę segmenty równolegle",
                current=len(translations),
                total=total,
            )

            while active:
                for future in as_completed(list(active.keys())):
                    group = active.pop(future)
                    segment = group.segment
                    try:
                        translation = future.result()
                    except Exception:
                        if checkpoint_callback:
                            checkpoint_callback(
                                _partial_translation_state(
                                    state,
                                    translations,
                                    memory_hits,
                                    persistent_cache_hits,
                                    memory_misses,
                                    cache_scope,
                                    None,
                                    total,
                                )
                            )
                        raise
                    translations[segment.segment_id] = translation
                    memory_misses += 1

                    memory_key = normalize_translation_source(segment.source_text)
                    if memory_key:
                        memory.setdefault(memory_key, (segment.segment_id, translation))
                    translation_cache.store(segment, translation, job_id=state["job_id"])

                    for duplicate in group.duplicates:
                        translations[duplicate.segment_id] = copy_translation_from_cache(
                            duplicate,
                            translation,
                            source_label=f"segment {segment.segment_id}",
                        )
                        memory_hits += 1

                    submit_next()

                    partial_state = _partial_translation_state(
                        state,
                        translations,
                        memory_hits,
                        persistent_cache_hits,
                        memory_misses,
                        cache_scope,
                        _inflight_state(
                            active.values(),
                            operation="translate",
                            provider=config.translator_provider,
                            model=model_name,
                        ),
                        total,
                    )
                    if checkpoint_callback:
                        checkpoint_callback(partial_state)
                    log_debug(
                        "segment.translate.done",
                        segment_id=segment.segment_id,
                        total=total,
                        translated_chars=len(translation.translated_text),
                        confidence=translation.confidence,
                        uncertain_terms=translation.uncertain_terms,
                        duplicate_segments=len(group.duplicates),
                        translated_preview=text_preview(translation.translated_text),
                    )
                    emit_progress(
                        progress_callback,
                        stage="translate",
                        message="Segment przetłumaczony",
                        current=len(translations),
                        total=total,
                        segment_id=segment.segment_id,
                    )
                    for duplicate in group.duplicates:
                        emit_progress(
                            progress_callback,
                            stage="translate",
                            message="Używam zapisanego tłumaczenia identycznego segmentu",
                            current=len(translations),
                            total=total,
                            segment_id=duplicate.segment_id,
                        )
                    break

        log_debug(
            "segments.translate.parallel.done",
            translations=len(translations),
            memory_hits=memory_hits,
            persistent_cache_hits=persistent_cache_hits,
            memory_misses=memory_misses,
        )

    return {
        **state,
        "translations": translations,
        "translation_memory_hits": memory_hits,
        "persistent_translation_cache_hits": persistent_cache_hits,
        "translation_memory_misses": memory_misses,
        "translation_cache_scope": cache_scope,
        "llm_inflight": None,
        "status": "segments_translated",
    }


def revise_flagged_segments(
    state: TranslationState,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> TranslationState:
    config = state["config"]
    glossary = load_glossary(config.glossary_path)
    client = build_translator(config)
    translations = dict(state.get("translations", {}))
    attempts = dict(state.get("revision_attempts", {}))
    review_results = state.get("review_results", {})
    segment_ids = state.get("revision_required_segments", [])
    total = len(segment_ids)

    for index, segment_id in enumerate(segment_ids, start=1):
        segment = next(segment for segment in state.get("segments", []) if segment.segment_id == segment_id)
        current = translations[segment_id]
        findings: list[ReviewFinding] = review_results.get(segment_id).findings if segment_id in review_results else []
        started_at = time.perf_counter()
        log_debug(
            "segment.revise.start",
            segment_id=segment_id,
            index=index,
            total=total,
            findings=len(findings),
            current_preview=text_preview(current.translated_text),
        )
        llm_inflight = None
        if config.translator_provider != "mock":
            llm_inflight = {
                "operation": "revise",
                "provider": config.translator_provider,
                "model": getattr(client, "model_name", config.translator_provider),
                "segment_id": segment_id,
                "index": index,
                "total": total,
                "estimated_input_tokens": estimate_revision_request_tokens(segment, current, findings, glossary, config),
            }
        if checkpoint_callback:
            checkpoint_callback(
                {
                    **state,
                    "translations": translations,
                    "revision_attempts": attempts,
                    "llm_inflight": llm_inflight,
                    "status": f"revising {index}/{total}",
                }
            )
        emit_progress(
            progress_callback,
            stage="revise",
            message="Poprawiam zakwestionowany segment",
            current=index,
            total=total,
            segment_id=segment_id,
        )
        translations[segment_id] = client.revise(segment, current, findings, glossary, config)
        attempts[segment_id] = attempts.get(segment_id, 0) + 1
        partial_state = {
            **state,
            "translations": translations,
            "revision_attempts": attempts,
            "llm_inflight": None,
            "status": f"revising {index}/{total}",
        }
        if checkpoint_callback:
            checkpoint_callback(partial_state)
        log_debug(
            "segment.revise.done",
            segment_id=segment_id,
            index=index,
            total=total,
            duration_s=round(time.perf_counter() - started_at, 3),
            translated_preview=text_preview(translations[segment_id].translated_text),
        )

    return {
        **state,
        "translations": translations,
        "revision_attempts": attempts,
        "revision_required_segments": [],
        "llm_inflight": None,
        "status": "flagged_segments_revised",
    }


def _build_translation_memory(
    segments: list[DocumentSegment],
    translations: dict[str, TranslationResult],
) -> TranslationMemory:
    memory: TranslationMemory = {}
    for segment in segments:
        translation = translations.get(segment.segment_id)
        if not translation:
            continue

        key = normalize_translation_source(segment.source_text)
        if key:
            memory.setdefault(key, (segment.segment_id, translation))

    return memory


def _partial_translation_state(
    state: TranslationState,
    translations: dict[str, TranslationResult],
    memory_hits: int,
    persistent_cache_hits: int,
    memory_misses: int,
    cache_scope: dict,
    llm_inflight: dict | None,
    total: int,
) -> dict:
    return {
        **state,
        "translations": translations,
        "translation_memory_hits": memory_hits,
        "persistent_translation_cache_hits": persistent_cache_hits,
        "translation_memory_misses": memory_misses,
        "translation_cache_scope": cache_scope,
        "llm_inflight": llm_inflight,
        "status": f"translating {len(translations)}/{total}",
    }


def _inflight_state(
    groups: object,
    *,
    operation: str,
    provider: str,
    model: str,
) -> dict | None:
    if provider == "mock":
        return None

    group_list = list(groups)  # type: ignore[arg-type]
    if not group_list:
        return None

    requests = [
        {
            "segment_id": group.segment.segment_id,
            "estimated_input_tokens": group.estimated_input_tokens,
        }
        for group in group_list
    ]
    return {
        "operation": operation,
        "provider": provider,
        "model": model,
        "active": len(requests),
        "segment_id": requests[0]["segment_id"],
        "segments": requests,
        "estimated_input_tokens": sum(int(request["estimated_input_tokens"] or 0) for request in requests),
    }


def _concurrency_limit(configured: int, total: int) -> int:
    if total <= 0:
        return 1
    return max(1, min(int(configured or 1), total))
