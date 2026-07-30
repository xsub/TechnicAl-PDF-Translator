from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from translator.debug import log_debug, text_preview
from translator.schemas import DocumentSegment, JobConfig, TranslationResult
from translator.utils import dumps_json, ensure_dir, normalize_ws, sha256_file


CACHE_SCHEMA_VERSION = "translation-cache-v1"


class TranslationCache:
    def __init__(self, db_path: str | Path, config: JobConfig) -> None:
        self.db_path = Path(db_path)
        self.config = config
        self.scope = build_translation_cache_scope(config)
        self.scope_json = dumps_json(self.scope)
        ensure_dir(self.db_path.parent)
        self._init_db()

    @classmethod
    def for_config(cls, config: JobConfig) -> "TranslationCache":
        return cls(Path(config.output_dir).parent / "jobs.db", config)

    def lookup(self, segment: DocumentSegment) -> TranslationResult | None:
        normalized_source = normalize_translation_source(segment.source_text)
        if not normalized_source:
            return None

        cache_key = translation_cache_key(self.scope, normalized_source)
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                """
                SELECT translation_json, origin_job_id, origin_segment_id
                FROM translation_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
            if row is None:
                log_debug(
                    "translation_cache.lookup.miss",
                    segment_id=segment.segment_id,
                    db_path=str(self.db_path),
                    source_hash=_source_hash(normalized_source),
                )
                return None

            with conn:
                conn.execute(
                    """
                    UPDATE translation_cache
                    SET hits = hits + 1,
                        last_used_at = ?
                    WHERE cache_key = ?
                    """,
                    (datetime.now(timezone.utc).isoformat(), cache_key),
                )

        cached = TranslationResult.model_validate(json.loads(row[0]))
        origin_job_id = row[1]
        origin_segment_id = row[2]
        result = copy_translation_from_cache(
            segment,
            cached,
            source_label=f"persistent cache job {origin_job_id} segment {origin_segment_id}",
        )
        log_debug(
            "translation_cache.lookup.hit",
            segment_id=segment.segment_id,
            origin_job_id=origin_job_id,
            origin_segment_id=origin_segment_id,
            db_path=str(self.db_path),
            translated_preview=text_preview(result.translated_text),
        )
        return result

    def store(
        self,
        segment: DocumentSegment,
        translation: TranslationResult,
        *,
        job_id: str,
    ) -> None:
        self._store_with_scope(
            config=self.config,
            scope=self.scope,
            segment=segment,
            translation=translation,
            job_id=job_id,
        )

    def seed_from_state(self, state: dict[str, Any]) -> int:
        config = state.get("config")
        if not isinstance(config, JobConfig):
            return 0

        raw_scope = state.get("translation_cache_scope")
        scope = raw_scope if isinstance(raw_scope, dict) else build_translation_cache_scope(config)
        segments = state.get("segments", [])
        translations = state.get("translations", {})
        if not isinstance(translations, dict):
            return 0

        count = 0
        for segment in segments:
            if not isinstance(segment, DocumentSegment):
                continue
            translation = translations.get(segment.segment_id)
            if not isinstance(translation, TranslationResult):
                continue
            self._store_with_scope(
                config=config,
                scope=scope,
                segment=segment,
                translation=translation,
                job_id=str(state.get("job_id") or ""),
            )
            count += 1

        log_debug(
            "translation_cache.seed_from_state.done",
            job_id=state.get("job_id"),
            stored=count,
            db_path=str(self.db_path),
        )
        return count

    def backfill_from_jobs(self) -> int:
        with closing(sqlite3.connect(self.db_path)) as conn:
            try:
                rows = conn.execute("SELECT payload_json FROM jobs").fetchall()
            except sqlite3.OperationalError as exc:
                if "no such table" in str(exc).lower():
                    log_debug("translation_cache.backfill.no_jobs_table", db_path=str(self.db_path))
                    return 0
                raise

        total = 0
        for (payload_json,) in rows:
            try:
                state = _state_from_payload(payload_json)
            except Exception as exc:  # noqa: BLE001 - legacy payload should not break cache
                log_debug(
                    "translation_cache.backfill.skip_payload",
                    db_path=str(self.db_path),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                continue
            total += self.seed_from_state(state)

        log_debug(
            "translation_cache.backfill.done",
            jobs=len(rows),
            stored=total,
            db_path=str(self.db_path),
        )
        return total

    def _store_with_scope(
        self,
        *,
        config: JobConfig,
        scope: dict[str, Any],
        segment: DocumentSegment,
        translation: TranslationResult,
        job_id: str,
    ) -> None:
        normalized_source = normalize_translation_source(segment.source_text)
        if not normalized_source or not translation.translated_text.strip():
            return

        scope_json = dumps_json(scope)
        source_hash = _source_hash(normalized_source)
        cache_key = translation_cache_key(scope, normalized_source)
        now = datetime.now(timezone.utc).isoformat()
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO translation_cache(
                        cache_key,
                        source_hash,
                        source_language,
                        target_language,
                        translator_provider,
                        translator_model,
                        scope_json,
                        normalized_source,
                        translation_json,
                        origin_job_id,
                        origin_segment_id,
                        created_at,
                        updated_at,
                        last_used_at,
                        hits
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        translation_json = excluded.translation_json,
                        origin_job_id = excluded.origin_job_id,
                        origin_segment_id = excluded.origin_segment_id,
                        updated_at = excluded.updated_at
                    """,
                    (
                        cache_key,
                        source_hash,
                        config.source_language,
                        config.target_language,
                        config.translator_provider,
                        str(scope.get("translator_model") or ""),
                        scope_json,
                        normalized_source,
                        translation.model_dump_json(),
                        job_id,
                        segment.segment_id,
                        now,
                        now,
                        now,
                    ),
                )

        log_debug(
            "translation_cache.store.done",
            segment_id=segment.segment_id,
            job_id=job_id,
            db_path=str(self.db_path),
            source_hash=source_hash,
            translated_preview=text_preview(translation.translated_text),
        )

    def _init_db(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS translation_cache (
                        cache_key TEXT PRIMARY KEY,
                        source_hash TEXT NOT NULL,
                        source_language TEXT NOT NULL,
                        target_language TEXT NOT NULL,
                        translator_provider TEXT NOT NULL,
                        translator_model TEXT NOT NULL,
                        scope_json TEXT NOT NULL,
                        normalized_source TEXT NOT NULL,
                        translation_json TEXT NOT NULL,
                        origin_job_id TEXT NOT NULL,
                        origin_segment_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_used_at TEXT NOT NULL,
                        hits INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_translation_cache_lookup
                    ON translation_cache(
                        source_hash,
                        source_language,
                        target_language,
                        translator_provider,
                        translator_model
                    )
                    """
                )


def build_translation_cache_scope(config: JobConfig) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_language": _normalize_language(config.source_language),
        "target_language": _normalize_language(config.target_language),
        "domain": normalize_ws(config.domain),
        "translator_provider": config.translator_provider,
        "translator_model": _translator_model(config),
        "glossary_sha256": _optional_sha256(config.glossary_path),
        "translator_prompt_sha256": _optional_sha256(
            Path(__file__).resolve().parent / "domain" / "prompts" / "translator.txt"
        ),
    }


def translation_cache_key(scope: dict[str, Any], normalized_source: str) -> str:
    payload = {
        "scope": scope,
        "source": normalized_source,
    }
    return hashlib.sha256(dumps_json(payload).encode("utf-8")).hexdigest()


def normalize_translation_source(source_text: str) -> str:
    return normalize_ws(source_text)


def copy_translation_from_cache(
    segment: DocumentSegment,
    cached_translation: TranslationResult,
    *,
    source_label: str,
) -> TranslationResult:
    note = f"Reused exact-match translation from {source_label}."
    notes = list(cached_translation.translator_notes)
    if note not in notes:
        notes.append(note)

    return cached_translation.model_copy(
        update={
            "segment_id": segment.segment_id,
            "translator_notes": notes,
        }
    )


def _state_from_payload(payload_json: str) -> dict[str, Any]:
    raw = json.loads(payload_json)
    if raw.get("config"):
        raw["config"] = JobConfig.model_validate(raw["config"])
    raw["segments"] = [
        DocumentSegment.model_validate(segment)
        for segment in raw.get("segments", [])
    ]
    raw["translations"] = {
        str(segment_id): TranslationResult.model_validate(result)
        for segment_id, result in raw.get("translations", {}).items()
    }
    return raw


def _translator_model(config: JobConfig) -> str:
    if config.translator_provider == "openai":
        return os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-5-mini")
    return "mock"


def _optional_sha256(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.exists():
        return "missing"
    return sha256_file(resolved)


def _source_hash(normalized_source: str) -> str:
    return hashlib.sha256(normalized_source.encode("utf-8")).hexdigest()


def _normalize_language(language: str) -> str:
    return normalize_ws(language).lower()
