from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from translator.debug import log_debug, text_preview
from translator.schemas import DocumentSegment, JobConfig, ReviewFinding, ReviewResult, TranslationResult
from translator.utils import ensure_dir, normalize_ws


@dataclass(frozen=True)
class UserPhraseMemoryMatch:
    cache_key: str
    source_language: str
    target_language: str
    source_text: str
    replace_text: str
    preferred_text: str
    approved_text: str
    origin_job_id: str
    origin_segment_id: str
    hits: int
    reason: str


class UserPhraseMemory:
    def __init__(self, db_path: str | Path, config: JobConfig) -> None:
        self.db_path = Path(db_path)
        self.config = config
        ensure_dir(self.db_path.parent)
        self._init_db()

    @classmethod
    def for_config(cls, config: JobConfig) -> "UserPhraseMemory":
        return cls(Path(config.output_dir).parent / "jobs.db", config)

    def count(self) -> int:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM operator_phrase_memory
                WHERE source_language = ?
                  AND target_language = ?
                """,
                (self.config.source_language, self.config.target_language),
            ).fetchone()
        return int(row[0] if row else 0)

    def store(
        self,
        *,
        source_text: str,
        replace_text: str,
        preferred_text: str,
        approved_text: str,
        job_id: str,
        segment_id: str,
    ) -> bool:
        normalized_replace = normalize_phrase(replace_text)
        normalized_preferred = normalize_phrase(preferred_text)
        if not normalized_replace or not normalized_preferred or normalized_replace == normalized_preferred:
            return False

        normalized_source = normalize_phrase(source_text)
        cache_key = _memory_key(
            self.config.source_language,
            self.config.target_language,
            normalized_replace,
        )
        now = datetime.now(timezone.utc).isoformat()
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO operator_phrase_memory(
                        cache_key,
                        source_language,
                        target_language,
                        normalized_source,
                        normalized_replace,
                        normalized_preferred,
                        source_text,
                        replace_text,
                        preferred_text,
                        approved_text,
                        origin_job_id,
                        origin_segment_id,
                        created_at,
                        updated_at,
                        last_used_at,
                        hits
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        normalized_source = excluded.normalized_source,
                        normalized_preferred = excluded.normalized_preferred,
                        source_text = excluded.source_text,
                        replace_text = excluded.replace_text,
                        preferred_text = excluded.preferred_text,
                        approved_text = excluded.approved_text,
                        origin_job_id = excluded.origin_job_id,
                        origin_segment_id = excluded.origin_segment_id,
                        updated_at = excluded.updated_at
                    """,
                    (
                        cache_key,
                        self.config.source_language,
                        self.config.target_language,
                        normalized_source,
                        normalized_replace,
                        normalized_preferred,
                        normalize_ws(source_text),
                        normalize_ws(replace_text),
                        normalize_ws(preferred_text),
                        normalize_ws(approved_text),
                        job_id,
                        segment_id,
                        now,
                        now,
                        now,
                    ),
                )

        log_debug(
            "operator_phrase_memory.store.done",
            job_id=job_id,
            segment_id=segment_id,
            replace_preview=text_preview(replace_text),
            preferred_preview=text_preview(preferred_text),
        )
        return True

    def find_matches(
        self,
        *,
        source_text: str,
        translated_text: str,
        limit: int = 5,
    ) -> list[UserPhraseMemoryMatch]:
        normalized_source = normalize_phrase(source_text)
        normalized_translation = normalize_phrase(translated_text)
        if not normalized_translation:
            return []

        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                """
                SELECT
                    cache_key,
                    source_language,
                    target_language,
                    normalized_source,
                    normalized_replace,
                    normalized_preferred,
                    source_text,
                    replace_text,
                    preferred_text,
                    approved_text,
                    origin_job_id,
                    origin_segment_id,
                    hits
                FROM operator_phrase_memory
                WHERE source_language = ?
                  AND target_language = ?
                ORDER BY updated_at DESC
                """,
                (self.config.source_language, self.config.target_language),
            ).fetchall()

        matches: list[UserPhraseMemoryMatch] = []
        seen_replace: set[str] = set()
        for row in rows:
            normalized_replace = str(row[4] or "")
            normalized_preferred = str(row[5] or "")
            if not normalized_replace or not normalized_preferred or normalized_replace in seen_replace:
                continue

            reason = ""
            if normalized_replace in normalized_translation and normalized_preferred not in normalized_translation:
                reason = "target_phrase"
            elif normalized_source and normalized_source == str(row[3] or "") and normalized_preferred not in normalized_translation:
                reason = "source_exact"
            if not reason:
                continue

            seen_replace.add(normalized_replace)
            matches.append(
                UserPhraseMemoryMatch(
                    cache_key=str(row[0]),
                    source_language=str(row[1]),
                    target_language=str(row[2]),
                    source_text=str(row[6] or ""),
                    replace_text=str(row[7] or ""),
                    preferred_text=str(row[8] or ""),
                    approved_text=str(row[9] or ""),
                    origin_job_id=str(row[10] or ""),
                    origin_segment_id=str(row[11] or ""),
                    hits=int(row[12] or 0),
                    reason=reason,
                )
            )
            if len(matches) >= limit:
                break

        return sorted(matches, key=lambda match: (match.reason != "target_phrase", -len(match.replace_text)))

    def mark_used(self, matches: list[UserPhraseMemoryMatch]) -> None:
        if not matches:
            return

        now = datetime.now(timezone.utc).isoformat()
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.executemany(
                    """
                    UPDATE operator_phrase_memory
                    SET hits = hits + 1,
                        last_used_at = ?
                    WHERE cache_key = ?
                    """,
                    [(now, match.cache_key) for match in matches],
                )

    def _init_db(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS operator_phrase_memory (
                        cache_key TEXT PRIMARY KEY,
                        source_language TEXT NOT NULL,
                        target_language TEXT NOT NULL,
                        normalized_source TEXT NOT NULL,
                        normalized_replace TEXT NOT NULL,
                        normalized_preferred TEXT NOT NULL,
                        source_text TEXT NOT NULL,
                        replace_text TEXT NOT NULL,
                        preferred_text TEXT NOT NULL,
                        approved_text TEXT NOT NULL,
                        origin_job_id TEXT NOT NULL,
                        origin_segment_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_used_at TEXT NOT NULL,
                        hits INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )


def normalize_phrase(text: str) -> str:
    return normalize_ws(text).casefold()


def apply_user_phrase_memory_to_translation(
    segment: DocumentSegment,
    translation: TranslationResult,
    config: JobConfig,
) -> tuple[TranslationResult, list[UserPhraseMemoryMatch]]:
    if not getattr(config, "use_user_phrase_memory", False):
        return translation, []

    memory = UserPhraseMemory.for_config(config)
    matches = memory.find_matches(source_text=segment.source_text, translated_text=translation.translated_text)
    updated_text = translation.translated_text
    applied_matches: list[UserPhraseMemoryMatch] = []

    for match in matches:
        proposed_text = apply_phrase_match(updated_text, match)
        if proposed_text != updated_text:
            updated_text = proposed_text
            applied_matches.append(match)

    if not applied_matches:
        return translation, []

    memory.mark_used(applied_matches)
    notes = list(translation.translator_notes)
    for match in applied_matches:
        note = f"Applied trusted user phrase memory: {match.replace_text} -> {match.preferred_text}."
        if note not in notes:
            notes.append(note)

    return (
        translation.model_copy(
            update={
                "translated_text": updated_text,
                "translator_notes": notes,
                "confidence": "high",
            }
        ),
        applied_matches,
    )


def add_phrase_memory_review_findings(
    segment: DocumentSegment,
    translation: TranslationResult,
    config: JobConfig,
    review_result: ReviewResult,
) -> ReviewResult:
    if getattr(config, "use_user_phrase_memory", False):
        return review_result

    findings = build_phrase_memory_review_findings(segment, translation, config)
    if not findings:
        return review_result

    merged_findings = [*review_result.findings, *findings]
    verdict = _review_verdict(merged_findings, review_result.verdict)
    return review_result.model_copy(update={"findings": merged_findings, "verdict": verdict})


def build_phrase_memory_review_findings(
    segment: DocumentSegment,
    translation: TranslationResult,
    config: JobConfig,
) -> list[ReviewFinding]:
    memory = UserPhraseMemory.for_config(config)
    matches = memory.find_matches(source_text=segment.source_text, translated_text=translation.translated_text)
    findings: list[ReviewFinding] = []

    for match in matches:
        proposed_translation = apply_phrase_match(translation.translated_text, match)
        if proposed_translation == translation.translated_text:
            continue

        findings.append(
            ReviewFinding(
                segment_id=segment.segment_id,
                severity="major",
                category="user_phrase_memory",
                source_evidence=segment.source_text,
                translation_evidence=match.replace_text if match.reason == "target_phrase" else translation.translated_text,
                explanation=(
                    "Pamięć fraz operatora ma zatwierdzoną podmianę: "
                    f"użyj {match.preferred_text!r} zamiast {match.replace_text!r}."
                ),
                proposed_translation=proposed_translation,
                confidence="high",
            )
        )

    return findings


def apply_phrase_match(text: str, match: UserPhraseMemoryMatch) -> str:
    if match.reason == "source_exact" and match.approved_text:
        return match.approved_text

    replaced = _replace_once_case_insensitive(text, match.replace_text, match.preferred_text)
    if replaced != text:
        return replaced

    if normalize_phrase(text) == normalize_phrase(match.replace_text):
        return match.approved_text or match.preferred_text

    return text


def _replace_once_case_insensitive(text: str, old: str, new: str) -> str:
    if not old.strip() or not new.strip():
        return text
    pattern = re.compile(re.escape(old.strip()), flags=re.IGNORECASE)
    return pattern.sub(new.strip(), text, count=1)


def _review_verdict(findings: list[ReviewFinding], fallback: str) -> str:
    if any(finding.severity in {"critical", "major"} for finding in findings):
        return "human_review"
    if findings:
        return "revise" if fallback == "accept" else fallback
    return fallback


def _memory_key(source_language: str, target_language: str, normalized_replace: str) -> str:
    raw = "\0".join([source_language.casefold(), target_language.casefold(), normalized_replace])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
