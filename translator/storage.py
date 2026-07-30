from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from translator.debug import log_debug
from translator.schemas import (
    DocumentSegment,
    JobConfig,
    JobReport,
    OperatorDecision,
    OutputVerification,
    ReviewResult,
    TranslationResult,
    ValidationIssue,
)
from translator.state import TranslationState
from translator.utils import dumps_json, ensure_dir, sha256_file


class JobStore:
    def __init__(self, db_path: str | Path = "storage/jobs.db") -> None:
        self.db_path = Path(db_path)
        ensure_dir(self.db_path.parent)
        self._init_db()

    def save_state(self, state: TranslationState) -> None:
        payload = dumps_json(state)
        log_debug(
            "storage.save_state.start",
            job_id=state["job_id"],
            status=state.get("status", "unknown"),
            payload_chars=len(payload),
            db_path=str(self.db_path),
        )
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO jobs(job_id, status, source_pdf_path, output_pdf_path, payload_json, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        status=excluded.status,
                        output_pdf_path=excluded.output_pdf_path,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        state["job_id"],
                        state.get("status", "unknown"),
                        state.get("source_pdf_path"),
                        state.get("output_pdf_path"),
                        payload,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        log_debug(
            "storage.save_state.done",
            job_id=state["job_id"],
            status=state.get("status", "unknown"),
            db_path=str(self.db_path),
        )

    def load_state(self, job_id: str) -> TranslationState | None:
        log_debug("storage.load_state.start", job_id=job_id, db_path=str(self.db_path))
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT payload_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()

        if row is None:
            log_debug("storage.load_state.missing", job_id=job_id, db_path=str(self.db_path))
            return None

        state = _state_from_payload(row[0])
        log_debug(
            "storage.load_state.done",
            job_id=state.get("job_id"),
            status=state.get("status"),
            segments=len(state.get("segments", [])),
            translations=len(state.get("translations", {})),
        )
        return state

    def load_latest_state(self) -> TranslationState | None:
        log_debug("storage.load_latest_state.start", db_path=str(self.db_path))
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM jobs
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            log_debug("storage.load_latest_state.empty", db_path=str(self.db_path))
            return None

        state = _state_from_payload(row[0])
        log_debug(
            "storage.load_latest_state.done",
            job_id=state.get("job_id"),
            status=state.get("status"),
            segments=len(state.get("segments", [])),
            translations=len(state.get("translations", {})),
        )
        return state

    def write_report(self, state: TranslationState) -> Path:
        output_dir = Path(state["config"].output_dir)
        ensure_dir(output_dir)
        report_path = output_dir / f"{state['job_id']}_report.json"
        log_debug("storage.write_report.start", job_id=state["job_id"], report_path=str(report_path))
        report = build_report(state, report_path)
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        log_debug("storage.write_report.done", job_id=state["job_id"], report_path=str(report_path))
        return report_path

    def _init_db(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        source_pdf_path TEXT NOT NULL,
                        output_pdf_path TEXT,
                        payload_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )


def _state_from_payload(payload_json: str) -> TranslationState:
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
    raw["deterministic_issues"] = [
        ValidationIssue.model_validate(issue)
        for issue in raw.get("deterministic_issues", [])
    ]
    raw["review_results"] = {
        str(segment_id): ReviewResult.model_validate(result)
        for segment_id, result in raw.get("review_results", {}).items()
    }
    raw["operator_decisions"] = {
        str(segment_id): OperatorDecision.model_validate(decision)
        for segment_id, decision in raw.get("operator_decisions", {}).items()
    }

    if raw.get("output_verification"):
        raw["output_verification"] = OutputVerification.model_validate(raw["output_verification"])
    else:
        raw["output_verification"] = None

    return cast(TranslationState, raw)


def build_report(state: TranslationState, report_path: str | Path | None = None) -> JobReport:
    deterministic_issues = state.get("deterministic_issues", [])
    review_findings = [
        finding
        for result in state.get("review_results", {}).values()
        for finding in result.findings
    ]
    output_pdf_path = state.get("output_pdf_path")
    metrics = {
        "segments_total": len(state.get("segments", [])),
        "segments_auto_accepted": len(state.get("segments", [])) - len(state.get("unresolved_segments", [])),
        "segments_human_review_required": len(state.get("unresolved_segments", [])),
        "deterministic_issues": len(deterministic_issues),
        "critical_deterministic_issues": sum(1 for issue in deterministic_issues if issue.severity == "critical"),
        "review_findings": len(review_findings),
        "critical_or_major_review_findings": sum(
            1 for finding in review_findings if finding.severity in {"critical", "major"}
        ),
        "translation_memory_hits": state.get("translation_memory_hits", 0),
        "translation_memory_misses": state.get("translation_memory_misses", 0),
    }
    return JobReport(
        job_id=state["job_id"],
        source_pdf_path=state["source_pdf_path"],
        output_pdf_path=output_pdf_path,
        report_path=str(report_path) if report_path else None,
        source_language=state["config"].source_language,
        target_language=state["config"].target_language,
        source_sha256=sha256_file(state["source_pdf_path"]),
        output_sha256=sha256_file(output_pdf_path) if output_pdf_path else None,
        status=state.get("status", "unknown"),
        metrics=metrics,
        deterministic_issues=deterministic_issues,
        review_findings=review_findings,
        output_verification=state.get("output_verification"),
    )
