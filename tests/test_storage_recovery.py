from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from translator.schemas import DocumentSegment, JobConfig, ReviewResult, TranslationResult
from translator.storage import JobStore


class StorageRecoveryTests(unittest.TestCase):
    def test_load_normalizes_completed_translation_with_stale_translating_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir) / "jobs.db")
            config = JobConfig(source_pdf_path="dummy.pdf", output_dir=str(Path(temp_dir) / "out"))
            segments = [
                _segment("s1", "A"),
                _segment("s2", "B"),
            ]
            store.save_state(
                {
                    "job_id": "job-test",
                    "config": config,
                    "source_pdf_path": "dummy.pdf",
                    "segments": segments,
                    "translations": {
                        "s1": TranslationResult(segment_id="s1", translated_text="AA"),
                        "s2": TranslationResult(segment_id="s2", translated_text="BB"),
                    },
                    "review_results": {},
                    "operator_decisions": {},
                    "status": "translating 1/2",
                }
            )

            loaded = store.load_state("job-test")

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded["status"], "segments_translated")

    def test_load_normalizes_completed_review_with_stale_reviewing_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir) / "jobs.db")
            config = JobConfig(source_pdf_path="dummy.pdf", output_dir=str(Path(temp_dir) / "out"))
            segments = [
                _segment("s1", "A"),
                _segment("s2", "B"),
            ]
            store.save_state(
                {
                    "job_id": "job-test",
                    "config": config,
                    "source_pdf_path": "dummy.pdf",
                    "segments": segments,
                    "translations": {
                        "s1": TranslationResult(segment_id="s1", translated_text="AA"),
                        "s2": TranslationResult(segment_id="s2", translated_text="BB"),
                    },
                    "review_results": {
                        "s1": ReviewResult(segment_id="s1", verdict="accept"),
                        "s2": ReviewResult(segment_id="s2", verdict="accept"),
                    },
                    "operator_decisions": {},
                    "status": "reviewing 1/2",
                }
            )

            loaded = store.load_state("job-test")

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded["status"], "translation_reviewed")

    def test_load_normalizes_completed_pipeline_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir) / "jobs.db")
            config = JobConfig(source_pdf_path="dummy.pdf", output_dir=str(Path(temp_dir) / "out"))
            segments = [
                _segment("s1", "A"),
                _segment("s2", "B"),
            ]
            store.save_state(
                {
                    "job_id": "job-test",
                    "config": config,
                    "source_pdf_path": "dummy.pdf",
                    "segments": segments,
                    "translations": {
                        "s1": TranslationResult(segment_id="s1", translated_text="AA"),
                        "s2": TranslationResult(segment_id="s2", translated_text="BB"),
                    },
                    "review_results": {
                        "s1": ReviewResult(segment_id="s1", verdict="accept"),
                        "s2": ReviewResult(segment_id="s2", verdict="accept"),
                    },
                    "operator_decisions": {},
                    "status": "pipeline translated 2/2, reviewed 1/2",
                }
            )

            loaded = store.load_state("job-test")

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded["status"], "translation_reviewed")


def _segment(segment_id: str, source_text: str) -> DocumentSegment:
    order_index = int(segment_id.removeprefix("s"))
    return DocumentSegment(
        segment_id=segment_id,
        page_number=1,
        order_index=order_index,
        block_type="paragraph",
        source_text=source_text,
    )


if __name__ == "__main__":
    unittest.main()
