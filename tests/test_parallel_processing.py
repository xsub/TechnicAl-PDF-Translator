from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from translator.nodes.review import review_translation
from translator.nodes.translate import translate_segments
from translator.schemas import DocumentSegment, JobConfig, ReviewResult, TranslationResult


class ParallelTranslator:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.model_name = "test-translation-model"

    def translate(self, segment: DocumentSegment, glossary, config: JobConfig) -> TranslationResult:  # noqa: ANN001
        self.calls.append(segment.segment_id)
        return TranslationResult(
            segment_id=segment.segment_id,
            translated_text=f"PL: {segment.source_text}",
            confidence="high",
        )


class ParallelReviewer:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.model_name = "test-review-model"

    def review(self, segment: DocumentSegment, translation: TranslationResult, glossary, config: JobConfig) -> ReviewResult:  # noqa: ANN001
        self.calls.append(segment.segment_id)
        return ReviewResult(segment_id=segment.segment_id, verdict="accept", findings=[])


class ParallelProcessingTests(unittest.TestCase):
    def test_parallel_translation_checkpoint_tracks_active_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = JobConfig(
                source_pdf_path="dummy.pdf",
                output_dir=str(Path(temp_dir) / "out"),
                translator_provider="openai",
                translation_concurrency=2,
            )
            segments = [
                _segment("s1", "Declaration of compliance"),
                _segment("s2", "Overall migration below limit"),
                _segment("s3", "Specific migration not detected"),
            ]
            translator = ParallelTranslator()
            checkpoints: list[dict] = []

            with patch("translator.nodes.translate.build_translator", return_value=translator):
                result = translate_segments(
                    {
                        "job_id": "job-test",
                        "config": config,
                        "segments": segments,
                        "translations": {},
                        "translation_memory_hits": 0,
                        "persistent_translation_cache_hits": 0,
                        "translation_memory_misses": 0,
                    },
                    checkpoint_callback=checkpoints.append,
                )

            self.assertCountEqual(translator.calls, ["s1", "s2", "s3"])
            self.assertEqual(result["translation_memory_misses"], 3)
            self.assertTrue(
                any((checkpoint.get("llm_inflight") or {}).get("active") == 2 for checkpoint in checkpoints)
            )

    def test_parallel_review_checkpoint_tracks_active_requests(self) -> None:
        config = JobConfig(
            source_pdf_path="dummy.pdf",
            reviewer_provider="openai",
            review_concurrency=2,
        )
        segments = [
            _segment("s1", "Declaration of compliance"),
            _segment("s2", "Overall migration below limit"),
            _segment("s3", "Specific migration not detected"),
        ]
        translations = {
            segment.segment_id: TranslationResult(
                segment_id=segment.segment_id,
                translated_text=f"PL: {segment.source_text}",
                confidence="high",
            )
            for segment in segments
        }
        reviewer = ParallelReviewer()
        checkpoints: list[dict] = []

        with patch("translator.nodes.review.build_reviewer", return_value=reviewer):
            result = review_translation(
                {
                    "job_id": "job-test",
                    "config": config,
                    "segments": segments,
                    "translations": translations,
                    "review_results": {},
                },
                checkpoint_callback=checkpoints.append,
            )

        self.assertCountEqual(reviewer.calls, ["s1", "s2", "s3"])
        self.assertEqual(len(result["review_results"]), 3)
        self.assertTrue(
            any((checkpoint.get("llm_inflight") or {}).get("active") == 2 for checkpoint in checkpoints)
        )

    def test_translation_uses_default_concurrency_for_legacy_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = JobConfig(
                source_pdf_path="dummy.pdf",
                output_dir=str(Path(temp_dir) / "out"),
                translator_provider="openai",
            )
            delattr(config, "translation_concurrency")
            segments = [
                _segment("s1", "Declaration of compliance"),
                _segment("s2", "Overall migration below limit"),
            ]
            translator = ParallelTranslator()

            with patch("translator.nodes.translate.build_translator", return_value=translator):
                result = translate_segments(
                    {
                        "job_id": "job-test",
                        "config": config,
                        "segments": segments,
                        "translations": {},
                        "translation_memory_hits": 0,
                        "persistent_translation_cache_hits": 0,
                        "translation_memory_misses": 0,
                    },
                )

            self.assertCountEqual(translator.calls, ["s1", "s2"])
            self.assertEqual(len(result["translations"]), 2)

    def test_review_uses_default_concurrency_for_legacy_config(self) -> None:
        config = JobConfig(
            source_pdf_path="dummy.pdf",
            reviewer_provider="openai",
        )
        delattr(config, "review_concurrency")
        segments = [
            _segment("s1", "Declaration of compliance"),
            _segment("s2", "Overall migration below limit"),
        ]
        translations = {
            segment.segment_id: TranslationResult(
                segment_id=segment.segment_id,
                translated_text=f"PL: {segment.source_text}",
                confidence="high",
            )
            for segment in segments
        }
        reviewer = ParallelReviewer()

        with patch("translator.nodes.review.build_reviewer", return_value=reviewer):
            result = review_translation(
                {
                    "job_id": "job-test",
                    "config": config,
                    "segments": segments,
                    "translations": translations,
                    "review_results": {},
                },
            )

        self.assertCountEqual(reviewer.calls, ["s1", "s2"])
        self.assertEqual(len(result["review_results"]), 2)


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
