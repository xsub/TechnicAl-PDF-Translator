from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from translator.nodes.pipeline import translate_and_review_segments
from translator.schemas import DocumentSegment, JobConfig, ReviewResult, TranslationResult


class SlidingTranslator:
    def __init__(self, review_started: threading.Event, events: list[tuple[str, str, bool | None]]) -> None:
        self.review_started = review_started
        self.events = events
        self.model_name = "test-translation-model"

    def translate(self, segment: DocumentSegment, glossary, config: JobConfig) -> TranslationResult:  # noqa: ANN001
        self.events.append(("translate_start", segment.segment_id, None))
        if segment.segment_id == "s2":
            released_by_review = self.review_started.wait(timeout=1.0)
            self.events.append(("s2_released", segment.segment_id, released_by_review))
        self.events.append(("translate_done", segment.segment_id, None))
        return TranslationResult(
            segment_id=segment.segment_id,
            translated_text=f"PL: {segment.source_text}",
            confidence="high",
        )


class SlidingReviewer:
    def __init__(self, review_started: threading.Event, events: list[tuple[str, str, bool | None]]) -> None:
        self.review_started = review_started
        self.events = events
        self.model_name = "test-review-model"

    def review(self, segment: DocumentSegment, translation: TranslationResult, glossary, config: JobConfig) -> ReviewResult:  # noqa: ANN001
        self.events.append(("review_start", segment.segment_id, None))
        self.review_started.set()
        self.events.append(("review_done", segment.segment_id, None))
        return ReviewResult(segment_id=segment.segment_id, verdict="accept", findings=[])


class PipelineProcessingTests(unittest.TestCase):
    def test_review_starts_before_all_translations_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            review_started = threading.Event()
            events: list[tuple[str, str, bool | None]] = []
            config = JobConfig(
                source_pdf_path="dummy.pdf",
                output_dir=str(Path(temp_dir) / "out"),
                translator_provider="openai",
                reviewer_provider="openai",
                translation_concurrency=2,
                review_concurrency=1,
            )
            segments = [
                _segment("s1", "Declaration of compliance"),
                _segment("s2", "Overall migration below limit"),
            ]
            translator = SlidingTranslator(review_started, events)
            reviewer = SlidingReviewer(review_started, events)
            progress_events: list[dict] = []

            with (
                patch("translator.nodes.pipeline.build_translator", return_value=translator),
                patch("translator.nodes.pipeline.build_reviewer", return_value=reviewer),
            ):
                result = translate_and_review_segments(
                    {
                        "job_id": "job-test",
                        "config": config,
                        "segments": segments,
                        "translations": {},
                        "review_results": {},
                        "translation_memory_hits": 0,
                        "persistent_translation_cache_hits": 0,
                        "translation_memory_misses": 0,
                    },
                    progress_callback=progress_events.append,
                )

            self.assertEqual(len(result["translations"]), 2)
            self.assertEqual(len(result["review_results"]), 2)
            reviewed_events = [
                event
                for event in progress_events
                if event["message"] == "Pipeline: segment po recenzji"
            ]
            self.assertTrue(reviewed_events)
            self.assertEqual(reviewed_events[-1]["translations_done"], 2)
            self.assertEqual(reviewed_events[-1]["translations_total"], 2)
            self.assertEqual(reviewed_events[-1]["reviews_done"], 2)
            self.assertEqual(reviewed_events[-1]["reviews_total"], 2)
            self.assertIn(("s2_released", "s2", True), events)
            self.assertLess(
                events.index(("review_start", "s1", None)),
                events.index(("translate_done", "s2", None)),
            )

    def test_pipeline_reviews_existing_translations_without_retranslating(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            review_started = threading.Event()
            events: list[tuple[str, str, bool | None]] = []
            config = JobConfig(
                source_pdf_path="dummy.pdf",
                output_dir=str(Path(temp_dir) / "out"),
                translator_provider="openai",
                reviewer_provider="openai",
                translation_concurrency=2,
                review_concurrency=2,
            )
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
            reviewer = SlidingReviewer(review_started, events)

            with (
                patch("translator.nodes.pipeline.build_translator", side_effect=AssertionError("translator should not run")),
                patch("translator.nodes.pipeline.build_reviewer", return_value=reviewer),
            ):
                result = translate_and_review_segments(
                    {
                        "job_id": "job-test",
                        "config": config,
                        "segments": segments,
                        "translations": translations,
                        "review_results": {"s1": ReviewResult(segment_id="s1", verdict="accept")},
                        "translation_memory_hits": 0,
                        "persistent_translation_cache_hits": 0,
                        "translation_memory_misses": 0,
                    }
                )

            self.assertEqual(len(result["translations"]), 2)
            self.assertEqual(len(result["review_results"]), 2)
            self.assertEqual([event for event in events if event[0] == "review_start"], [("review_start", "s2", None)])


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
