from __future__ import annotations

import unittest
from unittest.mock import patch

from translator.nodes.translate import translate_segments
from translator.schemas import DocumentSegment, JobConfig, ReviewFinding, TranslationResult


class CountingTranslator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def translate(self, segment: DocumentSegment, glossary, config: JobConfig) -> TranslationResult:  # noqa: ANN001
        self.calls.append(segment.segment_id)
        return TranslationResult(
            segment_id=segment.segment_id,
            translated_text=f"PL: {' '.join(segment.source_text.split())}",
            translator_notes=["fresh llm translation"],
            confidence="high",
        )

    def revise(
        self,
        segment: DocumentSegment,
        current: TranslationResult,
        findings: list[ReviewFinding],
        glossary,
        config: JobConfig,
    ) -> TranslationResult:  # pragma: no cover - not used in this test
        return current


class TranslationMemoryTests(unittest.TestCase):
    def test_reuses_existing_exact_match_translation_before_calling_llm(self) -> None:
        segments = [
            DocumentSegment(
                segment_id="s1",
                page_number=1,
                order_index=1,
                block_type="paragraph",
                source_text="Declaration of compliance",
            ),
            DocumentSegment(
                segment_id="s2",
                page_number=2,
                order_index=2,
                block_type="paragraph",
                source_text="Declaration   of\ncompliance",
            ),
            DocumentSegment(
                segment_id="s3",
                page_number=2,
                order_index=3,
                block_type="paragraph",
                source_text="Overall migration was below 10 mg/dm2.",
            ),
        ]
        translator = CountingTranslator()
        checkpoints: list[dict] = []
        state = {
            "job_id": "job-test",
            "config": JobConfig(source_pdf_path="dummy.pdf"),
            "segments": segments,
            "translations": {},
            "translation_memory_hits": 0,
            "translation_memory_misses": 0,
        }

        with patch("translator.nodes.translate.build_translator", return_value=translator):
            result = translate_segments(state, checkpoint_callback=checkpoints.append)

        self.assertEqual(translator.calls, ["s1", "s3"])
        self.assertEqual(result["translation_memory_hits"], 1)
        self.assertEqual(result["translation_memory_misses"], 2)
        self.assertEqual(result["translations"]["s2"].segment_id, "s2")
        self.assertEqual(result["translations"]["s2"].translated_text, "PL: Declaration of compliance")
        self.assertIn("Reused exact-match translation from segment s1.", result["translations"]["s2"].translator_notes)
        self.assertTrue(any(checkpoint.get("translation_memory_hits") == 1 for checkpoint in checkpoints))

    def test_reuses_translation_loaded_from_checkpoint(self) -> None:
        segments = [
            DocumentSegment(
                segment_id="s1",
                page_number=1,
                order_index=1,
                block_type="paragraph",
                source_text="Repeated footer",
            ),
            DocumentSegment(
                segment_id="s2",
                page_number=2,
                order_index=2,
                block_type="paragraph",
                source_text="Repeated footer",
            ),
        ]
        translator = CountingTranslator()
        state = {
            "job_id": "job-test",
            "config": JobConfig(source_pdf_path="dummy.pdf"),
            "segments": segments,
            "translations": {
                "s1": TranslationResult(
                    segment_id="s1",
                    translated_text="Powtarzalna stopka",
                    confidence="high",
                )
            },
            "translation_memory_hits": 0,
            "translation_memory_misses": 1,
        }

        with patch("translator.nodes.translate.build_translator", return_value=translator):
            result = translate_segments(state)

        self.assertEqual(translator.calls, [])
        self.assertEqual(result["translation_memory_hits"], 1)
        self.assertEqual(result["translation_memory_misses"], 1)
        self.assertEqual(result["translations"]["s2"].translated_text, "Powtarzalna stopka")


if __name__ == "__main__":
    unittest.main()
