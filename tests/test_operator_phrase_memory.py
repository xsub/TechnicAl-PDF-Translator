from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from translator.nodes.resolve import resolve_findings
from translator.nodes.review import review_translation
from translator.operator_phrase_memory import (
    UserPhraseMemory,
    add_phrase_memory_review_findings,
    apply_user_phrase_memory_to_translation,
)
from translator.schemas import DocumentSegment, JobConfig, ReviewResult, TranslationResult


class OperatorPhraseMemoryTests(unittest.TestCase):
    def test_stores_and_finds_target_phrase_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            memory = UserPhraseMemory.for_config(config)

            stored = memory.store(
                source_text="European Regulatory statement for SunPak FSP EcoPace inks",
                replace_text="Europejska informacja regulacyjna dotycząca farb drukarskich SunPak FSP EcoPace",
                preferred_text="Europejskie oświadczenie regulacyjne dotyczące farb SunPak FSP EcoPace",
                approved_text="Europejskie oświadczenie regulacyjne dotyczące farb SunPak FSP EcoPace",
                job_id="job-a",
                segment_id="s1",
            )

            self.assertTrue(stored)
            self.assertEqual(memory.count(), 1)
            matches = memory.find_matches(
                source_text="European Regulatory statement for SunPak FSP EcoPace inks",
                translated_text="Europejska informacja regulacyjna dotycząca farb drukarskich SunPak FSP EcoPace",
            )
            self.assertEqual(len(matches), 1)
            self.assertEqual(
                matches[0].preferred_text,
                "Europejskie oświadczenie regulacyjne dotyczące farb SunPak FSP EcoPace",
            )

    def test_auto_mode_applies_user_phrase_before_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir).model_copy(update={"use_user_phrase_memory": True})
            segment = _segment()
            translation = TranslationResult(
                segment_id="s1",
                translated_text="Tekst o wysuszonej warstwie farby.",
                confidence="medium",
            )
            UserPhraseMemory.for_config(config).store(
                source_text=segment.source_text,
                replace_text="wysuszonej warstwie farby",
                preferred_text="wyschniętej powłoce farby",
                approved_text="Tekst o wyschniętej powłoce farby.",
                job_id="job-a",
                segment_id="s1",
            )

            updated, matches = apply_user_phrase_memory_to_translation(segment, translation, config)

            self.assertEqual(len(matches), 1)
            self.assertEqual(updated.translated_text, "Tekst o wyschniętej powłoce farby.")
            self.assertEqual(updated.confidence, "high")
            self.assertIn("Applied trusted user phrase memory", updated.translator_notes[-1])

    def test_review_gets_phrase_memory_finding_when_auto_mode_is_off(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            segment = _segment()
            translation = TranslationResult(
                segment_id="s1",
                translated_text="Tekst o wysuszonej warstwie farby.",
                confidence="high",
            )
            UserPhraseMemory.for_config(config).store(
                source_text=segment.source_text,
                replace_text="wysuszonej warstwie farby",
                preferred_text="wyschniętej powłoce farby",
                approved_text="Tekst o wyschniętej powłoce farby.",
                job_id="job-a",
                segment_id="s1",
            )

            review = add_phrase_memory_review_findings(
                segment,
                translation,
                config,
                ReviewResult(segment_id="s1", verdict="accept", findings=[]),
            )

            self.assertEqual(review.verdict, "human_review")
            self.assertEqual(review.findings[0].category, "user_phrase_memory")
            self.assertEqual(review.findings[0].proposed_translation, "Tekst o wyschniętej powłoce farby.")

    def test_phrase_memory_finding_routes_to_human_review_not_auto_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            segment = _segment()
            translation = TranslationResult(
                segment_id="s1",
                translated_text="Tekst o wysuszonej warstwie farby.",
                confidence="high",
            )
            UserPhraseMemory.for_config(config).store(
                source_text=segment.source_text,
                replace_text="wysuszonej warstwie farby",
                preferred_text="wyschniętej powłoce farby",
                approved_text="Tekst o wyschniętej powłoce farby.",
                job_id="job-a",
                segment_id="s1",
            )

            reviewed = review_translation(
                {
                    "job_id": "job-a",
                    "config": config,
                    "segments": [segment],
                    "translations": {"s1": translation},
                    "review_results": {},
                }
            )
            resolved = resolve_findings(reviewed)

            self.assertEqual(reviewed["review_results"]["s1"].findings[0].category, "user_phrase_memory")
            self.assertEqual(resolved["unresolved_segments"], ["s1"])
            self.assertEqual(resolved["revision_required_segments"], [])


def _config(temp_dir: str) -> JobConfig:
    return JobConfig(
        source_pdf_path="dummy.pdf",
        output_dir=str(Path(temp_dir) / "output"),
        source_language="English",
        target_language="Polish",
        translator_provider="mock",
        reviewer_provider="mock",
        max_revision_attempts=1,
    )


def _segment() -> DocumentSegment:
    return DocumentSegment(
        segment_id="s1",
        page_number=1,
        order_index=1,
        block_type="paragraph",
        source_text="Text about dry ink film.",
    )


if __name__ == "__main__":
    unittest.main()
