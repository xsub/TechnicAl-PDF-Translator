from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from translator.domain.locale_formatting import apply_locale_formatting, localize_decimal_separator
from translator.domain.protected import protect_segment, validate_segment_invariants
from translator.schemas import DocumentSegment, JobConfig, TranslationResult
from translator.storage import JobStore


class LocaleFormattingTests(unittest.TestCase):
    def test_localizes_decimal_dot_for_polish_target(self) -> None:
        config = JobConfig(source_pdf_path="dummy.pdf", source_language="English", target_language="Polish")
        translation = TranslationResult(
            segment_id="s1",
            translated_text="Migracja była poniżej 0.01 mg/kg and 1,234.56 ppm.",
        )

        localized = apply_locale_formatting(translation, config)

        self.assertEqual(localized.translated_text, "Migracja była poniżej 0,01 mg/kg and 1 234,56 ppm.")
        self.assertIn("Localized decimal separator for target locale.", localized.translator_notes)

    def test_keeps_decimal_dot_for_english_target(self) -> None:
        config = JobConfig(source_pdf_path="dummy.pdf", source_language="Polish", target_language="English")
        translation = TranslationResult(segment_id="s1", translated_text="Migration was below 0.01 mg/kg.")

        localized = apply_locale_formatting(translation, config)

        self.assertEqual(localized.translated_text, "Migration was below 0.01 mg/kg.")
        self.assertEqual(localized.translator_notes, [])

    def test_does_not_change_structural_references(self) -> None:
        text = "Section 1.2 and table 2.1 stay, but the result is 0.05 mg/kg."

        self.assertEqual(
            localize_decimal_separator(text),
            "Section 1.2 and table 2.1 stay, but the result is 0,05 mg/kg.",
        )

    def test_decimal_comma_keeps_numeric_invariant(self) -> None:
        segment = protect_segment(
            DocumentSegment(
                segment_id="s1",
                page_number=1,
                order_index=1,
                block_type="paragraph",
                source_text="Migration was below 0.05 mg/kg.",
            )
        )
        translation = TranslationResult(segment_id="s1", translated_text="Migracja była poniżej 0,05 mg/kg.")

        issues = validate_segment_invariants(segment, translation)

        self.assertFalse(any(issue.issue_type in {"missing_number", "unexpected_addition"} for issue in issues))

    def test_localizes_translations_loaded_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = JobConfig(
                source_pdf_path="dummy.pdf",
                output_dir=str(Path(temp_dir) / "out"),
                target_language="Polish",
            )
            store = JobStore(Path(temp_dir) / "jobs.db")
            store.save_state(
                {
                    "job_id": "job-test",
                    "config": config,
                    "source_pdf_path": "dummy.pdf",
                    "segments": [],
                    "translations": {
                        "s1": TranslationResult(
                            segment_id="s1",
                            translated_text="Migracja była poniżej 0.01 mg/kg.",
                        )
                    },
                    "operator_decisions": {},
                    "status": "segments_translated",
                }
            )

            loaded = store.load_state("job-test")

        assert loaded is not None
        self.assertEqual(
            loaded["translations"]["s1"].translated_text,
            "Migracja była poniżej 0,01 mg/kg.",
        )


if __name__ == "__main__":
    unittest.main()
