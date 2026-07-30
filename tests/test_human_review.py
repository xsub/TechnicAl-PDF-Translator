from __future__ import annotations

import unittest

from translator.nodes.human_review import apply_operator_decisions
from translator.schemas import DocumentSegment, JobConfig, TranslationResult


class HumanReviewTests(unittest.TestCase):
    def test_operator_edit_note_is_kept_in_translation_history(self) -> None:
        state = {
            "job_id": "job-test",
            "config": JobConfig(source_pdf_path="dummy.pdf"),
            "segments": [
                DocumentSegment(
                    segment_id="s1",
                    page_number=1,
                    order_index=1,
                    block_type="paragraph",
                    source_text="dry ink film",
                )
            ],
            "translations": {
                "s1": TranslationResult(
                    segment_id="s1",
                    translated_text="wysuszonej warstwie farby",
                    translator_notes=[],
                )
            },
            "unresolved_segments": ["s1"],
            "operator_decisions": {
                "s1": {
                    "action": "edit",
                    "text": "wyschnięta powłoka farby",
                    "note": "Fragment poprawiony przez użytkownika: preferowana fraza `wyschnięta powłoka farby`.",
                }
            },
        }

        result = apply_operator_decisions(state)

        translation = result["translations"]["s1"]
        self.assertEqual(translation.translated_text, "wyschnięta powłoka farby")
        self.assertIn("Operator edited translation.", translation.translator_notes)
        self.assertIn(
            "Fragment poprawiony przez użytkownika: preferowana fraza `wyschnięta powłoka farby`.",
            translation.translator_notes,
        )
        self.assertEqual(result["unresolved_segments"], [])


if __name__ == "__main__":
    unittest.main()
