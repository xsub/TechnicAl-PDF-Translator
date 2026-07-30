from __future__ import annotations

import unittest

from translator.domain.protected import extract_protected_tokens, protect_segment, validate_segment_invariants
from translator.schemas import DocumentSegment, TranslationResult


class ProtectedPatternTests(unittest.TestCase):
    def test_extracts_technical_invariants(self) -> None:
        text = "Overall migration was < 10 mg/dm² at 40°C for 10 days. CAS 123-45-6, EN 1186, ND."
        tokens = extract_protected_tokens(text)
        values = {(token.kind, token.normalized) for token in tokens}

        self.assertIn(("comparator", "<"), values)
        self.assertIn(("number", "10"), values)
        self.assertIn(("unit", "mg/dm2"), values)
        self.assertIn(("unit", "°c"), values)
        self.assertIn(("cas", "123-45-6"), values)
        self.assertIn(("standard", "EN1186"), values)
        self.assertIn(("abbreviation", "ND"), values)

    def test_changed_number_is_critical(self) -> None:
        segment = protect_segment(
            DocumentSegment(
                segment_id="s1",
                page_number=1,
                order_index=0,
                block_type="paragraph",
                source_text="Migration was below 0.05 mg/kg.",
            )
        )
        translation = TranslationResult(segment_id="s1", translated_text="Migracja była poniżej 0.5 mg/kg.")

        issues = validate_segment_invariants(segment, translation)

        self.assertTrue(any(issue.severity == "critical" for issue in issues))
        self.assertTrue(any(issue.issue_type in {"missing_number", "unexpected_addition"} for issue in issues))


if __name__ == "__main__":
    unittest.main()

