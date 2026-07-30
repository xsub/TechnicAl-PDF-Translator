from __future__ import annotations

import unittest

from translator.operator_refinement import suggest_operator_preferred_phrase, suggest_operator_replacement_text
from translator.schemas import ReviewFinding


class OperatorRefinementSuggestionTests(unittest.TestCase):
    def test_current_translation_is_the_default_text_to_replace(self) -> None:
        current = "Europejska informacja regulacyjna dotycząca farb drukarskich SunPak FSP EcoPace"

        self.assertEqual(suggest_operator_replacement_text(current), current)

    def test_reviewer_evidence_that_differs_from_current_translation_becomes_preferred_phrase(self) -> None:
        current = "Europejska informacja regulacyjna dotycząca farb drukarskich SunPak FSP EcoPace"
        better = "Europejskie oświadczenie regulacyjne dotyczące farb SunPak FSP EcoPace"
        finding = ReviewFinding(
            segment_id="s1",
            severity="major",
            category="terminology",
            source_evidence="European Regulatory statement for SunPak FSP EcoPace inks",
            translation_evidence=better,
            explanation="The wording should use regulatory statement rather than regulatory information.",
            proposed_translation=None,
            confidence="high",
        )

        self.assertEqual(suggest_operator_preferred_phrase([finding], current), better)

    def test_explicit_reviewer_proposal_wins_over_evidence(self) -> None:
        current = "wysuszonej warstwie farby"
        proposed = "wyschnięta powłoka farby"
        finding = ReviewFinding(
            segment_id="s1",
            severity="major",
            category="terminology",
            source_evidence="dry ink film",
            translation_evidence=current,
            explanation="Prefer the standard printing phrase.",
            proposed_translation=proposed,
            confidence="high",
        )

        self.assertEqual(suggest_operator_preferred_phrase([finding], current), proposed)

    def test_current_translation_evidence_is_not_repeated_as_preferred_phrase(self) -> None:
        current = "Migracja globalna była poniżej 10 mg/dm²."
        finding = ReviewFinding(
            segment_id="s1",
            severity="minor",
            category="style",
            source_evidence="Overall migration was below 10 mg/dm².",
            translation_evidence=current,
            explanation="No material issue.",
            proposed_translation=None,
            confidence="medium",
        )

        self.assertEqual(suggest_operator_preferred_phrase([finding], current), "")


if __name__ == "__main__":
    unittest.main()
