from __future__ import annotations

import unittest

from translator.domain.glossary import load_glossary
from translator.languages import LANGUAGE_OPTIONS, is_polish_language, language_index


class LanguageConfigurationTests(unittest.TestCase):
    def test_language_options_include_defaults_and_are_searchable_source_data(self) -> None:
        self.assertEqual(LANGUAGE_OPTIONS[language_index("English")], "English")
        self.assertEqual(LANGUAGE_OPTIONS[language_index("Polish")], "Polish")
        self.assertGreater(len(LANGUAGE_OPTIONS), 50)

    def test_polish_glossary_validation_is_only_enforced_for_polish_target(self) -> None:
        glossary = load_glossary("translator/domain/glossary.yaml")

        polish_issues = glossary.validate_translation(
            "s1",
            "Overall migration was below 10 mg/dm2.",
            "Overall migration was below 10 mg/dm2.",
            "Polish",
        )
        german_issues = glossary.validate_translation(
            "s1",
            "Overall migration was below 10 mg/dm2.",
            "Die Gesamtmigration lag unter 10 mg/dm2.",
            "German",
        )

        self.assertTrue(polish_issues)
        self.assertEqual(german_issues, [])

    def test_non_polish_prompt_terms_are_marked_as_concept_anchors(self) -> None:
        glossary = load_glossary("translator/domain/glossary.yaml")

        prompt_terms = glossary.terms_for_prompt("German")

        self.assertIn("domain concept anchor", prompt_terms)
        self.assertIn("approved Polish equivalent", prompt_terms)

    def test_polish_language_detection_accepts_ui_variants(self) -> None:
        self.assertTrue(is_polish_language("Polish"))
        self.assertTrue(is_polish_language("Polski"))
        self.assertTrue(is_polish_language("pl"))
        self.assertFalse(is_polish_language("Portuguese"))


if __name__ == "__main__":
    unittest.main()
