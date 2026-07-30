from __future__ import annotations

import unittest
from types import SimpleNamespace

from translator.domain.glossary import DomainGlossary
from translator.llm.clients import _extract_token_usage, estimate_translation_request_tokens
from translator.schemas import DocumentSegment, JobConfig, TokenUsage, TranslationResult
from translator.translation_cache import copy_translation_from_cache


class TokenUsageTests(unittest.TestCase):
    def test_extracts_openai_usage_metadata(self) -> None:
        raw = SimpleNamespace(
            usage_metadata={
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
            }
        )

        usage = _extract_token_usage(raw, provider="openai", model="gpt-test", operation="translate")

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.provider, "openai")
        self.assertEqual(usage.model, "gpt-test")
        self.assertEqual(usage.operation, "translate")
        self.assertEqual(usage.input_tokens, 120)
        self.assertEqual(usage.output_tokens, 30)
        self.assertEqual(usage.total_tokens, 150)

    def test_extracts_response_metadata_token_usage(self) -> None:
        raw = SimpleNamespace(
            usage_metadata=None,
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 12,
                    "total_tokens": 52,
                }
            },
        )

        usage = _extract_token_usage(raw, provider="openai", model="gpt-test", operation="review")

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.input_tokens, 40)
        self.assertEqual(usage.output_tokens, 12)
        self.assertEqual(usage.total_tokens, 52)

    def test_extracts_token_usage_from_dict_raw_message(self) -> None:
        raw = {
            "response_metadata": {
                "usage": {
                    "prompt_tokens": 75,
                    "completion_tokens": 25,
                }
            }
        }

        usage = _extract_token_usage(raw, provider="openai", model="gpt-test", operation="translate")

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.input_tokens, 75)
        self.assertEqual(usage.output_tokens, 25)
        self.assertEqual(usage.total_tokens, 100)

    def test_extracts_anthropic_usage_metadata(self) -> None:
        raw = SimpleNamespace(
            usage_metadata={
                "input_tokens": 90,
                "output_tokens": 15,
            }
        )

        usage = _extract_token_usage(raw, provider="anthropic", model="claude-test", operation="review")

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.provider, "anthropic")
        self.assertEqual(usage.input_tokens, 90)
        self.assertEqual(usage.output_tokens, 15)
        self.assertEqual(usage.total_tokens, 105)

    def test_cache_copy_clears_token_usage_so_hits_do_not_count_again(self) -> None:
        segment = DocumentSegment(
            segment_id="s2",
            page_number=1,
            order_index=2,
            block_type="paragraph",
            source_text="Repeated footer",
        )
        cached = TranslationResult(
            segment_id="s1",
            translated_text="Powtarzalna stopka",
            token_usage=TokenUsage(
                provider="openai",
                model="gpt-test",
                operation="translate",
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
            ),
        )

        copied = copy_translation_from_cache(segment, cached, source_label="segment s1")

        self.assertEqual(copied.segment_id, "s2")
        self.assertIsNone(copied.token_usage)

    def test_estimates_translation_request_tokens(self) -> None:
        segment = DocumentSegment(
            segment_id="s1",
            page_number=1,
            order_index=1,
            block_type="paragraph",
            source_text="Overall migration was below 10 mg/dm2.",
        )
        config = JobConfig(source_pdf_path="dummy.pdf", translator_provider="openai")

        estimated_tokens = estimate_translation_request_tokens(segment, DomainGlossary(), config)

        self.assertGreater(estimated_tokens, 0)


if __name__ == "__main__":
    unittest.main()
