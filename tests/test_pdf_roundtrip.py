from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pypdf import PdfReader

from translator.pdf.output_validator import verify_output_pdf
from translator.pdf.parser import PDFParser
from translator.pdf.renderer import render_translated_pdf
from translator.schemas import TranslationResult


class PDFRoundtripTests(unittest.TestCase):
    def test_parse_render_verify_simple_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "source.pdf"
            output = temp / "translated.pdf"
            _make_sample_pdf(source)

            segments, metadata = PDFParser().extract(source)
            self.assertEqual(metadata["pages"], 1)
            self.assertGreaterEqual(len(segments), 2)

            translations = {
                segment.segment_id: TranslationResult(
                    segment_id=segment.segment_id,
                    translated_text=segment.source_text
                    .replace("Declaration of compliance", "Deklaracja zgodności")
                    .replace("Overall migration was", "Migracja globalna była")
                    .replace("below", "poniżej"),
                    confidence="high",
                )
                for segment in segments
            }

            render_translated_pdf(segments, translations, output)
            verification = verify_output_pdf(output, translations)

            self.assertTrue(output.exists())
            self.assertTrue(verification.ok, verification.messages)

    def test_overlay_render_preserves_source_page_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "source.pdf"
            output = temp / "translated_overlay.pdf"
            _make_sample_pdf(source)

            segments, _ = PDFParser().extract(source)
            translations = {
                segment.segment_id: TranslationResult(
                    segment_id=segment.segment_id,
                    translated_text=segment.source_text.replace(
                        "Declaration of compliance",
                        "Deklaracja zgodności",
                    ),
                    confidence="high",
                )
                for segment in segments
            }

            render_translated_pdf(segments, translations, output, source_pdf_path=source)

            source_reader = PdfReader(str(source))
            output_reader = PdfReader(str(output))
            self.assertEqual(len(output_reader.pages), len(source_reader.pages))
            self.assertEqual(output_reader.pages[0].mediabox, source_reader.pages[0].mediabox)
            self.assertIn("Deklaracja zgodności", output_reader.pages[0].extract_text())


def _make_sample_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(72, 760, "Declaration of compliance")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(72, 730, "Overall migration was below 10 mg/dm2.")
    pdf.drawString(72, 710, "No specific migration was detected.")
    pdf.save()


if __name__ == "__main__":
    unittest.main()
