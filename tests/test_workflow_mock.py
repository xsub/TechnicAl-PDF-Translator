from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from translator.schemas import JobConfig
from translator.storage import JobStore
from translator.workflow import run_mvp_pipeline


class MockWorkflowTests(unittest.TestCase):
    def test_mock_pipeline_creates_pdf_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "source.pdf"
            output_dir = temp / "out"
            _make_source_pdf(source)

            state = run_mvp_pipeline(
                JobConfig(
                    source_pdf_path=str(source),
                    output_dir=str(output_dir),
                    translator_provider="mock",
                    reviewer_provider="mock",
                    require_human_review=False,
                )
            )

            self.assertIn(state["status"], {"completed", "completed_with_output_warnings"})
            self.assertTrue(Path(state["output_pdf_path"]).exists())
            self.assertTrue(Path(state["report_path"]).exists())
            self.assertFalse(state["output_verification"].placeholder_leaks)

            loaded = JobStore(temp / "jobs.db").load_latest_state()
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded["job_id"], state["job_id"])
            self.assertEqual(len(loaded["segments"]), len(state["segments"]))
            self.assertEqual(len(loaded["translations"]), len(state["translations"]))
            self.assertEqual(loaded["config"].translator_provider, "mock")


def _make_source_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(72, 760, "Declaration of compliance")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(72, 730, "Overall migration was below 10 mg/dm2.")
    pdf.drawString(72, 710, "No specific migration was detected.")
    pdf.save()


if __name__ == "__main__":
    unittest.main()
