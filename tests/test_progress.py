from __future__ import annotations

import logging
import unittest

from translator.debug import LOGGER_NAME
from translator.progress import emit_progress


class ProgressTests(unittest.TestCase):
    def test_progress_callback_error_does_not_interrupt_workflow(self) -> None:
        def broken_callback(event: dict) -> None:  # noqa: ARG001
            raise RuntimeError("streamlit render failed")

        logger = logging.getLogger(LOGGER_NAME)
        previous_disabled = logger.disabled
        logger.disabled = True
        try:
            emit_progress(
                broken_callback,
                stage="translate",
                message="Segment przetłumaczony",
                current=1,
                total=2,
                segment_id="s1",
            )
        finally:
            logger.disabled = previous_disabled

    def test_progress_event_preserves_extra_fields(self) -> None:
        events: list[dict] = []

        emit_progress(
            events.append,
            stage="pipeline",
            message="Pipeline: segment po recenzji",
            current=323,
            total=564,
            segment_id="p002-b0017",
            translations_done=282,
            translations_total=282,
            reviews_done=41,
            reviews_total=282,
        )

        self.assertEqual(events[0]["translations_done"], 282)
        self.assertEqual(events[0]["translations_total"], 282)
        self.assertEqual(events[0]["reviews_done"], 41)
        self.assertEqual(events[0]["reviews_total"], 282)


if __name__ == "__main__":
    unittest.main()
