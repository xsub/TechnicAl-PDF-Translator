from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from translator.debug import DebugTimer, log_debug
from translator.schemas import DocumentSegment, TranslationResult
from translator.utils import ensure_dir


def render_translated_pdf(
    segments: list[DocumentSegment],
    translations: dict[str, TranslationResult],
    output_path: str | Path,
    title: str = "Translated technical document",
) -> Path:
    output = Path(output_path)
    ensure_dir(output.parent)
    font_name = _register_unicode_font()
    styles = _build_styles(font_name)
    log_debug(
        "pdf.render.start",
        output_path=str(output),
        segments=len(segments),
        translations=len(translations),
        font_name=font_name,
    )

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
    )

    story = []
    current_page = None
    pending_table: list[DocumentSegment] = []
    pending_table_id: str | None = None

    def flush_table() -> None:
        nonlocal pending_table, pending_table_id
        if not pending_table:
            return
        story.append(_make_table(pending_table, translations, styles["table_cell"], font_name))
        story.append(Spacer(1, 4 * mm))
        pending_table = []
        pending_table_id = None

    for segment in sorted(segments, key=lambda item: item.order_index):
        if current_page is None:
            current_page = segment.page_number
        elif segment.page_number != current_page:
            flush_table()
            story.append(PageBreak())
            current_page = segment.page_number

        if segment.block_type == "table_cell":
            if pending_table_id and pending_table_id != segment.table_id:
                flush_table()
            pending_table_id = segment.table_id
            pending_table.append(segment)
            continue

        flush_table()
        text = translations.get(segment.segment_id)
        rendered_text = text.translated_text if text else segment.source_text
        style = styles["heading"] if segment.block_type == "heading" else styles["body"]
        story.append(Paragraph(escape(rendered_text), style))
        story.append(Spacer(1, 2.2 * mm))

    flush_table()
    with DebugTimer("pdf.render.reportlab_build", output_path=str(output), flowables=len(story)):
        document.build(story)
    log_debug("pdf.render.done", output_path=str(output), size_bytes=output.stat().st_size if output.exists() else None)
    return output


def _make_table(
    cells: list[DocumentSegment],
    translations: dict[str, TranslationResult],
    cell_style: ParagraphStyle,
    font_name: str,
) -> Table:
    max_row = max((cell.row_index or 0) for cell in cells)
    max_col = max((cell.column_index or 0) for cell in cells)
    matrix: list[list[Paragraph]] = [
        [Paragraph("", cell_style) for _ in range(max_col + 1)]
        for _ in range(max_row + 1)
    ]

    for cell in cells:
        row = cell.row_index or 0
        col = cell.column_index or 0
        translation = translations.get(cell.segment_id)
        text = translation.translated_text if translation else cell.source_text
        matrix[row][col] = Paragraph(escape(text), cell_style)

    table = Table(matrix, repeatRows=1 if max_row > 0 else 0, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDEFF3")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#A7ABB4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _build_styles(font_name: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "TechnicalBody",
        parent=base["BodyText"],
        fontName=font_name,
        fontSize=9.5,
        leading=12.5,
        spaceAfter=2,
    )
    heading = ParagraphStyle(
        "TechnicalHeading",
        parent=base["Heading2"],
        fontName=font_name,
        fontSize=13,
        leading=16,
        spaceBefore=4,
        spaceAfter=5,
    )
    table_cell = ParagraphStyle(
        "TechnicalTableCell",
        parent=body,
        fontName=font_name,
        fontSize=8,
        leading=10,
    )
    return {"body": body, "heading": heading, "table_cell": table_cell}


def _register_unicode_font() -> str:
    candidates = [
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            font_name = "TechnicalUnicode"
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, path))
            return font_name
    return "Helvetica"
