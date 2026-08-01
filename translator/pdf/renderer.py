from __future__ import annotations

from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from translator.debug import DebugTimer, log_debug
from translator.schemas import DocumentSegment, TranslationResult
from translator.utils import ensure_dir, normalize_ws


def render_translated_pdf(
    segments: list[DocumentSegment],
    translations: dict[str, TranslationResult],
    output_path: str | Path,
    title: str = "Translated technical document",
    source_pdf_path: str | Path | None = None,
) -> Path:
    if source_pdf_path and Path(source_pdf_path).exists():
        return _render_overlay_pdf(
            segments,
            translations,
            output_path,
            source_pdf_path=source_pdf_path,
            title=title,
        )
    return _render_reflow_pdf(segments, translations, output_path, title=title)


def _render_overlay_pdf(
    segments: list[DocumentSegment],
    translations: dict[str, TranslationResult],
    output_path: str | Path,
    *,
    source_pdf_path: str | Path,
    title: str,
) -> Path:
    from pypdf import PdfReader, PdfWriter

    source = Path(source_pdf_path)
    output = Path(output_path)
    ensure_dir(output.parent)
    regular_font, bold_font = _register_unicode_fonts()
    segments_by_page = _segments_by_page(segments)
    log_debug(
        "pdf.render.overlay.start",
        source_pdf_path=str(source),
        output_path=str(output),
        segments=len(segments),
        translations=len(translations),
        regular_font=regular_font,
        bold_font=bold_font,
    )

    with DebugTimer("pdf.render.overlay_build", output_path=str(output)):
        source_reader = PdfReader(str(source))
        overlay_buffer = BytesIO()
        overlay_canvas = canvas.Canvas(overlay_buffer)

        for page_index, source_page in enumerate(source_reader.pages, start=1):
            page_width = float(source_page.mediabox.width)
            page_height = float(source_page.mediabox.height)
            overlay_canvas.setPageSize((page_width, page_height))
            for segment in segments_by_page.get(page_index, []):
                _draw_overlay_segment(
                    overlay_canvas,
                    segment,
                    translations,
                    page_width=page_width,
                    page_height=page_height,
                    regular_font=regular_font,
                    bold_font=bold_font,
                )
            overlay_canvas.showPage()

        overlay_canvas.setTitle(title)
        overlay_canvas.save()
        overlay_buffer.seek(0)
        overlay_reader = PdfReader(overlay_buffer)

        writer = PdfWriter()
        for page_index, source_page in enumerate(source_reader.pages):
            writer.add_page(source_page)
            if page_index < len(overlay_reader.pages):
                writer.pages[page_index].merge_page(overlay_reader.pages[page_index])
        writer.add_metadata({"/Title": title, "/Producer": "TechnicAl overlay renderer"})
        with output.open("wb") as stream:
            writer.write(stream)

    log_debug("pdf.render.overlay.done", output_path=str(output), size_bytes=output.stat().st_size if output.exists() else None)
    return output


def _render_reflow_pdf(
    segments: list[DocumentSegment],
    translations: dict[str, TranslationResult],
    output_path: str | Path,
    *,
    title: str,
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


def _segments_by_page(segments: list[DocumentSegment]) -> dict[int, list[DocumentSegment]]:
    by_page: dict[int, list[DocumentSegment]] = {}
    for segment in sorted(segments, key=lambda item: item.order_index):
        by_page.setdefault(segment.page_number, []).append(segment)
    return by_page


def _draw_overlay_segment(
    pdf: canvas.Canvas,
    segment: DocumentSegment,
    translations: dict[str, TranslationResult],
    *,
    page_width: float,
    page_height: float,
    regular_font: str,
    bold_font: str,
) -> None:
    if not segment.bbox:
        return

    translation = translations.get(segment.segment_id)
    if not translation:
        return

    rendered_text = _clean_overlay_text(translation.translated_text)
    if not rendered_text:
        return

    if normalize_ws(rendered_text) == normalize_ws(segment.source_text):
        return

    x0, top, x1, bottom = _clamped_bbox(segment.bbox, page_width=page_width, page_height=page_height)
    original_width = max(1.0, x1 - x0)
    original_height = max(1.0, bottom - top)
    left_margin = max(18.0, x0)
    right_margin = left_margin if left_margin < page_width / 2 else 36.0
    available_width = max(original_width, page_width - x0 - right_margin)
    font_name = bold_font if segment.is_bold or segment.block_type == "heading" else regular_font
    font_size = _fit_single_line_font_size(
        rendered_text,
        font_name=font_name,
        start_size=_overlay_font_size(segment),
        max_width=available_width,
    )

    pad_x = 1.5
    pad_y = 1.2
    cover_x = max(0.0, x0 - pad_x)
    cover_top = max(0.0, top - pad_y)
    cover_bottom = min(page_height, bottom + pad_y)
    cover_width = min(page_width - cover_x, max(original_width + 2 * pad_x, available_width + 2 * pad_x))
    cover_height = cover_bottom - cover_top
    cover_y = page_height - cover_bottom

    pdf.setFillColor(colors.white)
    pdf.rect(cover_x, cover_y, cover_width, cover_height, stroke=0, fill=1)
    pdf.setFillColor(colors.black)
    pdf.setFont(font_name, font_size)
    baseline_y = page_height - bottom + max(1.0, (original_height - font_size) / 2)
    pdf.drawString(x0, baseline_y, rendered_text)


def _clamped_bbox(
    bbox: tuple[float, float, float, float],
    *,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    x0, top, x1, bottom = bbox
    return (
        min(max(float(x0), 0.0), page_width),
        min(max(float(top), 0.0), page_height),
        min(max(float(x1), 0.0), page_width),
        min(max(float(bottom), 0.0), page_height),
    )


def _clean_overlay_text(text: str) -> str:
    cleaned = normalize_ws(text)
    cleaned = cleaned.replace("\uf0b7", "•")
    if cleaned.startswith("□ "):
        cleaned = "• " + cleaned[2:]
    return cleaned


def _overlay_font_size(segment: DocumentSegment) -> float:
    source_size = float(segment.font_size or 9.5)
    if segment.block_type == "heading":
        return min(max(source_size, 9.5), 13.5)
    if segment.block_type == "footer":
        return min(max(source_size, 6.5), 8.5)
    return min(max(source_size, 7.0), 11.0)


def _fit_single_line_font_size(
    text: str,
    *,
    font_name: str,
    start_size: float,
    max_width: float,
) -> float:
    font_size = start_size
    while font_size > 5.5 and pdfmetrics.stringWidth(text, font_name, font_size) > max_width:
        font_size -= 0.25
    return max(font_size, 5.5)


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
    return _register_unicode_fonts()[0]


def _register_unicode_fonts() -> tuple[str, str]:
    regular_font = _register_font_from_candidates(
        "TechnicalUnicode",
        [
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
    )
    bold_font = _register_font_from_candidates(
        "TechnicalUnicodeBold",
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ],
    )
    return regular_font, bold_font or regular_font


def _register_font_from_candidates(font_name: str, candidates: list[str]) -> str:
    candidates = [
        path for path in candidates
        if Path(path).exists()
    ]
    for path in candidates:
        if font_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(font_name, path))
        return font_name
    return "Helvetica"
