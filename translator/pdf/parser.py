from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Any

from translator.debug import DebugTimer, log_debug, text_preview
from translator.schemas import DocumentSegment
from translator.utils import normalize_ws


class PDFParser:
    def extract(self, pdf_path: str | Path) -> tuple[list[DocumentSegment], dict[str, Any]]:
        path = Path(pdf_path)
        log_debug("pdf.extract.start", path=str(path), size_bytes=path.stat().st_size if path.exists() else None)
        try:
            with DebugTimer("pdf.extract.pdfplumber", path=str(path)):
                return self._extract_with_pdfplumber(path)
        except ImportError as exc:
            raise RuntimeError("Brakuje pdfplumber. Uruchom: python -m pip install pdfplumber") from exc

    def _extract_with_pdfplumber(self, path: Path) -> tuple[list[DocumentSegment], dict[str, Any]]:
        import pdfplumber

        segments: list[DocumentSegment] = []
        global_order = 0
        metadata: dict[str, Any] = {"source_file": str(path), "parser": "pdfplumber", "pages": 0}

        with pdfplumber.open(str(path)) as pdf:
            metadata["pages"] = len(pdf.pages)
            metadata["pdf_metadata"] = pdf.metadata or {}
            log_debug(
                "pdf.opened",
                path=str(path),
                pages=len(pdf.pages),
                metadata_keys=sorted((pdf.metadata or {}).keys()),
            )

            for page_index, page in enumerate(pdf.pages, start=1):
                page_started_count = len(segments)
                table_infos = _extract_tables(page, page_index)
                table_boxes = [info["bbox"] for info in table_infos]
                line_infos = _extract_lines_outside_tables(page, table_boxes)
                median_size = _median_font_size(line_infos)
                log_debug(
                    "pdf.page.extracted_raw",
                    page_number=page_index,
                    tables=len(table_infos),
                    lines=len(line_infos),
                    median_font_size=median_size,
                )

                page_items: list[tuple[float, int, DocumentSegment]] = []
                for local_index, line in enumerate(line_infos):
                    text = normalize_ws(line["text"])
                    if not text:
                        continue
                    block_type = _guess_block_type(text, line.get("font_size"), median_size)
                    segment = DocumentSegment(
                        segment_id=f"p{page_index:03d}-b{local_index:04d}",
                        page_number=page_index,
                        order_index=global_order,
                        block_type=block_type,
                        source_text=text,
                        bbox=line.get("bbox"),
                        font_name=line.get("font_name"),
                        font_size=line.get("font_size"),
                        is_bold=bool(line.get("is_bold")),
                    )
                    page_items.append((line["top"], local_index, segment))
                    global_order += 1

                for table_number, info in enumerate(table_infos, start=1):
                    table_id = f"p{page_index:03d}-t{table_number:02d}"
                    table = info["rows"]
                    base_top = info["bbox"][1]
                    headers = [normalize_ws(cell or "") for cell in table[0]] if table else []
                    for row_index, row in enumerate(table):
                        for column_index, cell in enumerate(row):
                            text = normalize_ws(cell or "")
                            if not text:
                                continue
                            segment = DocumentSegment(
                                segment_id=f"{table_id}-r{row_index:03d}-c{column_index:03d}",
                                page_number=page_index,
                                order_index=global_order,
                                block_type="table_cell",
                                source_text=text,
                                table_id=table_id,
                                row_index=row_index,
                                column_index=column_index,
                                column_header=headers[column_index] if row_index > 0 and column_index < len(headers) else None,
                            )
                            page_items.append((base_top + row_index * 0.01 + column_index * 0.0001, global_order, segment))
                            global_order += 1

                for _, _, item in sorted(page_items, key=lambda value: (value[0], value[1])):
                    segments.append(item.model_copy(update={"order_index": len(segments)}))
                page_segments = segments[page_started_count:]
                log_debug(
                    "pdf.page.segments",
                    page_number=page_index,
                    segments=len(page_segments),
                    first_segment_id=page_segments[0].segment_id if page_segments else None,
                    first_segment_preview=text_preview(page_segments[0].source_text) if page_segments else None,
                )

        segments = _attach_context(segments)
        log_debug("pdf.extract.done", path=str(path), segments=len(segments), pages=metadata["pages"])
        return segments, metadata


def _extract_tables(page: Any, page_number: int) -> list[dict[str, Any]]:
    tables = []
    try:
        found_tables = page.find_tables()
    except Exception:
        found_tables = []

    for table in found_tables:
        try:
            rows = table.extract()
        except Exception:
            rows = None
        if not rows:
            continue
        tables.append({"page": page_number, "bbox": table.bbox, "rows": rows})
    return tables


def _extract_lines_outside_tables(page: Any, table_boxes: list[tuple[float, float, float, float]]) -> list[dict[str, Any]]:
    try:
        words = page.extract_words(
            x_tolerance=2,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=["fontname", "size"],
        )
    except TypeError:
        words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False)

    outside_words = [
        word for word in words
        if not _point_inside_any_box(_word_center(word), table_boxes)
    ]

    outside_words.sort(key=lambda word: (round(float(word.get("top", 0)), 1), float(word.get("x0", 0))))
    lines: list[list[dict[str, Any]]] = []
    for word in outside_words:
        top = float(word.get("top", 0))
        if not lines or abs(float(lines[-1][0].get("top", 0)) - top) > 3:
            lines.append([word])
        else:
            lines[-1].append(word)

    line_infos: list[dict[str, Any]] = []
    for line in lines:
        line.sort(key=lambda word: float(word.get("x0", 0)))
        text = " ".join(str(word.get("text", "")) for word in line)
        font_sizes = [float(word.get("size", 0) or 0) for word in line]
        font_names = [str(word.get("fontname", "")) for word in line]
        bbox = (
            min(float(word.get("x0", 0)) for word in line),
            min(float(word.get("top", 0)) for word in line),
            max(float(word.get("x1", 0)) for word in line),
            max(float(word.get("bottom", 0)) for word in line),
        )
        line_infos.append(
            {
                "text": text,
                "top": bbox[1],
                "bbox": bbox,
                "font_size": median(font_sizes) if font_sizes else None,
                "font_name": font_names[0] if font_names else None,
                "is_bold": any("bold" in name.lower() for name in font_names),
            }
        )
    return line_infos


def _word_center(word: dict[str, Any]) -> tuple[float, float]:
    return (
        (float(word.get("x0", 0)) + float(word.get("x1", 0))) / 2,
        (float(word.get("top", 0)) + float(word.get("bottom", 0))) / 2,
    )


def _point_inside_any_box(point: tuple[float, float], boxes: list[tuple[float, float, float, float]]) -> bool:
    x, y = point
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in boxes)


def _median_font_size(lines: list[dict[str, Any]]) -> float | None:
    sizes = [float(line["font_size"]) for line in lines if line.get("font_size")]
    return median(sizes) if sizes else None


def _guess_block_type(text: str, font_size: float | None, median_size: float | None) -> str:
    if font_size and median_size and font_size >= median_size + 2:
        return "heading"
    if len(text) <= 90 and (text.isupper() or text.rstrip(":").istitle()):
        return "heading"
    if text.lstrip().startswith(("-", "•")):
        return "list_item"
    return "paragraph"


def _attach_context(segments: list[DocumentSegment]) -> list[DocumentSegment]:
    enriched = []
    for index, segment in enumerate(segments):
        preceding = segments[index - 1].source_text if index > 0 else None
        following = segments[index + 1].source_text if index + 1 < len(segments) else None
        enriched.append(
            segment.model_copy(
                update={
                    "preceding_context": preceding,
                    "following_context": following,
                    "order_index": index,
                }
            )
        )
    return enriched
