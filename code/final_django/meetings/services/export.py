import math
import re
from io import BytesIO
from pathlib import Path
from typing import Dict

from django.conf import settings
from django.http import HttpResponse
from django.utils.html import strip_tags
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

KOREAN_FONT_NAME = settings.KOREAN_FONT_NAME


def register_korean_font():
    """한글 폰트(맑은 고딕)가 미등록 상태라면 ReportLab에 등록한다."""
    if KOREAN_FONT_NAME in pdfmetrics.getRegisteredFontNames():
        return

    font_path: Path = settings.KOREAN_FONT_PATH
    if not font_path.exists():
        raise FileNotFoundError(f"Korean font file not found: {font_path}")

    pdfmetrics.registerFont(TTFont(KOREAN_FONT_NAME, str(font_path)))


def parse_minutes_sections(html: str) -> Dict[str, str]:
    """meeting.meeting_notes 에 저장된 HTML에서 data-minutes-section 별 텍스트 추출."""
    if not html:
        return {}

    pattern = re.compile(
        r'<div[^>]*data-minutes-section="(?P<key>[^"]+)"[^>]*>(?P<body>.*?)</div>',
        re.DOTALL | re.IGNORECASE,
    )

    sections: Dict[str, str] = {}
    for match in pattern.finditer(html):
        key = match.group("key")
        body_html = match.group("body")
        text = strip_tags(body_html).replace("&nbsp;", " ")
        lines = [ln.rstrip() for ln in text.splitlines()]
        sections[key] = "\n".join(ln for ln in lines if ln.strip())

    return sections


def extract_major_agenda(html: str) -> str:
    """회의록 HTML 안에서 '주요안건' 행 아래 minutes-editbox 내용을 우선 추출."""
    if not html:
        return ""

    pattern = re.compile(
        r'<th[^>]*>\s*주요안건\s*</th>.*?<div[^>]*class="[^"]*minutes-editbox[^"]*"[^>]*>(?P<body>.*?)</div>',
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(html)
    if not match:
        return ""

    return strip_tags(match.group("body")).replace("&nbsp;", " ").strip()


def _clean_section(text: str, header_keywords) -> str:
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)

    if lines:
        first = lines[0].replace(" ", "")
        if all(keyword in first for keyword in header_keywords):
            lines.pop(0)

    return "\n".join(lines).strip()


def _first_non_empty_line(text: str) -> str:
    for ln in (text or "").splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def _prepare_minutes_data(meeting):
    attendees_qs = meeting.attendees.select_related("user", "user__dept").all()
    attendees_list = list(attendees_qs)
    attendees_count = len(attendees_list)

    sections = parse_minutes_sections(meeting.meeting_notes or "")
    contents_text = _clean_section(sections.get("contents", ""), ["회의", "내용"])
    results_text = _clean_section(sections.get("results", ""), ["회의", "결과"])
    todos_text = _clean_section(sections.get("todos", ""), ["해야", "할", "일"])
    base_text = _clean_section(sections.get("base", ""), ["기본", "정보"])

    main_agenda = extract_major_agenda(meeting.meeting_notes or "")
    if not main_agenda:
        main_agenda = _first_non_empty_line(base_text) or _first_non_empty_line(contents_text)

    if hasattr(meeting, "host") and meeting.host:
        host_name = meeting.host.name
    else:
        host_name = getattr(meeting, "responsible_name", "")

    if meeting.meeting_notes:
        raw_html = (
            meeting.meeting_notes
            .replace("<br>", "\n")
            .replace("<br/>", "\n")
            .replace("<br />", "\n")
        )
        text_body = strip_tags(raw_html).replace("&nbsp;", " ")
    elif meeting.summary:
        text_body = meeting.summary
    elif meeting.transcript:
        text_body = meeting.transcript
    else:
        text_body = "회의 내용이 아직 등록되지 않았습니다."

    return {
        "attendees_list": attendees_list,
        "attendees_count": attendees_count,
        "sections": sections,
        "contents_text": contents_text,
        "results_text": results_text,
        "todos_text": todos_text,
        "main_agenda": main_agenda,
        "agenda": meeting.title or "",
        "meeting_dt_str": (
            meeting.meet_date_time.strftime("%Y.%m.%d %H:%M")
            if meeting.meet_date_time
            else ""
        ),
        "place": meeting.place or "",
        "host_name": host_name,
        "text_body": text_body,
    }


def _wrap_line_by_width(line: str, max_width: float, font: str, font_size: int = 10):
    """주어진 폭 안에 단어 단위로 줄바꿈. 너무 긴 단어는 폭에 맞게 강제 분리."""
    words = line.split()
    if not words:
        return [""]
    lines_local = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if pdfmetrics.stringWidth(candidate, font, font_size) <= max_width:
            current = candidate
        else:
            lines_local.append(current)
            current = word
    if pdfmetrics.stringWidth(current, font, font_size) > max_width:
        tmp = current
        while pdfmetrics.stringWidth(tmp, font, font_size) > max_width:
            cut = len(tmp)
            while cut > 0 and pdfmetrics.stringWidth(tmp[:cut], font, font_size) > max_width:
                cut -= 1
            if cut <= 0:
                break
            lines_local.append(tmp[:cut])
            tmp = tmp[cut:]
        if tmp:
            lines_local.append(tmp)
    else:
        lines_local.append(current)
    return lines_local


def _render_pdf_minutes(meeting, data: dict) -> bytes:
    register_korean_font()

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    x_margin = 50
    top_margin = height - 50
    bottom_margin = 40
    line_height = 14
    font = KOREAN_FONT_NAME

    sections = data["sections"]
    has_base = "base" in sections
    has_contents = "contents" in sections
    has_results = "results" in sections
    has_todos = "todos" in sections
    has_attendees_section = "attendees" in sections

    attendees_list = data["attendees_list"]
    attendees_count = data["attendees_count"]

    y = top_margin
    p.setFont(font, 18)
    p.drawString(x_margin, y, meeting.title or "회의록")
    y -= 40

    p.setFont(font, 11)
    x_left = x_margin
    x_right = width - x_margin
    center_x = (x_left + x_right) / 2.0

    header_row_h = 24
    label_row_h = 24
    attend_header_h = 24
    attend_row_h = 24
    rows_per_side = max(4, math.ceil(attendees_count / 2))
    attend_rows_h = rows_per_side * attend_row_h

    table_top_y = y
    current_top = table_top_y

    if has_base:
        x_label = x_left + 80
        x_right_block = x_right - 120
        header_top = current_top

        for i in range(5):
            y_line = header_top - header_row_h * i
            p.line(x_left, y_line, x_right, y_line)

        p.line(x_left, header_top, x_left, header_top - 4 * header_row_h)
        p.line(x_right, header_top, x_right, header_top - 4 * header_row_h)
        for i in range(4):
            p.line(x_label, header_top - i * header_row_h, x_label, header_top - (i + 1) * header_row_h)
        p.line(x_right_block, header_top - header_row_h, x_right_block, header_top - 3 * header_row_h)

        def header_text_y(row_idx: int) -> float:
            return header_top - header_row_h * row_idx - header_row_h + 8

        p.drawString(x_left + 5, header_text_y(0), "제 목")
        p.drawString(x_left + 85, header_text_y(0), data["agenda"])

        p.drawString(x_left + 5, header_text_y(1), "일 시")
        p.drawString(x_left + 85, header_text_y(1), data["meeting_dt_str"])
        host_label = f"주최자명 {data['host_name']}" if data["host_name"] else "주최자명"
        p.drawString(x_right_block + 5, header_text_y(1), host_label)

        p.drawString(x_left + 5, header_text_y(2), "장 소")
        p.drawString(x_left + 85, header_text_y(2), data["place"])
        p.drawString(x_right_block + 5, header_text_y(2), f"참석인원  {attendees_count}")

        p.drawString(x_left + 5, header_text_y(3), "주요안건")
        if data["main_agenda"]:
            agenda_box_width = x_right - (x_left + 80) - 10
            agenda_y = header_text_y(3)
            for line in _wrap_line_by_width(data["main_agenda"], agenda_box_width, font, font_size=11):
                p.drawString(x_left + 85, agenda_y, line)
                agenda_y -= line_height

        current_top = header_top - 4 * header_row_h

    section_specs = [
        (has_contents, "회의 내용", 200, data["contents_text"]),
        (has_results, "회의 결과", 200, data["results_text"]),
        (has_todos, "해야 할 일", 150, data["todos_text"]),
    ]
    section_boxes = []
    for enabled, title, box_height, body_text in section_specs:
        if not enabled:
            continue
        title_bottom = current_top - label_row_h
        p.line(x_left, title_bottom, x_right, title_bottom)
        p.drawCentredString(center_x, title_bottom + 8, title)

        box_top = title_bottom
        box_bottom = box_top - box_height
        p.line(x_left, box_bottom, x_right, box_bottom)
        current_top = box_bottom
        section_boxes.append((box_top, box_bottom, body_text))

    p.setFont(font, 10)
    usable_width = (x_right - x_left) - 10
    for box_top, box_bottom, body_text in section_boxes:
        y_pos = box_top - 14
        stopped = False
        for raw_line in (body_text or "").splitlines():
            wrapped = _wrap_line_by_width(raw_line, usable_width, font, font_size=10)
            for line in wrapped:
                if not line and len(wrapped) == 1:
                    continue
                if y_pos < box_bottom + line_height:
                    stopped = True
                    break
                p.drawString(x_left + 5, y_pos, line)
                y_pos -= line_height
            if stopped:
                break

    p.setFont(font, 11)

    if has_attendees_section or not sections:
        attend_label_top = current_top
        attend_label_bottom = attend_label_top - label_row_h
        attend_block_height = label_row_h + attend_header_h + attend_rows_h

        if attend_label_top - attend_block_height < bottom_margin:
            p.line(x_left, table_top_y, x_left, current_top)
            p.line(x_right, table_top_y, x_right, current_top)
            p.showPage()
            p.setFont(font, 11)
            current_top = top_margin
            table_top_y = current_top
            attend_label_top = current_top
            attend_label_bottom = attend_label_top - label_row_h

        p.line(x_left, attend_label_bottom, x_right, attend_label_bottom)
        p.drawCentredString(center_x, attend_label_bottom + 8, "참석자")

        attend_header_top = attend_label_bottom
        attend_header_bottom = attend_header_top - attend_header_h
        p.line(x_left, attend_header_bottom, x_right, attend_header_bottom)

        col_w = (x_right - x_left) / 6.0
        x_cols = [x_left + col_w * i for i in range(7)]

        attend_table_bottom = attend_header_bottom - attend_rows_h
        for xv in x_cols:
            p.line(xv, attend_header_top, xv, attend_table_bottom)

        header_y = attend_header_bottom + 5
        for idx, label in enumerate(["소 속", "성 명", "서 명", "소 속", "성 명", "서 명"]):
            p.drawString(x_cols[idx] + 5, header_y, label)

        row_top = attend_header_bottom
        for _ in range(rows_per_side):
            row_top -= attend_row_h
            p.line(x_left, row_top, x_right, row_top)

        outer_bottom = attend_table_bottom
        p.line(x_left, table_top_y, x_left, outer_bottom)
        p.line(x_right, table_top_y, x_right, outer_bottom)

        for row_idx in range(rows_per_side):
            row_text_y = attend_header_bottom - attend_row_h * row_idx - attend_row_h + 5
            for col_offset, idx in ((0, row_idx), (3, row_idx + rows_per_side)):
                if idx < len(attendees_list):
                    att = attendees_list[idx]
                    dept = att.user.dept.dept_name if att.user.dept else ""
                    p.drawString(x_cols[col_offset] + 5, row_text_y, dept)
                    p.drawString(x_cols[col_offset + 1] + 5, row_text_y, att.user.name)
                p.drawString(x_cols[col_offset + 2] + 5, row_text_y, "(인)")
    else:
        p.line(x_left, table_top_y, x_left, current_top)
        p.line(x_right, table_top_y, x_right, current_top)

    p.showPage()
    p.save()

    pdf_value = buffer.getvalue()
    buffer.close()
    return pdf_value


def _render_docx_minutes(meeting, data: dict) -> bytes:
    doc = Document()
    doc.add_heading(meeting.title or "회의록", level=1)

    info_rows = [
        ("제목", data["agenda"] or "-"),
        ("일시", data["meeting_dt_str"] or "-"),
        ("장소", data["place"] or "-"),
        ("주최자", data["host_name"] or "-"),
        ("주요안건", data["main_agenda"] or "-"),
        ("참석자 수", str(data["attendees_count"])),
    ]
    info_table = doc.add_table(rows=len(info_rows), cols=2)
    info_table.style = "Table Grid"
    for idx, (label, value) in enumerate(info_rows):
        cells = info_table.rows[idx].cells
        cells[0].text = label
        cells[1].text = value

    def add_doc_section(title: str, text: str):
        cleaned = (text or "").strip()
        if not cleaned:
            return
        doc.add_paragraph("")
        doc.add_heading(title, level=2)
        for ln in cleaned.splitlines():
            doc.add_paragraph(ln)

    add_doc_section("회의 내용", data["contents_text"])
    add_doc_section("회의 결과", data["results_text"])
    add_doc_section("해야 할 일", data["todos_text"])

    doc.add_paragraph("")
    doc.add_heading("참석자", level=2)
    attendees_list = data["attendees_list"]
    if attendees_list:
        rows = max(1, math.ceil(len(attendees_list) / 2))
        table = doc.add_table(rows=rows + 1, cols=4)
        table.style = "Table Grid"
        header_cells = table.rows[0].cells
        for idx, label in enumerate(["소 속", "성 명", "소 속", "성 명"]):
            header_cells[idx].text = label

        for row_idx in range(rows):
            row_cells = table.rows[row_idx + 1].cells
            for col_offset, idx in ((0, row_idx), (2, row_idx + rows)):
                if idx < len(attendees_list):
                    att = attendees_list[idx]
                    dept = att.user.dept.dept_name if att.user.dept else ""
                    row_cells[col_offset].text = dept
                    row_cells[col_offset + 1].text = att.user.name
    else:
        doc.add_paragraph("참석자 정보가 없습니다.")

    if not meeting.meeting_notes and data["text_body"]:
        doc.add_paragraph("")
        doc.add_heading("회의 내용", level=2)
        for line in data["text_body"].splitlines():
            doc.add_paragraph(line)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def build_minutes_download_response(meeting, fmt: str) -> HttpResponse:
    """요청 포맷(pdf/docx)에 맞는 회의록 다운로드 HttpResponse를 생성한다."""
    if fmt not in ("pdf", "docx"):
        return HttpResponse("invalid format", status=400)

    data = _prepare_minutes_data(meeting)
    if fmt == "pdf":
        payload = _render_pdf_minutes(meeting, data)
        filename = f"meeting_{meeting.meeting_id}_minutes.pdf"
        content_type = "application/pdf"
    else:
        payload = _render_docx_minutes(meeting, data)
        filename = f"meeting_{meeting.meeting_id}_minutes.docx"
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    response = HttpResponse(payload, content_type=content_type)
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response
