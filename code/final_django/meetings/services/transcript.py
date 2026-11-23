import ast
import json
import re
from datetime import date, datetime

from django.db import transaction
from django.utils.html import escape

from meetings.models import Task
from users.models import User


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```") and s.endswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s


def _parse_raw_transcript(raw_transcript: str):
    try:
        return json.loads(raw_transcript)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(raw_transcript)
        except (ValueError, SyntaxError):
            return None


def transcript_to_plain_text(raw_transcript: str) -> str:
    """
    DB에 저장된 transcript(plain 문자열 혹은 JSON 리스트 형태)를
    모델에 넘길 수 있는 평문으로 변환한다.
    """
    if not raw_transcript:
        return ""

    parsed = _parse_raw_transcript(raw_transcript)
    if isinstance(parsed, list):
        lines = []
        for segment in parsed:
            if isinstance(segment, dict):
                for speaker, text in segment.items():
                    lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    return str(raw_transcript)


def normalize_tasks(full_tasks):
    """모델에서 내려준 태스크를 Task 모델에 저장할 수 있는 간단한 문자열 리스트로 정규화."""
    results = []
    if isinstance(full_tasks, str):
        for line in full_tasks.splitlines():
            line = line.strip(" -•\t")
            if line:
                results.append(line)
        return results

    if isinstance(full_tasks, list):
        for item in full_tasks:
            text = ""
            if isinstance(item, dict):
                who = item.get("who") or item.get("owner") or item.get("speaker")
                what = (
                    item.get("what")
                    or item.get("task")
                    or item.get("content")
                    or item.get("task_content")
                )
                when = item.get("when") or item.get("due_date")
                parts = [p for p in [who, what, when] if p]
                text = " - ".join(parts)
            elif isinstance(item, str):
                text = item.strip()
            if text:
                results.append(text)
    return results


def parse_due_date(due_str: str):
    """문자열로 내려온 기한을 DateField에 맞게 파싱. 실패 시 None."""
    if not due_str or not isinstance(due_str, str):
        return None
    due_str = due_str.strip()
    try:
        return date.fromisoformat(due_str)
    except Exception:
        pass
    for fmt in ("%Y.%m.%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(due_str, fmt).date()
        except Exception:
            pass
    return None


def extract_structured_tasks(full_tasks):
    """full_tasks(list/dict/str)에서 description/assignee/due를 뽑아낸 리스트 반환."""
    tasks = []
    if isinstance(full_tasks, str):
        s = _strip_code_fence(full_tasks)
        try:
            full_tasks = json.loads(s)
        except Exception:
            pass

    if isinstance(full_tasks, dict) and "tasks" in full_tasks:
        full_tasks = full_tasks.get("tasks")

    if isinstance(full_tasks, list):
        for item in full_tasks:
            if isinstance(item, dict):
                desc = (
                    item.get("description")
                    or item.get("what")
                    or item.get("task")
                    or item.get("content")
                    or item.get("task_content")
                )
                assignee_name = (
                    item.get("assignee")
                    or item.get("who")
                    or item.get("owner")
                    or item.get("speaker")
                )
                due_raw = (
                    item.get("due")
                    or item.get("when")
                    or item.get("due_text")
                )
                if isinstance(due_raw, str) and due_raw.strip().lower() in ("*", "null", "none", ""):
                    due_raw = ""
                due_date_val = item.get("due_date")
                parsed_due_date = None
                if due_raw:
                    parsed_due_date = parse_due_date(due_raw)
                elif due_date_val and str(due_date_val).lower() not in ("null", "none"):
                    parsed_due_date = parse_due_date(str(due_date_val))
                tasks.append(
                    {
                        "description": desc if desc is not None else "",
                        "assignee_name": assignee_name if assignee_name else "",
                        "due_date": parsed_due_date,
                        "due_raw": due_raw or "",
                    }
                )
            elif isinstance(item, str):
                tasks.append(
                    {
                        "description": item,
                        "assignee_name": "",
                        "due_date": None,
                        "due_raw": "",
                    }
                )
    elif isinstance(full_tasks, str) and full_tasks.strip():
        tasks.append(
            {
                "description": full_tasks.strip(),
                "assignee_name": "",
                "due_date": None,
                "due_raw": "",
            }
        )
    return tasks


def normalize_summary_text(full_summary):
    """
    full_summary가 문자열(코드펜스 포함 가능) 또는 dict/list로 올 때
    구조를 최대한 유지한 JSON 문자열로 반환.
    """
    if full_summary is None:
        return ""
    if isinstance(full_summary, str):
        s = _strip_code_fence(full_summary)
        try:
            parsed = json.loads(s)
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            return s
        except Exception:
            return s
    if isinstance(full_summary, (dict, list)):
        try:
            return json.dumps(full_summary, ensure_ascii=False, indent=2)
        except Exception:
            return str(full_summary)
    return str(full_summary)


def stringify_agenda_summary(summary_val):
    """
    Agenda summary가 dict로 내려올 때 보기 좋은 문자열로 변환한다.
    - agenda_summary 필드를 우선 사용
    - 그 외 주요 필드(who/what/when/where/why/how/how_much/how_many)를 라벨과 함께 표시
    """
    if summary_val is None:
        return ""
    if isinstance(summary_val, str):
        return summary_val.strip()
    if isinstance(summary_val, dict):
        agenda_summary = summary_val.get("agenda_summary")
        if agenda_summary:
            return str(agenda_summary).strip()
        keys = ["who", "what", "when", "where", "why", "how", "how_much", "how_many"]
        parts = [f"{key}: {summary_val[key]}" for key in keys if summary_val.get(key)]
        if parts:
            return " | ".join(parts)
        try:
            return json.dumps(summary_val, ensure_ascii=False)
        except Exception:
            return str(summary_val)
    try:
        return json.dumps(summary_val, ensure_ascii=False)
    except Exception:
        return str(summary_val)


def parse_summary_agendas(summary_text):
    """저장된 summary_text(JSON 문자열 예상)에서 agendas 배열을 파싱해 리스트 반환."""
    if not summary_text:
        return []
    try:
        text = _strip_code_fence(summary_text) if isinstance(summary_text, str) else summary_text
        parsed = json.loads(text)
    except Exception:
        parsed = summary_text if isinstance(summary_text, dict) else None
    if not isinstance(parsed, dict):
        return []
    agendas = parsed.get("agendas")
    if not isinstance(agendas, list):
        return []
    result = []
    for item in agendas:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "agenda": (item.get("agenda") or "").strip(),
                "agenda_description": (item.get("agenda_description") or "").strip(),
                "summary": stringify_agenda_summary(item.get("summary")),
            }
        )
    return result


def format_agendas_for_minutes(agendas):
    """회의록 기본값으로 넣기 위한 간단한 텍스트 형태로 변환."""
    if not agendas:
        return ""
    lines = []
    for idx, ag in enumerate(agendas, 1):
        title = ag.get("agenda") or ""
        desc = ag.get("agenda_description") or ""
        summ = stringify_agenda_summary(ag.get("summary"))
        parts = [p for p in [title, desc, summ] if p]
        if parts:
            lines.append(f"{idx}. " + " | ".join(parts))
    return "\n".join(lines)


def task_to_display(task):
    """
    Task 객체에서 화면에 표시할 who/what/when 정보를 추출.
    task_content 이 JSON(dict) 형태이면 description/due_text/assignee 사용, 아니면 문자열 그대로.
    """
    what = task.task_content or ""
    when_text = ""
    who_text_from_json = ""
    assignee_id_from_json = None
    try:
        parsed = json.loads(task.task_content)
        if isinstance(parsed, dict):
            target = (
                parsed["tasks"][0]
                if "tasks" in parsed and isinstance(parsed["tasks"], list) and parsed["tasks"]
                else parsed
            )
            what = target.get("description") or what
            when_text = target.get("due") or target.get("due_text") or ""
            who_text_from_json = target.get("assignee") or ""
            assignee_id_from_json = target.get("assignee_id")
    except Exception:
        pass

    who_text = "직접입력"
    if task.assignee:
        who_text = task.assignee.name
        if getattr(task.assignee, "dept", None):
            who_text += f" ({task.assignee.dept.dept_name})"
    elif who_text_from_json:
        who_text = who_text_from_json

    return {
        "id": getattr(task, "task_id", None),
        "who": who_text,
        "what": what,
        "when": when_text or "직접입력",
        "assignee_id": task.assignee_id or assignee_id_from_json,
    }


def plain_text_to_structured(text: str):
    """'발화자: 내용' 형태의 평문을 구조화 목록으로 변환한다."""
    segments = []
    speakers = set()
    if not text:
        return segments, speakers
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        speaker, value = line.split(":", 1)
        speaker = speaker.strip()
        value = value.strip()
        if speaker and value:
            segments.append({speaker: value})
            speakers.add(speaker)
    return segments, speakers


def render_transcript_html(raw_transcript: str) -> str:
    """
    상세 화면에서 쓸 전문 HTML을 생성한다.
    - 리스트(JSON/리터럴)인 경우: 각 발화자:내용을 굵게 표시
    - 평문에서도 '화자: 내용' 패턴이면 굵게 표시
    """
    if not raw_transcript:
        return ""

    html_lines = []
    parsed = _parse_raw_transcript(raw_transcript)

    def add_line(speaker, text):
        html_lines.append(
            f"<div><strong>{escape(str(speaker))}</strong>: {escape(str(text))}</div>"
        )

    if isinstance(parsed, list):
        for segment in parsed:
            if not isinstance(segment, dict):
                continue
            for speaker, text in segment.items():
                add_line(speaker, text)
        return "".join(html_lines)

    for line in raw_transcript.splitlines():
        if ":" in line:
            sp, txt = line.split(":", 1)
            add_line(sp.strip(), txt.strip())
        else:
            html_lines.append(f"<div>{escape(line)}</div>")
    return "".join(html_lines)


def save_sllm_result(meeting, payload):
    """SLLM 응답 payload에서 요약과 태스크를 파싱해 Meeting/Task 모델에 반영한다."""
    full_summary_raw = payload.get("full_summary") or payload.get("summary") or ""
    if not full_summary_raw and payload.get("agendas"):
        full_summary_raw = {"agendas": payload.get("agendas")}
    full_summary = normalize_summary_text(full_summary_raw)
    full_tasks = payload.get("full_tasks") or payload.get("tasks") or []
    tasks_structured = extract_structured_tasks(full_tasks)

    try:
        print(f"[SLLM] meeting_id={meeting.meeting_id} full_tasks={full_tasks}")
        print(f"[SLLM] meeting_id={meeting.meeting_id} tasks_structured={tasks_structured}")
    except Exception:
        pass

    with transaction.atomic():
        if full_summary is not None:
            meeting.summary = full_summary
            meeting.save(update_fields=["summary"])
        meeting.tasks.all().delete()
        if tasks_structured:
            task_objs = []
            for t in tasks_structured:
                desc = t.get("description") or ""
                assignee_name = (t.get("assignee_name") or "").strip()
                due_date = t.get("due_date")
                due_raw = t.get("due_raw") or ""

                assignee_obj = None
                if assignee_name:
                    try:
                        assignee_obj = User.objects.filter(name=assignee_name).first()
                    except Exception:
                        assignee_obj = None

                content_payload = {"description": desc}
                if assignee_name:
                    content_payload["assignee"] = assignee_name
                if due_raw:
                    content_payload["due"] = due_raw
                if due_date:
                    try:
                        content_payload["due_date"] = due_date.isoformat()
                    except Exception:
                        content_payload["due_date"] = str(due_date)

                task_objs.append(
                    Task(
                        meeting=meeting,
                        task_content=json.dumps(content_payload, ensure_ascii=False),
                        assignee=assignee_obj,
                        due_date=due_date,
                    )
                )
            Task.objects.bulk_create(task_objs)
