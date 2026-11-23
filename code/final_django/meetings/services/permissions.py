import json

from django.db.models import Prefetch

from meetings.models import Attendee, Meeting
from users.models import User


def get_privacy_from_domain(domain_value):
    """기존 BooleanField(domain) 호환 + legacy 문자열 처리."""
    if domain_value in [True, False]:
        return "private" if domain_value else "public"
    try:
        if isinstance(domain_value, str):
            parsed = json.loads(domain_value)
            if isinstance(parsed, dict):
                val = parsed.get("privacy")
                if isinstance(val, str) and val.lower() == "private":
                    return "private"
            if domain_value.strip().lower() == "private":
                return "private"
    except Exception:
        pass
    return "public"


def get_privacy(meeting: Meeting):
    if getattr(meeting, "private_yn", False):
        return "private"
    return get_privacy_from_domain(getattr(meeting, "domain", None))


def has_meeting_view_permission(
    meeting: Meeting,
    attendees_list,
    session_user_id,
    login_user_dept_id,
):
    """상세/다운로드 공통 접근 권한 체크."""
    is_host = session_user_id and str(meeting.host_id) == str(session_user_id)
    is_attendee = (
        session_user_id
        and any(str(a.user_id) == str(session_user_id) for a in attendees_list)
    )
    privacy = get_privacy(meeting)
    if privacy == "private":
        return bool(is_host or is_attendee)

    same_dept = False
    if login_user_dept_id:
        same_dept = any(
            getattr(a.user, "dept_id", None) == login_user_dept_id
            for a in attendees_list
        )
    return bool(is_host or is_attendee or same_dept)


def base_meeting_list_queryset():
    """회의 목록 조회용 공통 QuerySet(주최자 및 참석자 부서 prefetch 포함)."""
    return (
        Meeting.objects.select_related("host")
        .prefetch_related(
            Prefetch(
                "attendees",
                queryset=Attendee.objects.select_related("user", "user__dept"),
            )
        )
        .order_by("-meet_date_time")
    )


def serialize_meeting_row(meeting: Meeting, *, attendees=None, is_joined: bool, is_private=None):
    """회의 목록 템플릿에 전달할 단일 회의 행 딕셔너리를 생성한다."""
    if attendees is None:
        attendees = list(meeting.attendees.all())
    row = {
        "meeting_id": meeting.meeting_id,
        "title": meeting.title,
        "meet_date_time": meeting.meet_date_time,
        "place": meeting.place,
        "host_name": meeting.host.name,
        "attendee_count": len(attendees),
        "attendee_names": ", ".join(a.user.name for a in attendees),
        "is_joined": is_joined,
    }
    if is_private is not None:
        row["is_private"] = is_private
    return row


def build_meeting_list_context(meeting_qs, login_user_id=None):
    """열람 권한 필터링을 적용한 회의 목록 컨텍스트 데이터와 로그인 사용자를 반환한다."""
    meetings_data = []

    login_user = None
    login_user_dept_id = None
    if login_user_id:
        login_user = User.objects.select_related("dept").get(user_id=login_user_id)
        login_user_dept_id = getattr(login_user, "dept_id", None)

    for m in meeting_qs:
        attendees = list(m.attendees.all())
        is_host = str(m.host_id) == str(login_user_id) if login_user_id else False
        is_attendee = (
            any(str(a.user_id) == str(login_user_id) for a in attendees)
            if login_user_id and not is_host
            else False
        )
        is_joined = is_host or is_attendee

        if not has_meeting_view_permission(m, attendees, login_user_id, login_user_dept_id):
            continue

        privacy = get_privacy(m)
        meetings_data.append(
            serialize_meeting_row(
                m,
                attendees=attendees,
                is_joined=is_joined,
                is_private=(privacy == "private"),
            )
        )

    return meetings_data, login_user
