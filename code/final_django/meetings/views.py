import ast
import json
from datetime import date, timedelta
from urllib.parse import quote

import requests
from django.contrib import messages
from django.db import transaction
from django.db.models import Prefetch, Q
from django.http import FileResponse, Http404, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST
from django.views.generic import TemplateView

from core.views import LoginRequiredSessionMixin
from meetings.services.export import (
    build_minutes_download_response,
    extract_major_agenda,
    parse_minutes_sections,
)
from meetings.services.permissions import (
    base_meeting_list_queryset,
    build_meeting_list_context,
    has_meeting_view_permission,
    serialize_meeting_row,
)
from meetings.services.runpod_polling import (
    DOMAIN_API_MAP,
    extract_runpod_error,
    poll_runpod_job,
    start_runpod_job,
)
from meetings.services.transcript import (
    format_agendas_for_minutes,
    parse_due_date,
    parse_summary_agendas,
    render_transcript_html,
    save_sllm_result,
    task_to_display,
    transcript_to_plain_text,
)
from meetings.utils.runpod import (
    get_job_status,
    get_sllm,
    get_stt,
    start_sllm_job,
    start_stt_job,
    start_stt_job_from_bytes,
)
from meetings.utils.s3_upload import (
    get_local_audio_path,
    get_presigned_url,
    is_local_storage_key,
    upload_raw_file_bytes,
)
from users.models import Dept, User

from .models import Attendee, Meeting, S3File, Task


# ---------------------------------------------------------------------------
# 공통 뷰 헬퍼 / 믹스인
# ---------------------------------------------------------------------------

def _resolve_s3_file(meeting: Meeting):
    """meeting.record_url 또는 record_url_id(s3_key)로부터 S3File 객체를 조회한다."""
    if getattr(meeting, "record_url", None):
        try:
            return meeting.record_url
        except Exception:
            pass

    record_url_val = getattr(meeting, "record_url_id", None)
    if not record_url_val:
        return None

    return S3File.objects.filter(s3_key=record_url_val).first()


def _get_session_user_and_dept(request):
    session_user_id = request.session.get("login_user_id")
    if not session_user_id:
        return None, None, None
    user_obj = (
        User.objects.select_related("dept")
        .filter(user_id=session_user_id)
        .first()
    )
    dept_id = getattr(user_obj, "dept_id", None) if user_obj else None
    return session_user_id, user_obj, dept_id


def _is_meeting_host(request, meeting: Meeting) -> bool:
    if not meeting.host_id:
        return False
    host_user_id = str(meeting.host_id)
    session_user_id = request.session.get("login_user_id")
    if session_user_id and str(session_user_id) == host_user_id:
        return True
    if hasattr(request, "user") and request.user.is_authenticated:
        request_user_id = getattr(request.user, "user_id", None) or request.user.id
        if request_user_id and str(request_user_id) == host_user_id:
            return True
    return False


def _parse_json_body(request):
    try:
        return json.loads(request.body.decode("utf-8")), None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)


def _redirect_forbidden(request):
    referer = request.META.get("HTTP_REFERER")
    if referer:
        return redirect(referer)
    try:
        fallback = reverse("meetings:meeting_list_dept")
    except Exception:
        fallback = "/"
    return redirect(fallback)


class _MeetingAccessMixin(LoginRequiredSessionMixin):
    """회의 조회 권한(공개/비공개/부서)을 공통 검사하는 믹스인."""

    def _get_meeting_with_relations(self, meeting_id):
        meeting = (
            Meeting.objects.select_related("host")
            .prefetch_related("attendees__user__dept")
            .filter(pk=meeting_id)
            .first()
        )
        if not meeting:
            raise Http404()
        return meeting

    def _check_access(self, request, meeting):
        session_user_id, login_user_obj, login_user_dept_id = _get_session_user_and_dept(request)
        attendees_list = list(meeting.attendees.all())
        allowed = has_meeting_view_permission(
            meeting,
            attendees_list,
            session_user_id,
            login_user_dept_id,
        )
        return allowed, session_user_id, login_user_obj

    def get(self, request, *args, **kwargs):
        meeting_id = kwargs.get("meeting_id") or self.kwargs.get("meeting_id")
        meeting = self._get_meeting_with_relations(meeting_id)
        allowed, _, _ = self._check_access(request, meeting)
        if not allowed:
            return _redirect_forbidden(request)
        return super().get(request, *args, **kwargs)


class _BaseMeetingRenderingView(LoginRequiredSessionMixin, TemplateView):
    """STT/SLLM 대기 화면 공통 뷰."""

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        meeting_id = self.kwargs.get("meeting_id")
        meeting = get_object_or_404(Meeting, pk=meeting_id)
        context["meeting"] = meeting
        context["meeting_id"] = meeting_id
        return context


# ---------------------------------------------------------------------------
# 회의 목록 뷰
# ---------------------------------------------------------------------------

class MeetingListAllView(LoginRequiredSessionMixin, TemplateView):
    template_name = "meetings/meeting_list.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        login_user_id = self.request.session.get("login_user_id")
        meetings_data, login_user = build_meeting_list_context(
            base_meeting_list_queryset(),
            login_user_id,
        )
        context["login_user"] = login_user
        context["meetings"] = meetings_data
        context["meeting_list_type"] = "all"
        return context


class MeetingListMineView(LoginRequiredSessionMixin, TemplateView):
    template_name = "meetings/meeting_list.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["meeting_list_type"] = "mine"
        login_user_id = self.request.session.get("login_user_id")
        if not login_user_id:
            context["login_user"] = None
            context["meetings"] = []
            return context

        meeting_qs = (
            base_meeting_list_queryset()
            .filter(Q(host_id=login_user_id) | Q(attendees__user_id=login_user_id))
            .distinct()
        )
        meetings_data, login_user = build_meeting_list_context(meeting_qs, login_user_id)
        context["login_user"] = login_user
        context["meetings"] = meetings_data
        return context


class MeetingListDeptView(LoginRequiredSessionMixin, TemplateView):
    template_name = "meetings/meeting_list.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        login_user_id = self.request.session.get("login_user_id")
        login_user = None
        context["meeting_list_type"] = "dept"
        if login_user_id:
            login_user = User.objects.select_related("dept").get(user_id=login_user_id)

        dept_id = getattr(login_user, "dept_id", None)
        if dept_id:
            meeting_qs = (
                base_meeting_list_queryset()
                .exclude(Q(host_id=login_user_id) | Q(attendees__user_id=login_user_id))
                .filter(Q(attendees__user__dept_id=dept_id) | Q(host__dept_id=dept_id))
                .distinct()
            )
        else:
            meeting_qs = Meeting.objects.none()

        meetings_data = [serialize_meeting_row(m, is_joined=False) for m in meeting_qs]
        context["login_user"] = login_user
        context["meetings"] = meetings_data
        return context


# ---------------------------------------------------------------------------
# 회의 생성 / 녹음 / 전사 / 상세 뷰
# ---------------------------------------------------------------------------

class MeetingCreateView(LoginRequiredSessionMixin, TemplateView):
    template_name = "meetings/meeting_create.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        dept_qs = Dept.objects.prefetch_related(
            Prefetch("users", queryset=User.objects.order_by("name"))
        ).order_by("dept_name")
        context["departments"] = dept_qs

        login_user_id = self.request.session.get("login_user_id")
        if login_user_id:
            context["login_user"] = User.objects.get(user_id=login_user_id)
        return context

    def post(self, request, *args, **kwargs):
        attendee_ids = request.POST.getlist("attendees")
        domain_value = request.POST.get("domains") or ""
        privacy_private = request.POST.get("is_private") in ["on", "1", "true", True]
        if not attendee_ids:
            messages.error(request, "참석자를 최소 1명 이상 선택해 주세요.")
            context = self.get_context_data()
            context["form_title"] = request.POST.get("title", "")
            context["form_meet_date_time"] = request.POST.get("meet_date_time", "")
            context["form_place"] = request.POST.get("place", "")
            return render(request, self.template_name, context)

        title = request.POST.get("title")
        meet_date_time = request.POST.get("meet_date_time")
        place = request.POST.get("place")

        login_user_id = request.session.get("login_user_id")
        if not login_user_id:
            return redirect("login")

        host_user = User.objects.get(user_id=login_user_id)
        attendee_ids = [uid for uid in attendee_ids if str(uid) != str(login_user_id)]

        with transaction.atomic():
            meeting = Meeting.objects.create(
                title=title,
                meet_date_time=meet_date_time,
                place=place,
                host=host_user,
                transcript="",
                domain=domain_value,
                private_yn=privacy_private,
            )

        Attendee.objects.create(meeting=meeting, user=host_user)
        users = User.objects.filter(user_id__in=attendee_ids)
        Attendee.objects.bulk_create([Attendee(meeting=meeting, user=u) for u in users])

        return redirect("meetings:meeting_record", meeting_id=meeting.meeting_id)


class MeetingRecordView(_MeetingAccessMixin, TemplateView):
    template_name = "meetings/meeting_record.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        meeting_id = self.kwargs.get("meeting_id")
        meeting = Meeting.objects.get(pk=meeting_id)

        context["attendees"] = meeting.attendees.select_related("user").all()

        grace = timedelta(minutes=10)
        now = timezone.now()
        allow_recording = False
        try:
            if meeting.meet_date_time:
                allow_recording = meeting.meet_date_time >= (now - grace)
        except Exception:
            allow_recording = False

        context["allow_recording"] = allow_recording
        context["meeting"] = meeting
        context["meeting_id"] = meeting_id
        return context


class MeetingTranscriptView(_MeetingAccessMixin, TemplateView):
    template_name = "meetings/meeting_transcript.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        meeting_id = self.kwargs.get("meeting_id")
        context["meeting_id"] = meeting_id
        context["meeting"] = get_object_or_404(Meeting, pk=meeting_id)
        return context


@require_GET
def meeting_transcript_api(request, meeting_id):
    meeting = get_object_or_404(Meeting, pk=meeting_id)
    attendees_qs = meeting.attendees.select_related("user", "user__dept").all()

    raw_transcript = meeting.transcript or ""
    structured_transcript = []
    speakers = []
    transcript_plain = raw_transcript.replace("\r\n", "\n")

    if raw_transcript:
        try:
            parsed = json.loads(raw_transcript)
        except (ValueError, TypeError):
            try:
                parsed = ast.literal_eval(raw_transcript)
            except (ValueError, SyntaxError):
                parsed = None

        if isinstance(parsed, list):
            speaker_keys = set()
            plain_lines = []
            normalized_segments = []

            for segment in parsed:
                if not isinstance(segment, dict):
                    continue
                normalized_segment = {}
                for key, value in segment.items():
                    key_str = str(key)
                    value_str = str(value)
                    normalized_segment[key_str] = value_str
                    speaker_keys.add(key_str)
                    plain_lines.append(f"{key_str}: {value_str}")
                if normalized_segment:
                    normalized_segments.append(normalized_segment)

            if normalized_segments:
                structured_transcript = normalized_segments
                speakers = sorted(speaker_keys)
                transcript_plain = "\n".join(plain_lines)

    attendees_payload = [
        {
            "user_id": attendee.user_id,
            "name": attendee.user.name,
            "dept_name": attendee.user.dept.dept_name if attendee.user.dept else "",
        }
        for attendee in attendees_qs
    ]

    return JsonResponse(
        {
            "meeting_title": meeting.title,
            "transcript": raw_transcript,
            "transcript_plain": transcript_plain,
            "transcript_structured": structured_transcript,
            "record_url": str(meeting.record_url_id) if meeting.record_url_id else "",
            "attendees": attendees_payload,
            "speakers": speakers,
        }
    )


@require_POST
def meeting_transcript_save(request, meeting_id):
    meeting = get_object_or_404(Meeting, pk=meeting_id)
    if not _is_meeting_host(request, meeting):
        return JsonResponse(
            {"ok": False, "error": "전문을 저장할 권한이 없습니다."},
            status=403,
        )

    payload, err_response = _parse_json_body(request)
    if err_response is not None:
        return err_response

    transcript_text = (payload.get("transcript_text") or "").strip()
    transcript_structured = payload.get("transcript_structured")

    if transcript_structured:
        if not isinstance(transcript_structured, list):
            return JsonResponse({"ok": False, "error": "잘못된 전문 형식입니다."}, status=400)
        try:
            meeting.transcript = json.dumps(transcript_structured, ensure_ascii=False)
        except (TypeError, ValueError):
            return JsonResponse(
                {"ok": False, "error": "전문 데이터를 처리할 수 없습니다."},
                status=400,
            )
    elif transcript_text:
        meeting.transcript = transcript_text
    else:
        return JsonResponse(
            {"ok": False, "error": "저장할 전문 내용이 없습니다."},
            status=400,
        )

    meeting.save(update_fields=["transcript"])
    redirect_url = reverse("meetings:rendering_sllm", args=[meeting_id])
    return JsonResponse({"ok": True, "redirect_url": redirect_url})


class MeetingDetailView(_MeetingAccessMixin, TemplateView):
    template_name = "meetings/meeting_detail.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        meeting_id = self.kwargs.get("meeting_id")
        meeting = self._get_meeting_with_relations(meeting_id)
        _, session_user_id, login_user_obj = self._check_access(self.request, meeting)

        context["meeting"] = meeting
        context["login_user"] = login_user_obj
        context["login_user_id"] = session_user_id
        context["attendees"] = meeting.attendees.select_related("user", "user__dept").all()
        context["tasks"] = meeting.tasks.select_related("assignee", "assignee__dept").all()
        context["tasks_display"] = [task_to_display(t) for t in context["tasks"]]

        summary_agendas = parse_summary_agendas(meeting.summary)
        context["summary_agendas"] = summary_agendas
        context["summary_agendas_text"] = format_agendas_for_minutes(summary_agendas)
        context["summary_agenda_first"] = summary_agendas[0].get("agenda") if summary_agendas else ""

        tasks_for_minutes = []
        for t in context["tasks_display"]:
            parts = []
            if t.get("who"):
                parts.append(f"Who: {t['who']}")
            if t.get("what"):
                parts.append(f"What: {t['what']}")
            if t.get("when"):
                parts.append(f"When: {t['when']}")
            if parts:
                tasks_for_minutes.append(" | ".join(parts))
        context["tasks_for_minutes"] = "\n".join(tasks_for_minutes)

        context["all_users"] = list(
            User.objects.select_related("dept")
            .values("user_id", "name", "dept__dept_name")
            .order_by("name")
        )
        context["transcript_display_html"] = render_transcript_html(meeting.transcript)
        try:
            context["all_users_json"] = json.dumps(context["all_users"], ensure_ascii=False)
        except Exception:
            context["all_users_json"] = "[]"

        is_host = _is_meeting_host(self.request, meeting)
        context["is_host"] = is_host
        context["can_edit_minutes"] = is_host
        return context


# ---------------------------------------------------------------------------
# 음성 파일 업로드 / 다운로드 / URL 연결
# ---------------------------------------------------------------------------

@require_GET
def meeting_audio_download(request, meeting_id):
    meeting = (
        Meeting.objects.select_related("record_url", "host")
        .prefetch_related("attendees__user__dept")
        .filter(pk=meeting_id)
        .first()
    )
    if not meeting:
        raise Http404()

    session_user_id, _, login_user_dept_id = _get_session_user_and_dept(request)
    attendees_list = list(meeting.attendees.all())
    if not has_meeting_view_permission(meeting, attendees_list, session_user_id, login_user_dept_id):
        return JsonResponse(
            {"ok": False, "error": "음성 파일을 다운로드할 권한이 없습니다."},
            status=403,
        )

    if not meeting.record_url_id:
        return JsonResponse({"ok": False, "error": "등록된 음성 파일이 없습니다."}, status=404)

    s3_obj = _resolve_s3_file(meeting)
    if not s3_obj:
        return JsonResponse({"ok": False, "error": "등록된 음성 파일이 없습니다."}, status=404)

    if is_local_storage_key(s3_obj.s3_key):
        try:
            local_path = get_local_audio_path(s3_obj.s3_key)
        except ValueError:
            return JsonResponse({"ok": False, "error": "등록된 음성 파일이 없습니다."}, status=404)
        if not local_path.exists():
            return JsonResponse({"ok": False, "error": "등록된 음성 파일이 없습니다."}, status=404)
        filename = getattr(s3_obj, "original_name", None) or f"meeting_{meeting_id}.wav"
        return FileResponse(open(local_path, "rb"), as_attachment=True, filename=filename, content_type="audio/wav")

    try:
        presigned_url = get_presigned_url(s3_obj.s3_key)
    except Exception:
        return JsonResponse(
            {"ok": False, "error": "음성 파일 URL을 생성하는 중 오류가 발생했습니다."},
            status=500,
        )
    try:
        s3_response = requests.get(presigned_url, stream=True, timeout=30)
    except requests.RequestException:
        return JsonResponse(
            {"ok": False, "error": "음성 파일을 가져오는 중 오류가 발생했습니다."},
            status=502,
        )

    if s3_response.status_code != 200:
        s3_response.close()
        return JsonResponse(
            {"ok": False, "error": "음성 파일을 가져오는 데 실패했습니다."},
            status=502,
        )

    def stream_file():
        try:
            for chunk in s3_response.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        finally:
            s3_response.close()

    filename = getattr(s3_obj, "original_name", None) or f"meeting_{meeting_id}.wav"
    fallback_filename = "meeting_audio.wav"
    quoted_filename = quote(filename)
    content_type = s3_response.headers.get("Content-Type") or "audio/wav"
    content_length = s3_response.headers.get("Content-Length")

    response = StreamingHttpResponse(stream_file(), content_type=content_type)
    if content_length:
        response["Content-Length"] = content_length
    response["Content-Disposition"] = (
        f"attachment; filename*=UTF-8''{quoted_filename}; filename=\"{fallback_filename}\""
    )
    return response


def meeting_record_upload(request, meeting_id):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    meeting = get_object_or_404(Meeting, pk=meeting_id)
    uploaded_file = request.FILES.get("file")
    if not uploaded_file:
        return JsonResponse({"error": "file is required"}, status=400)

    filename = uploaded_file.name
    ext = filename.split(".")[-1].lower()
    if ext not in ["wav"]:
        return JsonResponse({"error": "WAV 파일만 업로드 가능합니다."}, status=400)

    file_bytes = uploaded_file.read()
    try:
        record_url = upload_raw_file_bytes(
            file_bytes=file_bytes,
            original_filename=filename,
            delete_after_seconds=172800,
        )
    except Exception as e:
        return JsonResponse({"error": f"S3 업로드 중 오류: {str(e)}"}, status=500)

    meeting.record_url_id = record_url
    meeting.save(update_fields=["record_url"])
    return JsonResponse({"ok": True, "record_url": record_url})


@require_POST
def meeting_record_url_set(request, meeting_id):
    """이미 존재하는 S3 키를 회의에 직접 연결한다. 주최자만 설정 가능."""
    meeting = get_object_or_404(Meeting, pk=meeting_id)
    if not _is_meeting_host(request, meeting):
        return JsonResponse(
            {"ok": False, "error": "녹음 파일을 설정할 권한이 없습니다."},
            status=403,
        )

    payload, err_response = _parse_json_body(request)
    if err_response is not None:
        return err_response

    s3_key = (payload.get("s3_key") or "").strip()
    if not s3_key:
        return JsonResponse({"ok": False, "error": "s3_key is required"}, status=400)

    original_name = (payload.get("original_name") or "").strip() or (
        s3_key.split("/")[-1] or "audio.wav"
    )
    delete_after_seconds = payload.get("delete_after_seconds")
    if not isinstance(delete_after_seconds, int) or delete_after_seconds <= 0:
        delete_after_seconds = 60 * 60 * 24 * 30
    delete_at = timezone.now() + timedelta(seconds=delete_after_seconds)

    s3_obj, created = S3File.objects.get_or_create(
        s3_key=s3_key,
        defaults={"original_name": original_name, "delete_at": delete_at},
    )
    if not created:
        update_fields = ["delete_at"]
        if original_name and s3_obj.original_name != original_name:
            s3_obj.original_name = original_name
            update_fields.append("original_name")
        s3_obj.delete_at = delete_at
        s3_obj.save(update_fields=update_fields)

    meeting.record_url_id = s3_key
    meeting.save(update_fields=["record_url"])
    return JsonResponse({"ok": True, "s3_key": s3_key})


# ---------------------------------------------------------------------------
# STT / SLLM 비동기 폴링 렌더링 및 API
# ---------------------------------------------------------------------------

def meeting_summary(request, meeting_id):
    return redirect("meetings:meeting_detail", meeting_id=meeting_id)


class MeetingSttRenderingView(_BaseMeetingRenderingView):
    """STT 대기 화면"""

    template_name = "meetings/rending_stt.html"


class MeetingSllmRenderingView(_BaseMeetingRenderingView):
    """SLLM 대기 화면"""

    template_name = "meetings/rending_sllm.html"


@require_GET
def meeting_transcript_prepare(request, meeting_id):
    """
    렌딩 페이지에서 폴링으로 호출하는 엔드포인트.
    - 이미 transcript가 있으면 바로 done 리턴
    - job_id가 없으면 Runpod STT 비동기 작업을 시작하고 job_id 리턴
    - job_id가 있으면 Runpod 작업 상태를 조회하고, 완료 시 DB 저장 후 done 리턴
    """
    meeting = get_object_or_404(Meeting, pk=meeting_id)
    if meeting.transcript:
        return JsonResponse({"status": "done"})

    job_id = (request.GET.get("job_id") or "").strip()

    if not job_id:
        s3_obj = _resolve_s3_file(meeting)
        if not s3_obj:
            return JsonResponse({"status": "error", "message": "등록된 음성 파일이 없습니다."}, status=404)

        if is_local_storage_key(s3_obj.s3_key):
            try:
                local_path = get_local_audio_path(s3_obj.s3_key)
                audio_bytes = local_path.read_bytes()
            except Exception as e:
                return JsonResponse(
                    {"status": "error", "message": f"로컬 음성 파일을 읽는 중 오류가 발생했습니다: {e}"},
                    status=500,
                )
            print(f"[STT] meeting_id={meeting_id} local_key={s3_obj.s3_key} bytes={len(audio_bytes)}")
            start_fn = lambda: start_stt_job_from_bytes(audio_bytes)
        else:
            try:
                presigned_url = get_presigned_url(s3_obj.s3_key)
            except Exception as e:
                return JsonResponse(
                    {"status": "error", "message": f"음성 URL 생성 중 오류가 발생했습니다: {e}"},
                    status=500,
                )

            print(f"[STT] meeting_id={meeting_id} s3_key={s3_obj.s3_key} presigned_url_generated")
            start_fn = lambda: start_stt_job(presigned_url)

        new_job_id, err_res = start_runpod_job(
            start_fn,
            comm_error_prefix="STT 작업 시작 중 통신 오류가 발생했습니다",
            start_error_msg="전사 작업을 시작하지 못했습니다.",
        )
        if err_res is not None:
            return err_res

        meeting.transcript_status = "processing"
        meeting.transcript_error = ""
        meeting.save(update_fields=["transcript_status", "transcript_error"])
        print(f"[STT][queued] meeting_id={meeting_id} job_id={new_job_id}")
        return JsonResponse({"status": "processing", "job_id": new_job_id})

    job_status, res_json, err_res = poll_runpod_job(
        get_job_status,
        job_id,
        comm_error_prefix="STT 상태 조회 중 통신 오류가 발생했습니다",
        not_found_msg="전사 작업 정보를 찾을 수 없습니다. 다시 시도해 주세요.",
        status_error_msg="전사 상태 조회 중 오류가 발생했습니다.",
    )
    if err_res is not None:
        return err_res

    if job_status in ("queued", "processing"):
        return JsonResponse({"status": "processing", "job_id": job_id})

    if job_status == "error":
        message = extract_runpod_error(res_json) or "전사 처리 중 오류가 발생했습니다."
        meeting.transcript_status = "error"
        meeting.transcript_error = message
        meeting.save(update_fields=["transcript_status", "transcript_error"])
        return JsonResponse({"status": "error", "message": message}, status=500)

    if job_status == "done":
        result = res_json.get("result") or {}
        if not isinstance(result, dict) or not result.get("success"):
            message = extract_runpod_error(result) or "전사 처리 중 오류가 발생했습니다."
            meeting.transcript_status = "error"
            meeting.transcript_error = message
            meeting.save(update_fields=["transcript_status", "transcript_error"])
            return JsonResponse({"status": "error", "message": message}, status=500)

        meeting.transcript = result.get("data", {}).get("full_text")
        meeting.transcript_status = "done"
        meeting.transcript_error = ""
        meeting.save(update_fields=["transcript", "transcript_status", "transcript_error"])
        print(f"[STT][done] meeting_id={meeting_id} job_id={job_id}")
        return JsonResponse({"status": "done"})

    return JsonResponse(
        {"status": "error", "message": f"알 수 없는 작업 상태입니다: {job_status}"},
        status=500,
    )


@require_GET
def meeting_sllm_prepare(request, meeting_id):
    """
    전문(화자 매핑 완료본)을 SLLM에 전달해 요약/태스크를 생성한다.
    렌딩 페이지에서 폴링으로 호출하며, 완료 시 detail 화면으로 넘어간다.
    """
    meeting = get_object_or_404(Meeting, pk=meeting_id)
    transcript_plain = transcript_to_plain_text(meeting.transcript)
    if not transcript_plain.strip():
        return JsonResponse({"status": "error", "message": "분석할 전문이 없습니다."}, status=400)

    job_id = (request.GET.get("job_id") or "").strip()

    if not job_id:
        domain_raw = (meeting.domain or "").strip()
        domain_for_api = DOMAIN_API_MAP.get(domain_raw, domain_raw)
        domain_payload = [domain_for_api] if domain_for_api else []
        print(
            f"[SLLM] meeting_id={meeting_id} domain_raw='{domain_raw}' "
            f"domain_for_api='{domain_for_api}' domain_payload={domain_payload}"
        )

        new_job_id, err_res = start_runpod_job(
            lambda: start_sllm_job(transcript_plain, domain=domain_payload),
            comm_error_prefix="SLLM 작업 시작 중 통신 오류가 발생했습니다",
            start_error_msg="SLLM 작업을 시작하지 못했습니다.",
            preview_len=1000,
        )
        if err_res is not None:
            return err_res

        print(f"[SLLM][queued] meeting_id={meeting_id} job_id={new_job_id}")
        return JsonResponse({"status": "processing", "job_id": new_job_id})

    job_status, res_json, err_res = poll_runpod_job(
        get_job_status,
        job_id,
        comm_error_prefix="SLLM 상태 조회 중 통신 오류가 발생했습니다",
        not_found_msg="요약 작업 정보를 찾을 수 없습니다. 다시 시도해 주세요.",
        status_error_msg="SLLM 상태 조회 중 오류가 발생했습니다.",
        preview_len=1000,
    )
    if err_res is not None:
        return err_res

    if job_status in ("queued", "processing"):
        return JsonResponse({"status": "processing", "job_id": job_id})

    if job_status == "error":
        message = extract_runpod_error(res_json) or "SLLM 호출 중 오류가 발생했습니다."
        print(f"[SLLM][error] job_id={job_id} message={message}")
        return JsonResponse({"status": "error", "message": message}, status=500)

    if job_status == "done":
        job_result = res_json.get("result") or {}
        payload = job_result.get("data") if isinstance(job_result, dict) else None
        if payload is None:
            payload = job_result

        if (isinstance(job_result, dict) and job_result.get("success") is False) or not payload:
            message = extract_runpod_error(job_result) or "SLLM 호출 중 오류가 발생했습니다."
            print(f"[SLLM][error] job_id={job_id} message={message}")
            return JsonResponse({"status": "error", "message": message}, status=500)

        save_sllm_result(meeting, payload)
        print(f"[SLLM][done] meeting_id={meeting_id} job_id={job_id}")
        return JsonResponse({"status": "done"})

    return JsonResponse(
        {"status": "error", "message": f"알 수 없는 작업 상태입니다: {job_status}"},
        status=500,
    )


# ---------------------------------------------------------------------------
# 컨텍스트 프로세서 / 회의록·태스크 저장 / 회의록 다운로드
# ---------------------------------------------------------------------------

def today_meetings(request):
    """
    모든 템플릿에서 'today_meetings'로
    meet_date_time 기준 '오늘 날짜'인 회의 목록에 접근할 수 있게 해주는 컨텍스트 프로세서
    """
    today = date.today()
    login_user_id, _, login_user_dept_id = _get_session_user_and_dept(request)

    base_qs = (
        Meeting.objects.filter(meet_date_time__date=today)
        .select_related("host")
        .prefetch_related("attendees__user__dept")
        .order_by("-meet_date_time")
    )

    meetings = []
    now = timezone.now()
    for m in base_qs:
        attendees = list(m.attendees.all())
        if not has_meeting_view_permission(m, attendees, login_user_id, login_user_dept_id):
            continue

        total_minutes = int((now - m.meet_date_time).total_seconds() / 60)
        if total_minutes < 60:
            m.time_ago = f"{total_minutes}분 전"
        else:
            m.time_ago = f"{total_minutes // 60}시간 전"

        meetings.append(m)
        if len(meetings) >= 3:
            break

    return {"today_meetings": meetings}


def minutes_save(request, meeting_id):
    """회의록 HTML(meeting_notes)을 저장하는 용도 : 주최자만 저장가능."""
    meeting = get_object_or_404(Meeting, pk=meeting_id)
    if not getattr(meeting, "host", None) or not _is_meeting_host(request, meeting):
        return JsonResponse(
            {"ok": False, "error": "회의록을 수정할 권한이 없습니다."},
            status=403,
        )

    data, err_response = _parse_json_body(request)
    if err_response is not None:
        return err_response

    meeting.meeting_notes = (data.get("content") or "").strip()
    meeting.save(update_fields=["meeting_notes"])
    return JsonResponse({"ok": True})


@require_POST
def tasks_save(request, meeting_id):
    """태스크(Who/What/When) 수정본을 저장. 주최자만 가능."""
    meeting = get_object_or_404(Meeting, pk=meeting_id)
    if not _is_meeting_host(request, meeting):
        return JsonResponse(
            {"ok": False, "error": "태스크를 저장할 권한이 없습니다."},
            status=403,
        )

    payload, err_response = _parse_json_body(request)
    if err_response is not None:
        return err_response

    tasks_payload = payload.get("tasks")
    if tasks_payload is None or not isinstance(tasks_payload, list):
        return JsonResponse({"ok": False, "error": "tasks 형식이 올바르지 않습니다."}, status=400)

    new_tasks = []
    for item in tasks_payload:
        if not isinstance(item, dict):
            continue
        who = (item.get("who") or item.get("assignee") or "").strip()
        assignee_id_payload = item.get("assignee_id")
        what = (item.get("what") or item.get("description") or "").strip()
        when = (item.get("when") or item.get("due") or item.get("due_text") or "").strip()

        if not (who or what or when):
            continue

        parsed_due = parse_due_date(when) if when else None
        assignee_obj = None
        if assignee_id_payload:
            try:
                assignee_obj = User.objects.filter(user_id=assignee_id_payload).first()
            except Exception:
                assignee_obj = None
        if not assignee_obj and who:
            try:
                who_for_lookup = who.split("(", 1)[0].strip()
                if meeting.host and who_for_lookup == meeting.host.name:
                    assignee_obj = meeting.host
                if not assignee_obj:
                    assignee_obj = User.objects.filter(name=who_for_lookup).first()
            except Exception:
                assignee_obj = None

        content_payload = {"description": what}
        if who:
            content_payload["assignee"] = who
            if assignee_obj and getattr(assignee_obj, "user_id", None):
                content_payload["assignee_id"] = assignee_obj.user_id
        if when and when.strip().lower() not in ("*", "null", "none", ""):
            content_payload["due"] = when
            if parsed_due:
                content_payload["due_date"] = parsed_due.isoformat()

        new_tasks.append(
            Task(
                meeting=meeting,
                task_content=json.dumps(content_payload, ensure_ascii=False),
                assignee=assignee_obj,
                due_date=parsed_due,
            )
        )

    with transaction.atomic():
        meeting.tasks.all().delete()
        if new_tasks:
            Task.objects.bulk_create(new_tasks)

    return JsonResponse({"ok": True})


def minutes_download(request, meeting_id, fmt):
    meeting = get_object_or_404(Meeting, pk=meeting_id)
    return build_minutes_download_response(meeting, fmt)
