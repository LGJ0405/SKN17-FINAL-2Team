import os
import json
from datetime import datetime, timedelta

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import redirect
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from users.models import User
from google_calendar.models import GoogleCalendarToken, OAuthState
from google_calendar.utils import get_google_credentials

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build


# -------------------------------------------------------------------
# Google OAuth 로그인 / 콜백
# -------------------------------------------------------------------
def google_login(request):
    """
    구글 OAuth 로그인/동의 화면으로 리다이렉트
    """
    # 로그인 확인
    login_user_id = request.session.get("login_user_id")
    if not login_user_id:
        return JsonResponse(
            {"error": "not_logged_in", "detail": "먼저 로그인해주세요."},
            status=400
        )

    if settings.DEBUG:
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    flow = Flow.from_client_secrets_file(
        settings.GOOGLE_OAUTH2_CLIENT_SECRETS_JSON,
        scopes=settings.GOOGLE_OAUTH2_SCOPES,
        redirect_uri=settings.GOOGLE_OAUTH2_REDIRECT_URI,
    )

    # authorization_url을 만들면서, 내부적으로 state를 생성해 준다.
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    # ✅ state와 user_id를 DB에 저장 (세션이 유지되지 않으므로)
    OAuthState.objects.update_or_create(
        state=state,
        defaults={"user_id": login_user_id}
    )

    return redirect(authorization_url)

def oauth2callback(request):
    """
    구글에서 redirect 되는 콜백 URL
    - 구글이 준 code, state를 이용해 토큰 발급
    - 로그인한 User 기준으로 GoogleCalendarToken 저장
    """
    if settings.DEBUG:
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    # URL에서 state 파라미터 가져오기
    state = request.GET.get("state")
    if not state:
        return JsonResponse({
            "error": "missing_state",
            "detail": "인증 상태 정보가 없습니다.",
        }, status=400)

    # DB에서 state로 user_id 조회
    try:
        oauth_state = OAuthState.objects.get(state=state)
        login_user_id = oauth_state.user_id
    except OAuthState.DoesNotExist:
        return JsonResponse({
            "error": "invalid_state",
            "detail": "유효하지 않은 인증 상태입니다. 다시 시도해주세요.",
        }, status=400)

    # ✅ Flow를 같은 state, 같은 redirect_uri로 생성해야 한다.
    try:
        flow = Flow.from_client_secrets_file(
            settings.GOOGLE_OAUTH2_CLIENT_SECRETS_JSON,
            scopes=settings.GOOGLE_OAUTH2_SCOPES,
            state=state,
        )
        flow.redirect_uri = settings.GOOGLE_OAUTH2_REDIRECT_URI

        # 구글이 현재 이 콜백으로 호출한 전체 URL
        authorization_response = request.build_absolute_uri()

        # ✅ 여기서 oauthlib이 "URL에 있는 state"와 "Flow에 지정된 state"를 비교한다.
        flow.fetch_token(authorization_response=authorization_response)

        credentials = flow.credentials
        token_json = credentials.to_json()

    except Exception as e:
        error_msg = str(e)
        # 스코프 변경 에러인 경우 기존 토큰 삭제 안내
        if "Scope has changed" in error_msg or "scope" in error_msg.lower():
            try:
                user = User.objects.get(user_id=login_user_id)
                GoogleCalendarToken.objects.filter(user=user).delete()
            except:
                pass
            return JsonResponse({
                "error": "oauth_scope_changed",
                "detail": "인증 범위가 변경되었습니다. 구글 계정 설정(https://myaccount.google.com/permissions)에서 이 앱의 액세스 권한을 삭제한 후 다시 로그인해주세요.",
                "action_required": "revoke_google_permission"
            }, status=400)

        return JsonResponse({
            "error": "oauth_error",
            "detail": error_msg
        }, status=500)

    try:
        user = User.objects.get(user_id=login_user_id)
    except User.DoesNotExist:
        return JsonResponse({"error": "user_not_found"}, status=404)

    # 유저별로 토큰 저장/업데이트
    GoogleCalendarToken.objects.update_or_create(
        user=user,
        defaults={"token_json": token_json},
    )

    # 사용한 state는 DB에서 삭제
    oauth_state.delete()

    # 세션을 복원하고 리다이렉트
    request.session["login_user_id"] = login_user_id
    request.session["login_user_name"] = user.name
    request.session["login_user_dept_name"] = getattr(user.dept, "dept_name", "")
    request.session["login_user_admin"] = user.admin_yn
    request.session["google_credentials"] = token_json
    request.session.save()

    return redirect("/")  # 홈으로 이동

# -------------------------------------------------------------------
# Google Calendar 이벤트 목록 조회
#  - /api/google-events/
# -------------------------------------------------------------------
def google_events(request):
    creds = get_google_credentials(request)
    if not creds:
        return JsonResponse({"error": "not_authenticated"}, status=401)

    service = build("calendar", "v3", credentials=creds)
    tasks_service = None

    events = []
    calendars = []
    page_token = None

    try:
        while True:
            cal_list = service.calendarList().list(pageToken=page_token).execute()
            calendars.extend(cal_list.get("items", []))
            page_token = cal_list.get("nextPageToken")
            if not page_token:
                break
    except Exception as e:
        return JsonResponse(
            {"error": "google_api_error", "detail": str(e)},
            status=500,
        )

    if not calendars:
        calendars.append({"id": "primary", "summary": "기본 캘린더", "primary": True})

    for cal in calendars:
        try:
            events_result = service.events().list(
                calendarId=cal.get("id", "primary"),
                timeMin="2020-01-01T00:00:00Z",
                timeMax="2030-12-31T23:59:59Z",
                maxResults=500,
                singleEvents=True,
                orderBy="startTime",
            ).execute()
        except Exception:
            # 특정 캘린더 조회 실패 시 다른 캘린더는 계속 조회
            continue

        for e in events_result.get("items", []):
            repeat_type = "none"
            if "recurrence" in e:
                rrule = e["recurrence"][0] if e["recurrence"] else ""
                if "FREQ=DAILY" in rrule:
                    repeat_type = "daily"
                elif "FREQ=WEEKLY" in rrule:
                    repeat_type = "weekly"
                elif "FREQ=MONTHLY" in rrule:
                    repeat_type = "monthly"
                elif "FREQ=YEARLY" in rrule:
                    repeat_type = "yearly"

            events.append(
                {
                    "id": e["id"],
                    "title": e.get("summary", ""),
                    "start": e["start"].get("dateTime") or e["start"].get("date"),
                    "end": e["end"].get("dateTime") or e["end"].get("date"),
                    "description": e.get("description", ""),
                    "repeat": repeat_type,
                    "calendarId": cal.get("id", "primary"),
                    "calendarSummary": cal.get("summary", ""),
                }
            )

    # Google Tasks -> FullCalendar 이벤트 형태로 변환하여 추가
    try:
        tasks_service = build("tasks", "v1", credentials=creds)
        default_tasklist = tasks_service.tasklists().get(tasklist="@default").execute()
        task_calendar_id = default_tasklist.get("id") or "@default"
        task_calendar_summary = default_tasklist.get("title") or "Tasks"

        task_page_token = None
        while True:
            task_list = tasks_service.tasks().list(
                tasklist="@default",
                pageToken=task_page_token,
                showCompleted=False,
                showDeleted=False,
                maxResults=100,
            ).execute()

            for t in task_list.get("items", []):
                due = t.get("due")
                if not due:
                    continue  # 마감일이 없는 태스크는 달력에 표시하지 않음

                start_iso = due
                end_iso = due

                # notes에 start/end를 JSON 형태로 저장했다면 그것을 우선 사용
                notes_raw = t.get("notes") or ""
                parsed_start = None
                parsed_end = None
                try:
                    notes_json = json.loads(notes_raw)
                    parsed_start = notes_json.get("start") or None
                    parsed_end = notes_json.get("end") or None
                except Exception:
                    parsed_start = None
                    parsed_end = None

                if parsed_start:
                    start_iso = parsed_start
                if parsed_end:
                    end_iso = parsed_end

                # end 시간이 없으면 +1시간 기본 설정
                is_timed = "T" in start_iso if start_iso else False
                if not parsed_end and is_timed:
                    try:
                        parsed = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
                        end_iso = (parsed + timedelta(hours=1)).isoformat()
                    except Exception:
                        end_iso = start_iso
                # 시각이 없으면 종일 이벤트로 취급
                if not is_timed:
                    end_iso = start_iso

                events.append(
                    {
                        "id": f"task-{t.get('id')}",
                        "title": t.get("title", ""),
                        "start": start_iso,
                        "end": end_iso,
                        "allDay": not is_timed,
                        "description": notes_raw,
                        "repeat": "none",
                        "calendarId": task_calendar_id,
                        "calendarSummary": task_calendar_summary,
                    }
                )

            task_page_token = task_list.get("nextPageToken")
            if not task_page_token:
                break
    except Exception:
        # Tasks 연동 실패 시에도 기존 캘린더 이벤트 반환
        pass

    return JsonResponse(events, safe=False)


# -------------------------------------------------------------------
# Google Calendar 이벤트 생성
#  - /api/google-events/create/
# -------------------------------------------------------------------
@csrf_exempt  # CSRF 토큰 제대로 쓰실 거면 이 데코레이터는 제거해도 됩니다.
@require_POST
def create_google_event(request):
    """
    홈 화면에서 보낸 일정 데이터를 사용자의 Google Calendar에 생성
    """
    creds = get_google_credentials(request)
    if not creds:
        return JsonResponse({"error": "not_authenticated"}, status=401)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid_json"}, status=400)

    title = data.get("title")
    start = data.get("start")
    end = data.get("end")
    description = data.get("description", "")
    repeat = data.get("repeat", "none")
    calendar_id = data.get("calendarId") or "primary"

    if not title or not start or not end:
        return JsonResponse({"error": "missing_fields"}, status=400)

    service = build("calendar", "v3", credentials=creds)

    time_zone = "Asia/Seoul"

    event_body = {
        "summary": title,
        "description": description,
        "start": {
            "dateTime": start,
            "timeZone": time_zone,
        },
        "end": {
            "dateTime": end,
            "timeZone": time_zone,
        },
    }

    # 반복 규칙 추가
    if repeat and repeat != "none":
        freq_map = {
            "daily": "DAILY",
            "weekly": "WEEKLY",
            "monthly": "MONTHLY",
            "yearly": "YEARLY",
        }
        freq = freq_map.get(repeat)
        if freq:
            event_body["recurrence"] = [f"RRULE:FREQ={freq}"]

    try:
        created_event = service.events().insert(
            calendarId=calendar_id,
            body=event_body,
        ).execute()
    except Exception as e:
        return JsonResponse(
            {"error": "google_api_error", "detail": str(e)},
            status=500,
        )

    return JsonResponse(
        {
            "id": created_event.get("id"),
            "status": created_event.get("status"),
        }
    )


# -------------------------------------------------------------------
# Google Tasks 생성
#  - /api/google-tasks/create/
# -------------------------------------------------------------------
@csrf_exempt
@require_POST
def create_google_task(request):
    """
    회의 상세 Add 버튼에서 전달한 태스크를 Google Tasks에 생성한다.
    기대 payload:
      {
        "title": "...",           # 필수
        "notes": "...",           # 선택
        "due": "RFC3339 string",  # 필수: 2024-12-20T15:00:00+09:00
        "tasklist_id": "@default" # 선택
      }
    """
    creds = get_google_credentials(request)
    if not creds:
        return JsonResponse({"error": "not_authenticated"}, status=401)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid_json"}, status=400)

    title = (data.get("title") or "").strip()
    due = (data.get("due") or "").strip()
    notes = (data.get("notes") or "").strip()
    tasklist_id = (data.get("tasklist_id") or "@default").strip() or "@default"

    if not title or not due:
        return JsonResponse({"error": "missing_fields"}, status=400)

    try:
        service = build("tasks", "v1", credentials=creds)
        body = {
            "title": title,
            "due": due,
        }
        if notes:
            body["notes"] = notes

        task = service.tasks().insert(tasklist=tasklist_id, body=body).execute()
        return JsonResponse({"ok": True, "task_id": task.get("id")})
    except Exception as e:
        return JsonResponse(
            {"error": "google_api_error", "detail": str(e)},
            status=500,
        )


# -------------------------------------------------------------------
# Google Calendar 인증 여부 확인
#  - /api/google-auth-status/
# -------------------------------------------------------------------
def google_auth_status(request):
    """
    현재 요청 기준으로 Google 인증이 되어 있는지 여부만 반환
    """
    creds = get_google_credentials(request)
    return JsonResponse({"authenticated": bool(creds)})


# -------------------------------------------------------------------
# Google Calendar 이벤트 수정
#  - /api/google-events/<event_id>/update/
# -------------------------------------------------------------------
@csrf_exempt
@require_POST
def update_google_event(request, event_id):
    """
    /api/google-events/<event_id>/update/ 로 들어오는 수정 요청 처리
    """
    creds = get_google_credentials(request)
    if not creds:
        return JsonResponse({"error": "not_authenticated"}, status=401)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid_json"}, status=400)

    title = data.get("title")
    start = data.get("start")
    end = data.get("end")
    description = data.get("description", "")
    repeat = data.get("repeat", "none")
    calendar_id = data.get("calendarId") or "primary"

    if not title or not start or not end:
        return JsonResponse({"error": "missing_fields"}, status=400)

    service = build("calendar", "v3", credentials=creds)
    time_zone = "Asia/Seoul"

    event_body = {
        "summary": title,
        "description": description,
        "start": {
            "dateTime": start,
            "timeZone": time_zone,
        },
        "end": {
            "dateTime": end,
            "timeZone": time_zone,
        },
    }

    # 반복 규칙 추가
    if repeat and repeat != "none":
        freq_map = {
            "daily": "DAILY",
            "weekly": "WEEKLY",
            "monthly": "MONTHLY",
            "yearly": "YEARLY",
        }
        freq = freq_map.get(repeat)
        if freq:
            event_body["recurrence"] = [f"RRULE:FREQ={freq}"]

    try:
        updated_event = service.events().update(
            calendarId=calendar_id,
            eventId=event_id,
            body=event_body,
        ).execute()
    except Exception as e:
        return JsonResponse(
            {"error": "google_api_error", "detail": str(e)},
            status=500,
        )

    return JsonResponse(
        {
            "id": updated_event.get("id"),
            "status": updated_event.get("status"),
        }
    )


# -------------------------------------------------------------------
# Google Calendar 이벤트 삭제
#  - /api/google-events/<event_id>/delete/
# -------------------------------------------------------------------
@csrf_exempt  # 마찬가지로 CSRF 토큰 처리 후에는 제거 가능
@require_POST
def google_events_delete(request, event_id):
    """
    /api/google-events/<event_id>/delete/ 로 들어오는 삭제 요청 처리
    """
    creds = get_google_credentials(request)
    if not creds:
        return JsonResponse({"error": "not_authenticated"}, status=401)

    calendar_id = "primary"
    try:
        body_data = json.loads(request.body.decode("utf-8"))
        calendar_id = body_data.get("calendarId") or "primary"
    except json.JSONDecodeError:
        pass

    # task- 로 시작하면 Google Tasks API로 삭제, 아니면 Calendar 이벤트 삭제
    if event_id.startswith("task-"):
        task_id = event_id.replace("task-", "", 1)
        tasklist_id = calendar_id or "@default"
        try:
            tasks_service = build("tasks", "v1", credentials=creds)
            tasks_service.tasks().delete(tasklist=tasklist_id, task=task_id).execute()
        except Exception as e:
            return JsonResponse(
                {"error": "google_api_error", "detail": str(e)},
                status=500,
            )
    else:
        service = build("calendar", "v3", credentials=creds)
        try:
            service.events().delete(
                calendarId=calendar_id,
                eventId=event_id,
            ).execute()
        except Exception as e:
            return JsonResponse(
                {"error": "google_api_error", "detail": str(e)},
                status=500,
            )

    return JsonResponse({"success": True})


def google_calendars(request):
    """
    사용자가 접근 가능한 Google 캘린더 목록 조회
    """
    creds = get_google_credentials(request)
    if not creds:
        return JsonResponse({"error": "not_authenticated"}, status=401)

    service = build("calendar", "v3", credentials=creds)
    calendars = []
    page_token = None

    try:
        while True:
            cal_list = service.calendarList().list(pageToken=page_token).execute()
            calendars.extend(cal_list.get("items", []))
            page_token = cal_list.get("nextPageToken")
            if not page_token:
                break
    except Exception as e:
        return JsonResponse(
            {"error": "google_api_error", "detail": str(e)},
            status=500,
        )

    result = []
    for cal in calendars:
        result.append(
            {
                "id": cal.get("id"),
                "summary": cal.get("summary", ""),
                "primary": cal.get("primary", False),
            }
        )

    # 기본 Tasks 리스트도 별도 캘린더로 노출
    try:
        tasks_service = build("tasks", "v1", credentials=creds)
        default_tasklist = tasks_service.tasklists().get(tasklist="@default").execute()
        result.append(
            {
                "id": default_tasklist.get("id") or "@default",
                "summary": default_tasklist.get("title") or "Tasks",
                "primary": False,
                "is_task_list": True,
            }
        )
    except Exception:
        pass

    if not result:
        result.append({"id": "primary", "summary": "기본 캘린더", "primary": True})

    return JsonResponse(result, safe=False)


@csrf_exempt
@require_POST
def revoke_google_auth(request):
    """
    현재 로그인한 사용자의 구글 인증 토큰 삭제
    - 스코프 변경 등으로 인한 에러 발생 시 재인증을 위해 사용
    """
    login_user_id = request.session.get("login_user_id")
    if not login_user_id:
        return JsonResponse(
            {"error": "not_logged_in", "detail": "로그인이 필요합니다."},
            status=400
        )

    try:
        user = User.objects.get(user_id=login_user_id)
        GoogleCalendarToken.objects.filter(user=user).delete()

        # 세션에서도 제거
        if "google_credentials" in request.session:
            del request.session["google_credentials"]
            request.session.save()

        return JsonResponse({"success": True, "message": "구글 연동이 해제되었습니다. 다시 로그인해주세요."})
    except User.DoesNotExist:
        return JsonResponse({"error": "user_not_found"}, status=404)
    except Exception as e:
        return JsonResponse(
            {"error": "revoke_failed", "detail": str(e)},
            status=500
        )
