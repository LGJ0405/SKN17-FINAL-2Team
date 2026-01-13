from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.hashers import check_password, make_password
from django.views.decorators.http import require_POST
from .models import User
from .forms import LoginForm
from django.contrib.auth import logout as django_logout
from django.utils import timezone
import re

def login_view(request):
    form = LoginForm()
    return render(request, "users/login.html", {"form": form})


@require_POST
def login_api(request):
    form = LoginForm(request.POST)

    if not form.is_valid():
        return JsonResponse(
            {"ok": False, "errors": form.errors},
            status=400
        )

    user_id = form.cleaned_data["user_id"]
    password = form.cleaned_data["password"]

    try:
        user = User.objects.get(user_id=user_id)
    except User.DoesNotExist:
        return JsonResponse(
            {"ok": False, "errors": {"__all__": ["존재하지 않는 아이디입니다."]}},
            status=400
        )

    # 계정 상태 확인 (비활성화된 계정은 로그인 불가)
    if user.status == User.STATUS_INACTIVE:
        return JsonResponse(
            {
                "ok": False,
                "locked": True,
                "errors": {"__all__": ["계정이 비활성화되었습니다. 관리자에게 문의하세요."]}
            },
            status=403
        )

    # 비밀번호 확인
    db_pw = user.password
    if db_pw.startswith("pbkdf2_"):
        pw_ok = check_password(password, db_pw)
    else:
        pw_ok = (password == db_pw)

    if not pw_ok:
        # 로그인 실패 카운트 증가
        user.login_fail_count += 1

        # 5회 이상 실패 시 계정 비활성화
        if user.login_fail_count >= 5:
            user.status = User.STATUS_INACTIVE
            user.save()
            return JsonResponse(
                {
                    "ok": False,
                    "locked": True,
                    "errors": {"__all__": ["비밀번호를 5회 이상 틀렸습니다. 계정이 비활성화되었습니다. 관리자에게 비밀번호 초기화를 문의하세요."]}
                },
                status=403
            )

        user.save()
        remaining_attempts = 5 - user.login_fail_count

        return JsonResponse(
            {
                "ok": False,
                "errors": {"__all__": [f"비밀번호가 일치하지 않습니다. ({remaining_attempts}회 남음)"]}
            },
            status=400
        )

    # 로그인 성공 - 실패 카운트 초기화
    user.login_fail_count = 0
    user.save()

    # 비밀번호 정책(8~15자, 영문+숫자) 미충족 시 로그인 막고 변경 플래그 반환
    # 주의: 세션에 정상 로그인 키(`login_user_id`)를 넣으면 사이트 접근이 허용되므로
    # 강제 변경 흐름은 별도 세션 키 `pending_pw_change_user_id`를 사용한다.
    pw_pattern = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,15}$")
    if not pw_pattern.match(password):
        # 로그인 상태로 처리하지 않음 — 변경 전에는 일반 접근 불가
        request.session["pending_pw_change_user_id"] = user.user_id
        request.session["pending_pw_change_user_name"] = user.name
        request.session.save()
        return JsonResponse(
            {
                "ok": False,
                "force_change": True,
                "message": "초기 비밀번호를 변경해 주세요.",
            },
            status=400,
        )

    # 세션 설정
    request.session["login_user_id"] = user.user_id
    request.session["login_user_name"] = user.name
    request.session["login_user_dept_name"] = getattr(user.dept, "dept_name", "")
    request.session["login_user_admin"] = user.admin_yn

    # 세션 명시적 저장
    request.session.save()

    print("[DEBUG] 로그인 성공 - 세션:", dict(request.session))

    next_url = request.POST.get("next") or "/"
    return JsonResponse({"ok": True, "redirect_url": next_url})

def logout_view(request):
    # Django auth + 직접 넣은 세션 모두 정리
    request.session.flush()
    return redirect("users:login")

@require_POST
def check_old_pw(request):
    """
    기존 비밀번호 실시간 확인용 API
    - ok: true  => 비밀번호 일치
    - ok: false => 비밀번호 불일치 또는 입력 없음
    """
    login_user_id = request.session.get("login_user_id")
    if not login_user_id:
        return JsonResponse({"ok": False}, status=401)

    # 세션에 user.user_id를 넣어두신 구조라면 이렇게 조회
    user = get_object_or_404(User, user_id=login_user_id)

    old_password = request.POST.get("old_password", "").strip()
    if not old_password:
        return JsonResponse({"ok": False}, status=400)

    if check_password(old_password, user.password):
        return JsonResponse({"ok": True})
    else:
        return JsonResponse({"ok": False})

@require_POST
def modify_pw_view(request):
    # 0. 로그인 여부 확인
    login_user_id = request.session.get("login_user_id")
    if not login_user_id:
        return JsonResponse(
            {"ok": False, "field": "common", "message": "로그인이 필요합니다."},
            status=401,
        )

    # 세션에 user.user_id를 넣어 두었다고 가정
    user = get_object_or_404(User, user_id=login_user_id)

    old_password = request.POST.get("old_password", "").strip()
    new_password = request.POST.get("new_password", "").strip()
    new_password2 = request.POST.get("new_password2", "").strip()

    # 1. (방어용) 빈 값 체크 – 프론트에서 이미 막지만 혹시 모를 경우 대비
    if not old_password or not new_password or not new_password2:
        return JsonResponse(
            {
                "ok": False,
                "field": "common",
                "message": "모든 항목을 입력해 주세요.",
            },
            status=400,
        )

    # 2. 비밀번호 정책(8~15자, 영문 + 숫자 포함) 체크
    pw_pattern = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,15}$")
    if not pw_pattern.match(new_password):
        return JsonResponse(
            {
                "ok": False,
                "field": "new_password",
                "message": "비밀번호 형식을 맞추어주세요.",
            },
            status=400,
        )

    # 3. 새 비밀번호 두 칸이 같은지 체크
    if new_password != new_password2:
        return JsonResponse(
            {
                "ok": False,
                "field": "new_password2",
                "message": "비밀번호를 확인해주세요.",
            },
            status=400,
        )

    # 4. 기존 비밀번호가 맞는지 체크
    if not check_password(old_password, user.password):
        return JsonResponse(
            {
                "ok": False,
                "field": "old_password",
                "message": "비밀번호를 확인해주세요.",
            },
            status=400,
        )

    # 5. 새 비밀번호가 기존과 같은지 방지 (선택)
    if check_password(new_password, user.password):
        return JsonResponse(
            {
                "ok": False,
                "field": "new_password",
                "message": "이전에 사용한 비밀번호와 다른 비밀번호를 입력해 주세요.",
            },
            status=400,
        )

    # 6. 실제 비밀번호 변경
    user.password = make_password(new_password)
    user.save()

    return JsonResponse({"ok": True})


@require_POST
def modify_pw_initial(request):
    """
    로그인 시 비밀번호 정책이 맞지 않아 막힌 경우,
    기존 비밀번호 입력 없이 새 비밀번호로만 변경
    """
    # allow flow both for fully-logged-in users and for pending initial-change users
    login_user_id = request.session.get("login_user_id")
    pending_user_id = request.session.get("pending_pw_change_user_id")
    if not login_user_id and not pending_user_id:
        return JsonResponse(
            {"ok": False, "field": "common", "message": "로그인이 필요합니다."},
            status=401,
        )

    user = None
    if login_user_id:
        user = get_object_or_404(User, user_id=login_user_id)
    else:
        user = get_object_or_404(User, user_id=pending_user_id)

    new_password = request.POST.get("new_password", "").strip()
    new_password2 = request.POST.get("new_password2", "").strip()

    if not new_password or not new_password2:
        return JsonResponse(
            {"ok": False, "field": "common", "message": "모든 항목을 입력해 주세요."},
            status=400,
        )

    pw_pattern = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,15}$")
    if not pw_pattern.match(new_password):
        return JsonResponse(
            {"ok": False, "field": "new_password", "message": "비밀번호 형식을 맞추어주세요."},
            status=400,
        )

    if new_password != new_password2:
        return JsonResponse(
            {"ok": False, "field": "new_password2", "message": "비밀번호를 확인해주세요."},
            status=400,
        )

    # 기존 비밀번호와 동일한지 체크 (평문/해시 모두 대응)
    if check_password(new_password, user.password) or user.password == new_password:
        return JsonResponse(
            {"ok": False, "field": "new_password", "message": "이전에 사용한 비밀번호와 다른 비밀번호를 입력해 주세요."},
            status=400,
        )

    user.password = make_password(new_password)
    user.save()

    # 초기 변경 흐름으로 왔다면 정상 로그인 세션으로 전환하고 pending 키 제거
    if pending_user_id:
        request.session.pop("pending_pw_change_user_id", None)
        request.session.pop("pending_pw_change_user_name", None)
        request.session["login_user_id"] = user.user_id
        request.session["login_user_name"] = user.name
        request.session["login_user_dept_name"] = getattr(user.dept, "dept_name", "")
        request.session["login_user_admin"] = user.admin_yn
        request.session.save()

    return JsonResponse({"ok": True, "redirect_url": "/"})


@require_POST
def admin_reset_password(request):
    """
    관리자가 잠긴 계정의 비밀번호를 초기화하고 계정 잠금을 해제
    """
    # 관리자 권한 확인
    login_user_id = request.session.get("login_user_id")
    if not login_user_id:
        return JsonResponse(
            {"ok": False, "message": "로그인이 필요합니다."},
            status=401
        )

    try:
        admin_user = User.objects.get(user_id=login_user_id)
        if not admin_user.admin_yn:
            return JsonResponse(
                {"ok": False, "message": "관리자 권한이 필요합니다."},
                status=403
            )
    except User.DoesNotExist:
        return JsonResponse({"ok": False, "message": "사용자를 찾을 수 없습니다."}, status=404)

    # 대상 사용자 ID 가져오기
    target_user_id = request.POST.get("target_user_id", "").strip()
    if not target_user_id:
        return JsonResponse(
            {"ok": False, "message": "초기화할 사용자 ID를 입력해주세요."},
            status=400
        )

    # 대상 사용자 조회
    try:
        target_user = User.objects.get(user_id=target_user_id)
    except User.DoesNotExist:
        return JsonResponse(
            {"ok": False, "message": "해당 사용자를 찾을 수 없습니다."},
            status=404
        )

    # 비밀번호 초기화 (생년월일 6자리로 설정)
    initial_password = target_user.birth_date.strftime("%y%m%d")
    target_user.password = make_password(initial_password)

    # 계정 활성화 및 실패 카운트 초기화
    target_user.status = User.STATUS_ACTIVE
    target_user.login_fail_count = 0
    target_user.save()

    return JsonResponse({
        "ok": True,
        "message": f"'{target_user_id}' 계정의 비밀번호가 초기화되고 활성화되었습니다. (초기 비밀번호: 생년월일 6자리)"
    })
