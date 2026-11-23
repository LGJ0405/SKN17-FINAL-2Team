import json
import re
from datetime import date, datetime
from functools import wraps

from django.contrib.auth.hashers import make_password
from django.http import JsonResponse
from django.shortcuts import redirect
from django.utils.decorators import method_decorator
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_http_methods
from django.views.generic import TemplateView

from users.models import Dept, User


class LoginRequiredSessionMixin:
    login_url = "/users/login/"

    @method_decorator(never_cache)
    def dispatch(self, request, *args, **kwargs):
        if not request.session.get("login_user_id"):
            return redirect(self.login_url)
        return super().dispatch(request, *args, **kwargs)


class HomeView(LoginRequiredSessionMixin, TemplateView):
    template_name = "includes/home.html"


class AdminHomeView(LoginRequiredSessionMixin, TemplateView):
    template_name = "includes/admin_home.html"

    def dispatch(self, request, *args, **kwargs):
        if not request.session.get("login_user_admin"):
            return redirect("/")
        try:
            user = User.objects.get(user_id=request.session.get("login_user_id"))
            if not user.admin_yn:
                return redirect("/")
        except Exception:
            return redirect("/")
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["departments"] = Dept.objects.all().order_by("dept_name")
        context["members"] = (
            User.objects.filter(status=User.STATUS_ACTIVE)
            .select_related("dept")
            .order_by("dept__dept_name", "name")
        )
        return context


def _admin_api_required(view_func):
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not request.session.get("login_user_admin"):
            return JsonResponse({"success": False, "message": "권한이 없습니다."}, status=403)
        return view_func(request, *args, **kwargs)

    return _wrapped


def _error_response(message: str, status: int = 400) -> JsonResponse:
    return JsonResponse({"success": False, "message": message}, status=status)


def _validate_name(name: str):
    if not re.match(r"^[가-힣]{1,32}$", name):
        return _error_response("이름은 한글만 입력 가능하며 1~32자여야 합니다.", 400)
    return None


def _validate_work_part(work_part: str):
    if not re.match(r"^[가-힣a-zA-Z\s]{1,15}$", work_part):
        return _error_response("업무는 한글 또는 영어만 입력 가능하며 1~15자여야 합니다.", 400)
    return None


def _parse_birth_date(birth_date: str):
    if not re.match(r"^\d{8}$", birth_date):
        return None, _error_response("생년월일은 8자리 숫자로 입력해주세요.", 400)
    try:
        birth_date_obj = datetime.strptime(birth_date, "%Y%m%d").date()
    except ValueError:
        return None, _error_response("유효하지 않은 생년월일입니다.", 400)

    today = date.today()
    hundred_years_ago = date(today.year - 100, today.month, today.day)
    nineteen_years_ago = date(today.year - 19, today.month, today.day)

    if birth_date_obj < hundred_years_ago:
        return None, _error_response("생년월일은 현재로부터 100년 이내여야 합니다.", 400)
    if birth_date_obj > nineteen_years_ago:
        return None, _error_response("만 19세 이상만 등록 가능합니다.", 400)

    return birth_date_obj, None


def _get_dept(dept_id):
    try:
        return Dept.objects.get(dept_id=dept_id), None
    except Dept.DoesNotExist:
        return None, _error_response("존재하지 않는 부서입니다.", 400)


def _serialize_member(user: User) -> dict:
    return {
        "user_id": user.user_id,
        "name": user.name,
        "dept_id": user.dept.dept_id,
        "dept_name": user.dept.dept_name,
        "birth_date": user.birth_date.strftime("%Y%m%d"),
        "work_part": user.work_part,
    }


@require_http_methods(["POST"])
@_admin_api_required
def admin_member_create(request):
    """부서원 추가 API"""
    try:
        data = json.loads(request.body)

        user_id = data.get("user_id", "").strip()
        name = data.get("name", "").strip()
        dept_id = data.get("dept_id")
        birth_date = data.get("birth_date", "").strip()
        work_part = data.get("work_part", "").strip()
        admin_yn = data.get("admin_yn", False)

        if not all([user_id, name, dept_id, birth_date, work_part]):
            return _error_response("모든 필드를 입력해주세요.", 400)

        if not re.match(r"^[a-z0-9]{6,20}$", user_id):
            return _error_response("ID는 영소문자와 숫자로 구성되며 6~20자여야 합니다.", 400)

        if User.objects.filter(user_id=user_id).exists():
            return _error_response("이미 존재하는 ID입니다.", 400)

        err = _validate_name(name) or _validate_work_part(work_part)
        if err is not None:
            return err

        dept, err = _get_dept(dept_id)
        if err is not None:
            return err

        birth_date_obj, err = _parse_birth_date(birth_date)
        if err is not None:
            return err

        user = User.objects.create(
            user_id=user_id,
            name=name,
            dept=dept,
            birth_date=birth_date_obj,
            work_part=work_part,
            password=make_password(birth_date),
            admin_yn=admin_yn,
            status=User.STATUS_ACTIVE,
        )

        return JsonResponse(
            {
                "success": True,
                "message": "부서원이 추가되었습니다.",
                "member": _serialize_member(user),
            }
        )

    except json.JSONDecodeError:
        return _error_response("잘못된 요청입니다.", 400)
    except Exception as e:
        return _error_response(f"오류가 발생했습니다: {str(e)}", 500)


@require_http_methods(["PUT"])
@_admin_api_required
def admin_member_update(request, user_id):
    """부서원 수정 API"""
    try:
        try:
            user = User.objects.get(user_id=user_id)
        except User.DoesNotExist:
            return _error_response("존재하지 않는 사용자입니다.", 404)

        data = json.loads(request.body)

        if "name" in data:
            name = data["name"].strip()
            err = _validate_name(name)
            if err is not None:
                return err
            user.name = name

        if "dept_id" in data:
            dept, err = _get_dept(data["dept_id"])
            if err is not None:
                return err
            user.dept = dept

        if "birth_date" in data:
            birth_date = data["birth_date"].strip()
            birth_date_obj, err = _parse_birth_date(birth_date)
            if err is not None:
                return err
            user.birth_date = birth_date_obj

        if "work_part" in data:
            work_part = data["work_part"].strip()
            err = _validate_work_part(work_part)
            if err is not None:
                return err
            user.work_part = work_part

        if "admin_yn" in data:
            user.admin_yn = data["admin_yn"]

        user.save()

        return JsonResponse(
            {
                "success": True,
                "message": "부서원 정보가 수정되었습니다.",
                "member": _serialize_member(user),
            }
        )

    except json.JSONDecodeError:
        return _error_response("잘못된 요청입니다.", 400)
    except Exception as e:
        return _error_response(f"오류가 발생했습니다: {str(e)}", 500)


@require_http_methods(["DELETE"])
@_admin_api_required
def admin_member_delete(request, user_id):
    """부서원 삭제(비활성화 또는 영구 삭제) API"""
    try:
        user = User.objects.get(user_id=user_id)

        permanent = False
        if request.body:
            try:
                data = json.loads(request.body)
                permanent = data.get("permanent", False)
            except json.JSONDecodeError:
                pass

        if permanent:
            user.delete()
            message = "부서원이 영구 삭제되었습니다."
        else:
            user.status = User.STATUS_INACTIVE
            user.save()
            message = "부서원이 비활성화되었습니다."

        return JsonResponse({"success": True, "message": message})

    except User.DoesNotExist:
        return _error_response("존재하지 않는 사용자입니다.", 404)
    except Exception as e:
        return _error_response(f"오류가 발생했습니다: {str(e)}", 500)


@require_http_methods(["POST"])
@_admin_api_required
def admin_member_reset_password(request, user_id):
    """부서원 비밀번호 초기화 API (생년월일로 초기화)"""
    try:
        user = User.objects.get(user_id=user_id)

        birth_date_str = user.birth_date.strftime("%Y%m%d")
        user.password = make_password(birth_date_str)
        user.save()

        return JsonResponse(
            {
                "success": True,
                "message": f"비밀번호가 생년월일({birth_date_str})로 초기화되었습니다.",
            }
        )

    except User.DoesNotExist:
        return _error_response("존재하지 않는 사용자입니다.", 404)
    except Exception as e:
        return _error_response(f"오류가 발생했습니다: {str(e)}", 500)
