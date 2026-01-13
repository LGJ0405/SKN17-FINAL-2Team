from django.shortcuts import redirect
from django.views.generic import TemplateView
from django.views.decorators.cache import never_cache
from django.utils.decorators import method_decorator
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib.auth.hashers import make_password
import json
import re

from users.models import User, Dept
from datetime import date, datetime, timedelta

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
        # 세션단에서 검증
        if not request.session.get("login_user_admin"):
            return redirect("/")
        try:
            # DB단에서 추가 검증
            user = User.objects.get(user_id=request.session.get("login_user_id"))
            if not user.admin_yn:
                return redirect("/")
        except:
            return redirect("/")
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["departments"] = Dept.objects.all().order_by("dept_name")
        # 전체 부서원 목록을 가져와서 템플릿에 전달
        context["members"] = User.objects.filter(
            status=User.STATUS_ACTIVE
        ).select_related("dept").order_by("dept__dept_name", "name")
        return context


@require_http_methods(["POST"])
def admin_member_create(request):
    """부서원 추가 API"""
    if not request.session.get("login_user_admin"):
        return JsonResponse({"success": False, "message": "권한이 없습니다."}, status=403)

    try:
        data = json.loads(request.body)

        # 필수 필드 검증
        user_id = data.get("user_id", "").strip()
        name = data.get("name", "").strip()
        dept_id = data.get("dept_id")
        birth_date = data.get("birth_date", "").strip()
        work_part = data.get("work_part", "").strip()
        admin_yn = data.get("admin_yn", False)

        if not all([user_id, name, dept_id, birth_date, work_part]):
            return JsonResponse({"success": False, "message": "모든 필드를 입력해주세요."}, status=400)

        # ID 유효성 검사: 영소문자+숫자, 6~20자
        if not re.match(r'^[a-z0-9]{6,20}$', user_id):
            return JsonResponse({"success": False, "message": "ID는 영소문자와 숫자로 구성되며 6~20자여야 합니다."}, status=400)

        # 중복 ID 체크
        if User.objects.filter(user_id=user_id).exists():
            return JsonResponse({"success": False, "message": "이미 존재하는 ID입니다."}, status=400)

        # 이름 유효성 검사: 한글만, 32자까지
        if not re.match(r'^[가-힣]{1,32}$', name):
            return JsonResponse({"success": False, "message": "이름은 한글만 입력 가능하며 1~32자여야 합니다."}, status=400)

        # 업무 유효성 검사: 한글+영어, 15자까지
        if not re.match(r'^[가-힣a-zA-Z\s]{1,15}$', work_part):
            return JsonResponse({"success": False, "message": "업무는 한글 또는 영어만 입력 가능하며 1~15자여야 합니다."}, status=400)

        # 부서 확인
        try:
            dept = Dept.objects.get(dept_id=dept_id)
        except Dept.DoesNotExist:
            return JsonResponse({"success": False, "message": "존재하지 않는 부서입니다."}, status=400)

        # 생년월일 파싱 및 유효성 검사 (8자리 YYYYMMDD)
        if not re.match(r'^\d{8}$', birth_date):
            return JsonResponse({"success": False, "message": "생년월일은 8자리 숫자로 입력해주세요."}, status=400)

        try:
            birth_date_obj = datetime.strptime(birth_date, "%Y%m%d").date()
        except ValueError:
            return JsonResponse({"success": False, "message": "유효하지 않은 생년월일입니다."}, status=400)

        # 생년월일 범위 검사: 현재-100년 ~ 만 19세 이상
        today = date.today()
        hundred_years_ago = date(today.year - 100, today.month, today.day)
        nineteen_years_ago = date(today.year - 19, today.month, today.day)

        if birth_date_obj < hundred_years_ago:
            return JsonResponse({"success": False, "message": "생년월일은 현재로부터 100년 이내여야 합니다."}, status=400)

        if birth_date_obj > nineteen_years_ago:
            return JsonResponse({"success": False, "message": "만 19세 이상만 등록 가능합니다."}, status=400)

        # 사용자 생성 (비밀번호는 생년월일과 동일)
        user = User.objects.create(
            user_id=user_id,
            name=name,
            dept=dept,
            birth_date=birth_date_obj,
            work_part=work_part,
            password=make_password(birth_date),  # 생년월일을 초기 비밀번호로
            admin_yn=admin_yn,
            status=User.STATUS_ACTIVE
        )

        return JsonResponse({
            "success": True,
            "message": "부서원이 추가되었습니다.",
            "member": {
                "user_id": user.user_id,
                "name": user.name,
                "dept_id": user.dept.dept_id,
                "dept_name": user.dept.dept_name,
                "birth_date": user.birth_date.strftime("%Y%m%d"),
                "work_part": user.work_part
            }
        })

    except json.JSONDecodeError:
        return JsonResponse({"success": False, "message": "잘못된 요청입니다."}, status=400)
    except Exception as e:
        return JsonResponse({"success": False, "message": f"오류가 발생했습니다: {str(e)}"}, status=500)


@require_http_methods(["PUT"])
def admin_member_update(request, user_id):
    """부서원 수정 API"""
    if not request.session.get("login_user_admin"):
        return JsonResponse({"success": False, "message": "권한이 없습니다."}, status=403)

    try:
        # 사용자 조회
        try:
            user = User.objects.get(user_id=user_id)
        except User.DoesNotExist:
            return JsonResponse({"success": False, "message": "존재하지 않는 사용자입니다."}, status=404)

        data = json.loads(request.body)

        # 수정 가능한 필드만 업데이트
        if "name" in data:
            name = data["name"].strip()
            # 이름 유효성 검사
            if not re.match(r'^[가-힣]{1,32}$', name):
                return JsonResponse({"success": False, "message": "이름은 한글만 입력 가능하며 1~32자여야 합니다."}, status=400)
            user.name = name

        if "dept_id" in data:
            try:
                dept = Dept.objects.get(dept_id=data["dept_id"])
                user.dept = dept
            except Dept.DoesNotExist:
                return JsonResponse({"success": False, "message": "존재하지 않는 부서입니다."}, status=400)

        if "birth_date" in data:
            birth_date = data["birth_date"].strip()

            # 생년월일 유효성 검사
            if not re.match(r'^\d{8}$', birth_date):
                return JsonResponse({"success": False, "message": "생년월일은 8자리 숫자로 입력해주세요."}, status=400)

            try:
                birth_date_obj = datetime.strptime(birth_date, "%Y%m%d").date()
            except ValueError:
                return JsonResponse({"success": False, "message": "유효하지 않은 생년월일입니다."}, status=400)

            # 생년월일 범위 검사
            today = date.today()
            hundred_years_ago = date(today.year - 100, today.month, today.day)
            nineteen_years_ago = date(today.year - 19, today.month, today.day)

            if birth_date_obj < hundred_years_ago:
                return JsonResponse({"success": False, "message": "생년월일은 현재로부터 100년 이내여야 합니다."}, status=400)

            if birth_date_obj > nineteen_years_ago:
                return JsonResponse({"success": False, "message": "만 19세 이상만 등록 가능합니다."}, status=400)

            user.birth_date = birth_date_obj

        if "work_part" in data:
            work_part = data["work_part"].strip()
            # 업무 유효성 검사
            if not re.match(r'^[가-힣a-zA-Z\s]{1,15}$', work_part):
                return JsonResponse({"success": False, "message": "업무는 한글 또는 영어만 입력 가능하며 1~15자여야 합니다."}, status=400)
            user.work_part = work_part

        if "admin_yn" in data:
            user.admin_yn = data["admin_yn"]

        user.save()

        return JsonResponse({
            "success": True,
            "message": "부서원 정보가 수정되었습니다.",
            "member": {
                "user_id": user.user_id,
                "name": user.name,
                "dept_id": user.dept.dept_id,
                "dept_name": user.dept.dept_name,
                "birth_date": user.birth_date.strftime("%Y%m%d"),
                "work_part": user.work_part
            }
        })

    except json.JSONDecodeError:
        return JsonResponse({"success": False, "message": "잘못된 요청입니다."}, status=400)
    except Exception as e:
        return JsonResponse({"success": False, "message": f"오류가 발생했습니다: {str(e)}"}, status=500)


@require_http_methods(["DELETE"])
def admin_member_delete(request, user_id):
    """부서원 삭제(비활성화 또는 영구 삭제) API"""
    if not request.session.get("login_user_admin"):
        return JsonResponse({"success": False, "message": "권한이 없습니다."}, status=403)

    try:
        user = User.objects.get(user_id=user_id)

        # 요청 본문에서 permanent 플래그 확인
        permanent = False
        if request.body:
            try:
                data = json.loads(request.body)
                permanent = data.get("permanent", False)
            except json.JSONDecodeError:
                pass

        if permanent:
            # 영구 삭제
            user.delete()
            message = "부서원이 영구 삭제되었습니다."
        else:
            # 비활성화 처리
            user.status = User.STATUS_INACTIVE
            user.save()
            message = "부서원이 비활성화되었습니다."

        return JsonResponse({
            "success": True,
            "message": message
        })

    except User.DoesNotExist:
        return JsonResponse({"success": False, "message": "존재하지 않는 사용자입니다."}, status=404)
    except Exception as e:
        return JsonResponse({"success": False, "message": f"오류가 발생했습니다: {str(e)}"}, status=500)

    
@require_http_methods(["POST"])
def admin_member_reset_password(request, user_id):
    """부서원 비밀번호 초기화 API (생년월일로 초기화)"""
    if not request.session.get("login_user_admin"):
        return JsonResponse({"success": False, "message": "권한이 없습니다."}, status=403)

    try:
        user = User.objects.get(user_id=user_id)

        # 생년월일을 8자리 문자열로 변환하여 비밀번호로 설정
        birth_date_str = user.birth_date.strftime("%Y%m%d")
        user.password = make_password(birth_date_str)
        user.save()

        return JsonResponse({
            "success": True,
            "message": f"비밀번호가 생년월일({birth_date_str})로 초기화되었습니다."
        })

    except User.DoesNotExist:
        return JsonResponse({"success": False, "message": "존재하지 않는 사용자입니다."}, status=404)
    except Exception as e:
        return JsonResponse({"success": False, "message": f"오류가 발생했습니다: {str(e)}"}, status=500)