from django.contrib import admin
from django.contrib.auth.hashers import make_password
from .models import Dept, User

@admin.register(Dept)
class DeptAdmin(admin.ModelAdmin):
    list_display = ("dept_id", "dept_name")
    search_fields = ("dept_name",)

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    fields = ("user_id", "dept", "name", "password", "work_part",
              "birth_date", "status", "admin_yn", "delete_at")

    def save_model(self, request, obj, form, change):
        raw_pw = obj.password
        # 변경이거나 생성일 때 비밀번호가 평문이면 해싱
        if raw_pw and not raw_pw.startswith("pbkdf2_"):
            obj.password = make_password(raw_pw)
        super().save_model(request, obj, form, change)