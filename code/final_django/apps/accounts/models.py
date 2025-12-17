from django.db import models
from django.db.models import Q, CheckConstraint   # ✅ CHECK용 import

class Dept(models.Model):
    dept_id = models.AutoField(primary_key=True)
    dept_name = models.CharField(max_length=40)

    class Meta:
        db_table = "dept_tbl"

    def __str__(self):
        return self.dept_name

class User(models.Model):
    STATUS_ACTIVE = "A"      # 활성
    STATUS_INACTIVE = "I"    # 비활성
    STATUS_DELETE_READY = "D"  # 삭제 예정

    STATUS_CHOICES = [
        (STATUS_ACTIVE, "활성"),
        (STATUS_INACTIVE, "비활성"),
        (STATUS_DELETE_READY, "삭제 예정"),
    ]

    user_id = models.CharField(max_length=20, primary_key=True)

    dept = models.ForeignKey(
        Dept,
        on_delete=models.PROTECT,
        related_name="users",
    )

    name = models.CharField(max_length=100)
    password = models.CharField(max_length=128)
    work_part = models.CharField(max_length=50)
    birth_date = models.DateField()
    status = models.CharField(max_length=1, choices=STATUS_CHOICES, default=STATUS_ACTIVE)
    admin_yn = models.BooleanField()
    delete_at = models.DateField(null=True, blank=True)

    # 로그인 실패 카운트 (5회 실패 시 status를 'I'로 변경)
    login_fail_count = models.IntegerField(default=0)

    class Meta:
        db_table = "user_tbl"
        # ✅ DB 레벨 CHECK 제약 조건
        constraints = [
            CheckConstraint(
                condition=Q(status__in=["A", "I", "D"]),
                name="user_status_valid",
            )
        ]

    def __str__(self):
        return f"{self.user_id} / {self.name}"