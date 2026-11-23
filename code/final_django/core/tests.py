import json
from datetime import date

from django.test import TestCase

from users.models import Dept, User


class AdminMemberApiTests(TestCase):
    def setUp(self):
        self.dept = Dept.objects.create(dept_name="개발팀")
        self.admin_user = User.objects.create(
            user_id="admin01",
            dept=self.dept,
            name="관리자",
            password="pw",
            work_part="관리",
            birth_date=date(1988, 1, 1),
            admin_yn=True,
        )

    def _login_as_admin(self):
        session = self.client.session
        session["login_user_id"] = self.admin_user.user_id
        session["login_user_admin"] = True
        session.save()

    def test_admin_api_requires_admin_session(self):
        res = self.client.post(
            "/admin/member/create/",
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.json()["message"], "권한이 없습니다.")

    def test_admin_member_lifecycle(self):
        self._login_as_admin()

        # 1. 추가
        create_res = self.client.post(
            "/admin/member/create/",
            data=json.dumps(
                {
                    "user_id": "newuser1",
                    "name": "신규유저",
                    "dept_id": self.dept.dept_id,
                    "birth_date": "19950505",
                    "work_part": "백엔드",
                    "admin_yn": False,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(create_res.status_code, 200)
        self.assertTrue(create_res.json()["success"])

        # 2. 수정
        update_res = self.client.put(
            "/admin/member/newuser1/update/",
            data=json.dumps({"name": "수정유저", "work_part": "Backend"}),
            content_type="application/json",
        )
        self.assertEqual(update_res.status_code, 200)
        self.assertEqual(update_res.json()["member"]["name"], "수정유저")

        # 3. 비밀번호 초기화
        reset_res = self.client.post("/admin/member/newuser1/reset-password/")
        self.assertEqual(reset_res.status_code, 200)
        self.assertIn("19950505", reset_res.json()["message"])

        # 4. 비활성화 삭제
        del_res = self.client.delete(
            "/admin/member/newuser1/delete/",
            data=json.dumps({"permanent": False}),
            content_type="application/json",
        )
        self.assertEqual(del_res.status_code, 200)
        user_obj = User.objects.get(user_id="newuser1")
        self.assertEqual(user_obj.status, User.STATUS_INACTIVE)
