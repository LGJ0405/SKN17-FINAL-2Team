import json
import os
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings
from django.utils import timezone

from meetings.models import Attendee, Meeting, S3File, Task
from meetings.services.export import extract_major_agenda, parse_minutes_sections
from meetings.services.permissions import (
    base_meeting_list_queryset,
    build_meeting_list_context,
    has_meeting_view_permission,
)
from meetings.services.transcript import (
    extract_structured_tasks,
    normalize_summary_text,
    parse_summary_agendas,
    render_transcript_html,
    transcript_to_plain_text,
)
from users.models import Dept, User


class PollingFlowTests(TestCase):
    def setUp(self):
        self.dept = Dept.objects.create(dept_name="개발팀")
        self.user = User.objects.create(
            user_id="test_user",
            dept=self.dept,
            name="홍길동",
            password="hashed_password",
            work_part="백엔드",
            birth_date=date(1995, 1, 1),
            admin_yn=False,
        )
        self.s3_file = S3File.objects.create(
            s3_key="meetings/test/sample.wav",
            original_name="sample.wav",
            delete_at=timezone.now() + timedelta(days=7),
        )
        self.meeting = Meeting.objects.create(
            host=self.user,
            title="폴링 테스트 회의",
            meet_date_time=datetime(2026, 11, 21, 14, 0, 0),
            place="회의실 A",
            record_url=self.s3_file,
            domain="IT",
        )

    @staticmethod
    def _mock_response(status_code, payload):
        res = MagicMock()
        res.status_code = status_code
        res.json.return_value = payload
        res.text = json.dumps(payload, ensure_ascii=False)
        return res

    @patch("meetings.views.get_job_status")
    @patch("meetings.views.start_stt_job")
    @patch("meetings.views.get_presigned_url", return_value="https://s3.example.com/sample.wav")
    def test_stt_polling_full_flow(self, mock_presigned, mock_start_stt, mock_get_status):
        mock_start_stt.return_value = self._mock_response(
            200,
            {"success": True, "job_id": "stt-job-123", "status": "queued"},
        )

        # 1. 최초 요청 -> 작업 시작 및 job_id 반환
        res1 = self.client.get(f"/meetings/{self.meeting.meeting_id}/transcript/prepare/")
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json(), {"status": "processing", "job_id": "stt-job-123"})

        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.transcript_status, "processing")

        # 2. 폴링 (진행 중)
        mock_get_status.return_value = self._mock_response(
            200,
            {"success": True, "job_id": "stt-job-123", "status": "processing", "result": None, "error": None},
        )
        res2 = self.client.get(f"/meetings/{self.meeting.meeting_id}/transcript/prepare/?job_id=stt-job-123")
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json(), {"status": "processing", "job_id": "stt-job-123"})

        # 3. 폴링 (완료)
        mock_get_status.return_value = self._mock_response(
            200,
            {
                "success": True,
                "job_id": "stt-job-123",
                "status": "done",
                "result": {
                    "success": True,
                    "data": {
                        "full_text": [{"SPEAKER_00": "안녕하세요. 회의를 시작하겠습니다."}],
                    },
                },
                "error": None,
            },
        )
        res3 = self.client.get(f"/meetings/{self.meeting.meeting_id}/transcript/prepare/?job_id=stt-job-123")
        self.assertEqual(res3.status_code, 200)
        self.assertEqual(res3.json(), {"status": "done"})

        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.transcript_status, "done")
        self.assertIn("SPEAKER_00", self.meeting.transcript)

        # 4. 이미 전사본이 있을 때 재요청 -> 즉시 done 반환 (Runpod 재호출 없음)
        mock_start_stt.reset_mock()
        res4 = self.client.get(f"/meetings/{self.meeting.meeting_id}/transcript/prepare/")
        self.assertEqual(res4.status_code, 200)
        self.assertEqual(res4.json(), {"status": "done"})
        mock_start_stt.assert_not_called()

    @patch("meetings.views.get_job_status")
    def test_stt_polling_error_and_not_found(self, mock_get_status):
        mock_get_status.return_value = self._mock_response(404, {"detail": {"error": "Job not found"}})
        res_404 = self.client.get(f"/meetings/{self.meeting.meeting_id}/transcript/prepare/?job_id=missing")
        self.assertEqual(res_404.status_code, 404)
        self.assertEqual(res_404.json()["status"], "error")

        mock_get_status.return_value = self._mock_response(
            200,
            {"success": True, "job_id": "stt-err", "status": "error", "result": None, "error": "CUDA OOM"},
        )
        res_err = self.client.get(f"/meetings/{self.meeting.meeting_id}/transcript/prepare/?job_id=stt-err")
        self.assertEqual(res_err.status_code, 500)
        self.assertEqual(res_err.json(), {"status": "error", "message": "CUDA OOM"})

        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.transcript_status, "error")
        self.assertEqual(self.meeting.transcript_error, "CUDA OOM")

    @patch("meetings.views.get_job_status")
    @patch("meetings.views.start_sllm_job")
    def test_sllm_polling_full_flow(self, mock_start_sllm, mock_get_status):
        self.meeting.transcript = json.dumps([{"홍길동": "다음 주까지 API 폴링 구조를 배포합니다."}], ensure_ascii=False)
        self.meeting.save(update_fields=["transcript"])

        mock_start_sllm.return_value = self._mock_response(
            200,
            {"success": True, "job_id": "sllm-job-456", "status": "queued"},
        )

        # 1. 최초 요청 -> 작업 시작
        res1 = self.client.get(f"/meetings/{self.meeting.meeting_id}/sllm/prepare/")
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json(), {"status": "processing", "job_id": "sllm-job-456"})
        mock_start_sllm.assert_called_once()
        _, kwargs = mock_start_sllm.call_args
        self.assertEqual(kwargs.get("domain"), ["IT"])

        # 2. 폴링 (진행 중)
        mock_get_status.return_value = self._mock_response(
            200,
            {"success": True, "job_id": "sllm-job-456", "status": "processing", "result": None, "error": None},
        )
        res2 = self.client.get(f"/meetings/{self.meeting.meeting_id}/sllm/prepare/?job_id=sllm-job-456")
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json(), {"status": "processing", "job_id": "sllm-job-456"})

        # 3. 폴링 (완료 -> summary + Task DB 저장)
        mock_get_status.return_value = self._mock_response(
            200,
            {
                "success": True,
                "job_id": "sllm-job-456",
                "status": "done",
                "result": {
                    "full_summary": "API 폴링 구조 배포 논의",
                    "full_tasks": [
                        {
                            "task": "API 폴링 구조 배포",
                            "assignee": "홍길동",
                            "due_date": "2026-11-28",
                        }
                    ],
                },
                "error": None,
            },
        )
        res3 = self.client.get(f"/meetings/{self.meeting.meeting_id}/sllm/prepare/?job_id=sllm-job-456")
        self.assertEqual(res3.status_code, 200)
        self.assertEqual(res3.json(), {"status": "done"})

        self.meeting.refresh_from_db()
        self.assertIn("API 폴링 구조 배포 논의", self.meeting.summary)
        tasks = list(Task.objects.filter(meeting=self.meeting))
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].assignee, self.user)

    @override_settings(
        STORAGES={
            "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
            "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
        }
    )
    def test_rendering_views_render_with_job_poller(self):
        session = self.client.session
        session["login_user_id"] = self.user.user_id
        session.save()

        stt_res = self.client.get(f"/meetings/{self.meeting.meeting_id}/rendering/stt/")
        self.assertEqual(stt_res.status_code, 200)
        self.assertContains(stt_res, "job_poller.js")
        self.assertContains(stt_res, "startJobPolling")

        sllm_res = self.client.get(f"/meetings/{self.meeting.meeting_id}/rendering/sllm/")
        self.assertEqual(sllm_res.status_code, 200)
        self.assertContains(sllm_res, "job_poller.js")
        self.assertContains(sllm_res, "startJobPolling")


class ServicesUnitTests(TestCase):
    def setUp(self):
        self.dept = Dept.objects.create(dept_name="기획팀")
        self.other_dept = Dept.objects.create(dept_name="영업팀")
        self.host = User.objects.create(
            user_id="host01",
            dept=self.dept,
            name="김주최",
            password="pw",
            work_part="PM",
            birth_date=date(1990, 5, 1),
            admin_yn=False,
        )
        self.same_dept_user = User.objects.create(
            user_id="member01",
            dept=self.dept,
            name="박팀원",
            password="pw",
            work_part="기획",
            birth_date=date(1992, 3, 2),
            admin_yn=False,
        )
        self.other_user = User.objects.create(
            user_id="other01",
            dept=self.other_dept,
            name="이타팀",
            password="pw",
            work_part="영업",
            birth_date=date(1993, 4, 3),
            admin_yn=False,
        )
        self.meeting = Meeting.objects.create(
            host=self.host,
            title="기획 회의",
            meet_date_time=datetime(2026, 11, 21, 10, 0, 0),
            place="3층",
            domain="마케팅",
            private_yn=False,
        )
        Attendee.objects.create(meeting=self.meeting, user=self.host)

    def test_transcript_helpers(self):
        raw = json.dumps([{"김주최": "안녕하세요"}, {"박팀원": "반갑습니다"}], ensure_ascii=False)
        self.assertEqual(transcript_to_plain_text(raw), "김주최: 안녕하세요\n박팀원: 반갑습니다")
        self.assertIn("<strong>김주최</strong>", render_transcript_html(raw))

        summary_json = normalize_summary_text('```json\n{"agendas": [{"agenda": "일정", "summary": "다음주"}]}\n```')
        agendas = parse_summary_agendas(summary_json)
        self.assertEqual(len(agendas), 1)
        self.assertEqual(agendas[0]["agenda"], "일정")

        structured = extract_structured_tasks(
            [{"task": "문서화", "assignee": "김주최", "due_date": "2026.11.30"}]
        )
        self.assertEqual(structured[0]["description"], "문서화")
        self.assertEqual(structured[0]["due_date"], date(2026, 11, 30))

    def test_permissions_and_queryset_helpers(self):
        attendees = list(self.meeting.attendees.select_related("user", "user__dept").all())
        self.assertTrue(has_meeting_view_permission(self.meeting, attendees, "member01", self.dept.dept_id))
        self.assertFalse(has_meeting_view_permission(self.meeting, attendees, "other01", self.other_dept.dept_id))

        rows, login_user = build_meeting_list_context(base_meeting_list_queryset(), "member01")
        self.assertEqual(login_user.user_id, "member01")
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["is_joined"])

    def test_export_html_parsers(self):
        html = (
            '<div data-minutes-section="base">'
            '<table><tr><th>주요안건</th><td><div class="minutes-editbox">배포 점검</div></td></tr></table>'
            '</div>'
            '<div data-minutes-section="contents">회의 내용\n폴링 점검 완료</div>'
        )
        sections = parse_minutes_sections(html)
        self.assertIn("contents", sections)
        self.assertEqual(extract_major_agenda(html), "배포 점검")

    def test_runpod_base_url_override(self):
        from meetings.utils import runpod

        with patch.dict(os.environ, {"RUNPOD_BASE_URL": "http://127.0.0.1:8001"}):
            self.assertEqual(runpod._base_url(), "http://127.0.0.1:8001/")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RUNPOD_BASE_URL", None)
            with override_settings(POD_ID="testpod123", RUNPOD_BASE_URL=None):
                self.assertEqual(runpod._base_url(), "https://testpod123-8000.proxy.runpod.net/")

    @patch("meetings.views.get_job_status")
    @patch("meetings.views.start_stt_job_from_bytes")
    def test_local_storage_upload_and_bytes_polling(self, mock_start_bytes, mock_get_status):
        from django.core.files.uploadedfile import SimpleUploadedFile
        from meetings.utils.s3_upload import get_local_audio_path, upload_raw_file_bytes

        local_path = None
        try:
            with patch.dict(os.environ, {"USE_LOCAL_STORAGE": "true"}):
                local_key = upload_raw_file_bytes(b"RIFF....WAVEfmt ", "local_test.wav", 3600)
                self.assertTrue(local_key.startswith("local/"))
                local_path = get_local_audio_path(local_key)
                self.assertTrue(local_path.exists())

                local_s3 = S3File.objects.get(s3_key=local_key)
                meeting = Meeting.objects.create(
                    host=self.host,
                    title="로컬 저장 테스트 회의",
                    meet_date_time=datetime(2026, 11, 24, 2, 0, 0),
                    place="로컬",
                    record_url=local_s3,
                    domain="IT",
                )

                start_res = MagicMock()
                start_res.status_code = 200
                start_res.json.return_value = {"success": True, "job_id": "local-stt-1", "status": "queued"}
                start_res.text = "{}"
                mock_start_bytes.return_value = start_res

                res1 = self.client.get(f"/meetings/{meeting.meeting_id}/transcript/prepare/")
                self.assertEqual(res1.status_code, 200)
                self.assertEqual(res1.json(), {"status": "processing", "job_id": "local-stt-1"})
                mock_start_bytes.assert_called_once_with(b"RIFF....WAVEfmt ")

                done_res = MagicMock()
                done_res.status_code = 200
                done_res.json.return_value = {
                    "success": True,
                    "job_id": "local-stt-1",
                    "status": "done",
                    "result": {"success": True, "data": {"full_text": [{"SPEAKER_00": "로컬 테스트 완료"}]}},
                    "error": None,
                }
                done_res.text = "{}"
                mock_get_status.return_value = done_res

                res2 = self.client.get(f"/meetings/{meeting.meeting_id}/transcript/prepare/?job_id=local-stt-1")
                self.assertEqual(res2.status_code, 200)
                self.assertEqual(res2.json(), {"status": "done"})
        finally:
            if local_path and local_path.exists():
                local_path.unlink()
