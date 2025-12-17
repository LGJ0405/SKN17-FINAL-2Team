from django.apps import AppConfig
import threading
import os, time

class MeetingsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.meetings'

    def ready(self):
        if os.environ.get("RUN_MAIN") != "true":
            return

        if getattr(self, "batch_started", False):
            return
        self.batch_started = True

        t = threading.Thread(target=self.batch_runner, daemon=True)
        t.start()

    @staticmethod
    def batch_runner():
        # 첫 5분 대기
        time.sleep(300)

        while True:
            from .batch import delete_expired_s3_files
            deleted = delete_expired_s3_files()
            if deleted > 0:
                print(f"[배치] 삭제된 파일 수: {deleted}")
            else:
                print("[배치] 삭제할 파일 없음")

            time.sleep(3600)  # 1시간 대기