import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class JobManager:
    """인메모리 백그라운드 작업 큐 및 상태 조회 관리자."""

    def __init__(
        self,
        max_workers: int = 2,
        ttl_seconds: int = 3600,
        max_jobs: int = 200,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._ttl_seconds = ttl_seconds
        self._max_jobs = max_jobs

    @staticmethod
    def extract_error_message(res: Any) -> str:
        """작업 결과 딕셔너리에서 사용자에게 노출할 오류 메시지를 추출한다."""
        if isinstance(res, dict):
            err = res.get("error")
            if isinstance(err, dict):
                return str(err.get("detail") or err.get("message") or err)
            if err:
                return str(err)
            if res.get("message"):
                return str(res["message"])
        return "작업 처리 중 오류가 발생했습니다."

    def _cleanup_locked(self, now: float) -> None:
        expired_ids = [
            jid
            for jid, job in self._jobs.items()
            if job["status"] in ("done", "error")
            and (now - job["updated_at"]) > self._ttl_seconds
        ]
        for jid in expired_ids:
            self._jobs.pop(jid, None)

        if len(self._jobs) > self._max_jobs:
            finished = sorted(
                (
                    (job["updated_at"], jid)
                    for jid, job in self._jobs.items()
                    if job["status"] in ("done", "error")
                ),
                key=lambda x: x[0],
            )
            for _, jid in finished[: len(self._jobs) - self._max_jobs]:
                self._jobs.pop(jid, None)

    def _run(self, job_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "processing"
                self._jobs[job_id]["updated_at"] = time.time()

        try:
            res = fn(*args, **kwargs)
            now = time.time()
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                if isinstance(res, dict) and res.get("success") is False:
                    job["status"] = "error"
                    job["result"] = res
                    job["error"] = self.extract_error_message(res)
                else:
                    job["status"] = "done"
                    job["result"] = res
                    job["error"] = None
                job["updated_at"] = now
        except Exception as exc:
            now = time.time()
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["status"] = "error"
                    job["error"] = str(exc)
                    job["updated_at"] = now

    def submit(self, job_type: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        """새 비동기 작업을 큐에 등록하고 job_id를 반환한다."""
        job_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._cleanup_locked(now)
            self._jobs[job_id] = {
                "job_id": job_id,
                "type": job_type,
                "status": "queued",
                "result": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
        self._executor.submit(self._run, job_id, fn, *args, **kwargs)
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        """job_id에 해당하는 작업 상태 복사본을 반환한다."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job["job_id"],
                "type": job["type"],
                "status": job["status"],
                "result": job["result"],
                "error": job["error"],
            }
