import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from job_manager import JobManager

MOCK_DELAY_SECONDS = float(os.getenv("MOCK_DELAY_SECONDS", "6"))

app = FastAPI(title="Local Mock Runpod Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

job_manager = JobManager()


class AudioRequest(BaseModel):
    audio_url: str


class InferenceRequest(BaseModel):
    transcript: str
    domain: list[str] = Field(default_factory=list)


def _mock_stt(audio_url: str) -> dict:
    time.sleep(MOCK_DELAY_SECONDS)
    full_text = [
        {"SPEAKER_00": "안녕하세요. 오늘 프로젝트 배포 일정 점검 회의를 시작하겠습니다."},
        {"SPEAKER_01": "네, API 폴링 구조로 전환한 부분 브라우저 테스트까지 진행하면 됩니다."},
        {"SPEAKER_00": "좋습니다. 내일까지 최종 확인 후 공유 부탁드립니다."},
    ]
    return {
        "success": True,
        "data": {
            "full_text": full_text,
            "segments": [
                {"speaker": "SPEAKER_00", "text": full_text[0]["SPEAKER_00"]},
                {"speaker": "SPEAKER_01", "text": full_text[1]["SPEAKER_01"]},
                {"speaker": "SPEAKER_00", "text": full_text[2]["SPEAKER_00"]},
            ],
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "raw_transcript": "\n".join(
                f"{k}: {v}" for item in full_text for k, v in item.items()
            ),
            "source_url": audio_url,
        },
    }


def _mock_inference(transcript: str, domain: list[str]) -> dict:
    time.sleep(MOCK_DELAY_SECONDS)
    domain_label = ", ".join(domain) if domain else "일반"
    return {
        "agendas": [
            {
                "agenda": f"[{domain_label}] 폴링 구조 전환 및 배포 점검",
                "agenda_description": "Runpod-Django 간 동기식 타임아웃 문제를 폴링 구조로 개선",
                "summary": {
                    "agenda_summary": "STT/SLLM 요청을 비동기 Job 큐와 3초 폴링 구조로 전환하여 장시간 요청 끊김을 해결하기로 논의함.",
                },
            }
        ],
        "tasks": [
            {
                "description": "로컬 Mock 서버 기반 브라우저 폴링 동작 확인",
                "assignee": "홍길동",
                "due": "내일까지",
                "due_date": "2026-11-22",
            }
        ],
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "mode": "mock", "timestamp": time.time()}


@app.post("/stt/async")
async def stt_async(req: AudioRequest):
    if not req.audio_url:
        raise HTTPException(status_code=422, detail="audio_url required")
    job_id = job_manager.submit("stt", _mock_stt, req.audio_url)
    return {"success": True, "job_id": job_id, "status": "queued"}


@app.post("/inference/async")
async def inference_async(req: InferenceRequest):
    if not req.transcript:
        raise HTTPException(status_code=422, detail="transcript required")
    job_id = job_manager.submit("inference", _mock_inference, req.transcript, req.domain)
    return {"success": True, "job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"success": False, "error": "Job not found"},
        )
    return {"success": True, **job}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
