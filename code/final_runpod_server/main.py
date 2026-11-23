import asyncio
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from job_manager import JobManager
from sllm_tool_binding import agent_main
from whisper_pannote import run_stt_diarization

app = FastAPI(title="Runpod STT + Pyannote Server")

# CORS 정책 때문에 응답 못 받는 거 해결용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (테스트용)
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # 필요한 메서드만 허용
    allow_headers=["*"],
)

job_manager = JobManager()


def _run_inference(transcript: str, domain: list[str]) -> dict:
    domain_arg = domain[0] if domain else None
    return agent_main(domain_input=domain_arg, transcript=transcript)


class AudioRequest(BaseModel):
    audio_url: str


class InferenceRequest(BaseModel):
    transcript: str
    domain: list[str] = Field(default_factory=list)


# 포트연결 정상 적동 되었는지 확인용
@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/stt")
async def stt(req: AudioRequest):
    if not req.audio_url:
        raise HTTPException(status_code=422, detail="audio_url required")
    try:
        result = await asyncio.to_thread(run_stt_diarization, req.audio_url)
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"success": False, "error": str(e)})


@app.post("/inference")
async def inference(req: InferenceRequest):
    if not req.transcript:
        raise HTTPException(status_code=422, detail="transcript required")
    try:
        return await asyncio.to_thread(_run_inference, req.transcript, req.domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"success": False, "error": str(e)})


@app.post("/stt/async")
async def stt_async(req: AudioRequest):
    if not req.audio_url:
        raise HTTPException(status_code=422, detail="audio_url required")
    job_id = job_manager.submit("stt", run_stt_diarization, req.audio_url)
    return {"success": True, "job_id": job_id, "status": "queued"}


@app.post("/stt/async-file")
async def stt_async_file(request: Request):
    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="audio body required")
    job_id = job_manager.submit("stt", run_stt_diarization, audio_bytes=audio_bytes)
    return {"success": True, "job_id": job_id, "status": "queued"}


@app.post("/inference/async")
async def inference_async(req: InferenceRequest):
    if not req.transcript:
        raise HTTPException(status_code=422, detail="transcript required")
    job_id = job_manager.submit("inference", _run_inference, req.transcript, req.domain)
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
