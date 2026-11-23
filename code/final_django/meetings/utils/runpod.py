import os

import requests
from django.conf import settings

ASYNC_REQUEST_TIMEOUT = 15
UPLOAD_REQUEST_TIMEOUT = 120


def _base_url() -> str:
    override = os.getenv("RUNPOD_BASE_URL") or getattr(settings, "RUNPOD_BASE_URL", None)
    if override:
        return override.rstrip("/") + "/"
    return f"https://{settings.POD_ID}-8000.proxy.runpod.net/"


def _normalize_domain(domain):
    if isinstance(domain, (list, tuple)):
        return list(domain)
    if domain:
        return [domain]
    return []


def _get(path: str, timeout=None):
    kwargs = {"timeout": timeout} if timeout is not None else {}
    return requests.get(_base_url() + path, **kwargs)


def _post(path: str, payload: dict, timeout=None):
    kwargs = {"timeout": timeout} if timeout is not None else {}
    return requests.post(_base_url() + path, json=payload, **kwargs)


def runpod_health():
    return _get("health", timeout=ASYNC_REQUEST_TIMEOUT).status_code


def get_stt(presigned_url):
    return _post("stt", {"audio_url": presigned_url})


def get_sllm(transcript, domain=""):
    """
    domain: string or list of strings. Normalized to list for API.
    """
    return _post(
        "inference",
        {"transcript": transcript, "domain": _normalize_domain(domain)},
    )


def start_stt_job(presigned_url):
    return _post(
        "stt/async",
        {"audio_url": presigned_url},
        timeout=ASYNC_REQUEST_TIMEOUT,
    )


def start_stt_job_from_bytes(audio_bytes: bytes):
    return requests.post(
        _base_url() + "stt/async-file",
        data=audio_bytes,
        headers={"Content-Type": "application/octet-stream"},
        timeout=UPLOAD_REQUEST_TIMEOUT,
    )


def start_sllm_job(transcript, domain=""):
    return _post(
        "inference/async",
        {"transcript": transcript, "domain": _normalize_domain(domain)},
        timeout=ASYNC_REQUEST_TIMEOUT,
    )


def get_job_status(job_id):
    return _get(f"jobs/{job_id}", timeout=ASYNC_REQUEST_TIMEOUT)
