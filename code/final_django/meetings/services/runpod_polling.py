import requests
from django.http import JsonResponse

DOMAIN_API_MAP = {
    "마케팅": "Marketing / Economy",
    "IT": "IT",
    "디자인": "Design",
    "회계": "Accounting",
}


def extract_runpod_error(res_json, fallback_text=""):
    """Runpod 응답 JSON에서 에러 메시지를 일관되게 추출한다."""
    if isinstance(res_json, dict):
        msg = res_json.get("error") or res_json.get("message")
        if isinstance(msg, dict):
            msg = msg.get("detail") or msg.get("message") or msg.get("error")
        if msg:
            return str(msg)
        detail = res_json.get("detail")
        if isinstance(detail, dict):
            msg = detail.get("message") or detail.get("error") or detail.get("detail")
            if msg:
                return str(msg)
        elif detail:
            return str(detail)
    return fallback_text


def _safe_json(res):
    try:
        return res.json()
    except Exception:
        return {}


def start_runpod_job(
    start_fn,
    *,
    comm_error_prefix: str,
    start_error_msg: str,
    preview_len: int = 300,
):
    """비동기 Runpod 작업을 시작하고 (job_id, error_response)를 반환한다."""
    try:
        res = start_fn()
    except requests.RequestException as e:
        return None, JsonResponse(
            {"status": "error", "message": f"{comm_error_prefix}: {e}"},
            status=502,
        )

    res_json = _safe_json(res)
    if res.status_code not in (200, 202) or not res_json.get("job_id"):
        message = extract_runpod_error(res_json, getattr(res, "text", "")[:preview_len])
        return None, JsonResponse(
            {"status": "error", "message": message or start_error_msg},
            status=500,
        )

    return res_json["job_id"], None


def poll_runpod_job(
    status_fn,
    job_id: str,
    *,
    comm_error_prefix: str,
    not_found_msg: str,
    status_error_msg: str,
    preview_len: int = 300,
):
    """Runpod 작업 상태를 조회하고 (job_status, res_json, error_response)를 반환한다."""
    try:
        res = status_fn(job_id)
    except requests.RequestException as e:
        return None, {}, JsonResponse(
            {"status": "error", "message": f"{comm_error_prefix}: {e}"},
            status=502,
        )

    res_json = _safe_json(res)
    if res.status_code == 404:
        return None, res_json, JsonResponse(
            {"status": "error", "message": not_found_msg},
            status=404,
        )

    if res.status_code != 200:
        message = extract_runpod_error(res_json, getattr(res, "text", "")[:preview_len])
        return None, res_json, JsonResponse(
            {"status": "error", "message": message or status_error_msg},
            status=500,
        )

    return res_json.get("status"), res_json, None
