import io
import os
import uuid
from datetime import timedelta
from pathlib import Path

import boto3
from botocore.config import Config
from django.conf import settings
from django.utils import timezone

from meetings.models import S3File

CONTENT_TYPE_MAP = {
    "wav": "audio/wav",
}

LOCAL_STORAGE_PREFIX = "local/"

BUCKET_NAME = settings.AWS_STORAGE_BUCKET_NAME
REGION_NAME = settings.AWS_S3_REGION_NAME


def is_local_storage_enabled() -> bool:
    env_val = os.getenv("USE_LOCAL_STORAGE", "").strip().lower()
    if env_val in ("1", "true", "yes"):
        return True
    if env_val in ("0", "false", "no"):
        return False
    return bool(getattr(settings, "USE_LOCAL_STORAGE", False))


def is_local_storage_key(s3_key: str) -> bool:
    return str(s3_key or "").startswith(LOCAL_STORAGE_PREFIX)


def get_local_audio_path(s3_key: str) -> Path:
    base_dir = (Path(settings.BASE_DIR) / "media" / "local").resolve()
    target = (Path(settings.BASE_DIR) / "media" / s3_key).resolve()
    if base_dir != target and base_dir not in target.parents:
        raise ValueError(f"Invalid local audio key: {s3_key}")
    return target


def upload_raw_file_bytes(file_bytes: bytes, original_filename: str, delete_after_seconds: int) -> str:
    ext = original_filename.split(".")[-1].lower()
    delete_at = timezone.now() + timedelta(seconds=delete_after_seconds)

    if is_local_storage_enabled():
        local_key = f"{LOCAL_STORAGE_PREFIX}{uuid.uuid4()}.{ext}"
        local_path = get_local_audio_path(local_key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(file_bytes)

        S3File.objects.create(
            s3_key=local_key,
            original_name=original_filename,
            delete_at=delete_at,
        )
        return local_key

    s3 = boto3.client(
        "s3",
        region_name=REGION_NAME,
        endpoint_url=f"https://s3.{REGION_NAME}.amazonaws.com",
        config=Config(signature_version="s3v4"),
    )
    s3_key = f"tests/{uuid.uuid4()}.{ext}"

    content_type = CONTENT_TYPE_MAP.get(ext, "application/octet-stream")
    file_obj = io.BytesIO(file_bytes)

    s3.upload_fileobj(
        Fileobj=file_obj,
        Bucket=BUCKET_NAME,
        Key=s3_key,
        ExtraArgs={"ContentType": content_type},
    )

    S3File.objects.create(
        s3_key=s3_key,
        original_name=original_filename,
        delete_at=delete_at,
    )
    return s3_key

def get_presigned_url(s3_key: str, expires_seconds: int = 60 * 60 * 48) -> str:
    s3 = boto3.client("s3", 
        region_name=REGION_NAME,
        endpoint_url=f"https://s3.{REGION_NAME}.amazonaws.com",
        config=Config(signature_version='s3v4')
    )
    presigned_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": BUCKET_NAME, "Key": s3_key},
        ExpiresIn=expires_seconds,
    )
    return presigned_url