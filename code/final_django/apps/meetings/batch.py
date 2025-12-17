# yourapp/batch.py
from django.utils import timezone
from django.conf import settings
import boto3
from .models import S3File

def delete_expired_s3_files():
    now = timezone.now()

    # ORM으로 delete_at 지난 row 조회
    expired_qs = S3File.objects.filter(delete_at__lt=now)

    if not expired_qs.exists():
        return 0  # 삭제할 것이 없음

    s3 = boto3.client("s3")
    bucket = settings.AWS_STORAGE_BUCKET_NAME

    deleted_count = 0

    # 메모리 절약을 위해 iterator 사용
    for obj in expired_qs.iterator():
        try:
            s3.delete_object(Bucket=bucket, Key=obj.s3_key)
            obj.delete()
            deleted_count += 1
        except Exception as e:
            print(f"삭제 실패: {obj.s3_key}, 오류: {e}")

    return deleted_count
