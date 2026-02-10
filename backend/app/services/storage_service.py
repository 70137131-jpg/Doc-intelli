import io
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import settings
from app.core.exceptions import StorageError
from app.core.logging import get_logger

logger = get_logger(__name__)


class StorageService:
    def __init__(self):
        endpoint_url = f"{'https' if settings.minio_use_ssl else 'http'}://{settings.minio_endpoint}"
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

    def ensure_buckets(self) -> None:
        for bucket in [settings.minio_bucket_raw, settings.minio_bucket_processed]:
            try:
                self.client.head_bucket(Bucket=bucket)
            except ClientError:
                try:
                    self.client.create_bucket(Bucket=bucket)
                    logger.info(f"Created bucket: {bucket}")
                except Exception as e:
                    logger.error(f"Failed to create bucket {bucket}: {e}")

    def upload_file(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        try:
            self.client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            logger.info(f"Uploaded {key} to {bucket}")
        except Exception as e:
            raise StorageError(f"Failed to upload {key}: {e}")

    def download_file(self, bucket: str, key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except Exception as e:
            raise StorageError(f"Failed to download {key}: {e}")

    def delete_file(self, bucket: str, key: str) -> None:
        try:
            self.client.delete_object(Bucket=bucket, Key=key)
            logger.info(f"Deleted {key} from {bucket}")
        except Exception as e:
            raise StorageError(f"Failed to delete {key}: {e}")

    def generate_presigned_url(
        self, bucket: str, key: str, expires_in: int = 3600
    ) -> str:
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        except Exception as e:
            raise StorageError(f"Failed to generate presigned URL: {e}")

    def file_exists(self, bucket: str, key: str) -> bool:
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False
