"""Initialize MinIO buckets for the document intelligence platform."""

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def create_buckets():
    client = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin123",
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    buckets = ["raw-documents", "processed-documents"]

    for bucket in buckets:
        try:
            client.head_bucket(Bucket=bucket)
            print(f"Bucket '{bucket}' already exists")
        except ClientError:
            client.create_bucket(Bucket=bucket)
            print(f"Created bucket '{bucket}'")


if __name__ == "__main__":
    create_buckets()
