import boto3
from pathlib import Path
import os

BUCKET = os.getenv("S3_BUCKET")
REGION = os.getenv("AWS_REGION")

s3 = boto3.client("s3")


def upload_blog(title, content):
    key = f"blogs/{title}.md"

    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=content.encode("utf-8"),
        ContentType="text/markdown"
    )

    return f"https://{BUCKET}.s3.{REGION}.amazonaws.com/{key}"


def upload_images():
    urls = []
    img_dir = Path("images")

    if img_dir.exists():
        for p in img_dir.iterdir():
            if p.is_file():
                key = f"images/{p.name}"
                s3.upload_file(str(p), BUCKET, key)

                url = f"https://{BUCKET}.s3.{REGION}.amazonaws.com/{key}"
                urls.append(url)

    return urls