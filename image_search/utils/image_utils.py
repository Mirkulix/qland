"""Image utility functions."""

import hashlib
import io
import os
from pathlib import Path

from PIL import Image


def validate_image(file_bytes: bytes, allowed_extensions: set) -> bool:
    """Validate that the bytes represent a valid image."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        return True
    except Exception:
        return False


def get_image_hash(file_bytes: bytes) -> str:
    """Compute SHA256 hash of image bytes."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def resize_for_upload(file_bytes: bytes, max_size: int = 1024) -> bytes:
    """Resize image to max dimension while maintaining aspect ratio.
    Many search APIs have file size limits, so we resize before uploading.
    """
    img = Image.open(io.BytesIO(file_bytes))
    if max(img.size) <= max_size:
        return file_bytes

    ratio = max_size / max(img.size)
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    fmt = img.format or "JPEG"
    if fmt.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=85)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def save_upload(file_bytes: bytes, upload_dir: str, filename: str) -> Path:
    """Save uploaded file to disk and return the path."""
    os.makedirs(upload_dir, exist_ok=True)
    image_hash = get_image_hash(file_bytes)
    ext = Path(filename).suffix or ".jpg"
    save_path = Path(upload_dir) / f"{image_hash}{ext}"
    save_path.write_bytes(file_bytes)
    return save_path


def image_to_base64(file_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    import base64
    return base64.b64encode(file_bytes).decode("utf-8")
