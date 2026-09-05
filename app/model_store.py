"""Locate or download the Llama GGUF used for CPU inference.

Hardening:
* Download URLs must be ``https://`` (override host via ``ROSEGOLD_LLAMA_URL``).
* Downloads stream to a ``.part`` file with a socket timeout, then are renamed
  atomically; a partial file is never left in place of a good one.
* Set ``ROSEGOLD_LLAMA_SHA256`` to pin the expected digest; a mismatch deletes
  the download and raises instead of loading tampered weights.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.parse
import urllib.request

DEFAULT_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
DEFAULT_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MIN_BYTES = 1_500_000_000
_CHUNK = 8 * 1024 * 1024
_ALLOWED_SCHEMES = {"https"}


def gguf_filename() -> str:
    name = os.getenv("ROSEGOLD_LLAMA_GGUF", DEFAULT_FILE).strip() or DEFAULT_FILE
    # A bare filename only: no directory components may be smuggled in via env.
    return os.path.basename(name)


def gguf_repo() -> str:
    return os.getenv("ROSEGOLD_LLAMA_REPO", DEFAULT_REPO).strip() or DEFAULT_REPO


def expected_sha256() -> str:
    return os.getenv("ROSEGOLD_LLAMA_SHA256", "").strip().lower()


def download_timeout() -> float:
    try:
        return max(5.0, float(os.getenv("ROSEGOLD_DOWNLOAD_TIMEOUT", "120")))
    except ValueError:
        return 120.0


def model_display_name() -> str:
    name = gguf_filename()
    return name.replace(".gguf", "")


def gcs_model_dir() -> str:
    return os.getenv("ROSEGOLD_MODEL_DIR", "/mnt/gcs/models")


def local_model_dir() -> str:
    return os.getenv("ROSEGOLD_LOCAL_MODEL_DIR", "/tmp/rosegold-models")


def _ok(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= MIN_BYTES
    except OSError:
        return False


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_download_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES or not parsed.netloc:
        raise ValueError(f"Refusing to download model weights from non-https URL: {url!r}")
    return url


def _download(url: str, dest: str) -> None:
    validate_download_url(url)
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = dest + ".part"
    print(f"[Model] Downloading {url} -> {dest}", flush=True)
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "rosegold-model-store/1.0"})
        with urllib.request.urlopen(request, timeout=download_timeout()) as response, open(tmp, "wb") as out:
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                digest.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        want = expected_sha256()
        got = digest.hexdigest()
        if want and got != want:
            raise RuntimeError(f"GGUF sha256 mismatch: expected {want}, got {got}")
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    if not _ok(dest):
        try:
            os.remove(dest)
        except OSError:
            pass
        raise RuntimeError(f"Downloaded GGUF is too small: {dest}")


def _verify_pinned(path: str) -> bool:
    """Return True when ``path`` matches the pinned digest (or no pin is configured)."""
    want = expected_sha256()
    if not want:
        return True
    try:
        return _sha256_file(path) == want
    except OSError:
        return False


def ensure_llama_gguf() -> str:
    filename = gguf_filename()
    local = os.path.join(local_model_dir(), filename)
    if _ok(local) and _verify_pinned(local):
        print(f"[Model] Using local GGUF {local}", flush=True)
        return local

    os.makedirs(local_model_dir(), exist_ok=True)
    cached = os.path.join(gcs_model_dir(), filename)
    if _ok(cached) and _verify_pinned(cached):
        print(f"[Model] Copying GCS GGUF {cached} -> {local}", flush=True)
        shutil.copy2(cached, local)
        return local

    url = (
        os.getenv("ROSEGOLD_LLAMA_URL", "").strip()
        or f"https://huggingface.co/{gguf_repo()}/resolve/main/{filename}"
    )
    validate_download_url(url)
    try:
        os.makedirs(gcs_model_dir(), exist_ok=True)
        if os.path.isdir(gcs_model_dir()) and os.access(gcs_model_dir(), os.W_OK):
            _download(url, cached)
            if cached != local:
                shutil.copy2(cached, local)
            return local
    except Exception as exc:
        print(f"[Model] GCS cache write skipped ({exc})", flush=True)

    _download(url, local)
    return local
