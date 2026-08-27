"""Locate or download the Llama GGUF used for CPU inference."""

from __future__ import annotations

import os
import shutil
import urllib.request

DEFAULT_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
DEFAULT_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MIN_BYTES = 1_500_000_000


def gguf_filename() -> str:
    return os.getenv("ROSEGOLD_LLAMA_GGUF", DEFAULT_FILE).strip() or DEFAULT_FILE


def gguf_repo() -> str:
    return os.getenv("ROSEGOLD_LLAMA_REPO", DEFAULT_REPO).strip() or DEFAULT_REPO


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


def _download(url: str, dest: str) -> None:
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = dest + ".part"
    print(f"[Model] Downloading {url} -> {dest}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    if not _ok(dest):
        raise RuntimeError(f"Downloaded GGUF is too small: {dest}")


def ensure_llama_gguf() -> str:
    filename = gguf_filename()
    local = os.path.join(local_model_dir(), filename)
    if _ok(local):
        print(f"[Model] Using local GGUF {local}", flush=True)
        return local

    os.makedirs(local_model_dir(), exist_ok=True)
    cached = os.path.join(gcs_model_dir(), filename)
    if _ok(cached):
        print(f"[Model] Copying GCS GGUF {cached} -> {local}", flush=True)
        shutil.copy2(cached, local)
        return local

    url = (
        os.getenv("ROSEGOLD_LLAMA_URL", "").strip()
        or f"https://huggingface.co/{gguf_repo()}/resolve/main/{filename}"
    )
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
