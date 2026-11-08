"""Small file-identity and exclusive-output helpers."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@contextmanager
def output_lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another evaluator owns {folder}; use a separate output directory"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def model_identity(value, revision="main"):
    path = Path(value).expanduser()
    if not path.is_dir():
        if path.is_absolute() or value.startswith("."):
            raise FileNotFoundError(f"Local model directory does not exist: {value}")
        if "/" in value:
            from huggingface_hub import model_info

            info = model_info(value, revision=revision)
            return {"hub_id": value, "resolved_revision": info.sha}
        return {"hub_id": value}
    from .data import sha256

    files = sorted(
        p
        for p in path.rglob("*")
        if p.is_file()
        and (p.suffix in {".safetensors", ".bin", ".json", ".model", ".txt", ".jinja"})
        and "training_state" not in p.name
    )
    return {
        "path": str(path.resolve()),
        "files": {
            str(p.relative_to(path)): {
                "size": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
                **({"sha256": sha256(p)} if p.stat().st_size < 2_000_000 else {}),
            }
            for p in files
        },
    }
