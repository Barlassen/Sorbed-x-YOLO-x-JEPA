"""A small, integrity-verified registry of segmentation model weights.

The registry is the single source of truth for *what* Sorbed is allowed to load
and *how to trust it*. Every remote artifact carries the SHA-256 it must hash to;
:func:`download_weights` fetches it, verifies the digest, and **fails closed**
(deletes the file and raises) on any mismatch, so a corrupted or swapped download
can never reach inference.

No weights are vendored in the source tree. Remote entries live in a local
``models/registry.json`` file (absent by default); the only built-in entry is the
weight-free ``classical`` backend, which needs no download. Only ``.onnx`` and
``.safetensors`` artifacts are accepted — Sorbed never loads pickle-based formats,
which can execute arbitrary code on load.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_REGISTRY_FILE = _REPO_ROOT / "models" / "registry.json"
_ENV_REGISTRY_FILE = "SORBED_REGISTRY_FILE"
_ENV_MODEL_DIR = "SORBED_MODEL_DIR"

_ALLOWED_SUFFIXES = (".onnx", ".safetensors")


class ModelEntry(BaseModel):
    """Provenance record and download contract for one set of weights."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    url: str | None = None  # None => weight-free backend, nothing to download
    sha256: str | None = None  # required when url is set; hex digest of the file
    license: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    input_size: tuple[int, int] = (512, 512)  # (height, width) the model expects
    description: str = ""

    @property
    def needs_download(self) -> bool:
        return self.url is not None

    @property
    def filename(self) -> str:
        """Cache filename for this entry, derived from its URL when present."""
        if self.url is None:
            return f"{self.name}"
        suffix = _suffix_of(self.url)
        return f"{self.name}{suffix}"


# The only built-in entry: the classical backend guarantees the pipeline runs with
# zero downloads. It carries no URL and no digest because there is nothing to fetch.
_CLASSICAL_ENTRY = ModelEntry(
    name="classical",
    url=None,
    sha256=None,
    license="Apache-2.0",
    backend="classical",
    input_size=(0, 0),
    description=(
        "Weight-free Lab-distance + GrabCut wound segmenter. Always available, "
        "fully offline, interpretable baseline (not clinical-grade)."
    ),
)


def _suffix_of(url_or_path: str) -> str:
    suffix = Path(url_or_path.split("?", 1)[0]).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise ValueError(
            f"refusing unsupported weight format {suffix!r} in {url_or_path!r}; "
            f"only {_ALLOWED_SUFFIXES} are accepted"
        )
    return suffix


def _registry_file() -> Path:
    override = os.environ.get(_ENV_REGISTRY_FILE, "").strip()
    if override:
        return Path(override).expanduser()
    return _DEFAULT_REGISTRY_FILE


def _load_file_entries() -> list[ModelEntry]:
    path = _registry_file()
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("models", raw) if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError(f"registry file {path} must hold a list of model records")
    entries: list[ModelEntry] = []
    for record in records:
        entry = ModelEntry.model_validate(record)
        if entry.url is not None:
            if not entry.sha256:
                raise ValueError(
                    f"registry entry {entry.name!r} has a url but no sha256 digest"
                )
            _suffix_of(entry.url)  # validate the format eagerly
        entries.append(entry)
    return entries


def list_models() -> list[ModelEntry]:
    """All known entries: the built-in classical backend plus any file entries."""
    entries = [_CLASSICAL_ENTRY]
    seen = {_CLASSICAL_ENTRY.name}
    for entry in _load_file_entries():
        if entry.name in seen:
            raise ValueError(f"duplicate registry entry name: {entry.name!r}")
        seen.add(entry.name)
        entries.append(entry)
    return entries


def get_entry(name: str) -> ModelEntry:
    """Return the entry named ``name`` or raise :class:`KeyError`."""
    for entry in list_models():
        if entry.name == name:
            return entry
    known = ", ".join(sorted(e.name for e in list_models()))
    raise KeyError(f"no registered model named {name!r}; known models: {known}")


def resolve_cache_dir() -> Path:
    """Directory where downloaded weights live.

    Honors ``SORBED_MODEL_DIR`` first, then ``XDG_CACHE_HOME/sorbed``, and finally
    ``~/.cache/sorbed``. The directory is created if missing.
    """
    override = os.environ.get(_ENV_MODEL_DIR, "").strip()
    if override:
        base = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
        root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
        base = root / "sorbed"
    base.mkdir(parents=True, exist_ok=True)
    return base


def verify_sha256(path: Path, expected: str) -> bool:
    """Return whether ``path`` hashes to ``expected`` (case-insensitive hex)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected.strip().lower()


def download_weights(entry: ModelEntry) -> Path:
    """Fetch ``entry`` into the cache, verifying its digest; fail closed on mismatch.

    Returns the path to the verified file. If the file already exists and verifies,
    the download is skipped. On a digest mismatch the partial/bad file is deleted
    and a :class:`RuntimeError` is raised, so unverified bytes never persist.
    """
    if entry.url is None or entry.sha256 is None:
        raise RuntimeError(
            f"model {entry.name!r} is weight-free (backend {entry.backend!r}); "
            "there is nothing to download."
        )
    _suffix_of(entry.url)  # reject unsupported formats before touching the network

    cache_dir = resolve_cache_dir()
    target = cache_dir / entry.filename
    if target.is_file() and verify_sha256(target, entry.sha256):
        return target

    fd, tmp_name = tempfile.mkstemp(prefix=f"{entry.name}.", dir=str(cache_dir))
    tmp_path = Path(tmp_name)
    os.close(fd)
    try:
        with urllib.request.urlopen(entry.url) as response, tmp_path.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if not verify_sha256(tmp_path, entry.sha256):
            actual = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
            raise RuntimeError(
                f"integrity check failed for {entry.name!r}: expected "
                f"{entry.sha256}, got {actual}. The download was discarded."
            )
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return target
