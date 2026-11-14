"""Immutable model bundles and atomic promotion on a shared POSIX filesystem."""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from src.utils.paths import MODELS_DIR

FILES = ("trend_model.keras", "preprocessor.joblib", "model_metadata.json")
BUNDLE_ID = re.compile(r"[0-9a-f]{32}")


class BundleError(RuntimeError):
    """A bundle is missing, incomplete, or inconsistent."""


class TrainingInProgressError(RuntimeError):
    """Another writer holds the shared training lock."""


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError) as exc:
        raise BundleError(f"Cannot read bundle document: {path.name}") from exc


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class ModelBundle:
    directory: Path
    bundle_id: str
    revision: str
    legacy: bool = False

    @property
    def model_path(self) -> Path:
        return self.directory / ("saved/trend_model.keras" if self.legacy else FILES[0])

    @property
    def preprocessor_path(self) -> Path:
        return self.directory / ("metadata/preprocessor.joblib" if self.legacy else FILES[1])

    @property
    def metadata_path(self) -> Path:
        return self.directory / ("metadata/model_metadata.json" if self.legacy else FILES[2])

    def validate(self) -> dict:
        """Verify all bytes before deserializing trusted model/preprocessor files."""
        if not self.legacy:
            manifest = _read_json(self.directory / "manifest.json")
            hashes = manifest.get("sha256")
            if (
                manifest.get("bundle_id") != self.bundle_id
                or not isinstance(hashes, dict)
                or set(hashes) != set(FILES)
            ):
                raise BundleError("Invalid model bundle manifest")
            for name in FILES:
                path = self.directory / name
                if not path.is_file() or _digest(path) != manifest["sha256"][name]:
                    raise BundleError(f"Model bundle checksum mismatch: {name}")
        metadata = _read_json(self.metadata_path)
        columns = metadata.get("feature_columns")
        if (
            not metadata.get("model_version")
            or not isinstance(columns, list)
            or not columns
            or any(not isinstance(column, str) or not column for column in columns)
            or len(set(columns)) != len(columns)
        ):
            raise BundleError("Model metadata must identify its version and input columns")
        if not self.legacy and metadata["model_version"] != self.bundle_id:
            raise BundleError("Model metadata version does not match the bundle")
        return metadata


def resolve_bundle(root: Path = MODELS_DIR) -> ModelBundle | None:
    """Read the pointer once. A malformed pointer never falls back to legacy files."""
    pointer = root / "current.json"
    if pointer.exists():
        value = _read_json(pointer)
        bundle_id = value.get("bundle_id")
        if (
            value.get("format_version") != 1
            or not isinstance(bundle_id, str)
            or not BUNDLE_ID.fullmatch(bundle_id)
        ):
            raise BundleError("Invalid active model pointer")
        return ModelBundle(root / "bundles" / bundle_id, bundle_id, bundle_id)
    paths = [
        root / "saved/trend_model.keras",
        root / "metadata/preprocessor.joblib",
        root / "metadata/model_metadata.json",
    ]
    if all(path.is_file() for path in paths):
        revision = "legacy:" + ":".join(
            f"{path.stat().st_mtime_ns}-{path.stat().st_size}" for path in paths
        )
        return ModelBundle(root, "legacy", revision, legacy=True)
    return None


@contextmanager
def training_lock(root: Path = MODELS_DIR):
    """Fail fast for competing writers; the OS releases the lock on process exit.

    All writers must use this entry point and share a filesystem supporting
    POSIX advisory locks across nodes. Never unlink the lock file.
    """
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / ".training.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o660)
    with os.fdopen(descriptor, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise TrainingInProgressError(
                "Another training or data-refresh process is running. Retry after it finishes."
            ) from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def stage_bundle(root: Path = MODELS_DIR):
    bundle_id = uuid.uuid4().hex
    directory = root / "bundles" / f".staging-{bundle_id}"
    directory.mkdir(parents=True)
    try:
        yield ModelBundle(directory, bundle_id, bundle_id)
    finally:
        # Published directories have been renamed and are retained for rollback.
        if directory.exists():
            shutil.rmtree(directory)


def publish_bundle(staged: ModelBundle) -> ModelBundle:
    """Seal a validated training result, then replace only the active pointer.

    Call under training_lock after a successful end-to-end prediction probe.
    A crash before pointer replacement leaves the previously active bundle intact.
    """
    if staged.legacy or staged.directory.name != f".staging-{staged.bundle_id}":
        raise BundleError("Only staged bundles can be published")
    hashes = {}
    for name in FILES:
        path = staged.directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise BundleError(f"Incomplete model bundle: {name}")
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        hashes[name] = _digest(path)
    _write_json(
        staged.directory / "manifest.json", {"bundle_id": staged.bundle_id, "sha256": hashes}
    )
    staged.validate()
    _sync_directory(staged.directory)
    final = staged.directory.parent / staged.bundle_id
    staged.directory.rename(final)
    _sync_directory(final.parent)
    root = final.parent.parent
    temporary = root / f".current-{staged.bundle_id}.json"
    try:
        _write_json(temporary, {"format_version": 1, "bundle_id": staged.bundle_id})
        os.replace(temporary, root / "current.json")
        _sync_directory(root)
    finally:
        temporary.unlink(missing_ok=True)
    return ModelBundle(final, staged.bundle_id, staged.bundle_id)
