"""Durable per-game beliefs; metadata and inactive games stay out of the hot path."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Iterator


logger = logging.getLogger(__name__)

def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise ValueError(f"Invalid checkpoint JSON at {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"Invalid checkpoint object at {path}")
    return value


def write_object(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(value, separators=(",", ":"), allow_nan=False))


class CheckpointStore:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.directory = state_path.with_name(state_path.name + ".games")
        self.lock = threading.RLock()
        self._ready = False
        self._active: dict[str, dict[str, Any]] = {}
        self._cleanup: Iterator[Path] | None = None

    def checkpoint_path(self, game_ref: str) -> Path:
        # References may be supplied by an API. Never interpret them as paths.
        return self.directory / (hashlib.sha256(game_ref.encode()).hexdigest() + ".json")

    def prepare(self) -> None:
        with self.lock:
            if self._ready:
                return
            metadata = read_object(self.state_path)
            migrated = 0
            if "beliefs" in metadata:
                beliefs = metadata["beliefs"]
                if not isinstance(beliefs, dict):
                    raise ValueError("Invalid legacy beliefs map; migration left untouched")
                original = self.state_path.read_text(encoding="utf-8")
                digest = hashlib.sha256(original.encode()).hexdigest()[:16]
                backup = self.state_path.with_name(self.state_path.name + f".legacy-{digest}.bak")
                if not backup.exists():
                    atomic_write(backup, original)
                if backup.read_text(encoding="utf-8") != original:
                    raise ValueError("Legacy checkpoint backup verification failed")
                for ref, snapshot in beliefs.items():
                    if not isinstance(ref, str) or not isinstance(snapshot, dict):
                        raise ValueError("Invalid legacy belief; migration left untouched")
                    path = self.checkpoint_path(ref)
                    record = {"game_ref": ref, "belief": snapshot}
                    if path.exists():
                        if read_object(path) != record:
                            raise ValueError("Conflicting migrated belief; migration left untouched")
                    else:
                        write_object(path, record)
                    if read_object(path) != record:
                        raise ValueError("Migrated belief verification failed")
                # Commit only after every snapshot and the protected backup are durable.
                migrated = len(beliefs)
                del metadata["beliefs"]
                write_object(self.state_path, metadata)
            self._ready = True
            logger.info("checkpoint store ready: migrated_games=%s metadata_bytes=%s",
                        migrated, self.state_path.stat().st_size if self.state_path.exists() else 0)

    def load_metadata(self) -> dict[str, Any]:
        with self.lock:
            self.prepare()
            return read_object(self.state_path)

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        with self.lock:
            self.prepare()
            if "beliefs" in metadata:
                raise ValueError("Beliefs must be saved through per-game checkpoints")
            write_object(self.state_path, metadata)

    def load_belief(self, game_ref: str) -> dict[str, Any] | None:
        with self.lock:
            self.prepare()
            if game_ref not in self._active:
                path = self.checkpoint_path(game_ref)
                if not path.exists():
                    return None
                record = read_object(path)
                if record.get("game_ref") != game_ref or not isinstance(record.get("belief"), dict):
                    raise ValueError("Invalid game checkpoint")
                self._active[game_ref] = record["belief"]
            return deepcopy(self._active[game_ref])

    def save_belief(self, game_ref: str, snapshot: dict[str, Any]) -> None:
        with self.lock:
            self.prepare()
            if self._active.get(game_ref) == snapshot:
                return
            write_object(self.checkpoint_path(game_ref), {"game_ref": game_ref, "belief": snapshot})
            # Publish only after a successful durable write.
            self._active[game_ref] = deepcopy(snapshot)

    def release(self, game_ref: str, *, completed: bool = False) -> None:
        with self.lock:
            self.prepare()
            if completed:
                self.checkpoint_path(game_ref).unlink(missing_ok=True)
            self._active.pop(game_ref, None)

    def next_cleanup_ref(self) -> str | None:
        """One pass per process, one cold checkpoint per discovery cycle."""
        with self.lock:
            self.prepare()
            if self._cleanup is None:
                self._cleanup = self.directory.glob("*.json")
            path = next(self._cleanup, None)
            if path is None:
                return None
            record = read_object(path)
            ref = record.get("game_ref")
            if not isinstance(ref, str) or self.checkpoint_path(ref) != path:
                raise ValueError("Invalid cleanup checkpoint reference")
            return ref
