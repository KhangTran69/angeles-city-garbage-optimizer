"""Persistence helpers for optimization experiment metadata."""
import errno
import json
import os
import stat
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - used on Windows.
    fcntl = None

from src.config import EXPERIMENTS_PATH, MAX_HISTORY_SIZE


@contextmanager
def _history_file_lock(path: str) -> Iterator[None]:
    """Serialize history reads and writes across threads and processes."""
    lock_path = os.path.abspath(path) + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    with open(lock_path, "a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return

        import msvcrt

        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()

        while True:
            lock_file.seek(0)
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as error:
                lock_contention = error.errno in {
                    errno.EACCES,
                    errno.EAGAIN,
                    errno.EDEADLK,
                } or getattr(error, "winerror", None) in {32, 33}
                if not lock_contention:
                    raise
                time.sleep(0.05)
        try:
            yield
        finally:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _read_experiment_history(
    path: str,
    *,
    refuse_invalid: bool = False,
) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        if refuse_invalid:
            raise ValueError(
                f"Experiment history at {path} is unreadable; refusing to overwrite it."
            ) from error
        return []

    if not isinstance(raw_data, list):
        if refuse_invalid:
            raise ValueError(
                f"Experiment history at {path} is not a JSON list; refusing to overwrite it."
            )
        return []
    return [item for item in raw_data if isinstance(item, dict)]


def load_experiment_history(path: str = EXPERIMENTS_PATH) -> List[Dict[str, Any]]:
    """Load previously persisted experiment records."""
    if not os.path.exists(path):
        return []
    with _history_file_lock(path):
        return _read_experiment_history(path)


def append_experiment_record(
    record: Dict[str, Any],
    path: str = EXPERIMENTS_PATH,
    max_items: int = MAX_HISTORY_SIZE * 5,
) -> List[Dict[str, Any]]:
    """Append an experiment record and persist capped history to disk."""
    absolute_path = os.path.abspath(path)
    directory = os.path.dirname(absolute_path)
    os.makedirs(directory, exist_ok=True)

    with _history_file_lock(absolute_path):
        history = _read_experiment_history(absolute_path, refuse_invalid=True)
        history.append(record)
        trimmed_history = history[-max_items:]

        try:
            existing_mode = stat.S_IMODE(os.stat(absolute_path).st_mode)
        except FileNotFoundError:
            existing_mode = None

        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(absolute_path)}.",
            suffix=".tmp",
            dir=directory,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(trimmed_history, file, indent=2, ensure_ascii=False)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            if existing_mode is not None:
                os.chmod(temporary_path, existing_mode)
            os.replace(temporary_path, absolute_path)
            temporary_path = ""
            try:
                directory_descriptor = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError:
                pass
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

    return trimmed_history


def latest_experiment(path: str = EXPERIMENTS_PATH) -> Optional[Dict[str, Any]]:
    """Return the most recent experiment record, if available."""
    history = load_experiment_history(path)
    return history[-1] if history else None
