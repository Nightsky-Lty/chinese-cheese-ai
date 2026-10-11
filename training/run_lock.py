"""防止两个进程同时写同一个标注或训练目录；进程退出后锁自动释放。"""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import os


@contextmanager
def run_lock(path: Path) -> Iterator[None]:
    """独占锁 path；锁文件可保留，但操作系统锁不会因崩溃永久占用。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f"another process owns this run: {path}") from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"another process owns this run: {path}") from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
