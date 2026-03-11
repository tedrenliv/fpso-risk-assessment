import queue
import sys

_q: queue.Queue = queue.Queue()
_orig_stdout = None


def get_queue() -> queue.Queue:
    return _q


def reset() -> None:
    while not _q.empty():
        try:
            _q.get_nowait()
        except queue.Empty:
            break


class _QueueWriter:
    def __init__(self, original):
        self._orig = original

    def write(self, text: str) -> None:
        stripped = text.rstrip("\n")
        if stripped.strip():
            _q.put(stripped)
        self._orig.write(text)

    def flush(self) -> None:
        self._orig.flush()


def install() -> None:
    global _orig_stdout
    if _orig_stdout is None:
        _orig_stdout = sys.stdout
        sys.stdout = _QueueWriter(_orig_stdout)


def uninstall() -> None:
    global _orig_stdout
    if _orig_stdout is not None:
        sys.stdout = _orig_stdout
        _orig_stdout = None
