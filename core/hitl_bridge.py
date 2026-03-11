import threading

_event = threading.Event()
_decision: dict = {"value": "approved", "note": ""}


def wait_for_decision() -> tuple:
    """Block until set_decision() is called. Returns (decision, note)."""
    _event.wait()
    return _decision["value"], _decision["note"]


def set_decision(decision: str, note: str = "") -> None:
    """Called by POST /hitl to unblock the waiting pipeline."""
    _decision["value"] = decision
    _decision["note"] = note
    _event.set()


def reset() -> None:
    _event.clear()
    _decision["value"] = "approved"
    _decision["note"] = ""
