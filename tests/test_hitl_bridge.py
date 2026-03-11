import threading
from core.hitl_bridge import wait_for_decision, set_decision, reset


def test_set_before_wait_returns_immediately():
    reset()
    set_decision("approved", "looks good")
    decision, note = wait_for_decision()
    assert decision == "approved"
    assert note == "looks good"


def test_wait_blocks_until_set():
    reset()

    def _setter():
        import time
        time.sleep(0.05)
        set_decision("skipped", "")

    t = threading.Thread(target=_setter)
    t.start()
    decision, note = wait_for_decision()
    t.join()
    assert decision == "skipped"


def test_reset_clears_event():
    reset()
    set_decision("rejected", "bad")
    reset()
    # After reset, wait_for_decision should block — verify event is cleared
    from core import hitl_bridge
    assert not hitl_bridge._event.is_set()


def test_default_decision_after_reset():
    reset()
    from core import hitl_bridge
    assert hitl_bridge._decision["value"] == "approved"
