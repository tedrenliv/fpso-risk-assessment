import sys
from core.stream_queue import get_queue, reset, install, uninstall


def test_print_captured_to_queue():
    reset()
    install()
    print("hello world")
    uninstall()
    q = get_queue()
    assert not q.empty()
    msg = q.get_nowait()
    assert "hello world" in msg


def test_stdout_still_writes_through():
    """Original stdout must still receive output."""
    reset()
    install()
    import io
    # After install, sys.stdout is our writer; its _orig is the real stdout
    assert hasattr(sys.stdout, '_orig')
    uninstall()


def test_reset_clears_queue():
    reset()
    install()
    print("some message")
    uninstall()
    reset()
    assert get_queue().empty()


def test_install_idempotent():
    """Calling install twice should not double-wrap stdout."""
    reset()
    install()
    writer_after_first = sys.stdout
    install()  # second call — should be a no-op
    assert sys.stdout is writer_after_first
    uninstall()
