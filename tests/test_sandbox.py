import time
import pytest
from engine.sandbox import run_sandboxed
from engine.preparse_check import safe_preparse


def _slow():
    time.sleep(5)
    return "done"


def _fine():
    return 42


def _boom():
    raise ValueError("bad")


def test_normal_function_returns_value():
    assert run_sandboxed(_fine) == 42


def test_timeout_is_enforced():
    with pytest.raises(TimeoutError):
        run_sandboxed(_slow, timeout_seconds=1)


def test_exception_propagates_as_runtimeerror():
    with pytest.raises(RuntimeError):
        run_sandboxed(_boom)


def test_safe_preparse_valid_csv():
    raw = b"a,b,c\n1,2,3\n4,5,6\n"
    result = safe_preparse("ok.csv", raw)
    assert result["columns"] == ["a", "b", "c"]


def test_safe_preparse_rejects_garbage_extension():
    with pytest.raises(RuntimeError):
        safe_preparse("file.exe", b"not a real file")
