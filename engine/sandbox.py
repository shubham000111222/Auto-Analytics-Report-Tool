from __future__ import annotations
import multiprocessing as mp
import sys
import traceback


def _child_target(fn, args, kwargs, conn, cpu_seconds, memory_mb):
    try:
        if sys.platform != "win32":
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024))
        result = fn(*args, **kwargs)
        conn.send(("ok", result))
    except Exception as e:
        conn.send(("error", f"{e}\n{traceback.format_exc()}"))
    finally:
        conn.close()


def run_sandboxed(fn, args=(), kwargs=None, timeout_seconds=15, cpu_seconds=10, memory_mb=512):
    """Runs fn in a short-lived subprocess with CPU/memory limits and a hard wall-clock
    timeout. Only picklable, small results should be returned -- this is meant for quick
    pre-checks, not for wrapping the full pipeline (which holds a live DuckDB connection
    that cannot cross process boundaries)."""
    kwargs = kwargs or {}
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_child_target, args=(fn, args, kwargs, child_conn, cpu_seconds, memory_mb))
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(f"Processing exceeded {timeout_seconds}s and was terminated.")

    if parent_conn.poll():
        status, payload = parent_conn.recv()
        if status == "error":
            raise RuntimeError(f"Sandboxed processing failed: {payload}")
        return payload

    raise RuntimeError("Sandboxed process exited without a result (likely killed for exceeding memory limits).")
