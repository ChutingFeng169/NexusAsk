"""
demo_timeout.py — visibly demonstrate the timeout process (mirrors demo_429.py).

Uses the SYNC call_with_timeout_sync so you see LIVE "waiting Xs/Ys" ticks tick up,
then either a normal response or a timeout. Prints the full flow to the terminal.

Run:  python demo_timeout.py
"""

import time
from timeout_handling import call_with_timeout_sync


def slow_model(prompt):
    time.sleep(5)  # stuck longer than the timeout
    return "不应该走到这里"


def fast_model(prompt):
    time.sleep(1)
    return "正常回答"


def reporter(ev):
    t = ev.get("type")
    if t == "calling":
        print(f"[timeout] calling model (timeout {ev['timeout']}s)...", flush=True)
    elif t == "waiting":
        print(f"[timeout]   waiting... {ev['elapsed']:.1f}s / {ev['timeout']}s", flush=True)
    elif t == "done":
        print(f"[timeout] ✅ responded in {ev['elapsed']:.2f}s", flush=True)
    elif t == "timeout":
        print(f"[timeout] ⏱ TIMED OUT after {ev['timeout']}s -> raising TimeoutError", flush=True)


def run(label, fn, timeout):
    print("=" * 64)
    print(f"DEMO: {label}")
    print("=" * 64)
    start = time.time()
    try:
        r = call_with_timeout_sync("测试问题", timeout_seconds=timeout, model_fn=fn, on_progress=reporter)
        print(f"[timeout] result: {r}  ({time.time() - start:.2f}s)\n", flush=True)
    except TimeoutError as e:
        print(f"[timeout] handled: {e}  ({time.time() - start:.2f}s)\n", flush=True)


if __name__ == "__main__":
    run("slow model (5s) with timeout=3s -> timeout fires live", slow_model, 3)
    run("fast model (1s) with timeout=5s -> normal return", fast_model, 5)
    print("DONE ✅")
