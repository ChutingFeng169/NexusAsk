"""
test_rate_limit_fallback.py
==========================
Precisely test the REAL retry/backoff/fallback logic in rate_limit_fallback.py
using mocks -- we only replace call_primary_model / call_fallback_model (the
network call) and the sleep function, so the tenacity retry controller, the
exponential-backoff computation, and the fallback switch are all exercised for
real.

Verified:
  1. retry count is exactly as expected (fail N-1 then succeed on N)
  2. exponential backoff with jitter (ceilings ~2x per step; each wait within its band)
  3. after MAX_RETRIES (10) all-rate-limited attempts -> fallback is used
"""

from unittest.mock import patch

import rate_limit_fallback as rlf
from rate_limit_fallback import RateLimitError, answer_with_retry_and_fallback


def _install_sleep_recorder():
    """Replace rlf._SLEEP with a recorder that returns instantly but captures the
    backoff durations tenacity computed. Returns the recordings list."""
    recorded = []

    def fake_sleep(seconds):
        recorded.append(seconds)  # do NOT actually sleep -> tests run instantly

    rlf._SLEEP = fake_sleep
    return recorded


def _reset_retry_globals():
    rlf.MAX_RETRIES = 10
    rlf.BACKOFF_BASE = 1
    rlf.BACKOFF_MIN = 1
    rlf.BACKOFF_MAX = 60
    rlf._SLEEP = time_sleep_orig


# keep the original sleep so tests that want real waiting can restore it
import time as _time

time_sleep_orig = _time.sleep


# ---------------------------------------------------------------- test 1
def test_retry_succeeds_after_failures():
    """First 2 calls rate-limited, 3rd succeeds -> exactly 3 attempts, success."""
    attempts = {"n": 0}

    def fake_primary(prompt):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimitError("simulated 429")
        return "成功的回答"

    _reset_retry_globals()
    sleeps = _install_sleep_recorder()

    with patch.object(rlf, "call_primary_model", side_effect=fake_primary):
        result = answer_with_retry_and_fallback("测试问题")

    assert attempts["n"] == 3, f"expected 3 attempts, got {attempts['n']}"
    assert result == "成功的回答", f"unexpected result: {result!r}"
    assert len(sleeps) == 2, f"expected 2 backoff sleeps (between 3 attempts), got {len(sleeps)}"
    print(f"✅ 测试1通过：重试{attempts['n']}次后成功，退避sleep {len(sleeps)}次")


# ---------------------------------------------------------------- test 2
def test_fallback_triggers_after_max_retries():
    """Always rate-limited -> exactly MAX_RETRIES (10) primary attempts, then fallback."""
    attempts = {"n": 0}

    def always_fail(prompt):
        attempts["n"] += 1
        raise RateLimitError("simulated 429, always fails")

    def fake_fallback(prompt):
        return "备用模型的回答"

    _reset_retry_globals()
    sleeps = _install_sleep_recorder()

    with patch.object(rlf, "call_primary_model", side_effect=always_fail), \
         patch.object(rlf, "call_fallback_model", side_effect=fake_fallback):
        result = answer_with_retry_and_fallback("测试问题")

    assert attempts["n"] == 10, f"expected 10 primary attempts, got {attempts['n']}"
    assert result == "备用模型的回答", f"expected fallback result, got {result!r}"
    assert len(sleeps) == 9, f"expected 9 backoff sleeps (between 10 attempts), got {len(sleeps)}"
    print(f"✅ 测试2通过：主模型重试{attempts['n']}次全失败后，正确切换到fallback")


# ---------------------------------------------------------------- test 3
def test_exponential_backoff_timing():
    """Exponential backoff with jitter: ceilings grow ~2x; each wait within its ceiling band."""
    attempts = {"n": 0}

    def fake_primary(prompt):
        attempts["n"] += 1
        if attempts["n"] < 4:  # fail first 3, succeed on 4th -> 3 backoff sleeps
            raise RateLimitError("simulated 429")
        return "ok"

    _reset_retry_globals()
    # raise the cap so no clamping interferes with the growth assertion
    rlf.BACKOFF_MAX = 10000
    sleeps = _install_sleep_recorder()

    with patch.object(rlf, "call_primary_model", side_effect=fake_primary):
        answer_with_retry_and_fallback("测试问题")

    print(f"   每次重试退避时间(秒): {sleeps}")
    assert len(sleeps) == 3, f"expected 3 backoff sleeps (between 4 attempts), got {len(sleeps)}"
    # ceilings: BACKOFF_BASE * 2^(n-1) = 1, 2, 4 (capped at BACKOFF_MAX)
    # each wait = max(BACKOFF_MIN, uniform(ceiling/2, ceiling)) -> jitter within band
    ceilings = [min(rlf.BACKOFF_MAX, rlf.BACKOFF_BASE * (2 ** i)) for i in range(len(sleeps))]
    for i, s in enumerate(sleeps):
        lo = max(rlf.BACKOFF_MIN, ceilings[i] / 2)
        hi = max(rlf.BACKOFF_MIN, ceilings[i])
        assert lo <= s <= hi, f"sleep {i} = {s:.2f} outside band [{lo:.2f}, {hi:.2f}] (ceiling={ceilings[i]})"
    # overall growth: ceilings double each step, so last sleep > first
    assert sleeps[-1] > sleeps[0], f"expected overall growth, got {sleeps}"
    print(f"✅ 测试3通过：退避时间指数增长(with jitter), sleeps={[round(s,2) for s in sleeps]}, ceilings={ceilings}")


def test_on_progress_event_sequence():
    """Verify on_progress fires the right event sequence for an all-429 run."""
    events = []
    attempts = {"n": 0}

    def always_fail(prompt):
        attempts["n"] += 1
        raise RateLimitError("simulated 429")

    def fake_fallback(prompt):
        return "fallback answer"

    _reset_retry_globals()
    rlf.BACKOFF_MIN = 0.0
    rlf.BACKOFF_MAX = 0.0
    _install_sleep_recorder()  # instant

    with patch.object(rlf, "call_primary_model", side_effect=always_fail), \
         patch.object(rlf, "call_fallback_model", side_effect=fake_fallback):
        result = answer_with_retry_and_fallback("q", on_progress=events.append)

    types = [e["type"] for e in events]
    # 10 attempts; before_sleep is NOT called on the final (stopped) attempt -> 9 rate_limited
    assert types.count("attempt") == 10, f"expected 10 attempts, got {types}"
    assert types.count("rate_limited") == 9, f"expected 9 rate_limited, got {types}"
    assert "exhausted" in types and "fallback" in types and "fallback_done" in types, types
    assert types[-1] == "fallback_done", f"expected last event fallback_done, got {types}"
    assert result == "fallback answer"
    print(f"✅ 测试4通过：on_progress事件序列正确 ({len(types)} events, seq OK)")


def test_retry_succeeds_before_max():
    """Retry succeeds BEFORE reaching MAX_RETRIES (recovery case): no fallback."""
    for fail_first in (1, 3, 5, 9):
        attempts = {"n": 0}

        def fake_primary(prompt, _ff=fail_first, _a=attempts):
            _a["n"] += 1
            if _a["n"] <= _ff:
                raise RateLimitError("simulated 429")
            return f"primary answer (after {_a['n']} attempts)"

        fb_calls = {"n": 0}

        def fake_fallback(prompt, _c=fb_calls):
            _c["n"] += 1
            return "SHOULD NOT HAPPEN"

        events = []
        _reset_retry_globals()
        rlf.BACKOFF_MIN = 0.0
        rlf.BACKOFF_MAX = 0.0
        _install_sleep_recorder()  # instant

        with patch.object(rlf, "call_primary_model", side_effect=fake_primary), \
             patch.object(rlf, "call_fallback_model", side_effect=fake_fallback):
            result = answer_with_retry_and_fallback("q", on_progress=events.append)

        types = [e["type"] for e in events]
        assert attempts["n"] == fail_first + 1, f"fail_first={fail_first}: expected {fail_first+1} attempts, got {attempts['n']}"
        assert fb_calls["n"] == 0, f"fail_first={fail_first}: fallback should NOT be called, got {fb_calls['n']}"
        assert result.startswith("primary answer"), f"unexpected result: {result}"
        assert "exhausted" not in types and "fallback" not in types and "fallback_done" not in types, types
        assert types and types[-1] == "done", f"expected last event 'done', got {types}"
    print("✅ 测试5通过：重试<10次后主模型成功响应(fail_first=1,3,5,9)，未触发fallback")


if __name__ == "__main__":
    test_retry_succeeds_after_failures()
    test_fallback_triggers_after_max_retries()
    test_exponential_backoff_timing()
    test_on_progress_event_sequence()
    test_retry_succeeds_before_max()
    print("\nALL TESTS PASSED ✅")
