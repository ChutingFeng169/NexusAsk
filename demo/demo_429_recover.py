"""
demo_429_recover.py
==================
Demo: the RECOVERY case — primary model returns 429 for the first N attempts,
then SUCCEEDS on attempt N+1 (so the answer comes from the PRIMARY model and
the fallback is NEVER used).

Run:  python demo_429_recover.py
"""

import time

import rate_limit_fallback as rlf
from rate_limit_fallback import RateLimitError

# Fast-but-real backoff so you can actually watch the timing.
rlf.BACKOFF_MIN = 0.2
rlf.BACKOFF_MAX = 0.5
rlf._SLEEP = time.sleep  # REAL sleeps -> you see the waits actually happen

FAIL_FIRST = 3
attempts = {"n": 0}


def fake_primary(prompt):
    attempts["n"] += 1
    if attempts["n"] <= FAIL_FIRST:
        raise RateLimitError(f"simulated 429 (attempt {attempts['n']}/{FAIL_FIRST})")
    return f"✅ [primary model] recovered on attempt {attempts['n']} and answered."


def fake_fallback(prompt):
    return "SHOULD NOT BE CALLED"


orig_p, orig_f = rlf.call_primary_model, rlf.call_fallback_model
rlf.call_primary_model = fake_primary
rlf.call_fallback_model = fake_fallback

print("=" * 64)
print(f"DEMO: 429 then RECOVER — primary fails first {FAIL_FIRST} attempts,")
print("then SUCCEEDS on the next attempt (NO fallback used).")
print("=" * 64)

start = time.time()
result = rlf.answer_with_retry_and_fallback("What is the capital of France?")
elapsed = time.time() - start

print("-" * 64)
print(f"FINAL RESULT      : {result}")
print(f"primary attempts  : {attempts['n']} (recovered, fallback NOT used)")
print(f"total elapsed     : {elapsed:.2f}s")

rlf.call_primary_model, rlf.call_fallback_model = orig_p, orig_f
