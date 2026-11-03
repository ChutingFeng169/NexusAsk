"""
demo_429.py
==========
Standalone demo: force EVERY primary call to return 429, then watch the full
retry -> exponential-backoff -> fallback flow printed live to the terminal.

Run:  python demo_429.py
"""

import time

import rate_limit_fallback as rlf
from rate_limit_fallback import RateLimitError

# Fast-but-real backoff so you can actually watch the timing (not minutes).
rlf.BACKOFF_MIN = 0.2
rlf.BACKOFF_MAX = 0.5
rlf._SLEEP = time.sleep  # REAL sleeps -> you see the waits actually happen

attempts = {"n": 0}


def fake_primary(prompt):
    attempts["n"] += 1
    raise RateLimitError(f"simulated 429 (attempt {attempts['n']})")


def fake_fallback(prompt):
    return "📦 [fallback model] answer from the backup endpoint."


# Patch the module-level callables the default path uses.
orig_p, orig_f = rlf.call_primary_model, rlf.call_fallback_model
rlf.call_primary_model = fake_primary
rlf.call_fallback_model = fake_fallback

print("=" * 64)
print("DEMO: simulate a 429 rate-limit on EVERY primary call")
print("Watch: retry attempts, exponential backoff waits, then fallback")
print("=" * 64)

start = time.time()
result = rlf.answer_with_retry_and_fallback("What is the capital of France?")  # default reporter prints
elapsed = time.time() - start

print("-" * 64)
print(f"FINAL RESULT      : {result}")
print(f"primary attempts  : {attempts['n']} (all rate-limited)")
print(f"total elapsed     : {elapsed:.2f}s")

rlf.call_primary_model, rlf.call_fallback_model = orig_p, orig_f
