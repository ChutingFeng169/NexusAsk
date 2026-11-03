"""
demo_async_pipelines.py — visibly compare sequential vs concurrent file processing.

Prints the total elapsed for each strategy so you can see the speedup
(sequential ~ N*per_file; concurrent ~ per_file; limited-concurrent in between).

Run:  python demo_async_pipelines.py
"""

import asyncio
from async_pipelines import process_sequential, process_concurrent


def _report(ev):
    t = ev.get("type")
    if t == "file_start":
        print(f"  start: {ev['name']}", flush=True)
    elif t == "file_done":
        print(f"  done : {ev['name']}", flush=True)


async def main():
    files = ["文件1.pdf", "文件2.pdf", "文件3.pdf", "文件4.pdf", "文件5.pdf"]

    print("=== Sequential (顺序) ===")
    _, t = await process_sequential(files, on_progress=_report)
    print(f"  total: {t:.2f}s\n")

    print("=== Concurrent unlimited (并发) ===")
    _, t = await process_concurrent(files, max_concurrent=5, on_progress=_report)
    print(f"  total: {t:.2f}s\n")

    print("=== Concurrent max=2 (限流并发) ===")
    _, t = await process_concurrent(files, max_concurrent=2, on_progress=_report)
    print(f"  total: {t:.2f}s\n")


if __name__ == "__main__":
    asyncio.run(main())
