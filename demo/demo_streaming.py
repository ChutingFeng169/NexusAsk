"""
demo_streaming.py — real streaming from the DeepSeek proxy, printed live (terminal typewriter).
"""
import os
from dotenv import load_dotenv
load_dotenv("/Users/ChuanZhou/rag-project/rag-project/.env")
from streaming_tokens_retry import stream_chat, count_tokens

print(f"prompt tokens: {count_tokens('用三句话介绍陶行知的教育思想。')}")
print("streaming: ", end="", flush=True)
full = []
for delta in stream_chat(
    os.getenv("OPENAI_BASE_URL"), os.getenv("OPENAI_API_KEY"), os.getenv("LLM_MODEL"),
    [{"role": "user", "content": "用三句话介绍陶行知的教育思想。"}],
):
    print(delta, end="", flush=True)
    full.append(delta)
print(f"\n\n[done] {len(full)} chunks, {count_tokens(''.join(full))} output tokens")
