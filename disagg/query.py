import json
import time
import requests

# 1. Prepare prompt
#prompt_data = "Mount Everest is the highest mountain on Earth. Exploration is the engine of human progress and knowledge. " * 236
#prompt = f"Summarize the following history of Everest: {prompt_data}"

prompt_data = "Quantum Physics " * 2000
prompt = f"Explain Quantum Physics : {prompt_data}"

model = "RedHatAI/Qwen3-Next-80B-A3B-Instruct-FP8-dynamic"
#model ="RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic"
#model = "RedHatAI/Qwen3.5-9B-FP8-dynamic"

url = "http://127.0.0.1:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "min_tokens": 5000,
    "max_tokens": 5000,
    "temperature": 0,
    "stream": True
}

# 2. Lifecycle Timestamps
start_time = time.perf_counter()
ttft_time = None

completion_tokens = 0
prompt_tokens = 0
full_text = ""

response = requests.post(url, headers=headers, json=payload, stream=True)

# 3. Parse SSE stream
for line in response.iter_lines():
    if not line:
        continue

    decoded_line = line.decode("utf-8")
    if not decoded_line.startswith("data: "):
        continue

    data_str = decoded_line[6:]
    if data_str.strip() == "[DONE]":
        break

    try:
        chunk = json.loads(data_str)

        # Usage object (usually in final chunk) — authoritative token counts
        if "usage" in chunk and chunk["usage"] is not None:
            prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
            completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)
            continue  # usage chunk typically has no content delta, skip manual count

        # Extract content and count tokens manually (no usage object on this chunk)
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        if content:
            full_text += content
            completion_tokens += 1
            if ttft_time is None:
                ttft_time = time.perf_counter()

    except Exception:
        continue

end_time = time.perf_counter()

# Fallback token estimation if usage was never broadcast
if prompt_tokens == 0:
    prompt_tokens = len(prompt.split())
if completion_tokens == 0:
    completion_tokens = len(full_text.split())

# 4. Telemetry calculations
e2e_sec = end_time - start_time
ttft_sec = ttft_time - start_time if ttft_time else e2e_sec
generation_sec = end_time - ttft_time if ttft_time else e2e_sec

ttft_ms = ttft_sec * 1000
e2e_ms = e2e_sec * 1000
inter_token_latency_ms = (generation_sec / completion_tokens) * 1000 if completion_tokens > 0 else 0

input_tokens_per_sec = prompt_tokens / ttft_sec if ttft_sec > 0 else 0
output_tokens_per_sec = completion_tokens / generation_sec if generation_sec > 0 else 0
overall_tokens_per_sec = completion_tokens / e2e_sec if e2e_sec > 0 else 0

# 5. Output
output_payload = {
    "object": "chat.completion",
    "model": model,
    "usage": {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    },
    "metrics": {
        "ttft_ms": round(ttft_ms, 1),
        "inter_token_latency_ms": round(inter_token_latency_ms, 2),
        "e2e_latency_ms": round(e2e_ms, 1),
        "input_tokens_per_sec": round(input_tokens_per_sec, 2),
        "output_tokens_per_sec": round(output_tokens_per_sec, 2),
        "overall_tokens_per_sec": round(overall_tokens_per_sec, 2)
    }
}

print(json.dumps(output_payload, indent=2))
