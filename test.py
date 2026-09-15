import os
import time
import statistics
import requests

BASE = "https://integrate.api.nvidia.com/v1"
MODELS_URL = BASE + "/models"
CHAT_URL = BASE + "/chat/completions"
MODEL = "openai/gpt-oss-20b"
ROUNDS = 10
TIMEOUT = 30

API_KEY = "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"
if not API_KEY:
    print("ERROR: NVIDIA_API_KEY environment variable is not set.")
    print('PowerShell: $env:NVIDIA_API_KEY="YOUR_KEY_HERE"')
    raise SystemExit(1)

auth = {"Authorization": f"Bearer {API_KEY}"}
headers = {
    **auth,
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
}

def models_test():
    print("\n=== NVIDIA EDGE / NETWORK TEST ===")
    t0 = time.perf_counter()
    try:
        r = requests.get(MODELS_URL, headers=auth, timeout=TIMEOUT)
        dt = time.perf_counter() - t0
        print(f"/models status : {r.status_code}")
        print(f"/models time   : {dt:.3f}s")
        return dt
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"/models ERROR  : {type(e).__name__}: {e}")
        print(f"/models time   : {dt:.3f}s")
        return dt

def chat_test(n):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hey bro"}],
        "temperature": 0.2,
        "max_tokens": 32,
        "stream": True,
    }

    print(f"\n--- Chat {n}/{ROUNDS} ---")
    t0 = time.perf_counter()
    ttft = None
    chunks = 0

    try:
        with requests.post(
            CHAT_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=TIMEOUT,
        ) as r:
            if r.status_code != 200:
                dt = time.perf_counter() - t0
                print(f"HTTP status : {r.status_code}")
                print(f"Time        : {dt:.3f}s")
                print(f"Error       : {r.text[:800]}")
                return r.status_code, None, dt

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
                    print(f"FIRST CHUNK : {ttft:.3f}s")

            total = time.perf_counter() - t0
            print(f"TOTAL       : {total:.3f}s")
            return 200, ttft, total

    except requests.exceptions.Timeout as e:
        dt = time.perf_counter() - t0
        print(f"TIMEOUT     : {dt:.3f}s")
        return "TIMEOUT", ttft, dt
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"ERROR       : {type(e).__name__}: {e}")
        print(f"Time        : {dt:.3f}s")
        return "ERROR", ttft, dt

print("=" * 68)
print("NVIDIA BACKEND vs INTERNET DIAGNOSTIC")
print("=" * 68)
print(f"Model: {MODEL}")
print(f"Rounds: {ROUNDS} sequential requests")
print("Do not run multiple copies at once.")

edge_time = models_test()
results = []

for i in range(1, ROUNDS + 1):
    results.append(chat_test(i))
    time.sleep(1)

ttft = [x[1] for x in results if isinstance(x[1], (int, float))]
totals = [x[2] for x in results if x[0] == 200]

print("\n" + "=" * 68)
print("SUMMARY")
print("=" * 68)

print(f"/models: {edge_time:.3f}s")

if ttft:
    print(f"TTFT   min / avg / max: {min(ttft):.3f}s / {statistics.mean(ttft):.3f}s / {max(ttft):.3f}s")
if totals:
    print(f"TOTAL  min / avg / max: {min(totals):.3f}s / {statistics.mean(totals):.3f}s / {max(totals):.3f}s")

print(f"Successful: {len(totals)}/{ROUNDS}")
print(f"Timeouts:   {sum(x[0] == 'TIMEOUT' for x in results)}/{ROUNDS}")

print("\nPer-request:")
for i, x in enumerate(results, 1):
    t = "-" if x[1] is None else f"{x[1]:.3f}s"
    print(f"{i:2d}: status={x[0]!s:8} TTFT={t:>8} TOTAL={x[2]:.3f}s")

print("\nINTERPRETATION:")
print("- If /models stays ~0.5-1s but chat TTFT jumps between ~2s and 10s,")
print("  the connection to NVIDIA is healthy and inference/backend latency")
print("  is the stronger suspect.")
print("- If /models also becomes slow/timeouts, investigate your internet,")
print("  VPN, DNS, or ISP routing.")
print("- Best confirmation: run once on Wi-Fi and once on a phone hotspot.")
print("  Similar chat timings on both strongly point to NVIDIA/backend.")
print("- Requests are sequential to avoid creating artificial congestion.")
