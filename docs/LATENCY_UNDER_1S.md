# Latency < 1s Checklist (Qwen3-VL-8B on Azure ML)

Hitting < 1s end-to-end is mostly a **serving engineering** problem.

## Biggest levers (in order)

### 1) Keep output short
- Set `max_tokens` low (start 256).
- Avoid chain-of-thought; require **JSON only**.

### 2) Downscale images aggressively (but safely)
- PDF pages are huge; sending full-res images kills latency.
- Start with long-side 1024–1536 px (tune to your fonts).

### 3) Use a fast inference engine
- Prefer **vLLM** with continuous batching.
- Keep model on GPU; avoid CPU offload.

### 4) Use quantization (if quality holds)
- 4-bit (AWQ/GPTQ) reduces memory and can improve throughput.
- FP8 can be very fast on newer GPUs.

### 5) Warmup + keep-alive
- Azure ML endpoints can scale-to-zero; avoid that if you need low p95.
- Keep minimum instances >= 1.
- Run warmup requests on startup.

### 6) Put endpoint in the same region as your app
- Network RTT matters a lot when your target is <1s.

## Settings that commonly blow latency
- Huge `max_tokens`
- Sending many pages at once
- High-res images (300 DPI page renders)
- Retry loops due to invalid JSON output

## Practical target
For a single page image + small JSON output:
- With a good GPU tier + tuned image size, **sub-1s p50** can be possible.
- **p95 < 1s** is harder; expect 1–2s until tuned.
