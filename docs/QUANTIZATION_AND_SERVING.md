# Quantization + Serving Plan (Qwen3-VL-8B)

## Serving engine choice
For low latency on GPU:
- Prefer **vLLM** (OpenAI-compatible server) if it supports Qwen3-VL multimodal for your model build.
- If multimodal support is limited, alternatives:
  - SGLang (often strong multimodal support)
  - TensorRT-LLM (fastest, but higher integration cost)

## Quantization options

### AWQ 4-bit
- Good memory reduction and often good throughput.
- Best when you want to pack more concurrency per GPU.

### GPTQ 4-bit
- Widely used; sometimes slower than AWQ depending on kernel support.

### FP8
- Very fast on newer GPUs (hardware dependent).

## Recommendation for Azure ML endpoint
- Start with **FP16/BF16** baseline on vLLM.
- If latency/cost is too high: move to **AWQ 4-bit**.
- Validate extraction accuracy on a held-out test set after quantization.

## Compatibility note
Quantization support varies by model architecture and serving stack.
When you’re ready, we’ll pick the exact toolchain (AWQ vs GPTQ vs FP8) based on:
- The Azure GPU SKU you choose
- The supported kernels in your container
- Your measured quality delta on extraction fields

## Troubleshooting: vLLM AWQ `cudaErrorUnsupportedPtxVersion`
If vLLM crashes while loading an AWQ model with an error like:
- `cudaErrorUnsupportedPtxVersion`
- `the provided PTX was compiled with an unsupported toolchain`

This usually means your **host NVIDIA driver** is not compatible with the CUDA/toolchain used by the kernels inside your vLLM container, or the selected kernel backend (e.g. **Marlin**) is not supported on that GPU generation.

Practical fixes:
- Prefer a host with a **newer NVIDIA driver** (often the simplest on marketplaces like Vast.ai).
- Avoid using an unpinned base image (don’t rely on `:latest`). Pin the serving image to a known-good vLLM/CUDA combo for your target fleet.
- If you must stay on that host, try a different quantization format/backend that doesn’t route through the failing kernel path.
