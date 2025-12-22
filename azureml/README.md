# Azure ML Managed Online Endpoint (Template)

This is a **template** for deploying Qwen3-VL-8B behind an Azure ML managed online endpoint.

Recommended approach for low latency:
- Custom container image
- Inference server: vLLM (OpenAI-compatible) if your multimodal model is supported

Files:
- `azureml/endpoint/Dockerfile` container template
- `azureml/endpoint/start.sh` starts the inference server
- `azureml/endpoint/score.py` minimal health route (optional)

You’ll still need:
- Azure ML workspace setup
- Model artifacts uploaded/registered (or mounted from storage)

When you share your intended Azure region + budget, I’ll suggest the best GPU SKU and scaling settings.
