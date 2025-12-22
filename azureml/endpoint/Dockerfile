# Template container for Azure ML Online Endpoint
# Note: exact base image + CUDA version must match your chosen Azure GPU SKU.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep dependencies minimal; pin later once we confirm the exact serving stack.
COPY azureml/endpoint/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY azureml/endpoint/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Optional: for AML health probes (not required if you expose vLLM only)
COPY azureml/endpoint/score.py /app/score.py

ENV PORT=8000
EXPOSE 8000

CMD ["/app/start.sh"]
