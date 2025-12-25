ARG VLLM_BASE_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip3 install --no-cache-dir --upgrade pip

# Extra deps (vLLM server is already included in the base image)
COPY runpod/serve/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY runpod/serve/start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000

# Use ENTRYPOINT instead of CMD so platforms that override CMD (e.g., serverless)
# still execute our bootstrap script.
ENTRYPOINT ["/app/start.sh"]
