FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip3 install --no-cache-dir --upgrade pip

# vLLM + server deps
COPY runpod/serve/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY runpod/serve/start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000

CMD ["/app/start.sh"]
