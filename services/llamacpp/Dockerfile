ARG LLAMA_SERVER_TAG=server-cuda
FROM ghcr.io/ggml-org/llama.cpp:${LLAMA_SERVER_TAG}

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
