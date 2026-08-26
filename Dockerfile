# syntax=docker/dockerfile:1.7

# Vast recommends deriving custom images from its pre-built base images. Hosts
# commonly cache these layers, and the image is already compatible with Vast's
# SSH launch mode, avoiding a slow first-boot package installation on many hosts.
ARG VAST_BASE=vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py312

FROM ${VAST_BASE} AS builder

ARG LLAMA_CPP_COMMIT=4df29be4f4c3673f428170fda944a5b19f743bb8
ARG FASTMTP_PATCH_URL=https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/993a5971fda8f30dd1b7eb2654792ba4415c7460/HauhauCS-FastMTP-llama.cpp.patch
ARG FASTMTP_PATCH_SHA256=981285400b59dc45cf99936b6ff66d4b3aa0f1b532f85fa51418cb407e51d615

WORKDIR /src
RUN git init llama.cpp \
    && cd llama.cpp \
    && git remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git fetch --depth 1 origin "${LLAMA_CPP_COMMIT}" \
    && git checkout --detach FETCH_HEAD

WORKDIR /src/llama.cpp
RUN curl -fL --retry 5 --retry-all-errors --connect-timeout 15 \
      "${FASTMTP_PATCH_URL}" -o /tmp/fastmtp.patch \
    && echo "${FASTMTP_PATCH_SHA256}  /tmp/fastmtp.patch" | sha256sum -c - \
    && git apply --check /tmp/fastmtp.patch \
    && git apply /tmp/fastmtp.patch

# GGML_NATIVE=OFF is important: GitHub-hosted builders have no NVIDIA GPU.
# CUDA 12.8 lets llama.cpp include Ampere/Ada/Blackwell CUDA targets.
RUN cmake -S . -B build -G Ninja \
      -DGGML_CUDA=ON \
      -DGGML_NATIVE=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release --target llama-server -j"$(nproc)"

FROM ${VAST_BASE} AS runtime

ARG HF_HUB_VERSION=1.28.0

# The Vast base image already contains Python, uv, SSH tooling and the CUDA
# runtime/development libraries. Add only the small Python dependency needed to
# fetch GGUFs at instance start.
RUN /venv/main/bin/pip install --no-cache-dir "huggingface_hub[hf-xet]==${HF_HUB_VERSION}"

COPY --from=builder /src/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY start.sh /usr/local/bin/start.sh

RUN chmod 0755 /usr/local/bin/start.sh \
    && mkdir -p /models

ENV PATH="/venv/main/bin:${PATH}" \
    HF_HOME=/models/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /models

# Vast's SSH launch mode overrides the image ENTRYPOINT and runs start.sh from
# the on-start command. Keeping this entrypoint is still useful for local Docker
# testing and for a future Vast args-mode deployment.
ENTRYPOINT ["/usr/local/bin/start.sh"]
