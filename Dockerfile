# syntax=docker/dockerfile:1.7

# Compile against CUDA in a full devel image, but ship a much smaller CUDA
# runtime image. The previous runtime inherited the ~multi-GB devel toolchain,
# which was unnecessary on disposable inference hosts.
# Older architectures (e.g. Volta V100 / SM70) are pinned to a CUDA 12.2 base.
ARG BUILDER_BASE=vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py312
ARG RUNTIME_BASE=nvidia/cuda:12.8.1-runtime-ubuntu24.04

FROM ${BUILDER_BASE} AS builder

ARG LLAMA_CPP_COMMIT=4df29be4f4c3673f428170fda944a5b19f743bb8
ARG FASTMTP_PATCH_URL=https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/993a5971fda8f30dd1b7eb2654792ba4415c7460/HauhauCS-FastMTP-llama.cpp.patch
ARG FASTMTP_PATCH_SHA256=981285400b59dc45cf99936b6ff66d4b3aa0f1b532f85fa51418cb407e51d615

# The GitHub Actions matrix builds one native CUDA target per image. Keeping a
# single SM target per image cuts cold compile time and avoids shipping unused
# cubins. Local builds may still pass a semicolon-separated list explicitly.
#   86  = Ampere GA102 (RTX A6000, A40)
#   89  = Ada Lovelace (RTX 4090, RTX 6000 Ada, RTX 5880 Ada, L40/L40S)
#   120 = Blackwell GeForce/RTX PRO (RTX 5090, RTX PRO 6000 Blackwell)
ARG CUDA_ARCHITECTURES=86
ARG QWEN_BUILD_PROFILE=custom
# Passed by CI from the compiler-cache key. Referencing this value in the
# compile RUN deliberately invalidates only that layer when the compiler-cache
# generation changes. This guarantees one real compile to populate an empty
# BuildKit cache mount instead of forever restoring an old compiled layer.
ARG CCACHE_SEED=manual

# ccache is intentionally installed only in the builder. The final runtime image
# stays unchanged. The compiler cache itself is a BuildKit cache mount, so it is
# never copied into the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ccache build-essential cmake curl git ninja-build \
    && rm -rf /var/lib/apt/lists/*

ENV CCACHE_DIR=/root/.cache/ccache \
    CCACHE_MAXSIZE=3G \
    CCACHE_COMPRESS=true \
    CCACHE_COMPRESSLEVEL=6

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
# CCACHE_SEED is intentionally consumed here. Without it, BuildKit may restore
# this entire RUN from the GHA layer cache before ccache ever gets a chance to
# populate its persistent cache mount (which is exactly what happened in v9).
RUN --mount=type=cache,id=qwen38-ccache,target=/root/.cache/ccache,sharing=locked \
    printf 'ccache seed: %s\n' "${CCACHE_SEED}" \
    && ccache --zero-stats \
    && cmake -S . -B build -G Ninja \
      -DGGML_CUDA=ON \
      -DGGML_NATIVE=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    && cmake --build build --config Release --target llama-server -j"$(nproc)" \
    && ccache --show-stats

FROM ${RUNTIME_BASE} AS runtime

ARG HF_HUB_VERSION=1.28.0
ARG LLAMA_CPP_COMMIT=4df29be4f4c3673f428170fda944a5b19f743bb8
ARG CUDA_ARCHITECTURES=86
ARG QWEN_BUILD_PROFILE=custom

LABEL io.qwen38.profile="${QWEN_BUILD_PROFILE}" \
      io.qwen38.cuda-arch="${CUDA_ARCHITECTURES}" \
      io.qwen38.llama-cpp-commit="${LLAMA_CPP_COMMIT}"

# Pre-install and upgrade everything needed by the disposable runtime. In v6 we
# asked Vast for SSH launch mode; Vast then built a child /ssh image at instance
# start and ran apt-get itself. v7 uses normal entrypoint mode and self-manages
# sshd, so all package work happens once here in CI instead of on paid GPU time.
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends \
         ca-certificates \
         openssh-server \
         openssh-client \
         tmux \
         git \
         rsync \
         wget \
         curl \
         less \
         locales \
         sudo \
         libgomp1 \
         python3 \
         python3-venv \
    && python3 -m venv /venv/main \
    && /venv/main/bin/pip install --no-cache-dir --upgrade pip \
    && /venv/main/bin/pip install --no-cache-dir "huggingface_hub[hf-xet]==${HF_HUB_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/ssh/ssh_host_*

COPY --from=builder /src/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY start.sh entrypoint.sh qwen-init-ssh.sh /usr/local/bin/
COPY ssh/ /etc/qwen38/ssh/

# llama-server is mostly statically linked, but GCC OpenMP remains a runtime
# dependency (libgomp.so.1). Catch this in CI instead of discovering it only
# after a paid Vast instance has downloaded the model and entered a restart loop.
RUN ldconfig \
    && ldd /usr/local/bin/llama-server > /tmp/llama-server.ldd \
    && cat /tmp/llama-server.ldd \
    && if grep -Eq 'libgomp\.so\.1[[:space:]]*=>[[:space:]]*not found' /tmp/llama-server.ldd; then \
         echo >&2 'ERROR: libgomp.so.1 is missing from runtime image'; exit 2; \
       fi

# Fail CI rather than publishing an entrypoint-mode image that nobody can SSH
# into. Workflows create ssh/authorized_keys.generated before docker build.
RUN chmod 0755 /usr/local/bin/start.sh /usr/local/bin/entrypoint.sh /usr/local/bin/qwen-init-ssh.sh \
    && mkdir -p /models /run/sshd /root/.ssh /etc/ssh/sshd_config.d \
    && chmod 0700 /root/.ssh \
    && if ! grep -hE '^[[:space:]]*(ssh-|ecdsa-|sk-)[^[:space:]]+[[:space:]]+[^[:space:]]+' /etc/qwen38/ssh/authorized_keys* >/dev/null 2>&1; then \
         echo >&2 'ERROR: image build contains no SSH public key; run scripts/prepare-authorized-keys first'; exit 2; \
       fi \
    && printf '{"llama_cpp_commit":"%s","cuda_arch":"%s","build_profile":"%s"}\n' \
         "$LLAMA_CPP_COMMIT" "$CUDA_ARCHITECTURES" "$QWEN_BUILD_PROFILE" > /etc/qwen38-build.json \
    && chmod 0444 /etc/qwen38-build.json \
    && printf '%s\n' \
         'PermitRootLogin prohibit-password' \
         'PubkeyAuthentication yes' \
         'PasswordAuthentication no' \
         'KbdInteractiveAuthentication no' \
         'ChallengeResponseAuthentication no' \
         'StrictModes yes' \
         'AllowTcpForwarding yes' \
         'GatewayPorts no' \
         'X11Forwarding no' \
         'ClientAliveInterval 15' \
         'ClientAliveCountMax 3' \
         'LogLevel VERBOSE' \
       > /etc/ssh/sshd_config.d/99-qwen38.conf

ENV PATH="/venv/main/bin:${PATH}" \
    HF_HOME=/models/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /models

# qwen-up requests `-p 22:22` explicitly. Vast maps that container port to a
# random external host port; the scripts discover it from show-instance JSON.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
