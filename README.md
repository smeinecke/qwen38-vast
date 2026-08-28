# Qwen3.8 27B FastMTP on Vast.ai

Reproducible, disposable deployment for:

- `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF`
- llama.cpp + HauhauCS FastMTP
- one architecture-matched NVIDIA GPU (A6000, Ada, or Blackwell)
- profile-driven 64k/128k context defaults
- Vast.ai billed only while the disposable instance exists/runs
- a local SSH tunnel instead of exposing the inference API directly to the internet
- persistent local run/benchmark telemetry under `.qwen-runs/`

> **v5 migration:** rebuild/push the GHCR images because `start.sh` now enables llama.cpp `/metrics`. Update your local scripts as well; run history is stored outside transient `.qwen-vast/` state.
>
> **v4 migration:** update your local `.env` from `.env.example`. Use `GHCR_IMAGE_BASE=ghcr.io/OWNER/REPO` instead of `:latest`; `qwen-up` now selects the architecture tag and context profile. Legacy `CTX_SIZE`/`GPU_QUERY` values are intentionally ignored in favor of explicit `*_OVERRIDE` variables.

The Docker image contains **no model weights and no credentials**. GitHub Actions builds the CUDA/llama.cpp runtime and publishes it to GHCR. The runtime now derives from Vast.ai's own CUDA base image so SSH-mode hosts can reuse commonly cached/compatible layers instead of spending minutes installing SSH tooling into a generic CUDA image. `qwen-up` rents a suitable Vast host, starts the image, downloads the GGUF from Hugging Face, creates an SSH tunnel, and waits for `/health`. `qwen-down` destroys the Vast instance.

## GitHub Actions builds and stable image tags

GitHub Actions builds three architecture-specific images in parallel on normal
GitHub-hosted runners. Each job compiles exactly one CUDA architecture, so one
slow target cannot consume the build time for all GPUs and no image carries
unused CUDA cubins.

| Profile | CUDA target | Stable GHCR tag | Runtime default |
|---|---:|---|---:|
| `a6000` | SM 86 | `:a6000` | 65,536 context |
| `ada-128k` | SM 89 | `:ada-128k` | 131,072 context |
| `blackwell-128k` | SM 120 | `:blackwell-128k` | 131,072 context |

Every build also publishes architecture-specific immutable SHA tags such as
`:a6000-sha-...`. Release tags become e.g. `:a6000-v5.0.0`. Normal deployment
uses the stable profile names, so `.env` never needs to contain a digest/hash.

The build matrix uses `ubuntu-24.04`, `fail-fast: false`, `max-parallel: 3` and a
330-minute per-job timeout. Ninja already parallelizes compilation across every
CPU core inside each runner; splitting one CMake build across additional fresh
runners would require distributing/interchanging a build tree and is not worth
the complexity. The useful parallelism boundary is therefore one runner per CUDA
architecture.

Each architecture has its own BuildKit and ccache scope. This avoids mixing
SM86/SM89/SM120 CUDA object files while retaining incremental build acceleration.

## Defaults

- Model: `Q4_K_P` (17.9 GB)
- `a6000`: 64k context, throughput-first
- `ada-128k`: 128k context
- `blackwell-128k`: 128k context, throughput-first Blackwell target
- FastMTP depth/runtime: configured by `start.sh`
- Maximum all-in Vast price: `$0.80/h`
- Local OpenAI-compatible endpoint: `http://127.0.0.1:18080/v1`

Qwen3.8-27B has native 262,144-token context and a hybrid attention layout. The
profiles intentionally use smaller defaults because long occupied context reduces
decode throughput substantially during interactive coding sessions.

## 1. Create the public GitHub repository

Create an empty public repository, copy these files into it, and push to `main`.
The workflow publishes:

```text
ghcr.io/YOUR_USER/YOUR_REPO:a6000
ghcr.io/YOUR_USER/YOUR_REPO:ada-128k
ghcr.io/YOUR_USER/YOUR_REPO:blackwell-128k
```

After the first successful build, open the generated GHCR package settings and
change package visibility to **Public**.

## 2. Hugging Face access

The HauhauCS repository is currently publicly downloadable, so **no Hugging Face token is required** for the default deployment. `hf download` runs anonymously when `HF_TOKEN` is empty.

You may still set a fine-grained/read-only token in local `.env` for authenticated downloads/rate limits or as insurance if the upstream repository's access policy changes later. Never commit it. If present, `start.sh` unsets the token before `exec`-ing `llama-server`.

## 3. Install local dependencies

Linux/macOS/WSL:

```bash
python3 -m pip install --upgrade vastai
sudo apt-get install -y jq curl openssh-client openssl   # Debian/Ubuntu/WSL
```

Authenticate Vast either with:

```bash
vastai set api-key YOUR_VAST_API_KEY
```

or set `VAST_API_KEY` in `.env`.

Also add an SSH public key to your Vast account. `qwen-up` uses Vast's SSH launch mode and an SSH local-forward for the API.

## 4. Configure

```bash
cp .env.example .env
chmod 600 .env
```

Edit at least:

```dotenv
GHCR_IMAGE_BASE=ghcr.io/YOUR_USER/YOUR_REPO
```

`HF_TOKEN` is optional. The image tag is selected by the `qwen-up` profile.
Context and GPU search defaults are profile-owned; use explicit `*_OVERRIDE`
variables only for one-off experiments.

## 5. Start

```bash
./qwen-up a6000
./qwen-up ada-128k
./qwen-up blackwell-128k
```

Calling `./qwen-up` without an argument uses `QWEN_PROFILE` from `.env` (default
`a6000`). Aliases such as `ampere`, `ada`, `blackwell`, `5090`, `sm86`, `sm89`
and `sm120` are accepted.

The script:

1. selects a GPU query, GHCR tag and context from the requested profile;
2. rejects anything above `MAX_DPH`;
3. rents the cheapest remaining offer;
4. starts your GHCR image in Vast SSH mode;
5. downloads the target GGUF + ~903 MB FastMTP sidecar with `hf_xet`;
6. starts patched `llama-server` on `127.0.0.1:8080` inside the instance;
7. creates an SSH tunnel to local port `18080`;
8. waits until `/health` responds;
9. writes client environment variables to `.qwen-vast/env`;
10. creates `.qwen-runs/<timestamp>-<profile>-<pid>/` and records startup metadata, selected offer, GPU snapshot and local `qwen-up` output.

Load them:

```bash
source .qwen-vast/env
```

Then test:

```bash
curl "$OPENAI_BASE_URL/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

Example OpenAI-compatible chat call:

```bash
curl "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf",
    "messages": [{"role":"user","content":"Review this function for race conditions."}],
    "max_tokens": 1024
  }'
```

## 6. Logs, telemetry and benchmarks

```bash
./qwen-status
./qwen-logs
```

Every launch gets a persistent local session directory:

```text
.qwen-runs/
└── 20260827T101530Z-a6000-12345/
    ├── metadata.json
    ├── client.log
    ├── gpu-start.txt
    ├── metrics-ready.prom
    └── benchmarks/
```

`.qwen-runs/` is gitignored and is **not** removed by `qwen-down`.

`qwen-logs` still tails `/var/log/qwen38/server.log` through SSH, but it also
appends the live stream to `server-live.log` in the current session. Use
`./qwen-logs --no-save` if you only want terminal output.

`qwen-status` is also self-healing: it refreshes `ssh_host`/`ssh_port` from Vast on every call and recreates the local SSH tunnel if the original `qwen-up` process did not get far enough to persist it or if the tunnel died later. `qwen-logs`, `qwen-bench`, and `qwen-down` use the same refreshed endpoint logic, so an already-paid instance does not need to be recreated just because local SSH state is incomplete.

`qwen-status` now shows profile/context/image, elapsed runtime, estimated compute
cost, current GPU utilization/VRAM/power, health, and selected llama.cpp metric
counters when available.

### Reproducible request benchmark

Run a small built-in coding benchmark:

```bash
./qwen-bench
```

Or benchmark a real code-analysis prompt/context:

```bash
./qwen-bench \
  --label repo-analysis \
  --prompt-file /tmp/repo-context.txt \
  --max-tokens 1024
```

The request is streamed so the client can measure **TTFB/TTFT**, while the final
llama.cpp OpenAI-compatible response supplies its server-side `timings` fields.
For current llama.cpp versions those include prompt/decode throughput and, when
speculative decoding is active, `draft_n` plus `draft_n_accepted`. The script
also scrapes `/metrics` before/after as a fallback and records a 1-second
`nvidia-smi` time series while the request runs.

Each benchmark produces:

```text
benchmarks/<timestamp>-<label>/
├── metrics.json
├── metrics-before.prom
├── metrics-after.prom
├── gpu.csv
├── server-tail.log
├── response.txt
├── reasoning.txt
└── response-final-chunk.json
```

`metrics.json` contains the useful comparison fields directly: prompt tokens,
prompt t/s, generated tokens, decode t/s, TTFT, total latency, MTP draft
acceptance, peak VRAM, average/peak power and estimated compute cost for the
request. The input prompt itself is **not copied by default**; only its SHA-256
and size are stored. Add `--save-prompt` when you explicitly want a copy.

Compare every recorded benchmark:

```bash
./qwen-results
./qwen-results --csv > qwen-benchmarks.csv
./qwen-results --json > qwen-benchmarks.json
```

This is the intended workflow for comparing e.g. A6000 32k/64k against 5090
64k/128k without relying on subjective response speed.

## 7. Stop billing and archive the remote log

```bash
./qwen-down --yes
```

Before destroying the instance, `qwen-down` now best-effort archives the full
remote `/var/log/qwen38/server.log`, final `/metrics`, Vast instance metadata and
a final `nvidia-smi` snapshot into the current `.qwen-runs/<session>/` folder.
It then calculates session duration and estimated GPU compute cost in
`metadata.json`.

If you deliberately do not want the archive step:

```bash
./qwen-down --yes --no-archive
```

The command still calls `vastai destroy instance`, not `stop`. Destroying is
intentional for this disposable workflow: it removes the container/disk instead
of leaving storage charges behind.

If `qwen-up` fails or is interrupted before readiness, it destroys the paid instance automatically unless you explicitly set:

```dotenv
KEEP_ON_FAILURE=1
```

Use that only for debugging.

## Cost controls

`MAX_DPH` is a hard local guard before rental. Example:

```dotenv
MAX_DPH=0.55
```

If no matching host is available below the cap, `qwen-up` exits without renting anything.

Profile-specific GPU queries are built into `qwen-up`. For a one-off custom
Vast query, set `GPU_QUERY_OVERRIDE`. Examples:

Fast Ada-only:

```dotenv
GPU_QUERY_OVERRIDE='num_gpus=1 gpu_name=RTX_6000Ada gpu_ram>=48 cpu_ram>=32 reliability>0.98 inet_down>=500 disk_bw>=200 cuda_vers>=12.8 disk_space>=60 rented=False rentable=True direct_port_count>=1'
```

Cheapest 48 GB option:

```dotenv
GPU_QUERY_OVERRIDE='num_gpus=1 gpu_ram>=48 cpu_ram>=32 reliability>0.98 inet_down>=500 disk_bw>=200 cuda_vers>=12.8 disk_space>=60 rented=False rentable=True direct_port_count>=1'
```

Europe example:

```dotenv
GPU_QUERY_OVERRIDE='num_gpus=1 gpu_name in ["RTX_6000Ada","L40S","RTX_5880Ada","RTX_A6000"] gpu_ram>=48 cpu_ram>=32 reliability>0.98 inet_down>=500 disk_bw>=200 cuda_vers>=12.8 disk_space>=60 rented=False rentable=True direct_port_count>=1 geolocation in [DE,NL,FR,SE,FI]'
```

### SSH says the instance is running but `qwen-up` is still waiting

Vast publishes the direct SSH endpoint as `ssh_host` and `ssh_port` in the instance object. The scripts now use those fields first and only use `vastai ssh-url` as a fallback. Run:

```bash
./qwen-status
```

If the instance is already reachable, this refreshes `.qwen-vast/state.json` and starts/restarts the local `localhost:18080` tunnel. `./qwen-logs` can then connect directly even while the model is still downloading.

If you are upgrading from v5 while an **old v5 `qwen-up` process is still hung in its SSH polling loop**, do not press `Ctrl+C` if you want to keep the paid instance: v5 treats `SIGINT` as startup failure and destroys it. Stop that old process without running its cleanup (for example suspend the shell job with `Ctrl+Z`, then `kill -9 %1`), update the scripts, and run `./qwen-status` to adopt the existing instance. This special recovery step is only needed for a currently-running old script.

## Why the SSH tunnel?

The inference port is deliberately bound to `127.0.0.1` inside the Vast container. Your local client reaches it through SSH port forwarding. This means source code and prompts are encrypted in transit and you do not need to expose a random public HTTP port.

The local generated API key is still enabled as defense in depth.

## Docker build details

The runtime derives from:

```text
vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py312
```

Vast recommends extending its pre-built base images for custom images. They are designed for Vast instance setup and their large base layers are commonly cached on hosts, which reduces cold-start overhead compared with a generic CUDA runtime image. The same base is used by both Docker build stages so GitHub Actions does not need two unrelated multi-gigabyte CUDA base-image families.

The image pins llama.cpp to the commit specified by the HauhauCS FastMTP model card:

```text
4df29be4f4c3673f428170fda944a5b19f743bb8
```

At image-build time it downloads the small FastMTP runtime patch from the model repository and verifies the release-manifest SHA-256 before applying it:

```text
981285400b59dc45cf99936b6ff66d4b3aa0f1b532f85fa51418cb407e51d615
```

The model download is also pinned to the release commit `993a5971fda8f30dd1b7eb2654792ba4415c7460` by default (`HF_REVISION`), so a later change on the model repo's `main` branch does not silently change your deployment.

The model weights are intentionally not baked into the image. This keeps your custom GHCR layers small and lets you switch Q4/Q5 without rebuilding.

`start.sh` enables llama.cpp's Prometheus-compatible `--metrics` endpoint. The
benchmark scripts primarily trust the per-request `timings` object (including
speculative draft counters) and use cumulative `/metrics` counters as a fallback;
this avoids depending on instantaneous throughput gauges.

## Fast download path

The runtime pins `huggingface_hub` with `hf_xet` and sets `HF_XET_HIGH_PERFORMANCE=1`. The default search asks for at least 32 GB system RAM, 500 Mbps download, and 200 MB/s local disk read bandwidth.

The disk filter matters for cold starts: Vast's SSH compatibility/bootstrap steps and model loading can be painfully slow on low-end SATA hosts even when internet bandwidth is good. Raise `disk_bw` to 300–500 if you prefer faster starts over the absolute cheapest hourly offer.

If a host's model download is still poor, increase `inet_down` in `GPU_QUERY` rather than adding persistent Vast storage first. The Q4 target plus FastMTP sidecar is under 20 GB, so a disposable download is usually economical.

## Updating llama.cpp / FastMTP

Do **not** blindly change the pinned llama.cpp commit while continuing to use the same FastMTP patch. Update both together from the model author's current instructions, then rebuild the GHCR image.

The normal embedded-MTP path can be used for troubleshooting without the FastMTP sidecar:

```dotenv
USE_FASTMTP=0
```

## Sources / upstream

- Model and FastMTP instructions: https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Vast CLI docs: https://docs.vast.ai/cli/hello-world
- Vast connection modes: https://docs.vast.ai/guides/instances/connect/overview
- Hugging Face Xet downloads: https://huggingface.co/docs/huggingface_hub/guides/download

Upstream model/patch licensing remains governed by the upstream repository. This deployment repository does not redistribute model weights.
