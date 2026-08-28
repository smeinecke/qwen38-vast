# Qwen3.8 27B FastMTP on Vast.ai

Disposable Vast.ai deployment for:

- `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF`
- llama.cpp + HauhauCS FastMTP
- architecture-specific CUDA images for Ampere, Ada and Blackwell
- editable runtime profiles in `profiles.json`
- self-managed SSH with a public key baked into the image
- local SSH tunnel for the OpenAI-compatible API
- persistent benchmark/telemetry history in `.qwen-runs/`

## v7 architecture change

v7 deliberately **does not use Vast's SSH launch mode anymore**. Vast SSH mode
replaces the image entrypoint and injects/builds additional SSH setup layers at
instance start. That caused two real problems in testing: extra paid cold-start
work and non-deterministic `/root/.ssh/authorized_keys` permissions.

The image now contains `openssh-server`, all utility packages and your public key
already. Vast runs the image in normal `args`/entrypoint mode and only maps
container port 22 to a random public port. `qwen-up` discovers that mapping from
the instance JSON and creates the same local API tunnel as before.

The runtime image also runs `apt-get upgrade` during the CI image build. No
package upgrade/install should be required on the paid Vast GPU instance.

## Images vs runtime profiles

`profiles.json` is the single editable configuration file for architectures,
context defaults and Vast search queries.

There are three compiled CUDA images:

| Image name | CUDA | Stable GHCR tag | GPUs |
|---|---:|---|---|
| `a6000` | SM86 | `:a6000` | RTX A6000 |
| `ada` | SM89 | `:ada-128k` | RTX 6000 Ada, L40S, RTX 5880 Ada |
| `blackwell` | SM120 | `:blackwell-128k` | RTX 5090, RTX PRO 6000 Blackwell |

Runtime profiles can reuse one compiled image. The included profiles are:

| Runtime profile | Image | Context |
|---|---|---:|
| `a6000` | `:a6000` | 65,536 |
| `a6000-128k` | `:a6000` | 131,072 |
| `ada-128k` | `:ada-128k` | 131,072 |
| `blackwell-128k` | `:blackwell-128k` | 131,072 |

So testing 64k vs 128k on an A6000 does **not** require another Docker build:

```bash
./qwen-up a6000
./qwen-up a6000-128k
```

Edit `profiles.json` to add e.g. 32k/96k profiles or tighten a region/GPU query.
`qwen-up` no longer contains hard-coded profile cases.

## GitHub Actions builds

### Normal GitHub-hosted workflow

`.github/workflows/docker.yml` builds the architectures from the `images` array
in `profiles.json`. Each architecture is a separate matrix job and can run in
parallel.

The workflow publishes stable tags plus SHA/release tags, for example:

```text
ghcr.io/YOUR_USER/YOUR_REPO:a6000
ghcr.io/YOUR_USER/YOUR_REPO:a6000-sha-abc1234
ghcr.io/YOUR_USER/YOUR_REPO:ada-128k
ghcr.io/YOUR_USER/YOUR_REPO:blackwell-128k
```

### Manual 56-core self-hosted workflow

`.github/workflows/docker-self-hosted.yml` is **workflow_dispatch only**. It
never runs on a push automatically. In the Actions UI select:

```text
Manual CUDA build on local runner
```

and choose one of:

```text
a6000
ada
blackwell
all
```

For a single selected architecture the Docker build sees all CPUs exposed by
your local runner and llama.cpp builds with `-j$(nproc)`. With one registered
self-hosted runner, choosing `all` queues the three matrix jobs serially; choose a
single target when you want the full 56 cores focused on one rebuild.

The workflows use the current Node-24 Docker Actions. Keep the self-hosted GitHub
Actions runner updated (Node-24 actions require a recent Actions runner).

## SSH public key baked into the image

Before each CI Docker build `scripts/prepare-authorized-keys` creates
`ssh/authorized_keys.generated`.

By default it downloads the public SSH keys of the GitHub repository owner from:

```text
https://github.com/<repository-owner>.keys
```

For a personal repository that requires no additional configuration. The full
key is not printed into CI logs; only key fingerprints are shown.

For an organization repository or a different key, set one of these GitHub
**repository variables**:

```text
QWEN_SSH_GITHUB_USER = your-github-user
```

or:

```text
QWEN_SSH_PUBLIC_KEY = ssh-ed25519 AAAA... user@host
```

You can also commit one or more public keys to `ssh/authorized_keys`. Public keys
are not credentials/secrets, but remember that every private key matching a key
baked into the image receives root SSH access to instances created from it.

At container startup `qwen-init-ssh.sh` merges the baked keys, forces:

```text
/root/.ssh                 0700 root:root
/root/.ssh/authorized_keys 0600 root:root
```

generates fresh per-instance SSH host keys and starts `sshd`. This avoids the
Vast `authorized_keys` ownership/mode failure seen with injected SSH mode.

As an emergency runtime override you may also put a quoted public key in local
`.env`:

```dotenv
SSH_PUBLIC_KEY='ssh-ed25519 AAAA... user@host'
```

`qwen-up` base64-encodes and injects that public key into the disposable
container. Normally this is unnecessary because the CI image already contains
the key.

## 1. Local dependencies

Linux/macOS/WSL:

```bash
python3 -m pip install --upgrade vastai
sudo apt-get install -y jq curl openssh-client openssl
```

Authenticate Vast with either:

```bash
vastai set api-key YOUR_VAST_API_KEY
```

or `VAST_API_KEY` in `.env`.

A Vast account SSH key is no longer required for this deployment path; SSH is
provided by the container itself.

## 2. Configure

```bash
cp .env.example .env
chmod 600 .env
```

At minimum set:

```dotenv
GHCR_IMAGE_BASE=ghcr.io/YOUR_USER/YOUR_REPO
```

`HF_TOKEN` is optional because the current model repository is public.

## 3. Start

```bash
./qwen-up a6000
./qwen-up a6000-128k
./qwen-up ada-128k
./qwen-up blackwell-128k
```

Calling `./qwen-up` without an argument uses `QWEN_PROFILE` from `.env`, falling
back to `default_profile` from `profiles.json`.

One-off context test:

```bash
CTX_SIZE_OVERRIDE=98304 ./qwen-up a6000
```

The script:

1. resolves the runtime profile from `profiles.json`;
2. selects the corresponding architecture image/tag;
3. finds the cheapest matching Vast offer below `MAX_DPH`;
4. creates the instance in normal entrypoint/args mode with `-p 22:22`;
5. waits for Vast's public port-22 mapping;
6. connects to the image's own `sshd`;
7. creates a local `localhost:18080 -> instance:127.0.0.1:8080` tunnel;
8. waits while `hf_xet` downloads the GGUF/FastMTP sidecar and llama.cpp loads;
9. waits for `/health`;
10. stores local state and telemetry.

Load the generated client environment:

```bash
source .qwen-vast/env
```

Then:

```bash
curl "$OPENAI_BASE_URL/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

The inference API itself is still bound only to `127.0.0.1` inside the Vast
container. Only SSH port 22 is public.

## 4. Logs and status

```bash
./qwen-status
./qwen-logs
```

`qwen-status` discovers either:

- v7 custom port mapping: `.public_ipaddr` + `.ports["22/tcp"].HostPort`, or
- legacy v6 Vast SSH fields: `.ssh_host` + `.ssh_port`.

It recreates a missing local tunnel automatically.

`qwen-logs` tails:

```text
/var/log/qwen38/server.log
```

and saves a local live copy under the current `.qwen-runs/<session>/` directory.
The container entrypoint also mirrors the same log to Vast's normal container
stdout so the web UI remains useful.

## 5. Telemetry and benchmarks

Every run receives a local directory such as:

```text
.qwen-runs/
└── 20260827T101530Z-a6000-12345/
    ├── metadata.json
    ├── client.log
    ├── gpu-start.txt
    ├── metrics-ready.prom
    └── benchmarks/
```

Run the built-in coding benchmark:

```bash
./qwen-bench
```

Or a real prompt/context:

```bash
./qwen-bench \
  --label repo-analysis \
  --prompt-file /tmp/repo-context.txt \
  --max-tokens 1024
```

The benchmark records prompt/decode throughput, TTFT, latency, MTP draft and
acceptance counts, GPU utilization, peak VRAM, power and estimated request cost.
The prompt is not copied by default; add `--save-prompt` if desired.

Compare runs:

```bash
./qwen-results
./qwen-results --csv > qwen-benchmarks.csv
./qwen-results --json > qwen-benchmarks.json
```

## 6. Stop billing

```bash
./qwen-down --yes
```

Before destroy, the script best-effort archives remote server logs, final
`/metrics`, Vast metadata and a final GPU snapshot. `404 / instance not found` is
treated as a successful already-absent end state.

If startup fails, the paid instance is destroyed automatically unless:

```dotenv
KEEP_ON_FAILURE=1
```

is explicitly set.

## Cost/search controls

`MAX_DPH` is a hard local guard. Example:

```dotenv
MAX_DPH=0.55
```

The normal GPU queries are in `profiles.json`, not in shell code. For a one-off
query use `GPU_QUERY_OVERRIDE`.

Example Europe-only override:

```dotenv
GPU_QUERY_OVERRIDE='num_gpus=1 gpu_ram>=48 cpu_ram>=32 reliability>0.98 inet_down>=800 disk_bw>=300 cuda_vers>=12.8 disk_space>=60 rented=False rentable=True direct_port_count>=1 geolocation in [DE,NL,FR,SE,FI]'
```

## Cold-start changes in v7

The model weights are intentionally **not** baked into the image. The Q4 model +
FastMTP sidecar still have to be fetched onto each disposable Vast disk.
`HF_XET_HIGH_PERFORMANCE=1` is enabled and the default queries require reasonably
fast download/disk performance.

What v7 removes is Vast's additional SSH compatibility build. In the problematic
v6 logs Vast first pulled the GHCR image, then generated an extra `.../ssh`
child image, ran `apt-get update/install`, rewrote SSH configuration, and only
then started the instance. v7 runs the already-prepared image as-is.

The builder still uses Vast's CUDA devel image, but the final inference image now
uses the much smaller NVIDIA CUDA 12.8 runtime base. This avoids shipping the
compiler/toolkit/cuDNN development stack to every disposable Vast host. We still
do **not** squash the final image into one giant layer: squashing destroys layer
reuse and makes retries/cache hits worse.

## Docker build details

Builder base:

```text
vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py312
```

Runtime base:

```text
nvidia/cuda:12.8.1-runtime-ubuntu24.04
```

llama.cpp commit:

```text
4df29be4f4c3673f428170fda944a5b19f743bb8
```

FastMTP patch SHA-256:

```text
981285400b59dc45cf99936b6ff66d4b3aa0f1b532f85fa51418cb407e51d615
```

Model revision:

```text
993a5971fda8f30dd1b7eb2654792ba4415c7460
```

The image contains no model weights or private credentials. It does contain the
configured **public** SSH key(s).

## Updating llama.cpp / FastMTP

Do not change the pinned llama.cpp commit while blindly keeping the same FastMTP
patch. Update both together from the model author's instructions, then rebuild.

Embedded-MTP fallback:

```dotenv
USE_FASTMTP=0
```

## Upstream

- Model: https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Vast Docker/launch modes: https://docs.vast.ai/guides/instances/docker-environment
- Vast networking/ports: https://docs.vast.ai/guides/instances/connect/networking
- Hugging Face downloads: https://huggingface.co/docs/huggingface_hub/guides/download
