# Qwen3.8 27B FastMTP on Vast.ai

Disposable Vast.ai deployment for:

- `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF`
- llama.cpp + HauhauCS FastMTP
- architecture-specific CUDA images for Ampere, Ada and Blackwell
- editable runtime profiles in `profiles.json`
- self-managed SSH with a public key baked into the image
- local SSH tunnel for the OpenAI-compatible API
- persistent benchmark/telemetry history in `.qwen-runs/`
- automatic cross-instance llama.cpp slot/KV snapshots on the external cache server

## v8 persistent slot/KV cache

v8 automatically persists the single llama.cpp slot between disposable Vast
instances. The configured cache server is currently:

```text
94.16.105.121
```

The remote server does **not** run llama.cpp. It only needs SSH and `rsync` and
stores `current.bin` plus a small JSON metadata file. The default cache root is
`~/qwen-slot-cache` for the configured cache user, so no root access is needed.

A dedicated SSH key is generated locally and is **not** baked into the image or
committed. One-time setup:

```bash
cp .env.example .env
# If the existing remote user is not qwen-cache, edit QWEN_SLOT_CACHE_USER.
./qwen-cache-setup
```

If the cache account has another name:

```bash
./qwen-cache-setup youruser@94.16.105.121
```

The setup command uses your existing SSH access once, adds a dedicated
`restrict`ed Ed25519 key, verifies that `rsync` exists on the cache server,
creates the remote cache directory and verifies the new key. Afterwards the
whole lifecycle is automatic. If `rsync` is missing, install it once on the
cache server (Ubuntu/Debian: `sudo apt install rsync`).

For independent coding contexts, choose a logical session name when starting:

```bash
./qwen-up a6000-128k --session my-project
```

The session name is persisted in the Vast run state, so `qwen-down` automatically
uploads back into the same namespace. No download/upload command is needed.

### Automatic startup

`qwen-up`:

1. checks that the dedicated cache key exists before renting;
2. copies that limited cache key to the new ephemeral Vast host;
3. computes a compatibility signature from llama.cpp commit, model, HF revision,
   context size, FastMTP setting and KV precision;
4. starts downloading a matching `current.bin` from the cache server **in
   parallel with model loading**;
5. after `/health` is ready, calls llama.cpp
   `POST /slots/0?action=restore` automatically;
6. records `n_restored`, bytes read and restore state in `.qwen-runs/`.

No cache file for the exact signature simply means a normal cold context.
A6000-128k and Ada/Blackwell-128k can share a snapshot when all signature inputs
match; a 64k and 128k context intentionally do not share one.

### Automatic shutdown

`qwen-down` now does this before destroying the paid host:

```text
slot 0 -> llama.cpp save -> Vast NVMe current.bin -> rsync -> 94.16.105.121
        -> atomic current.bin replacement -> retention -> destroy Vast instance
```

Normal use is unchanged; the slot save and upload happen automatically:

```bash
./qwen-down --yes
```

Emergency opt-out:

```bash
./qwen-down --yes --no-cache
```

By default a cache upload failure is logged but the instance is still destroyed
so a storage outage cannot accidentally keep GPU billing running. Set:

```dotenv
QWEN_SLOT_CACHE_REQUIRE_SAVE=1
```

if losing the latest snapshot is worse than leaving the Vast instance running.

The 100 GB cache server defaults to `QWEN_SLOT_CACHE_MAX_GB=80`. After a
successful upload, v8 deletes the oldest *other* session/signature snapshots
until usage is below the budget. The snapshot just uploaded is never pruned.

### Important Qwen3.8 restore verification

Qwen3.8-27B uses the `qwen3_5` hybrid architecture (linear/recurrent attention
plus periodic full attention). Recent upstream llama.cpp versions have an open
issue where slot restore on hybrid/recurrent models can report `n_restored`
successfully but still re-prefill the next request. The pinned FastMTP commit in
this repo is different, so v8 does not assume either outcome.

`qwen-bench` now records the actual llama.cpp `cache_n` value and displays cache
hit percentage. After the first save/restart/restore, run the same long-prefix
request again and trust the cache only if `cache_n` is non-zero/high:

```bash
./qwen-bench --label restored-context --prompt-file /tmp/same-long-prefix.txt
./qwen-results
```

The table now contains a `cache` column. This distinguishes "restore API said
success" from real avoided prompt processing.

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
| `a6000` | SM86 | `:a6000` | RTX A6000, A40 |
| `ada` | SM89 | `:ada-128k` | RTX 4090, RTX 6000 Ada, L40/L40S, RTX 5880 Ada |
| `blackwell` | SM120 | `:blackwell-128k` | RTX 5090, RTX PRO 6000 Blackwell |

Runtime profiles can reuse one compiled image. The included profiles are:

| Runtime profile | Image | Context | Search intent |
|---|---|---:|---|
| `4090-32k` | `:ada-128k` | 32,768 | cheapest fast 24 GB Ada option |
| `a6000` | `:a6000` | 65,536 | exact RTX A6000 |
| `ampere-value` | `:a6000` | 65,536 | cheapest RTX A6000 or A40 |
| `a6000-128k` | `:a6000` | 131,072 | exact RTX A6000 |
| `ampere-value-128k` | `:a6000` | 131,072 | cheapest RTX A6000 or A40 |
| `a6000-256k` | `:a6000` | 262,144 | exact RTX A6000, full native-context test profile |
| `ampere-value-256k` | `:a6000` | 262,144 | RTX A6000 or A40, full native-context test profile |
| `ada-64k` | `:ada-128k` | 65,536 | 48 GB Ada-class (6000 Ada/L40/L40S/5880 Ada) |
| `ada-128k` | `:ada-128k` | 131,072 | 48 GB Ada-class (6000 Ada/L40/L40S/5880 Ada) |
| `ada-256k` | `:ada-128k` | 262,144 | 48 GB Ada-class, full native-context test profile |
| `5090-64k` | `:blackwell-128k` | 65,536 | exact RTX 5090 |
| `5090-128k` | `:blackwell-128k` | 131,072 | exact RTX 5090 |
| `blackwell-128k` | `:blackwell-128k` | 131,072 | RTX 5090 or RTX PRO 6000 |
| `blackwell-256k` | `:blackwell-128k` | 262,144 | RTX PRO 6000 96 GB only |

`256k` means the model's full 262,144-token native context. These are runtime
profiles only; they reuse the existing SM86/SM89/SM120 images and therefore do
not add CUDA builds. The 48 GB 256k profiles are deliberately explicit test
profiles: actual headroom still depends on the selected model, KV-cache type and
FastMTP configuration. The 96 GB RTX PRO 6000 profile is the conservative 256k
Blackwell choice.

The market monitor uses the exact running context size plus `monitor_hardware` in
`profiles.json`. `gpu_ranks` is an intentionally editable ordering, not a benchmark
score. A candidate offer is accepted only when its concrete GPU rank is **equal to
or higher than the currently running GPU**. This prevents an A6000 run from being
replaced by a cheaper but slower A40. Unknown GPU models are rejected
conservatively until added to the rank table. Broad value profiles are marked
`monitor_search=true`; overlapping exact-GPU profiles can set it to `false` to
avoid duplicate Vast API searches.

Testing 64k, 128k and 256k on an A6000 does **not** require another Docker build:

```bash
./qwen-up a6000
./qwen-up a6000-128k
./qwen-up a6000-256k
```

Edit `profiles.json` to tune GPU ordering, add another context profile or tighten a region/GPU query.
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
sudo apt-get install -y jq curl openssh-client openssl rsync
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
QWEN_SLOT_CACHE_USER=qwen-cache   # change if your existing server user differs
```

Then prepare the external cache once:

```bash
./qwen-cache-setup
```

`HF_TOKEN` is optional because the current model repository is public.

## 3. Start

```bash
./qwen-up a6000
./qwen-up ampere-value-128k
./qwen-up ada-128k
./qwen-up 5090-128k
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
7. installs the dedicated external-cache key on the disposable host;
8. prefetches a compatible persistent slot snapshot from `94.16.105.121` while the model loads;
9. creates a local `localhost:18080 -> instance:127.0.0.1:8080` tunnel;
10. waits while `hf_xet` downloads the GGUF/FastMTP sidecar and llama.cpp loads;
11. waits for `/health` and restores slot 0 on a cache hit;
12. stores local state and telemetry.

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

### Local tunnel port and concurrent commands

`LOCAL_PORT=18080` is only the preferred local port. With the default
`LOCAL_PORT_AUTO=1`, qwen scripts automatically select the next free localhost
port if 18080 is occupied and store the actual port in `.qwen-vast/state.json`.
Tunnel creation is serialized, so running `qwen-status` while `qwen-up` is still
provisioning can no longer race `qwen-up` for the same `ssh -L` listener.

A transient SSH disconnect while the model is loading is retried until the
overall `START_TIMEOUT`; it no longer causes immediate instance destruction.
Set `LOCAL_PORT_AUTO=0` only if a fixed local port is a hard requirement.

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

## Live Vast price monitor

`qwen-monitor` compares the **all-in hourly price of the currently running
instance** (`dph_total`) with currently rentable Vast offers. A candidate must
use the **same context size** and its concrete GPU must have an equal-or-higher
rank in `profiles.json -> monitor_hardware.gpu_ranks`. Search uses the same
configured disk size, so the comparison is not just the bare GPU price.

The default ordering is conservative and editable. For example, A40 ranks below
RTX A6000, while Ada 48 GB, RTX 5090 and RTX PRO 6000 rank above it. An unknown
GPU is never treated as an upgrade automatically.

One-shot check:

```bash
./qwen-monitor once
```

Foreground watch:

```bash
./qwen-monitor watch --threshold 10 --interval 180
```

Background daemon:

```bash
./qwen-monitor start
./qwen-monitor status
./qwen-monitor logs
./qwen-monitor stop
```

Defaults in `.env`:

```dotenv
QWEN_MONITOR_THRESHOLD_PCT=10
QWEN_MONITOR_INTERVAL=180
QWEN_MONITOR_MAX_RESULTS=5
QWEN_MONITOR_AUTO_START=0
```

When a qualifying offer appears, the monitor prints the profile, GPU, all-in
`$/h`, percentage saving, location, download speed, disk speed, reliability and
Vast offer ID. It also stores the best alert in
`.qwen-vast/monitor-alert.json`; `qwen-status` surfaces that alert. If a desktop
notification service is available, `notify-send` is used best-effort.

For a compatible profile switch, preserve the external slot cache normally:

```bash
uv run ./qwen-down --yes
uv run ./qwen-up 5090-128k --session my-project
```

`qwen-down` automatically stops a background monitor before saving the slot and
destroying the instance. Set `QWEN_MONITOR_AUTO_START=1` if every successful
`qwen-up` should start the monitor automatically.

If `CTX_SIZE_OVERRIDE` uses a context size for which no explicit profile exists,
the monitor falls back to the running profile. Any suggested restart preserves
the exact current context with `CTX_SIZE_OVERRIDE`, so a price alert never
silently changes context length.

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

## Compiler cache diagnostics (v9.1)

There are intentionally **two independent build caches**:

1. BuildKit GHA layer cache (`cache-from/cache-to type=gha`) can restore the entire
   llama.cpp compile `RUN` without invoking the compiler at all.
2. `ccache` is persisted from the BuildKit cache mount via
   `buildkit-cache-dance` + `actions/cache`. It becomes useful when the compile
   layer is invalidated but most translation units are unchanged.

A message such as `Cache not found for input keys` only means the exact
`actions/cache` key was not found. A following `Cache hit for restore-key` is a
valid fallback hit. What matters for ccache is the restored archive size: an old
~300-byte archive was effectively empty. v9.1 deliberately executes the compile
layer once for a new compiler-cache generation so that cache is seeded.

On the first v9.1 build expect a real compile and a sizeable cache save. On the
next build with unchanged build inputs expect an exact v4 cache hit; BuildKit may
also report the compile layer itself as `CACHED`, which is even faster.

For the manual self-hosted workflow, Buildx additionally uses a persistent named
`qwen38-local-builder` (`keep-state: true`, `cleanup: false`). On the 56-core
runner this makes the local BuildKit state the fastest first-level cache; the GHA
ccache archive remains a recovery/portability layer.
