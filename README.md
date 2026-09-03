# Qwen3.8 27B FastMTP on Vast.ai

Disposable Vast.ai deployments for:

- `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF`
- llama.cpp + HauhauCS FastMTP
- architecture-specific CUDA images for Ampere, Ada and Blackwell
- editable runtime profiles in `profiles.json`
- self-managed SSH with a public key baked into the image
- local SSH tunnel for the OpenAI-compatible API
- persistent benchmark/telemetry history in `.hostai-runs/`
- persistent cross-instance llama.cpp slot/KV snapshots via **SSH/rsync or rclone**

## Quickstart

A step-by-step getting-started guide is in [QUICKSTART.md](QUICKSTART.md). It
covers dependencies, `.env` setup, SSH/rsync and rclone cache backends, starting
an instance, the OpenAI-compatible API, logs, benchmarks, the price monitor and
stopping billing.

## What this does

`hostai` is a small CLI that rents a disposable Vast.ai GPU instance, runs a
Docker image with llama.cpp, loads the Qwen3.8 27B GGUF model and exposes an
OpenAI-compatible API through a local SSH tunnel. The instance is destroyed when
you are done, and the llama.cpp slot/KV cache can be uploaded to an external
backend so the next deployment reuses the previous context.

The model weights are downloaded at instance startup, not baked into the image.
`HF_XET_HIGH_PERFORMANCE=1` is enabled to speed up transfers.

## Tokenized-only mode

`hostai` supports client-side tokenization as a defense-in-depth option. When
`HOSTAI_TOKENIZED_ONLY=1` is enabled:

- `hostai up` launches the remote `hostai-guard` in front of `llama-server`.
- The guard blocks all text-bearing endpoints (`/v1/chat/completions`,
  `/tokenize`, `/apply-template`, `/detokenize`, `/infill`, etc.).
- Only the native `/completion` endpoint is allowed, and only when the request
  `prompt` is a JSON array of integer token IDs.
- You run `hostai proxy` locally. It listens on a Unix socket, applies the model
  chat template with `transformers`, and forwards token IDs to the remote
  `llama-server` through the SSH tunnel.
- The generated text is then decoded and streamed back as an OpenAI-compatible
  response.

This makes prompt extraction from the remote VM more difficult because the
remote never sees the raw user text. The feature is **not** absolute protection:
a root-level attacker with the tokenizer can still reverse token IDs.

Enable it in `hostai.toml` or as a shell environment variable:

```dotenv
HOSTAI_TOKENIZED_ONLY=1
```

By default the proxy uses a Unix socket. Clients that cannot use Unix sockets
can request a local TCP port as well:

```dotenv
HOSTAI_PROXY_PORT=18081
```

The proxy tokenizer is pinned to the Qwen3.8-27B base model at a known-good
revision so tokenization stays reproducible.  Override it only when you have
verified a new revision with `hostai test-tokenizer` or the golden tests:

```dotenv
HOSTAI_PROXY_TOKENIZER_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

After `hostai up`:

```bash
# In one terminal
uv run hostai proxy

# In another terminal
source .hostai-vast/env
curl "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"hello"}]}'
```

If `HOSTAI_PROXY_PORT` is not set, `OPENAI_BASE_URL` is omitted from the
`.hostai-vast/env` file and the comment explains how to use
`$HOSTAI_PROXY_SOCKET` or set `HOSTAI_PROXY_PORT`.

`hostai bench` does not support tokenized-only mode yet. Disable the toggle to
use benchmarks.

## Slot/KV cache

The cache persists the llama.cpp slot between disposable Vast instances. The
remote cache stores `current.bin` plus a small JSON metadata file. The default
cache root is `~/qwen-slot-cache` for the configured cache user, so no root
access is needed.

The cache is configured in `.env` (see `.env.example`). You can use either of
two backends.

### Option A: SSH/rsync backend

A dedicated SSH key is generated locally and is not baked into the image or
committed. One-time setup:

```bash
cp .env.example .env
# If the existing remote user is not qwen-cache, edit HOSTAI_SLOT_CACHE_USER.
uv run hostai cache setup
```

If the cache account has another name or host:

```bash
uv run hostai cache setup youruser@your-cache-host.example.com
```

The setup command uses your existing SSH access once, adds a dedicated
`restrict`ed Ed25519 key, verifies that `rsync` exists on the cache server,
creates the remote cache directory and verifies the new key. Afterwards the
whole lifecycle is automatic. If `rsync` is missing, install it once on the
cache server (Ubuntu/Debian: `sudo apt install rsync`).

### Option B: rclone backend

Instead of SSH/rsync, the cache can use any rclone-supported backend (WebDAV,
S3, SFTP, etc.). Set in `.env`:

```dotenv
HOSTAI_SLOT_CACHE_RCLONE=1
HOSTAI_SLOT_CACHE_RCLONE_TYPE=webdav
HOSTAI_SLOT_CACHE_RCLONE_URL=https://your-cache-server.example.com/
HOSTAI_SLOT_CACHE_RCLONE_PASSWORD=your-webdav-password
```

`HOSTAI_SLOT_CACHE_HOST`, `PORT` and `KEY` are ignored when rclone is enabled.
`HOSTAI_SLOT_CACHE_USER` still sets the default backend user unless you override
it with `HOSTAI_SLOT_CACHE_RCLONE_USER`. Use `HOSTAI_SLOT_CACHE_RCLONE_REMOTE` to
use a preconfigured rclone remote in the image instead of supplying a URL.

For independent coding contexts, choose a logical session name when starting:

```bash
uv run hostai up a6000-128k --session my-project
```

The session name is persisted in the Vast run state, so `hostai down`
automatically uploads back into the same namespace.

### Lifecycle

`hostai up`:

1. checks that the dedicated cache key exists before renting (SSH/rsync only);
2. copies that limited cache key to the new ephemeral Vast host (SSH/rsync only);
3. computes a compatibility signature from llama.cpp commit, model, HF revision,
   context size, FastMTP setting and KV precision;
4. starts downloading a matching `current.bin` from the cache server **in
   parallel with model loading**;
5. after `/health` is ready, calls llama.cpp `POST /slots/0?action=restore`
   automatically;
6. records `n_restored`, bytes read and restore state in `.hostai-runs/`.

No cache file for the exact signature simply means a normal cold context.
A6000-128k and Ada/Blackwell-128k can share a snapshot when all signature inputs
match; a 64k and 128k context intentionally do not share one.

`hostai down` saves slot 0, uploads it to the cache and then destroys the paid
instance:

```text
slot 0 -> llama.cpp save -> Vast NVMe current.bin -> rsync/rclone -> cache server
        -> atomic current.bin replacement -> retention -> destroy Vast instance
```

Normal use is unchanged; the slot save and upload happen automatically:

```bash
uv run hostai down --yes
```

Emergency opt-out:

```bash
uv run hostai down --yes --no-cache
```

By default a cache upload failure is logged but the instance is still destroyed
so a storage outage cannot accidentally keep GPU billing running. Set:

```dotenv
HOSTAI_SLOT_CACHE_REQUIRE_SAVE=1
```

if losing the latest snapshot is worse than leaving the Vast instance running.

The cache server defaults to `HOSTAI_SLOT_CACHE_MAX_GB=80`. After a successful
upload, the oldest *other* session/signature snapshots are deleted until usage is
below the budget. The snapshot just uploaded is never pruned.

### Restore verification

Qwen3.8-27B uses the `qwen3_5` hybrid architecture (linear/recurrent attention
plus periodic full attention). Recent upstream llama.cpp versions have an open
issue where slot restore on hybrid/recurrent models can report `n_restored`
successfully but still re-prefill the next request. The pinned FastMTP commit in
this repo is different, so do not assume either outcome.

`hostai bench` records the actual llama.cpp `cache_n` value and displays cache
hit percentage. After the first save/restart/restore, run the same long-prefix
request again and trust the cache only if `cache_n` is non-zero/high:

```bash
uv run hostai bench --label restored-context --prompt-file /tmp/same-long-prefix.txt
uv run hostai results
```

The results table contains a `cache` column. This distinguishes "restore API
said success" from real avoided prompt processing.

## Runtime profiles and images

`profiles.json` is the single editable configuration file for architectures,
context defaults and Vast search queries.

There are four compiled CUDA images:

| Image name | CUDA | Stable GHCR tag | GPUs |
|---|---|---:|---|
| `a6000` | SM86 | `:a6000` | RTX A6000, A40 |
| `ada` | SM89 | `:ada-128k` | RTX 4090, RTX 6000 Ada, L40/L40S, RTX 5880 Ada |
| `blackwell` | SM120 | `:blackwell-128k` | RTX 5090, RTX PRO 6000 Blackwell |
| `v100` | SM70 | `:v100` | Tesla V100 |

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
| `blackwell-256k` | `:blackwell-128k` | 262,144 | exact RTX PRO 6000, 96 GB |
| `pro6000-256k` | `:blackwell-128k` | 262,144 | RTX PRO 6000 family (WS/S), 96 GB |
| `v100-128k` | `:v100` | 131,072 | Tesla V100 |

`256k` means the model's full 262,144-token native context. The 256k runtime
profiles reuse the existing SM86/SM89/SM120 images and therefore do not add
CUDA builds; `v100-128k` uses the dedicated `:v100` image. The 48 GB 256k
profiles are deliberately explicit test profiles: actual headroom still depends
on the selected model, KV-cache type and FastMTP configuration. The exact
`blackwell-256k` profile is the conservative 256k Blackwell choice; the broader
`pro6000-256k` profile also matches RTX PRO 6000 WS/S variants.

Edit `profiles.json` to tune GPU ordering, add another context profile or
tighten a region/GPU query.

### Free-traffic market policy

All rental and monitor searches also apply the central `market_policy` from
`profiles.json`. By default both `inet_down_cost` and `inet_up_cost` must be zero.
Vast reports these fields in USD/GB. This is intentionally separate from the GPU
profiles because downloading a ~20 GB container/model can otherwise cost more
than the short GPU rental itself. `hostai up` prints the selected offer's
transfer prices before renting and also validates the raw offer response. Adjust
`HOSTAI_MAX_INET_DOWN_COST` and `HOSTAI_MAX_INET_UP_COST` to allow paid traffic
up to a specific price, or set them to `0` to require free traffic.

### Market monitoring

The market monitor uses the exact running context size plus `monitor_hardware` in
`profiles.json`. `gpu_ranks` is an intentionally editable ordering, not a
benchmark score. A candidate offer is accepted only when its concrete GPU rank is
**equal to or higher than the currently running GPU**. This prevents an A6000 run
from being replaced by a cheaper but slower A40. Unknown GPU models are rejected
conservatively until added to the rank table. Broad value profiles are marked
`monitor_search=true`; overlapping exact-GPU profiles can set it to `false` to
avoid duplicate Vast API searches.

`hostai monitor` compares the **hourly Vast rental price of the currently running
instance** (`dph_total`) with currently rentable Vast offers. A candidate must
match the **active or selected local profile** (same context size, model, and
disk constraints) and its concrete GPU must have an equal-or-higher rank in
`profiles.json -> monitor_hardware.gpu_ranks`. Search uses the same configured
disk size, so the comparison is not just the bare GPU price.  Interruptible
instances are compared against bid offers at their stored bid price.

One-shot check:

```bash
uv run hostai monitor once
```

Foreground watch:

```bash
uv run hostai monitor watch --threshold 10 --interval 180
```

Background daemon:

```bash
uv run hostai monitor start
uv run hostai monitor status
uv run hostai monitor logs
uv run hostai monitor stop
```

Defaults in `.env`:

```dotenv
HOSTAI_MONITOR_THRESHOLD_PCT=10
HOSTAI_MONITOR_INTERVAL=180
HOSTAI_MONITOR_MAX_RESULTS=5
HOSTAI_MONITOR_AUTO_START=0
```

For a compatible profile switch, preserve the external slot cache normally:

```bash
uv run hostai down --yes
uv run hostai up 5090-128k --session my-project
```

`hostai down` automatically stops a background monitor before saving the slot and
destroying the instance. Set `HOSTAI_MONITOR_AUTO_START=1` if every successful
`hostai up` should start the monitor automatically.

## SSH

The image does **not** use Vast's SSH launch mode. Vast SSH mode replaces the
image entrypoint and injects/builds additional SSH setup layers at instance
start. That caused extra paid cold-start work and non-deterministic
`/root/.ssh/authorized_keys` permissions. Instead, the image contains
`openssh-server`, all utility packages and the public key already. Vast runs the
image in normal `args`/entrypoint mode and only maps container port 22 to a
random public port. `hostai up` discovers that mapping from the instance JSON and
creates the same local API tunnel as before.

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
HOSTAI_SSH_GITHUB_USER = your-github-user
```

or:

```text
HOSTAI_SSH_PUBLIC_KEY = ssh-ed25519 AAAA... user@host
```

You can also commit one or more public keys to `ssh/authorized_keys`. Public keys
are not credentials/secrets, but remember that every private key matching a key
baked into the image receives root SSH access to instances created from it.

At container startup `hostai-init-ssh.sh` merges the baked keys, forces:

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

`hostai up` base64-encodes and injects that public key into the disposable
container. Normally this is unnecessary because the CI image already contains the
key.

## Docker build details

Builder base (Ampere/Ada/Blackwell):

```text
vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py312
```

Runtime base (Ampere/Ada/Blackwell):

```text
nvidia/cuda:12.8.1-runtime-ubuntu24.04
```

The `v100` image uses older Volta-compatible bases:

```text
nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
```

The runtime image uses the much smaller NVIDIA CUDA runtime base to avoid
shipping the compiler/toolkit/cuDNN development stack to every disposable Vast
host. We do **not** squash the final image into one giant layer: squashing
destroys layer reuse and makes retries/cache hits worse. The image also runs
`apt-get upgrade` during the CI image build, so no package upgrade/install should
be required on the paid Vast GPU instance.

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
ghcr.io/YOUR_USER/YOUR_REPO:v100
```

The workflow uses Node-24 Docker Actions.

### Cumulative feature validation

The `hostai validate` command checks the repository layout and records a
`validation.json` digest.  With `--production`, it also requires Docker, the
integration image, and passes `tests/test_local_integration.py`, exercising:
`start.sh`, `entrypoint.sh`, SSH/tunnel setup, `llama-server` health, slot/KV
cache handling, and `hostai down` cleanup.  On success, `validation-last-success.json`
is updated with the immutable Docker image ID, git commit, `profiles.json` hash,
and validation level.  `hostai validate --compare` compares the current state
against that last *successful* record.

Set `[vast].require_production_validation = true` or
`HOSTAI_REQUIRE_PRODUCTION_VALIDATION=1` to make `hostai up` refuse real Vast
rentals until the current state matches the last successful production
validation.  Use `hostai up ... --allow-unvalidated` to bypass the gate.

Build the integration image and run it locally with:

```bash
docker build -f tests/integration/Dockerfile.test -t hostai-test:latest .
uv run hostai validate --production
```

### Compiler cache

There are intentionally **two independent build caches**:

1. BuildKit GHA layer cache (`cache-from/cache-to type=gha`) can restore the
   entire llama.cpp compile `RUN` without invoking the compiler at all.
2. `ccache` is persisted from the BuildKit cache mount via
   `buildkit-cache-dance` + `actions/cache`. It becomes useful when the compile
   layer is invalidated but most translation units are unchanged.

A message such as `Cache not found for input keys` only means the exact
`actions/cache` key was not found. A following `Cache hit for restore-key` is a
valid fallback hit. What matters for ccache is the restored archive size. The
first build with a new compiler-cache generation executes the compile layer once
to seed the cache. On the next build with unchanged build inputs expect an exact
`actions/cache` v4 hit; BuildKit may also report the compile layer itself as `CACHED`,
which is even faster.

## Cost/search controls

`MAX_DPH` is a hard local guard. Example:

```dotenv
MAX_DPH=0.55
```

Additional search and scoring controls live in `hostai.toml` under `[market]`
and `[vast]`:

- `disk_gb` — default container disk in GB; can also be set per-profile with `disk_gb`.
  The default `35` is sized for the Q4_K_P main model (~16.7 GiB = 17.92 decimal GB),
  the FastMTP-32K draft (~0.84 GiB = 0.90 decimal GB), ~5 GB of image/runtime
  overhead, and a safety margin.  The `disk_space>=N` host-eligibility constraint
  is derived from this value automatically, so do not hard-code `disk_space` in
  profile queries.
- `max_inet_down_cost` / `max_inet_up_cost` — reject paid traffic beyond these
  USD/GB limits; set to `0.0` to require free traffic.  These limits are also
  used in `session` scoring to include transfer cost in startup/session cost.
- `scoring_mode` — `dph` (default), `perf`, or `session`
- `scoring_prompt_weight` / `scoring_decode_weight` — balance prompt vs decode
  TPS when scoring in `perf` or `session` mode
- `allow_unverified` — include unverified hosts in the search

The normal GPU queries are in `profiles.json`, not in shell code. For a one-off
query use `GPU_QUERY_OVERRIDE`.

Example Europe-only override:

```dotenv
GPU_QUERY_OVERRIDE='num_gpus=1 gpu_ram>=48 cpu_ram>=32 reliability>0.98 inet_down>=800 disk_bw>=300 cuda_vers>=12.8 rented=False rentable=True direct_port_count>=1 geolocation in [DE,NL,FR,SE,FI]'
```

After a successful `hostai up`, `run-*/disk-telemetry.json` records per-stage
`df` and `du` data so you can verify the real container disk requirement and
adjust `disk_gb` if needed.

## Runtime failure handling

The slim runtime explicitly installs `libgomp1`, which provides `libgomp.so.1`
required by the compiled llama-server. The image build validates that dependency,
`start.sh` validates the binary before downloading model files, and `hostai up`
performs a remote preflight as soon as SSH is available.

PID 1 inside the container supervises `sshd -D` and `start.sh` as separate child
processes. If `start.sh` or `llama-server` exits, the container stays alive,
writes the exit code to `/run/qwen38/start.exitcode`, and leaves SSH running. If
`sshd` itself exits unexpectedly, PID 1 restarts only sshd after
`SSH_RESTART_DELAY_SECONDS` (default `2`) rather than restarting the whole
container. A Vast `TERM`/`INT` is forwarded cleanly to both children.

`hostai up` notices the model exit marker, prints the server log, and performs
normal failure cleanup. Set `KEEP_ON_FAILURE=1` when intentionally debugging a
failed instance and you want `hostai up` to leave it running and SSH-accessible.
SSH daemon diagnostics are available in `/var/log/qwen38/sshd.log`.

## Updating llama.cpp / FastMTP

Do not change the pinned llama.cpp commit while blindly keeping the same FastMTP
patch. Update both together from the model author's instructions, then rebuild.

Embedded-MTP fallback:

```dotenv
USE_FASTMTP=0
```

## Idle and maximum-runtime safeguards

`hostai up` can start a background watchdog when `watchdog_auto_start = true` is
set in the `[vast]` section.  The watchdog polls `llama-server` metrics and the
`/slots` endpoint to detect active requests.

| option | meaning |
|--------|---------|
| `idle_timeout_seconds` | destroy if no request has been active for this long |
| `max_runtime_seconds` | destroy once this lifetime is reached and the current request finishes |
| `idle_poll_interval_seconds` | seconds between watchdog checks |

Both safeguards always wait for the current request to finish before invoking
`hostai down`, so they reuse the normal cache-save/telemetry-archive path.
Manual `hostai down` also stops the watchdog automatically.

## Interruptible / bid instances

`hostai up` accepts `--interruptible` and `--bid <dph>` to rent Vast.ai
interruptible (bid) instances.  You can also set `interruptible = true` and a
`bid_price` in `[vast]`.  The bid price is used both as the search `price <=`
constraint and as the actual bid submitted to Vast.

```bash
uv run hostai up a6000-64k --interruptible --bid 0.35
```

## Cost-efficiency scoring

The market layer supports three scoring modes configured with
`market.scoring_mode` or `--scoring-mode`:

- `dph`: hourly price (legacy default)
- `perf`: cost per token from historical prompt/decode tokens-per-second
- `session`: estimated total cost for the expected session, including startup
  time and transfer cost

Historical data comes from `.hostai-runs/*/benchmarks/*/metrics.json` and is
aggregated per GPU.  You can tune the prompt/decode weights and the minimum
number of historical samples in `[market]`.

## Persistent model volume break-even

`hostai cost volume-break-even` estimates whether a persistent model volume is
cheaper than re-downloading the model weights on every start.  It only credits
savings for the main model and any draft/MTP files; Docker image pulls,
container startup, and unrelated init are not avoided.  It uses the current (or
configured) `dph`, bandwidth, and the number of recent starts from `.hostai-runs`.

```bash
uv run hostai cost volume-break-even --volume-gb 100 --volume-cost-month 5.00
```

## Shutdown observability

`hostai down` and the watchdog now write `shutdown-tail.json` into the run
directory, including:

- shutdown tail duration and estimated cost
- slot cache save/upload duration and bytes
- telemetry archive duration
- the shutdown reason (`manual`, `idle-timeout`, `max-runtime`)

The rsync cache upload also attempts to delta-seed the remote
`.current.bin.part` from an existing `current.bin` before sending changed blocks.

## Upstream

- Model: https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Vast Docker/launch modes: https://docs.vast.ai/guides/instances/docker-environment
- Vast networking/ports: https://docs.vast.ai/guides/instances/connect/networking
- Hugging Face downloads: https://huggingface.co/docs/huggingface_hub/guides/download
