# Quickstart

This is the fastest path from a fresh checkout to a running Qwen3.8 27B
FastMTP inference server on a disposable Vast.ai GPU instance.

The command-line tool is `hostai`. All examples below use `uv run`, which works
whether or not the package is installed globally.

## 1. Local dependencies

You need a Vast.ai account and API key:

```bash
python3 -m pip install --upgrade vastai
vastai set api-key YOUR_VAST_API_KEY
```

or set `VAST_API_KEY` in `.env`.

You also need the local tools the scripts call directly:

```bash
# Debian/Ubuntu
sudo apt-get install -y jq curl openssh-client openssl rsync

# macOS (Homebrew)
brew install jq curl openssh openssl rsync
```

A Vast account SSH key is **not** required for this deployment path; the
container has its own `sshd`.

## 2. Configure

```bash
cp .env.example .env
chmod 600 .env
```

At minimum edit:

```dotenv
GHCR_IMAGE_BASE=ghcr.io/YOUR_USER/YOUR_REPO
VAST_API_KEY=your-vast-api-key
HOSTAI_SLOT_CACHE_USER=qwen-cache   # change if your cache user differs
```

Other common settings in `.env`:

```dotenv
HOSTAI_PROFILE=a6000
MAX_DPH=0.80
HOSTAI_SLOT_CACHE_HOST=your-cache-server.example.com
HOSTAI_SLOT_CACHE_ENABLED=1
HOSTAI_SLOT_CACHE_MAX_GB=80
```

The first `hostai` command you run will create `hostai.toml` from `.env` if
`hostai.toml` does not already exist. After that, **non-secret** values in
`.env` are ignored in favor of `hostai.toml`. Either edit `hostai.toml` for
configuration, or delete it and re-run `cp .env.example .env` to re-import.

## 3. Choose a slot-cache backend

The slot cache persists the llama.cpp KV state between disposable Vast
instances. It is optional, but strongly recommended because it makes profiles
with the same model, context size and FastMTP settings reuse a previous
`current.bin`.

### Option A: SSH/rsync cache

One-time setup, using the same SSH identity you already have to the cache host:

```bash
uv run hostai cache setup
```

If the cache account has a different user or host:

```bash
uv run hostai cache setup youruser@your-cache-host.example.com
```

This creates a dedicated Ed25519 key in `.hostai-cache/cache_key` and installs
the public key on the cache server. The key is never baked into the image.

### Option B: rclone cache

Use any rclone-supported backend (WebDAV, S3, SFTP, etc.). No SSH key is needed.

```dotenv
HOSTAI_SLOT_CACHE_RCLONE=1
HOSTAI_SLOT_CACHE_RCLONE_TYPE=webdav
HOSTAI_SLOT_CACHE_RCLONE_URL=https://your-cache-server.example.com/
HOSTAI_SLOT_CACHE_RCLONE_PASSWORD=your-webdav-password
```

Then validate the setup:

```bash
uv run hostai cache setup
```

`HOSTAI_SLOT_CACHE_HOST`, `PORT` and `KEY` are ignored when rclone is enabled.
`HOSTAI_SLOT_CACHE_USER` still sets the default backend user unless you override
it with `HOSTAI_SLOT_CACHE_RCLONE_USER`. Use `HOSTAI_SLOT_CACHE_RCLONE_REMOTE` to
use a preconfigured rclone remote in the image.

## 4. Start an instance

```bash
# Use the default profile from HOSTAI_PROFILE or profiles.json
uv run hostai up

# Or pass a profile name
uv run hostai up a6000
uv run hostai up a6000-128k
uv run hostai up ada-128k
uv run hostai up 5090-128k
uv run hostai up blackwell-128k
```

`hostai up` does the following:

1. resolves the profile from `profiles.json`;
2. selects the architecture image/tag;
3. searches Vast for the cheapest matching offer below `MAX_DPH`;
4. creates the instance with container port 22 published to a random public port;
5. waits for the public SSH mapping and connects to the image's `sshd`;
6. installs the dedicated cache key (SSH/rsync backend only);
7. prefetches a compatible slot snapshot while the model loads;
8. creates a local `localhost:<port> -> instance:127.0.0.1:8080` tunnel;
9. waits for `/health` and restores slot 0 on a cache hit;
10. stores local state and telemetry in `.hostai-vast/` and `.hostai-runs/`.

If `HOSTAI_SLOT_CACHE_USE_SHM=1`, the slot snapshot is staged under `/dev/shm`
instead of on Vast disk.

One-off context size override:

```bash
CTX_SIZE_OVERRIDE=98304 uv run hostai up a6000
```

Independent contexts can be separated with `--session`:

```bash
uv run hostai up a6000 --session my-project
```

## 5. Talk to the model

`hostai up` prints the local API URL and writes client environment variables to
`.hostai-vast/env`:

```bash
source .hostai-vast/env
echo "$OPENAI_BASE_URL"
```

List models:

```bash
curl "$OPENAI_BASE_URL/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -k
```

The inference API is only bound to `127.0.0.1` inside the container. Only SSH
port 22 is public; the local `ssh -L` tunnel is the access path.

## 6. Logs and status

```bash
# Rich status table + current cost
uv run hostai status

# Tail the remote llama-server log
uv run hostai status --logs

# Follow it live
uv run hostai status --logs --follow
```

`hostai status` also re-creates the local API tunnel if it disappeared, and it
surfaces any monitor alerts.

## 7. Benchmarks

Built-in coding benchmark:

```bash
uv run hostai bench
```

Custom prompt or context:

```bash
uv run hostai bench \
  --label repo-analysis \
  --prompt-file /tmp/repo-context.txt \
  --max-tokens 1024
```

Compare runs:

```bash
uv run hostai results
uv run hostai results --csv > hostai-benchmarks.csv
uv run hostai results --json > hostai-benchmarks.json
```

Every run stores artifacts under `.hostai-runs/`:

```text
.hostai-runs/
└── 20260827T101530Z-a6000-12345/
    ├── metadata.json
    ├── client.log
    ├── gpu-start.txt
    ├── metrics-ready.prom
    └── benchmarks/
```

## 8. Live price monitor

Check once for a cheaper equivalent offer:

```bash
uv run hostai monitor once
```

Watch continuously:

```bash
uv run hostai monitor watch --threshold 10 --interval 180
```

Run a background daemon:

```bash
uv run hostai monitor start
uv run hostai monitor status
uv run hostai monitor logs
uv run hostai monitor stop
```

`hostai down` automatically stops the monitor before saving the slot and
destroying the instance. Set `HOSTAI_MONITOR_AUTO_START=1` to start the monitor
automatically after a successful `hostai up`.

## 9. Stop billing

```bash
# Save slot 0, upload it to the cache, then destroy the instance
uv run hostai down --yes
```

Emergency opt-out to avoid a cache upload:

```bash
uv run hostai down --yes --no-cache
```

If startup fails, the paid instance is destroyed automatically unless you set:

```dotenv
KEEP_ON_FAILURE=1
```

## 10. Switch to a better GPU

```bash
uv run hostai down --yes
uv run hostai up 5090-128k --session my-project
```

The `--session` value is persisted with the run, so `hostai down` uploads back
to the same cache namespace.

## Common `.env` overrides

```dotenv
MAX_DPH=0.55
HOSTAI_MAX_INET_DOWN_COST=0.001
HOSTAI_MAX_INET_UP_COST=0.001
HOSTAI_SLOT_CACHE_REQUIRE_SAVE=1
HOSTAI_SLOT_CACHE_USE_SHM=1
HOSTAI_MONITOR_AUTO_START=1
```

Traffic costs are controlled purely by `HOSTAI_MAX_INET_DOWN_COST` and
`HOSTAI_MAX_INET_UP_COST`. Set both to `0` to require free Vast traffic.

## Local directories

```text
.
├── .env                  # secrets + initial config (keep 0600)
├── hostai.toml           # migrated from .env on first run
├── .hostai-vast/         # state, known_hosts, env, monitor-alert.json
├── .hostai-runs/         # per-run telemetry + benchmarks
└── .hostai-cache/        # cache key and local TLS material
```

See `README.md` for architecture notes, Docker build instructions, and
updating llama.cpp / FastMTP.
