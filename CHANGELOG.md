# Changelog

## v9.6.1 — supervised SSH daemon

- Run `sshd` with `-D` as an explicit child of the container PID 1 supervisor instead of daemonizing it away from supervision.
- Restart `sshd` in-place if it exits unexpectedly; a transient SSH-daemon failure no longer requires or triggers a whole-container restart.
- Keep `start.sh`/`llama-server` one-shot and independent: inference failure records `/run/qwen38/start.exitcode` while SSH continues serving.
- Forward container `TERM`/`INT` cleanly to both inference and SSH children.
- Add `SSH_RESTART_DELAY_SECONDS` (default `2`) for bounded sshd restart backoff and extend the repo regression validator to require the supervised-SSH design.

## v9.6 — runtime dependency + SSH survivability

- Add `libgomp1` to the slim CUDA runtime image; `llama-server` requires `libgomp.so.1`.
- Validate the final binary linkage during Docker build so a broken runtime image cannot be published silently.
- Validate `llama-server --version` before any GGUF download and again remotely from `qwen-up` immediately after SSH is ready.
- Decouple SSH lifetime from `start.sh`: if the model process exits, the container remains alive and SSH stays reachable for diagnostics instead of entering a Vast restart loop.
- Persist `/run/qwen38/start.exitcode`; `qwen-up` detects it and prints the remote server log immediately instead of spending the full startup timeout reconnecting tunnels.
- Extend `scripts/validate-repo` to guard all of the above against future branch reconciliation regressions.

## v9.5 - reconciled cumulative fixes

- Restored the non-interactive, timeout-bounded `qwen_destroy_instance` path in `qwen-up` failure cleanup; later tunnel work had accidentally reintroduced a direct interactive `vastai destroy` call.
- Added the free-traffic Vast market policy on top of the user working tree: both rental and monitor searches reject non-zero upload/download traffic cost by default.
- Added `scripts/validate-repo` and wired it into both Docker workflows to catch loss of cumulative features (slot cache, 256k profiles, hardware ranking, traffic policy, tunnel helper, destroy helper and compiler cache).
- Release ZIP is sanitized: local `.env`, `.qwen-cache/`, `.qwen-runs/` and `.qwen-vast/` state are not included.

## v9.4 - free-traffic offer policy

- Rejects Vast offers with non-zero `inet_down_cost` or `inet_up_cost` by default.
- Applies the same traffic policy to `qwen-up` and `qwen-monitor`.
- Centralizes traffic limits in `profiles.json -> market_policy`; no duplicated profile query edits.
- Validates raw search results as a second guard and prints selected traffic prices before rental.
- Persists selected upload/download traffic prices in local instance state.

## v9.3.1 - race-free, self-healing SSH tunnels

- Centralize tunnel creation in `.qwen-lib.sh`; `qwen-up` no longer launches a second independent `ssh -L`.
- Serialize tunnel creation across `qwen-up`, `qwen-status`, `qwen-bench`, and `qwen-down` with an atomic local lock.
- Reuse a tunnel created concurrently by another qwen command instead of racing for port 18080.
- Add `LOCAL_PORT_AUTO=1` (default): if the preferred localhost port is occupied by an unrelated process, automatically select the next free port and persist it in state.
- Treat a tunnel disconnect during model loading as recoverable: refresh Vast SSH endpoint and reconnect until `START_TIMEOUT` rather than immediately destroying the instance.

## v9.3 - 256k profiles and hardware-safe market monitoring

- Added explicit 262,144-token (`256k`) runtime profiles for RTX A6000/A40, 48 GB Ada-class GPUs, and a conservative RTX PRO 6000 96 GB Blackwell profile; all reuse the existing three CUDA images.
- Added editable `monitor_hardware.gpu_ranks` metadata to `profiles.json`. The numbers are ordering values, not claimed benchmark scores.
- `qwen-monitor` now requires the candidate's concrete GPU to be the same hardware class or higher-ranked than the running GPU, so e.g. an A6000 is never replaced by a cheaper A40.
- Market comparisons require the exact current context size. Unknown GPU models are excluded conservatively.
- Custom `CTX_SIZE_OVERRIDE` values with no named profile fall back to the current profile, and the generated switch command preserves the exact current context.
- Monitor output now shows both current/candidate hardware ranks for auditability.

## v9.2 — non-interactive Vast destroy fix

- Fixed `qwen-down` hanging indefinitely at `Destroying Vast instance ...`: Vast CLI requires `-y` to skip its own irreversible-action confirmation. Because qwen-down captured CLI output, that second prompt was invisible.
- Fixed the same missing `-y` in `qwen-up` failure cleanup.
- Centralized destroy handling in `qwen_destroy_instance`.
- Added `HOSTAI_DESTROY_TIMEOUT_SECONDS` (default 45s) so a genuine Vast CLI/API stall cannot block forever.
- HTTP/API-style 404 `not found` remains a successful shutdown outcome.
- On timeout or real error, local state is retained for a safe retry instead of pretending the rental was destroyed.


## v9.1 - compiler-cache seeding fix

- Fixed the persistent compiler cache remaining an empty ~300-byte archive when BuildKit restored the entire llama.cpp compile RUN from its separate GHA layer cache.
- Added `CCACHE_SEED`, tied to the Dockerfile compiler-cache key, so a new cache generation executes the compile layer once and actually populates the BuildKit ccache mount.
- Bumped compiler-cache schema to v4 while retaining v3 restore fallbacks.
- Enabled `save-always` for both the GitHub cache and buildkit-cache-dance so useful partial CUDA compilation survives failed or timed-out builds.
- Added pre-build cache diagnostics (exact hit, disk size, file count) to both hosted and self-hosted workflows.
- The manual self-hosted workflow now keeps a fixed `qwen38-local-builder` BuildKit instance/state on the persistent 56-core runner, so local layer and cache-mount state survives workflow runs even without a GHA cache restore.
- The existing BuildKit `type=gha` layer cache remains enabled; it is complementary to ccache rather than a replacement.

## v9 - Vast market profiles and live price monitor

- Added value-oriented Vast profiles without adding new CUDA builds: A40 reuses SM86, RTX 4090/L40/L40S reuse SM89, and RTX 5090 reuses SM120.
- Added exact/value profiles for 32k, 64k and 128k operation plus `monitor_group` metadata so only context-equivalent profiles are compared.
- Added `qwen-monitor` with one-shot, foreground watch and background-daemon modes. It compares the running instance's live `dph_total` with current rentable offers and defaults to alerting at >=10% savings.
- Market comparisons use the same Vast storage size and preserve the normal profile reliability/network/disk constraints. Alerts include profile, GPU, price, saving, location, network/disk speed, reliability and offer ID.
- `qwen-status` shows monitor state and the latest qualifying alert; `qwen-down` stops a background monitor before cache save/destroy. Optional `HOSTAI_MONITOR_AUTO_START=1` starts it after a successful `qwen-up`.
- `qwen-up` now persists `disk_gb`, resolved `gpu_query` and `monitor_group` in run state/telemetry so market comparisons remain auditable.

## v8 - automatic cross-instance slot/KV persistence

- Added `.qwen-cache-lib.sh` and `qwen-cache-setup` for a dedicated restricted SSH key and an external rsync-backed llama.cpp slot cache. Setup now verifies remote `rsync` before any GPU is rented.
- Cache host is configured in `.env` via `HOSTAI_SLOT_CACHE_HOST`; cache user/root/session remain editable in `.env`. `qwen-up --session NAME` selects an independent persistent context without manual transfers.
- `qwen-up` automatically prefetches a compatibility-scoped snapshot from the external server in parallel with model loading and restores slot 0 after llama-server becomes healthy.
- `qwen-down` automatically saves slot 0, uploads `current.bin`/metadata atomically, prunes old snapshots above the configured size budget, and only then destroys the Vast instance.
- Added `HOSTAI_SLOT_CACHE_REQUIRE_SAVE` and `--no-cache` shutdown policies.
- Cache signatures include llama.cpp commit, model/revision, context size, FastMTP and KV precision to avoid restoring incompatible state.
- `start.sh` enables `--slots` and `--slot-save-path`; runtime build metadata now records the llama.cpp commit.
- `qwen-bench` records `cache_n`, evaluated prompt tokens and cache-hit percentage; `qwen-results` includes a `cache` column so hybrid-model restore effectiveness can be verified rather than inferred from the restore API response.
- Increased the example Vast scratch disk to 100 GB because slot snapshots are materialized locally before upload.

## v7 - self-managed SSH, external profiles and manual 56-core builds

- Switched Vast creation from injected SSH launch mode to normal args/entrypoint mode with a regular `-p 22:22` mapping. The image now starts its own `sshd`, so Vast no longer builds a runtime `.../ssh` child image or mutates `authorized_keys`.
- Added `hostai-init-ssh.sh` with deterministic `0700`/`0600` ownership/modes and fresh per-instance host keys.
- CI bakes the repository owner's GitHub public SSH keys into the image by default; repository-variable and committed-key overrides are supported.
- Moved `apt-get upgrade` plus SSH/runtime utility installation into the Docker image build, off paid GPU startup time.
- Changed the final stage from the full Vast CUDA/cuDNN devel base to `nvidia/cuda:12.8.1-runtime-ubuntu24.04`; the builder remains on the devel image. This substantially reduces the image pulled by each disposable host.
- Added `profiles.json`; runtime profiles and image architecture metadata no longer live inside `qwen-up`. Added `a6000-128k`, which reuses the same SM86 image as the 64k A6000 profile.
- Both Docker workflows derive their build matrix from `profiles.json`.
- Added `.github/workflows/docker-self-hosted.yml`, a manual-only workflow for the local 56-core self-hosted runner with selectable A6000/Ada/Blackwell/all targets.
- Updated Docker Actions to Node-24-capable major versions and `buildkit-cache-dance` v3.4.0.
- SSH endpoint discovery now understands normal Vast port mappings (`public_ipaddr` + `ports["22/tcp"].HostPort`) while retaining compatibility with legacy Vast SSH-mode instances.

## v6.1 - idempotent shutdown and locale-safe accounting

- Treat Vast `404 / instance not found` as successful shutdown (`already_absent`).
- Fail only for non-404 destroy errors.
- Force locale-independent numeric handling so German decimal locales do not break cost formatting.
- Recover legacy state files without `run_dir` into `.qwen-runs/<timestamp>-recovered-<instance>/`.
- Record `destroy_outcome` in final metadata.

## v6 - resilient Vast SSH discovery and self-healing tunnels

- Fixed startup hanging at `waiting for SSH` when Vast exposes `ssh_host`/`ssh_port` but `vastai ssh-url` is not in the previously assumed `ssh://...` text format.
- Added `.qwen-lib.sh` with shared SSH endpoint discovery. It prefers `ssh_host`/`ssh_port` from `show instance --raw` and falls back to both URL and `ssh -p PORT user@host` CLI formats.
- `qwen-up` persists the local port immediately, logs provisioning progress every 15 seconds, and reports when the SSH endpoint and daemon become available.
- `qwen-status` refreshes stale connection data and recreates a dead/missing local tunnel automatically.
- `qwen-logs`, `qwen-bench`, and `qwen-down` refresh SSH connection details from Vast instead of requiring an `ssh_url` that was captured during the original startup process.
- Existing paid instances created by v5 can therefore be adopted by the updated helper scripts without rerenting them.

## v5 - persistent telemetry and benchmark comparison

- Added persistent `.qwen-runs/<session>/` directories outside transient Vast state.
- `qwen-up` now records selected offer/profile/image/context, startup timing, GPU snapshot, metrics snapshot and its own console log without persisting the API key.
- Enabled llama.cpp `--metrics` in `start.sh` and log the llama-server version plus GPU hardware at startup.
- Added `qwen-bench` for streamed TTFT measurement plus llama.cpp server timings, prompt/decode t/s, MTP draft acceptance, Prometheus counter snapshots, GPU utilization/VRAM/power sampling and per-request compute cost.
- Added `qwen-results` to compare all saved benchmarks as a terminal table, CSV or JSON.
- `qwen-logs` now saves a local live-log copy by default.
- `qwen-status` shows session/run metadata, estimated running cost, selected llama.cpp counters and a live GPU snapshot.
- `qwen-down` archives the full remote server log, final metrics/GPU/Vast snapshots and writes final duration/estimated compute cost before destroying the instance.

## v4 - parallel architecture builds and profile-driven startup

- Replaced the 90-minute self-hosted build with three parallel GitHub-hosted matrix jobs (`SM86`, `SM89`, `SM120`), each with its own 330-minute limit.
- Each image now compiles exactly one CUDA architecture and uses an architecture-specific BuildKit/ccache scope.
- Added stable GHCR tags: `:a6000`, `:ada-128k`, and `:blackwell-128k`, plus per-profile SHA/release tags.
- Added `qwen-up PROFILE` / `--profile PROFILE` selection with aliases (`ampere`, `ada`, `blackwell`, `5090`, `sm86`, `sm89`, `sm120`).
- Profiles select the matching image tag, Vast GPU query and runtime context together.
- `a6000` and direct-container startup now default to 64k context for better interactive throughput; Ada/Blackwell profiles default to 128k.
- Replaced implicit `CTX_SIZE`/`GPU_QUERY` profile overrides with explicit `CTX_SIZE_OVERRIDE`/`GPU_QUERY_OVERRIDE` to avoid stale `.env` values silently defeating the selected profile.
- Added `GHCR_IMAGE_BASE` so `.env` stores only `ghcr.io/OWNER/REPO`; `qwen-up` appends the stable profile tag automatically.

## v3 - compiler cache and faster CUDA builds

- Added `ccache` for C/C++/CUDA compilation.
- Persisted the BuildKit ccache mount across GitHub Actions runs with `actions/cache` + `reproducible-containers/buildkit-cache-dance`.
- Scoped the existing Docker GHA layer cache to `qwen38-vast`.
- Limited the default CUDA build to SM 86 + SM 89, matching RTX A6000 and Ada/L40S Vast targets. This substantially reduces cold compile work compared with building every CUDA architecture.
- `CUDA_ARCHITECTURES` is a Docker build arg and can be overridden later (for example `86;89;120` for native Blackwell support).

## v2 - Vast cold-start fixes

- Runtime/build stages now derive from `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py312`.
  This is Vast's recommended extension path and should avoid the multi-minute SSH compatibility package install seen with the previous generic CUDA runtime image on slow hosts.
- `qwen-up` forces `LC_NUMERIC=C` so Vast decimal prices work on German/localized shells.
- Fixed Vast query units: `cpu_ram` is GB, so the default is now `cpu_ram>=32` rather than `>=64000`.
- Added `disk_bw>=200` to avoid extremely slow local disks during bootstrap/model load.
- Added `RTX_5880Ada` and explicit `rentable=True` to the default offer search.
- Rental output now shows the selected host's disk bandwidth.
- `HF_TOKEN` is optional because the current HauhauCS repository is publicly downloadable. If supplied, it is still forwarded only for the download phase and removed before llama-server starts.
