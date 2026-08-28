# Changelog

## v7 - self-managed SSH, external profiles and manual 56-core builds

- Switched Vast creation from injected SSH launch mode to normal args/entrypoint mode with a regular `-p 22:22` mapping. The image now starts its own `sshd`, so Vast no longer builds a runtime `.../ssh` child image or mutates `authorized_keys`.
- Added `qwen-init-ssh.sh` with deterministic `0700`/`0600` ownership/modes and fresh per-instance host keys.
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

# Changelog

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
