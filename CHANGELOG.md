# Changelog

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
