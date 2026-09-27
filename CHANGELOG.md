# Changelog

## 2026-09-27

- `hostai log` renders reasoning/thinking in italic at normal brightness instead of `bright_black`, which was too dark to read.
- Fixed tokenized-only sessions recording `local_port: 0` and status/down reporting an invalid endpoint.
  - `ensure_unix_tunnel` no longer zeroes `state.local_port`; the field is the client-facing port owned by the proxy (`_start_proxy` already saves it).
  - `is_tunnel_healthy` now checks the recorded `upstream_socket` Unix socket when present instead of a TCP port.
  - `hostai down` skips the raw TCP `ensure_tunnel` for unix-socket upstreams (it would forward to the TLS socket, unusable by the plain-HTTP proxy client) and stops the proxy only after the slot-cache save and telemetry archive, which go through the proxy.
  - `status` renders the proxy unix socket / "-" instead of `127.0.0.1:0` when no TCP port exists.
- `hostai status` now shows a `Perf (avg)` row with decode/prompt tok/s (and MTP draft-accept rate) derived from llama.cpp `/metrics` counters.
- Added `hostai down --skip-llama` to skip the remote `llama-server` shutdown while still destroying/pausing the instance.
- Fixed `hostai up` aborting GB10 rentals at the VRAM preflight.
  - GB10 is unified memory (UMA): `nvidia-smi` reports `memory.total` as `[N/A]` / "Not Supported", so the parse found no VRAM and provisioning was destroyed.
  - When every detected GPU is a UMA part (GB10), the preflight now verifies total system RAM from `/proc/meminfo` against `min_gpu_vram_mb` instead.
  - Unparseable output on non-UMA GPUs still fails closed, and the failure log now includes a snippet of the raw `nvidia-smi` output.

## 2026-09-10

- Added NVIDIA GB10 / Grace Blackwell support via a dedicated `:gb10` image.
  - ARM64 / `linux/arm64`, CUDA 13.x, sm121, separate from x86_64 SM120 `blackwell`.
  - Added `gb10-128k` and `gb10-256k` profiles with `min_gpu_vram_mb=115000`.
  - Vast query uses exact `gpu_name=GB10` and `cpu_arch=arm64` to avoid RTX PRO 6000.
  - CPU architecture (`uname -m`) is now captured at startup for runtime metadata.
- Architecture-aware GitHub workflows: x86_64 builds stay on `ubuntu-24.04`, GB10 builds on `ubuntu-24.04-arm`.
- Added `docker-gb10.yml` for manual GB10-only ARM64 builds.

## 2026-09-10

- Added the `ga100` compiled image for GA100 / SM80 (NVIDIA A100 SXM4 and unlocked CMP 170HX).
- Added `a100-128k` runtime profile (A100 SXM4 40 GB, 131,072 context).
- Added `cmp170hx-256k` runtime profile (unlocked CMP 170HX 64 GB, 262,144 context).
  - Vast query targets `CMP_170HX` with `gpu_ram>=60` so only the unlocked 64 GB variant is selected.
  - Post-SSH `min_gpu_vram_mb` check rejects any card exposing less than 60,000 MiB.
- Added optional `min_gpu_vram_mb` profile field and generic post-SSH VRAM preflight.
- Updated README and `.env.example` profile list.
