# Changelog

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
