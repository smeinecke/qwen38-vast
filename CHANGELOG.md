# Changelog

## v2 - Vast cold-start fixes

- Runtime/build stages now derive from `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py312`.
  This is Vast's recommended extension path and should avoid the multi-minute SSH compatibility package install seen with the previous generic CUDA runtime image on slow hosts.
- `qwen-up` forces `LC_NUMERIC=C` so Vast decimal prices work on German/localized shells.
- Fixed Vast query units: `cpu_ram` is GB, so the default is now `cpu_ram>=32` rather than `>=64000`.
- Added `disk_bw>=200` to avoid extremely slow local disks during bootstrap/model load.
- Added `RTX_5880Ada` and explicit `rentable=True` to the default offer search.
- Rental output now shows the selected host's disk bandwidth.
- `HF_TOKEN` is optional because the current HauhauCS repository is publicly downloadable. If supplied, it is still forwarded only for the download phase and removed before llama-server starts.
