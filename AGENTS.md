# Agent Notes

## Local Development

- Dependencies are managed with `uv` and locked in `uv.lock`.
- Use `uv run pytest` to run the test suite.
- Use `uv run ruff check src/hostai` for linting and `uv run ruff check --fix src/hostai` for auto-fixes.
- Use `uv run pyright src/hostai` for type checking.
- Use `bash -n start.sh` to validate shell script syntax.
- Some tests download the Qwen/Qwen3.8-27B tokenizer and are marked `slow`; use `uv run pytest -m "not slow"` to skip them.

## Tokenized-Only Mode

- Set `HOSTAI_TOKENIZED_ONLY=1` in `hostai.toml` (or as an environment variable) and run `hostai up`.
- After `up`, run `hostai proxy` to start the local Unix-socket proxy that tokenizes prompts before forwarding them.
- Clients that cannot speak Unix sockets should set `HOSTAI_PROXY_PORT` to a local TCP port; `OPENAI_BASE_URL` in the generated `env` file will then point at the proxy.
- The proxy tokenizer is pinned to a known-good Qwen3.8-27B commit (`tokenizer_revision` / `HOSTAI_PROXY_TOKENIZER_REVISION`).  Changing it should be followed by regenerating `tests/fixtures/tokenizer_golden.json` and running the tokenizer golden tests.
- The remote container image must be rebuilt/pushed when `Dockerfile`, `start.sh`, or `src/hostai/remote_guard.py` change because the guard runs inside the image.

## Lifecycle & Cost

- `hostai up` can auto-start a watchdog when `watchdog_auto_start = true` is set in `[vast]`.
- `hostai down` records shutdown-tail metrics and reuses the cache-save/telemetery path.
- `hostai cost volume-break-even` estimates whether a persistent model volume is cheaper than re-downloading.

## Container disk and model storage

- The default `market.disk_gb` is 35, sized for the Q4_K_P main model (~16.7 GiB = 17.92 decimal GB) + FastMTP-32K draft (~0.84 GiB = 0.90 decimal GB) + ~5 GB image/runtime overhead + safety margin.  You can override per-profile with `disk_gb` in `profiles.json`.
- Do not put `disk_space>=N` constraints in `profiles.json` queries. `market.build_search_query` derives `disk_space>=N` from `resolved_disk_gb(profile, config)` so searches, lookups, monitor checks, and cost/startup estimates stay consistent.
- `start.sh` now logs per-stage disk usage (`after-preflight`, `after-main-model`, `after-draft-model`, `before-serve`) to `/dev/shm/qwen38/log/disk-usage.log`. `hostai up` copies this plus a final snapshot to `run-*/disk-telemetry.json` after a successful cold start.
- The container image must be rebuilt after changes to `start.sh`.

## Local provider and integration tests

- Set `HOSTAI_PROVIDER=local` to run `hostai up`/`down` against a local Docker container instead of Vast.
- `HOSTAI_LOCAL_IMAGE` overrides the Docker image used by `LocalProvider` (default: `ghcr.io/smeinecke/qwen38-vast:<tag>`).
- `HOSTAI_LOCAL_SHM_SIZE_GB` controls the container `--shm-size` (default: 32 GB).
- Build the integration test image with:
    docker build -f tests/integration/Dockerfile.test -t hostai-test:latest .
- Run the integration acceptance test with `uv run pytest -q -m "not slow" tests/test_local_integration.py`.
- The integration image uses the real `start.sh` and `entrypoint.sh` but swaps expensive binaries (`hf`, `llama-server`, `nvidia-smi`) for fixtures in `tests/integration`.
- Deterministic fault injection is available through these environment variables (forwarded by `up.py` to the container):
  - `HOSTAI_FAULT_SSHD_DELAY_SECONDS=<int>`: delay sshd startup inside the container
  - `HOSTAI_FAULT_SOCKET_DELAY_SECONDS=<int>`: delay `llama-server` socket bind
  - `HOSTAI_FAULT_HEALTH_DELAY_SECONDS=<int>`: return 503 from `/health` for N seconds
  - `HOSTAI_FAULT_llama_EXIT_AFTER_SECONDS=<int>`: exit the fake server after N seconds
  - `HOSTAI_FAULT_METRICS_UNAVAILABLE=1`, `HOSTAI_FAULT_SLOTS_UNAVAILABLE=1`
- The `VastProvider` refuses construction unless `config.provider.backend == "vast"`, preventing accidental production Vast API calls when `HOSTAI_PROVIDER=local` or tests are running.

## Validation gate

- Run `hostai validate` to check the repository layout and record a `.hostai-vast/validation.json` digest.
- Run `hostai validate --production` to also require Docker, the integration image, and pass `tests/test_local_integration.py`.
- Run `hostai validate --compare` to compare the current state against the last recorded validation and warn about drift (git commit, image digest, `profiles.json`).
- Use this gate before a real Vast rental to confirm the local integration image still boots cleanly and the working tree matches the last validated state.
