# Agent Notes

## Local Development

- Dependencies are managed with `uv` and locked in `uv.lock`.
- Use `uv run pytest` to run the test suite.
- Use `uv run ruff check src/hostai` for linting and `uv run ruff check --fix src/hostai` for auto-fixes.
- Use `uv run pyright src/hostai` for type checking.
- Use `bash -n start.sh` to validate shell script syntax.

## Tokenized-Only Mode

- Set `HOSTAI_TOKENIZED_ONLY=1` in `hostai.toml` (or as an environment variable) and run `hostai up`.
- After `up`, run `hostai proxy` to start the local Unix-socket proxy that tokenizes prompts before forwarding them.
- Clients that cannot speak Unix sockets should set `HOSTAI_PROXY_PORT` to a local TCP port; `OPENAI_BASE_URL` in the generated `env` file will then point at the proxy.
- The remote container image must be rebuilt/pushed when `Dockerfile`, `start.sh`, or `src/hostai/remote_guard.py` change because the guard runs inside the image.

## Lifecycle & Cost

- `hostai up` can auto-start a watchdog when `watchdog_auto_start = true` is set in `[vast]`.
- `hostai down` records shutdown-tail metrics and reuses the cache-save/telemetery path.
- `hostai cost volume-break-even` estimates whether a persistent model volume is cheaper than re-downloading.
