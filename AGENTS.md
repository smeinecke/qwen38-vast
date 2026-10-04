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
- The proxy sends `return_tokens`/`token_only` so the remote returns generated token IDs only; the proxy detokenizes locally and applies `stop` strings client-side (truncating + warning, since server-side stop matching is text-based and cannot run without detokenization).
- Requests without `max_tokens`/`max_completion_tokens` get `n_predict` from `[proxy].default_max_tokens` (default `-1` = request-defined, run to EOS/context end). The same value is forwarded to the remote as `N_PREDICT`/`--n-predict` so passthrough mode and direct `/completion` calls behave consistently.
- When the request declares `tools`, the proxy also parses the model's `<tool_call><function=name><parameter=key>` markup (the Hermes-style format the chat template emits, with a JSON-in-tags fallback) into OpenAI `tool_calls` — in both streaming (`delta.tool_calls`, `finish_reason="tool_calls"`) and non-streaming responses. Parameter values are coerced using the declared tool schema types.
- The proxy logs operational metadata to `.hostai-cache/proxy.log` — never prompt or output content.
- Opt-in content logging is available via `[proxy] log_content = true` (or `HOSTAI_PROXY_LOG_CONTENT=1`). It writes one JSON record per line (request, delta, response, done, error events) to `.hostai-cache/proxy-content.jsonl`, flushed per line so `tail -f` shows prompts and streaming responses live. It is off by default; `proxy.log` stays content-free either way.
- `hostai log` renders that JSONL as a live chat-style transcript (follows by default; `-n` for backlog size, `--no-follow` to print once, `--ops` to tail the raw operational `proxy.log`).
- The remote container image must be rebuilt/pushed when `Dockerfile`, `start.sh`, `patches/`, or `src/hostai/remote_guard.py` change because the guard runs inside the image.

## Lifecycle & Cost

- `hostai info` shows provider ground truth: account balance/credit (via `get_account_info`) and every instance on the account (via `list_instances`), marking which ids are tracked by local state and summing the running $/h burn.
- `hostai down --id <provider-id>` (repeatable) destroys/pauses by raw provider id: ids matching tracked instances run the full shutdown path; untracked ids go through `down_remote_instance`, which skips cache save, telemetry archive, and daemon cleanup entirely. `--id` conflicts with `--all`/`--name`.
- `hostai up` can auto-start a watchdog when `watchdog_auto_start = true` is set in `[vast]`.
- `hostai down` records shutdown-tail metrics and reuses the cache-save/telemetery path.
- `hostai cost volume-break-even` estimates whether a persistent model volume is cheaper than re-downloading.

## Container disk and model storage

- The default `market.disk_gb` is 35, sized for the Q4_K_P main model (~16.7 GiB = 17.92 decimal GB) + FastMTP-32K draft (~0.84 GiB = 0.90 decimal GB) + ~5 GB image/runtime overhead + safety margin.  You can override per-profile with `disk_gb` in `profiles.json`.
- Do not put `disk_space>=N` constraints in `profiles.json` queries. `market.build_search_query` derives `disk_space>=N` from `resolved_disk_gb(profile, config)` so searches, lookups, monitor checks, and cost/startup estimates stay consistent.
- `start.sh` now logs per-stage disk usage (`after-preflight`, `after-main-model`, `after-draft-model`, `before-serve`) to `/dev/shm/qwen38/log/disk-usage.log`. `hostai up` copies this plus a final snapshot to `run-*/disk-telemetry.json` after a successful cold start.
- The container image must be rebuilt after changes to `start.sh`.

## Multiple instances

- `hostai up --name <n>` provisions a parallel deployment under `.hostai-vast/instances/<n>/`; the reserved name `default` maps to the legacy `.hostai-vast/` layout (`state.json`, `env`, `known_hosts`, `proxy.sock`, `proxy.pid`, `.lifecycle.lock`, …).
- All instance-bound commands accept `-n/--name` (env `HOSTAI_INSTANCE`); a numeric provider instance id also resolves via `state.resolve_instance_selector`. `status` shows a fleet table when several are tracked; `down --all` stops all.
- Locks: `.lifecycle.lock` per instance dir (serializes `up`/`restart` for that name only) plus `.hostai-vast/.allocate.lock` (short global lock covering the live-instance check, port claim, provider create and initial state write).
- Ports: `claimed_local_ports`/`_sibling_claimed_ports` treat ports recorded in sibling `state.json` files as taken even before they are bound.
- Daemons: monitor/watchdog pid+log files get a `-<name>` suffix in `.hostai-cache/` (`monitor-foo.pid`, `watchdog-foo.log`); proxy uses `proxy-<name>.log`/`proxy-content-<name>.jsonl`. Daemon argv always ends in `--name <instance>` so identity-checked signaling never stops a sibling's daemon.
- `LocalProvider` serializes its shared `.hostai-vast/local-provider.json` registry under `.local-provider.lock` (`_state_mutation`) and re-reads it on each access.
- `LocalProvider.list_instances` merges the registry with `docker ps` results for the `hostai.provider=local` label so orphaned containers still show up in `hostai info` (marked untracked); `get_instance`/`start`/`stop`/`destroy` fall back to the `hostai.instance_id=<id>` docker label when the registry entry is missing.
- TLS: `.hostai-cache/tls` is shared; regeneration is serialized under the allocation lock.

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
- Run `hostai validate --production` to require a clean Git tree, build the integration image from the current source, and pass `tests/test_local_integration.py`.
- Run `hostai validate --compare` to compare the current state against the last *successful* validation and warn about drift (git commit, image ID, `profiles.json`, validation level).
- On success, a production validation is also written to `.hostai-vast/validation-last-success.json`.
- Set `HOSTAI_REQUIRE_PRODUCTION_VALIDATION=1` or `[vast].require_production_validation = true` to make `hostai up` and `hostai up --restart` refuse any paid Vast `create_instance`/`start_instance` until the current git/image/profiles match the last successful production validation. Use `--allow-unvalidated` to bypass the gate explicitly.
- When the gate is enabled, `hostai up` selects the immutable production image tag `<profile>-sha-<validated-commit>` instead of the mutable profile tag, creating a complete provenance chain from Git commit -> integration test -> CI-built runtime image -> Vast rental.
- Use this gate before a real Vast rental to confirm the local integration image still boots cleanly and the working tree matches the last validated state.
