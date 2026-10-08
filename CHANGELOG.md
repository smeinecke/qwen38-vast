# Changelog

## 2026-10-07

- `hostai info` and `hostai status` now show the Vast `machine` id: a `Machine` column in `info` and the fleet overview, and a `Machine` row in single-instance `status` — provider payload first, falling back to `state.data["machine_id"]` recorded by `up`.
- `hostai up --machine <id>` pins provisioning to a specific Vast machine: `machine_id=N` is added to the server-side search query and enforced client-side in `filter_eligible_offers` (covers the local/fake providers). The `max_dph` cap still applies — the cheapest offer on that machine within budget wins. Conflicts with `--skip-machine`/blocklist entries are rejected, and `--restart` warns that `--machine`/`--offer` are ignored.
- `hostai replace` hardening from the local end-to-end run: `_resolve_fresh_offer` no longer resolves a client port for `replace` (the running proxy already owns it), and the staged state derives `upstream_socket`/`proxy_port` defaults when the old state lacks them.
- Fixed a stale-write race in `_start_proxy`: the spawned proxy daemon persists `upstream_socket`/`proxy_port` before binding, but `up` then saved `proxy_pid`/`local_port` from its pre-spawn state copy and could clobber them — leaving `state.json` without the endpoint fields the proxy itself later expects. `up` now merges the daemon-written fields from disk before saving, and `state.json` reliably records the client-facing `proxy_port` (distinct from the unsecure-mode tunnel `local_port`).
- `LocalProvider` gained a second same-profile offer (Tesla V100 on machine 8004) so `replace` can exercise the default auto-exclusion path locally; the fake/local providers also propagate `machine_id` into instance dicts.

## 2026-10-05

- Added `hostai replace`: swaps the running instance for a freshly provisioned machine with the same offer-selection options as `up` (`--offer`, `--machine`, `--skip-*`, `--max-price`, `--bid`/`--interruptible`, `--session`, `--dry-run`, `--allow-unvalidated`, `--name`, …) plus `--allow-same-machine` and `--no-archive`. The current `machine_id` is auto-excluded so the replacement lands on a different host unless pinned explicitly.
  - Zero client-facing downtime in tokenized mode: the new run is staged under `state.replace.json` (sidecar) while the old machine keeps serving; `state.json` flips atomically once the new machine's remote `/health` answers over SSH. The running proxy's upstream supervisor watches `state.json`, rebuilds its aiohttp session (`TokenizedProxy.retarget`), drops and re-creates the SSH tunnel against the new `ssh_url`, and re-fetches `/props` — `proxy.sock` and the TCP port stay bound the whole time.
  - The old `api_key`, `local_port`, `upstream_socket` path, and `proxy_pid` are carried into the new state, so the generated `env` file and proxy auth are unchanged across the swap. New `GET /_hostai/backend` route reports the upstream `instance_id` for cutover detection (older proxies fall back to a health-transition heuristic).
  - The old instance is destroyed only after the proxy reports healthy on the new backend (`/_hostai/backend` id match + `/health` 200). A retarget timeout destroys nothing: `state.json` stays on the new instance and the old one is left running (the proxy keeps watching and can still retarget; `hostai down --id <old>` removes it). Provisioning failures before cutover clean up the staged instance and sidecar without touching the live deployment.
  - Slot-cache warm swap: with cache enabled, the old machine's slot is saved+uploaded before provisioning so the new machine's prefetch restores the same signature (same model/ctx); the restore POST runs remotely over SSH after cutover.
- `hostai lookup` shows each offer's Vast `machine` id and accepts the same `--skip-machine`/`--skip-offer`/`--skip-country` exclusions as `up`/`monitor` (shared `OfferExclusions.is_excluded`), so you can preview the market while ruling out a bad host. Active skips are echoed as `[exclude] machines [...], ...` before the per-profile search lines.
- New `[blocklist]` config section (`machines`/`offers`/`countries` arrays; env `HOSTAI_BLOCKLIST_*`, comma-separated): a global exclusion list merged into every offer search — `up`, `lookup`, `monitor once|watch` — on top of per-invocation `--skip-*` flags and `state.json` skips.
- `start.sh` downloads now carry a progress watchdog: hf (especially hf_xet) can hang forever on half-broken egress — a reachable `huggingface.co` but dead xet CAS endpoints or poisoned DNS — without ever exiting. `_download_once` watches `/models` growth and kills hf after `HOSTAI_DL_STALL_SECONDS` (default 120s, poll 15s, up to 3 restarts); each stall retry first disables xet and then switches `HF_ENDPOINT` to `HF_MIRROR_ENDPOINT` if reachable, so downloads degrade gracefully instead of wedging the boot. A clean hf failure still propagates immediately.
- `hostai up --restart` now resurrects a dead `start.sh` inside a still-running container: the entrypoint keeps sshd alive after a boot failure (failed download, etc.) while Vast reports the instance "running", so `--restart` previously waited on a `/health` that could never appear. When `/run/qwen38/start.exitcode` exists and no start.sh/llama-server process remains, `up --restart` relaunches `/usr/local/bin/start.sh` over SSH with the original docker env (from `/proc/1/environ`); `hf download` resumes partial files.
- Wired up four config knobs that were documented in `hostai.toml.example` but never read: `[model] hf_repo` and `draft` are now forwarded to the container (`HF_REPO`/`DRAFT` env), `[image] unsecure` is honored by `hostai up` (`--unsecure` still wins), and `[monitor] max_results` now controls the per-profile search limit (previously hardcoded to 10).
- `monitor watch` actually sends a desktop notification on ALERT again — `notify.py` existed but was never wired into the Python rewrite (alerts only landed in the daemon log). Deduped per distinct offer/price so a persistent deal doesn't re-ping every interval.
- `hostai monitor once|watch|start` accept `--max-price <$/h>` (same option as `up`/`lookup`). A running instance already caps searches at its own dph; `--max-price` can only tighten that cap further — it never surfaces offers pricier than the current deployment. `monitor start` forwards the cap into the spawned daemon argv.
- Added `hostai down --force`: skips every remote-dependent step (provider status check, SSH tunnel, slot-cache save/upload, remote telemetry, remote llama stop) and goes straight to the provider destroy/pause call; local daemon cleanup and state bookkeeping still run. Useful when the remote is wedged mid-boot (e.g. a stalled model download) and the graceful path would wait on dead timeouts.

## 2026-10-04

- Requests without `max_tokens` are no longer silently capped at 512 (`bench.max_tokens`). The proxy now sends `n_predict` from the new `[proxy] default_max_tokens` option (env `HOSTAI_PROXY_DEFAULT_MAX_TOKENS`, default `-1` = run to EOS/context end), and `max_completion_tokens` is accepted as an alias for `max_tokens`. The same value is forwarded to the remote as `N_PREDICT` so `start.sh` passes it as `llama-server --n-predict`, keeping passthrough mode and direct `/completion` calls consistent.

## 2026-10-03

- Added `hostai info`: shows provider account ground truth — remaining credit balance (`Provider.get_account_info`, Vast `/users/current`) and every instance on the account (`Provider.list_instances`, Vast `/api/v1/instances/`), with the local tracked name next to each id, per-instance status/GPU/$-per-hour/SSH/age, and a running burn-rate total. Untracked instances are flagged with a `hostai down --id <id>` hint.
- Added `hostai down --id <provider-id>` (repeatable) for destroying/pausing instances by raw provider id — including machines with no local hostai state (e.g. listed by `hostai info`). Ids matching a tracked instance run the normal full shutdown path; untracked ids go through the new `down_remote_instance`, which skips cache save, telemetry archive, and daemon cleanup. `--id` conflicts with `--all`/`--name`.
- `LocalProvider` parity: `list_instances` merges the shared registry with labeled docker containers so orphans appear in `info`; `get_instance`/`start`/`stop`/`destroy` resolve the `hostai.instance_id=<id>` docker label when the registry entry was lost.
- `start.sh` probes `huggingface.co` reachability before downloading and falls back to `HF_MIRROR_ENDPOINT` (default `https://hf-mirror.com`) when egress is filtered — covers CN hosts where HF is DNS-poisoned/SNI-reset as well as hosts with broken DNS. `HOSTAI_HF_ENDPOINT` overrides detection entirely; `HF_HUB_DISABLE_XET` is set on mirror endpoints since xet CAS is not proxied.
- New `[model] model_sha256`/`draft_sha256` pins (env overrides `MODEL_SHA256`/`DRAFT_SHA256`): `start.sh` verifies each downloaded blob and aborts boot on mismatch, which keeps mirror downloads trustworthy.

## 2026-10-01

- Multiple parallel instances: `hostai up --name <name>` provisions an additional deployment that runs alongside the default one.
  - Named instances store all artifacts under `.hostai-vast/instances/<name>/` (own `state.json`, `env`, `known_hosts`, `proxy.sock`, `upstream.sock`, `proxy.pid`, `proxy.log`, `.lifecycle.lock`); the default instance keeps the legacy `.hostai-vast/` layout untouched.
  - `-n/--name` (or `HOSTAI_INSTANCE`) selects the instance on `up`, `down`, `status`, `bench`, `cache copy`, `cost volume-break-even`, `proxy`, `log`, `monitor *` and `watchdog *`; a numeric provider instance id also works as the selector. When exactly one instance is tracked the selector is optional.
  - `hostai status` without `--name` prints a fleet overview table when more than one instance is tracked; `hostai down --all` tears every tracked instance down.
  - Local ports are claimed per instance: `up`, `ensure_tunnel` and proxy startup skip ports recorded in sibling state files even when the sibling has not bound them yet.
  - Lifecycle locking is per instance (`.lifecycle.lock` inside each instance directory); a short global `.allocate.lock` serializes the live-instance check, port claim, provider create and initial state write.
  - Monitor, watchdog and proxy daemons are scoped per instance (`monitor-<name>.pid`, `proxy-<name>.log`, `proxy-content-<name>.jsonl`) and carry `--name <instance>` in their argv, so identity-checked pid-file signaling can never stop a sibling's daemon.
  - `LocalProvider` serializes its shared `.hostai-vast/local-provider.json` container registry under `.local-provider.lock` and re-reads it on each access, so parallel local `up`/`down` runs cannot drop each other's entries.
  - TLS certificate regeneration is serialized under the allocation lock so concurrent `up` runs cannot produce a mismatched cert/key pair.
- Expanded cheap single-GPU coverage after a Vast market scan (all reuse existing images):
  - New `turing` image (SM75) + `turing-128k` profile for the Quadro RTX 8000 48 GB — roughly V100 money (~$0.26/h) with 1.4x the VRAM. Built on the default CUDA 12.8 bases (Turing is still supported there; only Volta needs the 12.2 pin).
  - New `5000ada-128k` profile (SM89 `ada` image) for the 32 GB Ada value tier: RTX 5000 Ada (~$0.34/h) and modded RTX 4080S 32 GB. Same ctx/cache settings as `5090-128k`, which proves 128k fits in ~32 GB.
  - `a100-128k` now also matches `A100_PCIE` offers (~$0.42/h) in addition to `A100_SXM4`.
  - `blackwell-128k` now also matches `RTX_PRO_4500` (32 GB, ~$0.37/h) and `RTX_PRO_5000` (48 GB).
  - The 48 GB Ada profiles (`ada-64k`/`ada-128k`/`ada-256k`) now also match `RTX_4090` 48 GB modded variants; the `gpu_ram>=48` floor keeps stock 24 GB 4090s out.
- Fixed all profile queries using `num_gpus>=1`: now `num_gpus=1`. `llama-server` runs `--split-mode none` on a single GPU, so multi-GPU listings only billed idle cards, and `gpu_ram` reporting on multi-GPU offers is ambiguous for the VRAM floor.
- `monitor_hardware.gpu_ranks` gained entries for the new GPUs (Q RTX 8000, RTX 5000 Ada, RTX 4080S, RTX PRO 4500/5000) plus the previously unranked CMP 170HX, A100 PCIE and A100 SXM4 so `same_or_better` monitoring covers every profiled card.
- Added offer-exclusion options `--skip-offer` and `--skip-country` (repeatable), alongside the existing `--skip-machine`.
  - `hostai up` accepts all three; `--skip-country` takes alpha-2/alpha-3 codes or country names (exact lookups only — no fuzzy matching).
  - `--offer` combined with the same `--skip-offer` id fails fast with a conflict error; all `--skip-*` flags are ignored (with a warning) on `--restart`.
  - `hostai monitor once|watch|start` accept the same flags; `monitor start` forwards them to the spawned `watch` daemon.
  - Exclusions given to `up` are recorded in `state.json` (via `market.OfferExclusions`), so a running or auto-started monitor never recommends an offer/host the user already ruled out.
  - Market layer: new `market.OfferExclusions` dataclass replaces the per-flag `skip_machines` kwarg on `filter_eligible_offers`/`select_offer` and adds offer-ID and country filtering.
- Monitor output (`monitor once`/`watch`) now prints the candidate's location (`loc=...`).
- Fixed `--skip-country` never matching real Vast offers: `geolocation` is reported as `"Region/City, CC"` (e.g. `"Arizona, US"`, `"Jiangsu, CN"`); the normalizer now extracts the trailing alpha-2 code.
- `hostai proxy` now self-heals after the SSH upstream tunnel dies: a steady-state supervisor re-establishes the unix (or TCP) tunnel, flips `ready` so clients get a clean 503 while the model restarts instead of a bare 500, and exits once the provider confirms the instance is gone. `_chat` also maps upstream connect failures to 502 instead of an unhandled 500.
- `ssh.run_remote` reports the exception class name when `str(exc)` is empty (e.g. `TimeoutError`), and the `up` llama-server preflight error now shows the return code and falls back to stdout.
- `hostai status` renders `Elapsed` as `hh:mm:ss` (hours not capped at 24) via the new `utils.format_duration` instead of raw seconds.

## 2026-09-30

- Fixed `hostai up -l/--local-port` silently losing to a configured `[proxy] port`: `_resolve_client_port` stored the CLI port only in `ssh.local_port`, so `_start_proxy` re-derived `proxy.port || ssh.local_port` and bound the stale configured port. An explicit `--local-port` now overrides the proxy port for that run.

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
