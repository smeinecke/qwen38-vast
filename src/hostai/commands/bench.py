import hashlib
import json
import re
import shlex
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import click

from hostai import ssh, utils
from hostai.api import LlamaClient
from hostai.config import Config
from hostai.state import State, state_dir

_DEFAULT_PROMPT = """\
You are reviewing a performance-sensitive Python service. Analyze the code below as if this were a real production code review. Identify correctness, concurrency, resource-management and performance problems, then propose a concrete refactor. Be precise and include revised code for the most important section.

```python
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3

executor = ThreadPoolExecutor(max_workers=32)
conn = sqlite3.connect("cache.db")

def analyze(paths, model):
    futures = []
    for path in paths:
        text = open(path).read()
        cached = conn.execute("select result from cache where path=?", (path,)).fetchone()
        if cached:
            return json.loads(cached[0])
        futures.append(executor.submit(model, text))

    results = [f.result() for f in futures]
    for path, result in zip(paths, results):
        conn.execute("insert into cache(path,result) values (?,?)", (path, json.dumps(result)))
    return results
```
"""


_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_label(label: str) -> str:
    label = re.sub(r"[\s/]+", "-", label)
    label = _SAFE_LABEL_RE.sub("", label)
    return label or "bench"


def _delta_metrics(before: Dict[str, float], after: Dict[str, float], name: str) -> Optional[float]:
    if name not in before or name not in after:
        return None
    value = after[name] - before[name]
    return value if value >= 0 else None


def _read_prompt(prompt_file: Optional[Path]) -> str:
    if prompt_file:
        return prompt_file.read_text(errors="replace")
    return _DEFAULT_PROMPT


def _stream_chat(
    client: LlamaClient,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Tuple[
    bool,
    Optional[str],
    List[str],
    List[str],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    float,
    Optional[float],
    Optional[float],
    Optional[str],
]:
    """Stream a chat completion and measure timing/usage.

    Returns a large tuple rather than a class to keep this self-contained.
    """
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    usage: Dict[str, Any] = {}
    timings: Dict[str, Any] = {}
    last_obj: Dict[str, Any] = {}
    error: Optional[str] = None
    first_byte_s: Optional[float] = None
    first_token_s: Optional[float] = None
    total_s = 0.0
    finish_reason: Optional[str] = None

    t0 = time.perf_counter()
    try:
        stream = client.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            timeout=timeout,
        )
        # chat() returns an iterator when stream=True.
        for obj in stream:  # type: ignore[call-overload]
            if not isinstance(obj, dict):
                continue
            if first_byte_s is None:
                first_byte_s = time.perf_counter() - t0
            last_obj = obj
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            if isinstance(obj.get("timings"), dict):
                timings = obj["timings"]
            choices = obj.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice.get("finish_reason")
                    delta = choice.get("delta")
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            content_parts.append(content)
                            if first_token_s is None:
                                first_token_s = time.perf_counter() - t0
                        reasoning = delta.get("reasoning_content")
                        if isinstance(reasoning, str) and reasoning:
                            reasoning_parts.append(reasoning)
                            if first_token_s is None:
                                first_token_s = time.perf_counter() - t0
                        reasoning_alt = delta.get("reasoning")
                        if isinstance(reasoning_alt, str) and reasoning_alt:
                            reasoning_parts.append(reasoning_alt)
                            if first_token_s is None:
                                first_token_s = time.perf_counter() - t0
        total_s = time.perf_counter() - t0
    except Exception as exc:
        total_s = time.perf_counter() - t0
        error = f"{type(exc).__name__}: {exc}"

    return (
        error is None,
        error,
        content_parts,
        reasoning_parts,
        usage,
        timings,
        last_obj,
        total_s,
        first_byte_s,
        first_token_s,
        finish_reason,
    )


@click.command("bench", help="Run a benchmark against the running instance.")
@click.option("--label", default="coding-smoke", help="Human-readable benchmark label.")
@click.option("--prompt-file", type=click.Path(path_type=Path), default=None, help="Use FILE as the user prompt.")
@click.option("--max-tokens", type=int, default=None, help="Max tokens to generate.")
@click.option("--timeout", type=int, default=None, help="HTTP timeout in seconds.")
@click.option("--save-prompt", is_flag=True, help="Copy the prompt into the benchmark directory.")
@click.pass_obj
def cmd_bench(
    config: Config,
    label: str,
    prompt_file: Optional[Path],
    max_tokens: Optional[int],
    timeout: Optional[int],
    save_prompt: bool,
) -> None:
    state = State.load(state_dir(config.root_dir) / "state.json")
    if not state.exists or not state.instance_id:
        raise click.ClickException("no running instance; run hostai up first")

    if not ssh.is_tunnel_healthy(config, state, timeout=3):
        ssh.ensure_tunnel(config, state)

    client = LlamaClient(config, state)
    if not client.health():
        raise click.ClickException("llama-server is not healthy")

    tokens = max_tokens if max_tokens is not None else config.bench.max_tokens
    sec = timeout if timeout is not None else config.bench.timeout

    run_dir = state.run_dir
    if not run_dir:
        raise click.ClickException("state predates telemetry support; start a new instance with hostai up")
    run_dir = Path(run_dir)
    bench_dir = run_dir / "benchmarks" / f"{utils.now_epoch()}-{_safe_label(label)}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.chmod(0o700)

    prompt = _read_prompt(prompt_file)
    prompt_bytes = prompt.encode("utf-8")
    if save_prompt:
        (bench_dir / "prompt.txt").write_text(prompt)
        (bench_dir / "prompt.txt").chmod(0o600)

    safe = _safe_label(label)
    click.echo(
        f"[bench] session={run_dir.name} profile={state.profile} ctx={state.ctx_size} label={safe} max_tokens={tokens}"
    )

    known_hosts = state.state_file.parent / "known_hosts"

    sampler, stop_file, gpu_result_box = _start_gpu_sampler(state, sec, known_hosts)

    # Metrics before
    metrics_before_text = client.get_metrics_text() or ""
    (bench_dir / "metrics-before.prom").write_text(metrics_before_text)
    metrics_before = client.get_metrics()

    started_wall = time.time()
    (
        ok,
        error,
        content_parts,
        reasoning_parts,
        usage,
        timings,
        last_obj,
        total_s,
        first_byte_s,
        first_token_s,
        finish_reason,
    ) = _stream_chat(
        client,
        [{"role": "user", "content": prompt}],
        max_tokens=tokens,
        temperature=config.bench.temperature if config.bench.temperature is not None else 0.7,
        timeout=sec,
    )
    ended_wall = time.time()

    # Metrics after
    metrics_after_text = client.get_metrics_text() or ""
    (bench_dir / "metrics-after.prom").write_text(metrics_after_text)
    metrics_after = client.get_metrics()

    _collect_gpu_result(state, sampler, stop_file, known_hosts, bench_dir, gpu_result_box)
    _write_server_tail(state, known_hosts, bench_dir)

    result, request_cost = _build_result(
        state,
        started_wall,
        ended_wall,
        prompt,
        prompt_bytes,
        tokens,
        label,
        content_parts,
        reasoning_parts,
        last_obj,
        finish_reason,
        usage,
        timings,
        metrics_before,
        metrics_after,
        ok,
        error,
        first_byte_s,
        first_token_s,
        total_s,
        bench_dir,
    )

    (bench_dir / "response.txt").write_text("".join(content_parts))
    (bench_dir / "reasoning.txt").write_text("".join(reasoning_parts))
    (bench_dir / "response-final-chunk.json").write_text(json.dumps(last_obj, indent=2, ensure_ascii=False))
    (bench_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    if error:
        raise click.ClickException(f"benchmark request failed: {error}")

    _print_summary(result)
    click.echo(f"\nBenchmark artifacts: {bench_dir}")


def _metric_deltas(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Prometheus metric deltas for a benchmark run."""
    return {
        "prompt_tokens": _delta_metrics(before, after, "llamacpp:prompt_tokens_total"),
        "prompt_seconds": _delta_metrics(before, after, "llamacpp:prompt_seconds_total"),
        "predicted_tokens": _delta_metrics(before, after, "llamacpp:tokens_predicted_total"),
        "predicted_seconds": _delta_metrics(before, after, "llamacpp:tokens_predicted_seconds_total"),
        "draft_tokens": _delta_metrics(before, after, "llamacpp:spec_decode_num_draft_tokens_total"),
        "draft_accepted": _delta_metrics(before, after, "llamacpp:spec_decode_num_accepted_tokens_total"),
        "draft_steps": _delta_metrics(before, after, "llamacpp:spec_decode_num_drafts_total"),
    }


def _token_counts(usage: Dict[str, Any], timings: Dict[str, Any], metrics: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """Derive prompt/completion token counts and cache/prompt evaluated counts."""
    timing_cache_n = int(timings.get("cache_n") or 0)
    timing_prompt_n = int(timings.get("prompt_n") or 0)
    if usage.get("prompt_tokens") is not None:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
    elif timing_cache_n or timing_prompt_n:
        prompt_tokens = timing_cache_n + timing_prompt_n
    else:
        prompt_tokens = int(metrics.get("prompt_tokens") or 0)
    completion_tokens = int(
        usage.get("completion_tokens") or timings.get("predicted_n") or metrics.get("predicted_tokens") or 0
    )
    return prompt_tokens, completion_tokens, timing_cache_n, timing_prompt_n


def _tps_values(
    timings: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return (prompt_tps, decode_tps, prompt_tps_fallback, decode_tps_fallback)."""
    prompt_tokens = metrics.get("prompt_tokens")
    prompt_seconds = metrics.get("prompt_seconds")
    pred_tokens = metrics.get("predicted_tokens")
    pred_seconds = metrics.get("predicted_seconds")
    prompt_tps_fallback = None
    if prompt_tokens is not None and prompt_seconds and prompt_seconds > 0:
        prompt_tps_fallback = prompt_tokens / prompt_seconds
    pred_tps_fallback = None
    if pred_tokens is not None and pred_seconds and pred_seconds > 0:
        pred_tps_fallback = pred_tokens / pred_seconds
    prompt_tps = timings.get("prompt_per_second")
    if not isinstance(prompt_tps, (int, float)):
        prompt_tps = prompt_tps_fallback
    pred_tps = timings.get("predicted_per_second")
    if not isinstance(pred_tps, (int, float)):
        pred_tps = pred_tps_fallback
    return prompt_tps, pred_tps, prompt_tps_fallback, pred_tps_fallback


def _draft_counts(timings: Dict[str, Any], metrics: Dict[str, Any]) -> Tuple[int, int, Optional[float]]:
    """Return (draft_n, draft_accepted, accept_rate)."""
    draft_n = int(timings.get("draft_n") or metrics.get("draft_tokens") or 0)
    draft_accepted = int(timings.get("draft_n_accepted") or metrics.get("draft_accepted") or 0)
    accept_rate = (draft_accepted / draft_n) if draft_n > 0 else None
    return draft_n, draft_accepted, accept_rate


def _build_result(
    state: State,
    started_wall: float,
    ended_wall: float,
    prompt: str,
    prompt_bytes: bytes,
    tokens: int,
    label: str,
    content_parts: List[str],
    reasoning_parts: List[str],
    last_obj: Dict[str, Any],
    finish_reason: Optional[str],
    usage: Dict[str, Any],
    timings: Dict[str, Any],
    metrics_before: Dict[str, Any],
    metrics_after: Dict[str, Any],
    ok: bool,
    error: Optional[str],
    first_byte_s: Optional[float],
    first_token_s: Optional[float],
    total_s: float,
    bench_dir: Path,
) -> Tuple[Dict[str, Any], float]:
    """Compute telemetry and assemble the benchmark result document."""
    metrics = _metric_deltas(metrics_before, metrics_after)
    prompt_tokens, completion_tokens, timing_cache_n, timing_prompt_n = _token_counts(usage, timings, metrics)
    cache_hit_rate = (timing_cache_n / prompt_tokens) if prompt_tokens > 0 and timing_cache_n >= 0 else None
    prompt_tps, pred_tps, prompt_tps_fallback, pred_tps_fallback = _tps_values(timings, metrics)
    draft_n, draft_accepted, accept_rate = _draft_counts(timings, metrics)

    stream_decode_tps = None
    if completion_tokens > 0 and first_token_s is not None and total_s > first_token_s:
        stream_decode_tps = completion_tokens / (total_s - first_token_s)

    dph = float(state.dph or 0)
    request_cost = dph * total_s / 3600.0

    gpu_summary = _summarize_gpu_csv(bench_dir / "gpu.csv")

    result = {
        "schema_version": 1,
        "label": label,
        "ok": ok,
        "error": error,
        "started_at_unix": started_wall,
        "ended_at_unix": ended_wall,
        "session": _public_state(state),
        "request": {
            "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "prompt_bytes": len(prompt_bytes),
            "prompt_chars": len(prompt),
            "max_tokens": tokens,
            "stream": True,
            "cache_prompt": True,
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens") or (prompt_tokens + completion_tokens)),
            "cached_tokens": (
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                if isinstance(usage.get("prompt_tokens_details"), dict)
                else None
            ),
        },
        "server_timings": timings,
        "performance": {
            "prompt_tps": prompt_tps,
            "decode_tps": pred_tps,
            "ttfb_s": first_byte_s,
            "ttft_s": first_token_s,
            "total_s": total_s,
            "stream_effective_decode_tps": stream_decode_tps,
            "draft_n": draft_n,
            "draft_n_accepted": draft_accepted,
            "draft_accept_rate": accept_rate,
            "request_compute_cost_usd": request_cost,
            "cache_n": timing_cache_n,
            "prompt_evaluated_n": timing_prompt_n,
            "cache_hit_rate": cache_hit_rate,
        },
        "metrics_delta": {
            "prompt_tokens": metrics["prompt_tokens"],
            "prompt_seconds": metrics["prompt_seconds"],
            "predicted_tokens": metrics["predicted_tokens"],
            "predicted_seconds": metrics["predicted_seconds"],
            "draft_tokens": metrics["draft_tokens"],
            "draft_accepted": metrics["draft_accepted"],
            "draft_verification_steps": metrics["draft_steps"],
            "prompt_tps": prompt_tps_fallback,
            "decode_tps": pred_tps_fallback,
        },
        "finish_reason": finish_reason,
        "gpu_summary": gpu_summary,
    }
    return result, request_cost


def _start_gpu_sampler(
    state: State, sec: int, known_hosts: Path
) -> Tuple[Optional[threading.Thread], Optional[Path], Dict[str, Any]]:
    """Start a background thread that samples nvidia-smi on the remote host."""
    if not state.ssh_url:
        return None, None, {}
    stop_file = Path("/tmp/.hostai_gpu_sampler")
    inner_cmd = (
        f"touch {stop_file} && "
        "while [ -f /tmp/.hostai_gpu_sampler ]; do "
        "nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu "
        "--format=csv,noheader,nounits 2>/dev/null || true; sleep 1; done; rm -f /tmp/.hostai_gpu_sampler"
    )
    cmd = f"timeout {sec + 60} bash -c {shlex.quote(inner_cmd)}"
    gpu_result_box: Dict[str, Any] = {}

    def _sample() -> None:
        ssh_url = state.ssh_url
        if not ssh_url:
            return
        gpu_result_box["result"] = ssh.run_remote(
            ssh_url,
            cmd,
            known_hosts=known_hosts,
            timeout=sec + 120,
        )

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    return sampler, stop_file, gpu_result_box


def _collect_gpu_result(
    state: State,
    sampler: Optional[threading.Thread],
    stop_file: Optional[Path],
    known_hosts: Path,
    bench_dir: Path,
    gpu_result_box: Dict[str, Any],
) -> None:
    """Stop the GPU sampler and write gpu.csv."""
    if sampler and stop_file and state.ssh_url:
        ssh.run_remote(state.ssh_url, f"rm -f {stop_file}", known_hosts=known_hosts, timeout=15)
        sampler.join(timeout=15)
        gpu_result = gpu_result_box.get("result")
        if gpu_result and gpu_result.stdout:
            gpu_csv = (
                "timestamp,index,name,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_w,temperature_c\n"
            )
            gpu_csv += gpu_result.stdout
            (bench_dir / "gpu.csv").write_text(gpu_csv)


def _write_server_tail(state: State, known_hosts: Path, bench_dir: Path) -> None:
    """Fetch the last 300 lines of the remote server log."""
    if state.ssh_url:
        server_res = ssh.run_remote(
            state.ssh_url,
            "tail -n 300 /var/log/qwen38/server.log 2>/dev/null || true",
            known_hosts=known_hosts,
            timeout=60,
        )
        if server_res.stdout:
            (bench_dir / "server-tail.log").write_text(server_res.stdout)


def _public_state(state: State) -> Dict[str, Any]:
    to_dict = getattr(state, "to_dict", None)
    public = cast(Dict[str, Any], to_dict() if callable(to_dict) else dict(state.data))
    public.pop("api_key", None)
    public.pop("tunnel_pid", None)
    return public


def _summarize_gpu_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return {}
    if len(lines) < 2:
        return {}

    rows = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            try:
                memory_used = float(parts[4])
                power = float(parts[5]) if parts[5] else 0.0
                temp = float(parts[6]) if len(parts) > 6 and parts[6] else 0.0
                rows.append({"memory_used_mib": memory_used, "power_w": power, "temperature_c": temp})
            except (ValueError, IndexError):
                continue

    if not rows:
        return {}

    peak_memory = max(r["memory_used_mib"] for r in rows)
    avg_power = sum(r["power_w"] for r in rows) / len(rows)
    peak_temp = max(r["temperature_c"] for r in rows)
    return {
        "peak_memory_mib": peak_memory,
        "avg_power_w": avg_power,
        "peak_temperature_c": peak_temp,
    }


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _print_summary(result: Dict[str, Any]) -> None:
    """Print the human-readable benchmark summary."""
    usage = result.get("usage", {})
    perf = result.get("performance", {})
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tps = perf.get("prompt_tps")
    pred_tps = perf.get("decode_tps")
    timing_cache_n = int(perf.get("cache_n") or 0)
    cache_hit_rate = perf.get("cache_hit_rate")
    first_token_s = perf.get("ttft_s")
    total_s = float(perf.get("total_s") or 0)
    draft_n = int(perf.get("draft_n") or 0)
    draft_accepted = int(perf.get("draft_n_accepted") or 0)
    accept_rate = perf.get("draft_accept_rate")
    request_cost = float(perf.get("request_compute_cost_usd") or 0)

    click.echo(f"  prompt: {prompt_tokens} tok @ {_fmt(prompt_tps)} t/s")
    if timing_cache_n:
        click.echo(
            f"  cache:  {timing_cache_n} tok reused ({_fmt(100.0 * cache_hit_rate if cache_hit_rate is not None else None, 1)}%)"
        )
    else:
        click.echo("  cache:  0 tok reused")
    click.echo(f"  decode: {completion_tokens} tok @ {_fmt(pred_tps)} t/s")
    click.echo(f"  TTFT:   {_fmt(first_token_s, 3)} s | total: {_fmt(total_s, 3)} s")
    if draft_n:
        click.echo(
            f"  MTP:    {draft_accepted}/{draft_n} accepted ({_fmt(100.0 * accept_rate if accept_rate is not None else None, 1)}%)"
        )
    else:
        click.echo("  MTP:    no draft counters in response")
    click.echo(f"  cost:   ${request_cost:.6f} compute for this request")
