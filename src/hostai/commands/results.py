import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.table import Table

from hostai.config import Config
from hostai.state import runs_dir


def _load_rows(runs_dir: Path) -> List[Dict[str, Any]]:
    if not runs_dir.exists():
        return []
    files = sorted(runs_dir.glob("*/benchmarks/*/metrics.json"))
    rows = []
    for path in files:
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        session = d.get("session") or {}
        perf = d.get("performance") or {}
        usage = d.get("usage") or {}
        gpu = d.get("gpu_summary") or {}
        rows.append(
            {
                "run": path.parents[2].name,
                "bench": path.parent.name,
                "label": d.get("label") or "",
                "profile": session.get("profile") or "",
                "gpu": session.get("gpu") or "",
                "ctx": session.get("ctx_size") or "",
                "prompt_tok": usage.get("prompt_tokens") or 0,
                "prompt_tps": perf.get("prompt_tps"),
                "output_tok": usage.get("completion_tokens") or 0,
                "decode_tps": perf.get("decode_tps"),
                "ttft_s": perf.get("ttft_s"),
                "cache_hit": perf.get("cache_hit_rate"),
                "cache_n": perf.get("cache_n") or 0,
                "mtp_accept": perf.get("draft_accept_rate"),
                "peak_vram_mib": gpu.get("peak_memory_mib"),
                "avg_power_w": gpu.get("avg_power_w"),
                "request_cost_usd": perf.get("request_compute_cost_usd"),
                "ok": d.get("ok", False),
                "path": str(path.parent),
            }
        )
    return rows


def _format_value(v: Any, digits: int = 1) -> str:
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _ctx_label(v: Any) -> str:
    if v is None or v == "":
        return "-"
    try:
        n = int(v)
        if n >= 1024 and n % 1024 == 0:
            return f"{n // 1024}k"
        return str(n)
    except Exception:
        return str(v)


def _cache_pct(v: Any) -> str:
    if v is None or v == "":
        return "-"
    try:
        return _format_value(100 * float(v), 1) + "%"
    except (TypeError, ValueError):
        return "-"


def _gb(v: Any) -> str:
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v) / 1024:.1f}G"
    except (TypeError, ValueError):
        return "-"


def _watts(v: Any) -> str:
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.0f}W"
    except (TypeError, ValueError):
        return "-"


def _cost(v: Any) -> str:
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.5f}"
    except (TypeError, ValueError):
        return "-"


def _short_gpu(v: str) -> str:
    s = str(v)
    for prefix in ("NVIDIA GeForce ", "NVIDIA "):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    return s


def _render_table(rows: List[Dict[str, Any]]) -> None:
    console = Console()
    table = Table(title="Benchmark results", show_header=True, header_style="bold")
    headers = [
        "run",
        "label",
        "profile",
        "gpu",
        "ctx",
        "pp t/s",
        "tg t/s",
        "TTFT",
        "cache",
        "MTP",
        "VRAM",
        "power",
        "req $",
    ]
    for h in headers:
        table.add_column(h)
    for r in rows:
        table.add_row(
            r["run"],
            r["label"],
            r["profile"],
            _short_gpu(r["gpu"]),
            _ctx_label(r["ctx"]),
            _format_value(r["prompt_tps"], 1),
            _format_value(r["decode_tps"], 1),
            _format_value(r["ttft_s"], 2),
            _cache_pct(r["cache_hit"]),
            _cache_pct(r["mtp_accept"]),
            _gb(r["peak_vram_mib"]),
            _watts(r["avg_power_w"]),
            _cost(r["request_cost_usd"]),
        )
    console.print(table)


def _render_csv(rows: List[Dict[str, Any]]) -> str:
    fields = [
        "run",
        "bench",
        "label",
        "profile",
        "gpu",
        "ctx",
        "prompt_tok",
        "prompt_tps",
        "output_tok",
        "decode_tps",
        "ttft_s",
        "cache_hit",
        "cache_n",
        "mtp_accept",
        "peak_vram_mib",
        "avg_power_w",
        "request_cost_usd",
        "ok",
        "path",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in fields})
    return buf.getvalue()


@click.command("results", help="Compare benchmark results from .hostai-runs.")
@click.option(
    "--runs-dir",
    "runs_dir_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the directory to scan for runs.",
)
@click.option("--csv", "output_format", flag_value="csv", help="Output CSV.")
@click.option("--json", "output_format", flag_value="json", help="Output JSON.")
@click.pass_obj
def cmd_results(config: Config, runs_dir_path: Optional[Path], output_format):
    rdir = runs_dir_path or runs_dir(config.root_dir)
    rows = _load_rows(rdir)
    if output_format == "json":
        click.echo(json.dumps(rows, indent=2, default=str))
    elif output_format == "csv":
        click.echo(_render_csv(rows))
    elif not rows:
        click.echo(f"No benchmark results found below {rdir}")
    else:
        _render_table(rows)
