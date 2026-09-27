"""Render the proxy content log as a human-readable transcript."""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterator

import click

from hostai.config import Config
from hostai.proxy import _content_log_path, _proxy_log_file

_ROLE_COLORS = {
    "system": "yellow",
    "user": "cyan",
    "assistant": "green",
    "tool": "magenta",
}


def _ts(rec: Dict[str, Any]) -> str:
    ts = rec.get("ts")
    if isinstance(ts, (int, float)):
        return time.strftime("%H:%M:%S", time.localtime(ts))
    return ""


def _role(role: str) -> str:
    return click.style(f"[{role}]", fg=_ROLE_COLORS.get(role, "white"), bold=True)


def _dim(text: str) -> str:
    return click.style(text, fg="bright_black")


def _thinking(text: str) -> str:
    # Reasoning is long-form content the user reads; distinguish it with
    # italics instead of a dark color so it stays readable on any theme.
    return click.style(text, italic=True)


def _message_text(content: Any) -> str:
    """Flatten OpenAI message content (string or content-part list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(p.get("text", ""))
            for p in content
            if isinstance(p, dict) and p.get("type", "text") == "text"
        ]
        return "\n".join(t for t in parts if t)
    return ""


def _openai_obj_deltas(obj: Dict[str, Any]) -> Iterator[Dict[str, str]]:
    """Extract content/reasoning text from an OpenAI chunk or response object."""
    for choice in obj.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or choice.get("message") or {}
        out: Dict[str, str] = {}
        for key in ("reasoning_content", "content"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                out[key] = value
        if out:
            yield out


def _extract_upstream_deltas(text: str) -> Iterator[Dict[str, str]]:
    """Pull response text out of raw pass-through bytes (SSE or a JSON body)."""
    saw_sse = False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        saw_sse = True
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield from _openai_obj_deltas(obj)
    if saw_sse:
        return
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return
    if isinstance(obj, dict):
        yield from _openai_obj_deltas(obj)


class _Transcript:
    """Render proxy-content.jsonl records as a chat-style transcript.

    Tracks one open assistant block per request id so streamed deltas append
    inline, reasoning renders dimmed, and the block closes on done/response.
    """

    def __init__(self) -> None:
        self._open: Dict[str, str] = {}  # req_id -> current output phase

    def feed(self, rec: Dict[str, Any]) -> None:
        event = rec.get("event")
        handler = getattr(self, f"_on_{event}", None)
        if handler:
            handler(rec)

    def _on_request(self, rec: Dict[str, Any]) -> None:
        req_id = rec.get("id", "")
        body = rec.get("body") or {}
        messages = rec.get("messages") or body.get("messages") or []
        meta = []
        if rec.get("stream") or body.get("stream"):
            meta.append("stream")
        if rec.get("prompt_tokens"):
            meta.append(f"{rec['prompt_tokens']} prompt tokens")
        if rec.get("passthrough"):
            meta.append("passthrough")
        suffix = f" · {' · '.join(meta)}" if meta else ""
        click.echo()
        click.echo(_dim(f"── {req_id} {_ts(rec)}{suffix} " + "─" * 20))
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            click.echo(_role(str(msg.get("role", "?"))))
            text = _message_text(msg.get("content"))
            if text:
                click.echo(text.rstrip())
            for tc in msg.get("tool_calls") or []:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                click.echo(_dim(f"  → tool_call {fn.get('name')}: {fn.get('arguments')}"))
            click.echo()

    def _assistant(self, req_id: str, phase: str) -> None:
        if req_id not in self._open:
            click.echo()
            click.echo(f"{_role('assistant')} ", nl=False)
            self._open[req_id] = phase
        elif self._open[req_id] != phase:
            click.echo("\n" + _dim("─── answer ───") + "\n", nl=False)
            self._open[req_id] = phase

    def _emit(self, req_id: str, delta: Dict[str, str]) -> None:
        reasoning = delta.get("reasoning_content")
        content = delta.get("content")
        if reasoning:
            self._assistant(req_id, "reasoning")
            click.echo(_thinking(reasoning), nl=False)
        if content:
            self._assistant(req_id, "content")
            click.echo(content, nl=False)

    def _on_delta(self, rec: Dict[str, Any]) -> None:
        self._emit(rec.get("id", ""), rec)

    def _on_upstream_chunk(self, rec: Dict[str, Any]) -> None:
        for delta in _extract_upstream_deltas(str(rec.get("data") or "")):
            self._emit(rec.get("id", ""), delta)

    def _on_upstream_response(self, rec: Dict[str, Any]) -> None:
        status = rec.get("status")
        if status and status != 200:
            click.echo(click.style(f"[upstream status={status}]", fg="red"))

    def _on_response(self, rec: Dict[str, Any]) -> None:
        req_id = rec.get("id", "")
        msg = rec.get("message") or {}
        delta: Dict[str, str] = {}
        for key in ("reasoning_content", "content"):
            if isinstance(msg.get(key), str) and msg[key]:
                delta[key] = msg[key]
        if delta:
            self._emit(req_id, delta)
        for tc in msg.get("tool_calls") or []:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            click.echo(click.style(f"→ tool_call {fn.get('name')}: {fn.get('arguments')}", fg="yellow"))
        usage = rec.get("usage") or {}
        self._close(req_id, f"finish={rec.get('finish_reason')} · {usage.get('completion_tokens', '?')} tokens")

    def _on_done(self, rec: Dict[str, Any]) -> None:
        self._close(
            rec.get("id", ""),
            f"finish={rec.get('finish_reason')} · {rec.get('completion_tokens', '?')} tokens",
        )

    def _on_error(self, rec: Dict[str, Any]) -> None:
        req_id = rec.get("id", "")
        if req_id in self._open:
            click.echo()
            del self._open[req_id]
        click.echo(click.style(f"[error] {rec.get('error', '')}", fg="red"))

    def _close(self, req_id: str, note: str) -> None:
        if req_id in self._open:
            click.echo()
            del self._open[req_id]
        click.echo(_dim(f"   [{note}]"))


def _tail_lines(path: Path, lines: int, follow: bool) -> Iterator[str]:
    """Yield the last ``lines`` lines of ``path``, then keep following it."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        backlog = deque(f, maxlen=lines) if lines else deque(f)
        pos = f.tell()
    yield from backlog
    if not follow:
        return
    while True:
        try:
            if path.stat().st_size < pos:
                pos = 0  # log was truncated/rotated; start over
            with path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                for line in f:
                    yield line
                pos = f.tell()
        except FileNotFoundError:
            pass
        time.sleep(0.25)


def _follow_records(path: Path, lines: int, follow: bool) -> None:
    transcript = _Transcript()
    for line in _tail_lines(path, lines, follow):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            transcript.feed(rec)


@click.command("log", help="Show the proxy content log as a readable transcript.")
@click.option(
    "-n",
    "--lines",
    type=int,
    default=200,
    show_default=True,
    help="Recent log lines to render first (0 = whole file).",
)
@click.option(
    "--follow/--no-follow",
    default=True,
    show_default=True,
    help="Keep watching for new records (Ctrl-C to stop).",
)
@click.option("--ops", is_flag=True, help="Tail the operational proxy.log instead (raw output).")
@click.pass_obj
def cmd_log(config: Config, lines: int, follow: bool, ops: bool) -> None:
    if ops:
        path = _proxy_log_file(config)
        if not path.exists():
            raise click.ClickException(f"{path} not found; run 'hostai proxy' first")
        try:
            for line in _tail_lines(path, lines, follow):
                click.echo(line, nl=False)
        except KeyboardInterrupt:
            return
        return

    path = _content_log_path(config)
    if not path.exists():
        raise click.ClickException(
            f"{path} not found; enable content logging with 'log_content = true' in [proxy] "
            "or HOSTAI_PROXY_LOG_CONTENT=1, then restart 'hostai proxy'"
        )
    try:
        _follow_records(path, lines, follow)
    except KeyboardInterrupt:
        pass
