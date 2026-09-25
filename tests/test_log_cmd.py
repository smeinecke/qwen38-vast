"""Tests for the `hostai log` transcript renderer."""

import json

from click.testing import CliRunner

from hostai.commands.log import (
    _extract_upstream_deltas,
    _tail_lines,
    _Transcript,
    cmd_log,
)


def test_transcript_stream(capsys):
    t = _Transcript()
    t.feed({
        "ts": 1700000000.0,
        "event": "request",
        "id": "req-1",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "prompt_tokens": 3,
    })
    t.feed({"ts": 1700000000.1, "event": "delta", "id": "req-1", "reasoning_content": "thinking "})
    t.feed({"ts": 1700000000.2, "event": "delta", "id": "req-1", "content": "hello"})
    t.feed({"ts": 1700000000.3, "event": "done", "id": "req-1",
            "finish_reason": "stop", "completion_tokens": 5})
    out = capsys.readouterr().out
    assert "req-1" in out
    assert "[user]" in out and "hi" in out
    assert "[assistant]" in out
    assert "thinking" in out and "hello" in out
    assert "answer" in out  # reasoning -> answer phase marker
    assert "finish=stop" in out and "5 tokens" in out


def test_transcript_response(capsys):
    t = _Transcript()
    t.feed({"event": "request", "id": "req-2",
            "messages": [{"role": "user", "content": "ping"}], "stream": False})
    t.feed({"event": "response", "id": "req-2",
            "message": {"role": "assistant", "content": "pong"},
            "finish_reason": "stop", "usage": {"completion_tokens": 2}})
    out = capsys.readouterr().out
    assert "pong" in out
    assert "2 tokens" in out


def test_transcript_error(capsys):
    t = _Transcript()
    t.feed({"event": "error", "id": "req-3", "error": "upstream returned 500"})
    assert "upstream returned 500" in capsys.readouterr().out


def test_extract_upstream_deltas_sse():
    text = (
        'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    deltas = list(_extract_upstream_deltas(text))
    assert deltas == [{"content": "hel"}, {"content": "lo"}]


def test_extract_upstream_deltas_json_body():
    body = json.dumps({"choices": [{"message": {"content": "full answer"}}]})
    assert list(_extract_upstream_deltas(body)) == [{"content": "full answer"}]


def test_extract_upstream_deltas_garbage():
    assert list(_extract_upstream_deltas("not json at all")) == []


def test_tail_lines(tmp_path):
    log = tmp_path / "l.jsonl"
    log.write_text("".join(f"line {i}\n" for i in range(10)))
    assert list(_tail_lines(log, 3, follow=False)) == ["line 7\n", "line 8\n", "line 9\n"]
    assert len(list(_tail_lines(log, 0, follow=False))) == 10


def test_cmd_log_renders(config):
    log_dir = config.root_dir / ".hostai-cache"
    log_dir.mkdir(parents=True)
    records = [
        {"ts": 1700000000.0, "event": "request", "id": "req-9",
         "messages": [{"role": "user", "content": "hello there"}], "prompt_tokens": 5},
        {"ts": 1700000000.1, "event": "response", "id": "req-9",
         "message": {"role": "assistant", "content": "general kenobi"},
         "finish_reason": "stop", "usage": {"completion_tokens": 2}},
    ]
    (log_dir / "proxy-content.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    result = CliRunner().invoke(cmd_log, ["--no-follow"], obj=config)
    assert result.exit_code == 0
    assert "hello there" in result.output
    assert "general kenobi" in result.output
    assert "finish=stop" in result.output


def test_cmd_log_missing(config):
    result = CliRunner().invoke(cmd_log, ["--no-follow"], obj=config)
    assert result.exit_code != 0
    assert "log_content" in result.output


def test_cmd_log_ops(config):
    log_dir = config.root_dir / ".hostai-cache"
    log_dir.mkdir(parents=True)
    (log_dir / "proxy.log").write_text("INFO proxy ready\n")
    result = CliRunner().invoke(cmd_log, ["--no-follow", "--ops"], obj=config)
    assert result.exit_code == 0
    assert "INFO proxy ready" in result.output
