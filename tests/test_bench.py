"""Tests for hostai.commands.bench with mocked API and SSH."""

import json
from pathlib import Path
from unittest import mock

import click
from click.testing import CliRunner

from hostai.commands.bench import cmd_bench


def bench_state(project_dir):
    from hostai.state import State

    state = State(
        project_dir / ".hostai-vast" / "state.json",
        {
            "instance_id": 12345,
            "profile": "test",
            "model": "model.gguf",
            "hf_revision": "rev",
            "ctx_size": 32768,
            "local_port": 18080,
            "api_key": "test-key",
            "dph": 0.5,
            "ssh_url": "ssh://root@10.0.0.1:2222",
            "run_dir": str(project_dir / ".hostai-runs" / "run-1"),
        },
    )
    state.save()
    return state


def test_bench_no_instance(config):
    runner = CliRunner()
    result = runner.invoke(cmd_bench, [], obj=config)
    assert result.exit_code != 0
    assert "no running instance" in result.output


def test_bench_happy_path(config, project_dir):
    state = bench_state(project_dir)
    Path(state.run_dir).mkdir(parents=True)

    chunk = {
        "choices": [{"delta": {"content": "hi"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }

    chat_mock = mock.Mock(return_value=iter([chunk]))

    class FakeClient:
        health = mock.Mock(return_value=True)
        get_metrics_text = mock.Mock(return_value="")
        get_metrics = mock.Mock(return_value={})
        chat = chat_mock

    with mock.patch("hostai.commands.bench.State.load", return_value=state):
        with mock.patch("hostai.commands.bench.ssh.is_tunnel_healthy", return_value=True):
            with mock.patch("hostai.commands.bench.LlamaClient", return_value=FakeClient()):
                with mock.patch("hostai.commands.bench.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout="")):
                    runner = CliRunner()
                    result = runner.invoke(cmd_bench, ["--max-tokens", "10"], obj=config)

    if result.exit_code != 0:
        raise result.exception or AssertionError(result.output)
    assert "prompt:" in result.output
    assert "t/s" in result.output
    bench_dir = list((Path(state.run_dir) / "benchmarks").iterdir())[0]
    assert (bench_dir / "metrics.json").exists()

    # Verify the client was called with the right model/parameters.
    assert chat_mock.call_count == 1
    call = chat_mock.call_args
    assert call.kwargs["max_tokens"] == 10
    assert call.kwargs["temperature"] == 0.7
    assert call.kwargs["stream"] is True
    assert call.args[0][0]["role"] == "user"


def test_bench_api_not_healthy(config, project_dir):
    state = bench_state(project_dir)
    Path(state.run_dir).mkdir(parents=True)

    class FakeClient:
        def health(self):
            return False

    with mock.patch("hostai.commands.bench.State.load", return_value=state):
        with mock.patch("hostai.commands.bench.ssh.is_tunnel_healthy", return_value=True):
            with mock.patch("hostai.commands.bench.LlamaClient", return_value=FakeClient()):
                runner = CliRunner()
                result = runner.invoke(cmd_bench, [], obj=config)

    assert result.exit_code != 0
    assert "not healthy" in result.output


def test_bench_streaming_error(config, project_dir):
    state = bench_state(project_dir)
    Path(state.run_dir).mkdir(parents=True)

    class FakeClient:
        health = mock.Mock(return_value=True)
        get_metrics_text = mock.Mock(return_value="")
        get_metrics = mock.Mock(return_value={})

        def chat(self, *args, **kwargs):
            raise RuntimeError("stream exploded")

    with mock.patch("hostai.commands.bench.State.load", return_value=state):
        with mock.patch("hostai.commands.bench.ssh.is_tunnel_healthy", return_value=True):
            with mock.patch("hostai.commands.bench.LlamaClient", return_value=FakeClient()):
                with mock.patch("hostai.commands.bench.ssh.run_remote", return_value=mock.Mock(returncode=0, stdout="")):
                    runner = CliRunner()
                    result = runner.invoke(cmd_bench, ["--max-tokens", "10"], obj=config)

    assert result.exit_code != 0
    assert "stream exploded" in result.output


def test_bench_invalid_max_tokens(config):
    runner = CliRunner()
    result = runner.invoke(cmd_bench, ["--max-tokens", "0"], obj=config)
    # Validation happens in the API, but cmd_bench doesn't validate it itself,
    # so we just check it does not crash when no state exists.
    assert result.exit_code != 0
    assert "running instance" in result.output
