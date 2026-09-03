"""Tests for hostai.commands.results with mocked benchmark directories."""

import json

from click.testing import CliRunner

from hostai.commands.results import _load_rows, _render_csv, _render_table, cmd_results


def _make_run(runs_dir, name, label="coding-smoke"):
    bench_dir = runs_dir / name / "benchmarks" / "bench-1"
    bench_dir.mkdir(parents=True)
    metrics = {
        "label": label,
        "ok": True,
        "session": {"profile": "test", "gpu": "RTX 4090", "ctx_size": 32768},
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        "performance": {
            "prompt_tps": 100.0,
            "decode_tps": 50.0,
            "ttft_s": 0.5,
            "cache_hit_rate": 0.25,
            "draft_accept_rate": 0.8,
            "request_compute_cost_usd": 0.0001,
        },
        "gpu_summary": {"peak_memory_mib": 20480, "avg_power_w": 300},
    }
    (bench_dir / "metrics.json").write_text(json.dumps(metrics))
    return bench_dir


def test_load_rows(tmp_path):
    _make_run(tmp_path, "run-1")
    rows = _load_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["run"] == "run-1"
    assert rows[0]["label"] == "coding-smoke"


def test_load_rows_missing_dir(tmp_path):
    assert _load_rows(tmp_path / "does-not-exist") == []


def test_render_csv():
    rows = [{"run": "run-1", "bench": "bench-1", "label": "test"}]
    out = _render_csv(rows)
    assert "run,bench,label" in out
    assert "run-1" in out


def test_render_table_runs_without_error():
    rows = [
        {
            "run": "run-1",
            "bench": "bench-1",
            "label": "test",
            "profile": "test",
            "gpu": "RTX 4090",
            "ctx": 32768,
            "prompt_tok": 10,
            "prompt_tps": 100.0,
            "output_tok": 2,
            "decode_tps": 50.0,
            "ttft_s": 0.5,
            "cache_hit": 0.25,
            "cache_n": 0,
            "mtp_accept": 0.8,
            "peak_vram_mib": 20480,
            "avg_power_w": 300,
            "request_cost_usd": 0.0001,
            "ok": True,
            "path": "/tmp",
        }
    ]
    # Should not raise even with rich Console.
    _render_table(rows)


def test_cmd_results_json(config, tmp_path):
    _make_run(tmp_path, "run-1")
    runner = CliRunner()
    result = runner.invoke(cmd_results, ["--runs-dir", str(tmp_path), "--json"], obj=config)
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1


def test_cmd_results_empty(config, tmp_path):
    runner = CliRunner()
    result = runner.invoke(cmd_results, ["--runs-dir", str(tmp_path)], obj=config)
    assert result.exit_code == 0
    assert "No benchmark results" in result.output


def test_cmd_results_default_runs_dir(config, project_dir):
    runner = CliRunner()
    result = runner.invoke(cmd_results, [], obj=config)
    assert result.exit_code == 0, result.output
    assert "No benchmark results" in result.output
