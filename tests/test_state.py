"""Tests for hostai.state persistence."""

from hostai.state import State, init_run_dir, runs_dir, state_dir


def test_state_load_and_save(project_dir):
    state_file = project_dir / ".hostai-vast" / "state.json"
    state = State(state_file, {"instance_id": 42, "profile": "test"})
    state.save()

    loaded = State.load(state_file)
    assert loaded.instance_id == 42
    assert loaded.profile == "test"
    assert loaded.exists is True


def test_state_get_set_delete(project_dir):
    state = State(project_dir / "state.json")
    state.set("foo", "bar")
    assert state.get("foo") == "bar"
    state.delete("foo")
    assert state.get("foo") is None


def test_state_public_dict_masks_secrets(project_dir):
    state = State(
        project_dir / "state.json",
        {"api_key": "secret", "tunnel_pid": 123, "profile": "test"},
    )
    public = state.public_dict()
    assert "api_key" not in public
    assert "tunnel_pid" not in public
    assert public["profile"] == "test"


def test_state_properties(project_dir):
    state = State(project_dir / "state.json")
    state.instance_id = 42
    state.status = "running"
    state.local_port = 18080
    assert state.instance_id == 42
    assert state.status == "running"
    assert state.local_port == 18080


def test_state_run_dir_conversion(project_dir):
    state = State(project_dir / "state.json", {"run_dir": str(project_dir / ".hostai-runs" / "run-1")})
    assert state.run_dir == project_dir / ".hostai-runs" / "run-1"


def test_state_dir_helpers(project_dir):
    assert state_dir(project_dir) == project_dir / ".hostai-vast"
    assert runs_dir(project_dir) == project_dir / ".hostai-runs"
    rdir = runs_dir(project_dir)
    run = init_run_dir(rdir, "test")
    assert run.parent == rdir
    assert rdir.exists()
