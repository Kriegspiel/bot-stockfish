from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from stockfish_bot import api, checkpoints
from stockfish_bot.checkpoints import CheckpointStore


def test_migration_preserves_metadata_and_restart_evidence(tmp_path):
    path = tmp_path / "state.json"
    legacy = {"token": "test-token", "config_migrations": {"v1": True},
              "last_bot_game_join_attempt_at": 123, "beliefs": {
                  "game-one": {"cursor": 7, "matrix": [0.25, 0.75]},
                  "../../unsafe": {"cursor": 9}}}
    original = json.dumps(legacy, indent=2)
    path.write_text(original)
    store = CheckpointStore(path)
    store.prepare()
    assert store.load_metadata() == {k: v for k, v in legacy.items() if k != "beliefs"}
    assert next(tmp_path.glob("*.bak")).read_text() == original
    assert next(tmp_path.glob("*.bak")).stat().st_mode & 0o777 == 0o600
    assert store._active == {}
    for ref, snapshot in legacy["beliefs"].items():
        assert CheckpointStore(path).load_belief(ref) == snapshot
        assert store.checkpoint_path(ref).parent == store.directory
        assert store.checkpoint_path(ref).stat().st_mode & 0o777 == 0o600


def test_interrupted_migration_retries_without_losing_original(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    original = json.dumps({"token": "test", "beliefs": {"a": {"n": 1}, "b": {"n": 2}}})
    path.write_text(original)
    real_write = checkpoints.write_object

    def interrupted(target, value):
        if value.get("game_ref") == "b":
            raise OSError("simulated disk failure")
        real_write(target, value)

    monkeypatch.setattr(checkpoints, "write_object", interrupted)
    with pytest.raises(OSError):
        CheckpointStore(path).prepare()
    assert path.read_text() == original
    monkeypatch.setattr(checkpoints, "write_object", real_write)
    store = CheckpointStore(path)
    assert store.load_belief("a") == {"n": 1}
    assert store.load_belief("b") == {"n": 2}
    assert store.load_metadata() == {"token": "test"}


def test_failed_atomic_replace_keeps_disk_and_cache_at_previous_checkpoint(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path / "state.json")
    store.save_belief("a", {"cursor": 1})

    def fail(*args):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(checkpoints.os, "replace", fail)
    with pytest.raises(OSError):
        store.save_belief("a", {"cursor": 2})
    assert store.load_belief("a") == {"cursor": 1}
    assert CheckpointStore(store.state_path).load_belief("a") == {"cursor": 1}
    assert not list(store.directory.glob(".checkpoint-*"))


def test_hot_path_never_reads_lifetime_state_or_rewrites_other_games(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"token": "test", "beliefs": {str(i): {"cursor": i} for i in range(100)}}))
    store = CheckpointStore(path)
    store.prepare()
    untouched = store.checkpoint_path("99").read_bytes()
    metadata = path.read_bytes()
    real_read = checkpoints.read_object
    reads = []

    def tracked_read(target):
        reads.append(target)
        assert target != path
        return real_read(target)

    monkeypatch.setattr(checkpoints, "read_object", tracked_read)
    assert store.load_belief("0") == {"cursor": 0}
    store.save_belief("0", {"cursor": 1})
    assert store.load_belief("0") == {"cursor": 1}
    assert reads == [store.checkpoint_path("0")]
    assert store.checkpoint_path("99").read_bytes() == untouched
    assert path.read_bytes() == metadata
    assert set(store._active) == {"0"}
    store.release("0")
    assert store._active == {}
    assert store.checkpoint_path("0").exists()


def test_concurrent_games_and_metadata_updates_are_independent(tmp_path):
    store = CheckpointStore(tmp_path / "state.json")

    def game(ref):
        for n in range(3):
            store.save_belief(ref, {"cursor": n})
        store.release(ref)

    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(game, str(i)) for i in range(3)]
        jobs.append(pool.submit(store.save_metadata, {"token": "test", "cooldown": 123}))
        for job in jobs:
            job.result()
    fresh = CheckpointStore(store.state_path)
    assert fresh.load_metadata() == {"token": "test", "cooldown": 123}
    for i in range(3):
        assert fresh.load_belief(str(i)) == {"cursor": 2}
    assert store._active == {}


@pytest.mark.parametrize("state", ["active", "waiting", "completed"])
def test_cleanup_deletes_only_authoritatively_completed_games(tmp_path, monkeypatch, state):
    monkeypatch.setattr(api, "STATE_PATH", tmp_path / "state.json")
    store = api.checkpoint_store()
    store.save_belief("a", {"cursor": 1})
    store.release("a")
    monkeypatch.setattr(api, "get_json", lambda path: {"state": state})
    api.prune_one_checkpoint()
    assert store.checkpoint_path("a").exists() == (state != "completed")
    assert store._active == {}


def test_cleanup_unknown_game_preserves_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "STATE_PATH", tmp_path / "state.json")
    store = api.checkpoint_store()
    store.save_belief("a", {"cursor": 1})

    def unavailable(path):
        raise api.requests.HTTPError("not found")

    monkeypatch.setattr(api, "get_json", unavailable)
    with pytest.raises(api.requests.HTTPError):
        api.prune_one_checkpoint()
    assert store.checkpoint_path("a").exists()


def test_unchanged_snapshot_does_not_write_again(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path / "state.json")
    store.save_belief("a", {"cursor": 1})
    monkeypatch.setattr(checkpoints, "write_object", lambda *args: pytest.fail("unchanged checkpoint rewritten"))
    store.save_belief("a", {"cursor": 1})


def test_invalid_legacy_file_is_not_overwritten(tmp_path):
    path = tmp_path / "state.json"
    original = '{"token":"private",broken'
    path.write_text(original)
    with pytest.raises(ValueError, match="Invalid checkpoint JSON") as exc:
        CheckpointStore(path).save_metadata({})
    assert "private" not in str(exc.value)
    assert path.read_text() == original


def test_corrupt_game_checkpoint_does_not_silently_reset_belief(tmp_path):
    store = CheckpointStore(tmp_path / "state.json")
    store.save_belief("a", {"cursor": 7})
    store.release("a")
    store.checkpoint_path("a").write_text("{}")
    with pytest.raises(ValueError, match="Invalid game checkpoint"):
        store.load_belief("a")
