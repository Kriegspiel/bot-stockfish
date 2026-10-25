"""Synthetic persistence benchmark; never opens production state or credentials."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import tracemalloc

from stockfish_bot.checkpoints import CheckpointStore


def benchmark(count: int) -> dict:
    snapshot = {name: [1 / 64] * 64 for name in ("opponent_king", "opponent_pawns", "opponent_pieces")}
    snapshot["observed_referee_log_size"] = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.json"
        original = {"token": "synthetic", "beliefs": {str(i): snapshot for i in range(count)}}
        path.write_text(json.dumps(original, indent=2, sort_keys=True))
        lifetime_bytes = path.stat().st_size
        started = time.process_time()
        for cursor in range(5):
            state = json.loads(path.read_text())
            state["beliefs"]["0"]["observed_referee_log_size"] = cursor
            path.write_text(json.dumps(state, indent=2, sort_keys=True))
        old_ms = (time.process_time() - started) * 1000 / 5
        store = CheckpointStore(path)
        started = time.perf_counter()
        store.prepare()
        migration_s = time.perf_counter() - started
        tracemalloc.start()
        started = time.process_time()
        wall = time.perf_counter()
        for cursor in range(20):
            store.save_belief("0", {**snapshot, "observed_referee_log_size": cursor})
        new_ms = (time.process_time() - started) * 1000 / 20
        new_wall_ms = (time.perf_counter() - wall) * 1000 / 20
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {"historical_games": count, "legacy_bytes_per_write": lifetime_bytes,
                "checkpoint_bytes_per_write": store.checkpoint_path("0").stat().st_size,
                "legacy_cpu_ms_per_update": old_ms, "new_cpu_ms_per_update": new_ms,
                "new_wall_ms_per_update_including_fsync": new_wall_ms,
                "migration_seconds": migration_s, "new_steady_peak_allocated_bytes": peak,
                "active_cached_games": len(store._active)}


if __name__ == "__main__":
    print(json.dumps([benchmark(count) for count in (100, 10000)], indent=2))
