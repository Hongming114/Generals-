from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time

from generals_training import TrainingEngine


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark persistent 12-AI self-play training."
    )
    parser.add_argument("--matches", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-turns", type=int, default=240)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    temporary = None
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        temporary = tempfile.TemporaryDirectory()
        data_dir = Path(temporary.name)
    engine = TrainingEngine(
        data_dir,
        seed=args.seed,
        max_turns=max(1, args.max_turns),
        parallel_workers=max(1, args.workers),
        live_render=not args.no_render,
    )
    started = time.perf_counter()
    engine.start()
    try:
        while (
            engine.completed_matches < max(1, args.matches)
            and time.perf_counter() - started < max(0.1, args.seconds)
        ):
            time.sleep(0.1)
    finally:
        engine.pause()
        engine.wait_for_stop(timeout=60.0)
        if temporary is not None:
            temporary.cleanup()
    elapsed = time.perf_counter() - started
    snapshot = engine.snapshot()
    rate = snapshot["completed_matches"] / max(0.001, elapsed)
    turns_per_second = (
        snapshot["completed_turns"] / max(0.001, elapsed)
    )
    print(
        {
            "matches": snapshot["completed_matches"],
            "simulated_turns": snapshot["completed_turns"],
            "generation": snapshot["generation"],
            "elapsed_seconds": round(elapsed, 3),
            "matches_per_second": round(rate, 4),
            "matches_per_hour": round(rate * 3600.0, 2),
            "turns_per_second": round(turns_per_second, 3),
            "best_fitness": snapshot["best_fitness"],
            "workers": args.workers,
            "live_render": not args.no_render,
            "training_mode": snapshot.get("training_mode", "ffa"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
