from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import time
from typing import Any

from generals_training import TRAINING_MAX_TURNS, TrainingEngine


EVALUATION_SOURCE = "mechanism_audit_6ai_20x20"


def _evaluation_task(
    payload: tuple[
        list[dict[str, float]],
        int,
        int,
        int,
        list[int],
        int,
    ],
) -> dict[str, Any]:
    (
        genomes,
        seed,
        map_size,
        max_turns,
        genome_indices,
        player_count,
    ) = payload
    return TrainingEngine.simulate_episode(
        genomes,
        seed,
        map_size=map_size,
        max_turns=max_turns,
        mode="ffa",
        genome_indices=genome_indices,
        player_count=player_count,
        source=EVALUATION_SOURCE,
        collect_diagnostics=True,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _summarize(
    items: list[dict[str, Any]],
    *,
    elapsed_seconds: float,
    player_count: int,
    map_size: int,
    max_turns: int,
    seed: int,
) -> dict[str, Any]:
    rows = [
        result
        for item in items
        for result in item.get("results", [])
    ]
    match_count = max(1, len(items))
    resolved = sum(1 for item in items if item.get("winner") is not None)
    contact_matches = sum(
        1
        for item in items
        if any(
            result.get("first_contact_turn") is not None
            for result in item.get("results", [])
        )
    )
    battle_matches = sum(
        1
        for item in items
        if any(
            result.get("first_attrition_turn") is not None
            for result in item.get("results", [])
        )
    )
    total_survival_turns = sum(
        int(result.get("survival_turns", 0))
        for result in rows
    )
    total_idle_turns = sum(
        int(result.get("idle_turns", 0))
        for result in rows
    )
    state_turns: Counter[str] = Counter()
    mission_turns: Counter[str] = Counter()
    gather_mode_turns: Counter[str] = Counter()
    for result in rows:
        state_turns.update(result.get("strategy_state_turns", {}))
        mission_turns.update(result.get("mission_turns", {}))
        gather_mode_turns.update(result.get("gather_mode_turns", {}))

    observed = {
        "home_guard": sum(
            int(result.get("home_guard_active_turns", 0)) > 0
            for result in rows
        ),
        "development": sum(
            int(result.get("development_turns", 0)) > 0
            for result in rows
        ),
        "border_muster": sum(
            int(result.get("border_muster_turns", 0)) > 0
            for result in rows
        ),
        "all_in": sum(
            int(result.get("all_in_turns", 0)) > 0
            for result in rows
        ),
        "capital_depth_ramp": sum(
            int(result.get("capital_depth_short_turns", 0)) > 0
            for result in rows
        ),
    }
    idle_rate = total_idle_turns / max(1, total_survival_turns)
    contact_rate = contact_matches / match_count
    battle_rate = battle_matches / match_count
    required_missions = {
        "defend_home_city",
        "defend_territory",
        "attack_enemy_city",
        "attack_enemy_land",
        "attack_neutral_city",
        "explore_expand",
    }
    observed_states = set(state_turns)
    observed_missions = set(mission_turns)
    checks = {
        "contact_coverage": contact_rate >= 0.95,
        "combat_coverage": battle_rate >= 0.95,
        "no_idle_with_movable_army": idle_rate <= 0.001,
        "four_strategy_states": {
            "attack",
            "defense",
            "development",
            "exploration",
        }.issubset(observed_states),
        "six_mission_categories": required_missions.issubset(
            observed_missions
        ),
        "home_guard_executed": observed["home_guard"] > 0,
        "development_executed": observed["development"] > 0,
        "capital_depth_task_executed": observed["capital_depth_ramp"] > 0,
        "major_battle_executed": sum(
            int(result.get("major_battle_turns", 0))
            for result in rows
        )
        > 0,
    }
    return {
        "generated_at": time.time(),
        "config": {
            "matches": len(items),
            "players_per_match": player_count,
            "map_size": map_size,
            "max_turns": max_turns,
            "seed": seed,
            "source": EVALUATION_SOURCE,
        },
        "matches": {
            "requested": len(items),
            "resolved": resolved,
            "resolution_rate": round(resolved / match_count, 4),
            "contact_matches": contact_matches,
            "contact_rate": round(contact_rate, 4),
            "battle_matches": battle_matches,
            "battle_rate": round(battle_rate, 4),
            "average_turns": round(
                sum(int(item.get("turn", 0)) for item in items)
                / match_count,
                2,
            ),
            "max_turns_observed": max(
                (int(item.get("turn", 0)) for item in items),
                default=0,
            ),
            "average_major_battle_turns_per_player": round(
                sum(
                    int(result.get("major_battle_turns", 0))
                    for result in rows
                )
                / max(1, len(rows)),
                3,
            ),
        },
        "ai_turns": {
            "sampled_player_turns": total_survival_turns,
            "idle_turns": total_idle_turns,
            "idle_rate": round(idle_rate, 6),
            "strategy_state_turns": dict(state_turns.most_common()),
            "mission_turns": dict(mission_turns.most_common()),
            "gather_mode_turns": dict(gather_mode_turns.most_common()),
        },
        "mechanism_evidence": {
            "players_with_home_guard": observed["home_guard"],
            "players_with_attrition_development": observed["development"],
            "players_with_border_muster": observed["border_muster"],
            "players_with_all_in": observed["all_in"],
            "players_with_capital_depth_task": observed[
                "capital_depth_ramp"
            ],
            "land_gather_seen": "land" in gather_mode_turns,
            "city_gather_seen": "city" in gather_mode_turns,
        },
        "checks": checks,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "matches_per_second": round(
            len(items) / max(0.001, elapsed_seconds),
            4,
        ),
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run and ingest a deterministic multi-player AI mechanism audit."
        )
    )
    parser.add_argument("--matches", type=_positive_int, default=100)
    parser.add_argument("--players", type=_positive_int, default=6)
    parser.add_argument("--map-size", type=_positive_int, default=20)
    parser.add_argument(
        "--max-turns",
        type=_positive_int,
        default=TRAINING_MAX_TURNS,
    )
    parser.add_argument("--workers", type=_positive_int, default=0)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--summary-name",
        default="mechanism_evaluation_6ai_20x20.json",
    )
    parser.add_argument("--no-ingest", action="store_true")
    args = parser.parse_args()

    if not 2 <= args.players <= 12:
        parser.error("--players must be between 2 and 12")
    if args.map_size < 20:
        parser.error("--map-size must be at least 20")

    data_dir = (
        Path(args.data_dir)
        if args.data_dir
        else Path(__file__).resolve().with_name("training_data")
    )
    engine = TrainingEngine(
        data_dir,
        seed=args.seed,
        population_size=12,
        map_size=args.map_size,
        max_turns=args.max_turns,
        parallel_workers=1,
        live_render=False,
    )
    if len(engine.population) < args.players:
        raise RuntimeError("training population is smaller than player count")

    workers = (
        max(1, int(args.workers))
        if args.workers
        else max(1, min(8, os.cpu_count() or 2))
    )
    rng = random.Random(args.seed)
    payloads: list[
        tuple[
            list[dict[str, float]],
            int,
            int,
            int,
            list[int],
            int,
        ]
    ] = []
    genomes = [genome.to_dict() for genome in engine.population]
    for match_index in range(args.matches):
        indices = sorted(
            rng.sample(range(len(engine.population)), args.players)
        )
        payloads.append(
            (
                genomes,
                args.seed + match_index * 7919,
                args.map_size,
                args.max_turns,
                indices,
                args.players,
            )
        )

    before_generation = engine.generation
    before_completed = engine.completed_matches
    started = time.perf_counter()
    items: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as pool:
        futures = [pool.submit(_evaluation_task, payload) for payload in payloads]
        for completed, future in enumerate(as_completed(futures), 1):
            items.append(future.result())
            if completed % max(1, args.matches // 10) == 0:
                print(
                    f"progress {completed}/{args.matches}",
                    flush=True,
                )
    items.sort(key=lambda item: int(item["seed"]))
    elapsed = time.perf_counter() - started

    summary = _summarize(
        items,
        elapsed_seconds=elapsed,
        player_count=args.players,
        map_size=args.map_size,
        max_turns=args.max_turns,
        seed=args.seed,
    )
    if not args.no_ingest:
        engine.record_training_batch(items, batch_elapsed=elapsed)
        summary["training_ingestion"] = {
            "ingested": True,
            "generation_before": before_generation,
            "generation_after": engine.generation,
            "completed_matches_before": before_completed,
            "completed_matches_after": engine.completed_matches,
            "history_file": str(data_dir / "history.jsonl"),
        }
    else:
        summary["training_ingestion"] = {"ingested": False}

    summary_path = data_dir / args.summary_name
    _atomic_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "matches": summary["matches"],
                "checks": summary["checks"],
                "training_ingestion": summary["training_ingestion"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if all(summary["checks"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
