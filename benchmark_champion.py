from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import statistics
from typing import Any

from generals_ai import AIGenome
from generals_training import TrainingEngine


MATCH_TYPES = (
    ("ffa4", 4, 25, 12),
    ("ffa12", 12, 39, 8),
    ("ffa20", 20, 55, 5),
    ("duel", 2, 25, 12),
)


def _task(
    payload: tuple[dict[str, float], str, int, int, int, int, str],
) -> dict[str, Any]:
    genome, mode, players, map_size, seed, max_turns, source = payload
    genome_dicts = [dict(genome) for _ in range(players)]
    if mode == "duel":
        return TrainingEngine.simulate_episode(
            genome_dicts,
            seed,
            map_size=map_size,
            max_turns=max_turns,
            mode="duel",
            genome_indices=[0, 0],
            player_count=2,
            source=source,
            collect_diagnostics=True,
        )
    return TrainingEngine.simulate_episode(
        genome_dicts,
        seed,
        map_size=map_size,
        max_turns=max_turns,
        mode="ffa",
        genome_indices=list(range(players)),
        player_count=players,
        source=source,
        collect_diagnostics=True,
    )


def _summarize(matches: list[dict[str, Any]]) -> dict[str, float]:
    rows = [row for match in matches for row in match["results"]]

    def mean(name: str) -> float:
        return statistics.fmean(float(row.get(name, 0)) for row in rows)

    return {
        "samples": float(len(rows)),
        "tiles": round(mean("tiles"), 2),
        "peak_tiles": round(mean("peak_tiles"), 2),
        "castles_captured": round(mean("castles_captured"), 3),
        "attrition_for": round(mean("attrition_for"), 2),
        "attrition_against": round(mean("attrition_against"), 2),
        "armies_created": round(mean("armies_created"), 3),
        "peak_army_units": round(mean("peak_army_units"), 3),
        "survival_turns": round(mean("survival_turns"), 2),
        "home_guard_active_turns": round(
            mean("home_guard_active_turns"),
            2,
        ),
        "home_guard_deficit_turns": round(
            mean("home_guard_deficit_turns"),
            2,
        ),
        "battle_turns": round(mean("battle_turns"), 2),
        "major_battle_turns": round(mean("major_battle_turns"), 3),
        "first_contact_turn": round(
            statistics.fmean(
                float(row["first_contact_turn"])
                for row in rows
                if row.get("first_contact_turn") is not None
            )
            if any(row.get("first_contact_turn") is not None for row in rows)
            else -1.0,
            2,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--genome",
        default="training_data/champion.json",
    )
    parser.add_argument("--output")
    parser.add_argument("--max-turns", type=int, default=900)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.genome)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        genome = payload.get("best_match", {}).get("results", [{}])[0].get(
            "genome",
            payload.get("genome", {}),
        )
    else:
        genome = {}
    genome = AIGenome.from_mapping(genome).to_dict()

    payloads = []
    for name, players, map_size, seeds in MATCH_TYPES:
        for seed in range(1, seeds + 1):
            payloads.append(
                (
                    genome,
                    name,
                    players,
                    map_size,
                    seed,
                    args.max_turns,
                    f"champion_eval_{name}",
                )
            )
    workers = args.workers or None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        items = list(pool.map(_task, payloads))

    summary: dict[str, Any] = {"genome": genome}
    for name, _players, _map_size, _seeds in MATCH_TYPES:
        matches = [
            item for item in items if item.get("source") == f"champion_eval_{name}"
        ]
        summary[name] = _summarize(matches)
    summary["overall"] = _summarize(items)

    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
