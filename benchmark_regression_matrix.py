from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
from time import perf_counter

from generals_ai import (
    AI_INDEPENDENT_ARMY_TROOP_DIVISOR,
    GeneralsAI,
    independent_army_limit,
)
from generals_core import Board, GENERAL, Move, TurnReport


def _is_enemy(owner: int, player: int) -> bool:
    return owner >= 0 and owner != player


def _run_game(payload: tuple[str, int, int, int, int, str]) -> dict[str, object]:
    name, seed, size, players, max_turns, difficulty = payload
    board = Board(size, size, players, seed)
    ais = [
        GeneralsAI(player, difficulty, seed * 10_000 + player)
        for player in range(players)
    ]
    started = perf_counter()
    illegal_moves = 0
    legal_attacks = 0
    main_idle = 0
    army_idle = 0
    main_targeted_armies = 0
    duplicate_move_sources = 0
    observed_army_moves = 0
    first_contact: int | None = None
    first_attrition: int | None = None
    battle_turns = 0
    attrition_total = 0
    max_total_army = 0
    max_army_limit = 0
    max_army_count = 0
    created_armies = 0
    active_player_turns = 0
    post_200_active_player_turns = 0
    post_200_near_limit_turns = 0
    army_count_samples = 0
    army_limit_samples = 0
    army_slot_ratio_samples = 0.0
    owned_tile_turns = 0
    max_army_count_before_200 = 0
    expanded_players: set[int] = set()
    fail_windows: list[int] = []
    rolling_attrition: deque[int] = deque()
    rolling_total = 0
    window_best_rolling = 0
    window_total_army = sum(
        board.stats(player)["armies"]
        for player in board.active_players
    )
    finished_turn: int | None = None

    for turn in range(1, max_turns + 1):
        for ai in ais:
            if ai.player not in board.active_players:
                continue
            controller = ai.army_controller
            before_ids = {army.unit_id for army in controller.armies}
            controller.begin_turn(board, ai)
            for army in controller.armies:
                army.force_disband = False
            controller.review_ai_armies(board, ai)
            controller.maybe_create_auto(board, ai)
            created_armies += len(
                {army.unit_id for army in controller.armies}
                - before_ids
            )

            owned = ai._positions_by_owner(board)[ai.player]
            total_army = sum(board.tile(*position).army for position in owned)
            army_limit = independent_army_limit(
                total_army,
                AI_INDEPENDENT_ARMY_TROOP_DIVISOR,
            )
            if total_army > max_total_army:
                max_total_army = total_army
                max_army_limit = army_limit
            if army_limit > 4 and total_army >= 8_000:
                expanded_players.add(ai.player)
            army_count = controller.count
            max_army_count = max(max_army_count, army_count)
            if turn < 200:
                max_army_count_before_200 = max(
                    max_army_count_before_200,
                    army_count,
                )
            active_player_turns += 1
            army_count_samples += army_count
            army_limit_samples += army_limit
            army_slot_ratio_samples += army_count / max(1, army_limit)
            owned_tile_turns += len(owned)
            if turn >= 200:
                post_200_active_player_turns += 1
                if army_count >= max(1, math.ceil(army_limit * 0.75)):
                    post_200_near_limit_turns += 1

        report = TurnReport()
        for phase in range(2):
            moves: list[Move] = []
            pending_sources = {
                (pending.sx, pending.sy)
                for pending in board.pending_moves
                if pending.owner in board.active_players
            }
            for ai in ais:
                if ai.player not in board.active_players:
                    continue
                controller = ai.army_controller
                before = {
                    army.unit_id: army.position
                    for army in controller.armies
                }
                waiting = {
                    army.unit_id
                    for army in controller.armies
                    if army.transit_target is not None
                }
                army_positions = set(before.values())
                planned = ai.plan_moves(board, phase=phase)
                army_sources = {
                    move.source
                    for move in planned
                    if move.source in army_positions
                }
                observed_army_moves += len(army_sources)
                surviving = {
                    army.unit_id
                    for army in controller.armies
                }
                for unit_id, position in before.items():
                    if (
                        unit_id in surviving
                        and unit_id not in waiting
                        and position not in army_sources
                    ):
                        army_idle += 1

                current_army_positions = controller.positions
                occupied_after = controller.occupied_positions(board)
                main_moves = [
                    move
                    for move in planned
                    if move.source not in current_army_positions
                ]
                main_targeted_armies += sum(
                    1
                    for move in main_moves
                    if move.target in occupied_after
                )
                phase_sources = Counter(move.source for move in planned)
                duplicate_move_sources += sum(
                    count - 1
                    for count in phase_sources.values()
                    if count > 1
                )
                if not main_moves:
                    field_movable = False
                    for x, y in ai._positions_by_owner(board)[ai.player]:
                        if (x, y) in current_army_positions or (x, y) in pending_sources:
                            continue
                        tile = board.tile(x, y)
                        if tile.terrain != GENERAL and tile.army > 1:
                            field_movable = True
                            break
                    if field_movable:
                        main_idle += 1

                for move in planned:
                    source = board.tile(move.sx, move.sy)
                    if (
                        source.owner != ai.player
                        or move.amount < 1
                        or move.amount > source.army - 1
                        or not board.move_legal(move, ai.player)
                    ):
                        illegal_moves += 1
                    target = board.tile(move.tx, move.ty)
                    if _is_enemy(target.owner, ai.player):
                        legal_attacks += 1
                moves.extend(planned)
            report.merge(board.resolve_movement(moves))
            if board.winner is not None:
                break

        if first_contact is None:
            for ai in ais:
                if ai.player not in board.active_players:
                    continue
                visible = board.visibility(ai.player)
                if any(
                    _is_enemy(board.tile(*position).owner, ai.player)
                    for position in visible
                ):
                    first_contact = turn
                    break

        pair_attrition = sum((report.attrition_by_pair or {}).values())
        rolling_attrition.append(pair_attrition)
        rolling_total += pair_attrition
        if len(rolling_attrition) > 10:
            rolling_total -= rolling_attrition.popleft()
        window_best_rolling = max(window_best_rolling, rolling_total)
        battle_turns += int(pair_attrition > 0)
        attrition_total += pair_attrition
        if first_attrition is None and pair_attrition > 0:
            first_attrition = turn

        board.finish_turn(report)
        if board.winner is not None:
            finished_turn = turn
        if turn % 500 == 0:
            print(
                f"[{name}] seed={seed} progress_turn={turn} "
                f"active={len(board.active_players)} "
                f"armies={max_army_count}/{max_army_limit}",
                flush=True,
            )
        if turn % 100 == 0:
            required = max(
                24,
                math.ceil(window_total_army * 0.01),
            )
            if window_best_rolling < required:
                fail_windows.append(turn)
            window_total_army = sum(
                board.stats(player)["armies"]
                for player in board.active_players
            )
            rolling_attrition.clear()
            rolling_total = 0
            window_best_rolling = 0
        if board.winner is not None:
            break

    elapsed = perf_counter() - started
    passed = (
        illegal_moves == 0
        and main_idle == 0
        and army_idle == 0
        and main_targeted_armies == 0
        and duplicate_move_sources == 0
        and first_contact is not None
        and first_attrition is not None
        and not fail_windows
    )
    return {
        "name": name,
        "seed": seed,
        "size": size,
        "players": players,
        "max_turns": max_turns,
        "turn": board.turn - 1,
        "winner": board.winner,
        "finished": board.winner is not None,
        "finished_turn": finished_turn,
        "passed": passed,
        "elapsed_seconds": round(elapsed, 3),
        "illegal_moves": illegal_moves,
        "legal_attacks": legal_attacks,
        "main_idle": main_idle,
        "army_idle": army_idle,
        "main_targeted_armies": main_targeted_armies,
        "duplicate_move_sources": duplicate_move_sources,
        "observed_army_moves": observed_army_moves,
        "first_contact": first_contact,
        "first_attrition": first_attrition,
        "battle_turns": battle_turns,
        "attrition_total": attrition_total,
        "fail_windows": fail_windows,
        "max_total_army": max_total_army,
        "max_army_limit": max_army_limit,
        "max_army_count": max_army_count,
        "created_armies": created_armies,
        "active_player_turns": active_player_turns,
        "post_200_active_player_turns": post_200_active_player_turns,
        "army_creation_density_per_1000_player_turns": round(
            created_armies * 1_000 / max(1, active_player_turns),
            3,
        ),
        "average_army_count": round(
            army_count_samples / max(1, active_player_turns),
            3,
        ),
        "average_army_limit": round(
            army_limit_samples / max(1, active_player_turns),
            3,
        ),
        "average_army_slot_utilization": round(
            army_slot_ratio_samples / max(1, active_player_turns),
            4,
        ),
        "army_density_per_1000_owned_tiles": round(
            army_count_samples * 1_000 / max(1, owned_tile_turns),
            3,
        ),
        "post_200_near_limit_ratio": round(
            post_200_near_limit_turns / max(1, post_200_active_player_turns),
            4,
        ),
        "max_army_count_before_200": max_army_count_before_200,
        "army_limit_expanded": bool(expanded_players),
        "expanded_players": len(expanded_players),
        "army_count_expanded": max_army_count > 4,
    }


def _run_series(
    name: str,
    size: int,
    players: int,
    seeds: list[int],
    max_turns: int,
    difficulty: str,
    workers: int,
) -> list[dict[str, object]]:
    payloads = [
        (name, seed, size, players, max_turns, difficulty)
        for seed in seeds
    ]
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_game, payload): payload[1]
            for payload in payloads
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[{name}] seed={result['seed']} "
                f"pass={result['passed']} turn={result['turn']} "
                f"contact={result['first_contact']} "
                f"attrition={result['first_attrition']} "
                f"army_idle={result['army_idle']} "
                f"max_army={result['max_army_count']}/{result['max_army_limit']} "
                f"army_density={result['army_creation_density_per_1000_player_turns']} "
                f"near_limit={result['post_200_near_limit_ratio']} "
                f"early_max={result['max_army_count_before_200']} "
                f"time={result['elapsed_seconds']}s",
                flush=True,
            )
    results.sort(key=lambda item: int(item["seed"]))
    return results


def _summarize(results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "games": len(results),
        "passed": sum(bool(result["passed"]) for result in results),
        "winner_games": sum(bool(result["finished"]) for result in results),
        "illegal_moves": sum(int(result["illegal_moves"]) for result in results),
        "main_idle": sum(int(result["main_idle"]) for result in results),
        "army_idle": sum(int(result["army_idle"]) for result in results),
        "main_targeted_armies": sum(
            int(result["main_targeted_armies"])
            for result in results
        ),
        "duplicate_move_sources": sum(
            int(result["duplicate_move_sources"])
            for result in results
        ),
        "contact_games": sum(
            result["first_contact"] is not None
            for result in results
        ),
        "attrition_games": sum(
            result["first_attrition"] is not None
            for result in results
        ),
        "max_total_army": max(
            (int(result["max_total_army"]) for result in results),
            default=0,
        ),
        "max_army_limit": max(
            (int(result["max_army_limit"]) for result in results),
            default=0,
        ),
        "max_army_count": max(
            (int(result["max_army_count"]) for result in results),
            default=0,
        ),
        "created_armies": sum(
            int(result["created_armies"])
            for result in results
        ),
        "army_creation_density_per_1000_player_turns": round(
            sum(int(result["created_armies"]) for result in results)
            * 1_000
            / max(
                1,
                sum(int(result["active_player_turns"]) for result in results),
            ),
            3,
        ),
        "average_army_count": round(
            sum(
                float(result["average_army_count"])
                * int(result["active_player_turns"])
                for result in results
            )
            / max(
                1,
                sum(int(result["active_player_turns"]) for result in results),
            ),
            3,
        ),
        "average_army_limit": round(
            sum(
                float(result["average_army_limit"])
                * int(result["active_player_turns"])
                for result in results
            )
            / max(
                1,
                sum(int(result["active_player_turns"]) for result in results),
            ),
            3,
        ),
        "average_army_slot_utilization": round(
            sum(
                float(result["average_army_slot_utilization"])
                * int(result["active_player_turns"])
                for result in results
            )
            / max(
                1,
                sum(int(result["active_player_turns"]) for result in results),
            ),
            4,
        ),
        "post_200_near_limit_ratio": round(
            sum(
                float(result["post_200_near_limit_ratio"])
                * int(result["post_200_active_player_turns"])
                for result in results
            )
            / max(
                1,
                sum(
                    int(result["post_200_active_player_turns"])
                    for result in results
                ),
            ),
            4,
        ),
        "max_army_count_before_200": max(
            (int(result["max_army_count_before_200"]) for result in results),
            default=0,
        ),
        "games_with_expanded_army_limit": sum(
            bool(result["army_limit_expanded"])
            for result in results
        ),
        "expanded_players": sum(
            int(result["expanded_players"])
            for result in results
        ),
        "elapsed_seconds": round(
            sum(float(result["elapsed_seconds"]) for result in results),
            3,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--completion-only", action="store_true")
    args = parser.parse_args()

    workers = args.workers or 5
    if args.quick:
        large_seeds = [1]
        large_turns = 100
        completion_turns = 1_000
        small_seeds = [1]
        medium_seeds = [1]
        small_turns = 100
        medium_turns = 100
    else:
        large_seeds = [1, 2, 3, 4, 5]
        large_turns = 1_000
        completion_turns = 50_000
        small_seeds = [11, 12, 13, 14, 15]
        medium_seeds = [21, 22, 23, 24, 25]
        small_turns = 1_000
        medium_turns = 1_000

    started = perf_counter()
    if args.completion_only:
        large: list[dict[str, object]] = []
        small: list[dict[str, object]] = []
        medium: list[dict[str, object]] = []
    else:
        large = _run_series(
            "large_75_36ai_1000",
            75,
            36,
            large_seeds,
            large_turns,
            args.difficulty,
            min(workers, len(large_seeds)),
        )
    completion = _run_series(
        "large_75_36ai_completion",
        75,
        36,
        [6],
        completion_turns,
        args.difficulty,
        1,
    )
    if not args.completion_only:
        small = _run_series(
            "small_20_4ai_1000",
            20,
            4,
            small_seeds,
            small_turns,
            args.difficulty,
            min(workers, len(small_seeds)),
        )
        medium = _run_series(
            "medium_51_16ai_1000",
            51,
            16,
            medium_seeds,
            medium_turns,
            args.difficulty,
            min(workers, len(medium_seeds)),
        )

    sections = (
        {"large_75_36ai_completion": completion}
        if args.completion_only
        else {
            "large_75_36ai_1000": large,
            "large_75_36ai_completion": completion,
            "small_20_4ai_1000": small,
            "medium_51_16ai_1000": medium,
        }
    )
    summary = {
        name: _summarize(results)
        for name, results in sections.items()
    }
    completion_passed = bool(
        completion
        and completion[0]["finished"]
        and completion[0]["passed"]
    )
    passed = (
        completion_passed
        if args.completion_only
        else (
            all(bool(result["passed"]) for result in large + small + medium)
            and completion_passed
        )
    )
    report = {
        "passed": passed,
        "difficulty": args.difficulty,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "summary": summary,
        "results": sections,
    }
    output_dir = Path("training_data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / (
        "regression_completion_latest.json"
        if args.completion_only
        else "regression_matrix_latest.json"
    )
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={output_path.resolve()}")
    print(f"PASS={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
