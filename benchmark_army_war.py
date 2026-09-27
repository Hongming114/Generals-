from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import statistics

import generals_ai
from generals_ai import GeneralsAI
from generals_core import Board, TurnReport


def _run_match(
    seed: int,
    width: int,
    height: int,
    players: int,
    max_turns: int,
    difficulty: str,
    eager_creation: bool,
) -> dict[str, object]:
    generals_ai.INDEPENDENT_ARMY_EAGER_CREATION_ENABLED = eager_creation
    board = Board(width, height, players, seed)
    ais = [
        GeneralsAI(player, difficulty, seed * 10_000 + player)
        for player in range(players)
    ]
    created: Counter[int] = Counter()
    armies_created = 0
    army_actions = 0
    battle_turns = 0
    attrition_total = 0
    first_contact: int | None = None
    first_attrition: int | None = None

    for turn in range(1, max_turns + 1):
        for ai in ais:
            if ai.player not in board.active_players:
                continue
            controller = ai.army_controller
            before = {army.unit_id for army in controller.armies}
            controller.begin_turn(board, ai)
            for army in controller.armies:
                army.force_disband = False
            controller.review_ai_armies(board, ai)
            controller.maybe_create_auto(board, ai)
            new_ids = {army.unit_id for army in controller.armies} - before
            if new_ids:
                created[ai.player] += len(new_ids)
                armies_created += len(new_ids)

        report = TurnReport()
        for phase in range(2):
            moves = []
            for ai in ais:
                if ai.player not in board.active_players:
                    continue
                planned = ai.plan_moves(board, phase=phase)
                army_actions += sum(
                    1
                    for move in planned
                    if move.source in ai.army_controller.positions
                )
                moves.extend(planned)
            report.merge(board.resolve_movement(moves))
            if board.winner is not None:
                break

        if first_contact is None:
            for ai in ais:
                visible = board.visibility(ai.player)
                if any(
                    board.tile(*position).owner >= 0
                    and board.tile(*position).owner != ai.player
                    for position in visible
                ):
                    first_contact = turn
                    break

        pair_attrition = sum((report.attrition_by_pair or {}).values())
        battle_turns += int(pair_attrition > 0)
        attrition_total += pair_attrition
        if first_attrition is None and pair_attrition > 0:
            first_attrition = turn

        board.finish_turn(report)
        if board.winner is not None:
            break

    return {
        "seed": seed,
        "turn": board.turn - 1,
        "winner": board.winner,
        "first_contact": first_contact,
        "first_attrition": first_attrition,
        "battle_turns": battle_turns,
        "attrition_total": attrition_total,
        "armies_created": armies_created,
        "army_actions": army_actions,
        "players_with_army": len([value for value in created.values() if value > 0]),
    }


def _task(
    payload: tuple[int, int, int, int, int, str, bool],
) -> dict[str, object]:
    return _run_match(*payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--width", type=int, default=39)
    parser.add_argument("--height", type=int, default=39)
    parser.add_argument("--players", type=int, default=12)
    parser.add_argument("--max-turns", type=int, default=900)
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--legacy-creation", action="store_true")
    args = parser.parse_args()

    payloads = [
        (
            seed,
            args.width,
            args.height,
            args.players,
            args.max_turns,
            args.difficulty,
            not args.legacy_creation,
        )
        for seed in range(1, args.seeds + 1)
    ]
    workers = args.workers or None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_task, payloads))

    def mean(name: str) -> float:
        return round(
            statistics.fmean(
                float(result[name])
                for result in results
            ),
            2,
        )

    contact = [
        int(result["first_contact"] or args.max_turns + 1)
        for result in results
    ]
    attrition = [
        int(result["first_attrition"] or args.max_turns + 1)
        for result in results
    ]
    print(
        f"matches={len(results)} players={args.players} "
        f"map={args.width}x{args.height} difficulty={args.difficulty}"
    )
    print(
        f"contact avg={statistics.fmean(contact):.1f} "
        f"max={max(contact)} | attrition avg={statistics.fmean(attrition):.1f} "
        f"max={max(attrition)}"
    )
    print(
        f"battle_turns={mean('battle_turns')} "
        f"attrition_total={mean('attrition_total')} "
        f"armies_created={mean('armies_created')} "
        f"army_actions={mean('army_actions')} "
        f"players_with_army={mean('players_with_army')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
