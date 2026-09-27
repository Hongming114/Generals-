from __future__ import annotations

import argparse
from collections import deque
import math
from time import perf_counter

from generals_ai import GeneralsAI
from generals_core import Board, TurnReport


def format_result(seed: int, winner: int | None, winner_turn: int | None) -> str:
    return (
        f"seed={seed} winner={winner if winner is not None else 'none'} "
        f"turn={winner_turn if winner_turn is not None else 'none'}"
    )


def run_battle(
    seed: int,
    width: int,
    height: int,
    player_count: int,
    difficulty: str,
    turn_limit: int,
    window_size: int,
    battle_ratio: float,
    min_battle: int,
    rolling_turns: int,
) -> bool:
    board = Board(width, height, player_count, seed)
    ais = [
        GeneralsAI(player, difficulty, seed * 10_000 + player)
        for player in range(player_count)
    ]
    violations = 0
    failed_windows: list[int] = []
    window_total_army = sum(
        board.stats(player)["armies"]
        for player in board.active_players
    )
    rolling: deque[int] = deque()
    rolling_total = 0
    window_best_rolling = 0
    winner: int | None = None
    winner_turn: int | None = None

    for turn in range(1, turn_limit + 1):
        report = TurnReport()
        for phase in range(2):
            moves = []
            for ai in ais:
                if ai.player not in board.active_players:
                    continue
                planned = ai.plan_moves(board, phase=phase)
                for move in planned:
                    source = board.tile(move.sx, move.sy)
                    if (
                        source.owner != ai.player
                        or move.amount < 1
                        or move.amount > source.army - 1
                        or not board.move_legal(move, ai.player)
                    ):
                        violations += 1
                moves.extend(planned)
            report.merge(board.resolve_movement(moves))

        battle_amount = sum((report.attrition_by_pair or {}).values())
        rolling.append(battle_amount)
        rolling_total += battle_amount
        if len(rolling) > rolling_turns:
            rolling_total -= rolling.popleft()
        window_best_rolling = max(window_best_rolling, rolling_total)

        board.finish_turn(report)
        if board.winner is not None:
            winner = board.winner
            winner_turn = turn
            break

        if turn % window_size == 0:
            required = max(
                min_battle,
                math.ceil(window_total_army * battle_ratio),
            )
            if window_best_rolling < required:
                failed_windows.append(turn)
            window_total_army = sum(
                board.stats(player)["armies"]
                for player in board.active_players
            )
            rolling.clear()
            rolling_total = 0
            window_best_rolling = 0

    finished = winner_turn is not None and winner_turn <= turn_limit
    if not finished and winner_turn is None and rolling:
        required = max(
            min_battle,
            math.ceil(window_total_army * battle_ratio),
        )
        if window_best_rolling < required:
            failed_windows.append(turn_limit)

    passed = finished and not failed_windows and violations == 0
    print(
        f"{'PASS' if passed else 'FAIL'} "
        f"{format_result(seed, winner, winner_turn)} "
        f"violations={violations} "
        f"failed_windows={failed_windows or 'none'}"
    )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Long multiplayer battle benchmark: winner within the turn "
            "limit, one scaled major battle per window, and legal attacks."
        ),
    )
    parser.add_argument("--width", type=int, default=45)
    parser.add_argument("--height", type=int, default=45)
    parser.add_argument("--players", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--difficulty",
        choices=("easy", "normal", "hard"),
        default="hard",
    )
    parser.add_argument("--turn-limit", type=int, default=3_000)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--battle-ratio", type=float, default=0.01)
    parser.add_argument("--min-battle", type=int, default=24)
    parser.add_argument("--rolling-turns", type=int, default=10)
    args = parser.parse_args()

    started = perf_counter()
    passed = True
    for seed in args.seeds:
        if not run_battle(
            seed,
            args.width,
            args.height,
            args.players,
            args.difficulty,
            args.turn_limit,
            args.window,
            args.battle_ratio,
            args.min_battle,
            args.rolling_turns,
        ):
            passed = False
    print(f"elapsed={perf_counter() - started:.1f}s")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
