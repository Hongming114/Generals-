from __future__ import annotations

import argparse

from generals_ai import GeneralsAI
from generals_core import Board, TurnReport


CONTACT_LIMITS = {
    "easy": 1_300,
    "normal": 500,
    "hard": 200,
}


def first_contact_and_attrition_turn(
    seed: int,
    difficulty: str,
    width: int,
    height: int,
    player_count: int,
    limit: int,
) -> tuple[int | None, int | None]:
    board = Board(width, height, player_count, seed)
    ais = [
        GeneralsAI(
            player,
            difficulty,
            seed * 10_000 + player,
        )
        for player in range(player_count)
    ]
    contact_turn: int | None = None
    attrition_turn: int | None = None
    for turn in range(1, limit + 1):
        report = TurnReport()
        for phase in range(2):
            moves = []
            for ai in ais:
                moves.extend(ai.plan_moves(board, phase=phase))
            report.merge(board.resolve_movement(moves))
            if contact_turn is None:
                for player in range(player_count):
                    visible = board.visibility(player)
                    if any(
                        board.tile(x, y).owner != player
                        and board.tile(x, y).owner >= 0
                        for x, y in visible
                    ):
                        contact_turn = turn
                        break
            if board.winner is not None:
                break
        for pair, amount in (report.attrition_by_pair or {}).items():
            if (
                len(pair) == 2
                and pair[0] >= 0
                and pair[1] >= 0
                and amount > 0
            ):
                attrition_turn = turn
                break
        if contact_turn is not None and attrition_turn is not None:
            return contact_turn, attrition_turn
        board.finish_turn(report)
        if board.winner is not None:
            break
    return contact_turn, attrition_turn


def winner_turn(
    seed: int,
    difficulty: str,
    width: int,
    height: int,
    player_count: int,
    limit: int,
    report_every: int = 0,
) -> tuple[int | None, int | None]:
    board = Board(width, height, player_count, seed)
    ais = [
        GeneralsAI(
            player,
            difficulty,
            seed * 10_000 + player,
        )
        for player in range(player_count)
    ]
    for turn in range(1, limit + 1):
        report = TurnReport()
        for phase in range(2):
            moves = []
            for ai in ais:
                moves.extend(ai.plan_moves(board, phase=phase))
            report.merge(board.resolve_movement(moves))
            if board.winner is not None:
                return turn, board.winner
        board.finish_turn(report)
        if report_every and turn % report_every == 0:
            stats = " ".join(
                f"P{player}:{board.stats(player)['tiles']}/{board.stats(player)['armies']}"
                for player in range(player_count)
            )
            print(f"turn={turn} {stats}", flush=True)
    return None, None


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError("seed range end must be >= start")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure first-contact aggression on random Generals maps.",
    )
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("1-10"))
    parser.add_argument(
        "--difficulty",
        choices=("all", "easy", "normal", "hard"),
        default="all",
    )
    parser.add_argument(
        "--mode",
        choices=("contact", "elimination"),
        default="contact",
    )
    parser.add_argument("--turn-limit", type=int, default=3_000)
    parser.add_argument("--report-every", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "elimination":
        passed = True
        results: list[tuple[int, int | None, int | None]] = []
        for seed in args.seeds:
            turn, winner = winner_turn(
                seed,
                "hard" if args.difficulty == "all" else args.difficulty,
                args.width,
                args.height,
                args.players,
                args.turn_limit,
                args.report_every,
            )
            results.append((seed, turn, winner))
            if turn is None:
                passed = False
        summary = ", ".join(
            f"{seed}:{turn if turn is not None else 'none'}"
            for seed, turn, _ in results
        )
        state = "PASS" if passed else "FAIL"
        print(
            f"ELIMINATION limit<{args.turn_limit} {state}  {summary}"
        )
        return 0 if passed else 1

    difficulties = (
        tuple(CONTACT_LIMITS)
        if args.difficulty == "all"
        else (args.difficulty,)
    )
    passed = True
    for difficulty in difficulties:
        limit = CONTACT_LIMITS[difficulty]
        results: list[tuple[int, int | None, int | None]] = []
        for seed in args.seeds:
            contact, attrition = first_contact_and_attrition_turn(
                seed,
                difficulty,
                args.width,
                args.height,
                args.players,
                limit,
            )
            results.append((seed, contact, attrition))
        summary = ", ".join(
            (
                f"{seed}:"
                f"{contact if contact is not None else 'none'}/"
                f"{attrition if attrition is not None else 'none'}"
            )
            for seed, contact, attrition in results
        )
        state = "PASS" if all(
            contact is not None
            and contact <= limit
            and attrition is not None
            and attrition <= limit
            for _, contact, attrition in results
        ) else "FAIL"
        if state == "FAIL":
            passed = False
        print(
            f"{difficulty.upper():6} limit<={limit:<4} {state}  {summary}"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
