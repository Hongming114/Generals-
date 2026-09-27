import argparse
from collections import Counter, deque
import json
import math
from multiprocessing import shared_memory
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

import main as main_module
import generals_training as training_module

from generals_ai import (
    AI_INDEPENDENT_ARMY_CREATION_COOLDOWN,
    AI_INDEPENDENT_ARMY_TROOP_DIVISOR,
    AIGenome,
    ARCHETYPE_ATTACK,
    ARCHETYPE_FORT,
    ARCHETYPE_STANDARD,
    ARCHETYPE_TURTLE,
    ARMY_STATE_ATTACK,
    ARMY_STATE_DEFEND,
    ARMY_STATE_RESUPPLY,
    ARMY_STATES,
    ATTACK_ENEMY_CITY,
    ATTACK_ENEMY_LAND,
    ATTACK_NEUTRAL_CITY,
    ATTRITION_STALE_TURNS,
    CAPITAL_REINFORCEMENT_RATIO_THRESHOLD,
    DEFEND_HOME_CITY,
    DEFEND_TERRITORY,
    DEVELOPMENT_MIN_DETACHMENT_ARMY,
    ENEMY_SEARCH_ATTACK_SCORE,
    EXPLORE_EXPAND,
    GatherRoute,
    GeneralsAI,
    INDEPENDENT_ARMY_CREATION_COOLDOWN,
    INDEPENDENT_ARMY_DAMAGE_PRESSURE,
    INDEPENDENT_ARMY_DISPLACEMENT_INTERVAL,
    INDEPENDENT_ARMY_DISPLACEMENT_REWARD_DISTANCE,
    INDEPENDENT_ARMY_EAGER_CREATE_STRENGTH,
    KNOWN_GENERAL_CAMPAIGN_SCORE,
    Mission,
    STRATEGY_ATTACK,
    STRATEGY_DEFENSE,
    STRATEGY_DEVELOPMENT,
    STRATEGY_EXPLORATION,
    ai_army_creation_limit,
    assign_ai_archetypes,
    independent_army_limit,
)
from generals_training import TrainingEngine
from generals_core import (
    AI,
    CITY,
    GENERAL,
    HILL,
    HUMAN,
    MAX_PLAYERS,
    MOUNTAIN,
    NEUTRAL,
    PLAIN,
    Board,
    Move,
    PendingMove,
    Tile,
    TurnReport,
)


def blank_board(seed: int = 1, player_count: int = 2, size: int = 27) -> Board:
    board = Board(size, size, player_count, seed)
    board.grid = [[Tile(PLAIN, NEUTRAL, 0) for _ in range(size)] for _ in range(size)]
    board.player_positions = {}
    for player in range(player_count):
        x = 3 + player * 4
        y = size - 4 - player * 4
        board.grid[y][x] = Tile(GENERAL, player, 2)
        board.player_positions[player] = (x, y)
    board.active_players = set(range(player_count))
    board.eliminated_players.clear()
    board.turn = 1
    board.winner = None
    return board


def set_tile(board: Board, x: int, y: int, tile: Tile) -> None:
    board.grid[y][x] = tile


def cell_screen_pos(app: main_module.GameApp, cell: tuple[int, int]) -> tuple[int, int]:
    (origin_x, origin_y), cell_size, _ = app.board_geometry()
    return (
        int(origin_x + (cell[0] + 0.5) * cell_size),
        int(origin_y + (cell[1] + 0.5) * cell_size),
    )


def route_app() -> main_module.GameApp:
    args = argparse.Namespace(
        difficulty="normal",
        size=39,
        players=2,
        theme="day",
        fog="on",
        turn_seconds=1.0,
        growth_interval=20,
        vision_radius=2,
        seed=7,
        quick_start=False,
    )
    app = main_module.GameApp(args, headless=True)
    app.board = blank_board()
    app.ais = {}
    app.state = "PLAYING"
    app.refresh_view_data()
    return app


def queue_adjacent_path(
    app: main_module.GameApp,
    source: tuple[int, int],
    targets: list[tuple[int, int]],
    mode: str = "all",
) -> bool:
    if not targets:
        return False
    for index, target in enumerate(targets):
        if not app.queue_command(
            source,
            target,
            mode,
            record_history=index == 0,
        ):
            return False
    return True


class BoardRuleTests(unittest.TestCase):
    def test_all_players_have_distinct_colors(self) -> None:
        colors = [
            main_module.player_color(player)
            for player in range(MAX_PLAYERS)
        ]

        self.assertEqual(len(colors), MAX_PLAYERS)
        self.assertEqual(len(set(colors)), MAX_PLAYERS)

    def test_default_vision_radius_is_two_square_tiles(self) -> None:
        board = blank_board(size=39)
        x, y = board.player_positions[AI]

        visible = board.visibility(AI)

        self.assertEqual(board.vision_radius, 2)
        self.assertIn((x + 2, y + 2), visible)
        self.assertNotIn((x + 3, y + 3), visible)
        self.assertTrue(board.is_visible(x + 2, y + 2, AI))
        self.assertFalse(board.is_visible(x + 3, y + 3, AI))

    def test_zero_vision_radius_only_reveals_owned_tiles(self) -> None:
        board = blank_board(size=39)
        board.vision_radius = 0
        owned = {
            (cx, cy)
            for cy, row in enumerate(board.grid)
            for cx, tile in enumerate(row)
            if tile.owner == AI
        }
        general = board.player_positions[AI]

        visible = board.visibility(AI)

        self.assertEqual(visible, owned)
        self.assertFalse(board.is_visible(general[0] + 1, general[1], AI))

    def test_five_tile_vision_radius_is_copied_by_clone(self) -> None:
        board = blank_board(size=39)
        board.vision_radius = 5
        x, y = board.player_positions[AI]

        visible = board.visibility(AI)
        clone = board.clone()

        self.assertIn((x + 5, y + 5), visible)
        self.assertNotIn((x + 6, y + 6), visible)
        self.assertEqual(clone.vision_radius, 5)
        self.assertEqual(clone.visibility(AI), visible)

    def test_hill_blocks_vision_behind_it_but_not_itself(self) -> None:
        board = blank_board(size=39)
        board.vision_radius = 3
        origin = (10, 10)
        hill = (11, 10)
        behind = (12, 10)
        side = (11, 12)
        set_tile(board, *origin, Tile(PLAIN, AI, 2))
        set_tile(board, *hill, Tile(HILL, NEUTRAL, 0))

        visible = board.visibility_from_positions([origin])

        self.assertIn(hill, visible)
        self.assertNotIn(behind, visible)
        self.assertIn(side, visible)
        self.assertTrue(board.is_visible(*hill, AI))
        self.assertFalse(board.is_visible(*behind, AI))

    def test_general_income(self) -> None:
        board = blank_board()
        board.resolve_turn([])
        self.assertEqual(board.tile(3, 23).army, 3)
        self.assertEqual(board.tile(7, 19).army, 3)

    def test_city_income_continues_past_fifty(self) -> None:
        board = blank_board()
        set_tile(board, 12, 12, Tile(CITY, HUMAN, 50))
        report = board.resolve_turn([])
        self.assertEqual(board.tile(12, 12).army, 51)
        self.assertEqual(report.income[HUMAN], 2)

    def test_land_growth_waits_for_interval(self) -> None:
        board = blank_board()
        board.growth_interval = 10
        set_tile(board, 12, 12, Tile(PLAIN, HUMAN, 3))

        for _ in range(9):
            report = board.resolve_turn([])
        self.assertEqual(board.tile(12, 12).army, 3)
        self.assertEqual(report.land_growth, {})

        report = board.resolve_turn([])
        self.assertEqual(board.tile(12, 12).army, 4)
        self.assertEqual(report.land_growth, {HUMAN: 1})

    def test_land_growth_does_not_replace_city_or_general_income(self) -> None:
        board = blank_board()
        board.growth_interval = 10
        set_tile(board, 12, 12, Tile(CITY, HUMAN, 8))

        for _ in range(10):
            report = board.resolve_turn([])

        self.assertEqual(board.tile(12, 12).army, 18)
        self.assertEqual(board.tile(3, 23).army, 12)
        self.assertEqual(report.land_growth, {})

    def test_land_growth_continues_past_fifty(self) -> None:
        board = blank_board()
        board.growth_interval = 10
        set_tile(board, 12, 12, Tile(PLAIN, HUMAN, 50))

        for _ in range(10):
            report = board.resolve_turn([])

        self.assertEqual(board.tile(12, 12).army, 51)
        self.assertEqual(report.land_growth, {HUMAN: 1})

    def test_hill_expansion_uses_double_interval(self) -> None:
        board = blank_board()
        board.growth_interval = 10
        set_tile(board, 12, 12, Tile(PLAIN, HUMAN, 5))
        set_tile(board, 13, 12, Tile(HILL, HUMAN, 5))

        for _ in range(9):
            report = board.resolve_turn([])
        self.assertEqual(board.tile(12, 12).army, 5)
        self.assertEqual(board.tile(13, 12).army, 5)

        report = board.resolve_turn([])
        self.assertEqual(board.tile(12, 12).army, 6)
        self.assertEqual(board.tile(13, 12).army, 5)

        for _ in range(9):
            report = board.resolve_turn([])
        self.assertEqual(board.tile(12, 12).army, 6)
        self.assertEqual(board.tile(13, 12).army, 5)

        report = board.resolve_turn([])
        self.assertEqual(board.tile(12, 12).army, 7)
        self.assertEqual(board.tile(13, 12).army, 6)
        self.assertEqual(report.land_growth[HUMAN], 2)

    def test_growth_interval_is_copied_by_clone(self) -> None:
        board = blank_board()
        board.growth_interval = 37
        self.assertEqual(board.clone().growth_interval, 37)

    def test_plain_to_neutral_hill_takes_two_turns(self) -> None:
        board = blank_board()
        set_tile(board, 10, 10, Tile(PLAIN, HUMAN, 5))
        set_tile(board, 11, 10, Tile(HILL, NEUTRAL, 0))

        first = board.resolve_turn([Move(10, 10, 11, 10, 4)])

        self.assertEqual(first.delayed_moves, 1)
        self.assertEqual(board.tile(10, 10).army, 5)
        self.assertEqual(board.tile(11, 10).owner, NEUTRAL)
        self.assertEqual(len(board.pending_moves), 1)

        second = board.resolve_turn([])

        self.assertEqual(second.arrived_moves, 1)
        self.assertEqual(board.tile(10, 10).army, 1)
        self.assertEqual(board.tile(11, 10).terrain, HILL)
        self.assertEqual(board.tile(11, 10).owner, HUMAN)
        self.assertEqual(board.tile(11, 10).army, 4)
        self.assertEqual(board.pending_moves, [])

    def test_pending_hill_move_reserves_source_and_can_be_cancelled(self) -> None:
        board = blank_board()
        set_tile(board, 10, 10, Tile(PLAIN, HUMAN, 5))
        set_tile(board, 11, 10, Tile(HILL, NEUTRAL, 0))

        board.resolve_turn([Move(10, 10, 11, 10, 4)])

        self.assertEqual(board.tile(10, 10).army, 5)
        self.assertFalse(board.move_legal(Move(10, 10, 10, 11, 4), HUMAN))
        self.assertEqual(board.cancel_pending_moves(HUMAN, (10, 10)), 1)

        board.resolve_turn([])
        self.assertEqual(board.tile(10, 10).army, 5)
        self.assertEqual(board.tile(11, 10).owner, NEUTRAL)

    def test_hill_to_hill_takes_one_turn(self) -> None:
        board = blank_board()
        set_tile(board, 10, 10, Tile(HILL, HUMAN, 5))
        set_tile(board, 11, 10, Tile(HILL, NEUTRAL, 0))

        report = board.resolve_turn([Move(10, 10, 11, 10, 4)])

        self.assertEqual(report.delayed_moves, 0)
        self.assertEqual(board.tile(11, 10).owner, HUMAN)
        self.assertEqual(board.tile(11, 10).army, 4)

    def test_hill_to_plain_takes_two_turns(self) -> None:
        board = blank_board()
        set_tile(board, 10, 10, Tile(HILL, HUMAN, 5))
        set_tile(board, 11, 10, Tile(PLAIN, NEUTRAL, 0))

        first = board.resolve_turn([Move(10, 10, 11, 10, 4)])
        self.assertEqual(first.delayed_moves, 1)
        self.assertEqual(board.tile(10, 10).army, 5)
        self.assertEqual(board.tile(11, 10).owner, NEUTRAL)

        board.resolve_turn([])
        self.assertEqual(board.tile(10, 10).army, 1)
        self.assertEqual(board.tile(11, 10).owner, HUMAN)
        self.assertEqual(board.tile(11, 10).army, 4)

    def test_moving_to_owned_hill_takes_one_turn(self) -> None:
        board = blank_board()
        set_tile(board, 10, 10, Tile(PLAIN, HUMAN, 5))
        set_tile(board, 11, 10, Tile(HILL, HUMAN, 1))

        report = board.resolve_turn([Move(10, 10, 11, 10, 4)])

        self.assertEqual(report.delayed_moves, 0)
        self.assertEqual(board.tile(11, 10).army, 5)

    def test_attacking_enemy_hill_takes_two_turns(self) -> None:
        board = blank_board()
        set_tile(board, 10, 10, Tile(HILL, HUMAN, 7))
        set_tile(board, 11, 10, Tile(HILL, AI, 1))

        first = board.resolve_turn([Move(10, 10, 11, 10, 6)])
        self.assertEqual(first.delayed_moves, 1)
        self.assertEqual(board.tile(10, 10).army, 7)
        self.assertEqual(board.tile(11, 10).owner, AI)

        board.resolve_turn([])
        self.assertEqual(board.tile(10, 10).army, 1)
        self.assertEqual(board.tile(11, 10).owner, HUMAN)
        self.assertEqual(board.tile(11, 10).army, 5)

    def test_capture_neutral_tile(self) -> None:
        board = blank_board()
        set_tile(board, 12, 12, Tile(PLAIN, HUMAN, 4))
        report = board.resolve_turn([Move(12, 12, 12, 13, 3)])
        self.assertEqual(board.tile(12, 12).army, 1)
        self.assertEqual(board.tile(12, 13).owner, HUMAN)
        self.assertEqual(board.tile(12, 13).army, 3)
        self.assertIn((12, 13, HUMAN), report.captures)

    def test_reinforcement_merges(self) -> None:
        board = blank_board()
        set_tile(board, 12, 12, Tile(PLAIN, HUMAN, 5))
        set_tile(board, 12, 13, Tile(PLAIN, HUMAN, 2))
        board.resolve_turn([Move(12, 12, 12, 13, 4)])
        self.assertEqual(board.tile(12, 13).army, 6)
        self.assertEqual(board.tile(12, 12).army, 1)

    def test_simultaneous_swap(self) -> None:
        board = blank_board()
        set_tile(board, 10, 10, Tile(PLAIN, HUMAN, 5))
        set_tile(board, 11, 10, Tile(PLAIN, AI, 5))
        board.resolve_turn(
            [
                Move(10, 10, 11, 10, 4),
                Move(11, 10, 10, 10, 4),
            ]
        )
        self.assertEqual(board.tile(10, 10).owner, AI)
        self.assertEqual(board.tile(10, 10).army, 3)
        self.assertEqual(board.tile(11, 10).owner, HUMAN)
        self.assertEqual(board.tile(11, 10).army, 3)

    def test_same_army_moves_two_tiles_and_income_applies_once(self) -> None:
        board = blank_board()
        set_tile(board, 10, 10, Tile(PLAIN, HUMAN, 5))
        set_tile(board, 11, 10, Tile(PLAIN, NEUTRAL, 0))
        set_tile(board, 12, 10, Tile(PLAIN, NEUTRAL, 0))
        human_general = board.player_positions[HUMAN]
        general_army = board.tile(*human_general).army

        report = board.resolve_turn(
            [Move(10, 10, 11, 10, 4)],
            follow_up_moves=[Move(11, 10, 12, 10, 3)],
        )

        self.assertEqual(report.accepted_moves, 2)
        self.assertEqual(board.tile(12, 10).owner, HUMAN)
        self.assertEqual(board.tile(12, 10).army, 3)
        self.assertEqual(board.tile(*human_general).army, general_army + 1)
        self.assertEqual(board.turn, 2)

    def test_multiple_attackers_fight_each_other(self) -> None:
        board = blank_board()
        set_tile(board, 12, 12, Tile(PLAIN, NEUTRAL, 0))
        set_tile(board, 11, 12, Tile(PLAIN, HUMAN, 6))
        set_tile(board, 13, 12, Tile(PLAIN, AI, 5))
        set_tile(board, 12, 11, Tile(PLAIN, HUMAN, 3))
        board.resolve_turn(
            [
                Move(11, 12, 12, 12, 5),
                Move(13, 12, 12, 12, 4),
                Move(12, 11, 12, 12, 2),
            ]
        )
        self.assertEqual(board.tile(12, 12).owner, HUMAN)
        self.assertEqual(board.tile(12, 12).army, 3)

    def test_battle_report_accumulates_attrition_for_the_opponent_pair(self) -> None:
        board = blank_board()
        set_tile(board, 10, 10, Tile(PLAIN, AI, 8))
        set_tile(board, 11, 10, Tile(PLAIN, HUMAN, 3))

        report = board.resolve_turn([Move(10, 10, 11, 10, 7)])

        self.assertEqual(report.attrition_by_pair[(HUMAN, AI)], 6)
        self.assertEqual(board.tile(11, 10).owner, AI)
        self.assertEqual(board.tile(11, 10).army, 4)

        clone = board.clone()
        self.assertEqual(clone.last_report.attrition_by_pair[(HUMAN, AI)], 6)

    def test_merged_turn_reports_preserve_human_as_winner(self) -> None:
        report = TurnReport()

        report.merge(TurnReport(winner=HUMAN))

        self.assertEqual(report.winner, HUMAN)

    def test_two_player_general_capture_ends_game(self) -> None:
        board = blank_board()
        set_tile(board, 7, 19, Tile(GENERAL, AI, 1))
        set_tile(board, 6, 19, Tile(PLAIN, HUMAN, 4))
        report = board.resolve_turn([Move(6, 19, 7, 19, 3)])
        self.assertEqual(board.winner, HUMAN)
        self.assertEqual(report.winner, HUMAN)
        self.assertIn(AI, report.eliminated)

    def test_eliminated_territory_transfers_half_army_to_conqueror(self) -> None:
        board = blank_board(player_count=4)
        set_tile(board, 7, 19, Tile(GENERAL, 1, 1))
        set_tile(board, 6, 19, Tile(PLAIN, HUMAN, 6))
        set_tile(board, 20, 20, Tile(CITY, 1, 8))
        set_tile(board, 21, 20, Tile(PLAIN, 1, 3))
        set_tile(board, 22, 20, Tile(HILL, 1, 4))
        set_tile(board, 23, 20, Tile(PLAIN, 1, 1))
        report = board.resolve_turn([Move(6, 19, 7, 19, 5)])
        self.assertIn(1, report.eliminated)
        self.assertNotIn(1, board.active_players)
        self.assertIn(1, board.eliminated_players)
        self.assertEqual(board.tile(20, 20).owner, HUMAN)
        self.assertEqual(board.tile(20, 20).army, 5)
        self.assertEqual(board.tile(21, 20).owner, HUMAN)
        self.assertEqual(board.tile(21, 20).army, 2)
        self.assertEqual(board.tile(22, 20).owner, HUMAN)
        self.assertEqual(board.tile(22, 20).army, 2)
        self.assertEqual(board.tile(23, 20).owner, HUMAN)
        self.assertEqual(board.tile(23, 20).army, 1)
        self.assertEqual(board.tile(7, 19).terrain, CITY)
        self.assertEqual(board.tile(7, 19).owner, HUMAN)
        self.assertGreater(len(board.active_players), 1)
        self.assertIsNone(board.winner)

    def test_moves_keep_one_defender(self) -> None:
        board = blank_board()
        set_tile(board, 12, 12, Tile(PLAIN, HUMAN, 2))
        self.assertFalse(board.move_legal(Move(12, 12, 12, 13, 2), HUMAN))
        self.assertTrue(board.move_legal(Move(12, 12, 12, 13, 1), HUMAN))

    def test_generated_two_player_maps_have_random_connected_spawns(self) -> None:
        first_positions: tuple[tuple[int, int], ...] | None = None
        for seed in range(6):
            board = Board(39, 39, 2, seed)
            human_general = board.general_position(HUMAN)
            ai_general = board.general_position(AI)
            self.assertIsNotNone(human_general)
            self.assertIsNotNone(ai_general)
            positions = tuple(board.player_positions.values())
            self.assertTrue(board._all_passable_connected(board.grid, list(positions)))
            radii = [
                math.hypot(x - board.width / 2, y - board.height / 2)
                for x, y in positions
            ]
            self.assertGreater(max(radii) - min(radii), 1.0)
            if first_positions is None:
                first_positions = positions
            else:
                self.assertNotEqual(first_positions, positions)

    def test_twenty_by_twenty_map_generates_connected_spawns(self) -> None:
        for seed in range(4):
            board = Board(20, 20, 2, seed)
            positions = list(board.player_positions.values())
            self.assertEqual(len(positions), 2)
            self.assertEqual(len(set(positions)), 2)
            self.assertTrue(
                board._all_passable_connected(board.grid, positions)
            )

    def test_twelve_player_map_has_unique_connected_spawns(self) -> None:
        for seed in range(4):
            board = Board(51, 51, 12, seed)
            positions = list(board.player_positions.values())
            self.assertEqual(len(positions), 12)
            self.assertEqual(len(set(positions)), 12)
            self.assertEqual(len(board.active_players), 12)
            self.assertTrue(board._all_passable_connected(board.grid, positions))
            radii = [
                math.hypot(x - board.width / 2, y - board.height / 2)
                for x, y in positions
            ]
            self.assertGreater(max(radii) - min(radii), 2.0)
            for player, position in board.player_positions.items():
                tile = board.tile(*position)
                self.assertEqual(tile.terrain, GENERAL)
                self.assertEqual(tile.owner, player)

    def test_hill_counts_follow_mountain_multiplier(self) -> None:
        for size, player_count, seed in ((39, 2, 9), (75, 12, 17)):
            board = Board(size, size, player_count, seed)
            mountains = sum(
                tile.terrain == MOUNTAIN
                for row in board.grid
                for tile in row
            )
            hills = sum(
                tile.terrain == HILL
                for row in board.grid
                for tile in row
            )
            plains = sum(
                tile.terrain == PLAIN
                for row in board.grid
                for tile in row
            )
            self.assertGreaterEqual(hills, mountains - 2)
            self.assertLessEqual(hills, mountains * 2 + 2)
            self.assertLessEqual(hills, plains)
            self.assertGreaterEqual(board.hill_multiplier, 1.0)
            self.assertLessEqual(board.hill_multiplier, 2.0)

            clustered = 0
            for y, row in enumerate(board.grid):
                for x, tile in enumerate(row):
                    if tile.terrain != HILL:
                        continue
                    if any(
                        board.in_bounds(x + dx, y + dy)
                        and board.tile(x + dx, y + dy).terrain == MOUNTAIN
                        for dx in range(-3, 4)
                        for dy in range(-3, 4)
                    ):
                        clustered += 1
            self.assertGreaterEqual(clustered * 2, hills)

    def test_mountains_form_ridges_with_increased_density(self) -> None:
        for width, height, player_count, seed in (
            (20, 20, 2, 1),
            (51, 51, 12, 7),
            (75, 75, 24, 17),
        ):
            board = Board(width, height, player_count, seed)
            mountains = {
                (x, y)
                for y, row in enumerate(board.grid)
                for x, tile in enumerate(row)
                if tile.terrain == MOUNTAIN
            }
            self.assertGreaterEqual(
                len(mountains),
                int(width * height * 0.15),
            )
            self.assertLessEqual(
                len(mountains),
                int(width * height * 0.21),
            )
            self.assertTrue(
                board._all_passable_connected(
                    board.grid,
                    list(board.player_positions.values()),
                )
            )
            connected = sum(
                any(
                    (x + dx, y + dy) in mountains
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )
                for x, y in mountains
            )
            self.assertGreaterEqual(connected, len(mountains) * 0.72)
            self.assertFalse(
                any(
                    all(
                        (x + dx, y + dy) in mountains
                        for dy in range(5)
                        for dx in range(5)
                    )
                    for y in range(height - 4)
                    for x in range(width - 4)
                )
            )
            self.assertFalse(
                any(
                    all(
                        (x + dx, y + dy) in mountains
                        for dy in range(2)
                        for dx in range(2)
                    )
                    for y in range(height - 1)
                    for x in range(width - 1)
                )
            )

            seen: set[tuple[int, int]] = set()
            longest = 0
            for start in mountains:
                if start in seen:
                    continue
                queue = deque([start])
                seen.add(start)
                size = 0
                while queue:
                    x, y = queue.popleft()
                    size += 1
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        neighbor = (x + dx, y + dy)
                        if neighbor in mountains and neighbor not in seen:
                            seen.add(neighbor)
                            queue.append(neighbor)
                longest = max(longest, size)
            self.assertGreaterEqual(longest, 4)
            self.assertLessEqual(longest, 40)

            quadrant_counts = [
                sum(
                    1
                    for y in range(top, top + (height + 1) // 2)
                    for x in range(left, left + (width + 1) // 2)
                    if (x, y) in mountains
                )
                for top in (0, height // 2)
                for left in (0, width // 2)
            ]
            if width >= 39:
                self.assertGreaterEqual(
                    sum(count > 0 for count in quadrant_counts),
                    2,
                )
            else:
                self.assertGreaterEqual(
                    sum(count > 0 for count in quadrant_counts),
                    2,
                )

    def test_mountain_distribution_is_random_organic_not_quadrant_tiled(
        self,
    ) -> None:
        signatures: set[frozenset[tuple[int, int]]] = set()
        quadrant_spreads: list[int] = []
        orientations = {
            "horizontal": False,
            "vertical": False,
            "diagonal": False,
        }
        for seed in range(1, 9):
            board = Board(51, 51, 12, seed)
            mountains = {
                (x, y)
                for y, row in enumerate(board.grid)
                for x, tile in enumerate(row)
                if tile.terrain == MOUNTAIN
            }
            signatures.add(frozenset(mountains))

            quadrant_counts = [
                sum(
                    1
                    for y in range(top, top + (board.height + 1) // 2)
                    for x in range(left, left + (board.width + 1) // 2)
                    if (x, y) in mountains
                )
                for top in (0, board.height // 2)
                for left in (0, board.width // 2)
            ]
            quadrant_spreads.append(
                max(quadrant_counts) - min(quadrant_counts)
            )

            for x, y in mountains:
                orientations["horizontal"] |= (x + 1, y) in mountains
                orientations["vertical"] |= (x, y + 1) in mountains
                orientations["diagonal"] |= any(
                    (x + dx, y + dy) in mountains
                    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                )

        self.assertGreater(len(signatures), 1)
        self.assertGreater(max(quadrant_spreads), 10)
        self.assertTrue(all(orientations.values()))

    def test_hills_allow_sparse_cells_away_from_mountains(self) -> None:
        board = Board(20, 20, 4, 31)
        board.grid = [
            [Tile(PLAIN, NEUTRAL, 0) for _ in range(board.width)]
            for _ in range(board.height)
        ]
        board.grid[0][0] = Tile(MOUNTAIN, NEUTRAL, 0)
        safe_cells = {
            (x, y)
            for y in range(board.height)
            for x in range(board.width)
            if (x, y) != (0, 0)
        }

        board._place_hills(
            board.grid,
            safe_cells,
            [],
            __import__("random").Random(31),
            multiplier=70.0,
        )

        hills = {
            (x, y)
            for y, row in enumerate(board.grid)
            for x, tile in enumerate(row)
            if tile.terrain == HILL
        }
        sparse_hills = {
            (x, y)
            for x, y in hills
            if all(
                not board.in_bounds(x + dx, y + dy)
                or board.tile(x + dx, y + dy).terrain != MOUNTAIN
                for dx in range(-3, 4)
                for dy in range(-3, 4)
            )
        }

        self.assertGreater(len(sparse_hills), 0)

    def test_twenty_four_player_map_has_unique_connected_spawns(self) -> None:
        board = Board(75, 75, 24, 20260925)
        positions = list(board.player_positions.values())
        self.assertEqual(len(positions), 24)
        self.assertEqual(len(set(positions)), 24)
        self.assertEqual(len(board.active_players), 24)
        self.assertTrue(board._all_passable_connected(board.grid, positions))
        radii = [
            math.hypot(x - board.width / 2, y - board.height / 2)
            for x, y in positions
        ]
        self.assertGreater(max(radii) - min(radii), 4.0)

    def test_thirty_seven_player_map_has_unique_connected_spawns(self) -> None:
        board = Board(75, 75, MAX_PLAYERS, 20260925)
        positions = list(board.player_positions.values())
        self.assertEqual(len(positions), MAX_PLAYERS)
        self.assertEqual(len(set(positions)), MAX_PLAYERS)
        self.assertEqual(len(board.active_players), MAX_PLAYERS)
        self.assertTrue(board._all_passable_connected(board.grid, positions))

    def test_mountains_block_moves(self) -> None:
        board = blank_board()
        set_tile(board, 12, 12, Tile(PLAIN, HUMAN, 3))
        set_tile(board, 12, 13, Tile(MOUNTAIN, NEUTRAL, 0))
        self.assertFalse(board.move_legal(Move(12, 12, 12, 13, 2), HUMAN))

    def test_mountain_and_hill_use_required_terrain_rendering(self) -> None:
        app = route_app()
        rect = pygame.Rect(0, 0, 24, 24)
        mountain_surface = pygame.Surface((24, 24))
        hill_surface = pygame.Surface((24, 24))
        plain_surface = pygame.Surface((24, 24))

        app.draw_tile_base(
            Tile(MOUNTAIN, NEUTRAL, 0),
            rect,
            True,
            0,
            0,
            24,
            surface=mountain_surface,
        )
        app.draw_tile_base(
            Tile(HILL, NEUTRAL, 0),
            rect,
            True,
            0,
            0,
            24,
            surface=hill_surface,
        )
        app.draw_tile_base(
            Tile(PLAIN, NEUTRAL, 0),
            rect,
            True,
            0,
            0,
            24,
            surface=plain_surface,
        )

        self.assertEqual(
            mountain_surface.get_at((2, 2))[:3],
            main_module.MOUNTAIN_COLOR,
        )
        self.assertEqual(
            hill_surface.get_at((2, 2))[:3],
            plain_surface.get_at((2, 2))[:3],
        )

class RouteCommandTests(unittest.TestCase):
    def tearDown(self) -> None:
        pygame.quit()

    def test_two_army_land_can_receive_long_route(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))

        queued = queue_adjacent_path(
            app,
            (10, 10),
            [(x, 10) for x in range(11, 16)],
        )

        self.assertTrue(queued)
        self.assertEqual(len(app.human_commands[(10, 10)]), 5)
        self.assertEqual(app.route_end((10, 10)), (15, 10))

    def test_two_army_land_can_start_long_route_with_clicks(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        for x in range(11, 15):
            set_tile(app.board, x, 10, Tile(PLAIN, NEUTRAL, 0))
        app.selected = (10, 10)

        for x in range(11, 15):
            app.handle_game_click(cell_screen_pos(app, (x, 10)), 1)

        self.assertIn((10, 10), app.human_commands)
        self.assertEqual(app.route_end((10, 10)), (14, 10))
        self.assertEqual(app.selected, (14, 10))

        app.advance_turn()

        self.assertEqual(app.board.tile(11, 10).owner, HUMAN)
        self.assertEqual(app.board.tile(11, 10).army, 1)
        self.assertIn((11, 10), app.human_commands)

    def test_single_left_click_only_selects_land(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))

        app.handle_game_click(cell_screen_pos(app, (10, 10)), 1)

        self.assertEqual(app.selected, (10, 10))
        self.assertNotIn((10, 10), app.human_commands)
        self.assertTrue(
            app.human_ai.army_controller.has_main_army_tag((10, 10))
        )

    def test_fullscreen_flag_uses_scaled_fullscreen_only_outside_headless(
        self,
    ) -> None:
        app = object.__new__(main_module.GameApp)
        app.fullscreen = False
        app.headless = False
        self.assertEqual(app._display_flags(), 0)

        app.fullscreen = True
        self.assertEqual(
            app._display_flags(),
            pygame.SCALED | pygame.FULLSCREEN,
        )

        app.headless = True
        self.assertEqual(app._display_flags(), 0)

    def test_single_left_click_moves_to_adjacent_tile_and_selects_it(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        set_tile(app.board, 11, 10, Tile(PLAIN, NEUTRAL, 0))
        app.selected = (10, 10)

        app.handle_game_click(cell_screen_pos(app, (11, 10)), 1)

        self.assertEqual(app.route_end((10, 10)), (11, 10))
        self.assertEqual(len(app.human_commands[(10, 10)]), 1)
        self.assertEqual(app.selected, (11, 10))

    def test_single_left_click_adjacent_owned_tile_moves_and_selects_it(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 4))
        set_tile(app.board, 11, 10, Tile(PLAIN, HUMAN, 2))
        app.selected = (10, 10)

        app.handle_game_click(cell_screen_pos(app, (11, 10)), 1)

        self.assertEqual(app.human_commands[(10, 10)][0].mode, "all")
        self.assertEqual(app.selected, (11, 10))

    def test_single_left_click_far_tile_only_changes_selection(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        set_tile(app.board, 14, 13, Tile(PLAIN, NEUTRAL, 0))
        app.selected = (10, 10)

        app.handle_game_click(cell_screen_pos(app, (14, 13)), 1)

        self.assertNotIn((10, 10), app.human_commands)
        self.assertEqual(app.selected, (14, 13))

    def test_successive_adjacent_left_clicks_extend_the_same_route(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        for x in range(11, 15):
            set_tile(app.board, x, 10, Tile(PLAIN, NEUTRAL, 0))
        set_tile(app.board, 14, 11, Tile(PLAIN, NEUTRAL, 0))
        app.selected = (10, 10)

        for x in range(11, 15):
            app.handle_game_click(cell_screen_pos(app, (x, 10)), 1)
        app.handle_game_click(cell_screen_pos(app, (14, 11)), 1)

        self.assertEqual(app.route_end((10, 10)), (14, 11))
        self.assertEqual(len(app.human_commands[(10, 10)]), 5)
        self.assertEqual(app.selected, (14, 11))

    def test_route_extensions_survive_turns_between_clicks(self) -> None:
        app = route_app()
        for x in range(10, 15):
            set_tile(app.board, x, 10, Tile(PLAIN, HUMAN, 1))
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        app.selected = (10, 10)
        self.assertTrue(
            queue_adjacent_path(
                app,
                (10, 10),
                [(x, 10) for x in range(11, 15)],
            )
        )

        app.advance_turn()
        self.assertEqual(app.selected, (12, 10))
        app.handle_game_click(cell_screen_pos(app, (14, 10)), 1)
        app.handle_game_click(cell_screen_pos(app, (14, 11)), 1)

        self.assertIn((12, 10), app.human_commands)
        self.assertEqual(app.route_end((12, 10)), (14, 11))
        self.assertEqual(app.selected, (14, 11))

    def test_single_click_command_is_immediate(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        set_tile(app.board, 11, 10, Tile(PLAIN, NEUTRAL, 0))
        app.selected = (10, 10)
        app.handle_game_click(cell_screen_pos(app, (11, 10)), 1)

        self.assertEqual(app.route_end((10, 10)), (11, 10))
        self.assertEqual(app.selected, (11, 10))

    def test_paused_commands_execute_in_chronological_phase_order(self) -> None:
        app = route_app()
        app.state = "PAUSED"
        sources = ((5, 10), (10, 10), (15, 10))
        targets = ((6, 10), (11, 10), (16, 10))
        for source, target in zip(sources, targets):
            set_tile(app.board, *source, Tile(PLAIN, HUMAN, 4))
            set_tile(app.board, *target, Tile(PLAIN, NEUTRAL, 0))
            self.assertTrue(app.queue_command(source, target, "all"))

        self.assertEqual(app.command_order, list(sources))
        planned = app.human_moves()
        self.assertEqual([move.source for move in planned], [sources[0]])

        app.advance_turn()

        self.assertEqual(app.board.tile(*targets[0]).owner, HUMAN)
        self.assertEqual(app.board.tile(*targets[1]).owner, HUMAN)
        self.assertEqual(app.board.tile(*targets[2]).owner, NEUTRAL)
        self.assertEqual(app.command_order, [sources[2]])

    def test_scheduled_route_rotates_behind_other_commands(self) -> None:
        app = route_app()
        set_tile(app.board, 5, 10, Tile(PLAIN, HUMAN, 6))
        set_tile(app.board, 6, 10, Tile(PLAIN, NEUTRAL, 0))
        set_tile(app.board, 7, 10, Tile(PLAIN, NEUTRAL, 0))
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 4))
        set_tile(app.board, 11, 10, Tile(PLAIN, NEUTRAL, 0))
        self.assertTrue(app.queue_command((5, 10), (6, 10), "all"))
        self.assertTrue(app.queue_command((5, 10), (7, 10), "all"))
        self.assertTrue(app.queue_command((10, 10), (11, 10), "all"))

        app.advance_turn()

        self.assertEqual(app.command_order, [(6, 10)])
        self.assertEqual(app.board.tile(7, 10).owner, NEUTRAL)
        app.advance_turn()
        self.assertEqual(app.board.tile(7, 10).owner, HUMAN)

    def test_drag_does_not_create_automatic_route(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        app.selected = (10, 10)

        pygame.event.post(
            pygame.event.Event(
                pygame.MOUSEMOTION,
                pos=cell_screen_pos(app, (14, 10)),
                rel=(0, 0),
                buttons=(1, 0, 0),
            )
        )
        app.handle_events()

        self.assertNotIn((10, 10), app.human_commands)
        self.assertEqual(app.route_end((10, 10)), (10, 10))

    def test_route_cannot_start_into_a_mountain(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        set_tile(app.board, 11, 10, Tile(MOUNTAIN, NEUTRAL, 0))

        self.assertFalse(
            queue_adjacent_path(app, (10, 10), [(11, 10), (12, 10)])
        )
        self.assertNotIn((10, 10), app.human_commands)

    def test_far_left_click_does_not_create_automatic_route(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        set_tile(app.board, 15, 10, Tile(PLAIN, HUMAN, 2))
        app.selected = (10, 10)
        target_pos = cell_screen_pos(app, (15, 10))

        app.handle_game_click(target_pos, 1)

        self.assertNotIn((10, 10), app.human_commands)
        self.assertEqual(app.selected, (15, 10))

    def test_far_click_changes_selection_without_extending_existing_route(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        set_tile(app.board, 11, 10, Tile(PLAIN, NEUTRAL, 0))
        set_tile(app.board, 15, 15, Tile(PLAIN, HUMAN, 2))
        app.selected = (10, 10)
        app.handle_game_click(cell_screen_pos(app, (11, 10)), 1)

        app.handle_game_click(cell_screen_pos(app, (15, 15)), 1)

        self.assertEqual(app.route_end((10, 10)), (11, 10))
        self.assertEqual(len(app.human_commands[(10, 10)]), 1)
        self.assertEqual(app.selected, (15, 15))

    def test_queue_command_rejects_a_non_adjacent_target(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 4))

        queued = app.queue_command((10, 10), (15, 10), "all")

        self.assertFalse(queued)
        self.assertNotIn((10, 10), app.human_commands)

    def test_clicking_outside_board_clears_selection(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        app.selected = (10, 10)

        app.handle_game_click((0, 0), 1)

        self.assertIsNone(app.selected)

    def test_z_undoes_last_whole_command(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        app.selected = (10, 10)
        queue_adjacent_path(
            app,
            (10, 10),
            [(x, 10) for x in range(11, 16)],
        )

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_z))
        app.handle_events()

        self.assertNotIn((10, 10), app.human_commands)
        self.assertEqual(app.route_end((10, 10)), (10, 10))
        self.assertEqual(app.selected, (10, 10))

    def test_right_single_click_adjacent_tile_uses_half_army(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 5))
        set_tile(app.board, 11, 10, Tile(PLAIN, NEUTRAL, 0))
        app.selected = (10, 10)

        app.handle_game_click(cell_screen_pos(app, (11, 10)), 3)

        self.assertEqual(app.human_commands[(10, 10)][0].mode, "half")
        self.assertEqual(app.selected, (11, 10))

    def test_far_right_click_only_changes_selection(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 5))
        set_tile(app.board, 15, 10, Tile(PLAIN, HUMAN, 2))
        app.selected = (10, 10)
        target_pos = cell_screen_pos(app, (15, 10))

        app.handle_game_click(target_pos, 3)

        self.assertNotIn((10, 10), app.human_commands)
        self.assertEqual(app.selected, (15, 10))

    def test_right_adjacent_clicks_build_half_army_route(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 5))
        for x in range(11, 15):
            set_tile(app.board, x, 10, Tile(PLAIN, HUMAN, 2))

        app.handle_game_click(cell_screen_pos(app, (10, 10)), 3)
        self.assertEqual(app.selected, (10, 10))

        for x in range(11, 15):
            app.handle_game_click(cell_screen_pos(app, (x, 10)), 3)
        self.assertEqual(len(app.human_commands[(10, 10)]), 4)
        self.assertTrue(
            all(step.mode == "half" for step in app.human_commands[(10, 10)])
        )

    def test_c_hotkey_cancels_selected_route(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        app.selected = (10, 10)
        queue_adjacent_path(
            app,
            (10, 10),
            [(x, 10) for x in range(11, 16)],
        )

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        app.handle_events()

        self.assertNotIn((10, 10), app.human_commands)

    def test_route_limit_is_sixty_four_steps(self) -> None:
        app = route_app()
        app.board = blank_board(size=75)
        set_tile(app.board, 0, 0, Tile(PLAIN, HUMAN, 2))

        self.assertTrue(
            queue_adjacent_path(
                app,
                (0, 0),
                [(x, 0) for x in range(1, 65)],
            )
        )
        self.assertEqual(len(app.human_commands[(0, 0)]), 64)
        self.assertFalse(app.queue_command((0, 0), (65, 0), "all"))

    def test_large_maps_share_the_same_minimum_view_span(self) -> None:
        app = route_app()
        self.assertLessEqual(main_module.MIN_VIEW_CELLS, 36)
        for size in (75, 100):
            app.board_size = size
            app.board = blank_board(size=size)
            app.zoom = app.min_zoom()
            _, cell, area = app.board_geometry()
            self.assertAlmostEqual(area.height / cell, main_module.MIN_VIEW_CELLS, delta=0.05)

    def test_keyboard_panning_preserves_the_current_zoom(self) -> None:
        app = route_app()
        app.zoom = app.min_zoom()
        initial_zoom = app.zoom
        initial_x = app.camera_center[0]

        class PressedKeys:
            @staticmethod
            def __getitem__(key: int) -> bool:
                return key in (pygame.K_d, pygame.K_RIGHT)

        with mock.patch.object(
            main_module.pygame.key,
            "get_pressed",
            return_value=PressedKeys(),
        ):
            app.handle_keyboard_held(0.1)

        self.assertEqual(app.zoom, initial_zoom)
        self.assertGreater(app.camera_center[0], initial_x)

    def test_max_zoom_is_large_enough_for_large_maps(self) -> None:
        app = route_app()
        app.board_size = 100
        app.board = blank_board(size=100)

        self.assertGreaterEqual(main_module.MAX_ZOOM, 7.0)
        app.zoom = main_module.MAX_ZOOM
        _, cell, _ = app.board_geometry()
        self.assertGreater(cell, 60.0)

    def test_fog_can_be_disabled_for_human_view(self) -> None:
        app = route_app()
        app.fog_enabled = False

        app.refresh_view_data()

        self.assertEqual(len(app.visible_human), app.board.width * app.board.height)

    def test_ai_captures_do_not_create_player_action_animations(self) -> None:
        app = route_app()
        app.recent_captures.clear()

        app.record_action_animations(
            [
                (1, 1, 1),
                (2, 2, HUMAN),
                (3, 3, 2),
            ]
        )

        self.assertEqual(
            [(x, y) for x, y, _started in app.recent_captures],
            [(2, 2)],
        )

    def test_disabling_fog_does_not_disable_ai_vision_radius(self) -> None:
        app = route_app()
        app.vision_radius = 0
        app.board.vision_radius = 0
        app.fog_enabled = False

        app.refresh_view_data()

        self.assertEqual(len(app.visible_human), app.board.width * app.board.height)
        self.assertLess(
            len(app.board.visibility(AI)),
            app.board.width * app.board.height,
        )

    def test_fog_setting_button_changes_view_mode(self) -> None:
        app = route_app()
        app.state = "SETTINGS"
        app.draw_settings()

        app.handle_settings_click(app.menu_fog_rects["off"].center)

        self.assertFalse(app.fog_enabled)

    def test_vision_radius_setting_applies_to_new_game(self) -> None:
        app = route_app()
        app.state = "SETTINGS"
        app.draw_settings()

        app.handle_settings_click(
            (app.menu_vision_rect.right - 1, app.menu_vision_rect.centery)
        )

        self.assertEqual(app.vision_radius, 5)

        app.new_game(seed=123)

        self.assertEqual(app.board.vision_radius, 5)
        self.assertEqual(app.board.visibility(HUMAN), app.visible_human)

    def test_fastest_turn_speed_has_no_artificial_floor(self) -> None:
        app = route_app()
        app.state = "SETTINGS"
        app.draw_settings()

        app.handle_settings_click(
            (app.menu_speed_rect.left, app.menu_speed_rect.centery)
        )

        self.assertEqual(app.turn_seconds, main_module.MIN_TURN_SECONDS)
        self.assertEqual(app.turn_seconds, 0.0)

        app.turn_deadline = 0.0
        app.draw_panel()

    def test_ai_count_slider_supports_thirty_six_ai_players(self) -> None:
        app = route_app()
        app.board_size = 75
        app.state = "SETTINGS"
        app.draw_settings()

        app.handle_settings_click(
            (app.menu_ai_count_rect.right - 1, app.menu_ai_count_rect.centery)
        )

        self.assertEqual(app.ai_count, 36)
        self.assertEqual(app.player_count, MAX_PLAYERS)

        app.new_game(seed=123)

        self.assertEqual(len(app.ais), 36)
        self.assertEqual(len(app.board.active_players), MAX_PLAYERS)

    def test_eliminated_human_keeps_watching_until_an_ai_wins(self) -> None:
        app = route_app()
        app.board = blank_board(player_count=3)
        app.player_count = 3
        app.ais = {}
        human_general = app.board.player_positions[HUMAN]
        ai_one_general = app.board.player_positions[1]
        set_tile(app.board, *human_general, Tile(GENERAL, 1, 5))

        app.advance_turn()

        self.assertTrue(app.spectating)
        self.assertEqual(app.state, "PLAYING")
        self.assertIsNone(app.board.winner)
        self.assertEqual(len(app.visible_human), app.board.width * app.board.height)

        set_tile(app.board, *ai_one_general, Tile(GENERAL, 2, 5))
        app.advance_turn()

        self.assertEqual(app.board.winner, 2)
        self.assertEqual(app.state, "GAME_OVER")

    def test_ai_takeover_uses_ai_moves_for_human_board(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 5))
        set_tile(app.board, 11, 10, Tile(PLAIN, NEUTRAL, 0))

        class StubAI:
            @staticmethod
            def plan_moves(
                board: Board,
                phase: int = 0,
                observed_moves: tuple[Move, ...] = (),
            ) -> list[Move]:
                return [Move(10, 10, 11, 10, 4)]

        app.human_ai = StubAI()
        app.ai_takeover = True

        app.advance_turn()

        self.assertEqual(app.board.tile(11, 10).owner, HUMAN)
        self.assertEqual(app.board.tile(11, 10).army, 4)

    def test_ai_takeover_blocks_route_commands(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        app.ai_takeover = True

        self.assertFalse(app.queue_command((10, 10), (11, 10), "all"))

    def test_ai_takeover_button_toggles_both_ways(self) -> None:
        app = route_app()

        app.draw_panel()
        app.handle_game_click(app.takeover_rect.center, 1)
        self.assertTrue(app.ai_takeover)

        app.draw_panel()
        app.handle_game_click(app.takeover_rect.center, 1)
        self.assertFalse(app.ai_takeover)

    def test_two_army_long_route_advances_through_friendly_territory(self) -> None:
        app = route_app()
        for x in range(10, 14):
            set_tile(app.board, x, 10, Tile(PLAIN, HUMAN, 1))
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        queue_adjacent_path(
            app,
            (10, 10),
            [(x, 10) for x in range(11, 14)],
        )

        for _ in range(3):
            app.advance_turn()

        self.assertEqual(app.board.tile(10, 10).army, 1)
        self.assertEqual(app.board.tile(13, 10).army, 2)
        self.assertNotIn((10, 10), app.human_commands)
        self.assertNotIn((12, 10), app.human_commands)
        self.assertIsNone(app.selected)

    def test_route_waits_for_hill_transit_and_then_continues(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 5))
        set_tile(app.board, 11, 10, Tile(HILL, NEUTRAL, 0))
        set_tile(app.board, 12, 10, Tile(PLAIN, NEUTRAL, 0))
        set_tile(app.board, 13, 10, Tile(PLAIN, NEUTRAL, 0))
        queue_adjacent_path(
            app,
            (10, 10),
            [(x, 10) for x in range(11, 14)],
        )

        app.advance_turn()
        self.assertIn((10, 10), app.human_commands)
        self.assertIn((10, 10), app.route_hill_waits)
        self.assertEqual(app.board.tile(11, 10).owner, NEUTRAL)

        app.advance_turn()
        self.assertIn((11, 10), app.human_commands)
        self.assertNotIn((10, 10), app.route_hill_waits)
        self.assertEqual(app.board.tile(11, 10).owner, HUMAN)
        self.assertIn((11, 10), app.route_hill_waits)
        self.assertEqual(app.board.tile(12, 10).owner, NEUTRAL)

        app.advance_turn()
        self.assertEqual(app.board.tile(12, 10).owner, HUMAN)
        self.assertEqual(app.board.tile(13, 10).owner, HUMAN)
        self.assertNotIn((12, 10), app.human_commands)

    def test_cancelling_hill_route_before_arrival_keeps_army_at_source(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 5))
        set_tile(app.board, 11, 10, Tile(HILL, NEUTRAL, 0))
        queue_adjacent_path(app, (10, 10), [(11, 10)])

        app.advance_turn()
        self.assertEqual(app.board.tile(10, 10).army, 5)
        self.assertIn((10, 10), app.route_hill_waits)
        self.assertEqual(len(app.board.pending_moves), 1)

        app.cancel_command((10, 10))
        self.assertEqual(app.board.tile(10, 10).army, 5)
        self.assertEqual(app.board.pending_moves, [])

        app.advance_turn()
        self.assertEqual(app.board.tile(10, 10).army, 5)
        self.assertEqual(app.board.tile(11, 10).owner, NEUTRAL)

    def test_route_cannot_cross_a_mountain(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 2))
        set_tile(app.board, 11, 10, Tile(PLAIN, NEUTRAL, 0))
        set_tile(app.board, 12, 10, Tile(MOUNTAIN, NEUTRAL, 0))

        queued = queue_adjacent_path(
            app,
            (10, 10),
            [(11, 10), (12, 10), (13, 10)],
        )

        self.assertFalse(queued)
        self.assertEqual(len(app.human_commands[(10, 10)]), 1)


class AITests(unittest.TestCase):
    def test_ai_archetype_assignment_uses_fixed_weight_quotas(self) -> None:
        titles = assign_ai_archetypes(10, seed=77)
        counts = Counter(titles)

        self.assertEqual(counts[ARCHETYPE_STANDARD], 4)
        self.assertEqual(counts[ARCHETYPE_ATTACK], 2)
        self.assertEqual(counts[ARCHETYPE_FORT], 2)
        self.assertEqual(counts[ARCHETYPE_TURTLE], 2)
        self.assertEqual(
            GeneralsAI(AI, "hard", 1).archetype,
            ARCHETYPE_STANDARD,
        )

    def test_attack_archetype_boosts_attacks_and_all_in(self) -> None:
        board = blank_board(player_count=2, size=39)
        general = (10, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 30))
        set_tile(board, 11, 10, Tile(PLAIN, AI, 1))
        ai = GeneralsAI(
            AI,
            "hard",
            42,
            archetype=ARCHETYPE_ATTACK,
        )
        ai.home_guard_hard_target_army = 10
        ai.all_in_target = (20, 20)

        self.assertEqual(
            ai._archetype_mission_multiplier(
                board,
                ATTACK_ENEMY_CITY,
                (20, 20),
            ),
            1.5,
        )
        self.assertEqual(ai._archetype_all_in_multiplier(), 2.0)
        capped = ai._cap_home_guard_move(
            board,
            Move(10, 10, 11, 10, 30),
        )
        self.assertIsNotNone(capped)
        self.assertEqual(capped.amount, 30)

    def test_fort_archetype_boosts_exploration_and_weak_castles(self) -> None:
        board = blank_board(player_count=2, size=39)
        neutral_city = (10, 10)
        weak_city = (12, 10)
        strong_city = (14, 10)
        set_tile(board, *neutral_city, Tile(CITY, NEUTRAL, 5))
        set_tile(board, *weak_city, Tile(CITY, HUMAN, 8))
        set_tile(board, *strong_city, Tile(CITY, HUMAN, 50))
        ai = GeneralsAI(
            AI,
            "hard",
            42,
            archetype=ARCHETYPE_FORT,
        )

        self.assertEqual(
            ai._archetype_mission_multiplier(
                board,
                EXPLORE_EXPAND,
                neutral_city,
            ),
            1.5,
        )
        self.assertEqual(
            ai._archetype_mission_multiplier(
                board,
                ATTACK_NEUTRAL_CITY,
                neutral_city,
            ),
            2.0,
        )
        self.assertEqual(
            ai._archetype_mission_multiplier(
                board,
                ATTACK_ENEMY_CITY,
                weak_city,
            ),
            2.0,
        )
        self.assertEqual(
            ai._archetype_mission_multiplier(
                board,
                ATTACK_ENEMY_CITY,
                strong_city,
            ),
            1.0,
        )

    def test_turtle_archetype_accumulates_then_releases_all_in(self) -> None:
        board = blank_board(player_count=2, size=20)
        general = (5, 5)
        enemy_general = (15, 15)
        set_tile(board, *general, Tile(GENERAL, AI, 5))
        set_tile(board, *enemy_general, Tile(GENERAL, HUMAN, 5))
        ai = GeneralsAI(
            AI,
            "hard",
            42,
            archetype=ARCHETYPE_TURTLE,
        )

        ai._update_archetype_state(board, general)

        self.assertTrue(ai.turtle_accumulating)
        self.assertEqual(ai.archetype_capital_target_army, 24)
        self.assertEqual(
            ai._archetype_mission_multiplier(
                board,
                EXPLORE_EXPAND,
                enemy_general,
            ),
            0.15,
        )

        set_tile(board, *general, Tile(GENERAL, AI, 24))
        ai.archetype_update_turn = -1
        ai._update_archetype_state(board, general)

        self.assertFalse(ai.turtle_accumulating)
        self.assertTrue(ai.turtle_released)
        self.assertTrue(ai.forced_offensive)
        self.assertEqual(ai._archetype_all_in_multiplier(), 2.0)
        self.assertEqual(
            ai._archetype_mission_multiplier(
                board,
                ATTACK_ENEMY_CITY,
                enemy_general,
            ),
            3.0,
        )

    def test_strategy_state_changes_mission_weights(self) -> None:
        ai = GeneralsAI(AI, "normal", 11)
        ai.strategy_state_ready = True

        ai.strategy_state = STRATEGY_ATTACK
        self.assertGreater(
            ai._state_score_multiplier(ATTACK_ENEMY_CITY),
            ai._state_score_multiplier(ATTACK_NEUTRAL_CITY),
        )
        ai.strategy_state = STRATEGY_DEFENSE
        self.assertGreater(
            ai._state_score_multiplier(DEFEND_HOME_CITY),
            ai._state_score_multiplier(ATTACK_ENEMY_LAND),
        )
        ai.strategy_state = STRATEGY_DEVELOPMENT
        self.assertGreater(
            ai._state_score_multiplier(ATTACK_NEUTRAL_CITY),
            ai._state_score_multiplier(ATTACK_ENEMY_CITY),
        )
        ai.strategy_state = STRATEGY_EXPLORATION
        self.assertGreater(
            ai._state_score_multiplier(EXPLORE_EXPAND),
            ai._state_score_multiplier(ATTACK_ENEMY_LAND),
        )

    def test_update_self_state_can_reach_all_four_states(self) -> None:
        board = blank_board(player_count=4, size=39)
        general = (20, 20)
        set_tile(board, *general, Tile(GENERAL, AI, 8))
        board.player_positions[AI] = general
        owned = [general, (21, 20), (22, 20)]
        for position in owned[1:]:
            set_tile(board, *position, Tile(PLAIN, AI, 4))
        ai = GeneralsAI(AI, "hard", 42)
        ai.strategy_update_turn = -1
        ai.war_state = "peace"
        ai.global_army_rank_low = True
        ai.global_army_rank_high = False
        ai.enemy_army_estimate = {1: 100, 2: 100, 3: 100}

        ai.known_neutral_cities = {(23, 20)}
        ai._update_self_state(board, set(), owned, general)
        self.assertEqual(ai.strategy_state, STRATEGY_DEVELOPMENT)

        ai.known_neutral_cities = set()
        ai.forced_offensive = True
        ai.strategy_update_turn = -1
        ai._update_self_state(board, set(), owned, general)
        self.assertEqual(ai.strategy_state, STRATEGY_ATTACK)

        ai.forced_offensive = False
        ai.home_threat_target = (21, 20)
        ai.strategy_update_turn = -1
        ai._update_self_state(board, set(), owned, general)
        self.assertEqual(ai.strategy_state, STRATEGY_DEFENSE)

        ai.home_threat_target = None
        ai.development_hold_turns = 0
        ai.gather_mode = None
        ai.global_army_rank_low = False
        ai.global_army_rank_high = False
        ai.enemy_army_estimate = {1: 1, 2: 1, 3: 1}
        ai.strategy_update_turn = -1
        ai._update_self_state(board, set(), owned, general)
        self.assertEqual(ai.strategy_state, STRATEGY_EXPLORATION)

    def test_attrition_marks_war_and_stale_exchange_returns_to_peace(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=39)
        board.last_report.record_attrition(AI, HUMAN, 9)
        ai = GeneralsAI(AI, "hard", 42)

        ai._update_attrition_state(board, [(10, 10)])
        ai._update_war_state(board, 40)

        self.assertEqual(ai.war_state, "war")
        self.assertEqual(ai.war_state_opponent, HUMAN)
        self.assertEqual(ai.war_last_exchange_turn, board.turn)

        board.turn += 61
        ai._update_war_state(board, 40)

        self.assertEqual(ai.war_state, "peace")
        self.assertEqual(ai.war_state_opponent, -1)

    def test_peace_rank_prefers_development_when_low_and_exploration_when_high(
        self,
    ) -> None:
        ai = GeneralsAI(AI, "hard", 42)
        ai.war_state = "peace"

        ai.global_army_rank_low = True
        ai.global_army_rank_high = False
        self.assertEqual(ai._peace_rank_state(), STRATEGY_DEVELOPMENT)

        ai.global_army_rank_low = False
        ai.global_army_rank_high = True
        self.assertEqual(ai._peace_rank_state(), STRATEGY_EXPLORATION)

    def test_war_state_scales_weaker_defense_or_stronger_attack(
        self,
    ) -> None:
        ai = GeneralsAI(AI, "hard", 42)
        ai.war_state = "war"
        ai.war_force_ratio = 0.8

        self.assertEqual(
            ai._war_strategy_multiplier(DEFEND_HOME_CITY),
            1.2,
        )
        self.assertEqual(
            ai._war_strategy_multiplier(ATTACK_ENEMY_LAND),
            1.0,
        )

        ai.war_force_ratio = 1.25
        self.assertEqual(
            ai._war_strategy_multiplier(DEFEND_TERRITORY),
            1.0,
        )
        self.assertEqual(
            ai._war_strategy_multiplier(ATTACK_ENEMY_CITY),
            1.2,
        )

    def test_ai_enters_defense_state_when_capital_is_threatened(self) -> None:
        board = blank_board(player_count=2, size=39)
        general = board.general_position(AI)
        self.assertIsNotNone(general)
        set_tile(board, *general, Tile(GENERAL, AI, 5))
        threat = (general[0] + 1, general[1])
        set_tile(board, *threat, Tile(PLAIN, HUMAN, 15))
        ai = GeneralsAI(AI, "hard", 42)

        moves = ai.plan_moves(board)

        self.assertEqual(ai.strategy_state, STRATEGY_DEFENSE)
        self.assertEqual(ai.mission.kind, DEFEND_HOME_CITY)
        self.assertEqual(len(moves), 1)

    def test_overwhelming_force_freezes_development_and_attacks(self) -> None:
        board = blank_board(player_count=2, size=75)
        general = board.general_position(AI)
        self.assertIsNotNone(general)
        field_stack = (20, 20)
        enemy_general = (30, 30)
        set_tile(board, *general, Tile(GENERAL, AI, 5))
        set_tile(board, *field_stack, Tile(PLAIN, AI, 80))
        set_tile(board, *enemy_general, Tile(GENERAL, HUMAN, 10))
        board.player_positions[HUMAN] = enemy_general
        ai = GeneralsAI(AI, "hard", 42)
        ai.campaign_target = enemy_general
        ai.campaign_target_player = HUMAN
        ai.campaign_target_kind = ATTACK_ENEMY_CITY
        ai.campaign_hold_turns = 180
        ai.enemy_army_estimate[HUMAN] = 20

        with mock.patch.object(ai.rng, "random", return_value=0.0):
            moves = ai.plan_moves(board)

        self.assertEqual(ai.strategy_state, STRATEGY_ATTACK)
        self.assertGreater(ai.overwhelming_hold_turns, 0)
        self.assertGreater(ai.development_frozen_turns, 0)
        self.assertEqual(ai.development_hold_turns, 0)
        self.assertEqual(ai.gather_mode, "overwhelming")
        self.assertEqual(ai.mission.kind, ATTACK_ENEMY_CITY)
        self.assertEqual(ai.mission.target, enemy_general)
        self.assertTrue(moves)

    def test_home_guard_move_keeps_hard_ten_percent_floor(self) -> None:
        board = blank_board(player_count=2, size=39)
        general = board.general_position(AI)
        self.assertIsNotNone(general)
        set_tile(board, *general, Tile(GENERAL, AI, 20))
        ai = GeneralsAI(AI, "hard", 42)
        ai.home_guard_hard_target_army = 3

        capped = ai._cap_home_guard_move(
            board,
            Move(general[0], general[1], general[0] + 1, general[1], 20),
        )

        self.assertIsNotNone(capped)
        self.assertEqual(capped.amount, 17)

    def test_adjacent_army_does_not_count_as_home_garrison(self) -> None:
        board = blank_board(player_count=2, size=39)
        general = board.general_position(AI)
        self.assertIsNotNone(general)
        set_tile(board, *general, Tile(GENERAL, AI, 2))
        neighbor = (general[0] + 1, general[1])
        set_tile(board, *neighbor, Tile(PLAIN, AI, 20))
        ai = GeneralsAI(AI, "hard", 42)
        ai.home_guard_hard_target_army = 3

        capped = ai._cap_home_guard_move(
            board,
            Move(neighbor[0], neighbor[1], general[0] + 2, general[1], 20),
        )

        self.assertIsNotNone(capped)
        self.assertEqual(capped.amount, 20)

    def test_distance_cache_tracks_owned_hill_revision(self) -> None:
        board = blank_board()
        set_tile(board, 1, 1, Tile(PLAIN, NEUTRAL, 0))
        set_tile(board, 2, 1, Tile(HILL, NEUTRAL, 0))
        ai = GeneralsAI(HUMAN, "normal", 11)

        neutral_distances = ai._distance_map(board, [(1, 1)])
        self.assertEqual(neutral_distances[(2, 1)], 2)

        hill = board.tile(2, 1)
        hill.owner = HUMAN
        board._record_ownership_change(hill, NEUTRAL)
        owned_distances = ai._distance_map(board, [(1, 1)])
        self.assertEqual(owned_distances[(2, 1)], 1)

    def test_opening_ai_produces_legal_moves(self) -> None:
        board = Board(39, 39, 2, 42)
        ai = GeneralsAI(AI, "normal", 42)
        moves = ai.plan_moves(board)
        self.assertTrue(moves)
        for move in moves:
            self.assertTrue(board.move_legal(move, AI))

    def test_ai_plans_at_most_one_move_per_turn(self) -> None:
        board = blank_board(player_count=2)
        for x in range(5, 15):
            set_tile(board, x, 8, Tile(PLAIN, AI, 8))
        ai = GeneralsAI(AI, "hard", 42)

        moves = ai.plan_moves(board)

        self.assertLessEqual(len(moves), 1)
        self.assertTrue(all(board.move_legal(move, AI) for move in moves))

    def test_ai_can_move_the_same_stack_in_two_turn_phases(self) -> None:
        board = blank_board(player_count=2, size=39)
        board.grid = [
            [Tile(PLAIN, NEUTRAL, 0) for _ in range(board.width)]
            for _ in range(board.height)
        ]
        source = (10, 10)
        human_general = (30, 30)
        set_tile(board, *source, Tile(GENERAL, AI, 5))
        set_tile(board, *human_general, Tile(GENERAL, HUMAN, 2))
        board.player_positions[AI] = source
        board.player_positions[HUMAN] = human_general
        ai = GeneralsAI(AI, "hard", 42)

        first_moves = ai.plan_moves(board, phase=0)
        report = board.resolve_movement(first_moves)
        second_moves = ai.plan_moves(board, phase=1)
        report.merge(board.resolve_movement(second_moves))
        board.finish_turn(report)
        occupied = [
            (x, y)
            for y in range(board.height)
            for x in range(board.width)
            if board.tile(x, y).owner == AI
            and board.tile(x, y).army > 1
        ]

        self.assertTrue(first_moves)
        self.assertTrue(second_moves)
        self.assertEqual(second_moves[0].source, first_moves[0].target)
        self.assertEqual(
            max(Board.manhattan(source, position) for position in occupied),
            2,
        )

    def test_ai_attacks_when_it_has_enough_troops(self) -> None:
        board = blank_board(player_count=2)
        general = board.general_position(AI)
        self.assertIsNotNone(general)
        set_tile(board, *general, Tile(GENERAL, AI, 10))
        set_tile(board, 10, 10, Tile(PLAIN, AI, 20))
        set_tile(board, 11, 10, Tile(PLAIN, HUMAN, 5))
        ai = GeneralsAI(AI, "hard", 42)

        moves = ai.plan_moves(board)

        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].source, (10, 10))
        self.assertEqual(moves[0].target, (11, 10))
        self.assertEqual(moves[0].amount, 19)

    def test_local_engulf_weight_requires_confirmed_margin(self) -> None:
        board = blank_board(player_count=2, size=39)
        attacker = (10, 10)
        defender = (11, 10)
        set_tile(board, *attacker, Tile(PLAIN, AI, 30))
        set_tile(board, *defender, Tile(PLAIN, HUMAN, 5))
        ai = GeneralsAI(AI, "hard", 42)
        move = Move(attacker[0], attacker[1], defender[0], defender[1], 29)

        multiplier = ai._local_engulf_weight_multiplier(
            board,
            board.visibility(AI),
            move,
            board.tile(*defender),
            True,
        )

        self.assertEqual(multiplier, 1.5)

    def test_local_engulf_weight_does_not_apply_without_margin(self) -> None:
        board = blank_board(player_count=2, size=39)
        attacker = (10, 10)
        defender = (11, 10)
        set_tile(board, *attacker, Tile(PLAIN, AI, 22))
        set_tile(board, *defender, Tile(PLAIN, HUMAN, 20))
        ai = GeneralsAI(AI, "hard", 42)
        move = Move(attacker[0], attacker[1], defender[0], defender[1], 21)

        multiplier = ai._local_engulf_weight_multiplier(
            board,
            board.visibility(AI),
            move,
            board.tile(*defender),
            True,
        )

        self.assertEqual(multiplier, 1.0)

    def test_local_engulf_weight_excludes_waiting_gather_stack(self) -> None:
        board = blank_board(player_count=2, size=39)
        stack = (10, 10)
        defender = (11, 10)
        set_tile(board, *stack, Tile(PLAIN, AI, 30))
        set_tile(board, *defender, Tile(PLAIN, HUMAN, 5))
        ai = GeneralsAI(AI, "hard", 42)
        ai.gather_mode = "land"
        ai.primary_stack = stack
        move = Move(stack[0], stack[1], defender[0], defender[1], 29)

        reserved = ai._local_engulf_weight_multiplier(
            board,
            board.visibility(AI),
            move,
            board.tile(*defender),
            True,
        )
        ai.gather_mode = None
        ai.planning_engulf_cache.clear()
        available = ai._local_engulf_weight_multiplier(
            board,
            board.visibility(AI),
            move,
            board.tile(*defender),
            True,
        )

        self.assertEqual(reserved, 1.0)
        self.assertEqual(available, 1.5)

    def test_local_engulf_weight_excludes_independent_army_cell(self) -> None:
        board = blank_board(player_count=2, size=39)
        attacker = (10, 10)
        defender = (11, 10)
        set_tile(board, *attacker, Tile(PLAIN, AI, 30))
        set_tile(board, *defender, Tile(PLAIN, HUMAN, 5))
        ai = GeneralsAI(AI, "hard", 42)
        created, _message = ai.army_controller.create_at(
            board,
            attacker,
            ignore_cooldown=True,
        )
        self.assertTrue(created)
        move = Move(attacker[0], attacker[1], defender[0], defender[1], 29)

        multiplier = ai._local_engulf_weight_multiplier(
            board,
            board.visibility(AI),
            move,
            board.tile(*defender),
            True,
        )

        self.assertEqual(multiplier, 1.0)

    def test_legacy_strategic_army_label_is_removed(self) -> None:
        ai = GeneralsAI(AI, "hard", 42)

        self.assertFalse(hasattr(ai, "army_label_active"))
        self.assertFalse(hasattr(main_module, "STRATEGIC_ARMY_LABEL"))
        self.assertFalse(hasattr(ai, "_update_army_label"))
        self.assertFalse(hasattr(ai, "_decisive_local_engulf_move"))
        self.assertFalse(hasattr(ai.army_controller, "_gather_target"))
        self.assertFalse(hasattr(ai.army_controller, "_retreat_target"))

    def test_local_engulf_is_a_weight_bonus_not_a_forced_priority(self) -> None:
        board = blank_board(player_count=2, size=39)
        attacker = (10, 10)
        defender = (11, 10)
        set_tile(board, *attacker, Tile(PLAIN, AI, 8))
        set_tile(board, *defender, Tile(PLAIN, HUMAN, 2))
        ai = GeneralsAI(AI, "hard", 42)
        move = Move(attacker[0], attacker[1], defender[0], defender[1], 7)

        multiplier = ai._local_engulf_weight_multiplier(
            board,
            board.visibility(AI),
            move,
            board.tile(*defender),
            True,
        )

        self.assertEqual(multiplier, 1.5)

    def test_adjacent_capturable_enemy_capital_is_taken_immediately(self) -> None:
        board = blank_board(player_count=2, size=39)
        board.grid = [
            [Tile(PLAIN, NEUTRAL, 0) for _ in range(board.width)]
            for _ in range(board.height)
        ]
        own_general = (10, 10)
        attacker = (11, 10)
        enemy_general = (12, 10)
        set_tile(board, *own_general, Tile(GENERAL, AI, 1))
        set_tile(board, *attacker, Tile(PLAIN, AI, 6))
        set_tile(board, *enemy_general, Tile(GENERAL, HUMAN, 4))
        board.player_positions[AI] = own_general
        board.player_positions[HUMAN] = enemy_general
        ai = GeneralsAI(AI, "hard", 42)

        moves = ai.plan_moves(board)

        self.assertEqual(moves[0].source, attacker)
        self.assertEqual(moves[0].target, enemy_general)
        self.assertEqual(moves[0].amount, 5)

    def test_adjacent_enemy_capital_waits_when_capture_army_is_too_small(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=39)
        board.grid = [
            [Tile(PLAIN, NEUTRAL, 0) for _ in range(board.width)]
            for _ in range(board.height)
        ]
        own_general = (10, 10)
        attacker = (11, 10)
        enemy_general = (12, 10)
        set_tile(board, *own_general, Tile(GENERAL, AI, 1))
        set_tile(board, *attacker, Tile(PLAIN, AI, 4))
        set_tile(board, *enemy_general, Tile(GENERAL, HUMAN, 4))
        board.player_positions[AI] = own_general
        board.player_positions[HUMAN] = enemy_general
        ai = GeneralsAI(AI, "hard", 42)

        moves = ai.plan_moves(board)

        self.assertFalse(
            any(move.target == enemy_general for move in moves)
        )

    def test_ai_moves_rear_troops_toward_frontline(self) -> None:
        board = blank_board(player_count=2)
        general = board.general_position(AI)
        self.assertIsNotNone(general)
        set_tile(board, *general, Tile(GENERAL, AI, 12))
        set_tile(board, 10, 10, Tile(PLAIN, AI, 20))
        for x in range(11, 15):
            set_tile(board, x, 10, Tile(PLAIN, AI, 1))
        set_tile(board, 15, 10, Tile(PLAIN, HUMAN, 1))
        ai = GeneralsAI(AI, "hard", 42)

        moves = ai.plan_moves(board)

        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].source, (10, 10))
        self.assertEqual(moves[0].target, (11, 10))
        self.assertEqual(moves[0].amount, 19)

    def test_ai_moves_rear_hill_troops_toward_frontline(self) -> None:
        board = blank_board(player_count=2)
        general = board.general_position(AI)
        self.assertIsNotNone(general)
        set_tile(board, *general, Tile(GENERAL, AI, 12))
        set_tile(board, 10, 10, Tile(HILL, AI, 20))
        for x in range(11, 15):
            set_tile(board, x, 10, Tile(PLAIN, AI, 1))
        set_tile(board, 15, 10, Tile(PLAIN, HUMAN, 1))
        ai = GeneralsAI(AI, "hard", 42)

        moves = ai.plan_moves(board)

        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].source, (10, 10))
        self.assertEqual(moves[0].target, (11, 10))
        self.assertEqual(moves[0].amount, 19)

    def test_ai_offensive_path_detours_around_unwinnable_blocker(self) -> None:
        board = blank_board(player_count=2, size=39)
        source = (10, 10)
        blocker = (11, 10)
        goal = (12, 10)
        set_tile(board, *source, Tile(PLAIN, AI, 6))
        set_tile(board, *blocker, Tile(PLAIN, HUMAN, 12))
        set_tile(board, *goal, Tile(PLAIN, HUMAN, 1))
        board.player_positions[AI] = source
        board.player_positions[HUMAN] = goal
        ai = GeneralsAI(AI, "hard", 42)
        ai.forced_offensive = True
        ai.primary_stack = source
        ai.mission = Mission(ATTACK_ENEMY_LAND, goal, 100_000)

        move = ai._forced_vanguard_move(board, board.visibility(AI))

        self.assertIsNotNone(move)
        self.assertEqual(move.source, source)
        self.assertNotEqual(move.target, blocker)
        self.assertIn(move.target, ((9, 10), (10, 9), (10, 11)))

    def test_defensive_path_does_not_use_offensive_detour_cost(self) -> None:
        board = blank_board(player_count=2, size=39)
        source = (10, 10)
        blocker = (11, 10)
        goal = (12, 10)
        set_tile(board, *source, Tile(PLAIN, AI, 6))
        set_tile(board, *blocker, Tile(PLAIN, HUMAN, 12))
        set_tile(board, *goal, Tile(PLAIN, HUMAN, 1))
        board.player_positions[AI] = source
        board.player_positions[HUMAN] = goal
        ai = GeneralsAI(AI, "hard", 42)
        ai.mission = Mission(DEFEND_HOME_CITY, goal, 100_000)

        mission_distances = ai._mission_distance_map(
            board,
            board.visibility(AI),
            [goal],
        )

        self.assertEqual(mission_distances, ai._distance_map(board, [goal]))

    def test_mission_distance_handles_visible_enemy_off_route(self) -> None:
        board = blank_board(player_count=2, size=20)
        source = (0, 0)
        goal = (10, 10)
        set_tile(board, *source, Tile(PLAIN, AI, 3))
        set_tile(board, *goal, Tile(PLAIN, HUMAN, 1))
        board.player_positions[AI] = source
        board.player_positions[HUMAN] = goal
        ai = GeneralsAI(AI, "hard", 42)
        ai.mission = Mission(ATTACK_ENEMY_LAND, goal, 100_000)
        ai.planning_owned_tiles = [source]
        ai.planning_max_mobile_army = 2

        mission_distances = ai._mission_distance_map(
            board,
            {goal},
            [goal],
        )

        self.assertEqual(mission_distances, ai._distance_map(board, [goal]))

    def test_ai_moves_far_reserve_to_stalled_frontline(self) -> None:
        board = blank_board(player_count=2, size=75)
        front = (20, 10)
        reserve = (5, 10)
        set_tile(board, *front, Tile(PLAIN, AI, 8))
        set_tile(board, *reserve, Tile(PLAIN, AI, 30))
        ai = GeneralsAI(AI, "hard", 42)
        ai.forced_offensive = True
        ai.primary_stack = front
        ai.frontline_stall_turns = 3
        ai.mission = Mission(EXPLORE_EXPAND, front, 20_000)

        move = ai._forced_vanguard_move(board, board.visibility(AI))

        self.assertIsNotNone(move)
        self.assertEqual(move.source, reserve)
        self.assertEqual(move.target, (6, 10))
        self.assertEqual(move.amount, 29)
        self.assertEqual(ai.primary_reinforce_target, front)
        self.assertEqual(ai.primary_stack, (6, 10))
        self.assertEqual(ai.frontline_stall_turns, 0)

    def test_ai_calls_far_reserve_before_a_weakened_vanguard_advances(self) -> None:
        board = blank_board(player_count=2, size=75)
        front = (20, 10)
        reserve = (5, 10)
        set_tile(board, *front, Tile(PLAIN, AI, 3))
        set_tile(board, *reserve, Tile(PLAIN, AI, 24))
        ai = GeneralsAI(AI, "hard", 42)
        ai.forced_offensive = True
        ai.primary_stack = front
        ai.mission = Mission(ATTACK_ENEMY_LAND, (30, 10), 100_000)

        move = ai._forced_vanguard_move(board, board.visibility(AI))

        self.assertIsNotNone(move)
        self.assertEqual(move.source, reserve)
        self.assertNotEqual(move.source, front)
        self.assertEqual(ai.primary_reinforce_target, front)
        self.assertEqual(ai.primary_stack, (6, 10))

    def test_ai_keeps_a_formed_vanguard_instead_of_swapping_to_home_reserve(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=75)
        front = (20, 10)
        home = (5, 10)
        set_tile(board, *front, Tile(PLAIN, AI, 6))
        set_tile(board, *home, Tile(PLAIN, AI, 8))
        ai = GeneralsAI(AI, "hard", 42)
        ai.forced_offensive = True
        ai.primary_stack = front
        ai.mission = Mission(EXPLORE_EXPAND, (40, 10), 200_000)

        move = ai._forced_vanguard_move(board, board.visibility(AI))

        self.assertIsNotNone(move)
        self.assertEqual(move.source, front)
        self.assertNotEqual(move.source, home)
        self.assertEqual(ai.primary_stack, move.target)

    def test_ai_triggers_development_when_attrition_exceeds_twice_its_army(self) -> None:
        board = blank_board(player_count=2, size=75)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 8))
        ai = GeneralsAI(AI, "hard", 42)
        board.last_report.record_attrition(AI, HUMAN, 17)

        ai._update_attrition_state(board, [(10, 10)])

        self.assertEqual(ai.development_trigger_opponent, HUMAN)
        self.assertGreater(ai.development_hold_turns, 0)
        self.assertEqual(ai.attrition_by_opponent[HUMAN], 0)

    def test_ai_resets_stale_attrition_instead_of_remembering_it_forever(self) -> None:
        board = blank_board(player_count=2, size=75)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 8))
        board.turn = ATTRITION_STALE_TURNS + 2
        ai = GeneralsAI(AI, "hard", 42)
        ai.attrition_by_opponent[HUMAN] = 10
        ai.attrition_last_update[HUMAN] = 1

        ai._update_attrition_state(board, [(10, 10)])

        self.assertNotIn(HUMAN, ai.attrition_by_opponent)
        self.assertNotIn(HUMAN, ai.attrition_last_update)

    def test_development_task_sends_a_rear_detachment_to_a_neutral_city(self) -> None:
        board = blank_board(player_count=2, size=75)
        primary = (10, 10)
        reserve = (55, 55)
        city = (60, 60)
        set_tile(board, *primary, Tile(PLAIN, AI, 20))
        set_tile(board, *reserve, Tile(PLAIN, AI, 12))
        set_tile(board, *city, Tile(CITY, NEUTRAL, 5))
        ai = GeneralsAI(AI, "hard", 42)
        ai.primary_stack = primary
        ai.development_hold_turns = 100
        ai.known_neutral_cities.add(city)
        ai._update_development_state(
            board,
            board.visibility(AI),
            [primary, reserve],
        )

        move = ai._forced_development_move(board)

        self.assertEqual(ai.development_target, city)
        self.assertIsNotNone(move)
        self.assertEqual(move.source, reserve)
        self.assertNotEqual(move.source, primary)
        self.assertGreaterEqual(
            board.tile(*reserve).army - 1,
            DEVELOPMENT_MIN_DETACHMENT_ARMY,
        )

    def test_development_task_explores_when_no_neutral_city_is_known(self) -> None:
        board = blank_board(player_count=2, size=75)
        primary = (10, 10)
        reserve = (50, 50)
        set_tile(board, *primary, Tile(PLAIN, AI, 20))
        set_tile(board, *reserve, Tile(PLAIN, AI, 12))
        ai = GeneralsAI(AI, "hard", 42)
        ai.primary_stack = primary
        ai.development_hold_turns = 100

        ai._update_development_state(
            board,
            board.visibility(AI),
            [primary, reserve],
        )
        move = ai._forced_development_move(board)

        self.assertEqual(ai.development_target_kind, "explore")
        self.assertIsNotNone(ai.development_target)
        self.assertIsNotNone(move)
        self.assertEqual(move.source, reserve)

    def test_ai_does_not_immediately_reverse_its_exploration_move(self) -> None:
        board = blank_board(player_count=2)
        ai = GeneralsAI(AI, "hard", 42)

        first = ai.plan_moves(board)
        self.assertEqual(len(first), 1)
        board.resolve_turn(first)
        second = ai.plan_moves(board)

        self.assertEqual(len(second), 1)
        self.assertFalse(
            second[0].source == first[0].target
            and second[0].target == first[0].source
        )

    def test_mission_candidates_cover_all_six_tasks(self) -> None:
        board = blank_board(player_count=2)
        set_tile(board, 10, 10, Tile(GENERAL, AI, 6))
        set_tile(board, 10, 11, Tile(CITY, AI, 4))
        set_tile(board, 10, 12, Tile(PLAIN, AI, 2))
        set_tile(board, 11, 10, Tile(CITY, NEUTRAL, 5))
        set_tile(board, 11, 11, Tile(PLAIN, HUMAN, 3))
        set_tile(board, 11, 12, Tile(CITY, HUMAN, 4))
        board.player_positions[AI] = (10, 10)
        owned = [(10, 10), (10, 11), (10, 12)]
        land = [(10, 10), (10, 12)]
        ai = GeneralsAI(AI, "hard", 42)

        missions = ai._mission_candidates(
            board,
            board.visibility(AI),
            owned,
            land,
            [(10, 11)],
            (10, 10),
        )
        kinds = {mission.kind for mission in missions}

        self.assertEqual(
            kinds,
            {
                DEFEND_HOME_CITY,
                DEFEND_TERRITORY,
                ATTACK_ENEMY_CITY,
                ATTACK_ENEMY_LAND,
                ATTACK_NEUTRAL_CITY,
                EXPLORE_EXPAND,
            },
        )

    def test_land_gather_triggers_only_when_majority_has_four_army(self) -> None:
        board = blank_board(player_count=2)
        positions = [(10, 10), (11, 10), (10, 11), (11, 11)]
        for position, army in zip(positions, (4, 4, 1, 1)):
            set_tile(board, position[0], position[1], Tile(PLAIN, AI, army))
        ai = GeneralsAI(AI, "hard", 42)

        ai._update_gather_mode(board, board.visibility(AI), positions, [])
        self.assertIsNone(ai.gather_mode)

        set_tile(board, 10, 11, Tile(PLAIN, AI, 4))
        ai._update_gather_mode(board, board.visibility(AI), positions, [])

        self.assertEqual(ai.gather_mode, "land")
        self.assertEqual(ai.gather_army_mode, "all")

    def test_city_gather_uses_pressure_and_chooses_largest_capacity(self) -> None:
        board = blank_board(player_count=2)
        city = (10, 10)
        land = [(9, 10), (10, 9), (9, 9)]
        set_tile(board, 10, 10, Tile(CITY, AI, 12))
        for position in land:
            set_tile(board, position[0], position[1], Tile(PLAIN, AI, 4))
        ai = GeneralsAI(AI, "hard", 42)

        ai._update_gather_mode(
            board,
            board.visibility(AI),
            land,
            [city],
        )
        self.assertEqual(ai.gather_mode, "city")
        self.assertEqual(ai.gather_army_mode, "half")

        set_tile(board, 11, 10, Tile(PLAIN, HUMAN, 20))
        ai._update_gather_mode(
            board,
            board.visibility(AI),
            land,
            [city],
        )
        self.assertEqual(ai.gather_mode, "city")
        self.assertEqual(ai.gather_army_mode, "all")

    def test_land_gather_selects_concrete_path_and_prefers_castle(self) -> None:
        board = blank_board(player_count=2, size=39)
        origin = (10, 10)
        castle = (12, 10)
        rally = (15, 10)
        path_tiles = (
            (10, 10),
            (11, 10),
            (13, 10),
            (14, 10),
            (15, 10),
        )
        for position in path_tiles:
            set_tile(
                board,
                *position,
                Tile(
                    PLAIN,
                    AI,
                    10 if position == origin else 4,
                ),
            )
        set_tile(board, *castle, Tile(CITY, AI, 8))
        ai = GeneralsAI(AI, "hard", 42)
        ai.border_muster_target = rally
        ai.mission = Mission(
            ATTACK_ENEMY_LAND,
            (20, 10),
            10_000,
            "all",
            "test",
        )

        ai._update_gather_mode(
            board,
            board.visibility(AI),
            list(path_tiles),
            [castle],
        )

        self.assertEqual(ai.gather_mode, "land")
        self.assertIsNotNone(ai.gather_route)
        self.assertEqual(ai.gather_route.mode, "land")
        self.assertEqual(ai.gather_route.rally, rally)
        self.assertEqual(ai.gather_route.origin, origin)
        self.assertEqual(ai.gather_route.path[0], origin)
        self.assertEqual(ai.gather_route.path[-1], rally)
        self.assertIn(castle, ai.gather_route.path)

    def test_castle_gather_uses_ratio_path_from_castle_to_rally(self) -> None:
        board = blank_board(player_count=2, size=39)
        castle = (10, 10)
        rally = (15, 10)
        path_tiles = ((11, 10), (12, 10), (13, 10), (14, 10))
        set_tile(board, *castle, Tile(CITY, AI, 12))
        for position in path_tiles:
            set_tile(board, *position, Tile(PLAIN, AI, 1))
        set_tile(board, *rally, Tile(PLAIN, AI, 1))
        set_tile(board, 9, 10, Tile(PLAIN, HUMAN, 2))
        set_tile(board, 10, 9, Tile(PLAIN, HUMAN, 2))
        set_tile(board, 10, 11, Tile(PLAIN, HUMAN, 2))
        ai = GeneralsAI(AI, "hard", 42)
        ai.border_muster_target = rally
        ai.mission = Mission(
            ATTACK_ENEMY_LAND,
            (20, 10),
            10_000,
            "all",
            "test",
        )

        ai._update_gather_mode(
            board,
            board.visibility(AI),
            list(path_tiles),
            [castle],
        )

        self.assertEqual(ai.gather_mode, "city")
        self.assertIsNotNone(ai.gather_route)
        self.assertEqual(ai.gather_route.mode, "castle")
        self.assertEqual(ai.gather_route.origin, castle)
        self.assertEqual(ai.gather_route.rally, rally)
        self.assertEqual(ai.gather_route.path[-1], rally)

    def test_castle_neighbor_pressure_uses_surrounding_average_for_mountain(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=39)
        mountain = (1, 0)
        set_tile(board, *mountain, Tile(MOUNTAIN, NEUTRAL, 0))
        set_tile(board, 0, 0, Tile(MOUNTAIN, NEUTRAL, 0))
        set_tile(board, 2, 0, Tile(PLAIN, HUMAN, 9))
        set_tile(board, 1, 1, Tile(PLAIN, HUMAN, 3))
        ai = GeneralsAI(AI, "hard", 42)

        effective = ai._mountain_effective_army(board, mountain)

        self.assertEqual(effective, 6.0)

    def test_gather_route_move_bonus_prefers_exact_next_path_tile(self) -> None:
        board = blank_board(player_count=2, size=39)
        first = (10, 10)
        second = (11, 10)
        third = (12, 10)
        set_tile(board, *first, Tile(PLAIN, AI, 8))
        set_tile(board, *second, Tile(PLAIN, AI, 2))
        set_tile(board, *third, Tile(PLAIN, AI, 1))
        ai = GeneralsAI(AI, "hard", 42)
        ai.gather_route = GatherRoute(
            "land",
            first,
            third,
            (first, second, third),
            10,
            1,
        )

        on_path = ai._gather_route_move_bonus(
            board,
            Move(first[0], first[1], second[0], second[1], 7),
        )
        off_path = ai._gather_route_move_bonus(
            board,
            Move(first[0], first[1], first[0] + 1, first[1] - 1, 7),
        )

        self.assertGreater(on_path, 0.0)
        self.assertEqual(off_path, 0.0)

    def test_reinforcement_prefers_large_distant_stack_over_small_nearby_tile(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=39)
        near = (20, 20)
        far = (10, 10)
        front = (22, 20)
        set_tile(board, *near, Tile(PLAIN, AI, 8))
        set_tile(board, *far, Tile(PLAIN, AI, 120))
        ai = GeneralsAI(AI, "hard", 42)

        selected = ai._select_reinforcement_stack(
            board,
            [near, far],
            front,
        )

        self.assertEqual(selected, far)

    def test_high_density_region_prefers_sending_troops_to_frontline(self) -> None:
        board = blank_board(player_count=2, size=75)
        front = (20, 20)
        cluster = [(15, 15), (15, 16), (16, 15)]
        isolated = (18, 18)
        for position in cluster:
            set_tile(board, *position, Tile(PLAIN, AI, 20))
        set_tile(board, *isolated, Tile(PLAIN, AI, 20))
        ai = GeneralsAI(AI, "hard", 42)
        ai.last_seen_enemy = (15, 14)

        selected = ai._select_reinforcement_stack(
            board,
            cluster + [isolated],
            front,
        )

        self.assertIsNotNone(selected)
        self.assertIn(selected, cluster)
        self.assertGreater(
            ai._regional_reinforcement_ratio(board, cluster[0]),
            3.0,
        )

    def test_regional_reinforcement_ratio_uses_distance_to_nearest_enemy(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=75)
        for position in ((10, 10), (10, 11), (11, 10)):
            set_tile(board, *position, Tile(PLAIN, AI, 6))
        ai = GeneralsAI(AI, "hard", 42)

        uncontacted_ratio = ai._regional_reinforcement_ratio(board, (10, 10))
        ai.last_seen_enemy = (14, 10)
        distant_ratio = ai._regional_reinforcement_ratio(board, (10, 10))

        self.assertGreater(uncontacted_ratio, 3.0)
        self.assertLess(distant_ratio, uncontacted_ratio)

    def test_idle_region_shortens_effective_distance_in_reinforcement_ratio(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=75)
        region = [(10, 10), (10, 11), (11, 10)]
        for position in region:
            set_tile(board, *position, Tile(PLAIN, AI, 6))
        ai = GeneralsAI(AI, "hard", 42)
        ai.last_seen_enemy = (14, 10)
        region_key = ai._gather_idle_region_key(region[0])
        ai.gather_idle_region_last_move[region_key] = 1

        board.turn = 20
        short_idle_ratio = ai._regional_reinforcement_ratio(
            board,
            region[0],
        )
        board.turn = 201
        long_idle_ratio = ai._regional_reinforcement_ratio(
            board,
            region[0],
        )
        ai._record_gather_activity(board, region[0])
        reset_ratio = ai._regional_reinforcement_ratio(board, region[0])

        self.assertEqual(short_idle_ratio, 1.5)
        self.assertGreater(long_idle_ratio, 3.0)
        self.assertEqual(reset_ratio, 1.5)

    def test_idle_distance_discount_uses_piecewise_steps(self) -> None:
        ai = GeneralsAI(AI, "hard", 42)

        self.assertEqual(ai._idle_distance_discount(0), 0.0)
        self.assertEqual(ai._idle_distance_discount(20), 0.15)
        self.assertEqual(ai._idle_distance_discount(50), 0.30)
        self.assertEqual(ai._idle_distance_discount(100), 0.45)
        self.assertEqual(ai._idle_distance_discount(200), 0.60)

    def test_high_density_region_adds_frontline_move_bonus(self) -> None:
        board = blank_board(player_count=2, size=75)
        source = (10, 10)
        target = (10, 9)
        set_tile(board, *source, Tile(PLAIN, AI, 6))
        for position in ((10, 11), (11, 10)):
            set_tile(board, *position, Tile(PLAIN, AI, 6))
        ai = GeneralsAI(AI, "hard", 42)
        ai.mission = Mission(
            ATTACK_ENEMY_LAND,
            (10, 1),
            10_000,
            "all",
            "test front",
        )
        ai.strategy_state = STRATEGY_ATTACK
        ai.strategy_state_ready = True
        move = Move(source[0], source[1], target[0], target[1], 5)

        bonus = ai._mission_move_bonus(
            board,
            move,
            board.tile(*source),
            board.tile(*target),
            board.visibility(AI),
            1,
            True,
        )

        self.assertGreaterEqual(bonus, 18_000.0)

    def test_attack_starts_when_local_enemy_is_half_of_mustered_army(self) -> None:
        board = blank_board(player_count=2, size=75)
        general = (10, 10)
        frontier = (12, 10)
        rear_stack = (20, 20)
        enemy = (14, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 8))
        set_tile(board, *frontier, Tile(PLAIN, AI, 2))
        set_tile(board, *rear_stack, Tile(PLAIN, AI, 60))
        set_tile(board, *enemy, Tile(PLAIN, HUMAN, 3))
        board.player_positions[AI] = general
        board.player_positions[HUMAN] = (30, 30)
        ai = GeneralsAI(AI, "hard", 42)

        moves = ai.plan_moves(board)

        self.assertEqual(ai.mission.kind, ATTACK_ENEMY_LAND)
        self.assertIsNotNone(ai.border_muster_target)
        self.assertEqual(ai.gather_mode, "border_launch")
        self.assertTrue(ai.border_muster_ready)
        self.assertEqual(ai.border_muster_army_target, 8)
        self.assertTrue(moves)

    def test_border_muster_waits_for_strong_local_enemy(self) -> None:
        board = blank_board(player_count=2, size=75)
        general = (10, 10)
        frontier = (12, 10)
        enemy = (14, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 20))
        set_tile(board, *frontier, Tile(PLAIN, AI, 2))
        set_tile(board, *enemy, Tile(PLAIN, HUMAN, 13))
        board.player_positions[AI] = general
        board.player_positions[HUMAN] = (30, 30)
        ai = GeneralsAI(AI, "hard", 42)

        ai.plan_moves(board)

        self.assertIsNotNone(ai.border_muster_target)
        self.assertFalse(ai.border_muster_ready)
        self.assertEqual(ai.gather_mode, "border")

    def test_major_invasion_doubles_defense_weights(self) -> None:
        board = blank_board(player_count=2, size=75)
        general = (10, 10)
        reserve = (20, 20)
        enemy = (22, 20)
        set_tile(board, *general, Tile(GENERAL, AI, 20))
        set_tile(board, *reserve, Tile(PLAIN, AI, 30))
        set_tile(board, *enemy, Tile(PLAIN, HUMAN, 11))
        board.player_positions[AI] = general
        board.player_positions[HUMAN] = (40, 40)
        owned = [general, reserve]
        ai = GeneralsAI(AI, "hard", 42)

        ai._update_defense_intent(
            board,
            {enemy},
            owned,
            general,
        )
        ai.strategy_state_ready = True
        ai.strategy_state = STRATEGY_DEFENSE
        ai.major_invasion = False
        normal_defense = ai._state_score_multiplier(DEFEND_HOME_CITY)
        normal_territory = ai._state_score_multiplier(DEFEND_TERRITORY)
        normal_attack = ai._state_score_multiplier(ATTACK_ENEMY_LAND)
        ai.major_invasion = True

        self.assertTrue(ai.major_invasion)
        self.assertEqual(ai.major_invasion_player, HUMAN)
        self.assertEqual(ai.major_invasion_army, 11)
        self.assertEqual(
            ai._state_score_multiplier(DEFEND_HOME_CITY),
            normal_defense * 2,
        )
        self.assertEqual(
            ai._state_score_multiplier(DEFEND_TERRITORY),
            normal_territory * 2,
        )
        self.assertEqual(
            ai._state_score_multiplier(ATTACK_ENEMY_LAND),
            normal_attack,
        )

    def test_visible_enemy_general_becomes_priority_all_in_target(self) -> None:
        board = blank_board(player_count=3, size=39)
        ai_general = (10, 10)
        exposed_general = (12, 10)
        other_general = (10, 14)
        set_tile(board, *ai_general, Tile(GENERAL, AI, 50))
        set_tile(board, *exposed_general, Tile(GENERAL, HUMAN, 6))
        set_tile(board, *other_general, Tile(GENERAL, 2, 6))
        board.player_positions[AI] = ai_general
        board.player_positions[HUMAN] = exposed_general
        board.player_positions[2] = other_general
        ai = GeneralsAI(AI, "hard", 42)

        moves = ai.plan_moves(board)

        self.assertEqual(ai.exposed_general_player, HUMAN)
        self.assertEqual(ai.all_in_target, exposed_general)
        self.assertEqual(ai._current_attack_opponent(
            board,
            board.visibility(AI),
        ), HUMAN)
        self.assertEqual(ai._exposed_general_multiplier(HUMAN), 1.5)
        self.assertEqual(ai._exposed_general_multiplier(2), 1.0)
        self.assertTrue(moves)

    def test_ai_switches_mission_only_after_hysteresis_margin(self) -> None:
        board = blank_board(player_count=2)
        target = (10, 10)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 4))
        ai = GeneralsAI(AI, "hard", 42)
        ai.mission = Mission(EXPLORE_EXPAND, target, 5_000, age=10)
        current_candidate = Mission(EXPLORE_EXPAND, target, 5_000)
        small_gain = Mission(ATTACK_ENEMY_LAND, (15, 15), 8_500)

        with mock.patch.object(
            ai,
            "_mission_candidates",
            return_value=[small_gain, current_candidate],
        ):
            mission, changed = ai._update_mission(
                board,
                set(),
                [target],
                [target],
                [],
                target,
            )

        self.assertEqual(mission.kind, EXPLORE_EXPAND)
        self.assertFalse(changed)

        large_gain = Mission(ATTACK_ENEMY_LAND, (15, 15), 10_200)
        with mock.patch.object(
            ai,
            "_mission_candidates",
            return_value=[large_gain, current_candidate],
        ):
            mission, changed = ai._update_mission(
                board,
                set(),
                [target],
                [target],
                [],
                target,
            )

        self.assertEqual(mission.kind, ATTACK_ENEMY_LAND)
        self.assertTrue(changed)

    def test_ai_actively_searches_when_no_enemy_is_visible(self) -> None:
        board = Board(100, 100, 2, 42)
        ai = GeneralsAI(AI, "hard", 42)

        moves = ai.plan_moves(board)

        self.assertTrue(ai.long_range_search)
        self.assertIsNotNone(ai.search_waypoint)
        self.assertEqual(ai.mission.kind, EXPLORE_EXPAND)
        self.assertTrue(moves)
        self.assertTrue(board.move_legal(moves[0], AI))

    def test_enemy_search_target_rotates_instead_of_repeating_visited_cell(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=75)
        ai_general = (10, 10)
        enemy_tile = (40, 40)
        set_tile(board, *ai_general, Tile(GENERAL, AI, 20))
        set_tile(board, *enemy_tile, Tile(PLAIN, HUMAN, 4))
        board.player_positions[AI] = ai_general
        ai = GeneralsAI(AI, "hard", 42)
        ai.planning_owned_tiles = [ai_general]
        ai.known_enemy_tiles[HUMAN] = {enemy_tile}
        ai.enemy_hunt_player = HUMAN

        first = ai._select_enemy_search_target(board, set())
        self.assertIsNotNone(first)
        self.assertNotEqual(first, enemy_tile)
        ai.enemy_search_visited.add(first)

        second = ai._select_enemy_search_target(board, set())

        self.assertIsNotNone(second)
        self.assertNotEqual(second, first)

    def test_sustained_peace_search_outranks_home_area_expansion(self) -> None:
        board = blank_board(player_count=2, size=75)
        general = (10, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 40))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)
        ai.planning_owned_tiles = [general]
        ai.search_waypoint = (55, 55)
        ai.turns_since_contact = 300
        ai.search_pressure = 260.0
        ai.forced_offensive = True
        ai.long_range_search = True
        ai.strategy_state = STRATEGY_EXPLORATION
        ai.strategy_state_ready = True

        missions = ai._mission_candidates(
            board,
            set(),
            [general],
            [general],
            [],
            general,
        )
        best = max(missions, key=lambda mission: mission.score)

        self.assertEqual(best.reason, "active search 300 turns")
        self.assertEqual(best.target, ai.search_waypoint)

    def test_war_tempo_creates_an_active_search_mission_without_enemy_intel(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=75)
        board.turn = 200
        general = (10, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 40))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)
        ai.planning_owned_tiles = [general]
        ai.last_attrition_turn = 0
        ai.war_tempo_cooldown_turns = 0

        ai._update_war_tempo(board, set(), [general], general)

        self.assertIsNotNone(ai.mission)
        self.assertEqual(ai.mission.reason, "war tempo active enemy search")
        self.assertIsNotNone(ai.search_waypoint)
        self.assertTrue(ai.forced_offensive)

    def test_war_tempo_reacts_to_a_gap_in_significant_battles(self) -> None:
        board = blank_board(player_count=4, size=75)
        board.turn = 1_000
        general = (10, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 30))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)
        ai.planning_owned_tiles = [general]
        ai.last_attrition_turn = 999
        ai.last_significant_attrition_turn = 900
        ai.war_tempo_cooldown_turns = 0

        ai._update_war_tempo(
            board,
            board.visibility(AI),
            [general],
            general,
        )

        self.assertTrue(ai.forced_offensive)
        self.assertIsNotNone(ai.mission)
        self.assertIn("war tempo", ai.mission.reason)

    def test_significant_battle_updates_war_tempo_memory(self) -> None:
        board = blank_board(player_count=2, size=39)
        general = (10, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 1_000))
        board.player_positions[AI] = general
        board.turn = 120
        board.last_report.attrition_by_pair = {(AI, HUMAN): 15}
        ai = GeneralsAI(AI, "hard", 42)

        ai._update_attrition_state(board, [general])

        self.assertEqual(ai.last_attrition_turn, 120)
        self.assertEqual(ai.last_significant_attrition_turn, 120)

    def test_significant_battle_escalation_is_late_game_only(self) -> None:
        board = blank_board(player_count=12, size=75)
        board.turn = 1_000
        general = (10, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 30))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)
        ai.last_attrition_turn = 999
        ai.last_significant_attrition_turn = 900
        ai.war_tempo_cooldown_turns = 0
        ai.mission = None

        ai._update_war_tempo(
            board,
            board.visibility(AI),
            [general],
            general,
        )

        self.assertIsNone(ai.mission)
        self.assertFalse(ai.forced_offensive)

    def test_peace_army_leaves_resupply_to_search_outward(self) -> None:
        board = blank_board(player_count=2, size=75)
        position = (10, 10)
        set_tile(board, *position, Tile(PLAIN, AI, 12))
        board.player_positions[AI] = position
        ai = GeneralsAI(AI, "hard", 42)
        ai.planning_owned_tiles = [position]
        ai.army_controller.create_at(board, position, ignore_cooldown=True)
        ai.war_state = "peace"
        ai.turns_since_contact = 120
        ai.search_waypoint = (30, 10)

        moves = ai.army_controller.plan_moves(board, ai)

        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].source, position)
        self.assertEqual(moves[0].target, (11, 10))
        self.assertEqual(ai.army_controller.armies[0].state, ARMY_STATE_ATTACK)

    def test_ai_persists_enemy_general_campaign_after_losing_sight(self) -> None:
        board = blank_board(player_count=2, size=75)
        ai_general = (10, 10)
        enemy_general = (20, 20)
        set_tile(board, *ai_general, Tile(GENERAL, AI, 12))
        set_tile(board, *enemy_general, Tile(GENERAL, HUMAN, 5))
        board.player_positions[AI] = ai_general
        board.player_positions[HUMAN] = enemy_general
        ai = GeneralsAI(AI, "hard", 42)

        ai._update_strategy_state(
            board,
            {enemy_general},
            [ai_general],
            ai_general,
        )
        mission, _ = ai._update_mission(
            board,
            board.visibility(AI),
            [ai_general],
            [ai_general],
            [],
            ai_general,
        )

        self.assertEqual(ai.campaign_target, enemy_general)
        self.assertEqual(ai.campaign_target_player, HUMAN)
        self.assertEqual(mission.kind, ATTACK_ENEMY_CITY)
        self.assertEqual(mission.target, enemy_general)
        self.assertGreater(ai.campaign_hold_turns, 0)

        board.active_players.discard(HUMAN)
        board.turn += 1
        ai._update_strategy_state(
            board,
            board.visibility(AI),
            [ai_general],
            ai_general,
        )

        self.assertIsNone(ai.campaign_target)

    def test_ai_remembers_enemy_territory_after_it_returns_to_fog(self) -> None:
        board = blank_board(player_count=2, size=75)
        ai_general = (10, 10)
        enemy_tile = (20, 20)
        set_tile(board, *ai_general, Tile(GENERAL, AI, 12))
        set_tile(board, *enemy_tile, Tile(PLAIN, HUMAN, 4))
        board.player_positions[AI] = ai_general
        ai = GeneralsAI(AI, "hard", 42)
        ai.planning_owned_tiles = [ai_general]

        ai._update_enemy_intel(board, {enemy_tile})

        self.assertIn(enemy_tile, ai.known_enemy_tiles[HUMAN])
        self.assertIn(enemy_tile, ai.enemy_search_visited)
        self.assertEqual(ai.enemy_hunt_player, HUMAN)
        self.assertIsNotNone(ai.enemy_hunt_target)

        target = ai.enemy_hunt_target
        ai._update_enemy_intel(board, set())

        self.assertIn(enemy_tile, ai.known_enemy_tiles[HUMAN])
        self.assertEqual(ai.enemy_hunt_target, target)

    def test_ai_remembers_all_seen_land_for_capital_search(self) -> None:
        board = blank_board(player_count=2, size=75)
        ai_general = (10, 10)
        enemy_tile = (20, 20)
        seen_neutral = (21, 20)
        other_seen = (20, 21)
        set_tile(board, *ai_general, Tile(GENERAL, AI, 12))
        set_tile(board, *enemy_tile, Tile(PLAIN, HUMAN, 4))
        board.player_positions[AI] = ai_general
        ai = GeneralsAI(AI, "hard", 42)
        ai.planning_owned_tiles = [ai_general]

        ai._update_enemy_intel(
            board,
            {enemy_tile, seen_neutral, other_seen},
        )

        self.assertIn(seen_neutral, ai.seen_land)
        self.assertIn(other_seen, ai.seen_land)
        self.assertNotEqual(
            ai._select_enemy_search_target(board, set()),
            seen_neutral,
        )

        ai._update_enemy_intel(board, set())

        self.assertIn(seen_neutral, ai.seen_land)
        self.assertIn(enemy_tile, ai.seen_land)

    def test_ai_global_capital_search_moves_beyond_seen_land(self) -> None:
        board = blank_board(player_count=2, size=75)
        ai_general = (37, 37)
        set_tile(board, *ai_general, Tile(GENERAL, AI, 12))
        board.player_positions[AI] = ai_general
        ai = GeneralsAI(AI, "hard", 42)
        ai.planning_owned_tiles = [ai_general]
        ai.long_range_search = True
        ai.turns_since_contact = 300
        for y in range(20, 51):
            for x in range(20, 51):
                ai.seen_land.add((x, y))

        ai._refresh_search_waypoint(board, [ai_general], ai_general)

        target = ai.search_waypoint
        self.assertIsNotNone(target)
        self.assertNotIn(target, ai.seen_land)

    def test_two_player_search_uses_passable_tile_when_center_is_mountain(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=75)
        ai_general = (10, 10)
        set_tile(board, *ai_general, Tile(GENERAL, AI, 12))
        center = (board.width // 2, board.height // 2)
        set_tile(board, *center, Tile(MOUNTAIN, NEUTRAL, 0))
        board.player_positions[AI] = ai_general
        ai = GeneralsAI(AI, "hard", 42)
        ai.long_range_search = True

        ai._refresh_search_waypoint(board, [ai_general], ai_general)

        self.assertIsNotNone(ai.search_waypoint)
        self.assertNotEqual(
            board.tile(*ai.search_waypoint).terrain,
            MOUNTAIN,
        )

    def test_ai_global_capital_search_does_not_repeat_seen_target(self) -> None:
        board = blank_board(player_count=2, size=75)
        ai_general = (37, 37)
        set_tile(board, *ai_general, Tile(GENERAL, AI, 12))
        board.player_positions[AI] = ai_general
        ai = GeneralsAI(AI, "hard", 42)
        ai.planning_owned_tiles = [ai_general]
        ai.long_range_search = True
        ai.turns_since_contact = 300
        for y in range(20, 51):
            for x in range(20, 51):
                ai.seen_land.add((x, y))

        ai._refresh_search_waypoint(board, [ai_general], ai_general)
        first_target = ai.search_waypoint
        self.assertIsNotNone(first_target)

        ai.seen_land.add(first_target)
        ai._refresh_search_waypoint(board, [ai_general], ai_general)

        self.assertIsNotNone(ai.search_waypoint)
        self.assertNotEqual(ai.search_waypoint, first_target)
        self.assertNotIn(ai.search_waypoint, ai.seen_land)

    def test_enemy_search_falls_back_to_global_unseen_region(self) -> None:
        board = blank_board(player_count=2, size=75)
        ai_general = (10, 10)
        enemy_tile = (40, 40)
        set_tile(board, *ai_general, Tile(GENERAL, AI, 12))
        set_tile(board, *enemy_tile, Tile(PLAIN, HUMAN, 4))
        board.player_positions[AI] = ai_general
        ai = GeneralsAI(AI, "hard", 42)
        ai.planning_owned_tiles = [ai_general]
        ai.known_enemy_tiles[HUMAN] = {enemy_tile}
        ai.enemy_hunt_player = HUMAN
        for y in range(35, 46):
            for x in range(35, 46):
                ai.seen_land.add((x, y))

        target = ai._select_enemy_search_target(board, set())

        self.assertIsNotNone(target)
        self.assertNotIn(target, ai.seen_land)
        self.assertLessEqual(
            Board.manhattan(target, enemy_tile),
            12,
        )

    def test_ai_prefers_enemy_territory_hunt_over_neutral_expansion(self) -> None:
        board = blank_board(player_count=2, size=75)
        ai_general = (10, 10)
        enemy_tile = (20, 20)
        neutral_city = (30, 30)
        set_tile(board, *ai_general, Tile(GENERAL, AI, 24))
        set_tile(board, *enemy_tile, Tile(PLAIN, HUMAN, 4))
        set_tile(board, *neutral_city, Tile(CITY, NEUTRAL, 5))
        board.player_positions[AI] = ai_general
        ai = GeneralsAI(AI, "hard", 42)
        ai.planning_owned_tiles = [ai_general]
        ai._update_enemy_intel(board, {enemy_tile})
        ai.forced_offensive = True

        missions = ai._mission_candidates(
            board,
            {neutral_city},
            [ai_general],
            [ai_general],
            [],
            ai_general,
        )
        hunt = next(
            mission
            for mission in missions
            if mission.reason == "enemy territory hunt"
        )
        city = next(
            mission
            for mission in missions
            if mission.kind == ATTACK_NEUTRAL_CITY
        )

        self.assertGreater(hunt.score, city.score)
        self.assertEqual(
            abs(hunt.target[0] - enemy_tile[0])
            + abs(hunt.target[1] - enemy_tile[1]),
            1,
        )

    def test_known_enemy_general_switches_off_territory_search(self) -> None:
        board = blank_board(player_count=2, size=75)
        ai_general = (10, 10)
        enemy_general = (20, 20)
        set_tile(board, *ai_general, Tile(GENERAL, AI, 30))
        set_tile(board, *enemy_general, Tile(GENERAL, HUMAN, 6))
        board.player_positions[AI] = ai_general
        board.player_positions[HUMAN] = enemy_general
        ai = GeneralsAI(AI, "hard", 42)
        ai.planning_owned_tiles = [ai_general]

        ai._update_enemy_intel(board, {enemy_general})

        self.assertEqual(ai.known_enemy_generals[HUMAN], enemy_general)
        self.assertEqual(ai.exposed_general_player, HUMAN)
        self.assertIsNone(ai.enemy_hunt_target)
        self.assertEqual(ai.campaign_target, enemy_general)
        self.assertEqual(ai.campaign_target_kind, ATTACK_ENEMY_CITY)
        self.assertEqual(ai._exposed_general_multiplier(HUMAN), 1.5)

        ai._update_enemy_intel(board, set())
        missions = ai._mission_candidates(
            board,
            set(),
            [ai_general],
            [ai_general],
            [],
            ai_general,
        )
        decapitation = next(
            mission
            for mission in missions
            if mission.reason == "known enemy general decapitation"
        )

        self.assertEqual(ai.known_enemy_generals[HUMAN], enemy_general)
        self.assertEqual(ai.campaign_target, enemy_general)
        self.assertFalse(
            any(
                mission.reason == "enemy territory hunt"
                for mission in missions
            )
        )
        self.assertEqual(decapitation.target, enemy_general)

    def test_ai_maintains_twenty_percent_home_guard(self) -> None:
        board = blank_board(player_count=2, size=75)
        general = (10, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 6))
        for position, army in (
            ((20, 10), 20),
            ((20, 11), 20),
            ((20, 12), 20),
        ):
            set_tile(board, *position, Tile(PLAIN, AI, army))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)
        ai.last_contact_turn = 0

        moves = ai.plan_moves(board)

        self.assertEqual(ai.home_guard_target_army, 14)
        self.assertGreater(ai.home_guard_deficit, 0)
        self.assertEqual(ai.mission.kind, DEFEND_HOME_CITY)
        self.assertEqual(len(moves), 1)
        self.assertTrue(board.move_legal(moves[0], AI))

    def test_home_guard_priority_starts_after_first_contact(self) -> None:
        board = blank_board(player_count=2, size=75)
        general = (10, 10)
        reserve = (20, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 6))
        set_tile(board, *reserve, Tile(PLAIN, AI, 60))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)

        ai.plan_moves(board)

        self.assertEqual(ai.home_guard_target_army, 0)
        self.assertEqual(ai.home_guard_hard_target_army, 0)
        self.assertNotEqual(ai.mission.kind, DEFEND_HOME_CITY)

        ai.last_contact_turn = 0
        ai.force_mission_reselect = True
        ai.plan_moves(board)

        self.assertGreater(ai.home_guard_target_army, 0)
        self.assertEqual(ai.mission.kind, DEFEND_HOME_CITY)

    def test_capital_depth_mission_builds_buffer_before_turn_hundred(self) -> None:
        board = blank_board(player_count=2, size=75)
        general = (20, 20)
        set_tile(board, *general, Tile(GENERAL, AI, 8))
        board.player_positions[AI] = general
        owned = [general]
        ai = GeneralsAI(AI, "hard", 42)

        ai._update_defense_intent(
            board,
            board.visibility(AI),
            owned,
            general,
        )
        missions = ai._mission_candidates(
            board,
            board.visibility(AI),
            owned,
            owned,
            [],
            general,
        )

        deepen = [
            mission
            for mission in missions
            if mission.reason.startswith("deepen capital buffer")
        ]
        self.assertEqual(ai.capital_depth, 1)
        self.assertEqual(len(deepen), 1)
        self.assertEqual(Board.manhattan(deepen[0].target, general), 1)

    def test_capital_depth_requirement_ramps_after_hundred_turns(self) -> None:
        ai = GeneralsAI(AI, "hard", 42)

        self.assertEqual(ai._capital_depth_requirement(0), 2)
        self.assertEqual(ai._capital_depth_requirement(1), 2)
        self.assertEqual(ai._capital_depth_requirement(100), 2)
        self.assertEqual(ai._capital_depth_requirement(101), 3)
        self.assertEqual(ai._capital_depth_requirement(200), 3)
        self.assertEqual(ai._capital_depth_requirement(201), 4)
        self.assertEqual(ai._capital_depth_requirement(300), 4)
        self.assertEqual(ai._capital_depth_requirement(301), 5)
        self.assertEqual(ai._capital_depth_requirement(999), 5)

    def test_capital_guard_ratios_follow_nearest_enemy_territory_distance(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=75)
        general = (20, 20)
        radius = 9
        owned: list[tuple[int, int]] = []
        for y in range(general[1] - radius, general[1] + radius + 1):
            for x in range(general[0] - radius, general[0] + radius + 1):
                set_tile(board, x, y, Tile(PLAIN, AI, 1))
                owned.append((x, y))
        set_tile(board, *general, Tile(GENERAL, AI, 20))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)
        ai.last_contact_turn = 0

        ai._update_defense_intent(board, set(), owned, general)

        self.assertEqual(ai.capital_depth, 10)
        ai.capital_enemy_distance = 7
        hard_ratio, preferred_ratio = ai._capital_guard_ratios()
        self.assertEqual(hard_ratio, 0.10)
        self.assertGreater(preferred_ratio, hard_ratio)

        ai.capital_enemy_distance = 9
        hard_ratio, preferred_ratio = ai._capital_guard_ratios()
        self.assertEqual(hard_ratio, 0.05)
        self.assertEqual(preferred_ratio, 0.05)
        with mock.patch.object(
            ai,
            "_nearest_enemy_territory_distance",
            return_value=9,
        ):
            ai._update_defense_intent(board, set(), owned, general)
        self.assertEqual(
            ai.home_guard_hard_target_army,
            math.ceil(sum(board.tile(*pos).army for pos in owned) * 0.05),
        )

        ai.capital_enemy_distance = 16
        hard_ratio, preferred_ratio = ai._capital_guard_ratios()
        self.assertEqual(hard_ratio, 0.02)
        self.assertEqual(preferred_ratio, 0.02)

    def test_ai_clears_discovered_home_intruder(self) -> None:
        board = blank_board(player_count=2, size=75)
        general = (10, 10)
        intruder = (11, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 4))
        set_tile(board, *intruder, Tile(PLAIN, HUMAN, 1))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)
        ai.last_contact_turn = 0

        moves = ai.plan_moves(board)

        self.assertEqual(ai.home_threat_target, intruder)
        self.assertEqual(ai.mission.kind, DEFEND_HOME_CITY)
        self.assertEqual(ai.mission.target, intruder)
        self.assertEqual(moves[0].source, general)
        self.assertEqual(moves[0].target, intruder)
        self.assertEqual(moves[0].amount, 3)
        self.assertEqual(ai.home_support_last_source, general)

    def test_home_support_keeps_clearly_better_local_fight_when_capital_is_safe(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=39)
        general = (10, 10)
        source = (12, 10)
        enemy = (13, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 2))
        set_tile(board, *source, Tile(PLAIN, AI, 100))
        set_tile(board, *enemy, Tile(PLAIN, HUMAN, 2))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)
        ai.mission = Mission(
            DEFEND_HOME_CITY,
            general,
            100_000,
        )
        ai.home_guard_hard_deficit = 5
        ai.home_support_need = 5

        move = ai._forced_home_defense_move(
            board,
            board.visibility(AI),
            [source],
            general,
        )

        self.assertIsNone(move)
        self.assertIsNone(ai.home_support_last_source)
        self.assertGreater(ai.home_support_last_score, 0)

    def test_home_support_never_uses_an_independent_army_cell(self) -> None:
        board = blank_board(player_count=2, size=39)
        general = (10, 10)
        army_source = (11, 10)
        support_source = (12, 11)
        set_tile(board, *general, Tile(GENERAL, AI, 2))
        set_tile(board, *army_source, Tile(PLAIN, AI, 80))
        set_tile(board, *support_source, Tile(PLAIN, AI, 20))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)
        created, _ = ai.army_controller.create_at(
            board,
            army_source,
            ignore_cooldown=True,
        )
        self.assertTrue(created)
        ai.mission = Mission(
            DEFEND_HOME_CITY,
            general,
            100_000,
        )
        ai.home_guard_hard_deficit = 5
        ai.home_support_need = 5

        move = ai._forced_home_defense_move(
            board,
            board.visibility(AI),
            [army_source, support_source],
            general,
        )

        self.assertIsNotNone(move)
        self.assertNotEqual(move.source, army_source)
        self.assertNotIn(move.target, ai.army_controller.positions)

    def test_all_in_archetype_does_not_break_home_support_suspension(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=39)
        general = (10, 10)
        reserve = (12, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 2))
        set_tile(board, *reserve, Tile(PLAIN, AI, 30))
        board.player_positions[AI] = general
        ai = GeneralsAI(
            AI,
            "hard",
            42,
            archetype=ARCHETYPE_ATTACK,
        )
        ai.all_in_target = (30, 30)
        ai.home_guard_hard_deficit = 5
        ai.home_support_need = 5

        move = ai._forced_home_defense_move(
            board,
            board.visibility(AI),
            [reserve],
            general,
        )

        self.assertIsNone(move)

    def test_ai_garrisons_home_before_chasing_nearby_intruder(self) -> None:
        board = blank_board(player_count=2, size=75)
        general = (10, 10)
        intruder = (11, 10)
        reserve = (20, 20)
        set_tile(board, *general, Tile(GENERAL, AI, 2))
        set_tile(board, *intruder, Tile(PLAIN, HUMAN, 1))
        set_tile(board, *reserve, Tile(PLAIN, AI, 20))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)
        ai.last_contact_turn = 0

        moves = ai.plan_moves(board)

        self.assertGreater(ai.home_guard_deficit, 0)
        self.assertEqual(ai.mission.kind, DEFEND_HOME_CITY)
        self.assertEqual(ai.mission.target, general)
        self.assertEqual(moves[0].source, reserve)
        self.assertNotEqual(moves[0].target, intruder)

    def test_ai_moves_adjacent_army_directly_onto_general(self) -> None:
        board = blank_board(player_count=2, size=75)
        general = (10, 10)
        reserve = (11, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 2))
        set_tile(board, *reserve, Tile(PLAIN, AI, 20))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)
        ai.last_contact_turn = 0

        moves = ai.plan_moves(board)

        self.assertGreater(ai.home_guard_deficit, 0)
        self.assertEqual(ai.mission.kind, DEFEND_HOME_CITY)
        self.assertEqual(ai.mission.target, general)
        self.assertEqual(moves[0].source, reserve)
        self.assertEqual(moves[0].target, general)
        self.assertEqual(moves[0].amount, 1)

    def test_ai_calls_far_high_ratio_stack_for_capital_defense(self) -> None:
        board = blank_board(player_count=2, size=39)
        general = (10, 10)
        nearby = (12, 10)
        distant = (20, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 2))
        set_tile(board, *nearby, Tile(PLAIN, AI, 3))
        set_tile(board, *distant, Tile(PLAIN, AI, 96))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)
        ai.home_guard_hard_deficit = 20

        move = ai._forced_home_defense_move(
            board,
            board.visibility(AI),
            [nearby, distant],
            general,
        )

        self.assertIsNotNone(move)
        self.assertEqual(move.source, distant)
        self.assertGreater(
            board.tile(*distant).army
            / Board.manhattan(distant, general),
            8,
        )

    def test_home_support_uses_score_without_a_fixed_candidate_radius(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=39)
        general = (10, 10)
        distant = (14, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 2))
        set_tile(board, *distant, Tile(PLAIN, AI, 7))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)
        ai.mission = Mission(
            DEFEND_HOME_CITY,
            general,
            100_000,
        )
        ai.home_guard_hard_deficit = 5
        ai.home_support_need = 5
        self.assertLess(
            board.tile(*distant).army
            / Board.manhattan(distant, general),
            CAPITAL_REINFORCEMENT_RATIO_THRESHOLD,
        )

        move = ai._forced_home_defense_move(
            board,
            board.visibility(AI),
            [distant],
            general,
        )

        self.assertIsNotNone(move)
        self.assertEqual(move.source, distant)

    def test_capital_depth_rewards_deficit_without_absolute_priority(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=75)
        general = (20, 20)
        set_tile(board, *general, Tile(GENERAL, AI, 8))
        board.player_positions[AI] = general
        owned = [general]
        ai = GeneralsAI(AI, "hard", 42)

        board.turn = 1
        early = ai._capital_depth_target(
            board,
            board.visibility(AI),
            owned,
        )
        board.turn = 301
        late = ai._capital_depth_target(
            board,
            board.visibility(AI),
            owned,
        )

        self.assertIsNotNone(early)
        self.assertIsNotNone(late)
        self.assertGreater(late.score, early.score)
        self.assertLess(late.score, KNOWN_GENERAL_CAMPAIGN_SCORE)
        self.assertLess(late.score, ENEMY_SEARCH_ATTACK_SCORE)

    def test_ai_prioritizes_occupying_home_vision_square(self) -> None:
        board = blank_board(player_count=2, size=75)
        board.vision_radius = 5
        general = (10, 10)
        target = (15, 10)
        outside = (16, 10)
        original_general = board.player_positions[AI]
        set_tile(
            board,
            *original_general,
            Tile(PLAIN, NEUTRAL, 0),
        )
        set_tile(board, *general, Tile(GENERAL, AI, 10))
        set_tile(board, *target, Tile(PLAIN, NEUTRAL, 0))
        set_tile(board, *outside, Tile(PLAIN, NEUTRAL, 0))
        board.player_positions[AI] = general
        owned = [general]
        ai = GeneralsAI(AI, "hard", 42)

        missions = ai._mission_candidates(
            board,
            board.visibility(AI),
            owned,
            owned,
            [],
            general,
        )

        priority = [
            mission
            for mission in missions
            if mission.reason == "priority occupy home vision 5"
            and mission.target == target
        ]
        self.assertEqual(len(priority), 1)
        self.assertEqual(priority[0].kind, EXPLORE_EXPAND)
        self.assertGreaterEqual(priority[0].score, 800_000)
        self.assertFalse(
            any(
                mission.reason == "priority occupy home vision 5"
                and mission.target == outside
                for mission in missions
            )
        )

    def test_ai_all_in_enemy_general_beats_home_threat_mission(self) -> None:
        board = blank_board(player_count=2, size=75)
        board.grid = [
            [Tile(PLAIN, NEUTRAL, 0) for _ in range(board.width)]
            for _ in range(board.height)
        ]
        general = (10, 10)
        enemy_general = (11, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 20))
        set_tile(board, *enemy_general, Tile(GENERAL, HUMAN, 5))
        board.player_positions[AI] = general
        board.player_positions[HUMAN] = enemy_general
        ai = GeneralsAI(AI, "hard", 42)

        moves = ai.plan_moves(board)

        self.assertEqual(ai.all_in_target, enemy_general)
        self.assertEqual(ai.mission.kind, ATTACK_ENEMY_CITY)
        self.assertEqual(ai.mission.target, enemy_general)
        self.assertEqual(moves[0].source, general)
        self.assertEqual(moves[0].target, enemy_general)

    def test_good_neutral_city_can_outrank_home_guard_maintenance(self) -> None:
        board = blank_board(player_count=2, size=75)
        board.grid = [
            [Tile(PLAIN, NEUTRAL, 0) for _ in range(board.width)]
            for _ in range(board.height)
        ]
        general = (10, 10)
        city = (11, 10)
        reserve = (20, 20)
        set_tile(board, *general, Tile(GENERAL, AI, 10))
        set_tile(board, *city, Tile(CITY, NEUTRAL, 5))
        set_tile(board, *reserve, Tile(PLAIN, AI, 50))
        board.player_positions[AI] = general
        owned = [general, reserve]
        ai = GeneralsAI(AI, "hard", 42)
        ai.last_contact_turn = 0

        ai._update_defense_intent(
            board,
            board.visibility(AI),
            owned,
            general,
        )
        missions = ai._mission_candidates(
            board,
            board.visibility(AI),
            owned,
            owned,
            [],
            general,
        )
        home_score = max(
            mission.score
            for mission in missions
            if mission.kind == DEFEND_HOME_CITY
        )
        city_score = max(
            mission.score
            for mission in missions
            if mission.kind == ATTACK_NEUTRAL_CITY
        )

        self.assertGreater(ai.home_guard_deficit, 0)
        self.assertGreater(city_score, home_score)

    def test_passive_defense_raises_enemy_general_commitment(self) -> None:
        board = blank_board(player_count=2, size=100)
        board.grid = [
            [Tile(PLAIN, NEUTRAL, 0) for _ in range(board.width)]
            for _ in range(board.height)
        ]
        general = (10, 10)
        reserve = (20, 20)
        enemy_general = (50, 50)
        set_tile(board, *general, Tile(GENERAL, AI, 8))
        set_tile(board, *reserve, Tile(PLAIN, AI, 30))
        set_tile(board, *enemy_general, Tile(GENERAL, HUMAN, 4))
        board.player_positions[AI] = general
        board.player_positions[HUMAN] = enemy_general
        owned = [general, reserve]
        ai = GeneralsAI(AI, "hard", 42)
        ai.campaign_target = enemy_general
        ai.campaign_target_player = HUMAN
        ai.campaign_target_kind = ATTACK_ENEMY_CITY
        ai.campaign_hold_turns = 180
        ai.passive_defense_turns = 900

        missions = ai._mission_candidates(
            board,
            board.visibility(AI),
            owned,
            owned,
            [],
            general,
        )
        score = max(
            mission.score
            for mission in missions
            if mission.reason == "persistent enemy general campaign"
        )
        distance = Board.manhattan(general, enemy_general)
        base_score = 190_000 - distance * 90

        self.assertGreaterEqual(score, base_score + 80_000)

    def test_ai_never_idles_while_it_has_a_movable_army(self) -> None:
        board = blank_board(player_count=2)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 6))
        set_tile(board, 11, 10, Tile(MOUNTAIN, NEUTRAL, 0))
        ai = GeneralsAI(AI, "hard", 42)

        with (
            mock.patch.object(ai, "_mission_candidates", return_value=[]),
            mock.patch.object(ai, "_update_defense_intent"),
        ):
            moves = ai.plan_moves(board)

        self.assertEqual(len(moves), 1)
        self.assertTrue(board.move_legal(moves[0], AI))

    def test_home_guard_cap_does_not_idle_a_movable_field_stack(self) -> None:
        board = blank_board(player_count=2)
        general = (10, 10)
        field = (12, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 20))
        set_tile(board, *field, Tile(PLAIN, AI, 3))
        board.player_positions[AI] = general
        ai = GeneralsAI(AI, "hard", 42)
        ai.last_contact_turn = 0
        ai.home_guard_hard_target_army = 20

        with (
            mock.patch.object(ai, "_mission_candidates", return_value=[]),
            mock.patch.object(ai, "_update_defense_intent"),
        ):
            moves = ai.plan_moves(board)

        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].source, field)
        self.assertTrue(board.move_legal(moves[0], AI))

    def test_expansion_shape_prefers_branching_frontier_over_corridor(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=20)
        corridor = (4, 7)
        branch = (10, 7)
        for position in (
            (3, 6),
            (4, 6),
            (5, 6),
            (3, 7),
            (3, 8),
            (4, 8),
            (5, 8),
        ):
            set_tile(board, *position, Tile(MOUNTAIN, NEUTRAL, 0))
        visible = {
            (x, y)
            for y in range(board.height)
            for x in range(board.width)
        }
        ai = GeneralsAI(AI, "hard", 42)

        corridor_score = ai._expansion_shape_score(
            board,
            *corridor,
            visible,
            AI,
        )
        branch_score = ai._expansion_shape_score(
            board,
            *branch,
            visible,
            AI,
        )

        self.assertGreater(branch_score, corridor_score * 3)

    def test_local_expansion_splits_troops_but_forced_offensive_does_not(
        self,
    ) -> None:
        board = blank_board(player_count=2, size=27)
        general = (5, 5)
        source = (10, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 2))
        board.player_positions[AI] = general
        set_tile(board, *source, Tile(PLAIN, AI, 10))
        board.state_revision += 1
        ai = GeneralsAI(AI, "hard", 42)
        ai.mission = Mission(EXPLORE_EXPAND, (26, 26), 20_000)

        moves = ai.plan_moves(board)

        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].source, source)
        self.assertEqual(moves[0].amount, 5)

        forced = GeneralsAI(AI, "hard", 43)
        forced.mission = Mission(EXPLORE_EXPAND, (26, 26), 200_000)
        forced.forced_offensive = True
        forced.primary_stack = source

        forced_moves = forced.plan_moves(board)

        self.assertEqual(len(forced_moves), 1)
        self.assertEqual(forced_moves[0].source, source)
        self.assertEqual(forced_moves[0].amount, 9)

    def test_ai_must_leave_after_reaching_muster_threshold(self) -> None:
        board = blank_board(player_count=2, size=75)
        general = board.general_position(AI)
        self.assertIsNotNone(general)
        set_tile(board, *general, Tile(GENERAL, AI, 30))
        set_tile(board, general[0] + 1, general[1], Tile(PLAIN, AI, 30))
        set_tile(board, 10, 10, Tile(PLAIN, AI, 60))
        for position in ((20, 20), (21, 20), (20, 21)):
            set_tile(board, position[0], position[1], Tile(PLAIN, AI, 4))
        set_tile(board, 22, 20, Tile(CITY, AI, 12))
        ai = GeneralsAI(AI, "hard", 42)

        moves = ai.plan_moves(board)

        self.assertTrue(ai.forced_offensive)
        self.assertIsNone(ai.gather_mode)
        self.assertEqual(ai.mission.kind, EXPLORE_EXPAND)
        self.assertFalse(ai.force_mission_reselect)
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].source, (10, 10))
        self.assertNotEqual(moves[0].target, (10, 10))
        self.assertEqual(ai.primary_stack, moves[0].target)
        self.assertTrue(board.move_legal(moves[0], AI))

    def test_twenty_four_ai_players_remain_legal_during_play(self) -> None:
        board = Board(75, 75, 24, 314)
        ais = [GeneralsAI(player, "hard", 700 + player) for player in range(24)]
        for _ in range(6):
            if board.winner is not None:
                break
            moves = []
            for ai in ais:
                moves.extend(ai.plan_moves(board))
            self.assertTrue(all(board.move_legal(move) for move in moves))
            board.resolve_turn(moves)
        self.assertEqual(len(board.active_players), 24)
        self.assertTrue(all(board.stats(player)["tiles"] > 1 for player in range(24)))


class TrainingTests(unittest.TestCase):
    def test_genome_is_bounded_and_mutation_changes_weights(self) -> None:
        genome = AIGenome.from_mapping(
            {"aggression": 99.0, "defense": -99.0}
        )

        self.assertEqual(
            genome.aggression,
            AIGenome.GENE_BOUNDS["aggression"][1],
        )
        self.assertEqual(
            genome.defense,
            AIGenome.GENE_BOUNDS["defense"][0],
        )
        mutated = genome.mutate(
            __import__("random").Random(7),
            rate=1.0,
            scale=0.2,
            reset_chance=0.0,
        )
        self.assertNotEqual(mutated.to_dict(), genome.to_dict())

    def test_training_match_persists_history_champion_and_population(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            engine = TrainingEngine(
                data_dir,
                seed=13,
                population_size=12,
                map_size=20,
                max_turns=2,
            )
            result = engine.run_match(seed=19, max_turns=2)

            self.assertEqual(engine.completed_matches, 1)
            self.assertEqual(result["match"], 1)
            self.assertEqual(len(engine.population), 12)
            self.assertTrue((data_dir / "manifest.json").exists())
            self.assertTrue((data_dir / "champion.json").exists())
            self.assertTrue((data_dir / "population.json").exists())
            self.assertTrue((data_dir / "history.jsonl").exists())

            restored = TrainingEngine(
                data_dir,
                seed=99,
                population_size=12,
                map_size=20,
                max_turns=2,
            )
            self.assertEqual(restored.completed_matches, 1)
            self.assertEqual(restored.generation, 0)
            self.assertEqual(len(restored.history), 1)
            self.assertEqual(len(restored.pending_results), 1)
            self.assertEqual(len(restored.population), 12)

    def test_training_supports_six_player_evaluation_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            engine = TrainingEngine(
                data_dir,
                seed=17,
                population_size=6,
                map_size=20,
                max_turns=2,
                parallel_workers=1,
                live_render=False,
            )
            training_match = engine._new_match(
                seed=19,
                shuffle_genomes=False,
                player_count=6,
                mode="ffa",
            )
            self.assertTrue(
                all(
                    ai.archetype == ARCHETYPE_STANDARD
                    for ai in training_match.ais.values()
                )
            )
            result = TrainingEngine.simulate_episode(
                [genome.to_dict() for genome in engine.population],
                seed=23,
                map_size=20,
                max_turns=2,
                player_count=6,
                genome_indices=list(range(6)),
                source="mechanism_evaluation",
                collect_diagnostics=True,
            )

            self.assertEqual(result["player_count"], 6)
            self.assertEqual(result["source"], "mechanism_evaluation")
            self.assertEqual(len(result["results"]), 6)
            self.assertTrue(
                all("strategy_state_turns" in item for item in result["results"])
            )

            engine.record_training_batch([result], batch_elapsed=0.01)

            self.assertEqual(engine.completed_matches, 1)
            self.assertEqual(engine.generation, 0)
            for offset in range(1, 5):
                repeated = json.loads(json.dumps(result))
                repeated["seed"] = int(result["seed"]) + offset
                engine.record_training_batch([repeated], batch_elapsed=0.01)
            self.assertEqual(engine.completed_matches, 5)
            self.assertEqual(engine.generation, 1)
            restored = TrainingEngine(
                data_dir,
                seed=99,
                population_size=6,
                map_size=20,
                max_turns=2,
                parallel_workers=1,
                live_render=False,
            )
            self.assertEqual(restored.completed_matches, 5)
            self.assertEqual(restored.history[-1]["source"], "mechanism_evaluation")

    def test_recent_champion_uses_last_hundred_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            engine = TrainingEngine(
                Path(temp),
                seed=29,
                population_size=2,
                map_size=20,
                max_turns=2,
                parallel_workers=1,
                live_render=False,
            )
            old_genome = AIGenome.baseline().to_dict()
            old_genome["aggression"] = 1.45
            recent_genome = AIGenome.baseline().to_dict()
            recent_genome["aggression"] = 1.30
            engine.history = [
                {
                    "match": 1,
                    "generation": 0,
                    "mode": "ffa",
                    "results": [
                        {
                            "fitness": 999_999.0,
                            "genome": old_genome,
                        }
                    ],
                },
                *[
                    {
                        "match": index + 2,
                        "generation": 0,
                        "mode": "ffa",
                        "results": [
                            {
                                "fitness": float(index),
                                "genome": (
                                    recent_genome
                                    if index == 99
                                    else old_genome
                                ),
                            }
                        ],
                    }
                    for index in range(100)
                ],
            ]

            self.assertTrue(engine._refresh_recent_champion())
            self.assertAlmostEqual(engine.best_fitness, 99.0)
            self.assertAlmostEqual(engine.champion.aggression, 1.30)
            self.assertNotEqual(engine.champion.aggression, 1.45)

    def test_duel_pairs_inject_recent_champion_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            engine = TrainingEngine(
                Path(temp),
                seed=37,
                population_size=4,
                map_size=20,
                max_turns=2,
                parallel_workers=1,
                live_render=False,
            )
            pairs = engine._next_duel_pairs(8)
            self.assertEqual(len(pairs), 8)
            self.assertTrue(all(first == 0 for first, _second in pairs))
            self.assertTrue(
                all(1 <= second < engine.population_size for _first, second in pairs)
            )

    def test_training_loads_legacy_hybrid_manifest_without_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            legacy_match = {
                "best_fitness": 123456.0,
                "results": [{"mode": "ffa", "fitness": 123456.0}],
            }
            (data_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "generation": 7,
                        "completed_matches": 42,
                        "best_fitness": 123456.0,
                        "best_match": legacy_match,
                    }
                ),
                encoding="utf-8",
            )

            engine = TrainingEngine(
                data_dir,
                seed=31,
                population_size=2,
                map_size=20,
                max_turns=2,
                parallel_workers=1,
                live_render=False,
            )

            self.assertEqual(engine.best_fitness, 123456.0)
            self.assertEqual(engine.best_match, legacy_match)
            self.assertEqual(engine.generation, 7)
            self.assertEqual(engine.completed_matches, 42)

    def test_training_live_snapshot_exposes_board_progress(self) -> None:
        map_size = 20
        slot_ints = training_module.LIVE_SLOT_HEADER + map_size * map_size * 2
        memory = shared_memory.SharedMemory(
            create=True,
            size=slot_ints * 4,
        )
        with tempfile.TemporaryDirectory() as temp:
            engine = TrainingEngine(
                Path(temp),
                seed=41,
                population_size=2,
                map_size=map_size,
                max_turns=1,
                parallel_workers=1,
                live_render=True,
            )
            engine._shared_memory = memory
            engine._live_worker_count = 1
            try:
                TrainingEngine.simulate_episode(
                    [AIGenome.baseline().to_dict()],
                    seed=43,
                    map_size=map_size,
                    max_turns=1,
                    shared_name=memory.name,
                )
                view = engine.live_view()

                self.assertIsNotNone(view)
                self.assertEqual(view["width"], map_size)
                self.assertEqual(view["height"], map_size)
                self.assertEqual(len(view["tiles"]), map_size * map_size)
                self.assertIsNotNone(view["stamp"])
            finally:
                cached = training_module._LIVE_MEMORY_HANDLES.pop(
                    memory.name,
                    None,
                )
                if cached is not None:
                    cached.close()
                memory.close()
                memory.unlink()

    def test_training_live_snapshot_uses_large_mixed_map_stride(self) -> None:
        base_map_size = 20
        live_map_size = 25
        slot_ints = (
            training_module.LIVE_SLOT_HEADER
            + live_map_size * live_map_size * 2
        )
        memory = shared_memory.SharedMemory(
            create=True,
            size=slot_ints * 4,
        )
        with tempfile.TemporaryDirectory() as temp:
            engine = TrainingEngine(
                Path(temp),
                seed=43,
                population_size=4,
                map_size=base_map_size,
                live_map_size=live_map_size,
                max_turns=1,
                parallel_workers=1,
                live_render=True,
            )
            engine._shared_memory = memory
            engine._live_worker_count = 1
            try:
                TrainingEngine.simulate_episode(
                    [AIGenome.baseline().to_dict()],
                    seed=47,
                    map_size=live_map_size,
                    live_map_size=live_map_size,
                    max_turns=1,
                    shared_name=memory.name,
                )
                view = engine.live_view()

                self.assertIsNotNone(view)
                self.assertEqual(view["width"], live_map_size)
                self.assertEqual(view["height"], live_map_size)
                self.assertEqual(
                    len(view["tiles"]),
                    live_map_size * live_map_size,
                )
            finally:
                cached = training_module._LIVE_MEMORY_HANDLES.pop(
                    memory.name,
                    None,
                )
                if cached is not None:
                    cached.close()
                memory.close()
                memory.unlink()

    def test_live_view_uses_allocated_slots_not_cpu_count(self) -> None:
        map_size = 20
        slot_ints = training_module.LIVE_SLOT_HEADER + map_size * map_size * 2
        worker_slots = 12
        memory = shared_memory.SharedMemory(
            create=True,
            size=slot_ints * worker_slots * 4,
        )
        with tempfile.TemporaryDirectory() as temp:
            engine = TrainingEngine(
                Path(temp),
                seed=71,
                population_size=12,
                map_size=map_size,
                max_turns=1,
                parallel_workers=24,
                live_render=True,
            )
            engine._shared_memory = memory
            engine._live_worker_count = worker_slots
            try:
                self.assertIsNone(engine.live_view())
            finally:
                memory.close()
                memory.unlink()

    def test_live_view_rejects_a_truncated_shared_buffer(self) -> None:
        memory = shared_memory.SharedMemory(create=True, size=16)
        with tempfile.TemporaryDirectory() as temp:
            engine = TrainingEngine(
                Path(temp),
                seed=79,
                population_size=2,
                map_size=20,
                max_turns=1,
                parallel_workers=1,
                live_render=True,
            )
            engine._shared_memory = memory
            engine._live_worker_count = 1
            try:
                self.assertIsNone(engine.live_view())
            finally:
                memory.close()
                memory.unlink()

    def test_training_board_ignores_an_incomplete_snapshot(self) -> None:
        app = route_app()
        app._draw_training_board(
            pygame.Rect(0, 0, 300, 300),
            view={
                "width": 20,
                "height": 20,
                "tiles": [(PLAIN, NEUTRAL)] * 3,
            },
        )

    def test_training_live_slot_cycles_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            engine = TrainingEngine(
                Path(temp),
                seed=73,
                population_size=4,
                map_size=20,
                max_turns=1,
                parallel_workers=1,
                live_render=False,
            )
            engine._live_worker_count = 3
            self.assertEqual(engine.live_slot, 0)
            self.assertEqual(engine.cycle_live_slot(), 1)
            self.assertEqual(engine.cycle_live_slot(), 2)
            self.assertEqual(engine.cycle_live_slot(), 0)
            snapshot = engine.snapshot()
            self.assertEqual(snapshot["live_slot"], 0)
            self.assertEqual(snapshot["live_slot_count"], 3)

    def test_training_preview_click_cycles_parallel_slot(self) -> None:
        app = route_app()
        app.training._live_worker_count = 3
        app.state = "TRAINING"
        app.draw_training()
        self.assertTrue(app.training_preview_rect.collidepoint(
            app.training_preview_rect.center
        ))
        app.handle_training_click(app.training_preview_rect.center)
        self.assertEqual(app.training.live_slot, 1)

    def test_hybrid_duel_episode_tracks_genome_indices(self) -> None:
        genome = AIGenome.baseline().to_dict()
        result = TrainingEngine.simulate_episode(
            [genome, genome, genome, genome],
            seed=47,
            map_size=20,
            max_turns=3,
            mode="duel",
            genome_indices=[0, 1, 2, 3],
        )

        self.assertEqual(result["mode"], "duel")
        self.assertEqual(result["match_count"], 2)
        self.assertEqual(
            {int(item["genome_index"]) for item in result["results"]},
            {0, 1, 2, 3},
        )

    def test_hybrid_fitness_weights_ffa_and_duel(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            engine = TrainingEngine(
                Path(temp),
                seed=53,
                population_size=12,
                map_size=20,
                max_turns=1,
            )
            genome = AIGenome.baseline().to_dict()
            result = engine._aggregate_results(
                [
                    {
                        "mode": "ffa",
                        "results": [
                            {
                                "player": 0,
                                "genome_index": 0,
                                "genome": genome,
                                "fitness": 100.0,
                                "rank": 1,
                                "tiles": 10,
                                "armies": 20,
                                "winner": False,
                            }
                        ],
                    },
                    {
                        "mode": "duel",
                        "results": [
                            {
                                "player": 0,
                                "genome_index": 0,
                                "genome": genome,
                                "fitness": 200.0,
                                "rank": 1,
                                "tiles": 5,
                                "armies": 8,
                                "winner": False,
                            }
                        ],
                    },
                ]
            )

            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(float(result[0]["fitness"]), 130.0)
            self.assertEqual(result[0]["samples"], {"ffa": 1, "duel": 1})

    def test_live_match_results_are_persisted_and_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            engine = TrainingEngine(
                data_dir,
                seed=59,
                population_size=12,
                map_size=20,
                max_turns=1,
            )
            genome = engine.champion.to_dict()
            recorded = engine.record_live_match(
                [
                    {
                        "player": 1,
                        "rank": 1,
                        "winner": True,
                        "tiles": 80,
                        "armies": 240,
                        "genome": genome,
                    }
                ],
                turn=120,
                player_count=3,
                seed=61,
            )

            self.assertIsNotNone(recorded)
            self.assertTrue(
                (data_dir / "online_history.jsonl").exists()
            )
            mapped = engine._online_evaluation_items()
            self.assertEqual(len(mapped), 1)
            self.assertEqual(mapped[0]["results"][0]["genome_index"], 0)

            restored = TrainingEngine(
                data_dir,
                seed=67,
                population_size=12,
                map_size=20,
                max_turns=1,
            )
            self.assertEqual(len(restored.online_history), 1)
            self.assertEqual(restored.history[-1]["mode"], "online")

    def test_champion_genome_is_used_for_new_human_matches(self) -> None:
        app = route_app()
        champion = AIGenome.from_mapping(
            {"aggression": 1.4, "expansion": 1.25}
        )
        app.training.champion = champion
        app.difficulty_key = "hard"
        app.player_count = 3
        app.board_size = 20

        app.new_game(seed=23)

        self.assertEqual(app.ais[1].genome, champion)
        self.assertEqual(app.ais[2].genome, champion)
        self.assertEqual(app.human_ai.genome, champion)

        app.difficulty_key = "normal"
        app.new_game(seed=24)

        self.assertEqual(app.ais[1].genome, AIGenome.baseline())
        self.assertEqual(app.human_ai.genome, AIGenome.baseline())

    def test_training_menu_and_page_are_available(self) -> None:
        app = route_app()
        app.training.parallel_workers = 1
        app.state = "MENU"
        app.draw_menu()

        app.handle_menu_click(app.menu_training_rect.center)

        self.assertEqual(app.state, "TRAINING")
        self.assertTrue(app.training.running)
        app.draw_training()
        app.training.pause()
        app.handle_training_click(app.training_back_rect.center)
        self.assertEqual(app.state, "MENU")


class IndependentArmyTests(unittest.TestCase):
    def tearDown(self) -> None:
        pygame.quit()

    def test_creation_respects_limit_cooldown_and_general_restriction(self) -> None:
        board = blank_board()
        ai = GeneralsAI(AI, "normal", 1)
        controller = ai.army_controller

        general = board.general_position(AI)
        self.assertIsNotNone(general)
        created, message = controller.create_at(board, general)
        self.assertFalse(created)
        self.assertIn("主城", message)

        for x in range(10, 14):
            set_tile(board, x, 10, Tile(PLAIN, AI, 3))
            board.state_revision += 1
        set_tile(board, 14, 10, Tile(PLAIN, AI, 7_000))
        board.state_revision += 1
        for x in range(10, 14):
            created, _ = controller.create_at(
                board,
                (x, 10),
                ignore_cooldown=True,
                enforce_army_limit=True,
            )
            self.assertTrue(created)

        self.assertEqual(controller.count, 4)
        self.assertEqual(
            [army.unit_id for army in controller.armies],
            [1, 2, 3, 4],
        )
        created, message = controller.create_at(
            board,
            (14, 10),
            enforce_army_limit=True,
        )
        self.assertFalse(created)
        self.assertIn("最多", message)
        self.assertEqual(
            controller.creation_cooldown,
            INDEPENDENT_ARMY_CREATION_COOLDOWN,
        )

    def test_main_army_tag_follows_moves_and_expires_after_idle_turns(self) -> None:
        board = blank_board()
        ai = GeneralsAI(AI, "normal", 31)
        controller = ai.army_controller
        set_tile(board, 10, 10, Tile(PLAIN, AI, 5))
        set_tile(board, 11, 10, Tile(PLAIN, AI, 4))
        board.turn = 1

        self.assertTrue(controller.mark_main_army(board, (10, 10)))
        controller.record_main_army_moves(
            board,
            [Move(10, 10, 11, 10, 4)],
        )

        self.assertEqual(controller.main_army_position, (11, 10))
        board.turn = 11
        controller.sync_main_army_tag(board)
        self.assertEqual(controller.main_army_position, (11, 10))
        board.turn = 12
        controller.sync_main_army_tag(board)
        self.assertIsNone(controller.main_army_position)

    def test_main_ai_move_activity_creates_the_hidden_main_army_tag(
        self,
    ) -> None:
        board = blank_board()
        ai = GeneralsAI(AI, "normal", 32)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 5))
        move = Move(10, 10, 11, 10, 4)

        ai._record_main_move_activity(board, move)

        self.assertTrue(ai.army_controller.has_main_army_tag((10, 10)))
        self.assertNotIn((10, 10), ai.army_controller.positions)

    def test_main_army_tag_can_be_reinforced_by_an_independent_army(self) -> None:
        board = blank_board()
        ai = GeneralsAI(AI, "normal", 32)
        controller = ai.army_controller
        set_tile(board, 10, 10, Tile(PLAIN, AI, 10))
        set_tile(board, 11, 10, Tile(PLAIN, AI, 5))
        board.turn = 1

        self.assertTrue(controller.mark_main_army(board, (10, 10)))
        self.assertTrue(
            controller.create_at(
                board,
                (11, 10),
                ignore_cooldown=True,
            )[0]
        )
        army = controller.armies[0]
        controller._record_army_move(army, (10, 10), board)
        board.resolve_movement([Move(11, 10, 10, 10, 4)])
        controller.sync(board, ai)

        self.assertEqual(controller.main_army_position, (10, 10))
        self.assertEqual(controller.count, 0)
        self.assertEqual(board.tile(10, 10).army, 14)

    def test_main_army_tag_allows_army_to_enter_own_general(self) -> None:
        board = blank_board()
        ai = GeneralsAI(AI, "normal", 33)
        controller = ai.army_controller
        general = (10, 10)
        set_tile(board, *general, Tile(GENERAL, AI, 8))
        set_tile(board, 11, 10, Tile(PLAIN, AI, 6))
        board.player_positions[AI] = general

        self.assertTrue(controller.mark_main_army(board, general))
        move = controller._move_toward(
            board,
            ai,
            (11, 10),
            general,
            5,
        )

        self.assertIsNotNone(move)
        self.assertEqual(move.target, general)

    def test_main_army_tag_is_removed_when_the_stack_becomes_an_army(
        self,
    ) -> None:
        board = blank_board()
        ai = GeneralsAI(AI, "normal", 34)
        controller = ai.army_controller
        set_tile(board, 10, 10, Tile(PLAIN, AI, 5))
        board.turn = 1

        self.assertTrue(controller.mark_main_army(board, (10, 10)))
        self.assertTrue(
            controller.create_at(
                board,
                (10, 10),
                ignore_cooldown=True,
            )[0]
        )

        self.assertIsNone(controller.main_army_position)
        self.assertEqual(controller.count, 1)

    def test_army_limit_scales_with_total_army_for_human_and_ai(self) -> None:
        for total, expected in (
            (0, 4),
            (7_999, 4),
            (8_000, 5),
            (15_999, 5),
            (16_000, 6),
            (79_999, 13),
            (80_000, 14),
            (1_000_000, 129),
        ):
            self.assertEqual(independent_army_limit(total), expected)
        self.assertEqual(AI_INDEPENDENT_ARMY_TROOP_DIVISOR, 8_000)
        for total, expected in (
            (0, 4),
            (7_999, 4),
            (8_000, 5),
            (15_999, 5),
            (16_000, 6),
            (79_999, 13),
            (80_000, 14),
        ):
            self.assertEqual(
                independent_army_limit(
                    total,
                    AI_INDEPENDENT_ARMY_TROOP_DIVISOR,
                ),
                expected,
            )

        board = blank_board()
        ai = GeneralsAI(AI, "normal", 21)
        for x in (10, 11, 12, 13):
            set_tile(board, x, 10, Tile(PLAIN, AI, 3))
            self.assertTrue(
                ai.army_controller.create_at(
                    board,
                    (x, 10),
                    ignore_cooldown=True,
                )[0]
            )
        self.assertEqual(ai.army_controller.count, 4)
        ai.army_controller.creation_cooldown = 0
        board.turn = 200
        set_tile(board, 14, 10, Tile(PLAIN, AI, 7_000))
        board.state_revision += 1
        self.assertFalse(ai.army_controller.maybe_create_auto(board, ai))
        set_tile(board, 14, 10, Tile(PLAIN, AI, 8_000))
        board.state_revision += 1
        self.assertTrue(ai.army_controller.maybe_create_auto(board, ai))
        self.assertEqual(ai.army_controller.count, 5)

    def test_ai_army_creation_limit_ramps_until_turn_two_hundred(self) -> None:
        self.assertEqual(ai_army_creation_limit(0, 4), 1)
        self.assertEqual(ai_army_creation_limit(50, 4), 1)
        self.assertEqual(ai_army_creation_limit(100, 4), 2)
        self.assertEqual(ai_army_creation_limit(150, 4), 3)
        self.assertEqual(ai_army_creation_limit(199, 4), 3)
        self.assertEqual(ai_army_creation_limit(200, 4), 4)
        self.assertEqual(ai_army_creation_limit(100, 9), 5)
        self.assertEqual(ai_army_creation_limit(200, 9), 9)

    def test_early_ai_army_limit_trims_extra_armies(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 28)
        for x in range(10, 13):
            set_tile(board, x, 10, Tile(PLAIN, AI, 5))
            self.assertTrue(
                ai.army_controller.create_at(
                    board,
                    (x, 10),
                    ignore_cooldown=True,
                )[0]
            )

        self.assertEqual(ai.army_controller.count, 3)
        ai.army_controller.review_ai_armies(board, ai)
        self.assertEqual(ai.army_controller.count, 1)

    def test_main_ai_merges_small_stacks_to_bootstrap_army(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 27)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 2))
        set_tile(board, 11, 10, Tile(PLAIN, AI, 2))
        board.state_revision += 1

        moves = ai.plan_moves(board, phase=0)
        main_moves = [
            move
            for move in moves
            if move.source not in ai.army_controller.positions
        ]

        self.assertTrue(main_moves)
        self.assertIn(
            (main_moves[0].source, main_moves[0].target),
            {((10, 10), (11, 10)), ((11, 10), (10, 10))},
        )

    def test_late_game_low_roster_army_is_preserved_for_resupply(self) -> None:
        board = blank_board(size=39)
        board.turn = 200
        ai = GeneralsAI(AI, "hard", 29)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 5))
        set_tile(board, 11, 10, Tile(PLAIN, HUMAN, 50))
        set_tile(board, 4, 10, Tile(PLAIN, AI, 30))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        ai.army_controller.update_conservation_mode(board, 4)

        move = ai.army_controller._plan_army_move(
            board,
            ai,
            ai.army_controller.armies[0],
            set(),
            set(),
            (),
            (),
        )

        self.assertIsNotNone(move)
        self.assertEqual(ai.army_controller.armies[0].state, ARMY_STATE_RESUPPLY)
        self.assertNotEqual(move.target, (11, 10))

    def test_ai_creates_extra_armies_when_total_army_scales_past_eight_thousand(
        self,
    ) -> None:
        board = blank_board(size=39)
        board.turn = 200
        ai = GeneralsAI(AI, "hard", 22)
        for x in range(5, 25):
            set_tile(board, x, 10, Tile(PLAIN, AI, 6_400))
        board.state_revision += 1
        expected_limit = independent_army_limit(
            sum(
                board.tile(*position).army
                for position in ai._positions_by_owner(board)[AI]
            ),
            AI_INDEPENDENT_ARMY_TROOP_DIVISOR,
        )
        self.assertEqual(expected_limit, 20)

        for expected_count in range(1, expected_limit + 1):
            ai.army_controller.creation_cooldown = 0
            self.assertTrue(ai.army_controller.maybe_create_auto(board, ai))
            self.assertEqual(
                ai.army_controller.count,
                expected_count,
            )

        ai.army_controller.creation_cooldown = 0
        self.assertFalse(ai.army_controller.maybe_create_auto(board, ai))
        self.assertEqual(ai.army_controller.count, expected_limit)

    def test_target_strength_excludes_tiles_that_become_armies(self) -> None:
        board = blank_board()
        ai = GeneralsAI(AI, "normal", 2)
        controller = ai.army_controller
        set_tile(board, 10, 10, Tile(PLAIN, AI, 30))
        set_tile(board, 11, 10, Tile(PLAIN, AI, 20))
        set_tile(board, 12, 10, Tile(GENERAL, AI, 999))

        created, _ = controller.create_at(board, (10, 10))
        self.assertTrue(created)

        self.assertEqual(controller.target_strength(board, (10, 10)), 50)

    def test_ai_titles_weak_spread_stack_when_minimum_slots_available(self) -> None:
        board = blank_board()
        board.turn = 200
        ai = GeneralsAI(AI, "normal", 3)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 4))
        set_tile(board, 11, 10, Tile(PLAIN, AI, 4))
        board.state_revision += 1

        self.assertTrue(ai.army_controller.maybe_create_auto(board, ai))
        self.assertEqual(ai.army_controller.count, 1)
        self.assertEqual(ai.army_controller.armies[0].position, (10, 10))

        set_tile(board, 10, 10, Tile(PLAIN, AI, 10040))
        set_tile(board, 11, 10, Tile(PLAIN, AI, 8))
        board.state_revision += 1
        ai.army_controller.creation_cooldown = 0
        self.assertTrue(ai.army_controller.maybe_create_auto(board, ai))
        self.assertEqual(ai.army_controller.count, 2)
        self.assertEqual(ai.army_controller.armies[1].position, (11, 10))
        self.assertEqual(
            ai.army_controller.creation_cooldown,
            AI_INDEPENDENT_ARMY_CREATION_COOLDOWN,
        )

    def test_main_ai_keeps_its_action_when_multiple_armies_act(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 8)
        set_tile(board, 15, 15, Tile(PLAIN, AI, 120))
        set_tile(board, 20, 15, Tile(PLAIN, AI, 120))
        set_tile(board, 25, 15, Tile(PLAIN, AI, 90))
        self.assertTrue(
            ai.army_controller.create_at(
                board,
                (15, 15),
                ignore_cooldown=True,
            )[0]
        )
        self.assertTrue(
            ai.army_controller.create_at(
                board,
                (20, 15),
                ignore_cooldown=True,
            )[0]
        )

        moves = ai.plan_moves(board, phase=0)
        army_sources = ai.army_controller.positions
        main_moves = [move for move in moves if move.source not in army_sources]
        army_moves = [move for move in moves if move.source in army_sources]

        self.assertEqual(len(main_moves), 1)
        self.assertEqual(len(army_moves), 2)
        self.assertEqual({move.source for move in army_moves}, army_sources)
        self.assertTrue(board.move_legal(main_moves[0], AI))

    def test_army_moves_every_phase_even_without_gather_target(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 9)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 3))
        self.assertTrue(
            ai.army_controller.create_at(
                board,
                (10, 10),
                ignore_cooldown=True,
            )[0]
        )

        moves = ai.army_controller.plan_moves(board, ai, phase=0)

        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].source, (10, 10))
        self.assertTrue(board.move_legal(moves[0], AI))

    def test_army_title_follows_delayed_hill_transit(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 10)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 20))
        set_tile(board, 11, 10, Tile(HILL, NEUTRAL, 0))
        self.assertTrue(
            ai.army_controller.create_at(
                board,
                (10, 10),
                ignore_cooldown=True,
            )[0]
        )
        army = ai.army_controller.armies[0]

        move = ai.army_controller._move_toward(
            board,
            ai,
            army.position,
            (11, 10),
            board.tile(10, 10).army - 1,
        )
        self.assertIsNotNone(move)
        ai.army_controller._record_army_move(
            army,
            move.target,
            board,
        )
        self.assertEqual(army.transit_target, (11, 10))

        board.resolve_movement([move])
        board.finish_turn(board.last_report)
        ai.army_controller.plan_moves(board, ai, phase=0)
        self.assertEqual(army.position, (10, 10))

        board.resolve_movement([])
        ai.army_controller.sync(board, ai)

        self.assertEqual(army.position, (11, 10))
        self.assertIsNone(army.transit_target)
        self.assertEqual(board.tile(11, 10).army, 19)

    def test_main_ai_does_not_duplicate_a_delayed_army_move_source(
        self,
    ) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 11)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 20))
        set_tile(board, 11, 10, Tile(HILL, NEUTRAL, 0))
        self.assertTrue(
            ai.army_controller.create_at(
                board,
                (10, 10),
                ignore_cooldown=True,
            )[0]
        )
        army = ai.army_controller.armies[0]
        move = ai.army_controller._move_toward(
            board,
            ai,
            army.position,
            (11, 10),
            board.tile(10, 10).army - 1,
        )
        self.assertIsNotNone(move)
        ai.army_controller._record_army_move(
            army,
            move.target,
            board,
        )
        report = board.resolve_movement([move])
        board.finish_turn(report)

        planned = ai.plan_moves(board, phase=0)

        sources = [move.source for move in planned]
        self.assertEqual(len(sources), len(set(sources)))
        self.assertNotIn((10, 10), sources)
        self.assertNotIn((11, 10), sources)

    def test_army_move_is_separate_from_human_route_action(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 6))
        set_tile(app.board, 11, 10, Tile(PLAIN, HUMAN, 6))
        set_tile(app.board, 15, 10, Tile(PLAIN, HUMAN, 3))
        set_tile(app.board, 16, 10, Tile(PLAIN, NEUTRAL, 0))
        created, _ = app.human_ai.army_controller.create_at(
            app.board,
            (10, 10),
        )
        self.assertTrue(created)
        app.selected = (15, 10)
        self.assertTrue(app.queue_command((15, 10), (16, 10), "all"))

        app.advance_turn()

        self.assertEqual(app.board.tile(16, 10).owner, HUMAN)
        self.assertEqual(app.board.tile(10, 10).army, 1)
        self.assertNotEqual(
            app.human_ai.army_controller.armies[0].position,
            (10, 10),
        )

    def test_human_commands_cannot_select_or_route_through_army_cells(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 8))
        set_tile(app.board, 11, 10, Tile(PLAIN, HUMAN, 8))
        created, _ = app.human_ai.army_controller.create_at(
            app.board,
            (11, 10),
        )
        self.assertTrue(created)

        self.assertFalse(app.queue_command((11, 10), (10, 10), "all"))
        self.assertFalse(app.queue_command((10, 10), (11, 10), "all"))
        self.assertTrue(app.queue_command((10, 10), (9, 10), "all"))

        app.selected = (10, 10)
        app.handle_game_click(cell_screen_pos(app, (11, 10)), 1)
        self.assertEqual(app.selected, (10, 10))

    def test_main_ai_never_uses_army_cells_as_source_or_target(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 11)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 80))
        set_tile(board, 12, 10, Tile(PLAIN, AI, 80))
        set_tile(board, 12, 11, Tile(PLAIN, AI, 60))
        set_tile(board, 13, 11, Tile(PLAIN, AI, 40))
        self.assertTrue(
            ai.army_controller.create_at(
                board,
                (10, 10),
                ignore_cooldown=True,
            )[0]
        )
        self.assertTrue(
            ai.army_controller.create_at(
                board,
                (12, 10),
                ignore_cooldown=True,
            )[0]
        )

        army_positions = ai.army_controller.positions
        moves = ai.plan_moves(board, phase=0)
        main_moves = [
            move
            for move in moves
            if move.source not in army_positions
        ]

        self.assertTrue(main_moves)
        for move in moves:
            self.assertNotIn(move.target, army_positions)
        for move in main_moves:
            self.assertNotIn(move.source, army_positions)

    def test_army_resupply_ignores_pending_and_incoming_moving_sources(
        self,
    ) -> None:
        scenarios = (
            (
                "pending",
                PendingMove(
                    11,
                    10,
                    12,
                    10,
                    5,
                    AI,
                    ready_turn=1,
                ),
                (),
            ),
            (
                "incoming",
                None,
                (Move(11, 10, 12, 10, 5),),
            ),
        )
        for label, pending, incoming in scenarios:
            with self.subTest(label=label):
                board = blank_board(size=39)
                ai = GeneralsAI(AI, "hard", 12)
                set_tile(board, 10, 10, Tile(PLAIN, AI, 3))
                set_tile(board, 11, 10, Tile(PLAIN, AI, 10))
                if pending is not None:
                    board.pending_moves.append(pending)
                self.assertTrue(
                    ai.army_controller.create_at(
                        board,
                        (10, 10),
                        ignore_cooldown=True,
                    )[0]
                )
                army = ai.army_controller.armies[0]

                resupply = ai.army_controller._resupply_move(
                    board,
                    ai,
                    army,
                    {(11, 10)},
                )

                self.assertTrue(
                    resupply is None or resupply.target != (11, 10)
                )
                planned = ai.army_controller.plan_moves(
                    board,
                    ai,
                    phase=0,
                    incoming_moves=incoming,
                )
                self.assertTrue(planned)
                self.assertNotEqual(planned[0].target, (11, 10))

    def test_army_movement_rejects_other_army_cells(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 13)
        set_tile(board, 10, 11, Tile(PLAIN, AI, 20))
        set_tile(board, 10, 12, Tile(PLAIN, AI, 20))
        self.assertTrue(
            ai.army_controller.create_at(
                board,
                (10, 11),
                ignore_cooldown=True,
            )[0]
        )
        self.assertTrue(
            ai.army_controller.create_at(
                board,
                (10, 12),
                ignore_cooldown=True,
            )[0]
        )
        army = ai.army_controller.armies[0]
        other = ai.army_controller.armies[1]

        self.assertIsNone(
            ai.army_controller._move_toward(
                board,
                ai,
                army.position,
                other.position,
                19,
            )
        )
        forced = ai.army_controller._forced_army_move(
            board,
            ai,
            army,
            other.position,
        )
        self.assertIsNotNone(forced)
        self.assertNotIn(forced.target, ai.army_controller.positions)

    def test_new_army_creation_cancels_its_pending_move(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "normal", 14)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 8))
        board.pending_moves.append(
            PendingMove(10, 10, 11, 10, 3, AI, ready_turn=board.turn)
        )

        created, _ = ai.army_controller.create_at(
            board,
            (10, 10),
            ignore_cooldown=True,
        )

        self.assertTrue(created)
        self.assertFalse(
            any(
                pending.owner == AI
                and pending.source == (10, 10)
                for pending in board.pending_moves
            )
        )

    def test_army_is_released_when_failed_attack_leaves_one_soldier(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 15)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 4))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]
        army.transit_target = (11, 10)
        army.transit_ready_turn = board.turn
        set_tile(board, 10, 10, Tile(PLAIN, AI, 1))
        set_tile(board, 11, 10, Tile(PLAIN, HUMAN, 10))

        ai.army_controller.sync(board, ai)

        self.assertEqual(ai.army_controller.count, 0)

    def test_army_is_released_if_it_ends_up_on_own_general(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 16)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 4))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        set_tile(board, 10, 10, Tile(GENERAL, AI, 4))

        ai.army_controller.sync(board, ai)

        self.assertEqual(ai.army_controller.count, 0)

    def test_unable_army_is_released_immediately(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 17)
        set_tile(board, 0, 0, Tile(PLAIN, AI, 4))
        set_tile(board, 1, 0, Tile(MOUNTAIN, NEUTRAL, 0))
        set_tile(board, 0, 1, Tile(MOUNTAIN, NEUTRAL, 0))
        created, _ = ai.army_controller.create_at(board, (0, 0))
        self.assertTrue(created)

        moves = ai.army_controller.plan_moves(board, ai, phase=0)

        self.assertEqual(moves, [])
        self.assertEqual(ai.army_controller.count, 0)

    def test_double_right_click_creates_human_army_and_shows_cooldown(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 20_008))
        set_tile(app.board, 12, 10, Tile(PLAIN, HUMAN, 8))

        target = cell_screen_pos(app, (10, 10))
        app.handle_right_click(target)
        app.handle_right_click(target)
        self.assertEqual(app.human_ai.army_controller.count, 1)
        self.assertIn("创建军队", app.toast_message)

        second = cell_screen_pos(app, (12, 10))
        app.handle_right_click(second)
        app.handle_right_click(second)
        self.assertEqual(app.human_ai.army_controller.count, 1)
        self.assertIn("冷却", app.toast_message)

    def test_right_click_releases_human_army_and_restores_control(self) -> None:
        app = route_app()
        set_tile(app.board, 10, 10, Tile(PLAIN, HUMAN, 8))
        created, _ = app.human_ai.army_controller.create_at(
            app.board,
            (10, 10),
        )
        self.assertTrue(created)

        app.handle_right_click(cell_screen_pos(app, (10, 10)))

        self.assertEqual(app.human_ai.army_controller.count, 0)
        self.assertEqual(app.selected, (10, 10))

    def test_army_immediately_captures_adjacent_enemy_general_when_stronger(
        self,
    ) -> None:
        board = blank_board(size=39)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 872))
        set_tile(board, 11, 10, Tile(GENERAL, HUMAN, 840))
        set_tile(board, 7, 31, Tile(PLAIN, NEUTRAL, 0))
        board.player_positions[HUMAN] = (11, 10)
        ai = GeneralsAI(AI, "hard", 42)
        ai.known_enemy_generals[HUMAN] = (11, 10)
        ai.campaign_target = (11, 10)
        ai.campaign_target_player = HUMAN
        ai.campaign_target_kind = ATTACK_ENEMY_CITY
        created, _ = ai.army_controller.create_at(
            board,
            (10, 10),
            ignore_cooldown=True,
        )
        self.assertTrue(created)

        moves = ai.plan_moves(board, phase=0)
        army_moves = [
            move
            for move in moves
            if move.source == (10, 10)
        ]

        self.assertEqual(
            [(move.target, move.amount) for move in army_moves],
            [((11, 10), 871)],
        )
        report = board.resolve_movement(army_moves)
        self.assertIn(HUMAN, report.eliminated)

    def test_army_does_not_suicide_into_stronger_enemy_general(self) -> None:
        board = blank_board(size=39)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 800))
        set_tile(board, 11, 10, Tile(GENERAL, HUMAN, 840))
        board.player_positions[HUMAN] = (11, 10)
        ai = GeneralsAI(AI, "hard", 42)
        ai.known_enemy_generals[HUMAN] = (11, 10)
        created, _ = ai.army_controller.create_at(
            board,
            (10, 10),
            ignore_cooldown=True,
        )
        self.assertTrue(created)

        moves = ai.plan_moves(board, phase=0)
        army_moves = [
            move
            for move in moves
            if move.source == (10, 10)
        ]

        self.assertTrue(army_moves)
        self.assertNotEqual(army_moves[0].target, (11, 10))

    def test_incoming_enemy_is_prioritized_over_attack(self) -> None:
        board = blank_board()
        ai = GeneralsAI(AI, "hard", 4)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 60))
        set_tile(board, 10, 12, Tile(PLAIN, AI, 4))
        set_tile(board, 10, 13, Tile(PLAIN, HUMAN, 35))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)

        moves = ai.army_controller.plan_moves(
            board,
            ai,
            phase=0,
            incoming_moves=(Move(10, 13, 10, 12, 35),),
        )

        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].source, (10, 10))
        self.assertEqual(moves[0].target, (10, 11))
        self.assertEqual(ai.army_controller.armies[0].mission, "defend")

    def test_attack_state_stays_until_adjacent_enemy_reaches_hold_ratio(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 18)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 12))
        set_tile(board, 10, 11, Tile(PLAIN, HUMAN, 4))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]
        army.state = ARMY_STATE_ATTACK

        moves = ai.army_controller.plan_moves(board, ai, phase=0)

        self.assertTrue(moves)
        self.assertEqual(army.state, ARMY_STATE_ATTACK)

        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 19)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 12))
        set_tile(board, 10, 11, Tile(PLAIN, HUMAN, 20))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]
        army.state = ARMY_STATE_ATTACK

        ai.army_controller.plan_moves(board, ai, phase=0)

        self.assertEqual(army.state, ARMY_STATE_RESUPPLY)

    def test_moving_enemy_forces_defense_at_ten_cells(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 20)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 60))
        set_tile(board, 10, 12, Tile(PLAIN, AI, 4))
        set_tile(board, 10, 13, Tile(PLAIN, HUMAN, 35))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)

        moves = ai.army_controller.plan_moves(
            board,
            ai,
            phase=0,
            incoming_moves=(Move(10, 13, 10, 12, 35),),
        )

        self.assertTrue(moves)
        self.assertEqual(moves[0].source, (10, 10))
        self.assertEqual(ai.army_controller.armies[0].state, ARMY_STATE_DEFEND)
        self.assertEqual(ai.army_controller.armies[0].mission, "defend")

    def test_moving_enemy_prefers_defense_between_eleven_and_twenty_cells(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 21)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 60))
        set_tile(board, 10, 25, Tile(PLAIN, AI, 4))
        set_tile(board, 10, 26, Tile(PLAIN, HUMAN, 35))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)

        ai.army_controller.plan_moves(
            board,
            ai,
            phase=0,
            incoming_moves=(Move(10, 26, 10, 25, 35),),
        )

        self.assertEqual(ai.army_controller.armies[0].state, ARMY_STATE_DEFEND)

    def test_resupply_army_moves_itself_toward_castle_or_local_stack(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 22)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 8))
        set_tile(board, 14, 10, Tile(PLAIN, AI, 30))
        set_tile(board, 17, 10, Tile(CITY, AI, 2))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]

        moves = ai.army_controller.plan_moves(board, ai, phase=0)

        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0].source, (10, 10))
        self.assertEqual(moves[0].target, (11, 10))
        self.assertEqual(army.state, ARMY_STATE_RESUPPLY)
        self.assertIn(army.target, {(14, 10), (17, 10)})

    def test_resupply_switches_to_attack_after_reaching_a(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 23)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 50))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]
        army.state = ARMY_STATE_RESUPPLY

        moves = ai.army_controller.plan_moves(board, ai, phase=0)

        self.assertTrue(moves)
        self.assertEqual(army.state, ARMY_STATE_ATTACK)
        self.assertIn(army.state, ARMY_STATES)

    def test_army_with_two_soldiers_is_disbanded_before_planning(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 24)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 3))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 2))

        moves = ai.army_controller.plan_moves(board, ai, phase=0)

        self.assertEqual(moves, [])
        self.assertEqual(ai.army_controller.count, 0)

    def test_army_creation_rejects_two_soldiers(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 26)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 2))

        created, message = ai.army_controller.create_at(board, (10, 10))

        self.assertFalse(created)
        self.assertIn("至少需要 3 兵", message)
        self.assertEqual(ai.army_controller.count, 0)

    def test_two_armies_never_choose_the_same_target_cell(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 27)
        first = (10, 10)
        second = (10, 12)
        enemy = (10, 11)
        set_tile(board, *first, Tile(PLAIN, AI, 20))
        set_tile(board, *second, Tile(PLAIN, AI, 20))
        set_tile(board, *enemy, Tile(PLAIN, HUMAN, 2))
        for position in (first, second):
            created, _ = ai.army_controller.create_at(
                board,
                position,
                ignore_cooldown=True,
            )
            self.assertTrue(created)
        initial_positions = set(ai.army_controller.positions)

        moves = ai.army_controller.plan_moves(board, ai, phase=0)

        self.assertEqual(len(moves), 2)
        self.assertEqual(len({move.target for move in moves}), 2)
        self.assertTrue(
            all(move.target not in initial_positions for move in moves)
        )
        board.resolve_movement(moves)
        reserved_after_phase = {
            army.transit_target
            for army in ai.army_controller.armies
            if army.transit_target is not None
        }

        second_phase = ai.army_controller.plan_moves(board, ai, phase=1)

        self.assertEqual(
            len({move.target for move in second_phase}),
            len(second_phase),
        )
        self.assertTrue(
            all(move.target not in reserved_after_phase for move in second_phase)
        )

    def test_growth_round_reactivates_each_numbered_army_ai(self) -> None:
        board = blank_board(size=39)
        board.growth_interval = 20
        board.turn = 21
        ai = GeneralsAI(AI, "hard", 28)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 8))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]
        army.state = "inactive"
        army.force_disband = True
        army.stalled_turns = 4

        ai.army_controller.begin_turn(board, ai)

        self.assertEqual(army.state, ARMY_STATE_RESUPPLY)
        self.assertFalse(army.force_disband)
        self.assertEqual(army.stalled_turns, 0)

    def test_forced_move_keeps_one_of_three_army_states(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 25)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 8))
        set_tile(board, 10, 11, Tile(MOUNTAIN, NEUTRAL, 0))
        set_tile(board, 11, 10, Tile(MOUNTAIN, NEUTRAL, 0))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]
        army.state = ARMY_STATE_DEFEND

        move = ai.army_controller._forced_army_move(board, ai, army)

        self.assertIsNotNone(move)
        self.assertEqual(army.state, ARMY_STATE_DEFEND)
        self.assertIn(army.state, ARMY_STATES)

    def test_army_title_is_removed_when_formation_anchor_is_captured(self) -> None:
        board = blank_board()
        ai = GeneralsAI(AI, "normal", 5)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 12))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)

        set_tile(board, 10, 10, Tile(PLAIN, HUMAN, 2))
        ai.army_controller.sync(board, ai)

        self.assertEqual(ai.army_controller.count, 0)

    def test_ai_cannot_release_army_with_strength_100_or_more(self) -> None:
        board = blank_board()
        ai = GeneralsAI(AI, "normal", 6)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 100))
        set_tile(board, 12, 10, Tile(PLAIN, AI, 9_900))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]
        army.poor_performance_turns = 100

        ai.army_controller.review_ai_armies(board, ai)

        self.assertEqual(ai.army_controller.count, 1)

    def test_ai_can_release_poorly_performing_army_below_100(self) -> None:
        board = blank_board()
        board.turn = 200
        ai = GeneralsAI(AI, "normal", 7)
        for x in range(10, 15):
            set_tile(board, x, 10, Tile(PLAIN, AI, 3))
        set_tile(board, 10, 10, Tile(PLAIN, AI, 99))
        set_tile(board, 12, 10, Tile(PLAIN, AI, 9_901))
        for x in range(10, 15):
            created, _ = ai.army_controller.create_at(
                board,
                (x, 10),
                ignore_cooldown=True,
            )
            self.assertTrue(created)
        ai.army_controller.armies[0].poor_performance_turns = 100

        ai.army_controller.review_ai_armies(board, ai)

        self.assertEqual(ai.army_controller.count, 4)

    def test_ai_keeps_underperforming_army_while_slots_remain(self) -> None:
        board = blank_board()
        board.turn = 200
        ai = GeneralsAI(AI, "normal", 8)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 99))
        set_tile(board, 12, 10, Tile(PLAIN, AI, 9_901))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        ai.army_controller.armies[0].poor_performance_turns = 100

        ai.army_controller.review_ai_armies(board, ai)

        self.assertEqual(ai.army_controller.count, 1)

    def test_castle_can_host_an_army(self) -> None:
        board = blank_board()
        ai = GeneralsAI(AI, "normal", 11)
        set_tile(board, 10, 10, Tile(CITY, AI, 9))

        created, message = ai.army_controller.create_at(board, (10, 10))

        self.assertTrue(created, message)
        self.assertEqual(ai.army_controller.armies[0].position, (10, 10))

    def test_ai_is_eager_to_title_leading_stack(self) -> None:
        board = blank_board()
        ai = GeneralsAI(AI, "normal", 12)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 10024))
        for x in range(11, 19):
            set_tile(board, x, 10, Tile(PLAIN, AI, 5))

        self.assertTrue(ai.army_controller.maybe_create_auto(board, ai))
        self.assertEqual(ai.army_controller.count, 1)
        self.assertEqual(ai.army_controller.armies[0].position, (10, 10))

    def test_ai_titles_top_half_four_army_stack_eagerly(self) -> None:
        board = blank_board()
        ai = GeneralsAI(AI, "normal", 22)
        general = board.general_position(AI)
        self.assertIsNotNone(general)
        set_tile(board, *general, Tile(GENERAL, AI, 9_990))
        set_tile(board, 10, 10, Tile(PLAIN, AI, 4))
        for position in ((11, 10), (12, 10), (13, 10)):
            set_tile(board, *position, Tile(PLAIN, AI, 3))

        self.assertTrue(ai.army_controller.maybe_create_auto(board, ai))
        self.assertEqual(ai.army_controller.armies[0].position, (10, 10))

    def test_ai_titles_any_three_stack_with_ai_budget(self) -> None:
        board = blank_board()
        ai = GeneralsAI(AI, "normal", 23)
        general = board.general_position(AI)
        self.assertIsNotNone(general)
        set_tile(board, *general, Tile(GENERAL, AI, 10_000))
        set_tile(board, 10, 10, Tile(PLAIN, AI, 3))
        set_tile(board, 11, 10, Tile(PLAIN, AI, 3))

        self.assertTrue(ai.army_controller.maybe_create_auto(board, ai))
        self.assertEqual(ai.army_controller.armies[0].position, (10, 10))

    def test_all_ai_armies_move_without_idling_or_main_ai_interference(
        self,
    ) -> None:
        board = blank_board(seed=91, player_count=2, size=75)
        board.turn = 200
        positions = [(8, 8), (66, 66)]
        ais: list[GeneralsAI] = []
        for player, position in enumerate(positions):
            x, y = position
            set_tile(board, x, y, Tile(GENERAL, player, 2))
            board.player_positions[player] = position
            ai = GeneralsAI(player, "hard", 91_000 + player)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                set_tile(
                    board,
                    x + dx,
                    y + dy,
                    Tile(PLAIN, player, 10_000),
                )
                created, _ = ai.army_controller.create_at(
                    board,
                    (x + dx, y + dy),
                    ignore_cooldown=True,
                )
                self.assertTrue(created)
            ais.append(ai)

        idled = 0
        main_targeted_armies = 0
        for _turn in range(30):
            for ai in ais:
                ai.army_controller.begin_turn(board, ai)
                ai.army_controller.review_ai_armies(board, ai)
            report = TurnReport()
            for phase in range(2):
                moves: list[Move] = []
                for ai in ais:
                    ai.army_controller.sync(board, ai)
                    before = {
                        army.unit_id: army.position
                        for army in ai.army_controller.armies
                    }
                    waiting = {
                        army.unit_id
                        for army in ai.army_controller.armies
                        if army.transit_target is not None
                    }
                    army_positions = set(before.values())
                    planned = ai.plan_moves(board, phase=phase)
                    army_sources = {
                        move.source
                        for move in planned
                        if move.source in army_positions
                    }
                    main_targeted_armies += sum(
                        1
                        for move in planned
                        if move.source not in army_positions
                        and move.target in army_positions
                    )
                    surviving = {
                        army.unit_id
                        for army in ai.army_controller.armies
                    }
                    for unit_id, position in before.items():
                        if (
                            unit_id in surviving
                            and unit_id not in waiting
                            and position not in army_sources
                        ):
                            idled += 1
                    moves.extend(planned)
                report.merge(board.resolve_movement(moves))
            board.finish_turn(report)

        self.assertEqual(idled, 0)
        self.assertEqual(main_targeted_armies, 0)

    def test_army_survival_weight_avoids_equal_enemy_counterattack(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 13)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 12))
        set_tile(board, 10, 11, Tile(PLAIN, HUMAN, 20))
        set_tile(board, 11, 10, Tile(PLAIN, AI, 1))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]
        army.target = (10, 11)

        move = ai.army_controller._forced_army_move(
            board,
            ai,
            army,
            goal=(10, 11),
        )

        self.assertIsNotNone(move)
        self.assertNotEqual(move.target, (10, 11))

    def test_damaged_army_retreats_to_resupply(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 14)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 4))
        set_tile(board, 11, 10, Tile(PLAIN, HUMAN, 12))
        set_tile(board, 5, 10, Tile(PLAIN, AI, 30))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]
        army.damage_pressure = INDEPENDENT_ARMY_DAMAGE_PRESSURE
        army.state = ARMY_STATE_ATTACK
        board.turn = 40

        moves = ai.army_controller.plan_moves(board, ai, phase=0)

        self.assertTrue(moves)
        self.assertEqual(army.state, ARMY_STATE_RESUPPLY)
        self.assertLess(moves[0].target[0], moves[0].source[0])

    def test_resupply_prefers_safe_rear_stack(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 15)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 5))
        set_tile(board, 11, 10, Tile(PLAIN, AI, 40))
        set_tile(board, 12, 10, Tile(PLAIN, HUMAN, 25))
        set_tile(board, 5, 10, Tile(PLAIN, AI, 30))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]

        move = ai.army_controller._resupply_move(board, ai, army, set())

        self.assertIsNotNone(move)
        self.assertEqual(move.target, (11, 10))
        self.assertLessEqual(
            Board.manhattan(move.target, (12, 10)),
            Board.manhattan((5, 10), (12, 10)),
        )

    def test_displacement_reward_requires_twenty_cells_per_interval(self) -> None:
        board = blank_board(size=39)
        board.turn = 10
        ai = GeneralsAI(AI, "hard", 16)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 20))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]

        army.position = (
            10 + INDEPENDENT_ARMY_DISPLACEMENT_REWARD_DISTANCE + 1,
            10,
        )
        board.turn = 10 + INDEPENDENT_ARMY_DISPLACEMENT_INTERVAL
        ai.army_controller.begin_turn(board, ai)

        self.assertEqual(army.displacement_reward, 1)
        self.assertEqual(
            army.displacement_last_position,
            army.position,
        )
        self.assertEqual(
            army.displacement_last_turn,
            board.turn,
        )

    def test_displacement_reward_ignores_short_trips(self) -> None:
        board = blank_board(size=39)
        board.turn = 10
        ai = GeneralsAI(AI, "hard", 17)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 20))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]

        army.position = (18, 10)
        board.turn = 10 + INDEPENDENT_ARMY_DISPLACEMENT_INTERVAL
        ai.army_controller.begin_turn(board, ai)

        self.assertEqual(army.displacement_reward, 0)

    def test_army_is_marked_loitering_after_sustained_tiny_area_motion(
        self,
    ) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 18)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 20))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]

        for turn, position in (
            (0, (10, 10)),
            (30, (11, 10)),
            (60, (10, 10)),
        ):
            board.turn = turn
            army.position = position
            ai.army_controller.begin_turn(board, ai)

        self.assertTrue(army.loitering)
        self.assertEqual(army.loiter_center, (10, 10))
        self.assertGreater(army.loiter_breakout_turns, 0)

        set_tile(board, 10, 11, Tile(PLAIN, AI, 2))
        move = ai.army_controller._loiter_breakout_move(
            board,
            ai,
            army,
            set(),
        )

        self.assertIsNotNone(move)
        self.assertEqual(move.source, (10, 10))
        self.assertNotEqual(move.target, (10, 10))

    def test_army_keeps_moving_when_loitering_is_detected(self) -> None:
        board = blank_board(size=39)
        ai = GeneralsAI(AI, "hard", 19)
        set_tile(board, 10, 10, Tile(PLAIN, AI, 20))
        set_tile(board, 10, 11, Tile(PLAIN, AI, 2))
        set_tile(board, 11, 10, Tile(PLAIN, AI, 2))
        created, _ = ai.army_controller.create_at(board, (10, 10))
        self.assertTrue(created)
        army = ai.army_controller.armies[0]
        army.loitering = True
        army.loiter_center = (10, 10)
        army.loiter_breakout_turns = 10

        moves = ai.plan_moves(board, phase=0)
        army_moves = [
            move
            for move in moves
            if move.source == (10, 10)
        ]

        self.assertTrue(army_moves)

    def test_ai_army_creation_cooldown_is_thirty_five_turns(self) -> None:
        self.assertEqual(AI_INDEPENDENT_ARMY_CREATION_COOLDOWN, 35)

    def test_large_map_ai_runtime_is_within_budget(self) -> None:
        """Keep a concrete performance guard for the 100x100 / 12-AI case."""
        started = time.perf_counter()
        board = Board(100, 100, 12, 2026)
        ais = [
            GeneralsAI(player, "hard", 2026_000 + player)
            for player in range(12)
        ]

        for _turn in range(24):
            for ai in ais:
                controller = ai.army_controller
                controller.begin_turn(board, ai)
                controller.review_ai_armies(board, ai)
                controller.maybe_create_auto(board, ai)

            report = TurnReport()
            for phase in range(2):
                moves: list[Move] = []
                for ai in ais:
                    moves.extend(ai.plan_moves(board, phase=phase))
                report.merge(board.resolve_movement(moves))
            board.finish_turn(report)

        elapsed = time.perf_counter() - started
        self.assertLess(
            elapsed,
            6.0,
            f"100x100/12-AI 24-turn probe took {elapsed:.3f}s",
        )


if __name__ == "__main__":
    unittest.main()
